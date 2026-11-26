"""Command-bar run controller: persistent asyncio thread, event relay, broker, Stop.

Threads:
- Qt main thread: bar, transcript, cards, QtInvoker callbacks, Orange preview.
- Agent loop thread (persistent asyncio loop): provider streaming and the loop.
- Agent worker (the bridge's single executor): ToolRuntime calls on GuiWorkspace.
Cards are non-modal; the loop awaits futures resolved from Qt.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import itertools
from threading import Event, Thread
import time

from PySide6.QtCore import QObject, Qt, Signal, Slot

from application.agent_rendering import RenderedImage
from application.ai.agent_conversation import AgentConversation
from application.ai.agent_events import LoopEvent, RunOutcome, RunResult
from application.ai.agent_loop import AgentLoop, ExecOutcome, LoopHooks, LoopPolicy
from application.ai.agent_wire import Effort
from application.ai.budget import SessionBudget
from application.ai.limits import AgentRunLimits
from application.commands import Origin, Surface, default_registry
from application.commands.action_log import ActionLog

CONFIRM_TIMEOUT_S = 300.0
QUESTION_TIMEOUT_S = 600.0


class _CancelFlag:
    def __init__(self, event: Event):
        self.event = event

    @property
    def cancellation_requested(self):
        return self.event.is_set()


@dataclass(frozen=True)
class PendingCard:
    identifier: int
    kind: str  # confirmation | question
    title: str
    text: str
    options: tuple = ()
    prompt: str = ""


class BridgeExecutor:
    """Runs registry tools on the bridge worker; records the action log."""

    def __init__(self, bridge, registry, action_log, cancel_event: Event):
        self.bridge, self.registry, self.log, self.cancel_event = bridge, registry, action_log, cancel_event
        self.run_id = ""

    def spec_for(self, tool):
        return self.registry.by_tool(tool) if self.registry.has_tool(tool) else None

    async def execute(self, tool, arguments, cancel):
        started = time.monotonic()
        future = self.bridge.submit_inapp(tool, dict(arguments), self.cancel_event)
        raw = await asyncio.wrap_future(future)
        images = tuple(RenderedImage(base64.b64decode(item["data"]), item.get("mime_type", "image/png"))
                       for item in raw.get("images", ()))
        payload = dict(raw.get("payload") or {})
        ok = bool(raw.get("ok")) and not raw.get("is_error")
        self.log.record(origin=Origin.AGENT_INAPP.value, run_id=self.run_id, command=tool, arguments=arguments,
                        status="ok" if ok else "error", code=str(payload.get("code", "")),
                        duration_ms=int((time.monotonic() - started) * 1000), revision=payload.get("revision"))
        return ExecOutcome(ok, payload, images)


class AgentRunController(QObject):
    event = Signal(object)          # LoopEvent
    card = Signal(object)           # PendingCard
    run_finished = Signal(object)   # RunResult
    busy_changed = Signal(bool)

    def __init__(self, window, *, provider_factory=None):
        super().__init__(window)
        self.window = window
        self.bridge = window.agent_bridge_controller
        self.registry = default_registry()
        self.action_log = ActionLog()
        self.bridge.action_log = self.action_log
        self.bridge.inapp_runtime = None
        self.provider_factory = provider_factory
        self.provider = None
        self.model = "claude-opus-5-5"
        self.effort = Effort.MEDIUM
        self.policy = LoopPolicy()
        self.limits = AgentRunLimits()
        self.session = SessionBudget(cap_usd=self.limits.max_session_cost_usd)
        self.conversation = AgentConversation(self.model)
        self.loop_obj = None
        self.running = False
        self.last_result: RunResult | None = None
        self._cancel = Event()
        self._cards: dict[int, tuple] = {}
        self._ids = itertools.count(1)
        self._aio = asyncio.new_event_loop()
        self._thread = Thread(target=self._aio.run_forever, name="ghostgui-agent-loop", daemon=True)
        self._thread.start()
        self._closed = False
        self.event.connect(self._on_event_main, Qt.ConnectionType.QueuedConnection)

    # ---------------------------------------------------------------- settings
    def set_model(self, model: str):
        if model != self.model:
            self.model = model
            self.new_conversation()

    def new_conversation(self):
        if self.running:
            raise RuntimeError("Stop the current run first")
        self.conversation = AgentConversation(self.model)
        self.loop_obj = None

    def set_provider(self, provider):
        old, self.provider = self.provider, provider
        self.loop_obj = None
        if old is not None and hasattr(old, "aclose"):
            asyncio.run_coroutine_threadsafe(old.aclose(), self._aio)

    def ensure_provider(self):
        if self.provider is None and self.provider_factory is not None:
            self.provider = self.provider_factory()
        return self.provider

    def _tools(self):
        ports = ("gui",)
        return tuple(self.registry.tool_definitions(Surface.AGENT_INAPP, ports=ports))

    # --------------------------------------------------------------------- run
    def can_start(self) -> str:
        if self.running:
            return "Claude is already working; Stop it first"
        if self.bridge.server is not None:
            return "External control is active; disconnect it to use the command bar"
        ai = self.window.ai_assistant_controller
        if ai.session is not None or ai.active_handle is not None:
            return "Resolve the Motion Assistant request first"
        if self.window.robot_model_3d is None:
            return "Load a robot first"
        return ""

    def start(self, prompt: str) -> bool:
        reason = self.can_start()
        if reason:
            self._emit_local("notice", text=reason)
            return False
        provider = self.ensure_provider()
        if provider is None:
            self._emit_local("notice", text="Add an Anthropic API key in Claude settings to use the command bar")
            return False
        self._cancel = Event()
        executor = BridgeExecutor(self.bridge, self.registry, self.action_log, self._cancel)
        executor.run_id = f"run-{int(time.time() * 1000)}"
        hooks = LoopHooks(emit=self.event.emit, confirm=self._confirm, ask=self._ask)
        if self.loop_obj is None or self.loop_obj.provider is not provider:
            self.loop_obj = AgentLoop(provider, executor, model=self.model, tools=self._tools(), effort=self.effort,
                                      conversation=self.conversation, hooks=hooks, limits=self.limits,
                                      session_budget=self.session, policy=self.policy)
        else:
            self.loop_obj.executor, self.loop_obj.hooks, self.loop_obj.policy = executor, hooks, self.policy
        try:
            digest = self.window.agent_command_port.digest()
        except Exception:
            digest = None
        self.running = True
        self.busy_changed.emit(True)
        self._lock_controls()
        future = asyncio.run_coroutine_threadsafe(
            self.loop_obj.run(prompt, digest, _CancelFlag(self._cancel)), self._aio)
        future.add_done_callback(lambda done: self.event.emit(LoopEvent("_done", data={"future": done})))
        return True

    def approve_plan(self, deferred):
        if self.running or self.loop_obj is None:
            return False
        self._cancel = Event()
        self.loop_obj.executor.cancel_event = self._cancel
        self.running = True
        self.busy_changed.emit(True)
        future = asyncio.run_coroutine_threadsafe(
            self.loop_obj.execute_plan(deferred, _CancelFlag(self._cancel)), self._aio)
        future.add_done_callback(lambda done: self.event.emit(LoopEvent("_plan_done", data={"future": done})))
        return True

    def stop(self):
        if not self.running:
            return
        self._cancel.set()
        for identifier in list(self._cards):
            self.answer(identifier, None)
        self._emit_local("notice", text="Stopping…")

    def _lock_controls(self):
        bridge = self.bridge
        if not bridge._owned_controls:
            bridge._owned_controls = True
            self.window.set_ai_motion_controls_enabled(False)

    # ------------------------------------------------------------ cards/broker
    async def _await_card(self, card: PendingCard, timeout: float):
        future = self._aio.create_future()
        self._cards[card.identifier] = (future, card)
        self.card.emit(card)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._cards.pop(card.identifier, None)

    async def _confirm(self, spec, arguments, preview, prompt):
        card = PendingCard(next(self._ids), "confirmation", spec.title, preview, prompt=prompt[:200])
        decision = await self._await_card(card, CONFIRM_TIMEOUT_S)
        return decision if decision in ("once", "run") else "deny"

    async def _ask(self, question, options):
        card = PendingCard(next(self._ids), "question", "Claude asks", question, tuple(options))
        return await self._await_card(card, QUESTION_TIMEOUT_S)

    def answer(self, identifier: int, value):
        """Called on the Qt thread by a card."""
        entry = self._cards.get(identifier)
        if entry is None:
            return
        future, _ = entry
        self._aio.call_soon_threadsafe(lambda: future.done() or future.set_result(value))

    # ------------------------------------------------------------------ relay
    def _emit_local(self, kind, **values):
        self.event.emit(LoopEvent(kind, **values))

    @Slot(object)
    def _on_event_main(self, event):
        if event.kind == "_done":
            self._finish(event.data["future"])
        elif event.kind == "_plan_done":
            self.running = False
            self.busy_changed.emit(False)

    def _finish(self, future):
        self.running = False
        try:
            result = future.result()
        except Exception as error:  # never leave the UI stuck in running state
            result = RunResult(RunOutcome.ERROR, message=str(error)[:300])
        self.last_result = result
        self.busy_changed.emit(False)
        if (self.policy.approval == "auto" and result.outcome is RunOutcome.COMPLETED
                and self.bridge.has_candidate):
            self.accept_candidate()
        self.run_finished.emit(result)

    # -------------------------------------------------------------- candidate
    def accept_candidate(self):
        return self.bridge.submit_local("_user_accept")

    def discard_candidate(self):
        return self.bridge.submit_local("discard_candidate")

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        provider = self.provider
        if provider is not None and hasattr(provider, "aclose"):
            try:
                asyncio.run_coroutine_threadsafe(provider.aclose(), self._aio).result(timeout=2)
            except Exception:
                pass
        self._aio.call_soon_threadsafe(self._aio.stop)
        self._thread.join(timeout=2)

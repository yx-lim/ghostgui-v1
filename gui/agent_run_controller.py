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
from application.commands import Effect, Origin, Surface, default_registry
from application.commands.action_log import ActionLog

CONFIRM_TIMEOUT_S = 300.0
QUESTION_TIMEOUT_S = 600.0
UNDO_LABEL_CHARS = 48
BRIDGE_BUSY = "GhostGUI is finishing the last step"
IMAGES_OFF_MESSAGE = ("The user has turned off viewport images (Claude settings → Allow viewport images). "
                      "Use get_app_state instead.")


def undo_label(prompt: str) -> str:
    """History entry name for an accepted command-bar edit: "Claude: <prompt excerpt>"."""
    from application.ai.secrets import scrub_secrets

    text = " ".join(scrub_secrets(str(prompt)).split())
    if len(text) > UNDO_LABEL_CHARS:
        text = text[:UNDO_LABEL_CHARS - 1].rstrip() + "…"
    return f"Claude: {text}" if text else "Claude edit"


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

    def __init__(self, bridge, registry, action_log, cancel_event: Event, *, allow_images=lambda: True):
        self.bridge, self.registry, self.log, self.cancel_event = bridge, registry, action_log, cancel_event
        self.allow_images = allow_images
        self.run_id = ""
        self.transitioned = False  # a project/model switch clears undo history

    def spec_for(self, tool):
        return self.registry.by_tool(tool) if self.registry.has_tool(tool) else None

    async def execute(self, tool, arguments, cancel):
        started = time.monotonic()
        spec = self.spec_for(tool)
        allow_images = bool(self.allow_images())
        if spec is not None and spec.returns_images and not allow_images:
            payload = {"ok": False, "code": "images_disabled", "message": IMAGES_OFF_MESSAGE}
            self.log.record(origin=Origin.AGENT_INAPP.value, run_id=self.run_id, command=tool, arguments=arguments,
                            status="error", code="images_disabled", duration_ms=0)
            return ExecOutcome(False, payload)
        future = self.bridge.submit_inapp(tool, dict(arguments), self.cancel_event)
        raw = await asyncio.wrap_future(future)
        images = tuple(RenderedImage(base64.b64decode(item["data"]), item.get("mime_type", "image/png"))
                       for item in raw.get("images", ())) if allow_images else ()
        payload = dict(raw.get("payload") or {})
        ok = bool(raw.get("ok")) and not raw.get("is_error")
        if ok and spec is not None and spec.effect is Effect.TRANSITION:
            self.transitioned = True
        self.log.record(origin=Origin.AGENT_INAPP.value, run_id=self.run_id, command=tool, arguments=arguments,
                        status="ok" if ok else "error", code=str(payload.get("code", "")),
                        duration_ms=int((time.monotonic() - started) * 1000), revision=payload.get("revision"))
        return ExecOutcome(ok, payload, images)


class AgentRunController(QObject):
    event = Signal(object)          # LoopEvent
    card = Signal(object)           # PendingCard
    run_finished = Signal(object)   # RunResult
    busy_changed = Signal(bool)
    undo_changed = Signal()         # Undo run availability may have changed

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
        self.prewarm_enabled = False
        self.allow_images = True
        # Undo run: the view before the last run, and the history entry its Accept made.
        self.run_snapshot = None
        self.run_entry = None
        self.run_transitioned = False
        self._last_activity = float("-inf")
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

    # ---------------------------------------------------------------- prewarm
    PREWARM_IDLE_S = 55 * 60

    def prewarm_if_idle(self):
        """Opt-in: refresh the 1-hour cache when the bar is focused after an idle period."""
        if not self.prewarm_enabled or self.running:
            return False
        provider = self.ensure_provider()
        if provider is None or not hasattr(provider, "prewarm"):
            return False
        if time.monotonic() - self._last_activity < self.PREWARM_IDLE_S:
            return False
        from application.ai.agent_prompts import COMMAND_BAR_SYSTEM_PROMPT
        from application.ai.agent_wire import AgentTurnRequest

        messages = self.conversation.wire() or ({"role": "user", "content": [{"type": "text", "text": "."}]},)
        if messages[-1]["role"] == "assistant":
            messages = messages + ({"role": "user", "content": [{"type": "text", "text": "."}]},)
        request = AgentTurnRequest(model=self.model, system=COMMAND_BAR_SYSTEM_PROMPT, tools=self._tools(),
                                   messages=tuple(messages), effort=self.effort, fallbacks=self.policy.fallbacks)
        self._last_activity = time.monotonic()
        asyncio.run_coroutine_threadsafe(provider.prewarm(request), self._aio)
        return True

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
        executor = BridgeExecutor(self.bridge, self.registry, self.action_log, self._cancel,
                                  allow_images=lambda: self.allow_images)
        executor.run_id = f"run-{int(time.time() * 1000)}"
        self.run_snapshot = self._capture_view()
        self.run_entry = None
        self.run_transitioned = False
        self.bridge.accept_label = undo_label(prompt)
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
        self._last_activity = time.monotonic()
        self.busy_changed.emit(True)
        self.undo_changed.emit()
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
            self.run_transitioned = self.run_transitioned or self.loop_obj.executor.transitioned
            self.busy_changed.emit(False)
            self.undo_changed.emit()
        elif event.kind == "_discarded":
            self.undo_changed.emit()
        elif event.kind == "_accepted":
            stack = self.window.undo_stack
            if len(stack) > event.data["before"]:
                self.run_entry = stack[-1]
            self.undo_changed.emit()

    def _finish(self, future):
        self.running = False
        try:
            result = future.result()
        except Exception as error:  # never leave the UI stuck in running state
            result = RunResult(RunOutcome.ERROR, message=str(error)[:300])
        self.last_result = result
        executor = getattr(self.loop_obj, "executor", None)
        self.run_transitioned = bool(getattr(executor, "transitioned", False))
        self.busy_changed.emit(False)
        if (self.policy.approval == "auto" and result.outcome is RunOutcome.COMPLETED
                and self.bridge.has_candidate):
            self.accept_candidate()
        self.run_finished.emit(result)
        self.undo_changed.emit()

    # -------------------------------------------------------------- candidate
    def accept_candidate(self):
        """Accept the Orange preview; remembers the history entry for Undo run."""
        before = len(self.window.undo_stack)
        future = self.bridge.submit_local("_user_accept")
        future.add_done_callback(lambda done: self.event.emit(LoopEvent("_accepted", data={"before": before})))
        return future

    # --------------------------------------------------------------- undo run
    def _capture_view(self):
        window = self.window
        try:
            return {"workspace": window.capture_project_workspace(),
                    "left_sidebar": not window._left_sidebar_collapsed,
                    "right_sidebar": not window._right_sidebar_collapsed}
        except Exception:
            return None

    def undo_run_blocker(self) -> str:
        """Why Undo run is unavailable, or "" when it can run."""
        if self.running:
            return "Claude is still working"
        if self.run_snapshot is None:
            return "No Claude run to undo"
        if self.run_transitioned:
            return "The run opened a project or robot, which clears undo history"
        if self.bridge.has_candidate:
            return "Accept or Discard the Orange preview first"
        if self.bridge.server is not None:
            return "External control is active"
        if self.run_entry is not None:
            stack = self.window.undo_stack
            if not any(entry is self.run_entry for entry in stack):
                return "This run's edit was already undone"
            if stack[-1] is not self.run_entry:
                return "Other edits were made after this run; use Edit → Undo"
        if self.bridge.owns_document:
            return BRIDGE_BUSY  # a just-finished tool is cleared on the bridge's next poll
        return ""

    def undo_run(self) -> str:
        """Undo the last run's accepted edit and restore the view it started from.

        Returns "" on success, otherwise the reason nothing was changed.
        """
        reason = self.undo_run_blocker()
        if reason:
            return reason
        window = self.window
        if self.run_entry is not None:
            window.undo_last_action()
            if window.undo_stack and window.undo_stack[-1] is self.run_entry:
                return "Undo was blocked"
        snapshot = self.run_snapshot
        try:
            window.restore_project_workspace(snapshot["workspace"])
            time_s = snapshot["workspace"].get("current_time")
            if time_s is not None:
                window.viewer_3d.set_current_time(float(time_s))
            window.set_left_sidebar_visible(bool(snapshot["left_sidebar"]))
            window.set_right_sidebar_visible(bool(snapshot["right_sidebar"]))
        except Exception as error:
            return f"Motion undone; the view could not be restored ({str(error)[:120]})"
        finally:
            self.run_snapshot, self.run_entry = None, None
            self.undo_changed.emit()
        return ""

    def discard_candidate(self):
        future = self.bridge.submit_local("discard_candidate")
        future.add_done_callback(lambda done: self.event.emit(LoopEvent("_discarded")))
        return future

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

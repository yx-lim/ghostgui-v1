"""Provider-neutral command-bar agent loop on the Messages API shape.

Invariants:
- append-only history (required by preserved thinking);
- every tool_use receives a tool_result, synthesised on Stop or denial;
- tools run only after the full turn has streamed;
- operator policy travels in mid-conversation system messages, appended to the
  conversation only together with the assistant turn that answers them.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
import json
import time
from typing import Any, Awaitable, Callable, Mapping, Protocol

from application.ai.agent_conversation import AgentConversation
from application.ai.agent_events import DeferredCall, LoopEvent, RunOutcome, RunResult
from application.ai.agent_prompts import COMMAND_BAR_SYSTEM_PROMPT
from application.ai.agent_wire import (
    AgentStopReason, AgentStreamEvent, AgentTurnRequest, Effort, canonical_json,
    tool_result_block, user_text,
)
from application.ai.budget import RunBudget, SessionBudget
from application.ai.errors import (
    ProviderCancelledError, ProviderError, ProviderMalformedStreamError,
)
from application.ai.limits import AGENT_MAX_OUTPUT_TOKENS, AgentRunLimits
from application.ai.secrets import contains_secret
from application.commands.spec import CONFIRM_EFFECTS, PLAN_DEFERRED_EFFECTS, Effect

RAISED_OUTPUT_TOKENS = 128_000
INTERACTION_TOOLS = frozenset({"ask_user"})


@dataclass(frozen=True)
class ExecOutcome:
    ok: bool
    payload: Mapping[str, Any]
    images: tuple = ()


class ToolExecutor(Protocol):
    def spec_for(self, tool: str): ...

    async def execute(self, tool: str, arguments: Mapping[str, Any], cancel) -> ExecOutcome: ...


Confirm = Callable[[Any, Mapping[str, Any], str, str], Awaitable[str]]  # → deny | once | run
Ask = Callable[[str, tuple], Awaitable[str | None]]


@dataclass
class LoopHooks:
    emit: Callable[[LoopEvent], None] = lambda event: None
    confirm: Confirm | None = None
    ask: Ask | None = None


@dataclass
class LoopPolicy:
    approval: str = "review"  # review | auto
    plan_mode: bool = False
    fallbacks: bool = True


class _Flag:
    def __init__(self):
        self.cancellation_requested = False


@dataclass
class _RunState:
    prompt: str
    budget: RunBudget
    started: float = field(default_factory=time.monotonic)
    turns: int = 0
    actions: int = 0
    images: int = 0
    confirmations: int = 0
    consecutive_errors: int = 0
    failures: Counter = field(default_factory=Counter)
    allow_for_run: set = field(default_factory=set)
    deferred: list = field(default_factory=list)
    stop: RunOutcome | None = None
    stop_message: str = ""
    text: str = ""
    candidate_touched: bool = False
    planning_halted: bool = False


def _preview(spec, arguments) -> str:
    if spec.preview is not None:
        try:
            return str(spec.preview(None, arguments))[:300]
        except Exception:
            pass
    summary = spec.summarize(arguments)
    return f"{spec.title}" + (f" · {summary}" if summary else "")


class AgentLoop:
    def __init__(self, provider, executor: ToolExecutor, *, model: str, tools: tuple,
                 effort: Effort = Effort.MEDIUM, conversation: AgentConversation | None = None,
                 hooks: LoopHooks | None = None, limits: AgentRunLimits | None = None,
                 session_budget: SessionBudget | None = None, policy: LoopPolicy | None = None,
                 system_prompt: str = COMMAND_BAR_SYSTEM_PROMPT):
        self.provider, self.executor = provider, executor
        self.model, self.effort = model, effort
        self.tools = tuple(sorted((dict(tool) for tool in tools), key=lambda tool: tool["name"]))
        self.conversation = conversation or AgentConversation(model)
        if self.conversation.model != model:
            raise ValueError("the model is fixed per conversation; start a new conversation")
        self.hooks = hooks or LoopHooks()
        self.limits = limits or AgentRunLimits()
        self.session = session_budget or SessionBudget(cap_usd=self.limits.max_session_cost_usd)
        self.policy = policy or LoopPolicy()
        self.system_prompt = system_prompt
        self.pending_note = ""

    # ----------------------------------------------------------------- helpers
    def _emit(self, kind, **values):
        try:
            self.hooks.emit(LoopEvent(kind, **values))
        except Exception:
            pass

    def _policy_text(self, state: _RunState) -> str:
        approval = ("Review: accept_candidate only stages the preview for the user's decision."
                    if self.policy.approval == "review" else
                    "Apply automatically: accept_candidate commits a validated preview.")
        plan = (" Plan mode is on: UI, save, export, history and project actions are recorded "
                "as a plan for the user to approve instead of running; motion edits still build "
                "the preview." if self.policy.plan_mode else "")
        remaining = max(0.0, min(state.budget.cap_usd - state.budget.spent_usd, self.session.remaining_usd))
        return (f"Run policy. Approval: {approval}{plan} Limits: at most {self.limits.max_turns} "
                f"turns and {self.limits.max_tool_calls} tool calls; about ${remaining:.2f} of budget "
                "remains for this run.")

    def _relay(self, event: AgentStreamEvent):
        if event.kind in ("text", "thinking"):
            self._emit(event.kind, text=event.text)
        elif event.kind == "tool_use_start":
            spec = self.executor.spec_for(event.name)
            self._emit("tool_preparing", tool=event.name, title=spec.title if spec else event.name)

    async def _request(self, pending_system, cancel):
        messages = self.conversation.wire() + tuple({"role": "system", "content": text}
                                                    for text in pending_system)
        max_tokens = AGENT_MAX_OUTPUT_TOKENS
        attempts = self.limits.max_stream_reissues + 1
        for attempt in range(attempts + 1):
            request = AgentTurnRequest(
                model=self.model, system=self.system_prompt, tools=self.tools, messages=messages,
                max_tokens=max_tokens, effort=self.effort, fallbacks=self.policy.fallbacks)
            try:
                response = await asyncio.wait_for(
                    self.provider.stream_turn(request, self._relay, cancel), self.limits.max_turn_seconds)
            except asyncio.TimeoutError as error:
                raise ProviderError("Claude took too long to respond") from error
            except ProviderMalformedStreamError:
                if attempt + 1 >= attempts:
                    raise
                self._emit("notice", text="Re-issuing a turn whose tool input did not parse")
                continue
            if (response.stop_reason is AgentStopReason.MAX_TOKENS and response.tool_uses
                    and max_tokens < RAISED_OUTPUT_TOKENS):
                max_tokens = RAISED_OUTPUT_TOKENS  # never append a truncated tool call
                continue
            return response
        raise ProviderError("Claude did not produce a complete turn")

    def _error(self, call_id, code, message, **extra):
        return tool_result_block(call_id, {"ok": False, "code": code, "message": message, **extra},
                                 is_error=True)

    # --------------------------------------------------------------- tool calls
    async def _run_call(self, call, state: _RunState, cancel, declined: bool):
        call_id, name, arguments = call.get("id", ""), call.get("name", ""), call.get("input")
        spec = self.executor.spec_for(name)
        if cancel.cancellation_requested:
            return self._error(call_id, "cancelled", "Cancelled by the user before execution"), declined
        if state.stop is not None:
            return self._error(call_id, "run_stopped", f"Not run: {state.stop_message or 'the run stopped'}"), declined
        if not isinstance(arguments, dict):
            return self._error(call_id, "INVALID_JSON", "Tool input was not a JSON object; resend it"), declined
        if spec is None:
            return self._error(call_id, "unknown_tool", f"No tool named {name!r}"), declined
        if declined and spec.effect is not Effect.READ:
            return self._error(call_id, "skipped", "skipped: earlier action declined"), declined
        if state.actions >= self.limits.max_tool_calls:
            state.stop, state.stop_message = RunOutcome.BUDGET, "tool call limit reached"
            return self._error(call_id, "limit", "Tool call limit reached for this run"), declined
        preview = _preview(spec, arguments)
        if name in INTERACTION_TOOLS:
            return await self._ask(call_id, arguments, state), declined
        if self.policy.plan_mode and spec.effect in PLAN_DEFERRED_EFFECTS:
            if state.planning_halted:
                return self._error(call_id, "plan_halted", "Planning stopped at a project/model change"), declined
            state.deferred.append(DeferredCall(name, dict(arguments), preview))
            if spec.effect is Effect.TRANSITION:
                state.planning_halted = True
            self._emit("tool_deferred", tool=name, call_id=call_id, title=spec.title, summary=preview)
            return tool_result_block(call_id, {"ok": True, "dry_run": True, "deferred": True,
                                               "preview": preview}), declined
        if spec.effect in CONFIRM_EFFECTS and name not in state.allow_for_run:
            state.confirmations += 1
            if state.confirmations > self.limits.max_confirmations or self.hooks.confirm is None:
                if state.confirmations > self.limits.max_confirmations:
                    state.stop, state.stop_message = RunOutcome.STOPPED, "too many confirmations in one run"
                return self._error(call_id, "user_declined", "This action needs confirmation and was not run"), True
            self._emit("confirmation", tool=name, call_id=call_id, title=spec.title, summary=preview)
            decision = await self.hooks.confirm(spec, arguments, preview, state.prompt)
            if cancel.cancellation_requested:
                return self._error(call_id, "cancelled", "Cancelled by the user before execution"), True
            if decision not in ("once", "run"):
                self._emit("tool_finished", tool=name, call_id=call_id, title=spec.title,
                           summary="Declined by you", ok=False)
                return self._error(call_id, "user_declined", "The user declined this action"), True
            if decision == "run":
                state.allow_for_run.add(name)
        self._emit("tool_started", tool=name, call_id=call_id, title=spec.title, summary=preview)
        started = time.monotonic()
        outcome = await self.executor.execute(name, arguments, cancel)
        duration = int((time.monotonic() - started) * 1000)
        state.actions += 1
        payload = dict(outcome.payload)
        images = tuple(outcome.images)
        if images and state.images + len(images) > self.limits.max_images:
            images = ()
            payload["images_omitted"] = "image budget for this run is used up"
        state.images += len(images)
        if spec.mutates_candidate or spec.tool in ("generate_trajectory", "accept_candidate", "discard_candidate"):
            state.candidate_touched = True
        key = (name, canonical_json(arguments))
        if outcome.ok:
            state.consecutive_errors = 0
        else:
            state.consecutive_errors += 1
            state.failures[key] += 1
            if state.failures[key] >= self.limits.max_identical_failures:
                state.stop, state.stop_message = RunOutcome.TOOL_ERRORS, "the same call failed repeatedly"
            elif state.consecutive_errors >= self.limits.max_consecutive_tool_errors:
                state.stop, state.stop_message = RunOutcome.TOOL_ERRORS, "too many tool errors in a row"
        self._emit("tool_finished", tool=name, call_id=call_id, title=spec.title, summary=preview,
                   ok=outcome.ok, duration_ms=duration,
                   data={"message": str(payload.get("message", ""))[:300], "images": len(images)})
        return tool_result_block(call_id, payload, is_error=not outcome.ok, images=images), declined

    async def _ask(self, call_id, arguments, state):
        question = str(arguments.get("question", ""))[:500]
        options = tuple(str(option)[:60] for option in arguments.get("options", ()))[:6]
        if not question or self.hooks.ask is None:
            return self._error(call_id, "no_answer", "The user could not be asked; make a reasonable choice")
        self._emit("question", tool="ask_user", call_id=call_id, text=question, data={"options": options})
        answer = await self.hooks.ask(question, options)
        if not answer:
            return self._error(call_id, "no_answer", "The user did not answer")
        return tool_result_block(call_id, {"ok": True, "answer": str(answer)[:500]})

    # --------------------------------------------------------------------- run
    async def run(self, prompt: str, digest: Mapping[str, Any] | str | None = None, cancel=None) -> RunResult:
        cancel = cancel or _Flag()
        if contains_secret(prompt):
            return RunResult(RunOutcome.ERROR, message=(
                "Your message looks like it contains an API key. Remove it and use Claude settings "
                "to store keys securely."))
        if self.session.remaining_usd <= 0:
            return RunResult(RunOutcome.BUDGET, message="Session spend cap reached")
        state = _RunState(prompt=prompt, budget=RunBudget(self.limits.max_run_cost_usd, self.session))
        content = [user_text(prompt)]
        if self.pending_note:
            content.append(user_text(self.pending_note))
            self.pending_note = ""
        if digest:
            encoded = digest if isinstance(digest, str) else json.dumps(digest, sort_keys=True, default=str)
            content.append(user_text(f"<ghostgui_state>{encoded}</ghostgui_state>"))
        self.conversation.add_user(content)
        pending_system = [self._policy_text(state)]
        self._emit("run_started", text=prompt)
        try:
            while state.stop is None:
                if cancel.cancellation_requested:
                    state.stop = RunOutcome.STOPPED
                    break
                if state.turns >= self.limits.max_turns:
                    state.stop, state.stop_message = RunOutcome.TURN_LIMIT, "turn limit reached"
                    break
                if time.monotonic() - state.started > self.limits.max_wall_seconds:
                    state.stop, state.stop_message = RunOutcome.TURN_LIMIT, "time limit reached"
                    break
                if state.budget.exhausted():
                    state.stop, state.stop_message = RunOutcome.BUDGET, "spend cap reached"
                    break
                state.turns += 1
                self._emit("turn_started", data={"turn": state.turns})
                response = await self._request(pending_system, cancel)
                state.budget.add(response.usage, self.model)
                self._emit("usage", data={"cost_usd": state.budget.spent_usd,
                                          "session_usd": self.session.spent_usd,
                                          "input": response.usage.total_input,
                                          "cache_read": response.usage.cache_read,
                                          "output": response.usage.output_tokens,
                                          "served_model": response.usage.served_model or response.model})
                if response.usage.total_input > self.limits.new_conversation_input_tokens:
                    self._emit("notice", text="This conversation is long; start a new one to keep costs down")
                stop = response.stop_reason
                if stop is AgentStopReason.CONTEXT_EXCEEDED:
                    state.stop, state.stop_message = RunOutcome.CONTEXT_FULL, "conversation is full"
                    break
                if stop is AgentStopReason.REFUSAL and response.tool_uses:
                    state.stop, state.stop_message = RunOutcome.REFUSED, response.refusal_category or ""
                    break
                for text in pending_system:
                    self.conversation.add_system(text)
                pending_system = []
                self.conversation.add_assistant(response.content)
                if response.text:
                    state.text = response.text
                if stop is AgentStopReason.REFUSAL:
                    state.stop, state.stop_message = RunOutcome.REFUSED, response.refusal_category or ""
                    break
                if stop is AgentStopReason.PAUSE_TURN:
                    continue
                calls = response.tool_uses
                if not calls:
                    state.stop = RunOutcome.COMPLETED
                    break
                results, declined = [], False
                for call in calls:
                    result, declined = await self._run_call(call, state, cancel, declined)
                    results.append(result)
                self.conversation.add_user(results)
                if cancel.cancellation_requested and state.stop is None:
                    state.stop = RunOutcome.STOPPED
                if state.planning_halted and state.stop is None:
                    state.stop = RunOutcome.PLAN_READY
                if (state.stop is None and not state.budget.notice_sent
                        and state.budget.fraction() >= self.limits.budget_notice_fraction):
                    state.budget.notice_sent = True
                    pending_system.append("Budget notice: about 80% of this run's budget is used. "
                                          "Finish the essential part of the request now.")
        except ProviderCancelledError:
            state.stop = RunOutcome.STOPPED
        except ProviderError as error:
            state.stop, state.stop_message = RunOutcome.ERROR, str(error)
            self._emit("notice", text=str(error), data={"error": type(error).__name__})
        finally:
            pending = self.conversation.pending_tool_uses()
            if pending:
                self.conversation.add_user([self._error(call_id, "cancelled", "Cancelled by the user before execution")
                                            for call_id in pending])
        outcome = state.stop
        if outcome is RunOutcome.COMPLETED and state.deferred:
            outcome = RunOutcome.PLAN_READY
        usage = state.budget.usage
        result = RunResult(
            outcome=outcome, text=state.text, turns=state.turns, actions=state.actions,
            cost_usd=state.budget.spent_usd, input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, cache_read=usage.cache_read,
            message=state.stop_message, deferred=tuple(state.deferred),
            refusal_category=state.stop_message if outcome is RunOutcome.REFUSED else None,
            candidate_touched=state.candidate_touched)
        self._emit("run_finished", text=state.text, data={"outcome": outcome.value, "message": state.stop_message})
        return result

    async def execute_plan(self, deferred, cancel=None) -> list[dict]:
        """Run approved deferred calls exactly as recorded; stop at the first failure."""
        cancel = cancel or _Flag()
        summary = []
        for call in deferred:
            if cancel.cancellation_requested:
                summary.append({"tool": call.tool, "status": "cancelled"})
                break
            spec = self.executor.spec_for(call.tool)
            self._emit("tool_started", tool=call.tool, title=spec.title if spec else call.tool, summary=call.preview)
            outcome = await self.executor.execute(call.tool, call.arguments, cancel)
            self._emit("tool_finished", tool=call.tool, title=spec.title if spec else call.tool,
                       summary=call.preview, ok=outcome.ok,
                       data={"message": str(outcome.payload.get("message", ""))[:300]})
            summary.append({"tool": call.tool, "status": "ok" if outcome.ok else "failed",
                            "message": str(outcome.payload.get("message", ""))[:200]})
            if not outcome.ok:
                break
        self.pending_note = "Approved plan results: " + json.dumps(summary, sort_keys=True)
        return summary

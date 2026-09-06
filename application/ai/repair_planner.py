"""Single-request repair planning for failed semantic operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from typing import Any

from application.ai.errors import (
    ProviderCancelledError,
    ProviderCapabilityError,
)
from application.ai.limits import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AI_OUTPUT_TOKENS,
    MAX_AI_RESPONSE_CHARACTERS,
)
from application.ai.motion_plan import (
    MotionEditPlan,
    editable_tool_definitions,
    motion_repair_response_schema,
    parse_motion_repair_plan,
)
from application.ai.plan_executor import PlanExecutionResult
from application.ai.providers.base import CancellationSignal, LLMProvider
from application.ai.schemas import (
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    Usage,
)
from application.ai.semantic_tools import SemanticToolContext
from application.ai.tool_registry import ToolRegistry


MAX_REPAIR_FAILURE_REASON_CHARACTERS = 1_000
MAX_REPAIR_PROMPT_CHARACTERS = 65_536

MOTION_REPAIR_SYSTEM_PROMPT = """Repair only the failed GhostGUI semantic motion
operations. Return replacement operations only and never repeat successful operations.
Never call tools or emit raw qpos trajectories, arbitrary code, shell commands, filesystem
operations, or GhostGUI method names. Preserve every listed user-authored or protected
constraint and account for successful operations already applied. Encode each replacement
operation's arguments field as a compact JSON object string matching its supplied schema.
GhostGUI will validate and execute replacements locally. This is the only repair attempt."""


class MotionRepairPlannerError(RuntimeError):
    """The bounded replacement-operation request could not complete safely."""


@dataclass(frozen=True)
class MotionRepairPlannerLimits:
    request_timeout_seconds: float = 60.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_response_characters: int = MAX_AI_RESPONSE_CHARACTERS
    max_prompt_characters: int = MAX_REPAIR_PROMPT_CHARACTERS

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0.0
        ):
            raise ValueError("repair request timeout must be positive and finite")
        if not 0 < self.max_output_tokens <= MAX_AI_OUTPUT_TOKENS:
            raise ValueError("repair output-token limit is invalid")
        if not 0 < self.max_response_characters <= MAX_AI_RESPONSE_CHARACTERS:
            raise ValueError("repair response limit is invalid")
        if not 0 < self.max_prompt_characters <= MAX_REPAIR_PROMPT_CHARACTERS:
            raise ValueError("repair prompt limit is invalid")


@dataclass(frozen=True)
class MotionRepairPlanningResult:
    plan: MotionEditPlan
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


class MotionRepairPlanner:
    """Request at most one compact set of replacement operations."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: MotionRepairPlannerLimits | None = None,
        system_prompt: str = MOTION_REPAIR_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("repair system prompt must not be empty")
        self.provider = provider
        self.tools = tools
        self.limits = limits or MotionRepairPlannerLimits()
        self.system_prompt = system_prompt
        self._last_request_started = False

    @property
    def last_request_started(self) -> bool:
        return self._last_request_started

    async def repair(
        self,
        original_intent: str,
        execution: PlanExecutionResult,
        *,
        model: str,
        context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> MotionRepairPlanningResult:
        self._last_request_started = False
        if not execution.failed_operations:
            raise ValueError("repair requires at least one failed operation")
        if not self.provider.capabilities.supports_structured_output:
            raise ProviderCapabilityError(
                "selected provider/model does not support structured motion repair"
            )
        _raise_if_cancelled(cancellation_token)

        updated_context = self.tools.execute(
            "inspect_motion",
            {},
            context=context,
        )
        payload = _repair_payload(original_intent, execution, updated_context)
        messages = self._messages(payload)
        if sum(len(message.text) for message in messages) > self.limits.max_prompt_characters:
            raise MotionRepairPlannerError("motion repair prompt exceeds the local size limit")
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=motion_repair_response_schema(self.tools),
            max_output_tokens=self.limits.max_output_tokens,
        )
        session = context.session
        session.begin_provider_request()
        self._last_request_started = True
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=session.has_changes)
            raise MotionRepairPlannerError("AI repair request timed out") from error
        except BaseException:
            session.finish_provider_request(result_staged=session.has_changes)
            raise
        session.finish_provider_request(result_staged=session.has_changes)

        _raise_if_cancelled(cancellation_token)
        if response.tool_calls:
            raise MotionRepairPlannerError(
                "motion repair returned tool calls instead of replacement operations"
            )
        if len(response.text) > self.limits.max_response_characters:
            raise MotionRepairPlannerError(
                "motion repair response exceeds the local size limit"
            )
        plan = parse_motion_repair_plan(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return MotionRepairPlanningResult(plan, response.usage, transcript)

    def _messages(self, payload: dict[str, Any]) -> list[ProviderMessage]:
        definitions = [
            {
                "name": definition.name,
                "description": definition.description,
                "arguments": dict(definition.input_schema),
            }
            for definition in editable_tool_definitions(self.tools)
        ]
        contract = (
            self.system_prompt
            + "\n\nAllowed replacement operation schemas:\n"
            + json.dumps(definitions, sort_keys=True, separators=(",", ":"))
        )
        user_text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if self.provider.capabilities.supports_system_messages:
            return [
                ProviderMessage(MessageRole.SYSTEM, text=contract),
                ProviderMessage(MessageRole.USER, text=user_text),
            ]
        return [ProviderMessage(
            MessageRole.USER,
            text=f"Instructions:\n{contract}\n\nRepair context:\n{user_text}",
        )]


def _repair_payload(
    original_intent: str,
    execution: PlanExecutionResult,
    updated_context: dict,
) -> dict[str, Any]:
    constraints = updated_context.get("constraints", {})
    recent_user_edits = [
        edit
        for edit in updated_context.get("recent_edits", [])
        if edit.get("author") == "user"
    ]
    return {
        "original_intent": original_intent.strip(),
        "failed_operations": [
            {
                "index": result.index,
                "tool": result.operation.tool,
                "arguments": dict(result.operation.arguments),
                "reason": _bounded_reason(result.error),
            }
            for result in execution.failed_operations
        ],
        "updated_motion_context": {
            key: updated_context.get(key)
            for key in (
                "robot",
                "motion",
                "selection",
                "validation_state",
            )
        },
        "important_user_constraints": {
            "keyframes": constraints.get("keyframes", []),
            "truncated": bool(constraints.get("truncated", False)),
            "recent_user_edits": recent_user_edits,
        },
        "successful_operations_already_applied": [
            {
                "index": result.index,
                "tool": result.operation.tool,
                "arguments": dict(result.operation.arguments),
                "changed": result.changed,
            }
            for result in execution.operations
            if result.succeeded
        ],
    }


def bounded_repair_error(error: Exception) -> str:
    """Return a compact local failure suitable for partial-result UI copy."""

    return _bounded_reason(str(error).strip() or type(error).__name__)


def _bounded_reason(value: str | None) -> str:
    reason = (value or "operation failed").strip() or "operation failed"
    if len(reason) <= MAX_REPAIR_FAILURE_REASON_CHARACTERS:
        return reason
    return reason[: MAX_REPAIR_FAILURE_REASON_CHARACTERS - 1] + "…"


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("motion repair was cancelled")

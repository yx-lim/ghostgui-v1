"""One-shot visual motion planning and optional before/after verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from typing import Mapping

from application.ai.errors import ProviderCancelledError, ProviderCapabilityError
from application.ai.limits import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AI_INSTRUCTION_CHARACTERS,
    MAX_AI_OUTPUT_TOKENS,
    MAX_AI_RESPONSE_CHARACTERS,
)
from application.ai.motion_plan import (
    MAX_PLANNED_OPERATIONS,
    MotionEditPlan,
    PlannedOperation,
    editable_tool_definitions,
    planned_operation_response_schema,
)
from application.ai.plan_executor import (
    PlanExecutionError,
    PlanExecutionResult,
    PlanExecutor,
    local_proposal,
)
from application.ai.providers import CancellationSignal, LLMProvider
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    Usage,
)
from application.ai.semantic_tools import SemanticToolContext
from application.ai.tool_registry import ToolRegistry
from application.ai.visual_critique import VisualObservation


VISUAL_MOTION_PLANNER_SYSTEM_PROMPT = """Inspect the timestamped GhostGUI motion frames
and return one complete semantic edit plan. Return observations at approximate motion times,
never image indexes. Use only the supplied semantic operations and never call tools. Never
emit raw qpos trajectories, arbitrary code, shell commands, filesystem operations, or
GhostGUI method names. Prefer body/logical-frame and End Effector targets. Use Joint Angles
only when the user explicitly requests a Joint Angle or joint group. Preserve user-authored
and protected motion content. GhostGUI will validate and execute the operations locally.
Encode each operation's arguments as a compact JSON object string matching its schema.
Return an empty operations array when no safe visual change is needed."""


VISUAL_VERIFICATION_SYSTEM_PROMPT = """Compare original and candidate GhostGUI motion.
This is read-only verification: do not call tools, propose operations, edit motion, calculate
raw qpos, or emit code. Judge the candidate against the user's goal and preserved constraints.
Refer to approximate motion times from timestamp labels, never image indexes. Return only the
requested structured JSON."""


_OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "time_seconds": {"type": "number", "minimum": 0.0},
        "body_part": {"type": "string"},
        "issue": {"type": "string"},
        "severity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["issue"],
    "additionalProperties": False,
}


VISUAL_VERIFICATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "preferred": {
            "type": "string",
            "enum": ["original", "candidate", "tie"],
        },
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "observations": {
            "type": "array",
            "items": _OBSERVATION_SCHEMA,
            "maxItems": 16,
        },
    },
    "required": ["summary", "preferred", "reasons", "observations"],
    "additionalProperties": False,
}

class VisualRefinementError(RuntimeError):
    """A visual plan or verification result is invalid."""


@dataclass(frozen=True)
class VisualMotionPlannerLimits:
    request_timeout_seconds: float = 60.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_instruction_characters: int = MAX_AI_INSTRUCTION_CHARACTERS
    max_response_characters: int = MAX_AI_RESPONSE_CHARACTERS

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0.0
        ):
            raise ValueError("visual planning timeout must be positive and finite")
        if not 0 < self.max_output_tokens <= MAX_AI_OUTPUT_TOKENS:
            raise ValueError("visual planning output-token limit is invalid")
        if not 0 < self.max_instruction_characters <= MAX_AI_INSTRUCTION_CHARACTERS:
            raise ValueError("visual planning instruction limit is invalid")
        if not 0 < self.max_response_characters <= MAX_AI_RESPONSE_CHARACTERS:
            raise ValueError("visual planning response limit is invalid")


@dataclass(frozen=True)
class VisualMotionPlanningResult:
    observations: tuple[VisualObservation, ...]
    plan: MotionEditPlan
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


@dataclass(frozen=True)
class VisualMotionRunResult:
    planning: VisualMotionPlanningResult
    execution: PlanExecutionResult
    text: str
    proposal_lines: tuple[str, ...]
    provider_requests: int = 1

    def __post_init__(self) -> None:
        if self.provider_requests != 1:
            raise ValueError("visual refinement must use exactly one provider request")

    @property
    def observations(self) -> tuple[VisualObservation, ...]:
        return self.planning.observations

    @property
    def usage(self) -> Usage:
        return self.planning.usage

    @property
    def transcript(self) -> tuple[ProviderMessage, ...]:
        return self.planning.transcript


@dataclass(frozen=True)
class VisualVerification:
    summary: str
    preferred: ImageVariant | None
    reasons: tuple[str, ...]
    observations: tuple[VisualObservation, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("visual verification summary must not be empty")
        if len(self.reasons) > 8 or len(self.observations) > 16:
            raise ValueError("visual verification exceeds its bounded schema")
        if any(not value.strip() for value in self.reasons):
            raise ValueError("visual verification reasons must not be empty")


@dataclass(frozen=True)
class VisualVerificationResult:
    verification: VisualVerification
    usage: Usage
    transcript: tuple[ProviderMessage, ...]
    provider_requests: int = 1

    def __post_init__(self) -> None:
        if self.provider_requests != 1:
            raise ValueError("visual verification must use exactly one provider request")


def visual_motion_plan_response_schema(registry: ToolRegistry) -> dict:
    """Build the combined observation-and-semantic-operation response schema."""

    return {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": _OBSERVATION_SCHEMA,
                "maxItems": 16,
            },
            "operations": {
                "type": "array",
                "items": planned_operation_response_schema(registry),
                "maxItems": MAX_PLANNED_OPERATIONS,
            },
        },
        "required": ["observations", "operations"],
        "additionalProperties": False,
    }


class VisualMotionPlanner:
    """Request observations and one complete semantic plan in one image request."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: VisualMotionPlannerLimits | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.limits = limits or VisualMotionPlannerLimits()

    async def plan(
        self,
        user_goal: str,
        *,
        model: str,
        motion_context: Mapping,
        motion_frames: tuple[MotionFrameImage, ...],
        semantic_context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> VisualMotionPlanningResult:
        _validate_goal(user_goal, self.limits.max_instruction_characters)
        _validate_motion_frames(motion_frames)
        _require_visual_capabilities(self.provider)
        _raise_if_cancelled(cancellation_token)

        messages = self._messages(user_goal, motion_context, motion_frames)
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=visual_motion_plan_response_schema(self.tools),
            max_output_tokens=self.limits.max_output_tokens,
        )
        session = semantic_context.session
        session.begin_provider_request()
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=session.has_changes)
            raise VisualRefinementError("visual planning request timed out") from error
        except BaseException:
            session.finish_provider_request(result_staged=session.has_changes)
            raise
        session.finish_provider_request(result_staged=session.has_changes)

        _raise_if_cancelled(cancellation_token)
        if response.tool_calls:
            raise VisualRefinementError(
                "visual planner returned tool calls instead of a structured plan"
            )
        if len(response.text) > self.limits.max_response_characters:
            raise VisualRefinementError(
                "visual planning response exceeds the local size limit"
            )
        observations, plan = parse_visual_motion_plan(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return VisualMotionPlanningResult(
            observations,
            plan,
            response.usage,
            transcript,
        )

    def _messages(
        self,
        user_goal: str,
        motion_context: Mapping,
        motion_frames: tuple[MotionFrameImage, ...],
    ) -> list[ProviderMessage]:
        operations = [
            {
                "name": definition.name,
                "description": definition.description,
                "arguments": dict(definition.input_schema),
            }
            for definition in editable_tool_definitions(self.tools)
        ]
        user_text = (
            "GhostGUI motion context:\n"
            + _compact_json(motion_context)
            + "\n\nAllowed semantic edit operations:\n"
            + _compact_json(operations)
            + f"\n\nUser goal:\n{user_goal.strip()}"
        )
        if self.provider.capabilities.supports_system_messages:
            return [
                ProviderMessage(
                    MessageRole.SYSTEM,
                    text=VISUAL_MOTION_PLANNER_SYSTEM_PROMPT,
                ),
                ProviderMessage(
                    MessageRole.USER,
                    text=user_text,
                    motion_frames=motion_frames,
                ),
            ]
        return [ProviderMessage(
            MessageRole.USER,
            text=(
                f"Instructions:\n{VISUAL_MOTION_PLANNER_SYSTEM_PROMPT}\n\n"
                f"{user_text}"
            ),
            motion_frames=motion_frames,
        )]


class VisualMotionWorkflow:
    """Compose one multimodal plan request with deterministic local execution."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: VisualMotionPlannerLimits | None = None,
    ) -> None:
        self.planner = VisualMotionPlanner(provider, tools, limits=limits)
        self.executor = PlanExecutor(tools)

    async def run(
        self,
        user_goal: str,
        *,
        model: str,
        motion_context: Mapping,
        motion_frames: tuple[MotionFrameImage, ...],
        semantic_context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> VisualMotionRunResult:
        planning = await self.planner.plan(
            user_goal,
            model=model,
            motion_context=motion_context,
            motion_frames=motion_frames,
            semantic_context=semantic_context,
            cancellation_token=cancellation_token,
        )
        execution = self.executor.execute(
            planning.plan,
            context=semantic_context,
            cancellation_token=cancellation_token,
        )
        if execution.validation_passed is False:
            issues = execution.validation.get("issues", [])
            detail = "; ".join(str(issue) for issue in issues) or "unknown issue"
            raise PlanExecutionError(f"staged motion validation failed: {detail}")
        text, lines = local_proposal(execution)
        return VisualMotionRunResult(planning, execution, text, lines)


class VisualVerifier:
    """Perform one optional read-only before/after comparison request."""

    def __init__(self, provider: LLMProvider, *, timeout_seconds: float = 60.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("visual verification timeout must be positive and finite")
        self.provider = provider
        self.timeout_seconds = float(timeout_seconds)

    async def run(
        self,
        user_goal: str,
        *,
        model: str,
        motion_context: Mapping,
        comparison_frames: tuple[MotionFrameImage, ...],
        cancellation_token: CancellationSignal | None = None,
    ) -> VisualVerificationResult:
        _validate_goal(user_goal, MAX_AI_INSTRUCTION_CHARACTERS)
        _validate_comparison_frames(comparison_frames)
        _require_visual_capabilities(self.provider)
        _raise_if_cancelled(cancellation_token)

        user_text = (
            "GhostGUI motion context:\n"
            + _compact_json(motion_context)
            + f"\n\nUser goal:\n{user_goal.strip()}"
        )
        if self.provider.capabilities.supports_system_messages:
            messages = [
                ProviderMessage(
                    MessageRole.SYSTEM,
                    text=VISUAL_VERIFICATION_SYSTEM_PROMPT,
                ),
                ProviderMessage(
                    MessageRole.USER,
                    text=user_text,
                    motion_frames=comparison_frames,
                ),
            ]
        else:
            messages = [ProviderMessage(
                MessageRole.USER,
                text=(
                    f"Instructions:\n{VISUAL_VERIFICATION_SYSTEM_PROMPT}\n\n"
                    f"{user_text}"
                ),
                motion_frames=comparison_frames,
            )]
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=VISUAL_VERIFICATION_RESPONSE_SCHEMA,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise VisualRefinementError("visual verification request timed out") from error
        _raise_if_cancelled(cancellation_token)
        if len(response.text) > MAX_AI_RESPONSE_CHARACTERS:
            raise VisualRefinementError(
                "visual verification response exceeds the local size limit"
            )
        if response.tool_calls:
            raise VisualRefinementError("visual verification cannot execute tool calls")
        verification = parse_visual_verification(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return VisualVerificationResult(verification, response.usage, transcript)


def parse_visual_motion_plan(
    text: str,
) -> tuple[tuple[VisualObservation, ...], MotionEditPlan]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise VisualRefinementError("provider returned malformed visual-plan JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"observations", "operations"}:
        raise VisualRefinementError("visual motion plan has invalid fields")
    if not isinstance(payload["observations"], list):
        raise VisualRefinementError("visual motion plan observations must be a list")
    if not isinstance(payload["operations"], list):
        raise VisualRefinementError("visual motion plan operations must be a list")
    observations = tuple(
        _parse_observation(value) for value in payload["observations"]
    )
    if len(observations) > 16:
        raise VisualRefinementError("visual motion plan has too many observations")
    operations = tuple(_parse_operation(value) for value in payload["operations"])
    if len(operations) > MAX_PLANNED_OPERATIONS:
        raise VisualRefinementError("visual motion plan exceeds the operation limit")
    plan = MotionEditPlan(
        summary="Visual semantic motion plan",
        needs_clarification=False,
        clarification_question=None,
        operations=operations,
    )
    return observations, plan


def parse_visual_verification(text: str) -> VisualVerification:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise VisualRefinementError(
            "provider returned malformed visual verification JSON"
        ) from error
    required = {"summary", "preferred", "reasons", "observations"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise VisualRefinementError("visual verification has invalid fields")
    if payload["preferred"] not in {"original", "candidate", "tie"}:
        raise VisualRefinementError("visual verification preferred value is invalid")
    if not isinstance(payload["summary"], str):
        raise VisualRefinementError("visual verification summary must be text")
    if not isinstance(payload["reasons"], list) or not all(
        isinstance(value, str) for value in payload["reasons"]
    ):
        raise VisualRefinementError("visual verification reasons must be text")
    if not isinstance(payload["observations"], list):
        raise VisualRefinementError("visual verification observations must be a list")
    preferred = (
        None
        if payload["preferred"] == "tie"
        else ImageVariant(payload["preferred"])
    )
    try:
        return VisualVerification(
            payload["summary"],
            preferred,
            tuple(payload["reasons"]),
            tuple(_parse_observation(value) for value in payload["observations"]),
        )
    except ValueError as error:
        raise VisualRefinementError(str(error)) from error


def _parse_operation(value) -> PlannedOperation:
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise VisualRefinementError("visual planned operation has invalid fields")
    if not isinstance(value["tool"], str) or not isinstance(value["arguments"], str):
        raise VisualRefinementError("visual planned operation fields are invalid")
    try:
        arguments = json.loads(value["arguments"])
    except json.JSONDecodeError as error:
        raise VisualRefinementError(
            "visual planned operation arguments contain malformed JSON"
        ) from error
    if not isinstance(arguments, Mapping):
        raise VisualRefinementError(
            "visual planned operation arguments must decode to an object"
        )
    try:
        return PlannedOperation(value["tool"], arguments)
    except (TypeError, ValueError) as error:
        raise VisualRefinementError(str(error)) from error


def _parse_observation(value) -> VisualObservation:
    allowed = {"time_seconds", "body_part", "issue", "severity"}
    if not isinstance(value, dict) or "issue" not in value or not set(value) <= allowed:
        raise VisualRefinementError("visual observation has invalid fields")
    time_seconds = value.get("time_seconds")
    severity = value.get("severity")
    body_part = value.get("body_part")
    if time_seconds is not None and (
        isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float))
    ):
        raise VisualRefinementError("visual observation time must be numeric")
    if severity is not None and (
        isinstance(severity, bool) or not isinstance(severity, (int, float))
    ):
        raise VisualRefinementError("visual observation severity must be numeric")
    if body_part is not None and not isinstance(body_part, str):
        raise VisualRefinementError("visual observation body part must be text")
    if not isinstance(value["issue"], str):
        raise VisualRefinementError("visual observation issue must be text")
    try:
        return VisualObservation(
            None if time_seconds is None else float(time_seconds),
            body_part,
            value["issue"],
            None if severity is None else float(severity),
        )
    except ValueError as error:
        raise VisualRefinementError(str(error)) from error


def _validate_goal(goal: str, max_characters: int) -> None:
    if not goal.strip():
        raise ValueError("visual motion goal must not be empty")
    if len(goal) > max_characters:
        raise VisualRefinementError("visual motion goal exceeds the local size limit")


def _validate_motion_frames(frames: tuple[MotionFrameImage, ...]) -> None:
    if not 4 <= len(frames) <= 8:
        raise VisualRefinementError("visual refinement requires 4--8 motion frames")
    ids = {frame.comparison_id for frame in frames}
    if len(ids) != len(frames):
        raise VisualRefinementError("visual refinement frame identifiers must be unique")
    if any(frame.variant is not ImageVariant.CANDIDATE for frame in frames):
        raise VisualRefinementError("visual refinement requires staged candidate frames")
    ProviderMessage(MessageRole.USER, motion_frames=frames)


def _validate_comparison_frames(frames: tuple[MotionFrameImage, ...]) -> None:
    pairs: dict[str, set[ImageVariant]] = {}
    if not 8 <= len(frames) <= 16:
        raise VisualRefinementError("visual verification requires 4--8 image pairs")
    for frame in frames:
        pairs.setdefault(frame.comparison_id, set()).add(frame.variant)
    required = {ImageVariant.ORIGINAL, ImageVariant.CANDIDATE}
    if not 4 <= len(pairs) <= 8 or any(variants != required for variants in pairs.values()):
        raise VisualRefinementError(
            "each visual verification timestamp requires original and candidate images"
        )
    ProviderMessage(MessageRole.USER, motion_frames=frames)


def _require_visual_capabilities(provider: LLMProvider) -> None:
    capabilities = provider.capabilities
    if not capabilities.supports_vision:
        raise ProviderCapabilityError("selected provider/model does not support vision")
    if not capabilities.supports_structured_output:
        raise ProviderCapabilityError(
            "selected provider/model does not support structured visual planning"
        )


def _compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("visual request was cancelled")

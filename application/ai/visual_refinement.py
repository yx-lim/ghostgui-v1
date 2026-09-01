"""Bounded observe-plan-edit steps for visual motion refinement."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Mapping

from application.ai.agent import AgentLimits, AgentRunResult, GhostGUIAgent
from application.ai.errors import ProviderCancelledError, ProviderCapabilityError
from application.ai.limits import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AI_INSTRUCTION_CHARACTERS,
    MAX_AI_RESPONSE_CHARACTERS,
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


VISUAL_COMPARISON_SYSTEM_PROMPT = """Compare original and candidate GhostGUI motion.
Observe first, then provide a small semantic refinement plan. Do not call tools, edit motion,
calculate raw qpos, or emit code. Judge the candidate against the user's stated goal and
preserved constraints. Refer to approximate motion times from timestamp labels, never image
indexes. Request another refinement only for a clear remaining visual issue. Return only the
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


VISUAL_COMPARISON_RESPONSE_SCHEMA = {
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
        "should_refine": {"type": "boolean"},
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
    },
    "required": [
        "summary",
        "preferred",
        "reasons",
        "observations",
        "should_refine",
        "plan",
    ],
    "additionalProperties": False,
}


class VisualRefinementError(RuntimeError):
    """Raised when a comparison or bounded refinement step is invalid."""


@dataclass(frozen=True)
class VisualComparison:
    summary: str
    preferred: ImageVariant | None
    reasons: tuple[str, ...]
    observations: tuple[VisualObservation, ...]
    should_refine: bool
    plan: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("visual comparison summary must not be empty")
        if len(self.reasons) > 8 or len(self.observations) > 16 or len(self.plan) > 4:
            raise ValueError("visual comparison exceeds its bounded schema")
        if any(not value.strip() for value in (*self.reasons, *self.plan)):
            raise ValueError("visual comparison reasons and plan items must not be empty")
        if self.should_refine and not self.plan:
            raise ValueError("visual comparison requesting refinement requires a plan")


@dataclass(frozen=True)
class VisualComparisonResult:
    comparison: VisualComparison
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


@dataclass(frozen=True)
class VisualRefinementStepResult:
    comparison_result: VisualComparisonResult
    edit_result: AgentRunResult | None
    motion_changed: bool


@dataclass(frozen=True)
class VisualRefinementLimits:
    max_edit_iterations: int = 2
    comparison_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_edit_iterations <= 3:
            raise ValueError("visual refinement must allow only 1--3 edit iterations")
        if (
            not math.isfinite(self.comparison_timeout_seconds)
            or self.comparison_timeout_seconds <= 0.0
        ):
            raise ValueError("visual comparison timeout must be positive and finite")


class VisualRefinementAction(str, Enum):
    REFINE = "refine"
    ASSESS_ONLY = "assess_only"
    COMPLETE = "complete"


@dataclass
class VisualRefinementProgress:
    limits: VisualRefinementLimits
    completed_edit_iterations: int = 0

    def after_step(self, result: VisualRefinementStepResult) -> VisualRefinementAction:
        if not result.motion_changed:
            return VisualRefinementAction.COMPLETE
        if self.completed_edit_iterations >= self.limits.max_edit_iterations:
            raise VisualRefinementError("visual refinement edit limit exceeded")
        self.completed_edit_iterations += 1
        if self.completed_edit_iterations < self.limits.max_edit_iterations:
            return VisualRefinementAction.REFINE
        return VisualRefinementAction.ASSESS_ONLY


class VisualComparator:
    """One read-only before/after comparison over paired timestamped frames."""

    def __init__(self, provider: LLMProvider, *, timeout_seconds: float = 60.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("visual comparison timeout must be positive and finite")
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
    ) -> VisualComparisonResult:
        if not user_goal.strip():
            raise ValueError("visual refinement goal must not be empty")
        if len(user_goal) > MAX_AI_INSTRUCTION_CHARACTERS:
            raise VisualRefinementError(
                "visual refinement goal exceeds the local size limit"
            )
        _validate_comparison_frames(comparison_frames)
        capabilities = self.provider.capabilities
        if not capabilities.supports_vision:
            raise ProviderCapabilityError("selected provider/model does not support vision")
        if not capabilities.supports_structured_output:
            raise ProviderCapabilityError(
                "selected provider/model does not support structured visual comparison"
            )
        _raise_if_cancelled(cancellation_token)
        context_text = json.dumps(
            dict(motion_context), sort_keys=True, separators=(",", ":")
        )
        user_text = (
            f"GhostGUI motion context:\n{context_text}\n\n"
            f"User goal:\n{user_goal.strip()}"
        )
        if capabilities.supports_system_messages:
            messages = [
                ProviderMessage(MessageRole.SYSTEM, text=VISUAL_COMPARISON_SYSTEM_PROMPT),
                ProviderMessage(
                    MessageRole.USER,
                    text=user_text,
                    motion_frames=comparison_frames,
                ),
            ]
        else:
            messages = [ProviderMessage(
                MessageRole.USER,
                text=f"Instructions:\n{VISUAL_COMPARISON_SYSTEM_PROMPT}\n\n{user_text}",
                motion_frames=comparison_frames,
            )]
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=VISUAL_COMPARISON_RESPONSE_SCHEMA,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise VisualRefinementError("visual comparison request timed out") from error
        _raise_if_cancelled(cancellation_token)
        if len(response.text) > MAX_AI_RESPONSE_CHARACTERS:
            raise VisualRefinementError(
                "visual comparison response exceeds the local size limit"
            )
        if response.tool_calls:
            raise VisualRefinementError("visual comparison cannot execute tool calls")
        comparison = parse_visual_comparison(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return VisualComparisonResult(comparison, response.usage, transcript)


class VisualRefinementStep:
    """Run one Observe/Plan step and optionally one semantic edit agent run."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: VisualRefinementLimits | None = None,
        agent_limits: AgentLimits | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.limits = limits or VisualRefinementLimits()
        self.agent_limits = agent_limits

    async def run(
        self,
        user_goal: str,
        *,
        model: str,
        motion_context: Mapping,
        comparison_frames: tuple[MotionFrameImage, ...],
        semantic_context: SemanticToolContext,
        allow_edit: bool = True,
        cancellation_token: CancellationSignal | None = None,
    ) -> VisualRefinementStepResult:
        if allow_edit and not self.provider.capabilities.supports_tools:
            raise ProviderCapabilityError(
                "selected provider/model does not support semantic visual refinement"
            )
        comparison_result = await VisualComparator(
            self.provider,
            timeout_seconds=self.limits.comparison_timeout_seconds,
        ).run(
            user_goal,
            model=model,
            motion_context=motion_context,
            comparison_frames=comparison_frames,
            cancellation_token=cancellation_token,
        )
        comparison = comparison_result.comparison
        if not allow_edit or not comparison.should_refine:
            return VisualRefinementStepResult(comparison_result, None, False)

        instruction = _refinement_instruction(user_goal, comparison)
        before_edit_count = len(semantic_context.session.edits)
        edit_result = await GhostGUIAgent(
            self.provider,
            self.tools,
            limits=self.agent_limits,
        ).run(
            instruction,
            model=model,
            context=semantic_context,
            cancellation_token=cancellation_token,
        )
        changed = len(semantic_context.session.edits) > before_edit_count
        return VisualRefinementStepResult(comparison_result, edit_result, changed)


def parse_visual_comparison(text: str) -> VisualComparison:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise VisualRefinementError("provider returned malformed comparison JSON") from error
    required = {
        "summary", "preferred", "reasons", "observations", "should_refine", "plan"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise VisualRefinementError("visual comparison has invalid fields")
    if payload["preferred"] not in {"original", "candidate", "tie"}:
        raise VisualRefinementError("visual comparison preferred value is invalid")
    if not isinstance(payload["summary"], str):
        raise VisualRefinementError("visual comparison summary must be text")
    if not isinstance(payload["reasons"], list) or not all(
        isinstance(value, str) for value in payload["reasons"]
    ):
        raise VisualRefinementError("visual comparison reasons must be text")
    if not isinstance(payload["plan"], list) or not all(
        isinstance(value, str) for value in payload["plan"]
    ):
        raise VisualRefinementError("visual comparison plan must be text")
    if not isinstance(payload["should_refine"], bool):
        raise VisualRefinementError("visual comparison should_refine must be boolean")
    if not isinstance(payload["observations"], list):
        raise VisualRefinementError("visual comparison observations must be a list")
    observations = tuple(_parse_observation(value) for value in payload["observations"])
    preferred = (
        None
        if payload["preferred"] == "tie"
        else ImageVariant(payload["preferred"])
    )
    try:
        return VisualComparison(
            payload["summary"],
            preferred,
            tuple(payload["reasons"]),
            observations,
            payload["should_refine"],
            tuple(payload["plan"]),
        )
    except ValueError as error:
        raise VisualRefinementError(str(error)) from error


def _parse_observation(value) -> VisualObservation:
    allowed = {"time_seconds", "body_part", "issue", "severity"}
    if not isinstance(value, dict) or "issue" not in value or not set(value) <= allowed:
        raise VisualRefinementError("visual comparison observation has invalid fields")
    time_seconds = value.get("time_seconds")
    severity = value.get("severity")
    body_part = value.get("body_part")
    if time_seconds is not None and (
        isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float))
    ):
        raise VisualRefinementError("comparison observation time must be numeric")
    if severity is not None and (
        isinstance(severity, bool) or not isinstance(severity, (int, float))
    ):
        raise VisualRefinementError("comparison observation severity must be numeric")
    if body_part is not None and not isinstance(body_part, str):
        raise VisualRefinementError("comparison observation body part must be text")
    if not isinstance(value["issue"], str):
        raise VisualRefinementError("comparison observation issue must be text")
    try:
        return VisualObservation(
            None if time_seconds is None else float(time_seconds),
            body_part,
            value["issue"],
            None if severity is None else float(severity),
        )
    except ValueError as error:
        raise VisualRefinementError(str(error)) from error


def _validate_comparison_frames(frames: tuple[MotionFrameImage, ...]) -> None:
    pairs: dict[str, set[ImageVariant]] = {}
    if not 8 <= len(frames) <= 16:
        raise VisualRefinementError("visual comparison requires 4--8 image pairs")
    for frame in frames:
        pairs.setdefault(frame.comparison_id, set()).add(frame.variant)
    required = {ImageVariant.ORIGINAL, ImageVariant.CANDIDATE}
    if not 4 <= len(pairs) <= 8 or any(variants != required for variants in pairs.values()):
        raise VisualRefinementError(
            "each visual comparison timestamp requires original and candidate images"
        )
    # ProviderMessage validates that each comparison id also uses one exact time.
    ProviderMessage(MessageRole.USER, motion_frames=frames)


def _refinement_instruction(user_goal: str, comparison: VisualComparison) -> str:
    observations = "\n".join(
        "- " + (
            ""
            if value.time_seconds is None
            else f"around t={value.time_seconds:.3f} s: "
        ) + value.issue
        for value in comparison.observations
    ) or "- No additional timestamped observation."
    plan = "\n".join(f"- {value}" for value in comparison.plan)
    return (
        f"Original user goal:\n{user_goal.strip()}\n\n"
        f"Visual observations:\n{observations}\n\n"
        f"Approved semantic refinement plan:\n{plan}\n\n"
        "Apply only this plan through GhostGUI semantic tools. Preserve user-authored "
        "and protected motion, do not generate raw qpos, and summarize the changes."
    )


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("visual refinement request was cancelled")

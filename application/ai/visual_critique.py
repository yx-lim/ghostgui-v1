"""Read-only, structured visual critique over timestamped motion frames."""

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
    MAX_AI_RESPONSE_CHARACTERS,
)
from application.ai.providers import CancellationSignal, LLMProvider
from application.ai.schemas import (
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    Usage,
)


VISUAL_CRITIQUE_SYSTEM_PROMPT = """You are observing robot motion in GhostGUI.
Critique only: do not request tools and do not edit or generate motion.
Describe visible motion issues conservatively. Associate each observation with an
approximate motion time in seconds from the supplied timestamp labels whenever possible;
do not answer with image indexes. Return only the requested structured JSON."""


VISUAL_CRITIQUE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "observations": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "time_seconds": {"type": "number", "minimum": 0.0},
                    "body_part": {"type": "string"},
                    "issue": {"type": "string"},
                    "severity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["issue"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "observations"],
    "additionalProperties": False,
}


class VisualCritiqueError(RuntimeError):
    """Raised when critique input or structured output is invalid."""


@dataclass(frozen=True)
class VisualObservation:
    time_seconds: float | None
    body_part: str | None
    issue: str
    severity: float | None

    def __post_init__(self) -> None:
        if self.time_seconds is not None and (
            not math.isfinite(self.time_seconds) or self.time_seconds < 0.0
        ):
            raise ValueError("observation time must be finite and non-negative")
        if self.body_part is not None and not self.body_part.strip():
            raise ValueError("observation body part must be non-empty when provided")
        if not self.issue.strip():
            raise ValueError("observation issue must not be empty")
        if self.severity is not None and (
            not math.isfinite(self.severity) or not 0.0 <= self.severity <= 1.0
        ):
            raise ValueError("observation severity must be between 0 and 1")


@dataclass(frozen=True)
class VisualCritique:
    summary: str
    observations: tuple[VisualObservation, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("visual critique summary must not be empty")
        if len(self.observations) > 16:
            raise ValueError("visual critique contains too many observations")


@dataclass(frozen=True)
class VisualCritiqueResult:
    critique: VisualCritique
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


class VisualCritic:
    """Perform one bounded multimodal observation request with no edit tools."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0.0:
            raise ValueError("critique request timeout must be positive and finite")
        self.provider = provider
        self.request_timeout_seconds = float(request_timeout_seconds)

    async def run(
        self,
        instruction: str,
        *,
        model: str,
        motion_context: Mapping,
        motion_frames: tuple[MotionFrameImage, ...],
        cancellation_token: CancellationSignal | None = None,
    ) -> VisualCritiqueResult:
        if not instruction.strip():
            raise ValueError("visual critique instruction must not be empty")
        if len(instruction) > MAX_AI_INSTRUCTION_CHARACTERS:
            raise VisualCritiqueError(
                "visual critique instruction exceeds the local size limit"
            )
        if not 4 <= len(motion_frames) <= 8:
            raise VisualCritiqueError("visual critique requires 4--8 motion frames")
        capabilities = self.provider.capabilities
        if not capabilities.supports_vision:
            raise ProviderCapabilityError("selected provider/model does not support vision")
        if not capabilities.supports_structured_output:
            raise ProviderCapabilityError(
                "selected provider/model does not support structured visual critique"
            )
        _raise_if_cancelled(cancellation_token)

        context_text = json.dumps(
            dict(motion_context), sort_keys=True, separators=(",", ":")
        )
        user_text = (
            f"GhostGUI motion context:\n{context_text}\n\n"
            f"Critique request:\n{instruction.strip()}"
        )
        if capabilities.supports_system_messages:
            messages = [
                ProviderMessage(MessageRole.SYSTEM, text=VISUAL_CRITIQUE_SYSTEM_PROMPT),
                ProviderMessage(
                    MessageRole.USER,
                    text=user_text,
                    motion_frames=motion_frames,
                ),
            ]
        else:
            messages = [ProviderMessage(
                MessageRole.USER,
                text=f"Instructions:\n{VISUAL_CRITIQUE_SYSTEM_PROMPT}\n\n{user_text}",
                motion_frames=motion_frames,
            )]
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=VISUAL_CRITIQUE_RESPONSE_SCHEMA,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise VisualCritiqueError("visual critique request timed out") from error
        _raise_if_cancelled(cancellation_token)
        if len(response.text) > MAX_AI_RESPONSE_CHARACTERS:
            raise VisualCritiqueError(
                "visual critique response exceeds the local size limit"
            )
        if response.tool_calls:
            raise VisualCritiqueError("critique-only mode cannot execute tool calls")
        critique = parse_visual_critique(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return VisualCritiqueResult(critique, response.usage, transcript)


def parse_visual_critique(text: str) -> VisualCritique:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise VisualCritiqueError("provider returned malformed visual critique JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"summary", "observations"}:
        raise VisualCritiqueError("visual critique must contain summary and observations")
    if not isinstance(payload["summary"], str) or not isinstance(
        payload["observations"], list
    ):
        raise VisualCritiqueError("visual critique fields have invalid types")
    observations = []
    for value in payload["observations"]:
        allowed_fields = {"time_seconds", "body_part", "issue", "severity"}
        if (
            not isinstance(value, dict)
            or "issue" not in value
            or not set(value).issubset(allowed_fields)
        ):
            raise VisualCritiqueError("visual observation has invalid fields")
        time_seconds = value.get("time_seconds")
        body_part = value.get("body_part")
        severity = value.get("severity")
        if time_seconds is not None and (
            isinstance(time_seconds, bool)
            or not isinstance(time_seconds, (int, float))
        ):
            raise VisualCritiqueError("visual observation time must be numeric or null")
        if body_part is not None and not isinstance(body_part, str):
            raise VisualCritiqueError("visual observation body part must be text or null")
        if not isinstance(value["issue"], str):
            raise VisualCritiqueError("visual observation issue must be text")
        if severity is not None and (
            isinstance(severity, bool)
            or not isinstance(severity, (int, float))
        ):
            raise VisualCritiqueError("visual observation severity must be numeric or null")
        try:
            observations.append(VisualObservation(
                None if time_seconds is None else float(time_seconds),
                body_part,
                value["issue"],
                None if severity is None else float(severity),
            ))
        except ValueError as error:
            raise VisualCritiqueError(str(error)) from error
    try:
        return VisualCritique(payload["summary"], tuple(observations))
    except ValueError as error:
        raise VisualCritiqueError(str(error)) from error


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("visual critique request was cancelled")

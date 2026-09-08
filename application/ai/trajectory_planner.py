"""One-shot provider planning for compact local trajectory edits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any, Mapping

from application.ai.context import AIContext
from application.ai.edit_session import AIEditSession
from application.ai.errors import ProviderCapabilityError, ProviderCancelledError
from application.ai.limits import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AI_INSTRUCTION_CHARACTERS,
    MAX_AI_OUTPUT_TOKENS,
    MAX_AI_RESPONSE_CHARACTERS,
    DEFAULT_MOTION_REQUEST_TIMEOUT_SECONDS,
    MAX_MOTION_CONTEXT_CHARACTERS,
    MAX_MOTION_IMAGES,
    MAX_MOTION_REQUEST_TIMEOUT_SECONDS,
)
from application.ai.providers.base import CancellationSignal, LLMProvider
from application.ai.schemas import (
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    Usage,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryEditSpec,
    parse_trajectory_edit_spec,
    trajectory_edit_spec_response_schema,
    trajectory_operation_argument_contracts,
)


MOTION_ASSISTANT_SYSTEM_PROMPT = """You are the motion-generation and editing assistant embedded inside GhostGUI.

The user speaks naturally and does not know GhostGUI internals. You receive the current robot state, relevant motion samples, selected timeline context, timestamped rendered views when available, and compact motion operations GhostGUI can execute.

Infer the intended motion edit and return one complete TrajectoryEditSpec. Encode each operation's arguments as compact JSON object text matching its supplied contract. Do not output raw qpos trajectories, CSV, Python, shell commands, or instructions for operating GhostGUI. Preserve motion outside the requested scope unless continuity requires otherwise. Respect protected content and explicit user constraints. For new motion, use a small set of meaningful numeric sparse Keyframes."""


class TrajectoryPlannerError(RuntimeError):
    """One-shot compact planning could not produce a usable specification."""


@dataclass(frozen=True)
class TrajectoryPlannerLimits:
    max_instruction_characters: int = MAX_AI_INSTRUCTION_CHARACTERS
    max_context_characters: int = MAX_MOTION_CONTEXT_CHARACTERS
    max_response_characters: int = MAX_AI_RESPONSE_CHARACTERS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_images: int = MAX_MOTION_IMAGES
    request_timeout_seconds: float = DEFAULT_MOTION_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.max_instruction_characters <= 0 or self.max_context_characters <= 0:
            raise ValueError("planner text limits must be positive")
        if not 0 < self.max_response_characters <= MAX_AI_RESPONSE_CHARACTERS:
            raise ValueError("planner response limit is invalid")
        if not 0 < self.max_output_tokens <= MAX_AI_OUTPUT_TOKENS:
            raise ValueError("planner output-token limit is invalid")
        if not 0 <= self.max_images <= MAX_MOTION_IMAGES:
            raise ValueError("planner image limit must be between zero and eight")
        if not 0.0 < self.request_timeout_seconds <= MAX_MOTION_REQUEST_TIMEOUT_SECONDS:
            raise ValueError("planner timeout must be positive")


@dataclass(frozen=True)
class TrajectoryPlanningResult:
    spec: TrajectoryEditSpec
    usage: Usage
    transcript: tuple[ProviderMessage, ...]
    provider_requests: int = 1


class TrajectoryPlanner:
    """Ask the configured provider for one complete compact motion spec."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        limits: TrajectoryPlannerLimits | None = None,
        system_prompt: str = MOTION_ASSISTANT_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("motion assistant system prompt must not be empty")
        self.provider = provider
        self.limits = limits or TrajectoryPlannerLimits()
        self.system_prompt = system_prompt

    async def plan(
        self,
        instruction: str,
        *,
        model: str,
        context: AIContext | Mapping[str, Any],
        session: AIEditSession,
        motion_frames: tuple[MotionFrameImage, ...] = (),
        conversation_context: Mapping[str, Any] | None = None,
        cancellation_token: CancellationSignal | None = None,
    ) -> TrajectoryPlanningResult:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("AI edit instruction must not be empty")
        if len(instruction) > self.limits.max_instruction_characters:
            raise TrajectoryPlannerError("AI edit instruction exceeds the local size limit")
        if not self.provider.capabilities.supports_structured_output:
            raise ProviderCapabilityError(
                "selected provider/model does not support structured motion planning"
            )
        if len(motion_frames) > self.limits.max_images:
            raise TrajectoryPlannerError("visual motion context exceeds the image limit")
        if motion_frames and not self.provider.capabilities.supports_vision:
            raise ProviderCapabilityError("selected provider/model does not support vision")
        provider_image_limit = self.provider.capabilities.max_images_per_request
        if motion_frames and (
            provider_image_limit <= 0 or len(motion_frames) > provider_image_limit
        ):
            raise ProviderCapabilityError(
                "selected provider/model image limit is below the supplied context"
            )
        _raise_if_cancelled(cancellation_token)

        payload = context.to_dict() if isinstance(context, AIContext) else dict(context)
        prompt_context = {
            "motion_context": payload,
            "conversation": dict(conversation_context or {}),
            "operation_argument_contracts": trajectory_operation_argument_contracts(),
        }
        encoded = json.dumps(prompt_context, sort_keys=True, separators=(",", ":"))
        if len(encoded) > self.limits.max_context_characters:
            raise TrajectoryPlannerError("automatic motion context exceeds the local size limit")
        user_text = f"Automatic GhostGUI context:\n{encoded}\n\nUser motion request:\n{instruction}"
        messages = self._messages(user_text, motion_frames)
        request = ProviderRequest(
            model=model,
            messages=messages,
            response_schema=trajectory_edit_spec_response_schema(),
            max_output_tokens=self.limits.max_output_tokens,
            temperature=self._temperature(),
        )

        response = await self._request(request, session, cancellation_token)
        _raise_if_cancelled(cancellation_token)

        if response.tool_calls:
            raise TrajectoryPlannerError(
                "motion planner returned tool calls instead of a compact specification"
            )
        if len(response.text) > self.limits.max_response_characters:
            raise TrajectoryPlannerError("motion planner response exceeds the local size limit")
        transcript = messages + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        try:
            spec = parse_trajectory_edit_spec(response.text)
        except Exception as initial_error:
            repair_messages = self._repair_messages(instruction, initial_error)
            repair_request = ProviderRequest(
                model=model,
                messages=repair_messages,
                response_schema=trajectory_edit_spec_response_schema(),
                max_output_tokens=self.limits.max_output_tokens,
                temperature=self._temperature(),
            )
            repair = await self._request(
                repair_request,
                session,
                cancellation_token,
            )
            _raise_if_cancelled(cancellation_token)
            if repair.tool_calls:
                raise TrajectoryPlannerError(
                    "motion specification repair returned unexpected tool calls"
                )
            if len(repair.text) > self.limits.max_response_characters:
                raise TrajectoryPlannerError(
                    "motion specification repair exceeds the local size limit"
                )
            try:
                spec = parse_trajectory_edit_spec(repair.text)
            except Exception as repair_error:
                raise TrajectoryPlannerError(
                    f"motion specification remained invalid after one repair: "
                    f"{repair_error}"
                ) from repair_error
            transcript += repair_messages + (
                ProviderMessage(MessageRole.ASSISTANT, text=repair.text),
            )
            usage = Usage(
                response.usage.input_tokens + repair.usage.input_tokens,
                response.usage.output_tokens + repair.usage.output_tokens,
            )
            return TrajectoryPlanningResult(spec, usage, transcript, 2)
        return TrajectoryPlanningResult(spec, response.usage, transcript)

    async def _request(self, request, session, cancellation_token):
        session.begin_provider_request()
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=False)
            raise TrajectoryPlannerError("motion planning request timed out") from error
        except BaseException:
            session.finish_provider_request(result_staged=False)
            raise
        session.finish_provider_request(result_staged=session.has_changes)
        return response

    def _repair_messages(self, instruction, error):
        payload = json.dumps({
            "original_instruction": instruction,
            "parser_error": str(error),
            "operation_argument_contracts": trajectory_operation_argument_contracts(),
            "repair_requirement": (
                "Return one complete corrected TrajectoryEditSpec. Do not explain."
            ),
        }, sort_keys=True, separators=(",", ":"))
        return self._messages(payload, ())

    def _messages(self, user_text, motion_frames):
        user = ProviderMessage(
            MessageRole.USER,
            text=user_text,
            motion_frames=motion_frames,
        )
        if self.provider.capabilities.supports_system_messages:
            return (
                ProviderMessage(MessageRole.SYSTEM, text=self.system_prompt),
                user,
            )
        return (ProviderMessage(
            MessageRole.USER,
            text=f"Instructions:\n{self.system_prompt}\n\n{user_text}",
            motion_frames=motion_frames,
        ),)

    def _temperature(self):
        return 0.0 if self.provider.capabilities.supports_temperature else None


def _raise_if_cancelled(token):
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("motion planning was cancelled")

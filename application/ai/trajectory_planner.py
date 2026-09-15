"""One-shot provider planning for compact local trajectory edits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import re
from typing import Any, Mapping

from application.ai.context import AIContext
from application.ai.edit_session import AIEditSession
from application.ai.errors import ProviderCapabilityError, ProviderCancelledError
from application.ai.limits import (
    DEFAULT_MOTION_PLANNER_OUTPUT_TOKENS,
    MAX_AI_INSTRUCTION_CHARACTERS,
    MAX_AI_OUTPUT_TOKENS,
    MAX_AI_RESPONSE_CHARACTERS,
    DEFAULT_MOTION_REQUEST_TIMEOUT_SECONDS,
    MAX_MOTION_CONTEXT_CHARACTERS,
    MAX_MOTION_IMAGES,
    MAX_MOTION_REQUEST_TIMEOUT_SECONDS,
)
from application.ai.providers.base import (
    CancellationSignal,
    LLMProvider,
    supports_temperature_for_model,
)
from application.ai.schemas import (
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    Usage,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryEditMode,
    TrajectoryEditSpec,
    TrajectoryOperation,
    TrajectoryOperationType,
    parse_trajectory_edit_spec,
    trajectory_edit_spec_response_schema,
    trajectory_operation_argument_contracts,
)


MOTION_ASSISTANT_SYSTEM_PROMPT = """You are the motion-generation and editing assistant embedded inside GhostGUI.

The user speaks naturally and does not know GhostGUI internals. You receive the current robot state, relevant motion samples, selected timeline context, timestamped rendered views when available, and compact motion operations GhostGUI can execute.

Infer the intended motion edit and return one complete TrajectoryEditSpec. Encode each operation's arguments as compact JSON object text matching its supplied contract. Use semantic operations for simple, precise edits naturally expressed as Joint Angle or Cartesian targets. Use repeat_motion whenever the user asks to repeat, duplicate, loop, or append copies of existing motion; GhostGUI will copy the complete local timeline exactly, so never re-author that request with qpos_keyframes. Use lock_end_effector for spatial holds of a hand, foot, ankle, or similar contact point; map the user's anatomy wording to a registered End Effector name from the supplied model context (for example, a right-ankle spatial hold normally targets right_foot). hold_pose is only for whole-body, Joint Angle, or joint-group values. When the requested whole-body motion appears in robot.motion_primitives, use motion_primitive so GhostGUI can synthesize the model-owned, semantically checked closest available motion; do not free-form qpos for that motion. Otherwise use qpos_keyframes for a new whole-body motion, or for a contact-rich, highly coordinated, novel, or otherwise IK-fragile edit. Always attempt a useful best-effort motion rather than rejecting a request merely because semantic IK is unsuitable. A qpos Keyframe must contain the complete ordered vector described by the active model's qpos layout. Keep qpos plans compact: use 4-8 meaningful anchors whenever that is sufficient and round finite numeric values to no more than six decimal places. Do not output CSV, dense sample-by-sample trajectories, Python, shell commands, direct hardware commands, or instructions for operating GhostGUI. Preserve motion outside the requested scope unless continuity requires otherwise. Respect protected content and explicit user constraints. Use only a small set of meaningful sparse Keyframes."""


class TrajectoryPlannerError(RuntimeError):
    """One-shot compact planning could not produce a usable specification."""

    def __init__(self, message: str, *, diagnostic_details=None) -> None:
        super().__init__(message)
        self.diagnostic_details = dict(diagnostic_details or {})


@dataclass(frozen=True)
class TrajectoryPlannerLimits:
    max_instruction_characters: int = MAX_AI_INSTRUCTION_CHARACTERS
    max_context_characters: int = MAX_MOTION_CONTEXT_CHARACTERS
    max_response_characters: int = MAX_AI_RESPONSE_CHARACTERS
    max_output_tokens: int = DEFAULT_MOTION_PLANNER_OUTPUT_TOKENS
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
    requests: tuple[ProviderRequest, ...] = ()
    responses: tuple[ProviderResponse, ...] = ()
    parser_errors: tuple[str | None, ...] = ()


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
            temperature=self._temperature(model),
        )

        requests = [request]
        responses = [
            await self._request(request, session, cancellation_token)
        ]
        attempt_errors: list[str | None] = [None]
        response = responses[-1]
        if response.stop_reason is StopReason.MAX_TOKENS:
            attempt_errors[-1] = _output_limit_attempt_error(request)
            retry_tokens = min(
                MAX_AI_OUTPUT_TOKENS,
                int(request.max_output_tokens or self.limits.max_output_tokens) * 2,
            )
            if retry_tokens <= int(request.max_output_tokens or 0):
                raise _output_limit_error(requests, responses)
            retry_request = replace(request, max_output_tokens=retry_tokens)
            requests.append(retry_request)
            responses.append(
                await self._request(retry_request, session, cancellation_token)
            )
            attempt_errors.append(None)
            response = responses[-1]
            if response.stop_reason is StopReason.MAX_TOKENS:
                attempt_errors[-1] = _output_limit_attempt_error(retry_request)
                raise _output_limit_error(requests, responses)
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
            spec = _apply_registered_primitive_policy(spec, instruction, payload)
        except Exception as initial_error:
            attempt_errors[-1] = str(initial_error)
            repair_messages = self._repair_messages(instruction, initial_error)
            repair_request = ProviderRequest(
                model=model,
                messages=repair_messages,
                response_schema=trajectory_edit_spec_response_schema(),
                max_output_tokens=self.limits.max_output_tokens,
                temperature=self._temperature(model),
            )
            requests.append(repair_request)
            repair = await self._request(
                repair_request,
                session,
                cancellation_token,
            )
            responses.append(repair)
            attempt_errors.append(None)
            _raise_if_cancelled(cancellation_token)
            if repair.stop_reason is StopReason.MAX_TOKENS:
                attempt_errors[-1] = _output_limit_attempt_error(repair_request)
                raise _output_limit_error(requests, responses)
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
                spec = _apply_registered_primitive_policy(spec, instruction, payload)
            except Exception as repair_error:
                raise TrajectoryPlannerError(
                    f"motion specification remained invalid after one repair: "
                    f"{repair_error}"
                ) from repair_error
            transcript += repair_messages + (
                ProviderMessage(MessageRole.ASSISTANT, text=repair.text),
            )
            return TrajectoryPlanningResult(
                spec,
                _combined_usage(responses),
                transcript,
                len(requests),
                tuple(requests),
                tuple(responses),
                tuple(attempt_errors),
            )
        return TrajectoryPlanningResult(
            spec,
            _combined_usage(responses),
            transcript,
            len(requests),
            tuple(requests),
            tuple(responses),
            tuple(attempt_errors),
        )

    async def _request(self, request, session, cancellation_token):
        session.begin_provider_request()
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=False)
            metrics = _request_metrics(request)
            metrics["timeout_seconds"] = self.limits.request_timeout_seconds
            raise TrajectoryPlannerError(
                "motion planning request timed out after "
                f"{self.limits.request_timeout_seconds:g} seconds",
                diagnostic_details=metrics,
            ) from error
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

    def _temperature(self, model):
        return 0.0 if supports_temperature_for_model(self.provider, model) else None


def _raise_if_cancelled(token):
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("motion planning was cancelled")


def _request_metrics(request: ProviderRequest) -> dict[str, Any]:
    frames = tuple(
        frame
        for message in request.messages
        for frame in message.motion_frames
    )
    return {
        "model": request.model,
        "max_output_tokens": request.max_output_tokens,
        "message_characters": sum(len(message.text) for message in request.messages),
        "image_count": len(frames),
        "image_bytes": sum(len(frame.data) for frame in frames),
    }


def _apply_registered_primitive_policy(spec, instruction, context):
    """Replace free-form qpos for narrowly recognized registered primitives."""

    robot = context.get("robot", {}) if isinstance(context, Mapping) else {}
    available = (
        set(robot.get("motion_primitives", ()))
        if isinstance(robot, Mapping)
        else set()
    )
    if "burpee" not in available or not _is_burpee_request(instruction):
        return spec
    if len(spec.operations) != 1 or spec.operations[0].operation_type not in {
        TrajectoryOperationType.MOTION_PRIMITIVE,
        TrajectoryOperationType.QPOS_KEYFRAMES,
        TrajectoryOperationType.SPARSE_KEYFRAMES,
    }:
        return spec
    duration = _requested_primitive_duration(instruction, spec, context)
    return TrajectoryEditSpec(
        TrajectoryEditMode.GENERATE,
        spec.summary,
        (TrajectoryOperation(
            TrajectoryOperationType.MOTION_PRIMITIVE,
            {"primitive": "burpee", "duration_seconds": duration},
        ),),
    )


def _is_burpee_request(instruction):
    normalized = " ".join(str(instruction).lower().split())
    create = re.search(
        r"\b(create|generate|make|perform|do)\b.{0,80}\bburpee\b",
        normalized,
    )
    front_down_refinement = (
        re.search(r"\b(current|this)\s+burpee\b", normalized)
        and re.search(r"\b(front|face[- ]?down|prone)\b", normalized)
    )
    return bool(create or front_down_refinement)


def _requested_primitive_duration(instruction, spec, context):
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:-|\s)?seconds?\b",
        str(instruction).lower(),
    )
    if match is not None:
        return float(match.group(1))
    normalized = " ".join(str(instruction).lower().split())
    motion = context.get("motion", {}) if isinstance(context, Mapping) else {}
    if (
        re.search(r"\b(current|this)\s+burpee\b", normalized)
        and isinstance(motion, Mapping)
        and motion.get("duration_seconds") is not None
    ):
        return float(motion["duration_seconds"])
    operation_duration = spec.operations[0].arguments.get("duration_seconds")
    if operation_duration is not None:
        return float(operation_duration)
    if isinstance(motion, Mapping) and motion.get("duration_seconds") is not None:
        return float(motion["duration_seconds"])
    return 5.0


def _output_limit_attempt_error(request: ProviderRequest) -> str:
    return (
        "provider stopped at the bounded output limit "
        f"({request.max_output_tokens} tokens)"
    )


def _output_limit_error(
    requests: list[ProviderRequest],
    responses: list[ProviderResponse],
) -> TrajectoryPlannerError:
    final_limit = requests[-1].max_output_tokens
    return TrajectoryPlannerError(
        f"The motion response exceeded the {final_limit:,}-token bounded output "
        "limit; try a shorter motion or fewer motion phases",
        diagnostic_details={
            "stop_reason": StopReason.MAX_TOKENS.value,
            "attempts": [
                {
                    "max_output_tokens": request.max_output_tokens,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                }
                for request, response in zip(requests, responses)
            ],
        },
    )


def _combined_usage(responses: list[ProviderResponse]) -> Usage:
    return Usage(
        sum(response.usage.input_tokens for response in responses),
        sum(response.usage.output_tokens for response in responses),
    )

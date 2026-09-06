"""One-request text planning followed by deterministic local execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math

from application.ai.errors import (
    ProviderCancelledError,
    ProviderCapabilityError,
)
from application.ai.limits import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AI_INSTRUCTION_CHARACTERS,
    MAX_AI_OUTPUT_TOKENS,
    MAX_AI_RESPONSE_CHARACTERS,
)
from application.ai.motion_plan import (
    MotionEditPlan,
    editable_tool_definitions,
    motion_edit_plan_response_schema,
    parse_motion_edit_plan,
)
from application.ai.plan_executor import (
    PlanExecutionError,
    PlanExecutionResult,
    PlanExecutor,
    local_proposal,
)
from application.ai.providers.base import CancellationSignal, LLMProvider
from application.ai.schemas import (
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    Usage,
)
from application.ai.semantic_tools import SemanticToolContext
from application.ai.tool_registry import ToolRegistry


TEXT_MOTION_PLANNER_SYSTEM_PROMPT = """Plan robot motion edits using only the supplied
GhostGUI semantic edit operations. Return one complete structured plan; never call tools.
Never emit raw qpos trajectories, arbitrary code, shell commands, filesystem operations, or
GhostGUI method names. Prefer body/logical-frame and End Effector targets. Use Joint Angles
only when the user explicitly requests a Joint Angle or joint group. Preserve user-authored
and protected motion content. If the request is ambiguous, return a clarification question
and no operations. Otherwise set clarification_question to an empty string. GhostGUI will
validate and execute the plan locally. Encode each operation's arguments field as a compact
JSON object string matching that operation's supplied argument schema."""


class TextMotionPlannerError(RuntimeError):
    """A one-shot text-planning request could not complete safely."""


@dataclass(frozen=True)
class TextMotionPlannerLimits:
    request_timeout_seconds: float = 60.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_instruction_characters: int = MAX_AI_INSTRUCTION_CHARACTERS
    max_response_characters: int = MAX_AI_RESPONSE_CHARACTERS

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0.0
        ):
            raise ValueError("planner request timeout must be positive and finite")
        if not 0 < self.max_output_tokens <= MAX_AI_OUTPUT_TOKENS:
            raise ValueError("planner output-token limit is invalid")
        if not 0 < self.max_instruction_characters <= MAX_AI_INSTRUCTION_CHARACTERS:
            raise ValueError("planner instruction limit is invalid")
        if not 0 < self.max_response_characters <= MAX_AI_RESPONSE_CHARACTERS:
            raise ValueError("planner response limit is invalid")


@dataclass(frozen=True)
class TextMotionPlanningResult:
    plan: MotionEditPlan
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


@dataclass(frozen=True)
class TextMotionRunResult:
    plan: MotionEditPlan
    execution: PlanExecutionResult
    text: str
    proposal_lines: tuple[str, ...]
    provider_requests: int
    usage: Usage
    transcript: tuple[ProviderMessage, ...]

    def __post_init__(self) -> None:
        if self.provider_requests != 1:
            raise ValueError("text motion workflow must use exactly one provider request")

    @property
    def validation(self) -> dict | None:
        return self.execution.validation


class TextMotionPlanner:
    """Request one complete semantic plan from a normalized provider."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: TextMotionPlannerLimits | None = None,
        system_prompt: str = TEXT_MOTION_PLANNER_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("planner system prompt must not be empty")
        self.provider = provider
        self.tools = tools
        self.limits = limits or TextMotionPlannerLimits()
        self.system_prompt = system_prompt

    async def plan(
        self,
        instruction: str,
        *,
        model: str,
        context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> TextMotionPlanningResult:
        if not instruction.strip():
            raise ValueError("AI edit instruction must not be empty")
        if len(instruction) > self.limits.max_instruction_characters:
            raise TextMotionPlannerError(
                "AI edit instruction exceeds the local size limit"
            )
        if not self.provider.capabilities.supports_structured_output:
            raise ProviderCapabilityError(
                "selected provider/model does not support structured motion planning"
            )
        _raise_if_cancelled(cancellation_token)

        compact_context = self.tools.execute("inspect_motion", {}, context=context)
        messages = self._messages(instruction, compact_context)
        request = ProviderRequest(
            model=model,
            messages=tuple(messages),
            response_schema=motion_edit_plan_response_schema(self.tools),
            max_output_tokens=self.limits.max_output_tokens,
        )
        session = context.session
        session.begin_provider_request()
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=False)
            raise TextMotionPlannerError("AI planning request timed out") from error
        except BaseException:
            session.finish_provider_request(result_staged=False)
            raise
        session.finish_provider_request(result_staged=session.has_changes)

        _raise_if_cancelled(cancellation_token)
        if response.tool_calls:
            raise TextMotionPlannerError(
                "motion planner returned tool calls instead of a structured plan"
            )
        if len(response.text) > self.limits.max_response_characters:
            raise TextMotionPlannerError(
                "motion planner response exceeds the local size limit"
            )
        plan = parse_motion_edit_plan(response.text)
        transcript = tuple(messages) + (
            ProviderMessage(MessageRole.ASSISTANT, text=response.text),
        )
        return TextMotionPlanningResult(plan, response.usage, transcript)

    def _messages(self, instruction: str, compact_context) -> list[ProviderMessage]:
        operations = [
            {
                "name": definition.name,
                "description": definition.description,
                "arguments": dict(definition.input_schema),
            }
            for definition in editable_tool_definitions(self.tools)
        ]
        user_text = (
            "GhostGUI context:\n"
            + json.dumps(compact_context, sort_keys=True, separators=(",", ":"))
            + "\n\nAllowed semantic edit operations:\n"
            + json.dumps(operations, sort_keys=True, separators=(",", ":"))
            + f"\n\nUser instruction:\n{instruction.strip()}"
        )
        if self.provider.capabilities.supports_system_messages:
            return [
                ProviderMessage(MessageRole.SYSTEM, text=self.system_prompt),
                ProviderMessage(MessageRole.USER, text=user_text),
            ]
        return [ProviderMessage(
            MessageRole.USER,
            text=f"Instructions:\n{self.system_prompt}\n\n{user_text}",
        )]


class TextMotionWorkflow:
    """Compose one provider plan request with local ToolRegistry execution."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        planner_limits: TextMotionPlannerLimits | None = None,
    ) -> None:
        self.planner = TextMotionPlanner(
            provider,
            tools,
            limits=planner_limits,
        )
        self.executor = PlanExecutor(tools)

    async def run(
        self,
        instruction: str,
        *,
        model: str,
        context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> TextMotionRunResult:
        planning = await self.planner.plan(
            instruction,
            model=model,
            context=context,
            cancellation_token=cancellation_token,
        )
        execution = self.executor.execute(
            planning.plan,
            context=context,
            cancellation_token=cancellation_token,
        )
        if execution.validation_passed is False:
            issues = execution.validation.get("issues", [])
            detail = "; ".join(str(issue) for issue in issues) or "unknown issue"
            raise PlanExecutionError(
                f"staged motion validation failed: {detail}"
            )
        text, lines = local_proposal(execution)
        return TextMotionRunResult(
            plan=planning.plan,
            execution=execution,
            text=text,
            proposal_lines=lines,
            provider_requests=1,
            usage=planning.usage,
            transcript=planning.transcript,
        )


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("text motion planning was cancelled")

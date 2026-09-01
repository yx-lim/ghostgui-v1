"""Bounded provider/tool orchestration for staged GhostGUI edits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math

from application.ai.edit_session import AIEditSessionState
from application.ai.errors import ProviderCancelledError
from application.ai.providers import CancellationSignal, LLMProvider
from application.ai.schemas import (
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    ToolResult,
    Usage,
)
from application.ai.semantic_tools import SemanticToolContext
from application.ai.tool_registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """You edit robot motion only through the supplied GhostGUI semantic tools.
Never request raw qpos trajectories, code execution, shell access, or filesystem access.
Prefer body/logical-frame and End Effector targets over calculating Joint Angles.
Use Joint Angle tools only when the user explicitly refers to a joint or joint group.
Preserve user-authored and protected motion content. Ask for clarification when selection
context does not resolve an ambiguous instruction. Plan the complete edit before calling tools,
and send independent changes as parallel tool calls in one response. The current motion context
is already supplied, so do not call inspect_motion unless a tool failure makes reinspection
necessary. GhostGUI validates staged motion automatically after your final response; do not call
validate_motion merely to finish. Once the intended edits succeed, immediately return a concise
summary instead of making another tool call."""


class AgentError(RuntimeError):
    """Base class for bounded AI-agent failures."""


class AgentLimitError(AgentError):
    """The provider exceeded a configured iteration or tool-call bound."""


class AgentTimeoutError(AgentError):
    """A provider turn exceeded its configured timeout."""


class AgentValidationError(AgentError):
    """The staged motion did not pass deterministic validation."""


@dataclass(frozen=True)
class AgentLimits:
    # One provider turn per allowed tool call plus a final text response keeps
    # the independent safety bounds internally consistent. The previous value
    # of eight could terminate a progressing workflow before its 16-tool budget.
    max_provider_turns: int = 17
    max_tool_calls: int = 16
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_provider_turns <= 0 or self.max_tool_calls <= 0:
            raise ValueError("agent iteration limits must be positive")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0.0
        ):
            raise ValueError("agent request timeout must be positive and finite")


@dataclass(frozen=True)
class ToolExecutionRecord:
    call_id: str
    name: str
    succeeded: bool


@dataclass(frozen=True)
class AgentRunResult:
    text: str
    provider_turns: int
    tool_executions: tuple[ToolExecutionRecord, ...]
    validation: dict | None
    usage: Usage
    transcript: tuple[ProviderMessage, ...]


class GhostGUIAgent:
    """Run one user instruction against an existing detached edit session."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        limits: AgentLimits | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        auto_validate: bool = True,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("agent system prompt must not be empty")
        self.provider = provider
        self.tools = tools
        self.limits = limits or AgentLimits()
        self.system_prompt = system_prompt
        self.auto_validate = bool(auto_validate)

    async def run(
        self,
        instruction: str,
        *,
        model: str,
        context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> AgentRunResult:
        if not instruction.strip():
            raise ValueError("AI edit instruction must not be empty")
        if context.session.state in {
            AIEditSessionState.ACCEPTED,
            AIEditSessionState.REJECTED,
        }:
            raise AgentError("cannot run against a finished AI edit session")
        self._raise_if_cancelled(cancellation_token)
        compact_context = self.tools.execute(
            "inspect_motion",
            {},
            context=context,
        )
        messages = self._initial_messages(instruction, compact_context)
        executions = []
        failed_call_attempts: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for provider_turn in range(1, self.limits.max_provider_turns + 1):
            self._raise_if_cancelled(cancellation_token)
            response = await self._provider_turn(
                ProviderRequest(
                    model=model,
                    messages=tuple(messages),
                    tools=self.tools.definitions(),
                ),
                context,
                cancellation_token,
            )
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            if not response.text and not response.tool_calls:
                raise AgentError("provider returned an empty response")
            messages.append(ProviderMessage(
                MessageRole.ASSISTANT,
                text=response.text,
                tool_calls=response.tool_calls,
            ))

            if not response.tool_calls:
                validation = self._validate_if_needed(context)
                return AgentRunResult(
                    text=response.text,
                    provider_turns=provider_turn,
                    tool_executions=tuple(executions),
                    validation=validation,
                    usage=Usage(total_input_tokens, total_output_tokens),
                    transcript=tuple(messages),
                )

            if len(executions) + len(response.tool_calls) > self.limits.max_tool_calls:
                raise AgentLimitError("AI tool-call limit exceeded")
            call_ids = [call.identifier for call in response.tool_calls]
            if len(set(call_ids)) != len(call_ids):
                raise AgentError("provider returned duplicate tool call identifiers")

            tool_results = []
            for call in response.tool_calls:
                self._raise_if_cancelled(cancellation_token)
                try:
                    output = self.tools.execute(
                        call.name,
                        call.arguments,
                        context=context,
                    )
                    output = _json_value(output)
                    succeeded = True
                except Exception as error:
                    output = {"error": str(error)}
                    succeeded = False
                    signature = _call_signature(call)
                    failed_call_attempts[signature] = (
                        failed_call_attempts.get(signature, 0) + 1
                    )
                    if failed_call_attempts[signature] >= 2:
                        raise AgentLimitError(
                            f"AI repeated the failing {call.name} tool call "
                            "without progress"
                        ) from error
                executions.append(ToolExecutionRecord(
                    call.identifier,
                    call.name,
                    succeeded,
                ))
                tool_results.append(ToolResult(
                    call_id=call.identifier,
                    name=call.name,
                    output=output,
                    is_error=not succeeded,
                ))
                self._raise_if_cancelled(cancellation_token)
            messages.append(ProviderMessage(
                MessageRole.TOOL,
                tool_results=tuple(tool_results),
            ))

        raise AgentLimitError(
            "AI could not finish within the provider-turn budget of "
            f"{self.limits.max_provider_turns}"
        )

    async def _provider_turn(self, request, context, cancellation_token):
        session = context.session
        session.begin_provider_request()
        try:
            response = await asyncio.wait_for(
                self.provider.generate(request, cancellation_token),
                timeout=self.limits.request_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            session.finish_provider_request(result_staged=False)
            raise AgentTimeoutError("AI provider request timed out") from error
        except BaseException:
            session.finish_provider_request(result_staged=False)
            raise
        session.finish_provider_request(
            result_staged=session.has_changes
        )
        return response

    def _initial_messages(self, instruction, compact_context):
        context_text = json.dumps(
            compact_context,
            sort_keys=True,
            separators=(",", ":"),
        )
        user_text = f"GhostGUI context:\n{context_text}\n\nUser instruction:\n{instruction.strip()}"
        if self.provider.capabilities.supports_system_messages:
            return [
                ProviderMessage(MessageRole.SYSTEM, text=self.system_prompt),
                ProviderMessage(MessageRole.USER, text=user_text),
            ]
        return [ProviderMessage(
            MessageRole.USER,
            text=f"Instructions:\n{self.system_prompt}\n\n{user_text}",
        )]

    def _validate_if_needed(self, context):
        if not self.auto_validate or not context.session.has_changes:
            return None
        result = _json_value(self.tools.execute(
            "validate_motion",
            {},
            context=context,
        ))
        if not isinstance(result, dict) or result.get("valid") is not True:
            issues = result.get("issues", []) if isinstance(result, dict) else []
            detail = "; ".join(str(issue) for issue in issues) or "unknown issue"
            raise AgentValidationError(f"staged motion validation failed: {detail}")
        return result

    @staticmethod
    def _raise_if_cancelled(token):
        if token is not None and token.cancellation_requested:
            raise ProviderCancelledError("AI agent run was cancelled")


def _json_value(value):
    try:
        encoded = json.dumps(value, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise AgentError("tool returned a non-JSON result") from error


def _call_signature(call) -> str:
    try:
        arguments = json.dumps(
            dict(call.arguments),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        arguments = repr(dict(call.arguments))
    return f"{call.name}:{arguments}"

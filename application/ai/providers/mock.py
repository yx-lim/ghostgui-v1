"""Deterministic provider used before real network adapters are introduced."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

from application.ai.errors import (
    AIError,
    ProviderCancelledError,
    ProviderError,
)
from application.ai.providers.base import (
    CancellationSignal,
    validate_provider_request,
    validate_provider_response,
)
from application.ai.schemas import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
)


@dataclass(frozen=True)
class MockStep:
    """One scripted response or failure, optionally delivered after a delay."""

    response: ProviderResponse | None = None
    error: Exception | None = None
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (self.response is None) == (self.error is None):
            raise ValueError("mock step requires exactly one response or error")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0.0:
            raise ValueError("mock step delay must be finite and non-negative")


class MockProvider:
    """Replay a finite script while exercising real provider boundaries."""

    def __init__(
        self,
        steps: Iterable[MockStep | ProviderResponse],
        *,
        capabilities: ProviderCapabilities | None = None,
        provider_name: str = "mock",
    ) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name must not be empty")
        self._provider_name = provider_name
        self._capabilities = capabilities or ProviderCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_parallel_tool_calls=True,
            max_images_per_request=16,
        )
        self._steps = deque(
            step if isinstance(step, MockStep) else MockStep(response=step)
            for step in steps
        )
        self._requests: list[ProviderRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def requests(self) -> tuple[ProviderRequest, ...]:
        return tuple(self._requests)

    @property
    def remaining_steps(self) -> int:
        return len(self._steps)

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        validate_provider_request(request, self.capabilities)
        _raise_if_cancelled(cancellation_token)
        if not self._steps:
            raise ProviderError("MockProvider script is exhausted")

        self._requests.append(request)
        step = self._steps.popleft()
        await _wait_with_cancellation(step.delay_seconds, cancellation_token)
        _raise_if_cancelled(cancellation_token)

        if step.error is not None:
            if isinstance(step.error, AIError):
                raise step.error
            raise ProviderError(f"MockProvider scripted failure: {step.error}") from step.error

        response = step.response
        if response is None:  # Defensive: MockStep validates this invariant.
            raise ProviderError("MockProvider step has no response")
        validate_provider_response(response, self.capabilities)
        return response

    def assert_exhausted(self) -> None:
        if self._steps:
            raise AssertionError(f"MockProvider has {len(self._steps)} unconsumed step(s)")


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("provider request was cancelled")


async def _wait_with_cancellation(
    delay_seconds: float,
    token: CancellationSignal | None,
) -> None:
    remaining = delay_seconds
    while remaining > 0.0:
        interval = min(remaining, 0.01)
        await asyncio.sleep(interval)
        remaining -= interval
        _raise_if_cancelled(token)


# --- Agent-loop scripted provider -------------------------------------------------

class ScriptedAgentProvider:
    """Network-free agent provider: returns scripted raw turns in order.

    Each script entry is an ``AgentTurnResponse``, a callable taking the request and
    returning one, or an exception instance to raise. Requests are recorded.
    """

    provider_name = "mock"
    supports_agent_loop = True
    supports_mid_conversation_system = True

    def __init__(self, script=(), *, events=None):
        self._script = list(script)
        self._events = dict(events or {})
        self.requests = []

    def push(self, *entries):
        self._script.extend(entries)

    async def stream_turn(self, request, on_event=None, cancellation_token=None):
        from application.ai.agent_wire import AgentStreamEvent
        from application.ai.errors import ProviderCancelledError, ProviderError

        if cancellation_token is not None and cancellation_token.cancellation_requested:
            raise ProviderCancelledError("scripted request was cancelled")
        self.requests.append(request)
        index = len(self.requests) - 1
        if not self._script:
            raise ProviderError("scripted agent provider has no remaining turns")
        entry = self._script.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        response = entry(request) if callable(entry) else entry
        if on_event is not None:
            on_event(AgentStreamEvent("turn_start"))
            for event in self._events.get(index, ()):
                on_event(event)
            for block in response.content:
                if block.get("type") == "text":
                    on_event(AgentStreamEvent("text", text=block.get("text", "")))
                elif block.get("type") == "thinking":
                    on_event(AgentStreamEvent("thinking", text=block.get("thinking", "")))
                elif block.get("type") == "tool_use":
                    on_event(AgentStreamEvent("tool_use_start", name=block.get("name", "")))
        if cancellation_token is not None and cancellation_token.cancellation_requested:
            raise ProviderCancelledError("scripted request was cancelled")
        return response

    async def aclose(self):
        return None


def scripted_turn(*blocks, stop_reason=None, usage=None, model="claude-opus-5-5", stop_details=None):
    """Build an ``AgentTurnResponse`` from raw block dicts (tool_use implies tool_use stop)."""
    from application.ai.agent_wire import AgentStopReason, AgentTurnResponse, AgentUsage

    content = tuple(dict(block) for block in blocks)
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
    return AgentTurnResponse(content=content, stop_reason=AgentStopReason.parse(stop_reason),
                             usage=usage or AgentUsage(input_tokens=100, output_tokens=20, served_model=model),
                             model=model, request_id="req_mock", stop_details=stop_details)

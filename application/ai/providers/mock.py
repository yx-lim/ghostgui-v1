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

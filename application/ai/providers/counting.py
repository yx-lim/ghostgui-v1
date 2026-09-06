"""Provider-neutral request counting for AI workflow measurements."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from application.ai.providers.base import CancellationSignal, LLMProvider
from application.ai.schemas import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
)


@dataclass(frozen=True)
class ProviderRequestCounts:
    """Aggregate normalized-provider calls made by one measured workflow."""

    total: int
    succeeded: int
    failed: int
    in_flight: int

    def __post_init__(self) -> None:
        if min(self.total, self.succeeded, self.failed, self.in_flight) < 0:
            raise ValueError("provider request counts must not be negative")
        if self.total != self.succeeded + self.failed + self.in_flight:
            raise ValueError("provider request counts are inconsistent")


class ProviderRequestCounter:
    """Thread-safe aggregate counter shared by one or more provider wrappers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total = 0
        self._succeeded = 0
        self._failed = 0
        self._in_flight = 0

    @property
    def counts(self) -> ProviderRequestCounts:
        with self._lock:
            return ProviderRequestCounts(
                total=self._total,
                succeeded=self._succeeded,
                failed=self._failed,
                in_flight=self._in_flight,
            )

    def reset(self) -> None:
        """Clear a completed measurement without racing an active request."""

        with self._lock:
            if self._in_flight:
                raise RuntimeError("cannot reset provider counts during a request")
            self._total = 0
            self._succeeded = 0
            self._failed = 0

    def request_started(self) -> None:
        with self._lock:
            self._total += 1
            self._in_flight += 1

    def request_finished(self, *, succeeded: bool) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("provider request finished without a start")
            self._in_flight -= 1
            if succeeded:
                self._succeeded += 1
            else:
                self._failed += 1


class RequestCountingProvider:
    """Transparent provider decorator that counts normalized model requests.

    Counts intentionally describe GhostGUI calls to ``LLMProvider.generate``.
    Provider SDK retries below this boundary are transport behavior and are not
    counted as additional workflow requests.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        counter: ProviderRequestCounter | None = None,
    ) -> None:
        self._provider = provider
        self.counter = counter or ProviderRequestCounter()

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._provider.capabilities

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        self.counter.request_started()
        try:
            response = await self._provider.generate(request, cancellation_token)
        except BaseException:
            self.counter.request_finished(succeeded=False)
            raise
        self.counter.request_finished(succeeded=True)
        return response

    async def aclose(self) -> None:
        """Close an owned adapter when it exposes GhostGUI's optional hook."""

        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()

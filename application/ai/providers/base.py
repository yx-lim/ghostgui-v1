"""Provider protocol and shared capability checks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.ai.errors import ProviderCapabilityError
from application.ai.schemas import (
    MessageRole,
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
)


class CancellationSignal(Protocol):
    """Minimal contract implemented by the existing background-job token."""

    @property
    def cancellation_requested(self) -> bool:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Provider/model adapter consumed by the future bounded AI agent."""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        ...

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        ...


def validate_provider_request(
    request: ProviderRequest,
    capabilities: ProviderCapabilities,
) -> None:
    """Reject unsupported features before a provider performs any work."""

    if request.tools and not capabilities.supports_tools:
        raise ProviderCapabilityError("selected provider/model does not support tools")
    if request.response_schema is not None and not capabilities.supports_structured_output:
        raise ProviderCapabilityError(
            "selected provider/model does not support structured output"
        )
    if (
        any(message.role is MessageRole.SYSTEM for message in request.messages)
        and not capabilities.supports_system_messages
    ):
        raise ProviderCapabilityError(
            "selected provider/model does not support system messages"
        )
    image_count = sum(len(message.motion_frames) for message in request.messages)
    if image_count and not capabilities.supports_vision:
        raise ProviderCapabilityError("selected provider/model does not support vision")
    if image_count > capabilities.max_images_per_request:
        raise ProviderCapabilityError(
            "request exceeds the selected provider/model image limit"
        )


def validate_provider_response(
    response: ProviderResponse,
    capabilities: ProviderCapabilities,
) -> None:
    """Ensure a provider does not return calls outside its declared contract."""

    if response.tool_calls and not capabilities.supports_tools:
        raise ProviderCapabilityError(
            "provider returned tool calls despite declaring no tool support"
        )
    if len(response.tool_calls) > 1 and not capabilities.supports_parallel_tool_calls:
        raise ProviderCapabilityError(
            "provider returned parallel tool calls without declaring support"
        )

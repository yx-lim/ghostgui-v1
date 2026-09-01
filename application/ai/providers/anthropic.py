"""Anthropic Claude adapter for GhostGUI's provider-neutral AI contract."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any

from application.ai.credentials import CredentialSource, default_credential_source
from application.ai.errors import (
    ProviderAuthenticationError,
    ProviderCancelledError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from application.ai.providers.base import (
    CancellationSignal,
    validate_provider_request,
    validate_provider_response,
)
from application.ai.schemas import (
    MessageRole,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    ToolCall,
    Usage,
)


DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    supports_tools=True,
    supports_vision=True,
    supports_structured_output=True,
    supports_parallel_tool_calls=True,
    supports_system_messages=True,
    max_images_per_request=16,
)
_DEFAULT_MAX_OUTPUT_TOKENS = 4096


class AnthropicProvider:
    """Translate common requests to and from Anthropic's Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        credential_source: CredentialSource | None = None,
        capabilities: ProviderCapabilities = DEFAULT_ANTHROPIC_CAPABILITIES,
        cancellation_poll_seconds: float = 0.05,
    ) -> None:
        if cancellation_poll_seconds <= 0.0:
            raise ValueError("cancellation_poll_seconds must be positive")
        self._capabilities = capabilities
        self._cancellation_poll_seconds = float(cancellation_poll_seconds)
        self._owns_client = client is None
        if client is None:
            source = credential_source or default_credential_source()
            key = api_key or source.get_secret("anthropic")
            if not key:
                raise ProviderConfigurationError(
                    "Anthropic is not configured; add an Anthropic API key to the "
                    "system credential store or set ANTHROPIC_API_KEY"
                )
            try:
                from anthropic import AsyncAnthropic
            except ImportError as error:
                raise ProviderConfigurationError(
                    "Claude support is not installed; install GhostGUI with the ai extra"
                ) from error
            client = AsyncAnthropic(api_key=key)
        self._client = client

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        validate_provider_request(request, self._capabilities)
        if _cancelled(cancellation_token):
            raise ProviderCancelledError("Anthropic request was cancelled")
        arguments = _build_anthropic_request(request)
        sdk_request = asyncio.create_task(self._client.messages.create(**arguments))
        try:
            while not sdk_request.done():
                if _cancelled(cancellation_token):
                    sdk_request.cancel()
                    await _consume_cancellation(sdk_request)
                    raise ProviderCancelledError("Anthropic request was cancelled")
                await asyncio.sleep(self._cancellation_poll_seconds)
            raw_response = await sdk_request
        except ProviderCancelledError:
            raise
        except asyncio.CancelledError:
            sdk_request.cancel()
            raise
        except Exception as error:
            raise _normalize_error(error) from error
        response = _parse_anthropic_response(raw_response)
        validate_provider_response(response, self._capabilities)
        return response

    async def aclose(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _build_anthropic_request(request: ProviderRequest) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role is MessageRole.SYSTEM:
            if message.text:
                system_parts.append(message.text)
            continue
        content = _message_content(message)
        if content:
            messages.append({
                "role": _anthropic_role(message.role),
                "content": content,
            })

    arguments: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS,
        "messages": messages,
    }
    if system_parts:
        arguments["system"] = "\n\n".join(system_parts)
    if request.tools:
        arguments["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(dict(tool.input_schema)),
            }
            for tool in request.tools
        ]
    if request.response_schema is not None:
        arguments["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _structured_output_schema(request.response_schema),
            }
        }
    return arguments


def _message_content(message: ProviderMessage) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    for frame in message.motion_frames:
        content.append({"type": "text", "text": _frame_time_metadata(frame)})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": frame.mime_type,
                "data": base64.b64encode(frame.data).decode("ascii"),
            },
        })
    for call in message.tool_calls:
        content.append({
            "type": "tool_use",
            "id": call.identifier,
            "name": call.name,
            "input": dict(call.arguments),
        })
    for result in message.tool_results:
        content.append({
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": json.dumps(
                result.output,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "is_error": result.is_error,
        })
    return content


def _anthropic_role(role: MessageRole) -> str:
    return "assistant" if role is MessageRole.ASSISTANT else "user"


def _frame_time_metadata(frame: MotionFrameImage) -> str:
    label = frame.label.strip() or frame.comparison_id
    return f"{label} ({frame.variant.value}) = t={frame.time_seconds:.6f} s"


def _structured_output_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove constraints not accepted by Claude's constrained decoder.

    GhostGUI's parser still enforces the complete original contract.
    """

    unsupported = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    }

    def transform(value):
        if isinstance(value, Mapping):
            return {
                key: transform(item)
                for key, item in value.items()
                if key not in unsupported
            }
        if isinstance(value, (list, tuple)):
            return [transform(item) for item in value]
        return deepcopy(value)

    return transform(schema)


def _parse_anthropic_response(raw_response: Any) -> ProviderResponse:
    content = _field(raw_response, "content", ()) or ()
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in content:
        block_type = str(_field(block, "type", "") or "")
        if block_type == "text":
            text = _field(block, "text", "")
            if text:
                texts.append(str(text))
        elif block_type == "tool_use":
            identifier = str(_field(block, "id", "") or "")
            name = str(_field(block, "name", "") or "")
            arguments = _field(block, "input", None)
            if not identifier or not name or not isinstance(arguments, Mapping):
                raise ProviderResponseError("Anthropic returned a malformed tool call")
            calls.append(ToolCall(identifier, name, dict(arguments)))

    if not texts and not calls:
        raise ProviderResponseError("Anthropic returned an empty response")
    raw_stop_reason = str(_field(raw_response, "stop_reason", "") or "")
    if calls:
        stop_reason = StopReason.TOOL_CALLS
    elif raw_stop_reason == "max_tokens":
        stop_reason = StopReason.MAX_TOKENS
    else:
        stop_reason = StopReason.COMPLETE
    usage = _field(raw_response, "usage", None)
    return ProviderResponse(
        text="".join(texts),
        tool_calls=tuple(calls),
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=_nonnegative_int(_field(usage, "input_tokens", 0)),
            output_tokens=_nonnegative_int(_field(usage, "output_tokens", 0)),
        ),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_error(error: Exception) -> ProviderError:
    code = _error_status_code(error)
    if code in {401, 403}:
        return ProviderAuthenticationError("Anthropic rejected the configured API key")
    if code == 429:
        return ProviderRateLimitError(
            "Anthropic rate limit reached; wait before trying again"
        )
    if code == 404:
        return ProviderConfigurationError(
            "Claude model was not found or is unavailable for this API key (HTTP 404)"
        )
    if code == 400:
        return ProviderError("Anthropic rejected the request (HTTP 400)")
    if code in {408, 409, 500, 502, 503, 504}:
        return ProviderError(
            f"Anthropic service is temporarily unavailable (HTTP {code}); "
            "try again in a moment"
        )
    return ProviderError(f"Anthropic request failed ({type(error).__name__})")


def _error_status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _cancelled(token: CancellationSignal | None) -> bool:
    return token is not None and token.cancellation_requested


async def _consume_cancellation(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass

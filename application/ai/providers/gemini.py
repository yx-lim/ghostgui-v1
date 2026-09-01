"""Gemini adapter for GhostGUI's provider-neutral AI contract."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
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


DEFAULT_GEMINI_CAPABILITIES = ProviderCapabilities(
    supports_tools=True,
    supports_vision=True,
    supports_structured_output=True,
    supports_parallel_tool_calls=True,
    supports_system_messages=True,
    max_images_per_request=16,
)


class GeminiProvider:
    """Translate common requests to the Google Gen AI SDK and back again."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        credential_source: CredentialSource | None = None,
        capabilities: ProviderCapabilities = DEFAULT_GEMINI_CAPABILITIES,
        cancellation_poll_seconds: float = 0.05,
    ) -> None:
        if cancellation_poll_seconds <= 0.0:
            raise ValueError("cancellation_poll_seconds must be positive")
        self._capabilities = capabilities
        self._cancellation_poll_seconds = cancellation_poll_seconds
        self._owns_client = client is None
        if client is None:
            source = credential_source or default_credential_source()
            key = api_key or source.get_secret("gemini")
            if not key:
                raise ProviderConfigurationError(
                    "Gemini is not configured; add a Gemini API key to the system "
                    "credential store or set GOOGLE_API_KEY/GEMINI_API_KEY"
                )
            try:
                from google import genai
            except ImportError as error:
                raise ProviderConfigurationError(
                    "Gemini support is not installed; install GhostGUI with the ai extra"
                ) from error
            # Do not retain the plaintext key after the SDK client is constructed.
            client = genai.Client(api_key=key)
        self._client = client

    @property
    def provider_name(self) -> str:
        return "gemini"

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
            raise ProviderCancelledError("Gemini request was cancelled")

        contents, config = _build_gemini_request(request)
        sdk_request = asyncio.create_task(
            self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )
        )
        try:
            while not sdk_request.done():
                if _cancelled(cancellation_token):
                    sdk_request.cancel()
                    await _consume_cancellation(sdk_request)
                    raise ProviderCancelledError("Gemini request was cancelled")
                await asyncio.sleep(self._cancellation_poll_seconds)
            raw_response = await sdk_request
        except ProviderCancelledError:
            raise
        except asyncio.CancelledError:
            sdk_request.cancel()
            raise
        except Exception as error:
            raise _normalize_error(error) from error

        response = _parse_gemini_response(raw_response)
        validate_provider_response(response, self._capabilities)
        return response

    async def aclose(self) -> None:
        """Close a client created by this adapter; injected clients remain caller-owned."""

        if not self._owns_client:
            return
        close = getattr(getattr(self._client, "aio", None), "aclose", None)
        if close is not None:
            await close()


def _build_gemini_request(request: ProviderRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role is MessageRole.SYSTEM:
            if message.text:
                system_parts.append(message.text)
            continue
        parts = _message_parts(message)
        if parts:
            contents.append({"role": _gemini_role(message.role), "parts": parts})

    config: dict[str, Any] = {}
    if system_parts:
        config["system_instruction"] = "\n\n".join(system_parts)
    if request.max_output_tokens is not None:
        config["max_output_tokens"] = request.max_output_tokens
    if request.tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters_json_schema": deepcopy(dict(tool.input_schema)),
                    }
                    for tool in request.tools
                ]
            }
        ]
        # GhostGUI owns the bounded tool loop; the SDK must not execute functions.
        config["automatic_function_calling"] = {"disable": True}
    if request.response_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = deepcopy(dict(request.response_schema))
    return contents, config


def _message_parts(message: ProviderMessage) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if message.text:
        parts.append({"text": message.text})
    for frame in message.motion_frames:
        parts.append({"text": _frame_time_metadata(frame)})
        parts.append(
            {"inline_data": {"data": frame.data, "mime_type": frame.mime_type}}
        )
    for call in message.tool_calls:
        parts.append(
            {
                "function_call": {
                    "id": call.identifier,
                    "name": call.name,
                    "args": dict(call.arguments),
                }
            }
        )
    for result in message.tool_results:
        output = result.output if isinstance(result.output, Mapping) else {"result": result.output}
        response = dict(output)
        if result.is_error:
            response.setdefault("is_error", True)
        parts.append(
            {
                "function_response": {
                    "id": result.call_id,
                    "name": result.name,
                    "response": response,
                }
            }
        )
    return parts


def _frame_time_metadata(frame: MotionFrameImage) -> str:
    label = frame.label.strip() or frame.comparison_id
    return f"{label} ({frame.variant.value}) = t={frame.time_seconds:.6f} s"


def _gemini_role(role: MessageRole) -> str:
    if role is MessageRole.ASSISTANT:
        return "model"
    return "user"


def _parse_gemini_response(raw_response: Any) -> ProviderResponse:
    candidates = getattr(raw_response, "candidates", None) or ()
    if not candidates:
        raise ProviderResponseError("Gemini returned no response candidate")
    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or ()
    texts: list[str] = []
    calls: list[ToolCall] = []
    response_id = str(getattr(raw_response, "response_id", "") or "gemini")
    for index, part in enumerate(parts):
        part_text = getattr(part, "text", None)
        if part_text:
            texts.append(str(part_text))
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        name = str(getattr(function_call, "name", "") or "")
        arguments = getattr(function_call, "args", None)
        if not name or not isinstance(arguments, Mapping):
            raise ProviderResponseError("Gemini returned a malformed tool call")
        identifier = str(
            getattr(function_call, "id", "") or f"{response_id}-call-{index + 1}"
        )
        calls.append(ToolCall(identifier, name, dict(arguments)))

    if not texts and not calls:
        raise ProviderResponseError("Gemini returned an empty response")
    finish_reason = str(getattr(candidate, "finish_reason", "") or "").upper()
    if calls:
        stop_reason = StopReason.TOOL_CALLS
    elif "MAX_TOKENS" in finish_reason:
        stop_reason = StopReason.MAX_TOKENS
    else:
        stop_reason = StopReason.COMPLETE

    usage = getattr(raw_response, "usage_metadata", None)
    response = ProviderResponse(
        text="".join(texts),
        tool_calls=tuple(calls),
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=_nonnegative_int(getattr(usage, "prompt_token_count", 0)),
            output_tokens=_nonnegative_int(
                getattr(usage, "candidates_token_count", 0)
            ),
        ),
    )
    return response


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_error(error: Exception) -> ProviderError:
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if code in {401, 403}:
        return ProviderAuthenticationError("Gemini rejected the configured API key")
    if code == 429:
        return ProviderRateLimitError(
            "Gemini rate limit reached; wait before trying again"
        )
    # Avoid echoing provider exception messages because SDK errors may contain
    # request metadata. The exception type is sufficient for diagnostics.
    return ProviderError(f"Gemini request failed ({type(error).__name__})")


def _cancelled(token: CancellationSignal | None) -> bool:
    return token is not None and token.cancellation_requested


async def _consume_cancellation(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass

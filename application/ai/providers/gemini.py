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

_RETRYABLE_STATUS_CODES = frozenset({408, 500, 502, 503, 504})
_DEFAULT_MAX_ATTEMPTS = 1
_DEFAULT_RETRY_BASE_SECONDS = 0.5


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
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        retry_base_seconds: float = _DEFAULT_RETRY_BASE_SECONDS,
    ) -> None:
        if cancellation_poll_seconds <= 0.0:
            raise ValueError("cancellation_poll_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_base_seconds < 0.0:
            raise ValueError("retry_base_seconds must not be negative")
        self._capabilities = capabilities
        self._cancellation_poll_seconds = cancellation_poll_seconds
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._thought_signatures: dict[str, bytes] = {}
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

        contents, config = _build_gemini_request(
            request,
            thought_signatures=self._thought_signatures,
        )
        for attempt in range(1, self._max_attempts + 1):
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
                break
            except ProviderCancelledError:
                raise
            except asyncio.CancelledError:
                sdk_request.cancel()
                raise
            except Exception as error:
                if attempt >= self._max_attempts or not _retryable(error):
                    raise _normalize_error(error, attempts=attempt) from error
                await _wait_for_retry(
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                    cancellation_token,
                    poll_seconds=self._cancellation_poll_seconds,
                )

        response = _parse_gemini_response(
            raw_response,
            thought_signatures=self._thought_signatures,
        )
        validate_provider_response(response, self._capabilities)
        return response

    async def aclose(self) -> None:
        """Close a client created by this adapter; injected clients remain caller-owned."""

        if not self._owns_client:
            return
        close = getattr(getattr(self._client, "aio", None), "aclose", None)
        if close is not None:
            await close()


def _build_gemini_request(
    request: ProviderRequest,
    *,
    thought_signatures: Mapping[str, bytes] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role is MessageRole.SYSTEM:
            if message.text:
                system_parts.append(message.text)
            continue
        parts = _message_parts(message, thought_signatures=thought_signatures)
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


def _message_parts(
    message: ProviderMessage,
    *,
    thought_signatures: Mapping[str, bytes] | None = None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if message.text:
        parts.append({"text": message.text})
    for frame in message.motion_frames:
        parts.append({"text": _frame_time_metadata(frame)})
        parts.append(
            {"inline_data": {"data": frame.data, "mime_type": frame.mime_type}}
        )
    for call in message.tool_calls:
        part = {
            "function_call": {
                "id": call.identifier,
                "name": call.name,
                "args": dict(call.arguments),
            }
        }
        signature = (
            None
            if thought_signatures is None
            else thought_signatures.get(call.identifier)
        )
        if signature:
            # Gemini 3 strictly requires this opaque value to be returned on
            # the exact function-call part where the model supplied it.
            part["thought_signature"] = signature
        parts.append(part)
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


def _parse_gemini_response(
    raw_response: Any,
    *,
    thought_signatures: dict[str, bytes] | None = None,
) -> ProviderResponse:
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
        signature = getattr(part, "thought_signature", None)
        if thought_signatures is not None and signature:
            thought_signatures[identifier] = bytes(signature)
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


def _error_status_code(error: Exception) -> int | None:
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _retryable(error: Exception) -> bool:
    code = _error_status_code(error)
    return code in _RETRYABLE_STATUS_CODES or (
        code is None and type(error).__name__ == "ServerError"
    )


def _normalize_error(error: Exception, *, attempts: int = 1) -> ProviderError:
    code = _error_status_code(error)
    if code in {401, 403}:
        return ProviderAuthenticationError("Gemini rejected the configured API key")
    if code == 429:
        return ProviderRateLimitError(
            "Gemini rate limit reached; wait before trying again"
        )
    if code == 400:
        return ProviderError(
            "Gemini rejected the request (HTTP 400 INVALID_ARGUMENT)"
        )
    if code == 404:
        return ProviderConfigurationError(
            "Gemini model was not found or is unavailable for this API key (HTTP 404)"
        )
    if code in _RETRYABLE_STATUS_CODES or type(error).__name__ == "ServerError":
        status = f" (HTTP {code})" if code is not None else ""
        attempt_text = (
            f" after {attempts} attempts" if attempts > 1 else ""
        )
        return ProviderError(
            f"Gemini service is temporarily unavailable{status}{attempt_text}; "
            "try again in a moment"
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


async def _wait_for_retry(
    delay_seconds: float,
    token: CancellationSignal | None,
    *,
    poll_seconds: float,
) -> None:
    remaining = delay_seconds
    while remaining > 0.0:
        if _cancelled(token):
            raise ProviderCancelledError("Gemini request was cancelled")
        interval = min(remaining, poll_seconds)
        await asyncio.sleep(interval)
        remaining -= interval
    if _cancelled(token):
        raise ProviderCancelledError("Gemini request was cancelled")

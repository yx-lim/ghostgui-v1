"""Streaming, lossless Anthropic adapter for the in-app agent loop.

One ``stream_turn`` call issues exactly one streamed Messages request:

- tools (sorted by the caller) and a frozen system prompt carry the first 1-hour
  cache breakpoint; top-level ``cache_control`` adds an automatic tail breakpoint;
- adaptive thinking with summarized display, explicit effort;
- server-side refusal fallbacks on models that support them;
- every returned content block is kept verbatim for replay.
"""

from __future__ import annotations

import asyncio
from typing import Any

from application.ai.agent_wire import (
    AgentStopReason, AgentStreamEvent, AgentTurnRequest, AgentTurnResponse, AgentUsage,
    StreamCallback, UsageIteration,
)
from application.ai.errors import (
    ProviderAuthenticationError, ProviderBadRequestError, ProviderCancelledError,
    ProviderConfigurationError, ProviderContextLimitError, ProviderError,
    ProviderMalformedStreamError, ProviderModelUnavailableError, ProviderOfflineError,
    ProviderOverloadedError, ProviderPermissionError, ProviderRateLimitError,
)
from application.ai.secrets import SecretStr, install_log_scrubbing

FALLBACK_BETA = "server-side-fallback-2026-07-01"
AGENT_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
DEFAULT_AGENT_MODEL = "claude-opus-5-5"
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})


def supports_fallbacks(model: str) -> bool:
    return model in FALLBACK_MODELS


def build_stream_arguments(request: AgentTurnRequest) -> dict[str, Any]:
    """Pure request shaping; tested byte-for-byte for cache-prefix stability."""
    ttl = {"type": "ephemeral", "ttl": request.cache_ttl}
    tools = []
    for tool in request.tools:
        entry = {"name": tool["name"], "description": tool["description"],
                 "input_schema": tool["input_schema"]}
        if request.eager_input_streaming:
            entry["eager_input_streaming"] = True
        tools.append(entry)
    arguments: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": [{"type": "text", "text": request.system, "cache_control": dict(ttl)}],
        "messages": [dict(message) for message in request.messages],
        "cache_control": dict(ttl),
        "thinking": {"type": "adaptive", "display": request.thinking_display},
        "output_config": {"effort": request.effort.value},
    }
    if tools:
        arguments["tools"] = tools
    if request.fallbacks and supports_fallbacks(request.model):
        arguments["betas"] = [FALLBACK_BETA]
        arguments["fallbacks"] = "default"
    return arguments


def _to_dict(value: Any) -> Any:
    for method in ("to_dict", "model_dump"):
        function = getattr(value, method, None)
        if callable(function):
            try:
                return function(mode="json", exclude_none=True) if method == "model_dump" else function()
            except TypeError:
                return function()
    if isinstance(value, dict):
        return value
    return dict(vars(value))


def _int(mapping, key) -> int:
    try:
        return int(mapping.get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def parse_usage(raw: Any, served_model: str) -> AgentUsage:
    usage = _to_dict(raw) if raw is not None else {}
    creation = usage.get("cache_creation") or {}
    write_5m, write_1h = _int(creation, "ephemeral_5m_input_tokens"), _int(creation, "ephemeral_1h_input_tokens")
    if not creation:
        write_5m = _int(usage, "cache_creation_input_tokens")
    iterations = []
    for item in usage.get("iterations") or ():
        item_creation = item.get("cache_creation") or {}
        iterations.append(UsageIteration(
            type=str(item.get("type", "message")), model=str(item.get("model") or served_model),
            input_tokens=_int(item, "input_tokens"), output_tokens=_int(item, "output_tokens"),
            cache_read=_int(item, "cache_read_input_tokens"),
            cache_write_5m=_int(item_creation, "ephemeral_5m_input_tokens") if item_creation
            else _int(item, "cache_creation_input_tokens"),
            cache_write_1h=_int(item_creation, "ephemeral_1h_input_tokens"),
        ))
    return AgentUsage(
        input_tokens=_int(usage, "input_tokens"), output_tokens=_int(usage, "output_tokens"),
        cache_read=_int(usage, "cache_read_input_tokens"), cache_write_5m=write_5m,
        cache_write_1h=write_1h, served_model=served_model, iterations=tuple(iterations),
    )


def parse_message(message: Any) -> AgentTurnResponse:
    data = _to_dict(message)
    content = tuple(dict(block) for block in data.get("content") or ())
    model = str(data.get("model") or "")
    request_id = str(getattr(message, "_request_id", "") or data.get("_request_id") or "")
    return AgentTurnResponse(
        content=content, stop_reason=AgentStopReason.parse(data.get("stop_reason")),
        usage=parse_usage(data.get("usage"), model), model=model, request_id=request_id,
        stop_details=data.get("stop_details") or None,
    )


def map_error(error: Exception) -> ProviderError:
    """Most-specific first; never echo error bodies, which may contain input."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the ai extra is optional
        anthropic = None
    status = getattr(error, "status_code", None)
    request_id = str(getattr(error, "request_id", "") or "")
    name = type(error).__name__

    def is_(class_name, code=None):
        cls = getattr(anthropic, class_name, None) if anthropic else None
        return (cls is not None and isinstance(error, cls)) or name == class_name or (
            code is not None and status == code)

    if is_("AuthenticationError", 401):
        return ProviderAuthenticationError("Anthropic rejected the API key (401)")
    if is_("PermissionDeniedError", 403):
        return ProviderPermissionError("This API key lacks permission for the request (403)")
    if is_("NotFoundError", 404):
        return ProviderModelUnavailableError("The selected Claude model is unavailable for this key (404)")
    if is_("RateLimitError", 429):
        failure = ProviderRateLimitError("Anthropic rate limit reached (429)")
        failure.retry_after = _retry_after(error)
        return failure
    if is_("RequestTooLargeError", 413):
        return ProviderContextLimitError("The request is too large (413); start a new conversation")
    if is_("OverloadedError", 529) or is_("ServiceUnavailableError", 503) or is_("InternalServerError") or (
            isinstance(status, int) and status >= 500):
        return ProviderOverloadedError(f"Anthropic is temporarily unavailable ({status or name})",
                                       retry_after=_retry_after(error))
    if is_("BadRequestError", 400):
        return ProviderBadRequestError("Anthropic rejected the request (400)", request_id=request_id)
    if is_("APIStatusError"):
        return ProviderError(f"Anthropic request failed ({status})")
    if is_("APITimeoutError") or is_("APIConnectionError"):
        return ProviderOfflineError("Claude is unreachable; GhostGUI commands still work offline")
    return ProviderError(f"Anthropic request failed ({name})")


def _retry_after(error) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class AnthropicAgentProvider:
    """Owns one ``AsyncAnthropic`` client per key/base URL, reused across runs."""

    provider_name = "anthropic"
    supports_agent_loop = True
    supports_mid_conversation_system = True

    def __init__(self, *, api_key: SecretStr | str | None = None, client: Any | None = None,
                 cancellation_poll_seconds: float = 0.05):
        install_log_scrubbing()
        self._poll = float(cancellation_poll_seconds)
        self._owns_client = client is None
        if client is None:
            if api_key is None:
                raise ProviderConfigurationError("Add an Anthropic API key in Claude settings")
            secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
            try:
                import anthropic
            except ImportError as error:
                raise ProviderConfigurationError(
                    "Claude support is not installed; install GhostGUI with the ai extra") from error
            client = anthropic.AsyncAnthropic(
                api_key=secret.reveal(), max_retries=2, timeout=anthropic.Timeout(600, connect=10))
        self._client = client

    async def stream_turn(self, request: AgentTurnRequest, on_event: StreamCallback | None = None,
                          cancellation_token=None) -> AgentTurnResponse:
        arguments = build_stream_arguments(request)
        emit = on_event or (lambda event: None)
        task = asyncio.ensure_future(self._run_stream(arguments, emit))
        try:
            while not task.done():
                if cancellation_token is not None and cancellation_token.cancellation_requested:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    raise ProviderCancelledError("Claude request was cancelled")
                await asyncio.wait({task}, timeout=self._poll)
            return task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _run_stream(self, arguments, emit) -> AgentTurnResponse:
        try:
            manager = self._client.beta.messages.stream(**arguments)
            async with manager as stream:
                emit(AgentStreamEvent("turn_start"))
                async for event in stream:
                    self._relay(event, emit)
                message = await stream.get_final_message()
        except (ProviderError, asyncio.CancelledError):
            raise
        except ValueError as error:  # eager tool input that cannot be parsed
            raise ProviderMalformedStreamError("Claude streamed an unparseable tool input") from error
        except Exception as error:
            raise map_error(error) from error
        return parse_message(message)

    @staticmethod
    def _relay(event, emit) -> None:
        kind = getattr(event, "type", "")
        if kind == "text":
            emit(AgentStreamEvent("text", text=str(getattr(event, "text", ""))))
        elif kind == "thinking":
            emit(AgentStreamEvent("thinking", text=str(getattr(event, "thinking", ""))))
        elif kind == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", "") == "tool_use":
                emit(AgentStreamEvent("tool_use_start", name=str(getattr(block, "name", "")),
                                      index=int(getattr(event, "index", -1))))

    async def prewarm(self, request: AgentTurnRequest):
        """Write the tools+system cache with max_tokens=0 (no output tokens, not streamed).

        Used only when the user opts in and the bar is focused after an idle period.
        """
        arguments = build_stream_arguments(request)
        arguments["max_tokens"] = 0
        try:
            message = await self._client.beta.messages.create(**arguments)
        except Exception as error:
            raise map_error(error) from error
        return parse_usage(_to_dict(message).get("usage"), request.model)

    async def validate_key(self, models=AGENT_MODELS) -> dict[str, dict]:
        """Zero-token check: retrieve each model; returns id → {ok, max_input_tokens | error}."""
        results = {}
        for model in models:
            try:
                info = _to_dict(await self._client.models.retrieve(model))
                results[model] = {"ok": True, "max_input_tokens": info.get("max_input_tokens"),
                                  "display_name": info.get("display_name", model)}
            except Exception as error:
                mapped = map_error(error)
                results[model] = {"ok": False, "error": type(mapped).__name__, "message": str(mapped)}
                if isinstance(mapped, (ProviderAuthenticationError, ProviderOfflineError)):
                    break
        return results

    async def aclose(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if close is not None:
                await close()

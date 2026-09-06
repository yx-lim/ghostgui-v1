"""Explicit development-only recording and replay of normalized responses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from application.ai.errors import ProviderCancelledError, ProviderError
from application.ai.providers.base import (
    CancellationSignal,
    LLMProvider,
    validate_provider_request,
    validate_provider_response,
)
from application.ai.schemas import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    ToolCall,
    Usage,
)


RECORDING_FORMAT_VERSION = 1
REQUEST_FINGERPRINT_VERSION = 1


class ProviderRecordingError(ProviderError):
    """A development recording is missing, malformed, or unsafe to create."""


class ProviderRecordingStore(Protocol):
    def get(self, fingerprint: str) -> ProviderResponse | None:
        ...

    def put(self, fingerprint: str, response: ProviderResponse) -> None:
        ...


class InMemoryRecordingStore:
    """Non-persistent recording store for deterministic tests."""

    def __init__(self) -> None:
        self._responses: dict[str, ProviderResponse] = {}

    def get(self, fingerprint: str) -> ProviderResponse | None:
        return self._responses.get(fingerprint)

    def put(self, fingerprint: str, response: ProviderResponse) -> None:
        self._responses[fingerprint] = response


class JsonRecordingStore:
    """Persist only fingerprints and caller-sanitized normalized responses."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("recording path must name a file")

    def get(self, fingerprint: str) -> ProviderResponse | None:
        entries = self._load_entries()
        payload = entries.get(_validate_fingerprint(fingerprint))
        if payload is None:
            return None
        return _response_from_payload(payload)

    def put(self, fingerprint: str, response: ProviderResponse) -> None:
        fingerprint = _validate_fingerprint(fingerprint)
        if not isinstance(response, ProviderResponse):
            raise TypeError("recording response must use the normalized contract")
        entries = self._load_entries()
        entries[fingerprint] = _response_payload(response)
        payload = {
            "format_version": RECORDING_FORMAT_VERSION,
            "entries": entries,
        }
        parent = self.path.parent
        if not parent.is_dir():
            raise ProviderRecordingError("recording directory does not exist")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        except (OSError, TypeError, ValueError) as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise ProviderRecordingError("could not write provider recording") from error

    def _load_entries(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ProviderRecordingError("provider recording is unreadable") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"format_version", "entries"}
            or payload["format_version"] != RECORDING_FORMAT_VERSION
            or not isinstance(payload["entries"], dict)
        ):
            raise ProviderRecordingError("provider recording format is invalid")
        entries = {}
        for fingerprint, response in payload["entries"].items():
            entries[_validate_fingerprint(fingerprint)] = response
        return entries


class RecordedProvider:
    """Delegate live work and record only an explicitly sanitized response."""

    def __init__(
        self,
        delegate: LLMProvider,
        store: ProviderRecordingStore,
        *,
        sanitizer: Callable[[ProviderResponse], ProviderResponse],
        development_mode: bool = False,
    ) -> None:
        if not development_mode:
            raise ProviderRecordingError(
                "provider recording requires explicit development_mode=True"
            )
        if not callable(sanitizer):
            raise TypeError("provider recording requires a response sanitizer")
        self.delegate = delegate
        self.store = store
        self.sanitizer = sanitizer

    @property
    def provider_name(self) -> str:
        return self.delegate.provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.delegate.capabilities

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        response = await self.delegate.generate(request, cancellation_token)
        sanitized = self.sanitizer(response)
        if not isinstance(sanitized, ProviderResponse):
            raise ProviderRecordingError(
                "provider recording sanitizer returned an invalid response"
            )
        validate_provider_response(sanitized, self.capabilities)
        self.store.put(
            provider_request_fingerprint(self.provider_name, request),
            sanitized,
        )
        return response

    async def aclose(self) -> None:
        close = getattr(self.delegate, "aclose", None)
        if close is not None:
            await close()


class ReplayProvider:
    """Replay a sanitized response only for an identical normalized request."""

    def __init__(
        self,
        provider_name: str,
        capabilities: ProviderCapabilities,
        store: ProviderRecordingStore,
    ) -> None:
        if not provider_name.strip():
            raise ValueError("replay provider name must not be empty")
        self._provider_name = provider_name.strip().lower()
        self._capabilities = capabilities
        self.store = store

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(
        self,
        request: ProviderRequest,
        cancellation_token: CancellationSignal | None = None,
    ) -> ProviderResponse:
        validate_provider_request(request, self.capabilities)
        if cancellation_token is not None and cancellation_token.cancellation_requested:
            raise ProviderCancelledError("provider replay was cancelled")
        fingerprint = provider_request_fingerprint(self.provider_name, request)
        response = self.store.get(fingerprint)
        if response is None:
            raise ProviderRecordingError(
                "no sanitized provider recording matches this request"
            )
        validate_provider_response(response, self.capabilities)
        return response

    async def aclose(self) -> None:
        return None


def provider_request_fingerprint(provider_name: str, request: ProviderRequest) -> str:
    """Hash normalized prompt/context, images, model, and semantic schemas."""

    provider = provider_name.strip().lower()
    if not provider:
        raise ValueError("provider fingerprint name must not be empty")
    payload = {
        "fingerprint_version": REQUEST_FINGERPRINT_VERSION,
        "provider": provider,
        "model": request.model,
        "messages": [
            {
                "role": message.role.value,
                "text": message.text,
                "motion_frames": [
                    {
                        "content_sha256": hashlib.sha256(frame.data).hexdigest(),
                        "mime_type": frame.mime_type,
                        "time_seconds": frame.time_seconds,
                        "variant": frame.variant.value,
                        "comparison_id": frame.comparison_id,
                        "label": frame.label,
                    }
                    for frame in message.motion_frames
                ],
                "tool_calls": [
                    {
                        "identifier": call.identifier,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                    for call in message.tool_calls
                ],
                "tool_results": [
                    {
                        "call_id": result.call_id,
                        "name": result.name,
                        "output": result.output,
                        "is_error": result.is_error,
                    }
                    for result in message.tool_results
                ],
            }
            for message in request.messages
        ],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ],
        "response_schema": (
            None if request.response_schema is None else dict(request.response_schema)
        ),
        "max_output_tokens": request.max_output_tokens,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProviderRecordingError(
            "provider request cannot be fingerprinted safely"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _response_payload(response: ProviderResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "tool_calls": [
            {
                "identifier": call.identifier,
                "name": call.name,
                "arguments": deepcopy(dict(call.arguments)),
            }
            for call in response.tool_calls
        ],
        "stop_reason": response.stop_reason.value,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def _response_from_payload(payload: Any) -> ProviderResponse:
    if not isinstance(payload, Mapping) or set(payload) != {
        "text",
        "tool_calls",
        "stop_reason",
        "usage",
    }:
        raise ProviderRecordingError("recorded provider response is invalid")
    calls = payload["tool_calls"]
    usage = payload["usage"]
    if (
        not isinstance(payload["text"], str)
        or not isinstance(payload["stop_reason"], str)
        or not isinstance(calls, list)
        or not isinstance(usage, Mapping)
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(
            not isinstance(value, Mapping)
            or set(value) != {"identifier", "name", "arguments"}
            for value in calls
        )
    ):
        raise ProviderRecordingError("recorded provider response is invalid")
    try:
        return ProviderResponse(
            text=payload["text"],
            tool_calls=tuple(
                ToolCall(
                    value["identifier"],
                    value["name"],
                    value["arguments"],
                )
                for value in calls
            ),
            stop_reason=StopReason(payload["stop_reason"]),
            usage=Usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderRecordingError("recorded provider response is invalid") from error


def _validate_fingerprint(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderRecordingError("provider recording fingerprint is invalid")
    return value

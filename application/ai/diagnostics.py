"""Opt-in, bounded developer diagnostics for the compact Motion Assistant."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from application.ai.schemas import ProviderRequest, ProviderResponse


AI_DEBUG_ENVIRONMENT_VARIABLE = "GHOSTGUI_AI_DEBUG"
AI_DEBUG_DIRECTORY = ".ghostgui-ai-debug"
MAX_DIAGNOSTIC_STRING_CHARACTERS = 120_000
_SECRET_KEY = re.compile(
    r"api[_-]?key|authorization|credential|keyring|access[_-]?token|secret",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(?:sk-ant-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)


class MotionAssistantDiagnostics:
    """Collect one workflow trace; no-op unless explicitly enabled."""

    def __init__(self, *, enabled: bool = False, directory: str | Path | None = None):
        self.enabled = bool(enabled)
        self.directory = Path(directory or AI_DEBUG_DIRECTORY)
        self._payload: dict[str, Any] = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_environment(cls, *, directory: str | Path | None = None):
        enabled = os.environ.get(AI_DEBUG_ENVIRONMENT_VARIABLE, "").strip().lower()
        return cls(enabled=enabled in {"1", "true", "yes", "on"}, directory=directory)

    def record_planning(
        self,
        *,
        provider_name: str,
        request: ProviderRequest,
        response: ProviderResponse,
        parsed_spec,
        latency_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        self._payload["planning"] = _sanitize({
            "provider": provider_name,
            "model": request.model,
            "messages": [
                {
                    "role": message.role.value,
                    "text": message.text,
                    "frames": [
                        {
                            "timestamp_seconds": frame.time_seconds,
                            "variant": frame.variant.value,
                            "comparison_id": frame.comparison_id,
                            "mime_type": frame.mime_type,
                            "content_sha256": hashlib.sha256(frame.data).hexdigest(),
                        }
                        for frame in message.motion_frames
                    ],
                }
                for message in request.messages
            ],
            "normalized_response": asdict(response),
            "parsed_spec": asdict(parsed_spec),
            "token_usage": asdict(response.usage),
            "latency_seconds": float(latency_seconds),
        })

    def record_execution(self, execution_result) -> None:
        if not self.enabled:
            return
        self._payload["execution"] = _sanitize(
            asdict(execution_result)
            if is_dataclass(execution_result)
            else execution_result
        )

    def write(self) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime("motion-assistant-%Y%m%dT%H%M%S%fZ.json")
        destination = self.directory / name
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".motion-assistant-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(_sanitize(self._payload), handle, sort_keys=True, indent=2)
                handle.write("\n")
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, str):
        redacted = _SECRET_TEXT.sub("[REDACTED]", value)
        if len(redacted) > MAX_DIAGNOSTIC_STRING_CHARACTERS:
            return redacted[:MAX_DIAGNOSTIC_STRING_CHARACTERS] + "...[truncated]"
        return redacted
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    return enum_value if isinstance(enum_value, (str, int, float, bool)) else repr(value)

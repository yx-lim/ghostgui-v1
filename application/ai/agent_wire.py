"""Provider-neutral values for the in-app agent loop (lossless, append-only turns).

The conversation is held as Messages-API-shaped JSON blocks. Assistant content is
stored exactly as the provider returned it (thinking, fallback and tool_use blocks
included) and replayed verbatim, which preserved thinking requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Callable, Mapping


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class AgentStopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"
    CONTEXT_EXCEEDED = "model_context_window_exceeded"
    STOP_SEQUENCE = "stop_sequence"
    OTHER = "other"

    @classmethod
    def parse(cls, value: str | None) -> "AgentStopReason":
        try:
            return cls(value or "end_turn")
        except ValueError:
            return cls.OTHER


@dataclass(frozen=True)
class AgentTurnRequest:
    model: str
    system: str
    tools: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, Any], ...]
    max_tokens: int = 64_000
    effort: Effort = Effort.MEDIUM
    thinking_display: str = "summarized"
    cache_ttl: str = "1h"
    fallbacks: bool = True
    eager_input_streaming: bool = True

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("agent model must not be empty")
        if not self.messages:
            raise ValueError("agent request needs at least one message")
        if self.messages[0].get("role") != "user":
            raise ValueError("the first agent message must be a user message")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.cache_ttl not in {"5m", "1h"}:
            raise ValueError("cache_ttl must be 5m or 1h")


@dataclass(frozen=True)
class UsageIteration:
    type: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0


@dataclass(frozen=True)
class AgentUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    served_model: str = ""
    iterations: tuple[UsageIteration, ...] = ()

    def __add__(self, other: "AgentUsage") -> "AgentUsage":
        return AgentUsage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
            self.cache_read + other.cache_read, self.cache_write_5m + other.cache_write_5m,
            self.cache_write_1h + other.cache_write_1h, other.served_model or self.served_model,
            self.iterations + other.iterations,
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write_5m + self.cache_write_1h


@dataclass(frozen=True)
class AgentTurnResponse:
    content: tuple[Mapping[str, Any], ...]
    stop_reason: AgentStopReason
    usage: AgentUsage = field(default_factory=AgentUsage)
    model: str = ""
    request_id: str = ""
    stop_details: Mapping[str, Any] | None = None

    @property
    def tool_uses(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(block for block in self.content if block.get("type") == "tool_use")

    @property
    def text(self) -> str:
        return "".join(block.get("text", "") for block in self.content if block.get("type") == "text")

    @property
    def refusal_category(self) -> str | None:
        return None if not self.stop_details else self.stop_details.get("category")


@dataclass(frozen=True)
class AgentStreamEvent:
    """Incremental UI signal; never needed to reconstruct the final message."""

    kind: str  # text | thinking | tool_use_start | turn_start
    text: str = ""
    name: str = ""
    index: int = -1


StreamCallback = Callable[[AgentStreamEvent], None]


def user_text(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_result_block(tool_use_id: str, payload: Any, *, is_error: bool = False,
                      images: tuple = ()) -> dict:
    """A tool_result whose content holds JSON text plus optional base64 image parts."""
    import base64

    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, allow_nan=False)
    content: list[dict] = [{"type": "text", "text": text}]
    for image in images:
        data = getattr(image, "data", image)
        mime = getattr(image, "mime_type", "image/png")
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": mime, "data": base64.b64encode(data).decode("ascii")}})
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

"""Provider-neutral values exchanged by GhostGUI's AI application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    COMPLETE = "complete"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"


class ImageVariant(str, Enum):
    ORIGINAL = "original"
    CANDIDATE = "candidate"


class EditAuthor(str, Enum):
    IMPORTED = "imported"
    AI = "ai"
    USER = "user"


@dataclass(frozen=True)
class MotionEntityRef:
    """Opaque motion-entity identity used at AI boundaries.

    Metadata services own creation and resolution of this value. Callers must
    not parse it or infer that it contains a frame name or timestamp.
    """

    identifier: str

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("motion entity identifier must not be empty")


@dataclass(frozen=True)
class MotionFrameImage:
    """Rendered motion evidence with mandatory time correspondence."""

    data: bytes
    mime_type: str
    time_seconds: float
    variant: ImageVariant
    comparison_id: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("motion frame image data must not be empty")
        if self.mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported motion frame image MIME type")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("motion frame time must be finite and non-negative")
        if not self.comparison_id.strip():
            raise ValueError("motion frame comparison_id must not be empty")


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_tools: bool
    supports_vision: bool
    supports_structured_output: bool = False
    supports_parallel_tool_calls: bool = False
    supports_system_messages: bool = True
    max_images_per_request: int = 0

    def __post_init__(self) -> None:
        if self.max_images_per_request < 0:
            raise ValueError("max_images_per_request must be non-negative")
        if not self.supports_vision and self.max_images_per_request:
            raise ValueError("a text-only provider cannot accept images")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    identifier: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("tool call identifier must not be empty")
        if not self.name.strip():
            raise ValueError("tool call name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: Any
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.name.strip():
            raise ValueError("tool result call_id and name must not be empty")


@dataclass(frozen=True)
class ProviderMessage:
    role: MessageRole
    text: str = ""
    motion_frames: tuple[MotionFrameImage, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        if not (self.text or self.motion_frames or self.tool_results):
            raise ValueError("provider message must contain text, images, or tool results")
        if self.motion_frames and self.role is not MessageRole.USER:
            raise ValueError("motion frame images are only valid on user messages")
        if self.tool_results and self.role is not MessageRole.TOOL:
            raise ValueError("tool results require the tool message role")
        comparison_times: dict[str, float] = {}
        comparison_variants: set[tuple[str, ImageVariant]] = set()
        for frame in self.motion_frames:
            previous_time = comparison_times.setdefault(
                frame.comparison_id,
                frame.time_seconds,
            )
            if previous_time != frame.time_seconds:
                raise ValueError(
                    "original and candidate comparison frames must use identical times"
                )
            comparison_variant = (frame.comparison_id, frame.variant)
            if comparison_variant in comparison_variants:
                raise ValueError("comparison contains a duplicate image variant")
            comparison_variants.add(comparison_variant)


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    messages: tuple[ProviderMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    response_schema: Mapping[str, Any] | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("provider model must not be empty")
        if not self.messages:
            raise ValueError("provider request must contain at least one message")
        if self.response_schema is not None and not isinstance(
            self.response_schema,
            Mapping,
        ):
            raise TypeError("provider response_schema must be an object")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token usage must be non-negative")


@dataclass(frozen=True)
class ProviderResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = StopReason.COMPLETE
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if self.stop_reason is StopReason.TOOL_CALLS and not self.tool_calls:
            raise ValueError("tool_calls stop reason requires at least one tool call")
        if self.tool_calls and self.stop_reason is not StopReason.TOOL_CALLS:
            raise ValueError("tool calls require the tool_calls stop reason")

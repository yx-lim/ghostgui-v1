"""Ephemeral visible transcript state for the Motion Assistant UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChatEntryKind(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    ACTIVITY = "activity"
    RESULT = "result"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"


@dataclass(frozen=True)
class ChatTranscriptEntry:
    kind: ChatEntryKind
    text: str
    turn_id: int | None = None
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ChatEntryKind(self.kind))
        text = str(self.text).strip()
        if not text:
            raise ValueError("chat transcript entry text must not be empty")
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self,
            "details",
            tuple(str(value).strip() for value in self.details if str(value).strip()),
        )


class MotionChatTranscript:
    """Append-only visible history, deliberately separate from provider context."""

    def __init__(self) -> None:
        self._entries: list[ChatTranscriptEntry] = []
        self._turn_id = 0

    @property
    def entries(self) -> tuple[ChatTranscriptEntry, ...]:
        return tuple(self._entries)

    @property
    def current_turn_id(self) -> int | None:
        return self._turn_id or None

    def start_turn(self, instruction: str) -> ChatTranscriptEntry:
        self._turn_id += 1
        return self.append(ChatEntryKind.USER, instruction, turn_id=self._turn_id)

    def append(
        self,
        kind: ChatEntryKind,
        text: str,
        *,
        turn_id: int | None = None,
        details=(),
    ) -> ChatTranscriptEntry:
        entry = ChatTranscriptEntry(
            kind,
            text,
            self.current_turn_id if turn_id is None else turn_id,
            tuple(details),
        )
        self._entries.append(entry)
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self._turn_id = 0

"""Append-only command-bar conversation in Messages-API wire format."""

from __future__ import annotations

from copy import deepcopy
import json
from threading import Lock
from typing import Any, Mapping


class AgentConversation:
    """History is never edited or truncated; earlier entries are frozen once appended.

    Placement rules enforced here:
    - the first entry is a user message;
    - a system message follows a user message and is either last or followed by an
      assistant message.
    """

    def __init__(self, model: str):
        self.model = model
        self._messages: list[dict] = []
        self._frozen: list[str] = []
        self._lock = Lock()

    def _append(self, message: dict) -> None:
        with self._lock:
            role = message["role"]
            previous = self._messages[-1]["role"] if self._messages else None
            if previous is None and role != "user":
                raise ValueError("the conversation must start with a user message")
            if role == "system" and previous not in ("user", "system"):
                raise ValueError("a system message must follow a user message")
            if previous == "system" and role == "user":
                raise ValueError("a system message must be last or followed by an assistant turn")
            frozen = json.dumps(message, sort_keys=True, default=str)
            self._messages.append(json.loads(frozen))
            self._frozen.append(frozen)

    def add_user(self, content: list[Mapping[str, Any]]) -> None:
        if not content:
            raise ValueError("user content must not be empty")
        self._append({"role": "user", "content": [dict(block) for block in content]})

    def add_system(self, text: str) -> None:
        self._append({"role": "system", "content": text})

    def add_assistant(self, content) -> None:
        self._append({"role": "assistant", "content": [dict(block) for block in content]})

    @property
    def last_role(self) -> str | None:
        return self._messages[-1]["role"] if self._messages else None

    def __len__(self) -> int:
        return len(self._messages)

    def wire(self) -> tuple[dict, ...]:
        """Deep copies, verified against the frozen serialisation (append-only check)."""
        with self._lock:
            for message, frozen in zip(self._messages, self._frozen):
                if json.dumps(message, sort_keys=True, default=str) != frozen:
                    raise RuntimeError("conversation history was edited; start a new conversation")
            return tuple(deepcopy(message) for message in self._messages)

    def pending_tool_uses(self) -> tuple[str, ...]:
        """tool_use ids in the last assistant message that have no result yet."""
        if not self._messages or self._messages[-1]["role"] != "assistant":
            return ()
        return tuple(block["id"] for block in self._messages[-1]["content"]
                     if block.get("type") == "tool_use")

"""Bounded, scrubbed record of executed commands for transcripts and get_recent_actions."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import re
from threading import Lock
import time

_SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}|AIza[0-9A-Za-z_\-]{20,}")


def scrub(text: str, limit: int = 200) -> str:
    cleaned = _SECRET.sub("[redacted]", str(text))
    cleaned = "".join(character for character in cleaned if character.isprintable())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


@dataclass(frozen=True)
class ActionRecord:
    timestamp: float
    origin: str
    run_id: str
    command: str
    arguments: str
    status: str
    code: str = ""
    duration_ms: int = 0
    revision: int | None = None


class ActionLog:
    def __init__(self, capacity: int = 2000):
        self._records: deque[ActionRecord] = deque(maxlen=capacity)
        self._lock = Lock()

    def record(self, *, origin, command, arguments, status, run_id="", code="", duration_ms=0, revision=None):
        try:
            encoded = json.dumps({k: v for k, v in dict(arguments).items() if k != "expected_revision"},
                                 sort_keys=True, default=str)
        except Exception:
            encoded = "{}"
        record = ActionRecord(time.time(), str(origin), scrub(run_id, 40), scrub(command, 64), scrub(encoded),
                              scrub(status, 32), scrub(code, 64), int(duration_ms), revision)
        with self._lock:
            self._records.append(record)
        return record

    def tail(self, limit: int = 20) -> list[dict]:
        with self._lock:
            records = list(self._records)[-max(0, int(limit)):]
        return [asdict(record) for record in records]

    def __len__(self):
        with self._lock:
            return len(self._records)

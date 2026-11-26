"""Events emitted by the agent loop for the transcript UI (Qt-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RunOutcome(str, Enum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    REFUSED = "refused"
    CONTEXT_FULL = "context_full"
    TURN_LIMIT = "turn_limit"
    BUDGET = "budget"
    TOOL_ERRORS = "tool_errors"
    ERROR = "error"
    PLAN_READY = "plan_ready"


@dataclass(frozen=True)
class LoopEvent:
    kind: str
    # kinds: run_started, turn_started, text, thinking, tool_preparing, tool_started,
    # tool_finished, tool_deferred, confirmation, question, notice, usage, run_finished
    text: str = ""
    tool: str = ""
    call_id: str = ""
    title: str = ""
    summary: str = ""
    ok: bool = True
    duration_ms: int = 0
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeferredCall:
    tool: str
    arguments: Mapping[str, Any]
    preview: str


@dataclass(frozen=True)
class RunResult:
    outcome: RunOutcome
    text: str = ""
    turns: int = 0
    actions: int = 0
    cost_usd: float | None = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    message: str = ""
    refusal_category: str | None = None
    deferred: tuple[DeferredCall, ...] = ()
    candidate_touched: bool = False

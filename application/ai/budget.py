"""Run and session spend tracking for the command bar (estimates, checked between turns)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from application.ai.agent_wire import AgentUsage
from application.ai.pricing import cost_usd


@dataclass
class SessionBudget:
    cap_usd: float = 10.0
    spent_usd: float = 0.0
    usage: AgentUsage = field(default_factory=AgentUsage)
    unpriced_turns: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, usage: AgentUsage, model: str) -> float | None:
        cost = cost_usd(usage, model)
        with self._lock:
            self.usage = self.usage + usage
            if cost is None:
                self.unpriced_turns += 1
            else:
                self.spent_usd += cost
        return cost

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)


@dataclass
class RunBudget:
    cap_usd: float
    session: SessionBudget
    spent_usd: float = 0.0
    usage: AgentUsage = field(default_factory=AgentUsage)
    notice_sent: bool = False

    def add(self, usage: AgentUsage, model: str) -> None:
        cost = self.session.add(usage, model)
        self.usage = self.usage + usage
        if cost is not None:
            self.spent_usd += cost

    def fraction(self) -> float:
        run = self.spent_usd / self.cap_usd if self.cap_usd else 1.0
        session = self.session.spent_usd / self.session.cap_usd if self.session.cap_usd else 1.0
        return max(run, session)

    def exhausted(self) -> bool:
        return self.fraction() >= 1.0

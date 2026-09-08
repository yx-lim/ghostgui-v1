"""Bounded high-level conversation state for compact motion refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.ai.limits import MAX_AI_INSTRUCTION_CHARACTERS


MAX_RETAINED_REFINEMENTS = 3


@dataclass
class TrajectoryConversation:
    """Retain intent, never an unbounded provider transcript."""

    original_goal: str
    _refinements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.original_goal = _instruction(self.original_goal)
        self._refinements = [
            _instruction(value)
            for value in self._refinements[-MAX_RETAINED_REFINEMENTS:]
        ]

    @property
    def refinements(self) -> tuple[str, ...]:
        return tuple(self._refinements)

    def record_refinement(self, instruction: str) -> None:
        self._refinements.append(_instruction(instruction))
        del self._refinements[:-MAX_RETAINED_REFINEMENTS]

    def to_context(self) -> dict:
        return {
            "original_goal": self.original_goal,
            "recent_refinements": list(self._refinements),
            "source": "current_staged_candidate",
        }


def _instruction(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("motion conversation instruction must not be empty")
    if len(value) > MAX_AI_INSTRUCTION_CHARACTERS:
        raise ValueError("motion conversation instruction exceeds the local size limit")
    return value

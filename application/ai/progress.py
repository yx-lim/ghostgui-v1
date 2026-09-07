"""Provider-neutral progress events for AI motion workflow presentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class AIProgressStage(str, Enum):
    PLANNING_STARTED = "planning_started"
    TEXT_DELTA = "text_delta"
    STRUCTURED_PLAN_COMPLETED = "structured_plan_completed"
    LOCAL_OPERATION = "local_operation"
    VALIDATION = "validation"
    DONE = "done"


@dataclass(frozen=True)
class AIProgressEvent:
    """A deterministic workflow event; provider text is never executed."""

    stage: AIProgressStage
    operation_index: int | None = None
    operation_count: int | None = None
    text_delta: str = ""
    repair: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", AIProgressStage(self.stage))
        is_operation = self.stage is AIProgressStage.LOCAL_OPERATION
        if is_operation:
            if (
                self.operation_index is None
                or self.operation_count is None
                or self.operation_count < 1
                or not 1 <= self.operation_index <= self.operation_count
            ):
                raise ValueError("local operation progress requires a valid index/count")
        elif self.operation_index is not None:
            raise ValueError("operation index is only valid for local operation progress")
        if self.stage is AIProgressStage.STRUCTURED_PLAN_COMPLETED:
            if self.operation_count is None or self.operation_count < 0:
                raise ValueError("completed plan progress requires an operation count")
        elif not is_operation and self.operation_count is not None:
            raise ValueError(
                "operation count is only valid for plan and operation progress"
            )
        if self.stage is AIProgressStage.TEXT_DELTA:
            if not self.text_delta:
                raise ValueError("text delta progress must not be empty")
        elif self.text_delta:
            raise ValueError("text delta is only valid for text delta progress")

    @property
    def message(self) -> str:
        if self.stage is AIProgressStage.PLANNING_STARTED:
            return (
                "Planning replacement operations…"
                if self.repair
                else "Planning motion…"
            )
        if self.stage is AIProgressStage.TEXT_DELTA:
            return self.text_delta
        if self.stage is AIProgressStage.STRUCTURED_PLAN_COMPLETED:
            count = int(self.operation_count or 0)
            noun = "operation" if count == 1 else "operations"
            return f"Plan ready: {count} {noun}."
        if self.stage is AIProgressStage.LOCAL_OPERATION:
            return (
                f"Executing operation {self.operation_index}/"
                f"{self.operation_count}…"
            )
        if self.stage is AIProgressStage.VALIDATION:
            return "Validating candidate…"
        return "Motion candidate ready."


AIProgressCallback = Callable[[AIProgressEvent], None]


def report_progress(
    callback: AIProgressCallback | None,
    event: AIProgressEvent,
) -> None:
    if callback is not None:
        callback(event)

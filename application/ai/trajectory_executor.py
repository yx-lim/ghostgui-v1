"""Deterministic local execution boundary for compact trajectory specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from application.ai.edit_session import AIEditSession
from application.ai.errors import ProviderCancelledError
from application.ai.motion_services import MotionValidationReport
from application.ai.providers.base import CancellationSignal
from application.ai.progress import (
    AIProgressCallback,
    AIProgressEvent,
    AIProgressStage,
    report_progress,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryEditSpec,
    TrajectoryOperation,
    TrajectoryOperationType,
)


class TrajectoryExecutionError(RuntimeError):
    """A compact operation could not be executed safely and locally."""


@dataclass(frozen=True)
class TrajectoryExecutionContext:
    session: AIEditSession
    services: Any


@dataclass(frozen=True)
class TrajectoryOperationResult:
    operation_type: TrajectoryOperationType
    output: Mapping[str, Any]


@dataclass(frozen=True)
class TrajectoryExecutionResult:
    spec: TrajectoryEditSpec
    operations: tuple[TrajectoryOperationResult, ...]
    validation: MotionValidationReport


TrajectoryOperationHandler = Callable[
    [TrajectoryOperation, TrajectoryExecutionContext],
    Mapping[str, Any],
]


class TrajectorySpecExecutor:
    """Execute one complete spec locally, then validate its exact revision."""

    def __init__(
        self,
        handlers: Mapping[TrajectoryOperationType, TrajectoryOperationHandler],
        validator: Callable[[object], MotionValidationReport],
    ) -> None:
        self.handlers = {
            TrajectoryOperationType(key): value
            for key, value in handlers.items()
        }
        self.validator = validator

    def execute(
        self,
        spec: TrajectoryEditSpec,
        *,
        context: TrajectoryExecutionContext,
        cancellation_token: CancellationSignal | None = None,
        progress_callback: AIProgressCallback | None = None,
    ) -> TrajectoryExecutionResult:
        checkpoint = context.session.checkpoint()
        results = []
        try:
            operation_count = len(spec.operations)
            for operation_index, operation in enumerate(spec.operations, start=1):
                _raise_if_cancelled(cancellation_token)
                report_progress(
                    progress_callback,
                    AIProgressEvent(
                        AIProgressStage.LOCAL_OPERATION,
                        operation_index=operation_index,
                        operation_count=operation_count,
                    ),
                )
                handler = self.handlers.get(operation.operation_type)
                if handler is None:
                    raise TrajectoryExecutionError(
                        f"unsupported local trajectory operation: "
                        f"{operation.operation_type.value}"
                    )
                output = handler(operation, context)
                if not isinstance(output, Mapping):
                    raise TrajectoryExecutionError(
                        "trajectory operation returned an invalid local result"
                    )
                results.append(TrajectoryOperationResult(
                    operation.operation_type,
                    dict(output),
                ))
            _raise_if_cancelled(cancellation_token)
        except BaseException:
            context.session.restore_checkpoint(checkpoint)
            raise

        if not context.session.has_changes:
            raise TrajectoryExecutionError("trajectory specification made no motion change")
        context.session.invalidate_validation()
        report_progress(
            progress_callback,
            AIProgressEvent(AIProgressStage.VALIDATION),
        )
        validation = self.validator(context.session.working_document)
        if not isinstance(validation, MotionValidationReport):
            raise TrajectoryExecutionError("motion validator returned an invalid result")
        if not validation.valid:
            detail = "; ".join(validation.issues) or "unknown issue"
            raise TrajectoryExecutionError(
                f"staged motion structural/kinematic validation failed: {detail}"
            )
        context.session.mark_current_revision_validated()
        report_progress(
            progress_callback,
            AIProgressEvent(AIProgressStage.DONE),
        )
        return TrajectoryExecutionResult(spec, tuple(results), validation)


def _raise_if_cancelled(token):
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("trajectory execution was cancelled")

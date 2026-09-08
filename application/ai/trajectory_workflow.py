"""Default compact planning and local execution workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from application.ai.diagnostics import MotionAssistantDiagnostics
from application.ai.metadata import MotionMetadataService
from application.ai.schemas import MotionFrameImage, Usage
from application.ai.trajectory_executor import (
    TrajectoryExecutionContext,
    TrajectoryExecutionResult,
    TrajectorySpecExecutor,
)
from application.ai.trajectory_operations import build_trajectory_operation_handlers
from application.ai.trajectory_planner import TrajectoryPlanner, TrajectoryPlanningResult


@dataclass(frozen=True)
class CompactMotionRunResult:
    planning: TrajectoryPlanningResult
    execution: TrajectoryExecutionResult
    context_warnings: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.planning.spec.summary

    @property
    def usage(self) -> Usage:
        return self.planning.usage

    @property
    def provider_requests(self) -> int:
        return self.planning.provider_requests

    @property
    def proposal_lines(self) -> tuple[str, ...]:
        changes = tuple(
            f"{item.operation_type.value.replace('_', ' ')}"
            for item in self.execution.operations
        )
        operation_warnings = tuple(
            warning
            for item in self.execution.operations
            for warning in item.output.get("collision_warnings", ())
        )
        warnings = tuple(
            f"Warning: {warning}"
            for warning in (
                *self.context_warnings,
                *operation_warnings,
                *self.execution.validation.warnings,
            )
        )
        return changes + tuple(dict.fromkeys(warnings))


class CompactMotionWorkflow:
    """One compact provider plan followed by deterministic local execution."""

    def __init__(
        self,
        provider,
        motion_service,
        metadata_service: MotionMetadataService,
        *,
        diagnostics: MotionAssistantDiagnostics | None = None,
    ) -> None:
        self.provider = provider
        self.motion_service = motion_service
        self.metadata_service = metadata_service
        self.diagnostics = diagnostics or MotionAssistantDiagnostics()

    async def run(
        self,
        instruction,
        *,
        model,
        context,
        session,
        motion_frames: tuple[MotionFrameImage, ...] = (),
        conversation_context=None,
        context_warnings: tuple[str, ...] = (),
        cancellation_token=None,
    ) -> CompactMotionRunResult:
        started = monotonic()
        planning = await TrajectoryPlanner(self.provider).plan(
            instruction,
            model=model,
            context=context,
            session=session,
            motion_frames=motion_frames,
            conversation_context=conversation_context,
            cancellation_token=cancellation_token,
        )
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(
                self.motion_service,
                self.metadata_service,
            ),
            self.motion_service.validate_motion,
        )
        execution = executor.execute(
            planning.spec,
            context=TrajectoryExecutionContext(session, self.motion_service),
            cancellation_token=cancellation_token,
        )
        elapsed = monotonic() - started
        diagnostic_request = planning.requests[-1]
        diagnostic_response = planning.responses[-1]
        self.diagnostics.record_planning(
            provider_name=self.provider.provider_name,
            request=diagnostic_request,
            response=diagnostic_response,
            parsed_spec=planning.spec,
            latency_seconds=elapsed,
        )
        self.diagnostics.record_execution(execution)
        self.diagnostics.write()
        return CompactMotionRunResult(planning, execution, tuple(context_warnings))

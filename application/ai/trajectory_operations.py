"""Local handlers for compact trajectory operations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from application.ai.metadata import MotionMetadataService
from application.ai.motion_state import (
    ReplaceMotionState,
    capture_motion_state,
    detached_document,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import TrajectoryExecutionContext


class TrajectoryOperationError(ValueError):
    """A locally valid compact operation cannot be applied to this motion."""


def build_trajectory_operation_handlers(motion_service, metadata_service):
    """Build handlers around existing model and provenance services."""

    return {
        TrajectoryOperationType.ROOT_OFFSET: (
            lambda operation, context: _root_offset(
                operation,
                context,
                motion_service,
                metadata_service,
            )
        ),
    }


def _root_offset(
    operation: TrajectoryOperation,
    context: TrajectoryExecutionContext,
    motion,
    metadata: MotionMetadataService,
):
    document = context.session.working_document
    timeline = document.qpos_timeline
    if timeline is None:
        raise TrajectoryOperationError("root_offset requires an editable qpos timeline")
    free_joints = tuple(motion.adapter.free_joints_by_body.values())
    if not free_joints:
        raise TrajectoryOperationError("robot model has no floating root to offset")
    arguments = operation.arguments
    start = float(arguments["start_time"])
    end = float(arguments["end_time"])
    if end > document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("root_offset scope exceeds the motion duration")
    translation = np.asarray(arguments["translation_m"], dtype=float)
    state_times = tuple(
        float(time)
        for time in timeline.times()
        if start - 1e-9 <= float(time) <= end + 1e-9
    )
    frames = tuple(
        frame
        for frame in document.trajectory.frames
        if start - 1e-9 <= float(frame.time) <= end + 1e-9
    )
    if not state_times and not frames:
        raise TrajectoryOperationError("root_offset scope contains no Keyframes")

    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    affected = tuple(
        working_metadata.reference_for_keyframe(frame) for frame in frames
    ) + tuple(
        working_metadata.reference_for_qpos_keyframe(time) for time in state_times
    )
    candidate = detached_document(document)
    address = int(free_joints[0].qpos_address)
    for time in state_times:
        qpos = np.asarray(candidate.qpos_timeline.get_state(time), dtype=float).copy()
        if address < 0 or address + 7 > len(qpos):
            raise TrajectoryOperationError("floating-root qpos layout is invalid")
        qpos[address:address + 3] += translation
        candidate.qpos_timeline.set_state(time, qpos)
    for frame in frames:
        candidate.trajectory.upsert_frame(replace(
            frame,
            x=float(frame.x) + float(translation[0]),
            y=float(frame.y) + float(translation[1]),
            z=float(frame.z) + float(translation[2]),
        ))
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation="root_offset",
        ),
        affected_entities=affected,
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError("root_offset made no motion change")
    return {
        "start_time": start,
        "end_time": end,
        "translation_m": [float(value) for value in translation],
        "affected_qpos_keyframes": len(state_times),
        "affected_logical_keyframes": len(frames),
    }

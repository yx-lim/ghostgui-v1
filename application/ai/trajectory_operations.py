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
from application.ai.motion_services import JointAngleEditResult, LogicalFrameSolveResult
from application.ai.trajectory_edit_spec import (
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import TrajectoryExecutionContext
from application.ai.semantic_tools import SemanticToolContext, retime_segment


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
        TrajectoryOperationType.HOLD_POSE: (
            lambda operation, context: _hold_pose(
                operation,
                context,
                motion_service,
                metadata_service,
            )
        ),
        TrajectoryOperationType.RETIME_INTERVAL: (
            lambda operation, context: _retime_interval(
                operation,
                context,
                metadata_service,
            )
        ),
        TrajectoryOperationType.SET_JOINT_TARGET: (
            lambda operation, context: _set_joint_target(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.SET_JOINT_GROUP_TARGET: (
            lambda operation, context: _set_joint_group_target(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.SET_END_EFFECTOR_TARGET: (
            lambda operation, context: _set_end_effector_target(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.LOCK_END_EFFECTOR: (
            lambda operation, context: _lock_end_effector(
                operation, context, motion_service, metadata_service
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


def _hold_pose(
    operation: TrajectoryOperation,
    context: TrajectoryExecutionContext,
    motion,
    metadata: MotionMetadataService,
):
    document = context.session.working_document
    timeline = document.qpos_timeline
    if timeline is None:
        raise TrajectoryOperationError("hold_pose requires an editable qpos timeline")
    arguments = operation.arguments
    source_time = float(arguments["source_time"])
    start = float(arguments["start_time"])
    end = float(arguments["end_time"])
    if max(source_time, end) > document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("hold_pose time exceeds the motion duration")
    scope = arguments["body_scope"]
    body_name = arguments["body_name"]
    if scope == "end_effector":
        raise TrajectoryOperationError(
            "End Effector holds require lock_end_effector"
        )

    source_qpos = timeline.sample_state(source_time)
    if source_qpos is None:
        raise TrajectoryOperationError("hold_pose source state is unavailable")
    names = _held_joint_names(scope, body_name, motion)
    source_state = motion.adapter.create_state()
    source_state.set_qpos(source_qpos)
    values = {
        name: source_state.get_joint_value(name)
        for name in names
    }
    existing_times = tuple(float(time) for time in timeline.times())
    target_times = tuple(sorted({
        start,
        end,
        *(
            time for time in existing_times
            if start - 1e-9 <= time <= end + 1e-9
        ),
    }))
    candidate = detached_document(document)
    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    protected_frames = _protected_logical_frames(document, working_metadata)
    affected = []
    created = []
    for time in target_times:
        existed = any(abs(time - value) <= 1e-9 for value in existing_times)
        qpos_reference = working_metadata.reference_for_qpos_keyframe(time)
        affected.append(qpos_reference)
        if not existed:
            created.append(qpos_reference)
        if scope == "whole_body":
            qpos = np.asarray(source_qpos, dtype=float).copy()
            changed_frames = _synchronize_logical_frames(candidate, motion, time, qpos)
        else:
            solved = motion.set_joint_angles(
                candidate,
                time_seconds=time,
                values=values,
                protected_logical_frames=protected_frames,
            )
            if not isinstance(solved, JointAngleEditResult):
                raise TrajectoryOperationError(
                    "Joint Angle service returned an invalid hold result"
                )
            qpos = solved.qpos
            changed_frames = solved.logical_frames
            for frame in changed_frames:
                candidate.trajectory.upsert_frame(frame)
        candidate.qpos_timeline.set_state(time, qpos)
        affected.extend(
            working_metadata.reference_for_keyframe(frame)
            for frame in changed_frames
        )
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation="hold_pose",
        ),
        affected_entities=tuple(dict.fromkeys(affected)),
        created_entities=tuple(dict.fromkeys(created)),
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError("hold_pose made no motion change")
    return {
        "source_time": source_time,
        "start_time": start,
        "end_time": end,
        "body_scope": scope,
        "body_name": body_name,
        "held_qpos_keyframes": len(target_times),
    }


def _retime_interval(
    operation: TrajectoryOperation,
    context: TrajectoryExecutionContext,
    metadata: MotionMetadataService,
):
    arguments = operation.arguments
    start = float(arguments["start_time"])
    end = float(arguments["end_time"])
    scale = float(arguments["scale"])
    if end > context.session.working_document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("retime_interval exceeds the motion duration")
    output = retime_segment(
        SemanticToolContext(context.session, metadata),
        {
            "start_time_seconds": start,
            "end_time_seconds": end,
            "speed": 1.0 / scale,
        },
        allow_user_override=True,
    )
    return {
        **output,
        "duration_scale": scale,
    }


def _set_joint_target(operation, context, motion, metadata):
    arguments = operation.arguments
    return _apply_joint_values(
        context,
        motion,
        metadata,
        float(arguments["time_seconds"]),
        {arguments["joint"]: float(arguments["angle_rad"])},
        "set_joint_target",
    )


def _set_joint_group_target(operation, context, motion, metadata):
    arguments = operation.arguments
    group = arguments["joint_group"]
    if group not in motion.joint_groups:
        raise TrajectoryOperationError(f"unknown Joint Angle group: {group}")
    allowed = set(motion.joint_groups[group])
    values = {
        item["joint"]: float(item["angle_rad"])
        for item in arguments["joint_angles_rad"]
    }
    unknown = set(values) - allowed
    if unknown:
        raise TrajectoryOperationError(
            f"Joint Angle {sorted(unknown)[0]} is not part of group {group}"
        )
    return _apply_joint_values(
        context,
        motion,
        metadata,
        float(arguments["time_seconds"]),
        values,
        "set_joint_group_target",
    )


def _apply_joint_values(context, motion, metadata, time, values, operation_name):
    document = context.session.working_document
    if time > document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("Joint Angle target exceeds the motion duration")
    timeline = document.qpos_timeline
    if timeline is None:
        raise TrajectoryOperationError("Joint Angle target requires a qpos timeline")
    existing_times = tuple(float(value) for value in timeline.times())
    existed = any(abs(time - value) <= 1e-9 for value in existing_times)
    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    solved = motion.set_joint_angles(
        document,
        time_seconds=time,
        values=values,
        protected_logical_frames=_protected_logical_frames(
            document,
            working_metadata,
        ),
    )
    if not isinstance(solved, JointAngleEditResult):
        raise TrajectoryOperationError(
            "Joint Angle service returned an invalid target result"
        )
    candidate = detached_document(document)
    for frame in solved.logical_frames:
        candidate.trajectory.upsert_frame(frame)
    candidate.qpos_timeline.set_state(time, solved.qpos)
    qpos_reference = working_metadata.reference_for_qpos_keyframe(time)
    affected = tuple(
        working_metadata.reference_for_keyframe(frame)
        for frame in solved.logical_frames
    ) + (qpos_reference,)
    context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation=operation_name,
        ),
        affected_entities=affected,
        created_entities=() if existed else (qpos_reference,),
        allow_user_override=True,
    )
    return {
        "time_seconds": time,
        "joint_angles_rad": dict(values),
        "updated_logical_frames": [
            frame.frame_name for frame in solved.logical_frames
        ],
    }


def _set_end_effector_target(operation, context, motion, metadata):
    arguments = operation.arguments
    end_effector = arguments["end_effector"]
    if end_effector not in motion.end_effectors:
        raise TrajectoryOperationError(f"unknown End Effector: {end_effector}")
    mode = "delta" if arguments["mode"] == "relative" else "absolute"
    times = _interval_keyframe_times(
        context.session.working_document,
        float(arguments["start_time"]),
        float(arguments["end_time"]),
    )
    return _apply_logical_frame_targets(
        context,
        motion,
        metadata,
        targets={end_effector: tuple(arguments["position_m"])},
        times=times,
        mode=mode,
        operation_name="set_end_effector_target",
    )


def _lock_end_effector(operation, context, motion, metadata):
    arguments = operation.arguments
    names = tuple(arguments["end_effectors"])
    unknown = set(names) - set(motion.end_effectors)
    if unknown:
        raise TrajectoryOperationError(
            f"unknown End Effector: {sorted(unknown)[0]}"
        )
    document = context.session.working_document
    source_time = float(arguments["source_time"])
    if document.qpos_timeline is None:
        raise TrajectoryOperationError(
            "lock_end_effector requires an editable qpos timeline"
        )
    source_qpos = document.qpos_timeline.sample_state(source_time)
    if source_qpos is None:
        raise TrajectoryOperationError("End Effector lock source state is unavailable")
    state = motion.adapter.create_state()
    state.set_qpos(source_qpos)
    targets = {}
    orientations = {}
    from core.trajectory import quat_to_rpy

    for name in names:
        kind, object_name = motion.adapter.logical_frame_bindings[name]
        position, quaternion = state.get_body_pose(object_name, kind)
        targets[name] = tuple(float(value) for value in position)
        orientations[name] = tuple(float(value) for value in quat_to_rpy(quaternion))
    times = _interval_keyframe_times(
        document,
        float(arguments["start_time"]),
        float(arguments["end_time"]),
    )
    result = _apply_logical_frame_targets(
        context,
        motion,
        metadata,
        targets=targets,
        orientations=orientations,
        times=times,
        mode="absolute",
        operation_name="lock_end_effector",
    )
    return {**result, "source_time": source_time}


def _apply_logical_frame_targets(
    context,
    motion,
    metadata,
    *,
    targets,
    times,
    mode,
    operation_name,
    orientations=None,
):
    document = context.session.working_document
    if document.qpos_timeline is None:
        raise TrajectoryOperationError(
            f"{operation_name} requires an editable qpos timeline"
        )
    candidate = detached_document(document)
    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    protected = _protected_logical_frames(document, working_metadata)
    existing_qpos_times = tuple(float(value) for value in document.qpos_timeline.times())
    existing_logical = {
        (frame.frame_name, float(frame.time))
        for frame in document.trajectory.frames
    }
    affected = []
    created = []
    warnings = []
    for time in times:
        solved_names = []
        for name, position in targets.items():
            solved = motion.solve_logical_frame_target(
                candidate,
                logical_frame=name,
                time_seconds=time,
                position_m=position,
                orientation_rpy_rad=(
                    None if orientations is None else orientations.get(name)
                ),
                mode=mode,
                protected_logical_frames=tuple(dict.fromkeys(
                    (*protected, *solved_names)
                )),
            )
            if not isinstance(solved, LogicalFrameSolveResult):
                raise TrajectoryOperationError(
                    "IK service returned an invalid End Effector result"
                )
            candidate.trajectory.upsert_frame(solved.frame)
            candidate.qpos_timeline.set_state(time, solved.qpos)
            frame_reference = working_metadata.reference_for_keyframe(solved.frame)
            affected.append(frame_reference)
            if (name, time) not in existing_logical:
                created.append(frame_reference)
            warnings.extend(solved.collisions)
            solved_names.append(name)
        qpos_reference = working_metadata.reference_for_qpos_keyframe(time)
        affected.append(qpos_reference)
        if not any(abs(time - value) <= 1e-9 for value in existing_qpos_times):
            created.append(qpos_reference)
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation=operation_name,
        ),
        affected_entities=tuple(dict.fromkeys(affected)),
        created_entities=tuple(dict.fromkeys(created)),
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError(f"{operation_name} made no motion change")
    return {
        "end_effectors": list(targets),
        "start_time": times[0],
        "end_time": times[-1],
        "solved_keyframes": len(times) * len(targets),
        "collision_warnings": list(dict.fromkeys(warnings)),
    }


def _interval_keyframe_times(document, start, end):
    if document.qpos_timeline is None:
        raise TrajectoryOperationError("operation requires an editable qpos timeline")
    if end > document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("operation interval exceeds the motion duration")
    return tuple(sorted({
        start,
        end,
        *(
            float(time)
            for time in document.qpos_timeline.times()
            if start - 1e-9 <= float(time) <= end + 1e-9
        ),
    }))


def _held_joint_names(scope, body_name, motion):
    if scope == "whole_body":
        return tuple(motion.joint_names)
    if scope == "joint":
        if body_name not in motion.joint_names:
            raise TrajectoryOperationError(f"unknown Joint Angle: {body_name}")
        return (body_name,)
    if scope == "joint_group":
        try:
            names = tuple(motion.joint_groups[body_name])
        except KeyError as error:
            raise TrajectoryOperationError(
                f"unknown Joint Angle group: {body_name}"
            ) from error
        if not names:
            raise TrajectoryOperationError(
                f"Joint Angle group is empty: {body_name}"
            )
        return names
    raise TrajectoryOperationError(f"unsupported hold_pose scope: {scope}")


def _protected_logical_frames(document, metadata):
    return tuple(sorted({
        frame.frame_name
        for frame in document.trajectory.frames
        if (
            metadata.metadata_for_keyframe(frame) is not None
            and metadata.metadata_for_keyframe(frame).protected
        )
    }))


def _synchronize_logical_frames(document, motion, time, qpos):
    state = motion.adapter.create_state()
    state.set_qpos(qpos)
    changed = []
    for frame in document.frames_at_time(time):
        kind, object_name = motion.adapter.logical_frame_bindings[frame.frame_name]
        position, quaternion = state.get_body_pose(object_name, kind)
        from core.trajectory import quat_to_rpy

        roll, pitch, yaw = quat_to_rpy(quaternion)
        updated = replace(
            frame,
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
        document.trajectory.upsert_frame(updated)
        changed.append(updated)
    return tuple(changed)

"""Local handlers for compact trajectory operations."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import numpy as np

from application.ai.metadata import MotionMetadataService
from application.ai.limits import MAX_GENERATED_MOTION_DURATION_SECONDS
from application.ai.motion_state import (
    ReplaceMotionState,
    capture_motion_state,
    detached_document,
)
from application.ai.motion_services import JointAngleEditResult, LogicalFrameSolveResult
from application.ai.motion_primitives import build_motion_primitive
from application.ai.qpos_trajectory import (
    normalize_qpos_quaternions,
    validate_qpos_anchors,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import TrajectoryExecutionContext
from application.ai.semantic_tools import SemanticToolContext, retime_segment
from application.motion_clipboard import capture_motion_clip, plan_repeat_motion
from application.timeline_editing import ApplyTimelineEditPlan, TimelineEditError
from core.trajectory import TargetFrame, Trajectory, quat_to_rpy


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
        TrajectoryOperationType.REPEAT_MOTION: (
            lambda operation, context: _repeat_motion(
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
        TrajectoryOperationType.SET_LOGICAL_FRAME_TARGET: (
            lambda operation, context: _set_logical_frame_target(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.LOCK_END_EFFECTOR: (
            lambda operation, context: _lock_end_effector(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.SPARSE_KEYFRAMES: (
            lambda operation, context: _generate_sparse_keyframes(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.QPOS_KEYFRAMES: (
            lambda operation, context: _apply_qpos_keyframes(
                operation, context, motion_service, metadata_service
            )
        ),
        TrajectoryOperationType.MOTION_PRIMITIVE: (
            lambda operation, context: _apply_motion_primitive(
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


def _repeat_motion(
    operation: TrajectoryOperation,
    context: TrajectoryExecutionContext,
    metadata: MotionMetadataService,
):
    """Append exact local copies of a committed interval to the working copy."""

    document = context.session.working_document
    arguments = operation.arguments
    start = float(arguments["start_time"])
    end = float(arguments["end_time"])
    copies = int(arguments["additional_copies"])
    ping_pong = bool(arguments["ping_pong"])
    if end > document.timeline_duration + 1e-9:
        raise TrajectoryOperationError("repeat_motion exceeds the motion duration")

    try:
        clip = capture_motion_clip(document, start, end)
        plan = plan_repeat_motion(
            document,
            clip,
            document.timeline_duration,
            copies,
            ping_pong=ping_pong,
            maximum_time=MAX_GENERATED_MOTION_DURATION_SECONDS,
        )
    except TimelineEditError as error:
        raise TrajectoryOperationError(str(error)) from error

    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    before = set(_motion_references(document, working_metadata))
    after = tuple(dict.fromkeys((
        *(
            working_metadata.reference_for_keyframe(frame)
            for frame in plan.frames
        ),
        *(
            working_metadata.reference_for_qpos_keyframe(time)
            for time, _qpos in plan.states
        ),
    )))
    created = tuple(reference for reference in after if reference not in before)
    result = context.session.apply_ai(
        ApplyTimelineEditPlan(plan),
        affected_entities=created,
        created_entities=created,
    )
    if not result.changed:
        raise TrajectoryOperationError("repeat_motion made no motion change")
    return {
        "start_time": start,
        "end_time": end,
        "additional_copies": copies,
        "ping_pong": ping_pong,
        "duration_seconds": plan.timeline_duration,
        "inserted_logical_keyframes": plan.inserted_frame_count,
        "inserted_qpos_keyframes": plan.inserted_state_count,
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


def _set_logical_frame_target(operation, context, motion, metadata):
    arguments = operation.arguments
    logical_frame = arguments["logical_frame"]
    if logical_frame not in motion.logical_frames:
        raise TrajectoryOperationError(f"unknown logical frame: {logical_frame}")
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
        targets={
            logical_frame: (
                None
                if arguments["position_m"] is None
                else tuple(arguments["position_m"])
            ),
        },
        orientations={
            logical_frame: (
                None
                if arguments["orientation_rpy_rad"] is None
                else tuple(arguments["orientation_rpy_rad"])
            ),
        },
        times=times,
        mode=mode,
        operation_name="set_logical_frame_target",
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
            if position is None:
                if mode == "delta":
                    position = (0.0, 0.0, 0.0)
                else:
                    qpos = candidate.qpos_timeline.sample_state(time)
                    state = motion.adapter.create_state()
                    state.set_qpos(qpos)
                    kind, object_name = motion.adapter.logical_frame_bindings[name]
                    position, _quaternion = state.get_body_pose(object_name, kind)
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
        synchronized = _synchronize_logical_frames(
            candidate,
            motion,
            time,
            candidate.qpos_timeline.get_state(time),
        )
        affected.extend(
            working_metadata.reference_for_keyframe(frame)
            for frame in synchronized
        )
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
            float(frame.time)
            for frame in document.trajectory.frames
            if start - 1e-9 <= float(frame.time) <= end + 1e-9
        ),
    }))


def _apply_qpos_keyframes(operation, context, motion, metadata):
    arguments = operation.arguments
    document = context.session.working_document
    timeline = document.qpos_timeline
    if timeline is None:
        raise TrajectoryOperationError(
            "qpos_keyframes requires an editable qpos timeline"
        )
    try:
        anchors = validate_qpos_anchors(motion.adapter, arguments["keyframes"])
    except (TypeError, ValueError) as error:
        raise TrajectoryOperationError(str(error)) from error

    mode = arguments["mode"]
    duration = float(arguments["duration_seconds"])
    start = float(arguments["start_time"])
    end = float(arguments["end_time"])
    candidate = detached_document(document)
    if mode == "replace":
        _replace_with_qpos_anchors(candidate, motion, anchors, duration)
        operation_name = "qpos_keyframes_replace"
        logical_times = tuple(anchor.time_seconds for anchor in anchors)
        candidate.trajectory = Trajectory()
        candidate.active_index = -1
    elif mode == "patch":
        if abs(duration - document.timeline_duration) > 1e-6:
            raise TrajectoryOperationError(
                "qpos patch duration_seconds must match the current motion duration"
            )
        if end > document.timeline_duration + 1e-9:
            raise TrajectoryOperationError(
                "qpos patch interval exceeds the current motion duration"
            )
        logical_times = _patch_with_qpos_anchors(
            candidate,
            document,
            motion,
            anchors,
            start=start,
            end=end,
        )
        operation_name = "qpos_keyframes_patch"
        candidate.active_index = -1
    else:  # The structured contract rejects this before execution.
        raise TrajectoryOperationError(f"unsupported qpos mode: {mode}")

    for time in logical_times:
        _capture_generation_frames(
            candidate,
            motion,
            time,
            candidate.qpos_timeline.sample_state(time),
            phase="ai_qpos",
        )

    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    if mode == "replace":
        existing = set(_motion_references(document, working_metadata))
        replacement = set(_motion_references(candidate, working_metadata))
    else:
        existing = set(_motion_references_in_interval(
            document, working_metadata, start, end
        ))
        replacement = set(_motion_references_in_interval(
            candidate, working_metadata, start, end
        ))
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation=operation_name,
        ),
        affected_entities=tuple(existing | replacement),
        created_entities=tuple(replacement - existing),
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError(f"{operation_name} made no motion change")
    return {
        "mode": mode,
        "duration_seconds": candidate.timeline_duration,
        "start_time": start,
        "end_time": end,
        "qpos_width": int(motion.adapter.mj_model.nq),
        "sparse_keyframes": len(anchors),
        "dense_qpos_samples": len(candidate.qpos_timeline.times()),
        "synchronized_logical_keyframes": (
            len(logical_times)
            * len(_trajectory_frame_names(motion.adapter))
        ),
    }


def _apply_motion_primitive(operation, context, motion, metadata):
    document = context.session.working_document
    if document.qpos_timeline is None:
        raise TrajectoryOperationError(
            "motion_primitive requires an editable qpos timeline"
        )
    arguments = operation.arguments
    try:
        plan = build_motion_primitive(
            motion.adapter,
            primitive=arguments["primitive"],
            duration_seconds=float(arguments["duration_seconds"]),
        )
    except (TypeError, ValueError) as error:
        raise TrajectoryOperationError(str(error)) from error

    candidate = detached_document(document)
    _replace_with_motion_primitive(
        candidate,
        motion,
        plan.anchors,
        float(arguments["duration_seconds"]),
    )
    candidate.trajectory = Trajectory()
    candidate.active_index = -1
    for anchor, phase in zip(plan.anchors, plan.phases):
        _capture_generation_frames(
            candidate,
            motion,
            anchor.time_seconds,
            candidate.qpos_timeline.sample_state(anchor.time_seconds),
            phase=f"ai_{phase}",
        )

    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    existing = set(_motion_references(document, working_metadata))
    replacement = set(_motion_references(candidate, working_metadata))
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation=f"motion_primitive_{plan.name}_{plan.variant}",
        ),
        affected_entities=tuple(existing | replacement),
        created_entities=tuple(replacement - existing),
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError("motion_primitive made no motion change")
    return {
        "primitive": plan.name,
        "variant": plan.variant,
        "duration_seconds": candidate.timeline_duration,
        "sparse_keyframes": len(plan.anchors),
        "dense_qpos_samples": len(candidate.qpos_timeline.times()),
        "quality_checks": list(plan.quality_checks),
        "concessions": list(plan.concessions),
    }


def _replace_with_motion_primitive(candidate, motion, anchors, duration):
    """Interpolate a primitive while preventing between-anchor floor cuts."""

    from application.ai.motion_primitives import clear_environment_collisions

    sparse_timeline = _empty_timeline_like(candidate.qpos_timeline)
    for anchor in anchors:
        sparse_timeline.set_state(anchor.time_seconds, anchor.qpos)
    dense_times = tuple(sorted({
        *_uniform_times(duration, 0.01),
        *(anchor.time_seconds for anchor in anchors),
    }))
    candidate.qpos_timeline.states = {}
    previous = None
    previous_velocity = None
    joint_addresses = np.asarray([
        int(motion.adapter.joints[name].qpos_address)
        for name in motion.adapter.joint_names
    ])
    for time in dense_times:
        sample_time = _smooth_primitive_sample_time(time, anchors)
        qpos = normalize_qpos_quaternions(
            motion.adapter,
            sparse_timeline.sample_state(sample_time),
            context=f"interpolated motion primitive at {time:.3f} s",
        )
        qpos = clear_environment_collisions(motion.adapter, qpos)
        if previous is not None:
            elapsed = time - previous[0]
            root = tuple(motion.adapter.free_joints_by_body.values())[0]
            address = int(root.qpos_address)
            root_speed = float(np.linalg.norm(
                qpos[address:address + 3] - previous[1][address:address + 3]
            )) / elapsed
            if root_speed > 3.0:
                raise TrajectoryOperationError(
                    "motion primitive exceeds the root continuity speed limit"
                )
            velocity = (
                qpos[joint_addresses]
                - previous[1][joint_addresses]
            ) / elapsed
            if previous_velocity is not None:
                acceleration = (velocity - previous_velocity) / elapsed
                if float(np.max(np.abs(acceleration))) > 100.0:
                    raise TrajectoryOperationError(
                        "motion primitive exceeds the Joint Angle acceleration limit"
                    )
            previous_velocity = velocity
        candidate.qpos_timeline.set_state(time, qpos)
        previous = (time, qpos)
    candidate.set_timeline_duration(duration)
    candidate.current_time = min(candidate.current_time, duration)


def _smooth_primitive_sample_time(time, anchors):
    for start, end in zip(anchors, anchors[1:]):
        if time <= end.time_seconds + 1e-12:
            elapsed = end.time_seconds - start.time_seconds
            fraction = min(1.0, max(0.0, (time - start.time_seconds) / elapsed))
            eased = fraction * fraction * (3.0 - 2.0 * fraction)
            return start.time_seconds + eased * elapsed
    return anchors[-1].time_seconds


def _replace_with_qpos_anchors(candidate, motion, anchors, duration):
    sparse_timeline = _empty_timeline_like(candidate.qpos_timeline)
    for anchor in anchors:
        sparse_timeline.set_state(anchor.time_seconds, anchor.qpos)
    dense_times = tuple(sorted({
        *_uniform_times(duration, 0.01),
        *(anchor.time_seconds for anchor in anchors),
    }))
    candidate.qpos_timeline.states = {}
    for time in dense_times:
        candidate.qpos_timeline.set_state(
            time,
            normalize_qpos_quaternions(
                motion.adapter,
                sparse_timeline.sample_state(time),
                context=f"interpolated qpos at {time:.3f} s",
            ),
        )
    candidate.set_timeline_duration(duration)
    candidate.current_time = min(candidate.current_time, duration)


def _patch_with_qpos_anchors(
    candidate,
    source_document,
    motion,
    anchors,
    *,
    start,
    end,
):
    source_timeline = source_document.qpos_timeline
    sparse_timeline = _empty_timeline_like(source_timeline)
    sparse_timeline.set_state(start, source_timeline.sample_state(start))
    for anchor in anchors:
        sparse_timeline.set_state(anchor.time_seconds, anchor.qpos)
    sparse_timeline.set_state(end, source_timeline.sample_state(end))

    retained_states = tuple(
        (float(time), source_timeline.get_state(time))
        for time in source_timeline.times()
        if float(time) < start - 1e-9 or float(time) > end + 1e-9
    )
    patch_times = tuple(sorted({
        *_uniform_interval_times(start, end, 0.01),
        *(anchor.time_seconds for anchor in anchors),
    }))
    candidate.qpos_timeline.states = {}
    for time, qpos in retained_states:
        candidate.qpos_timeline.set_state(time, qpos)
    for time in patch_times:
        candidate.qpos_timeline.set_state(
            time,
            normalize_qpos_quaternions(
                motion.adapter,
                sparse_timeline.sample_state(time),
                context=f"interpolated qpos at {time:.3f} s",
            ),
        )

    for track in candidate.trajectory.tracks.values():
        track[:] = [
            frame
            for frame in track
            if float(frame.time) < start - 1e-9 or float(frame.time) > end + 1e-9
        ]
    return tuple(sorted({
        start,
        end,
        *(anchor.time_seconds for anchor in anchors),
    }))


def _empty_timeline_like(timeline):
    clone = copy(timeline)
    clone.states = {}
    return clone


def _generate_sparse_keyframes(operation, context, motion, metadata):
    arguments = operation.arguments
    document = context.session.working_document
    timeline = document.qpos_timeline
    if timeline is None:
        raise TrajectoryOperationError(
            "sparse_keyframes requires an editable qpos timeline"
        )
    initial_qpos = timeline.sample_state(document.current_time)
    if initial_qpos is None:
        raise TrajectoryOperationError("current robot pose is unavailable")

    candidate = detached_document(document)
    candidate.trajectory = Trajectory()
    candidate.qpos_timeline.states = {}
    candidate.set_timeline_duration(float(arguments["duration_seconds"]))
    candidate.current_time = min(candidate.current_time, candidate.timeline_duration)
    previous_qpos = np.asarray(initial_qpos, dtype=float).copy()
    sparse_times = []
    warnings = []
    for keyframe in arguments["keyframes"]:
        time = float(keyframe["time_seconds"])
        candidate.qpos_timeline.set_state(time, previous_qpos)
        qpos = _apply_sparse_root(candidate, motion, time, keyframe)
        candidate.qpos_timeline.set_state(time, qpos)

        joint_values = {
            target["joint"]: float(target["angle_rad"])
            for target in keyframe["joint_targets"]
        }
        if joint_values:
            solved_joints = motion.set_joint_angles(
                candidate,
                time_seconds=time,
                values=joint_values,
                protected_logical_frames=(),
            )
            if not isinstance(solved_joints, JointAngleEditResult):
                raise TrajectoryOperationError(
                    "Joint Angle service returned an invalid generation result"
                )
            candidate.qpos_timeline.set_state(time, solved_joints.qpos)

        torso_orientation = keyframe["torso_rpy_rad"]
        if torso_orientation is not None:
            _solve_sparse_logical_target(
                candidate,
                motion,
                time=time,
                name="torso",
                position=None,
                orientation=tuple(torso_orientation),
                protected=(),
                warnings=warnings,
            )

        solved_names = []
        for target in keyframe["end_effector_targets"]:
            name = target["end_effector"]
            if name not in motion.end_effectors:
                raise TrajectoryOperationError(f"unknown End Effector: {name}")
            _solve_sparse_logical_target(
                candidate,
                motion,
                time=time,
                name=name,
                position=tuple(target["position_m"]),
                orientation=None,
                protected=tuple(solved_names),
                warnings=warnings,
            )
            solved_names.append(name)

        previous_qpos = candidate.qpos_timeline.get_state(time)
        _capture_generation_frames(candidate, motion, time, previous_qpos)
        sparse_times.append(time)

    sparse_timeline = detached_document(candidate).qpos_timeline
    candidate.qpos_timeline.states = {}
    dense_times = _uniform_times(candidate.timeline_duration, 0.01)
    for time in dense_times:
        candidate.qpos_timeline.set_state(time, sparse_timeline.sample_state(time))

    working_metadata = MotionMetadataService(
        context.session.metadata,
        metadata.resolver,
    )
    existing = set(_motion_references(document, working_metadata))
    replacement = set(_motion_references(candidate, working_metadata))
    affected = tuple(existing | replacement)
    created = tuple(replacement - existing)
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation="sparse_keyframes",
        ),
        affected_entities=affected,
        created_entities=created,
        allow_user_override=True,
    )
    if not result.changed:
        raise TrajectoryOperationError("sparse_keyframes made no motion change")
    return {
        "duration_seconds": candidate.timeline_duration,
        "sparse_keyframes": len(sparse_times),
        "dense_qpos_samples": len(dense_times),
        "collision_warnings": list(dict.fromkeys(warnings)),
    }


def _apply_sparse_root(document, motion, time, keyframe):
    qpos = np.asarray(document.qpos_timeline.get_state(time), dtype=float).copy()
    position = keyframe["root_position_m"]
    if position is None:
        return qpos
    free_joints = tuple(motion.adapter.free_joints_by_body.values())
    if not free_joints:
        raise TrajectoryOperationError("robot model has no floating root target")
    address = int(free_joints[0].qpos_address)
    if address < 0 or address + 7 > len(qpos):
        raise TrajectoryOperationError("floating-root qpos layout is invalid")
    qpos[address:address + 3] = np.asarray(position, dtype=float)
    return qpos


def _solve_sparse_logical_target(
    document,
    motion,
    *,
    time,
    name,
    position,
    orientation,
    protected,
    warnings,
):
    if name not in motion.logical_frames:
        raise TrajectoryOperationError(f"unknown logical frame: {name}")
    if position is None:
        qpos = document.qpos_timeline.get_state(time)
        state = motion.adapter.create_state()
        state.set_qpos(qpos)
        kind, object_name = motion.adapter.logical_frame_bindings[name]
        position, _quaternion = state.get_body_pose(object_name, kind)
    solved = motion.solve_logical_frame_target(
        document,
        logical_frame=name,
        time_seconds=time,
        position_m=tuple(float(value) for value in position),
        orientation_rpy_rad=orientation,
        mode="absolute",
        protected_logical_frames=protected,
    )
    if not isinstance(solved, LogicalFrameSolveResult):
        raise TrajectoryOperationError("IK service returned an invalid generation result")
    document.trajectory.upsert_frame(solved.frame)
    document.qpos_timeline.set_state(time, solved.qpos)
    warnings.extend(solved.collisions)


def _capture_generation_frames(
    document,
    motion,
    time,
    qpos,
    *,
    phase="ai_generate",
):
    state = motion.adapter.create_state()
    state.set_qpos(qpos)
    for name in _trajectory_frame_names(motion.adapter):
        kind, object_name = motion.adapter.logical_frame_bindings[name]
        position, quaternion = state.get_body_pose(object_name, kind)
        roll, pitch, yaw = quat_to_rpy(quaternion)
        document.trajectory.upsert_frame(TargetFrame(
            time=time,
            phase=phase,
            frame_name=name,
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        ))


def _uniform_times(duration, dt):
    count = int(np.floor(float(duration) / dt + 1e-9))
    times = [round(index * dt, 9) for index in range(count + 1)]
    if not times or abs(times[-1] - duration) > 1e-9:
        times.append(float(duration))
    else:
        times[-1] = float(duration)
    return tuple(times)


def _uniform_interval_times(start, end, dt):
    duration = float(end) - float(start)
    return tuple(
        round(float(start) + offset, 9)
        for offset in _uniform_times(duration, dt)
    )


def _trajectory_frame_names(adapter):
    names = tuple(getattr(adapter, "trajectory_frames", ()))
    return names or tuple(adapter.logical_frame_bindings)


def _motion_references(document, metadata):
    references = [
        metadata.reference_for_keyframe(frame)
        for frame in document.trajectory.frames
    ]
    if document.qpos_timeline is not None:
        references.extend(
            metadata.reference_for_qpos_keyframe(time)
            for time in document.qpos_timeline.times()
        )
    return tuple(dict.fromkeys(references))


def _motion_references_in_interval(document, metadata, start, end):
    references = [
        metadata.reference_for_keyframe(frame)
        for frame in document.trajectory.frames
        if start - 1e-9 <= float(frame.time) <= end + 1e-9
    ]
    if document.qpos_timeline is not None:
        references.extend(
            metadata.reference_for_qpos_keyframe(time)
            for time in document.qpos_timeline.times()
            if start - 1e-9 <= float(time) <= end + 1e-9
        )
    return tuple(dict.fromkeys(references))


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

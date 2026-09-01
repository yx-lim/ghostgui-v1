"""Strict semantic tools that operate only on an AI working copy."""

from __future__ import annotations

from dataclasses import dataclass

from application.ai.context import (
    ContextBuilder,
    EditorSelectionContext,
    RobotCapabilityContext,
)
from application.ai.edit_session import AIEditSession
from application.ai.metadata import MotionMetadataService
from application.ai.motion_services import SemanticMotionError, SemanticMotionService
from application.ai.schemas import EditAuthor
from application.ai.motion_state import (
    ReplaceMotionState,
    capture_motion_state,
    detached_document,
)
from application.ai.tool_registry import ToolCategory, ToolRegistry, ToolSpec
from application.timeline_editing import ApplyTimelineEditPlan, plan_scale_time_range


@dataclass(frozen=True)
class SemanticToolContext:
    session: AIEditSession
    metadata: MotionMetadataService
    selection: EditorSelectionContext = EditorSelectionContext()
    motion_name: str | None = None
    validation_state: str | None = None

    @property
    def working_metadata(self) -> MotionMetadataService:
        return MotionMetadataService(self.session.metadata, self.metadata.resolver)


def build_semantic_tool_registry(
    motion: SemanticMotionService,
    *,
    context_builder: ContextBuilder | None = None,
) -> ToolRegistry:
    """Build the allowlist supported by the active robot model."""

    context_builder = context_builder or ContextBuilder()
    registry = ToolRegistry()
    registry.register(_spec(
        "inspect_motion",
        "Inspect compact motion, selection, protection, and capability context.",
        _closed_object({}),
        ToolCategory.INSPECT,
        False,
        lambda arguments, context: _inspect(
            _context(context), motion, context_builder
        ),
    ))
    registry.register(_spec(
        "ensure_keyframe",
        "Ensure an editable qpos Keyframe exists at a motion time.",
        _closed_object(
            {"time_seconds": _number(0.0, 3600.0)},
            required=("time_seconds",),
        ),
        ToolCategory.EDIT,
        True,
        lambda arguments, context: _ensure_keyframe(
            _context(context), motion, arguments
        ),
    ))
    if motion.logical_frames:
        registry.register(_spec(
            "set_logical_frame_target",
            "Set an absolute or relative body/logical-frame target using GhostGUI IK.",
            _logical_target_schema(motion.logical_frames),
            ToolCategory.EDIT,
            True,
            lambda arguments, context: _set_logical_frame_target(
                _context(context), motion, arguments
            ),
        ))
        registry.register(_spec(
            "protect_keyframe",
            "Protect or unprotect an existing logical-frame Keyframe.",
            _closed_object(
                {
                    "logical_frame": _enum(motion.logical_frames),
                    "time_seconds": _number(0.0, 3600.0),
                    "protected": {"type": "boolean"},
                },
                required=("logical_frame", "time_seconds", "protected"),
            ),
            ToolCategory.EDIT,
            True,
            lambda arguments, context: _protect_keyframe(
                _context(context), arguments
            ),
        ))
    if motion.end_effectors:
        registry.register(_spec(
            "move_end_effector",
            "Move an End Effector by a Cartesian and optional orientation delta using IK.",
            _closed_object(
                {
                    "end_effector": _enum(motion.end_effectors),
                    "time_seconds": _number(0.0, 3600.0),
                    "delta_m": _vector_schema(-1.0, 1.0),
                    "delta_rpy_rad": _vector_schema(-3.141593, 3.141593),
                },
                required=("end_effector", "time_seconds", "delta_m"),
            ),
            ToolCategory.EDIT,
            True,
            lambda arguments, context: _move_end_effector(
                _context(context), motion, arguments
            ),
        ))
    if motion.joint_names:
        registry.register(_spec(
            "set_joint_angle",
            "Set one explicitly named Joint Angle at a Keyframe.",
            _closed_object(
                {
                    "joint": _enum(motion.joint_names),
                    "time_seconds": _number(0.0, 3600.0),
                    "angle_rad": _number(-12.566371, 12.566371),
                },
                required=("joint", "time_seconds", "angle_rad"),
            ),
            ToolCategory.EDIT,
            True,
            lambda arguments, context: _set_joint_values(
                _context(context),
                motion,
                arguments["time_seconds"],
                {arguments["joint"]: arguments["angle_rad"]},
                "set_joint_angle",
            ),
        ))
    if motion.joint_groups:
        registry.register(_spec(
            "set_joint_group_angles",
            "Set ordered Joint Angles for an explicitly named joint group.",
            _closed_object(
                {
                    "joint_group": _enum(tuple(motion.joint_groups)),
                    "time_seconds": _number(0.0, 3600.0),
                    "angles_rad": {
                        "type": "array",
                        "items": _number(-12.566371, 12.566371),
                        "minItems": 1,
                        "maxItems": max(len(group) for group in motion.joint_groups.values()),
                    },
                },
                required=("joint_group", "time_seconds", "angles_rad"),
            ),
            ToolCategory.EDIT,
            True,
            lambda arguments, context: _set_joint_group(
                _context(context), motion, arguments
            ),
        ))
    registry.register(_spec(
        "retime_segment",
        "Scale a Keyframe interval by motion speed; values above 1 are faster.",
        _closed_object(
            {
                "start_time_seconds": _number(0.0, 3600.0),
                "end_time_seconds": _number(0.0, 3600.0),
                "speed": _number(0.01, 100.0),
            },
            required=("start_time_seconds", "end_time_seconds", "speed"),
        ),
        ToolCategory.EDIT,
        True,
        lambda arguments, context: _retime_segment(_context(context), arguments),
    ))
    registry.register(_spec(
        "validate_motion",
        "Validate the staged working motion without changing it.",
        _closed_object({}),
        ToolCategory.TEST,
        False,
        lambda arguments, context: _validate(_context(context), motion),
    ))
    return registry


def _inspect(context, motion, builder):
    capabilities = RobotCapabilityContext(
        logical_frames=tuple(motion.logical_frames),
        end_effectors=tuple(motion.end_effectors),
        joints=tuple(motion.joint_names),
        joint_groups=tuple(
            (name, tuple(values))
            for name, values in motion.joint_groups.items()
        ),
    )
    return builder.build_for_session(
        context.session,
        selection=context.selection,
        robot_capabilities=capabilities,
        metadata=context.metadata,
        motion_name=context.motion_name,
        validation_state=context.validation_state,
    ).to_dict()


def _ensure_keyframe(context, motion, arguments):
    time_seconds = float(arguments["time_seconds"])
    qpos = motion.ensure_qpos_keyframe(
        context.session.working_document,
        time_seconds=time_seconds,
    )
    candidate = detached_document(context.session.working_document)
    candidate.qpos_timeline.set_state(time_seconds, qpos)
    reference = context.working_metadata.reference_for_qpos_keyframe(time_seconds)
    result = context.session.apply_ai(
        ReplaceMotionState(
            capture_motion_state(candidate),
            operation="ensure_keyframe",
        ),
        affected_entities=(reference,),
    )
    return {"changed": result.changed, "time_seconds": time_seconds}


def _set_logical_frame_target(context, motion, arguments):
    return _solve_and_apply(
        context,
        motion,
        logical_frame=arguments["logical_frame"],
        time_seconds=float(arguments["time_seconds"]),
        position=tuple(arguments["position_m"]),
        orientation=(
            None
            if "orientation_rpy_rad" not in arguments
            else tuple(arguments["orientation_rpy_rad"])
        ),
        mode=arguments["mode"],
        operation="set_logical_frame_target",
    )


def _move_end_effector(context, motion, arguments):
    return _solve_and_apply(
        context,
        motion,
        logical_frame=arguments["end_effector"],
        time_seconds=float(arguments["time_seconds"]),
        position=tuple(arguments["delta_m"]),
        orientation=(
            None
            if "delta_rpy_rad" not in arguments
            else tuple(arguments["delta_rpy_rad"])
        ),
        mode="delta",
        operation="move_end_effector",
    )


def _solve_and_apply(
    context,
    motion,
    *,
    logical_frame,
    time_seconds,
    position,
    orientation,
    mode,
    operation,
):
    metadata = context.working_metadata
    protected = _protected_logical_frames(
        context.session.working_document,
        metadata,
    )
    solved = motion.solve_logical_frame_target(
        context.session.working_document,
        logical_frame=logical_frame,
        time_seconds=time_seconds,
        position_m=position,
        orientation_rpy_rad=orientation,
        mode=mode,
        protected_logical_frames=protected,
    )
    candidate = detached_document(context.session.working_document)
    candidate.trajectory.upsert_frame(solved.frame)
    if candidate.qpos_timeline is None:
        raise SemanticMotionError("motion has no editable qpos timeline")
    candidate.qpos_timeline.set_state(time_seconds, solved.qpos)
    references = (
        metadata.reference_for_keyframe(solved.frame),
        metadata.reference_for_qpos_keyframe(time_seconds),
    )
    context.session.apply_ai(
        ReplaceMotionState(capture_motion_state(candidate), operation=operation),
        affected_entities=references,
    )
    return {
        "logical_frame": solved.frame.frame_name,
        "time_seconds": solved.frame.time,
        "achieved_position_m": [solved.frame.x, solved.frame.y, solved.frame.z],
        "achieved_orientation_rpy_rad": [
            solved.frame.roll,
            solved.frame.pitch,
            solved.frame.yaw,
        ],
        "status": solved.status,
        "position_error": solved.position_error,
        "collision_warnings": list(solved.collisions),
    }


def _set_joint_group(context, motion, arguments):
    group_name = arguments["joint_group"]
    names = tuple(motion.joint_groups[group_name])
    angles = tuple(arguments["angles_rad"])
    if len(names) != len(angles):
        raise SemanticMotionError(
            f"joint group {group_name} requires {len(names)} Joint Angles"
        )
    return _set_joint_values(
        context,
        motion,
        arguments["time_seconds"],
        dict(zip(names, angles)),
        "set_joint_group_angles",
    )


def _set_joint_values(context, motion, time_seconds, values, operation):
    time_seconds = float(time_seconds)
    qpos = motion.set_joint_angles(
        context.session.working_document,
        time_seconds=time_seconds,
        values=values,
        protected_logical_frames=_protected_logical_frames(
            context.session.working_document,
            context.working_metadata,
        ),
    )
    candidate = detached_document(context.session.working_document)
    candidate.qpos_timeline.set_state(time_seconds, qpos)
    reference = context.working_metadata.reference_for_qpos_keyframe(time_seconds)
    context.session.apply_ai(
        ReplaceMotionState(capture_motion_state(candidate), operation=operation),
        affected_entities=(reference,),
    )
    return {
        "time_seconds": time_seconds,
        "joint_angles_rad": {name: float(value) for name, value in values.items()},
    }


def _protect_keyframe(context, arguments):
    document = context.session.working_document
    matches = tuple(
        frame
        for frame in document.trajectory.frames
        if frame.frame_name == arguments["logical_frame"]
        and abs(frame.time - float(arguments["time_seconds"])) <= 1e-6
    )
    if not matches:
        raise SemanticMotionError("cannot protect a nonexistent logical Keyframe")
    reference = context.working_metadata.reference_for_keyframe(matches[0])
    changed = context.session.protect(
        reference,
        bool(arguments["protected"]),
        author=EditAuthor.AI,
    )
    return {
        "logical_frame": matches[0].frame_name,
        "time_seconds": matches[0].time,
        "protected": bool(arguments["protected"]),
        "changed": changed,
    }


def _retime_segment(context, arguments):
    document = context.session.working_document
    start = float(arguments["start_time_seconds"])
    end = float(arguments["end_time_seconds"])
    plan = plan_scale_time_range(document, start, end, float(arguments["speed"]))
    metadata = context.working_metadata
    before_frames = tuple(document.trajectory.frames)
    before_state_times = tuple(document.qpos_timeline.times()) if document.qpos_timeline else ()
    affected = tuple(
        metadata.reference_for_keyframe(frame)
        for frame in before_frames
        if start - 1e-6 <= frame.time <= end + 1e-6
    ) + tuple(
        metadata.reference_for_qpos_keyframe(time)
        for time in before_state_times
        if start - 1e-6 <= time <= end + 1e-6
    )
    context.session.apply_ai(
        ApplyTimelineEditPlan(plan),
        affected_entities=affected,
    )
    _remap_timeline_metadata(
        metadata,
        before_frames,
        plan.frames,
        before_state_times,
        tuple(time for time, _qpos in plan.states),
    )
    return {
        "start_time_seconds": start,
        "end_time_seconds": end,
        "speed": float(arguments["speed"]),
        "new_duration_seconds": plan.timeline_duration,
        "affected_keyframes": plan.affected_count,
    }


def _validate(context, motion):
    report = motion.validate_motion(context.session.working_document)
    return {"valid": report.valid, "issues": list(report.issues)}


def _protected_logical_frames(document, metadata):
    protected = set()
    for frame in document.trajectory.frames:
        value = metadata.metadata_for_keyframe(frame)
        if value is not None and value.protected:
            protected.add(frame.frame_name)
    return tuple(sorted(protected))


def _remap_timeline_metadata(
    metadata,
    before_frames,
    after_frames,
    before_state_times,
    after_state_times,
):
    before_tracks = _tracks(before_frames)
    after_tracks = _tracks(after_frames)
    for name, frames in before_tracks.items():
        for before, after in zip(frames, after_tracks.get(name, ())):
            metadata.remap_keyframe(before, after)
    for before, after in zip(before_state_times, after_state_times):
        metadata.remap_qpos_keyframe(before, after)


def _tracks(frames):
    result = {}
    for frame in frames:
        result.setdefault(frame.frame_name, []).append(frame)
    for values in result.values():
        values.sort(key=lambda frame: frame.time)
    return result


def _context(value):
    if not isinstance(value, SemanticToolContext):
        raise TypeError("semantic tools require SemanticToolContext")
    return value


def _spec(name, description, schema, category, mutates, handler):
    return ToolSpec(
        name=name,
        description=description,
        input_schema=schema,
        handler=handler,
        category=category,
        mutates_working_copy=mutates,
    )


def _logical_target_schema(logical_frames):
    return _closed_object(
        {
            "logical_frame": _enum(logical_frames),
            "time_seconds": _number(0.0, 3600.0),
            "position_m": _vector_schema(-5.0, 5.0),
            "orientation_rpy_rad": _vector_schema(-12.566371, 12.566371),
            "mode": _enum(("absolute", "delta")),
        },
        required=("logical_frame", "time_seconds", "position_m", "mode"),
    )


def _closed_object(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _number(minimum, maximum):
    return {"type": "number", "minimum": minimum, "maximum": maximum}


def _enum(values):
    return {"type": "string", "enum": list(values)}


def _vector_schema(minimum, maximum):
    return {
        "type": "array",
        "items": _number(minimum, maximum),
        "minItems": 3,
        "maxItems": 3,
    }

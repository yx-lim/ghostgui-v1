"""Qt-free capture and paste planning for editable Motion Clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from core.trajectory import TargetFrame, quat_to_rpy, rpy_to_quat

from .project_document import ProjectDocument
from .timeline_editing import (
    TIME_PRECISION,
    TIME_TOLERANCE,
    TimelineEditError,
    TimelineEditPlan,
)


@dataclass(frozen=True)
class MotionClip:
    """Detached, model-specific Keyframes stored on a relative timeline."""

    model_key: str
    duration: float
    frames: tuple[TargetFrame, ...]
    states: tuple[tuple[float, np.ndarray], ...]
    qpos_width: int | None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def state_count(self) -> int:
        return len(self.states)


def capture_motion_clip(
    document: ProjectDocument,
    start_time: float,
    end_time: float,
) -> MotionClip:
    """Capture an inclusive committed range and normalize it to start at zero."""
    start_time = _nonnegative_time(start_time, "Range start")
    end_time = _nonnegative_time(end_time, "Range end")
    if end_time <= start_time + TIME_TOLERANCE:
        raise TimelineEditError("Range end must be later than range start.")

    original_frames = tuple(document.trajectory.frames)
    original_states = _timeline_states(document.qpos_timeline)
    content_times = [frame.time for frame in original_frames]
    content_times.extend(time for time, _qpos in original_states)
    if not content_times:
        raise TimelineEditError("There are no committed Keyframes to copy.")
    content_start = min(content_times)
    content_end = max(content_times)
    if (
        start_time < content_start - TIME_TOLERANCE
        or end_time > content_end + TIME_TOLERANCE
    ):
        raise TimelineEditError(
            "The copied range must stay within the committed motion bounds "
            f"({content_start:.2f}–{content_end:.2f} s)."
        )

    duration = _time_key(end_time - start_time)
    if duration <= TIME_TOLERANCE:
        raise TimelineEditError(
            "The copied range is below the timeline's time precision."
        )

    state_map: dict[float, np.ndarray] = {}
    boundary_states: dict[float, np.ndarray] = {}
    qpos_width = None
    for time, qpos in original_states:
        if _in_range(time, start_time, end_time):
            relative_time = _relative_time(time, start_time, duration)
            qpos_width = _capture_state(
                state_map,
                relative_time,
                qpos,
                expected_width=qpos_width,
            )

    if original_states:
        for boundary_time, relative_time in (
            (start_time, 0.0),
            (end_time, duration),
        ):
            sampled = document.qpos_timeline.sample_state(boundary_time)
            boundary_states[boundary_time] = np.asarray(
                sampled, dtype=float
            ).copy()
            qpos_width = _capture_state(
                state_map,
                relative_time,
                sampled,
                expected_width=qpos_width,
            )

    frame_map: dict[tuple[str, float], TargetFrame] = {}
    for frame in original_frames:
        if _in_range(frame.time, start_time, end_time):
            relative_time = _relative_time(frame.time, start_time, duration)
            _capture_frame(
                frame_map,
                replace(frame, time=relative_time),
            )

    # Materialized endpoints make an arbitrary interpolated subrange portable.
    # Exact stored targets win because interpolation intentionally carries the
    # previous Keyframe's phase at a segment boundary.
    for boundary_time, relative_time in (
        (start_time, 0.0),
        (end_time, duration),
    ):
        exact_names = {
            frame.frame_name
            for frame in original_frames
            if abs(frame.time - boundary_time) <= TIME_TOLERANCE
        }
        targets = _boundary_targets(
            document,
            boundary_time,
            boundary_states.get(boundary_time),
        )
        for frame in targets.values():
            if frame.frame_name in exact_names:
                continue
            _capture_frame(
                frame_map,
                replace(frame, time=relative_time),
            )

    if not frame_map and not state_map:
        raise TimelineEditError(
            f"No committed Keyframes exist from {start_time:.2f} s to "
            f"{end_time:.2f} s."
        )

    frames = tuple(
        replace(frame)
        for frame in sorted(
            frame_map.values(),
            key=lambda item: (item.time, item.frame_name),
        )
    )
    states = tuple(
        (time, state_map[time].copy()) for time in sorted(state_map)
    )
    return MotionClip(
        model_key=str(document.model_key),
        duration=duration,
        frames=frames,
        states=states,
        qpos_width=qpos_width,
    )


def plan_paste_motion(
    document: ProjectDocument,
    clip: MotionClip,
    destination_start: float,
    *,
    reverse: bool = False,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Preflight one forward or time-reversed Motion Clip paste."""
    destination_start = _nonnegative_time(
        destination_start, "Destination start"
    )
    operation = "paste_motion_reversed" if reverse else "paste_motion"
    return _plan_placements(
        document,
        clip,
        ((destination_start, bool(reverse)),),
        operation=operation,
        final_time=_time_key(destination_start + _clip_duration(clip)),
        maximum_time=maximum_time,
    )


def plan_repeat_motion(
    document: ProjectDocument,
    clip: MotionClip,
    destination_start: float,
    additional_copies: int,
    *,
    ping_pong: bool = False,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Preflight additional forward or alternating reversed/forward copies."""
    destination_start = _nonnegative_time(
        destination_start, "Destination start"
    )
    additional_copies = _positive_integer(
        additional_copies, "Additional copies"
    )
    duration = _clip_duration(clip)
    placements = tuple(
        (
            _time_key(destination_start + index * duration),
            bool(ping_pong and index % 2 == 0),
        )
        for index in range(additional_copies)
    )
    return _plan_placements(
        document,
        clip,
        placements,
        operation=(
            "repeat_motion_ping_pong"
            if ping_pong
            else "repeat_motion_forward"
        ),
        final_time=_time_key(
            destination_start + additional_copies * duration
        ),
        maximum_time=maximum_time,
    )


def _plan_placements(
    document,
    clip,
    placements,
    *,
    operation,
    final_time,
    maximum_time,
):
    duration, clip_qpos_width = _validate_clip(document, clip)
    _check_placement_overlaps(document, clip, placements, duration)

    frame_map: dict[tuple[str, float], TargetFrame] = {}
    for frame in document.trajectory.frames:
        key = (frame.frame_name, _time_key(frame.time))
        if key in frame_map:
            raise TimelineEditError(
                "The document contains duplicate "
                f"{frame.frame_name} targets at t={key[1]:.2f} s."
            )
        frame_map[key] = replace(frame)

    state_map: dict[float, np.ndarray] = {}
    robot_model = getattr(document.qpos_timeline, "robot_model", None)
    for time, qpos in _timeline_states(document.qpos_timeline):
        key = _time_key(time)
        if key in state_map:
            raise TimelineEditError(
                f"The document contains duplicate robot poses at t={key:.2f} s."
            )
        state_map[key] = np.asarray(qpos, dtype=float).copy()

    inserted_frames = 0
    inserted_states = 0
    for destination_start, reverse in placements:
        destination_start = _nonnegative_time(
            destination_start, "Destination start"
        )
        for source in clip.frames:
            relative_time = _clip_item_time(source.time, duration)
            mapped_time = _mapped_time(
                destination_start,
                relative_time,
                duration,
                reverse,
            )
            candidate = replace(source, time=mapped_time)
            key = (candidate.frame_name, mapped_time)
            existing = frame_map.get(key)
            if existing is None:
                frame_map[key] = candidate
                inserted_frames += 1
            elif not _frames_equivalent(existing, candidate):
                raise TimelineEditError(
                    f"Motion paste conflicts with an existing "
                    f"{candidate.frame_name} target at t={mapped_time:.2f} s."
                )

        for source_time, source_qpos in clip.states:
            relative_time = _clip_item_time(source_time, duration)
            mapped_time = _mapped_time(
                destination_start,
                relative_time,
                duration,
                reverse,
            )
            candidate = _validated_qpos(
                source_qpos,
                expected_width=clip_qpos_width,
            )
            existing = state_map.get(mapped_time)
            if existing is None:
                state_map[mapped_time] = candidate
                inserted_states += 1
            elif not _states_equivalent(
                existing,
                candidate,
                robot_model=robot_model,
            ):
                raise TimelineEditError(
                    "Motion paste conflicts with an existing robot pose at "
                    f"t={mapped_time:.2f} s."
                )

    if inserted_frames == 0 and inserted_states == 0:
        raise TimelineEditError(
            "The pasted motion is already present at the destination."
        )

    latest_time = max(
        [time for _frame_name, time in frame_map] + list(state_map),
        default=0.0,
    )
    new_duration = max(
        0.1,
        float(document.timeline_duration),
        latest_time,
        float(final_time),
    )
    _check_maximum_time(new_duration, maximum_time)

    frames = tuple(
        sorted(
            frame_map.values(),
            key=lambda frame: (frame.time, frame.frame_name),
        )
    )
    states = tuple(
        (time, state_map[time].copy()) for time in sorted(state_map)
    )
    return TimelineEditPlan(
        operation=operation,
        frames=frames,
        states=states,
        timeline_duration=new_duration,
        current_time=float(final_time),
        moved_frame_count=0,
        moved_state_count=0,
        inserted_frame_count=inserted_frames,
        inserted_state_count=inserted_states,
    )


def _validate_clip(document, clip):
    if not isinstance(clip, MotionClip):
        raise TimelineEditError("Motion clipboard data is invalid.")
    if str(document.model_key) != str(clip.model_key):
        raise TimelineEditError(
            "The copied motion belongs to a different robot model."
        )
    duration = _clip_duration(clip)
    if not clip.frames and not clip.states:
        raise TimelineEditError("The copied motion is empty.")
    if clip.states and document.qpos_timeline is None:
        raise TimelineEditError(
            "The active document has no robot-pose timeline for the copied "
            "Joint Angles."
        )

    declared_width = _declared_qpos_width(clip.qpos_width)
    state_width = None
    for frame in clip.frames:
        if not isinstance(frame, TargetFrame):
            raise TimelineEditError(
                "The copied motion contains an invalid target Keyframe."
            )
        _clip_item_time(frame.time, duration)
    for time, qpos in clip.states:
        _clip_item_time(time, duration)
        validated = _validated_qpos(qpos, expected_width=state_width)
        state_width = int(validated.size)
    if (
        declared_width is not None
        and state_width is not None
        and declared_width != state_width
    ):
        raise TimelineEditError(
            "The copied motion contains inconsistent Joint Angle widths."
        )
    clip_qpos_width = declared_width if declared_width is not None else state_width
    expected_width = _document_qpos_width(document)
    if (
        clip_qpos_width is not None
        and expected_width is not None
        and clip_qpos_width != expected_width
    ):
        raise TimelineEditError(
            "The copied Joint Angles do not match the active robot model."
        )
    return duration, clip_qpos_width


def _declared_qpos_width(value):
    if value is None:
        return None
    try:
        width = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimelineEditError(
            "Motion clipboard Joint Angle width is invalid."
        ) from exc
    if not math.isfinite(numeric) or width <= 0 or numeric != width:
        raise TimelineEditError(
            "Motion clipboard Joint Angle width is invalid."
        )
    return width


def _document_qpos_width(document):
    timeline = document.qpos_timeline
    robot_model = getattr(timeline, "robot_model", None)
    mj_model = getattr(robot_model, "mj_model", None)
    nq = getattr(mj_model, "nq", None)
    if nq is not None:
        return int(nq)
    states = _timeline_states(timeline)
    if states:
        return int(np.asarray(states[0][1]).size)
    return None


def _timeline_states(timeline):
    if timeline is None:
        return ()
    return tuple(
        (float(time), np.asarray(timeline.get_state(time), dtype=float).copy())
        for time in timeline.times()
    )


def _boundary_targets(document, time, qpos):
    targets = document.trajectory.targets_at_time(time)
    if qpos is None or not targets:
        return targets

    timeline = document.qpos_timeline
    robot_model = getattr(timeline, "robot_model", None)
    if robot_model is None:
        return targets
    try:
        state = robot_model.create_state()
        state.set_qpos(qpos)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return targets

    resolved = {}
    for frame_name, target in targets.items():
        try:
            binding = robot_model.resolve_logical_frame(frame_name)
            if binding is None:
                resolved[frame_name] = target
                continue
            kind, object_name = binding
            position, quaternion = state.get_body_pose(object_name, kind=kind)
            roll, pitch, yaw = quat_to_rpy(quaternion)
            resolved[frame_name] = replace(
                target,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            resolved[frame_name] = target
    return resolved


def _check_placement_overlaps(document, clip, placements, duration):
    original_tracks = {}
    clip_frame_names = {frame.frame_name for frame in clip.frames}
    for frame in document.trajectory.frames:
        if frame.frame_name in clip_frame_names:
            original_tracks.setdefault(frame.frame_name, []).append(frame.time)

    original_states = _timeline_states(document.qpos_timeline)
    state_times = [time for time, _qpos in original_states]
    for destination_start, _reverse in placements:
        destination_end = _time_key(destination_start + duration)
        for frame_name, times in original_tracks.items():
            if _ranges_overlap(
                min(times),
                max(times),
                destination_start,
                destination_end,
            ):
                raise TimelineEditError(
                    "Motion paste overlaps existing "
                    f"{frame_name} motion from {min(times):.2f}–"
                    f"{max(times):.2f} s. Use Insert Time or paste after it."
                )
        if clip.states and state_times and _ranges_overlap(
            min(state_times),
            max(state_times),
            destination_start,
            destination_end,
        ):
            raise TimelineEditError(
                "Motion paste overlaps existing committed robot poses from "
                f"{min(state_times):.2f}–{max(state_times):.2f} s. Use "
                "Insert Time or paste after them."
            )


def _capture_frame(frame_map, frame):
    key = (frame.frame_name, _time_key(frame.time))
    candidate = replace(frame, time=key[1])
    existing = frame_map.get(key)
    if existing is None:
        frame_map[key] = candidate
    elif not _frames_equivalent(existing, candidate):
        raise TimelineEditError(
            "The copied range contains conflicting "
            f"{frame.frame_name} targets at relative t={key[1]:.2f} s."
        )


def _capture_state(state_map, time, qpos, *, expected_width):
    key = _time_key(time)
    candidate = _validated_qpos(qpos, expected_width=expected_width)
    width = int(candidate.size)
    existing = state_map.get(key)
    if existing is None:
        state_map[key] = candidate
    elif not _states_equivalent(existing, candidate):
        raise TimelineEditError(
            "The copied range contains conflicting robot poses at relative "
            f"t={key:.2f} s."
        )
    return width


def _validated_qpos(qpos, *, expected_width):
    candidate = np.asarray(qpos, dtype=float)
    if candidate.ndim != 1 or not np.all(np.isfinite(candidate)):
        raise TimelineEditError(
            "The copied motion contains invalid Joint Angles."
        )
    if expected_width is not None and candidate.size != int(expected_width):
        raise TimelineEditError(
            "The copied motion contains inconsistent Joint Angle widths."
        )
    return candidate.copy()


def _frames_equivalent(first, second):
    if first.frame_name != second.frame_name or first.phase != second.phase:
        return False
    first_position = (
        first.x,
        first.y,
        first.z,
    )
    second_position = (
        second.x,
        second.y,
        second.z,
    )
    if not np.allclose(
        first_position,
        second_position,
        rtol=0.0,
        atol=1e-8,
    ):
        return False
    first_quaternion = np.asarray(
        rpy_to_quat(first.roll, first.pitch, first.yaw), dtype=float
    )
    second_quaternion = np.asarray(
        rpy_to_quat(second.roll, second.pitch, second.yaw), dtype=float
    )
    return bool(
        np.allclose(
            first_quaternion,
            second_quaternion,
            rtol=0.0,
            atol=1e-8,
        )
        or np.allclose(
            first_quaternion,
            -second_quaternion,
            rtol=0.0,
            atol=1e-8,
        )
    )


def _states_equivalent(first, second, *, robot_model=None):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        return False
    if np.allclose(first, second, rtol=0.0, atol=1e-8):
        return True

    adjusted = second.copy()
    free_joints = getattr(robot_model, "free_joints_by_body", {}).values()
    for joint in free_joints:
        quaternion_start = int(joint.qpos_address) + 3
        quaternion_end = quaternion_start + 4
        if quaternion_end > first.size:
            continue
        if np.allclose(
            first[quaternion_start:quaternion_end],
            -adjusted[quaternion_start:quaternion_end],
            rtol=0.0,
            atol=1e-8,
        ):
            adjusted[quaternion_start:quaternion_end] *= -1.0
    return bool(np.allclose(first, adjusted, rtol=0.0, atol=1e-8))


def _mapped_time(destination_start, relative_time, duration, reverse):
    offset = duration - relative_time if reverse else relative_time
    return _time_key(destination_start + offset)


def _relative_time(time, start_time, duration):
    relative = _time_key(float(time) - start_time)
    if abs(relative) <= TIME_TOLERANCE:
        return 0.0
    if abs(relative - duration) <= TIME_TOLERANCE:
        return duration
    return relative


def _clip_duration(clip):
    try:
        duration = _time_key(_finite_number(clip.duration, "Clip duration"))
    except AttributeError as exc:
        raise TimelineEditError("Motion clipboard data is invalid.") from exc
    if duration <= TIME_TOLERANCE:
        raise TimelineEditError("Copied motion duration must be greater than zero.")
    return duration


def _clip_item_time(value, duration):
    value = _time_key(_finite_number(value, "Clip Keyframe time"))
    if value < -TIME_TOLERANCE or value > duration + TIME_TOLERANCE:
        raise TimelineEditError(
            "The copied motion contains a Keyframe outside its duration."
        )
    if abs(value) <= TIME_TOLERANCE:
        return 0.0
    if abs(value - duration) <= TIME_TOLERANCE:
        return duration
    return value


def _in_range(time, start_time, end_time):
    return (
        float(time) >= start_time - TIME_TOLERANCE
        and float(time) <= end_time + TIME_TOLERANCE
    )


def _ranges_overlap(first_start, first_end, second_start, second_end):
    return (
        float(first_start) < float(second_end) - TIME_TOLERANCE
        and float(first_end) > float(second_start) + TIME_TOLERANCE
    )


def _check_maximum_time(duration, maximum_time):
    if maximum_time is None:
        return
    maximum_time = _finite_number(maximum_time, "Maximum timeline time")
    if duration > maximum_time + TIME_TOLERANCE:
        raise TimelineEditError(
            f"This edit would extend the timeline to {duration:.2f} s, beyond "
            f"the {maximum_time:.2f} s limit."
        )


def _positive_integer(value, label):
    if isinstance(value, bool):
        raise TimelineEditError(f"{label} must be a positive integer.")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimelineEditError(f"{label} must be a positive integer.") from exc
    if not math.isfinite(numeric) or integer < 1 or numeric != integer:
        raise TimelineEditError(f"{label} must be a positive integer.")
    return integer


def _finite_number(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimelineEditError(f"{label} must be finite.") from exc
    if not math.isfinite(value):
        raise TimelineEditError(f"{label} must be finite.")
    return value


def _nonnegative_time(value, label):
    value = _time_key(_finite_number(value, label))
    if value < 0.0:
        raise TimelineEditError(f"{label} cannot be negative.")
    return value


def _time_key(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimelineEditError("Timeline time must be finite.") from exc
    if not math.isfinite(value):
        raise TimelineEditError("Timeline time must be finite.")
    return round(value, TIME_PRECISION)

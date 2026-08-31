"""Atomic, Qt-free editing operations for keyframe timeline time."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from core.trajectory import TargetFrame

from .editor_commands import CommandResult
from .project_document import ProjectDocument


TIME_PRECISION = 6
TIME_TOLERANCE = 1e-6


class TimelineEditError(ValueError):
    """Raised when a requested timeline edit cannot be applied safely."""


@dataclass(frozen=True)
class TimelineEditPlan:
    """A fully validated replacement for both editable timeline sources."""

    operation: str
    frames: tuple[TargetFrame, ...]
    states: tuple[tuple[float, np.ndarray], ...]
    timeline_duration: float
    current_time: float
    moved_frame_count: int
    moved_state_count: int
    inserted_frame_count: int = 0
    inserted_state_count: int = 0

    @property
    def affected_count(self) -> int:
        return (
            self.moved_frame_count
            + self.moved_state_count
            + self.inserted_frame_count
            + self.inserted_state_count
        )


def snap_time(value: float, interval: float) -> float:
    """Snap a time value to the nearest export-sample interval."""
    value = _finite_number(value, "time")
    interval = _finite_number(interval, "Export interval")
    if interval <= 0.0:
        raise TimelineEditError("Export interval must be greater than zero.")
    return _time_key(round(value / interval) * interval)


def timeline_content_bounds(
    document: ProjectDocument,
) -> tuple[float, float] | None:
    """Return the earliest and latest logical-target/qpos Keyframe times."""
    times = [frame.time for frame in document.trajectory.frames]
    times.extend(time for time, _qpos in _timeline_states(document.qpos_timeline))
    if not times:
        return None
    return min(times), max(times)


def plan_insert_time(
    document: ProjectDocument,
    at_time: float,
    duration: float,
    *,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Open a held interval and shift everything at/after it to the right."""
    at_time = _nonnegative_time(at_time, "Insertion time")
    duration = _finite_number(duration, "Inserted duration")
    if duration <= 0.0:
        raise TimelineEditError("Inserted duration must be greater than zero.")
    duration = _time_key(duration)
    if duration <= 0.0:
        raise TimelineEditError(
            "Inserted duration is below the timeline's time precision."
        )

    original_frames = tuple(document.trajectory.frames)
    original_states = _timeline_states(document.qpos_timeline)
    boundary_targets = document.trajectory.targets_at_time(at_time)
    boundary_qpos = _sample_timeline(document.qpos_timeline, at_time)

    frame_map: dict[tuple[str, float], TargetFrame] = {}
    moved_frames = 0
    for frame in original_frames:
        should_move = frame.time >= at_time - TIME_TOLERANCE
        new_time = _time_key(frame.time + duration) if should_move else frame.time
        _add_unique_frame(
            frame_map,
            replace(frame, time=new_time) if should_move else frame,
            conflict_prefix="Insert Time produced",
        )
        moved_frames += int(should_move)

    inserted_frames = 0
    for target in boundary_targets.values():
        for boundary_time in (at_time, at_time + duration):
            key = (target.frame_name, _time_key(boundary_time))
            if key not in frame_map:
                frame_map[key] = replace(target, time=key[1])
                inserted_frames += 1

    state_map: dict[float, np.ndarray] = {}
    moved_states = 0
    for time, qpos in original_states:
        should_move = time >= at_time - TIME_TOLERANCE
        new_time = _time_key(time + duration) if should_move else time
        _add_unique_state(
            state_map,
            new_time,
            qpos,
            conflict_prefix="Insert Time produced",
        )
        moved_states += int(should_move)

    inserted_states = 0
    if boundary_qpos is not None:
        for boundary_time in (at_time, at_time + duration):
            key = _time_key(boundary_time)
            if key not in state_map:
                state_map[key] = boundary_qpos.copy()
                inserted_states += 1

    new_duration = max(
        0.1,
        document.timeline_duration + duration,
        _latest_time(frame_map, state_map),
    )
    _check_maximum_time(new_duration, maximum_time)
    current_time = (
        at_time
        if abs(document.current_time - at_time) <= TIME_TOLERANCE
        else (
            document.current_time + duration
            if document.current_time > at_time
            else document.current_time
        )
    )
    return _plan(
        "insert_time",
        frame_map,
        state_map,
        new_duration,
        current_time,
        moved_frames,
        moved_states,
        inserted_frames,
        inserted_states,
    )


def plan_shift_entire_motion(
    document: ProjectDocument,
    offset: float,
    *,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Translate every logical target and qpos keyframe by one offset."""
    offset = _finite_number(offset, "Time offset")
    offset = _time_key(offset)
    if abs(offset) <= TIME_TOLERANCE:
        raise TimelineEditError("Time offset must not be zero.")

    original_frames = tuple(document.trajectory.frames)
    original_states = _timeline_states(document.qpos_timeline)
    if not original_frames and not original_states:
        raise TimelineEditError("There are no Keyframes to shift.")

    frame_map: dict[tuple[str, float], TargetFrame] = {}
    for frame in original_frames:
        new_time = _time_key(frame.time + offset)
        if new_time < 0.0:
            raise TimelineEditError(
                f"Shifting by {offset:.2f} s would place a Keyframe before 0 s."
            )
        _add_unique_frame(
            frame_map,
            replace(frame, time=new_time),
            conflict_prefix="Shift Entire Motion produced",
        )

    state_map: dict[float, np.ndarray] = {}
    for time, qpos in original_states:
        new_time = _time_key(time + offset)
        if new_time < 0.0:
            raise TimelineEditError(
                f"Shifting by {offset:.2f} s would place a Keyframe before 0 s."
            )
        _add_unique_state(
            state_map,
            new_time,
            qpos,
            conflict_prefix="Shift Entire Motion produced",
        )

    requested_duration = (
        document.timeline_duration + offset
        if offset > 0.0
        else document.timeline_duration
    )
    new_duration = max(
        0.1,
        requested_duration,
        _latest_time(frame_map, state_map),
    )
    _check_maximum_time(new_duration, maximum_time)
    if state_map:
        current_time = min(state_map)
    elif frame_map:
        current_time = min(time for _name, time in frame_map)
    else:
        current_time = 0.0
    return _plan(
        "shift_entire_motion",
        frame_map,
        state_map,
        new_duration,
        current_time,
        len(original_frames),
        len(original_states),
    )


def plan_move_time_range(
    document: ProjectDocument,
    start_time: float,
    end_time: float,
    destination_start: float,
    *,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Move an inclusive Keyframe range after preflighting all conflicts."""
    start_time = _nonnegative_time(start_time, "Range start")
    end_time = _nonnegative_time(end_time, "Range end")
    destination_start = _nonnegative_time(
        destination_start, "Destination start"
    )
    if end_time <= start_time + TIME_TOLERANCE:
        raise TimelineEditError("Range end must be later than range start.")
    offset = _time_key(destination_start - start_time)
    if abs(offset) <= TIME_TOLERANCE:
        raise TimelineEditError("Destination start must differ from range start.")

    destination_end = _time_key(destination_start + end_time - start_time)
    ranges_overlap = (
        destination_start < end_time - TIME_TOLERANCE
        and destination_end > start_time + TIME_TOLERANCE
    )
    if ranges_overlap:
        raise TimelineEditError(
            "Source and destination ranges overlap. Use Insert Time or choose "
            "a non-overlapping destination."
        )

    original_frames = tuple(document.trajectory.frames)
    original_states = _timeline_states(document.qpos_timeline)
    frame_map: dict[tuple[str, float], TargetFrame] = {}
    moved_frames = 0
    for frame in original_frames:
        should_move = _in_range(frame.time, start_time, end_time)
        new_time = _time_key(frame.time + offset) if should_move else frame.time
        _add_unique_frame(
            frame_map,
            replace(frame, time=new_time) if should_move else frame,
            conflict_prefix="Move Time Range conflicts with",
        )
        moved_frames += int(should_move)

    state_map: dict[float, np.ndarray] = {}
    moved_states = 0
    for time, qpos in original_states:
        should_move = _in_range(time, start_time, end_time)
        new_time = _time_key(time + offset) if should_move else time
        _add_unique_state(
            state_map,
            new_time,
            qpos,
            conflict_prefix="Move Time Range conflicts with",
        )
        moved_states += int(should_move)

    if moved_frames == 0 and moved_states == 0:
        raise TimelineEditError(
            f"No Keyframes exist from {start_time:.2f} s to {end_time:.2f} s."
        )

    new_duration = max(
        0.1,
        document.timeline_duration,
        _latest_time(frame_map, state_map),
    )
    _check_maximum_time(new_duration, maximum_time)
    moved_state_times = [
        _time_key(time + offset)
        for time, _qpos in original_states
        if _in_range(time, start_time, end_time)
    ]
    moved_frame_times = [
        _time_key(frame.time + offset)
        for frame in original_frames
        if _in_range(frame.time, start_time, end_time)
    ]
    current_time = min(moved_state_times or moved_frame_times)
    return _plan(
        "move_time_range",
        frame_map,
        state_map,
        new_duration,
        current_time,
        moved_frames,
        moved_states,
    )


def plan_scale_time_range(
    document: ProjectDocument,
    start_time: float,
    end_time: float,
    speed: float,
    *,
    snap_interval: float | None = None,
    maximum_time: float | None = None,
) -> TimelineEditPlan:
    """Scale an inclusive Keyframe range around its fixed starting time."""
    start_time = _nonnegative_time(start_time, "Range start")
    end_time = _nonnegative_time(end_time, "Range end")
    speed = _finite_number(speed, "Motion speed")
    if speed <= 0.0:
        raise TimelineEditError("Motion speed must be greater than zero.")
    if abs(speed - 1.0) <= TIME_TOLERANCE:
        raise TimelineEditError("Motion speed must differ from 1.00×.")

    if snap_interval is not None:
        snap_interval = _finite_number(snap_interval, "Export interval")
        if snap_interval <= 0.0:
            raise TimelineEditError("Export interval must be greater than zero.")
    if end_time <= start_time + TIME_TOLERANCE:
        raise TimelineEditError("Range end must be later than range start.")

    original_frames = tuple(document.trajectory.frames)
    original_states = _timeline_states(document.qpos_timeline)
    selected_frames = tuple(
        frame
        for frame in original_frames
        if _in_range(frame.time, start_time, end_time)
    )
    selected_states = tuple(
        (time, qpos)
        for time, qpos in original_states
        if _in_range(time, start_time, end_time)
    )
    if not selected_frames and not selected_states:
        raise TimelineEditError(
            f"No Keyframes exist from {start_time:.2f} s to {end_time:.2f} s."
        )
    selected_times = [frame.time for frame in selected_frames]
    selected_times.extend(time for time, _qpos in selected_states)
    if not any(
        time > start_time + TIME_TOLERANCE for time in selected_times
    ):
        raise TimelineEditError(
            "Scaling requires at least two distinct Keyframe times."
        )

    result_end = _scaled_time(
        end_time,
        start_time,
        speed,
        snap_interval=snap_interval,
    )
    if result_end > end_time + TIME_TOLERANCE:
        outside_times = {
            _time_key(frame.time)
            for frame in original_frames
            if not _in_range(frame.time, start_time, end_time)
        }
        outside_times.update(
            _time_key(time)
            for time, _qpos in original_states
            if not _in_range(time, start_time, end_time)
        )
        blocked_time = min(
            (
                time
                for time in outside_times
                if time > end_time + TIME_TOLERANCE
                and time <= result_end + TIME_TOLERANCE
            ),
            default=None,
        )
        if blocked_time is not None:
            raise TimelineEditError(
                "Scaled range would overlap an existing Keyframe at "
                f"t={blocked_time:.2f} s. Move or insert time first."
            )

    frame_map: dict[tuple[str, float], TargetFrame] = {}
    moved_frame_times = []
    changed_frame_count = 0
    for frame in original_frames:
        should_scale = _in_range(frame.time, start_time, end_time)
        new_time = (
            _scaled_time(
                frame.time,
                start_time,
                speed,
                snap_interval=snap_interval,
            )
            if should_scale
            else frame.time
        )
        _add_unique_frame(
            frame_map,
            replace(frame, time=new_time) if should_scale else frame,
            conflict_prefix="Scale Time Range conflicts with",
        )
        if should_scale:
            moved_frame_times.append(new_time)
            changed_frame_count += int(
                abs(new_time - frame.time) > TIME_TOLERANCE
            )

    state_map: dict[float, np.ndarray] = {}
    moved_state_times = []
    changed_state_count = 0
    for time, qpos in original_states:
        should_scale = _in_range(time, start_time, end_time)
        new_time = (
            _scaled_time(
                time,
                start_time,
                speed,
                snap_interval=snap_interval,
            )
            if should_scale
            else time
        )
        _add_unique_state(
            state_map,
            new_time,
            qpos,
            conflict_prefix="Scale Time Range conflicts with",
        )
        if should_scale:
            moved_state_times.append(new_time)
            changed_state_count += int(
                abs(new_time - time) > TIME_TOLERANCE
            )

    if changed_frame_count == 0 and changed_state_count == 0:
        raise TimelineEditError(
            "This speed does not change any Keyframe time at the current "
            "precision or snapping interval."
        )

    new_duration = max(
        0.1,
        document.timeline_duration,
        _latest_time(frame_map, state_map),
    )
    _check_maximum_time(new_duration, maximum_time)
    current_time = min(moved_state_times or moved_frame_times)
    return _plan(
        "scale_time_range",
        frame_map,
        state_map,
        new_duration,
        current_time,
        len(selected_frames),
        len(selected_states),
    )


@dataclass(frozen=True)
class ApplyTimelineEditPlan:
    """Apply a preflighted timeline edit through the document command gateway."""

    plan: TimelineEditPlan

    @property
    def operation(self) -> str:
        return self.plan.operation

    def execute(self, document: ProjectDocument) -> CommandResult:
        previous_frames = tuple(document.trajectory.frames)
        previous_track_names = tuple(document.trajectory.tracks)
        previous_states = _timeline_states(document.qpos_timeline)
        previous_index = document.active_index
        previous_time = document.current_time
        previous_duration = document.timeline_duration
        try:
            document.trajectory.clear()
            for frame in self.plan.frames:
                document.trajectory.add_frame(frame)
            if document.qpos_timeline is not None:
                document.qpos_timeline.states.clear()
                for time, qpos in self.plan.states:
                    document.qpos_timeline.set_state(time, qpos)
            document.active_index = -1
            document.set_timeline_duration(self.plan.timeline_duration)
            document.set_current_time(self.plan.current_time)
        except Exception:
            document.trajectory.tracks = {
                name: [] for name in previous_track_names
            }
            for frame in previous_frames:
                document.trajectory.add_frame(frame)
            if document.qpos_timeline is not None:
                document.qpos_timeline.states.clear()
                for time, qpos in previous_states:
                    document.qpos_timeline.set_state(time, qpos)
            document.active_index = previous_index
            document.timeline_duration = previous_duration
            document.current_time = previous_time
            raise
        return CommandResult(
            True,
            self.operation,
            document.active_index,
            self.plan.affected_count,
        )


def _plan(
    operation,
    frame_map,
    state_map,
    timeline_duration,
    current_time,
    moved_frames,
    moved_states,
    inserted_frames=0,
    inserted_states=0,
):
    frames = tuple(
        sorted(frame_map.values(), key=lambda frame: (frame.time, frame.frame_name))
    )
    states = tuple(
        (time, state_map[time].copy()) for time in sorted(state_map)
    )
    return TimelineEditPlan(
        operation=operation,
        frames=frames,
        states=states,
        timeline_duration=float(timeline_duration),
        current_time=float(current_time),
        moved_frame_count=moved_frames,
        moved_state_count=moved_states,
        inserted_frame_count=inserted_frames,
        inserted_state_count=inserted_states,
    )


def _timeline_states(timeline):
    if timeline is None:
        return ()
    return tuple((time, timeline.get_state(time)) for time in timeline.times())


def _sample_timeline(timeline, time):
    if timeline is None:
        return None
    return timeline.sample_state(time)


def _add_unique_frame(frame_map, frame, *, conflict_prefix):
    key = (frame.frame_name, _time_key(frame.time))
    if key in frame_map:
        raise TimelineEditError(
            f"{conflict_prefix} an existing {frame.frame_name} target at "
            f"t={key[1]:.2f} s."
        )
    frame_map[key] = frame


def _add_unique_state(state_map, time, qpos, *, conflict_prefix):
    key = _time_key(time)
    if key in state_map:
        raise TimelineEditError(
            f"{conflict_prefix} an existing robot pose at t={key:.2f} s."
        )
    state_map[key] = np.asarray(qpos, dtype=float).copy()


def _latest_time(frame_map, state_map):
    times = [time for _name, time in frame_map]
    times.extend(state_map)
    return max(times, default=0.0)


def _check_maximum_time(duration, maximum_time):
    if maximum_time is None:
        return
    maximum_time = _finite_number(maximum_time, "Maximum timeline time")
    if duration > maximum_time + TIME_TOLERANCE:
        raise TimelineEditError(
            f"This edit would extend the timeline to {duration:.2f} s, beyond "
            f"the {maximum_time:.2f} s limit."
        )


def _in_range(time, start_time, end_time):
    return (
        time >= start_time - TIME_TOLERANCE
        and time <= end_time + TIME_TOLERANCE
    )


def _scaled_time(time, start_time, speed, *, snap_interval=None):
    scaled = start_time + (float(time) - start_time) / speed
    if snap_interval is not None:
        return snap_time(scaled, snap_interval)
    return _time_key(scaled)


def _time_key(value):
    return round(float(value), TIME_PRECISION)


def _finite_number(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise TimelineEditError(f"{label} must be finite.")
    return value


def _nonnegative_time(value, label):
    value = _time_key(_finite_number(value, label))
    if value < 0.0:
        raise TimelineEditError(f"{label} cannot be negative.")
    return value

"""Detached whole-motion snapshots and atomic document replacement."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

from application.editor_commands import CommandResult
from application.project_document import ProjectDocument, ProjectDocumentSnapshot
from core.trajectory import TargetFrame


@dataclass(frozen=True)
class MotionStateSnapshot:
    """Complete editable motion state, independent of presentation preview."""

    model_key: str
    trajectory: dict
    active_index: int
    current_time: float
    timeline_duration: float
    qpos_states: tuple[tuple[float, Any], ...]


def capture_motion_state(document: ProjectDocument) -> MotionStateSnapshot:
    timeline = document.qpos_timeline
    states = () if timeline is None else tuple(
        (float(time), _copy_value(timeline.get_state(time)))
        for time in timeline.times()
    )
    return MotionStateSnapshot(
        model_key=document.model_key,
        trajectory=document.trajectory.to_project_dict(),
        active_index=document.active_index,
        current_time=document.current_time,
        timeline_duration=document.timeline_duration,
        qpos_states=states,
    )


def detached_document(document: ProjectDocument) -> ProjectDocument:
    """Clone both logical Keyframes and qpos Keyframes for AI editing."""

    document_snapshot = document.snapshot()
    timeline = _clone_qpos_timeline(document.qpos_timeline)
    return ProjectDocument.from_snapshot(document_snapshot, qpos_timeline=timeline)


@dataclass(frozen=True)
class ReplaceMotionState:
    """Atomically replace both authoritative Keyframe representations."""

    state: MotionStateSnapshot
    operation: str = "replace_motion_state"

    def execute(self, document: ProjectDocument) -> CommandResult:
        if self.state.model_key != document.model_key:
            raise ValueError(
                "replacement motion model does not match the active document"
            )
        previous = capture_motion_state(document)
        _restore_motion_state(document, self.state)
        return CommandResult(
            changed=not _motion_states_equal(previous, self.state),
            operation=self.operation,
            active_index=document.active_index,
            affected_count=len(document.trajectory.frames) + len(self.state.qpos_states),
        )


def _clone_qpos_timeline(timeline: object | None) -> object | None:
    if timeline is None:
        return None
    if not hasattr(timeline, "states"):
        raise TypeError("qpos timeline must expose a mutable states mapping")
    clone = copy(timeline)
    clone.states = {}
    for time in timeline.times():
        clone.set_state(time, timeline.get_state(time))
    return clone


def _restore_motion_state(
    document: ProjectDocument,
    state: MotionStateSnapshot,
) -> None:
    replacement = ProjectDocument.from_snapshot(
        ProjectDocumentSnapshot(
            model_key=state.model_key,
            trajectory=state.trajectory,
            active_index=state.active_index,
            current_time=state.current_time,
            timeline_duration=state.timeline_duration,
            revision=document.revision,
            dirty=document.dirty,
        )
    )
    replacement_states = None
    if document.qpos_timeline is not None:
        replacement_timeline = copy(document.qpos_timeline)
        replacement_timeline.states = {}
        for time, qpos in state.qpos_states:
            replacement_timeline.set_state(time, _copy_value(qpos))
        replacement_states = replacement_timeline.states
    elif state.qpos_states:
        raise ValueError("cannot restore qpos Keyframes without a qpos timeline")

    # Everything failure-prone has completed on detached values. These swaps
    # now expose the logical and qpos Keyframes as one document transition.
    document.trajectory = replacement.trajectory
    if replacement_states is not None:
        document.qpos_timeline.states = replacement_states
    document.active_index = replacement.active_index
    document.timeline_duration = replacement.timeline_duration
    document.current_time = replacement.current_time


def _motion_states_equal(
    left: MotionStateSnapshot,
    right: MotionStateSnapshot,
) -> bool:
    if (
        left.model_key != right.model_key
        or left.trajectory != right.trajectory
        or left.active_index != right.active_index
        or left.current_time != right.current_time
        or left.timeline_duration != right.timeline_duration
        or len(left.qpos_states) != len(right.qpos_states)
    ):
        return False
    return all(
        left_time == right_time and _values_equal(left_value, right_value)
        for (left_time, left_value), (right_time, right_value)
        in zip(left.qpos_states, right.qpos_states)
    )


def _copy_value(value: Any) -> Any:
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else copy(value)


def _values_equal(left: Any, right: Any) -> bool:
    result = left == right
    all_method = getattr(result, "all", None)
    return bool(all_method()) if callable(all_method) else bool(result)

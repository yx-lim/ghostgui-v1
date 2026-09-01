"""Compact AI context assembled from authoritative editor state."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

from application.ai.edit_session import AIEditSession, SessionEditRecord
from application.ai.metadata import MotionMetadataService
from application.ai.schemas import EditAuthor
from application.project_document import ProjectDocument


@dataclass(frozen=True)
class EditorSelectionContext:
    """Ephemeral snapshot of GUI-owned selection at request time."""

    time_interval: tuple[float, float] | None = None
    logical_frame: str | None = None
    joint: str | None = None
    joint_group: str | None = None
    end_effector: str | None = None
    edit_mode: str | None = None
    camera_view: str | None = None

    def __post_init__(self) -> None:
        if self.time_interval is not None:
            start, end = (float(value) for value in self.time_interval)
            if not all(math.isfinite(value) and value >= 0.0 for value in (start, end)):
                raise ValueError("selected interval times must be finite and non-negative")
            if start > end:
                raise ValueError("selected interval start cannot exceed its end")
            object.__setattr__(self, "time_interval", (start, end))
        for field_name in (
            "logical_frame",
            "joint",
            "joint_group",
            "end_effector",
            "edit_mode",
            "camera_view",
        ):
            value = getattr(self, field_name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{field_name} must be non-empty when provided")


@dataclass(frozen=True)
class RobotCapabilityContext:
    """Compact model facts exposed by the model/UI adapter at request time."""

    logical_frames: tuple[str, ...] = ()
    end_effectors: tuple[str, ...] = ()
    joints: tuple[str, ...] = ()
    joint_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        for values in (self.logical_frames, self.end_effectors, self.joints):
            if any(not str(value).strip() for value in values):
                raise ValueError("robot capability names must not be empty")
        for group_name, members in self.joint_groups:
            if not str(group_name).strip() or any(
                not str(member).strip() for member in members
            ):
                raise ValueError("joint group names and members must not be empty")


@dataclass(frozen=True)
class AIContext:
    """JSON-safe compact context; raw qpos values are intentionally absent."""

    payload: dict

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self.payload))

    def to_prompt_text(self) -> str:
        return "GhostGUI editor context:\n" + json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
        )


class ContextBuilder:
    def __init__(
        self,
        *,
        max_times: int = 32,
        max_constraint_keyframes: int = 32,
        max_recent_edits: int = 8,
    ) -> None:
        if min(max_times, max_constraint_keyframes, max_recent_edits) <= 0:
            raise ValueError("context limits must be positive")
        self.max_times = int(max_times)
        self.max_constraint_keyframes = int(max_constraint_keyframes)
        self.max_recent_edits = int(max_recent_edits)

    def build(
        self,
        document: ProjectDocument,
        *,
        selection: EditorSelectionContext | None = None,
        robot_capabilities: RobotCapabilityContext | None = None,
        metadata: MotionMetadataService | None = None,
        recent_edits: Iterable[SessionEditRecord] = (),
        motion_name: str | None = None,
        validation_state: str | None = None,
        working_copy: bool = False,
    ) -> AIContext:
        selection = selection or EditorSelectionContext()
        robot_capabilities = robot_capabilities or RobotCapabilityContext()
        if (
            selection.time_interval is not None
            and selection.time_interval[1] > document.timeline_duration + 1e-9
        ):
            raise ValueError("selected interval exceeds the motion duration")
        frames = tuple(document.trajectory.frames)
        qpos_times = _qpos_times(document.qpos_timeline)
        all_times = sorted({frame.time for frame in frames}.union(qpos_times))
        active_frame = (
            frames[document.active_index]
            if 0 <= document.active_index < len(frames)
            else None
        )
        constraints, constraints_truncated = self._constraints(frames, metadata)
        edit_values = tuple(recent_edits)[-self.max_recent_edits :]

        payload = {
            "robot": {
                "model_key": document.model_key,
                "logical_frames": list(robot_capabilities.logical_frames),
                "end_effectors": list(robot_capabilities.end_effectors),
                "joints": list(robot_capabilities.joints),
                "joint_groups": {
                    name: list(members)
                    for name, members in robot_capabilities.joint_groups
                },
            },
            "motion": {
                "name": motion_name,
                "duration_seconds": document.timeline_duration,
                "current_time_seconds": document.current_time,
                "working_copy": bool(working_copy),
                "logical_keyframe_count": len(frames),
                "qpos_keyframe_count": len(qpos_times),
                "keyframe_times": _bounded_times(all_times, self.max_times),
                "tracks": {
                    name: {
                        "keyframe_count": len(track),
                        "times": _bounded_times(
                            (frame.time for frame in track),
                            self.max_times,
                        ),
                    }
                    for name, track in sorted(document.trajectory.tracks.items())
                    if track
                },
            },
            "selection": _selection_payload(selection, active_frame),
            "constraints": {
                "keyframes": constraints,
                "truncated": constraints_truncated,
            },
            "recent_edits": [
                {
                    "author": record.author.value,
                    "operation": record.operation,
                    "affected_entity_count": len(record.affected_entities),
                }
                for record in edit_values
            ],
            "validation_state": validation_state,
        }
        return AIContext(payload)

    def build_for_session(
        self,
        session: AIEditSession,
        *,
        selection: EditorSelectionContext | None = None,
        robot_capabilities: RobotCapabilityContext | None = None,
        metadata: MotionMetadataService | None = None,
        motion_name: str | None = None,
        validation_state: str | None = None,
    ) -> AIContext:
        working_metadata = (
            None
            if metadata is None
            else MotionMetadataService(session.metadata, metadata.resolver)
        )
        return self.build(
            session.working_document,
            selection=selection,
            robot_capabilities=robot_capabilities,
            metadata=working_metadata,
            recent_edits=session.edits,
            motion_name=motion_name,
            validation_state=validation_state,
            working_copy=True,
        )

    def _constraints(self, frames, metadata):
        if metadata is None:
            return [], False
        values = []
        for frame in frames:
            edit_metadata = metadata.metadata_for_keyframe(frame)
            if edit_metadata is None or (
                edit_metadata.author is not EditAuthor.USER
                and not edit_metadata.protected
            ):
                continue
            values.append(
                {
                    "logical_frame": frame.frame_name,
                    "time_seconds": frame.time,
                    "author": edit_metadata.author.value,
                    "protected": edit_metadata.protected,
                }
            )
        truncated = len(values) > self.max_constraint_keyframes
        return values[: self.max_constraint_keyframes], truncated


def _selection_payload(selection, active_frame):
    payload = {
        "time_interval_seconds": (
            None if selection.time_interval is None else list(selection.time_interval)
        ),
        "logical_frame": selection.logical_frame,
        "joint": selection.joint,
        "joint_group": selection.joint_group,
        "end_effector": selection.end_effector,
        "edit_mode": selection.edit_mode,
        "camera_view": selection.camera_view,
        "active_keyframe": None,
    }
    if active_frame is not None:
        payload["active_keyframe"] = {
            "time_seconds": active_frame.time,
            "phase": active_frame.phase,
            "logical_frame": active_frame.frame_name,
            "position_m": [active_frame.x, active_frame.y, active_frame.z],
            "orientation_rpy_rad": [
                active_frame.roll,
                active_frame.pitch,
                active_frame.yaw,
            ],
        }
    return payload


def _qpos_times(timeline) -> tuple[float, ...]:
    if timeline is None:
        return ()
    return tuple(float(time) for time in timeline.times())


def _bounded_times(values, limit):
    times = sorted({round(float(value), 6) for value in values})
    if len(times) <= limit:
        return {"values": times, "total_count": len(times), "truncated": False}
    if limit == 1:
        selected = [times[0]]
    else:
        selected = [
            times[round(index * (len(times) - 1) / (limit - 1))]
            for index in range(limit)
        ]
    return {"values": selected, "total_count": len(times), "truncated": True}

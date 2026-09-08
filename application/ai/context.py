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
            if edit_metadata is not None and (
                edit_metadata.author is not EditAuthor.USER
                and not edit_metadata.protected
            ):
                continue
            values.append(
                {
                    "logical_frame": frame.frame_name,
                    "time_seconds": frame.time,
                    "author": (
                        EditAuthor.USER.value
                        if edit_metadata is None
                        else edit_metadata.author.value
                    ),
                    "protected": (
                        False if edit_metadata is None else edit_metadata.protected
                    ),
                }
            )
        truncated = len(values) > self.max_constraint_keyframes
        return values[: self.max_constraint_keyframes], truncated


class MotionAssistantContextBuilder(ContextBuilder):
    """Add bounded sampled robot state using the existing timeline and FK."""

    def __init__(
        self,
        adapter,
        *,
        max_numerical_samples: int = 12,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 8 <= max_numerical_samples <= 20:
            raise ValueError("numerical motion samples must be bounded from 8 to 20")
        self.adapter = adapter
        self.max_numerical_samples = int(max_numerical_samples)

    def build(
        self,
        document: ProjectDocument,
        *,
        selection: EditorSelectionContext | None = None,
        robot_capabilities: RobotCapabilityContext | None = None,
        **kwargs,
    ) -> AIContext:
        selection = selection or EditorSelectionContext()
        capabilities = robot_capabilities or self._adapter_capabilities()
        base = super().build(
            document,
            selection=selection,
            robot_capabilities=capabilities,
            **kwargs,
        ).to_dict()
        base["robot"]["qpos_layout"] = self._qpos_layout()
        states = self._sampled_states(document, selection, capabilities)
        base["current_state"] = self._state_payload(
            document,
            document.current_time,
            tuple(capabilities.joints),
        )
        base["motion"]["numerical_samples"] = states
        base["motion"]["numerical_sample_count"] = len(states)
        return AIContext(base)

    def _adapter_capabilities(self) -> RobotCapabilityContext:
        return RobotCapabilityContext(
            logical_frames=tuple(self.adapter.logical_frame_bindings),
            end_effectors=tuple(self.adapter.end_effectors),
            joints=tuple(self.adapter.joint_names),
            joint_groups=tuple(
                (name, tuple(members))
                for name, members in sorted(self.adapter.joint_groups.items())
            ),
        )

    def _qpos_layout(self) -> dict:
        free_joints = tuple(self.adapter.free_joints_by_body.values())
        root = free_joints[0] if free_joints else None
        return {
            "width": int(self.adapter.mj_model.nq),
            "root": (
                None
                if root is None
                else {
                    "qpos_address": int(root.qpos_address),
                    "layout": "xyz + quaternion wxyz",
                }
            ),
            "ordered_joint_names": list(self.adapter.joint_names),
            "joint_units": "radians for hinge joints; metres for slide joints",
            "quaternion_convention": "wxyz",
        }

    def _sampled_states(self, document, selection, capabilities):
        if document.qpos_timeline is None:
            return []
        relevant_joints = self._relevant_joints(selection, capabilities)
        return [
            self._state_payload(document, time_seconds, relevant_joints)
            for time_seconds in _motion_sample_times(
                document.timeline_duration,
                document.current_time,
                selection.time_interval,
                self.max_numerical_samples,
            )
        ]

    def _relevant_joints(self, selection, capabilities):
        if selection.joint:
            return (selection.joint,)
        groups = dict(capabilities.joint_groups)
        if selection.joint_group and selection.joint_group in groups:
            return tuple(groups[selection.joint_group])
        selected_frame = selection.end_effector or selection.logical_frame
        if selected_frame:
            lower = selected_frame.lower()
            for side in ("left", "right"):
                if side in lower:
                    for suffix in ("arm", "leg"):
                        group = f"{side}_{suffix}"
                        if group in groups and (
                            ("hand" in lower and suffix == "arm")
                            or ("foot" in lower and suffix == "leg")
                        ):
                            return tuple(groups[group])
        return tuple(capabilities.joints)

    def _state_payload(self, document, time_seconds, joint_names):
        timeline = document.qpos_timeline
        if timeline is None:
            return {"time_seconds": _rounded(time_seconds), "available": False}
        qpos = timeline.sample_state(
            float(time_seconds),
            fallback_qpos=self.adapter.home_qpos,
        )
        state = self.adapter.create_state()
        state.set_qpos(qpos)
        return {
            "time_seconds": _rounded(time_seconds),
            "available": True,
            "root": self._root_payload(qpos),
            "joint_angles": {
                name: _rounded(state.get_joint_value(name))
                for name in joint_names
                if name in self.adapter.joints
            },
            "end_effectors": {
                name: self._pose_payload(state, name)
                for name in self.adapter.end_effectors
            },
            "pelvis": self._optional_pose_payload(state, "pelvis"),
            "torso": self._optional_pose_payload(state, "torso"),
        }

    def _root_payload(self, qpos):
        free_joints = tuple(self.adapter.free_joints_by_body.values())
        if not free_joints:
            return None
        address = int(free_joints[0].qpos_address)
        return {
            "position_m": [_rounded(value) for value in qpos[address:address + 3]],
            "orientation_quaternion_wxyz": [
                _rounded(value) for value in qpos[address + 3:address + 7]
            ],
        }

    def _optional_pose_payload(self, state, logical_name):
        if logical_name not in self.adapter.logical_frame_bindings:
            return None
        return self._pose_payload(state, logical_name)

    def _pose_payload(self, state, logical_name):
        kind, object_name = self.adapter.logical_frame_bindings[logical_name]
        position, quaternion = state.get_body_pose(object_name, kind)
        return {
            "position_m": [_rounded(value) for value in position],
            "orientation_quaternion_wxyz": [
                _rounded(value) for value in quaternion
            ],
        }


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


def _motion_sample_times(duration, current_time, interval, limit):
    duration = max(0.0, float(duration))
    current = min(duration, max(0.0, float(current_time)))
    if interval is not None:
        start, end = interval
        if abs(end - start) <= 1e-12:
            return (round(start, 6),)
        count = min(limit, 12)
        return tuple(
            round(start + (end - start) * index / (count - 1), 6)
            for index in range(count)
        )

    representative_count = min(8, limit)
    if duration <= 1e-12:
        return (0.0,)
    values = {
        round(duration * index / (representative_count - 1), 6)
        for index in range(representative_count)
    }
    local_step = min(0.1, duration / 20.0)
    values.update(
        round(min(duration, max(0.0, current + offset * local_step)), 6)
        for offset in (-2, -1, 0, 1, 2)
    )
    ordered = sorted(values)
    if len(ordered) <= limit:
        return tuple(ordered)
    keep = {0, len(ordered) - 1, min(range(len(ordered)), key=lambda i: abs(ordered[i] - current))}
    for index in range(len(ordered)):
        if len(keep) >= limit:
            break
        keep.add(index)
    return tuple(ordered[index] for index in sorted(keep))


def _rounded(value):
    return round(float(value), 6)

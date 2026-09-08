"""Robot-aware services backing semantic AI tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol

from core.math3d import quaternion_angle, rpy_to_quaternion
from core.robotics import QposContract
from core.trajectory import TargetFrame, quat_to_rpy


TARGET_FK_POSITION_TOLERANCE_M = 0.001
TARGET_FK_ORIENTATION_TOLERANCE_RAD = 0.01


class SemanticMotionError(ValueError):
    """A semantic motion request cannot be satisfied safely."""


@dataclass(frozen=True)
class LogicalFrameSolveResult:
    frame: TargetFrame
    qpos: Any
    status: str
    position_error: float = 0.0
    collisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointAngleEditResult:
    """A qpos edit and the existing logical Keyframes changed by its FK."""

    qpos: Any
    logical_frames: tuple[TargetFrame, ...] = ()


@dataclass(frozen=True)
class MotionValidationReport:
    """Result of local structural and kinematic checks, not dynamics."""

    valid: bool
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class SemanticMotionService(Protocol):
    @property
    def logical_frames(self) -> tuple[str, ...]:
        ...

    @property
    def end_effectors(self) -> tuple[str, ...]:
        ...

    @property
    def joint_names(self) -> tuple[str, ...]:
        ...

    @property
    def joint_groups(self) -> Mapping[str, tuple[str, ...]]:
        ...

    def solve_logical_frame_target(
        self,
        document,
        *,
        logical_frame: str,
        time_seconds: float,
        position_m: tuple[float, float, float],
        orientation_rpy_rad: tuple[float, float, float] | None,
        mode: str,
        protected_logical_frames: tuple[str, ...],
    ) -> LogicalFrameSolveResult:
        ...

    def set_joint_angles(
        self,
        document,
        *,
        time_seconds: float,
        values: Mapping[str, float],
        protected_logical_frames: tuple[str, ...],
    ) -> JointAngleEditResult:
        ...

    def ensure_qpos_keyframe(self, document, *, time_seconds: float) -> Any:
        ...

    def validate_motion(self, document) -> MotionValidationReport:
        ...


class GhostGUIMotionService:
    """Use existing model state, collision-aware IK, and Joint Angle APIs."""

    def __init__(
        self,
        adapter,
        *,
        collision_solver=None,
        validator: Callable[[object], MotionValidationReport] | None = None,
    ) -> None:
        self.adapter = adapter
        if collision_solver is None:
            from core.ik import CollisionAwareIKSolver

            collision_solver = CollisionAwareIKSolver(adapter)
        self.collision_solver = collision_solver
        self.validator = validator

    @property
    def logical_frames(self) -> tuple[str, ...]:
        return tuple(self.adapter.logical_frame_bindings)

    @property
    def end_effectors(self) -> tuple[str, ...]:
        return tuple(self.adapter.end_effectors)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.adapter.joint_names)

    @property
    def joint_groups(self) -> Mapping[str, tuple[str, ...]]:
        return {
            name: tuple(values)
            for name, values in self.adapter.joint_groups.items()
            if values
        }

    def solve_logical_frame_target(
        self,
        document,
        *,
        logical_frame,
        time_seconds,
        position_m,
        orientation_rpy_rad,
        mode,
        protected_logical_frames,
    ) -> LogicalFrameSolveResult:
        self._validate_time(document, time_seconds)
        try:
            kind, object_name = self.adapter.logical_frame_bindings[logical_frame]
        except KeyError as error:
            raise SemanticMotionError(
                f"unknown logical frame: {logical_frame}"
            ) from error
        current_qpos = self._sample_qpos(document, time_seconds)
        state = self.adapter.create_state()
        state.set_qpos(current_qpos)
        start_position, start_quaternion = state.get_body_pose(object_name, kind)
        target_position = _vector(position_m, "position")
        current_rpy = quat_to_rpy(start_quaternion)
        target_rpy = (
            tuple(current_rpy)
            if orientation_rpy_rad is None
            else _vector(orientation_rpy_rad, "orientation")
        )
        if mode == "delta":
            target_position = tuple(
                float(start_position[index]) + target_position[index]
                for index in range(3)
            )
            if orientation_rpy_rad is not None:
                target_rpy = tuple(
                    float(current_rpy[index]) + target_rpy[index]
                    for index in range(3)
                )
        elif mode != "absolute":
            raise SemanticMotionError(f"unsupported target mode: {mode}")

        from core.ik import BodyPoseTask
        from core.math3d import rpy_to_quaternion

        secondary_tasks = []
        for protected_name in protected_logical_frames:
            if protected_name == logical_frame:
                continue
            binding = self.adapter.logical_frame_bindings.get(protected_name)
            if binding is None:
                continue
            protected_kind, protected_object = binding
            position, quaternion = state.get_body_pose(
                protected_object,
                protected_kind,
            )
            secondary_tasks.append(BodyPoseTask(
                name=f"Protect {protected_name}",
                weight=2.0,
                priority=1,
                required=True,
                tolerance=0.005,
                object_name=protected_object,
                kind=protected_kind,
                target_position=position,
                target_quaternion=quaternion,
            ))
        result = self.collision_solver.solve_drag(
            current_qpos,
            start_position,
            start_quaternion,
            target_position,
            rpy_to_quaternion(*target_rpy),
            object_name=object_name,
            kind=kind,
            joint_weights=self.adapter.default_ik_joint_weights(),
            secondary_tasks=secondary_tasks,
        )
        if not result.success:
            raise SemanticMotionError(result.status)
        achieved_rpy = quat_to_rpy(result.quaternion)
        frame = TargetFrame(
            time=time_seconds,
            phase="ai_edit",
            frame_name=logical_frame,
            x=float(result.position[0]),
            y=float(result.position[1]),
            z=float(result.position[2]),
            roll=float(achieved_rpy[0]),
            pitch=float(achieved_rpy[1]),
            yaw=float(achieved_rpy[2]),
        )
        collisions = tuple(
            getattr(item, "pair_label", str(item)) for item in result.collisions
        )
        return LogicalFrameSolveResult(
            frame=frame,
            qpos=result.qpos,
            status=result.status,
            position_error=float(result.ik_error),
            collisions=collisions,
        )

    def set_joint_angles(
        self,
        document,
        *,
        time_seconds,
        values,
        protected_logical_frames,
    ):
        self._validate_time(document, time_seconds)
        unknown = set(values) - set(self.joint_names)
        if unknown:
            raise SemanticMotionError(f"unknown Joint Angle: {sorted(unknown)[0]}")
        for name, value in values.items():
            value = float(value)
            if not math.isfinite(value):
                raise SemanticMotionError(f"Joint Angle {name} must be finite")
            limits = self.adapter.get_joint_limits(name)
            if limits is not None and not (limits[0] <= value <= limits[1]):
                raise SemanticMotionError(
                    f"Joint Angle {name} is outside its model limits"
                )
        state = self.adapter.create_state()
        state.set_qpos(self._sample_qpos(document, time_seconds))
        protected_poses = {}
        for logical_frame in protected_logical_frames:
            binding = self.adapter.logical_frame_bindings.get(logical_frame)
            if binding is None:
                continue
            kind, object_name = binding
            protected_poses[logical_frame] = (
                kind,
                object_name,
                state.get_body_pose(object_name, kind),
            )
        state.set_joint_values(values)
        from core.math3d import quaternion_angle

        for logical_frame, (kind, object_name, before_pose) in protected_poses.items():
            before_position, before_quaternion = before_pose
            after_position, after_quaternion = state.get_body_pose(object_name, kind)
            position_error = math.sqrt(sum(
                (float(after_position[index]) - float(before_position[index])) ** 2
                for index in range(3)
            ))
            orientation_error = quaternion_angle(
                after_quaternion,
                before_quaternion,
            )
            if position_error > 0.001 or orientation_error > 0.01:
                raise SemanticMotionError(
                    f"Joint Angle edit would move protected {logical_frame}"
                )
        qpos = state.get_qpos()
        changed_joints = frozenset(values)
        logical_frames = []
        for frame in document.frames_at_time(time_seconds):
            if not changed_joints.intersection(
                self.adapter.joint_chain_for_frame(frame.frame_name)
            ):
                continue
            kind, object_name = self.adapter.logical_frame_bindings[
                frame.frame_name
            ]
            position, quaternion = state.get_body_pose(object_name, kind)
            roll, pitch, yaw = quat_to_rpy(quaternion)
            logical_frames.append(TargetFrame(
                time=frame.time,
                phase=frame.phase,
                frame_name=frame.frame_name,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            ))
        return JointAngleEditResult(qpos, tuple(logical_frames))

    def ensure_qpos_keyframe(self, document, *, time_seconds):
        self._validate_time(document, time_seconds)
        timeline = document.qpos_timeline
        if timeline is None:
            raise SemanticMotionError("motion has no editable qpos timeline")
        return timeline.sample_state(time_seconds)

    def validate_motion(self, document) -> MotionValidationReport:
        if self.validator is not None:
            return self.validator(document)
        issues = []
        warnings = []
        duration = _finite_float(document.timeline_duration)
        if duration is None or duration <= 0.0:
            issues.append("Motion duration must be positive and finite")
            duration = None

        current_time = _finite_float(document.current_time)
        if current_time is None or current_time < 0.0:
            issues.append("Current motion time must be finite and non-negative")
        elif duration is not None and current_time > duration + 1e-9:
            issues.append("Current motion time exceeds motion duration")

        adapter_model_key = str(
            getattr(getattr(self.adapter, "info", None), "key", "")
        )
        model_matches = (
            not adapter_model_key or document.model_key == adapter_model_key
        )
        if not model_matches:
            issues.append(
                f"Motion model {document.model_key} does not match active model "
                f"{adapter_model_key}"
            )

        valid_frames = []
        for frame in document.trajectory.frames:
            if frame.frame_name not in self.logical_frames:
                issues.append(f"Unknown logical frame {frame.frame_name}")
                continue
            frame_values = (
                frame.time,
                frame.x,
                frame.y,
                frame.z,
                frame.roll,
                frame.pitch,
                frame.yaw,
            )
            if any(_finite_float(value) is None for value in frame_values):
                issues.append(
                    f"Logical Keyframe {frame.frame_name} contains a "
                    "non-finite value"
                )
                continue
            frame_time = float(frame.time)
            if frame_time < 0.0:
                issues.append(
                    f"Logical Keyframe {frame.frame_name} has a negative time"
                )
                continue
            if duration is not None and frame_time > duration + 1e-9:
                issues.append(
                    f"Logical Keyframe at {frame_time:.3f} s exceeds motion "
                    "duration"
                )
                continue
            valid_frames.append(frame)

        for frame_name, track in document.trajectory.tracks.items():
            times = [
                float(frame.time)
                for frame in track
                if _finite_float(frame.time) is not None
            ]
            if any(
                earlier >= later
                for earlier, later in zip(times, times[1:])
            ):
                issues.append(
                    f"Logical Keyframe times for {frame_name} must be strictly "
                    "increasing"
                )

        timeline = document.qpos_timeline
        if timeline is not None:
            timeline_model_key = str(getattr(
                getattr(getattr(timeline, "robot_model", None), "info", None),
                "key",
                "",
            ))
            if (
                adapter_model_key
                and timeline_model_key
                and timeline_model_key != adapter_model_key
            ):
                issues.append(
                    f"qpos timeline model {timeline_model_key} does not match "
                    f"active model {adapter_model_key}"
                )
            try:
                raw_times = tuple(timeline.times())
            except (TypeError, ValueError):
                issues.append("qpos Keyframe times are unavailable or invalid")
                raw_times = ()
            normalized_times = []
            previous_time = None
            for raw_time in raw_times:
                time = _finite_float(raw_time)
                if time is None:
                    issues.append("qpos Keyframe has a non-finite time")
                    continue
                if time < 0.0:
                    issues.append("qpos Keyframe has a negative time")
                    continue
                if previous_time is not None and time <= previous_time:
                    issues.append(
                        "qpos Keyframe times must be strictly increasing"
                    )
                previous_time = time
                if duration is not None and time > duration + 1e-9:
                    issues.append(
                        f"qpos Keyframe at {time:.3f} s exceeds motion duration"
                    )
                    continue
                normalized_times.append((time, raw_time))

            collision_checker = getattr(
                self.collision_solver,
                "collision_checker",
                None,
            )
            qpos_by_time = {}
            contract = QposContract(int(self.adapter.mj_model.nq))
            for time, raw_time in normalized_times:
                try:
                    qpos = contract.validate(
                        timeline.get_state(raw_time),
                        context=f"qpos Keyframe at {time:.3f} s",
                    )
                except (TypeError, ValueError) as error:
                    issues.append(str(error))
                    continue

                limit_issues = self._joint_limit_issues(qpos, time)
                issues.extend(limit_issues)
                if limit_issues:
                    continue
                qpos_by_time[time] = qpos

                if collision_checker is not None:
                    collision_state = self.adapter.create_state()
                    collision_state.set_qpos(qpos)
                    blocking = tuple(
                        collision
                        for collision in collision_checker.get_collisions(collision_state)
                        if getattr(collision, "blocking", False)
                    )
                    if blocking:
                        warnings.append(
                            f"qpos Keyframe at {time:.3f} s has "
                            f"{len(blocking)} blocking collision(s)"
                        )
            if model_matches:
                issues.extend(
                    self._target_fk_issues(valid_frames, qpos_by_time)
                )
        return MotionValidationReport(not issues, tuple(issues), tuple(warnings))

    def _joint_limit_issues(self, qpos, time_seconds):
        issues = []
        for name in self.joint_names:
            limits = self.adapter.get_joint_limits(name)
            if limits is None:
                continue
            joint = self.adapter.joints.get(self.adapter.plain_name(name))
            if joint is None:
                continue
            value = float(qpos[joint.qpos_address])
            if value < limits[0] - 1e-9 or value > limits[1] + 1e-9:
                issues.append(
                    f"Joint Angle {name} at {time_seconds:.3f} s is outside "
                    "its model limits"
                )
        return issues

    def _target_fk_issues(self, frames, qpos_by_time):
        issues = []
        states = {}
        for frame in frames:
            qpos_time = next(
                (
                    time
                    for time in qpos_by_time
                    if abs(time - float(frame.time)) <= 1e-6
                ),
                None,
            )
            if qpos_time is None:
                continue
            state = states.get(qpos_time)
            if state is None:
                state = self.adapter.create_state()
                state.set_qpos(qpos_by_time[qpos_time])
                states[qpos_time] = state
            kind, object_name = self.adapter.logical_frame_bindings[
                frame.frame_name
            ]
            position, quaternion = state.get_body_pose(object_name, kind)
            position_error = math.sqrt(sum(
                (float(position[index]) - target) ** 2
                for index, target in enumerate((frame.x, frame.y, frame.z))
            ))
            orientation_error = quaternion_angle(
                quaternion,
                rpy_to_quaternion(frame.roll, frame.pitch, frame.yaw),
            )
            if (
                position_error > TARGET_FK_POSITION_TOLERANCE_M
                or orientation_error > TARGET_FK_ORIENTATION_TOLERANCE_RAD
            ):
                issues.append(
                    f"Logical Keyframe {frame.frame_name} at "
                    f"{float(frame.time):.3f} s does not match qpos forward "
                    "kinematics"
                )
        return issues

    @staticmethod
    def _validate_time(document, time_seconds):
        value = float(time_seconds)
        if not math.isfinite(value) or value < 0.0:
            raise SemanticMotionError("Keyframe time must be finite and non-negative")
        if value > document.timeline_duration + 1e-9:
            raise SemanticMotionError("Keyframe time exceeds motion duration")

    @staticmethod
    def _sample_qpos(document, time_seconds):
        timeline = document.qpos_timeline
        if timeline is None:
            raise SemanticMotionError("motion has no editable qpos timeline")
        return timeline.sample_state(time_seconds)


def _finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(values, label):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise SemanticMotionError(f"{label} must contain three finite values")
    return values

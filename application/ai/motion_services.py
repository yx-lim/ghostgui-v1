"""Robot-aware services backing semantic AI tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol

from core.trajectory import TargetFrame, quat_to_rpy


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
class MotionValidationReport:
    valid: bool
    issues: tuple[str, ...] = ()


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
    ) -> Any:
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
        return state.get_qpos()

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
        for frame in document.trajectory.frames:
            if frame.frame_name not in self.logical_frames:
                issues.append(f"Unknown logical frame {frame.frame_name}")
            if frame.time > document.timeline_duration + 1e-9:
                issues.append(
                    f"Keyframe at {frame.time:.3f} s exceeds motion duration"
                )
        timeline = document.qpos_timeline
        if timeline is not None:
            collision_checker = getattr(
                self.collision_solver,
                "collision_checker",
                None,
            )
            collision_state = (
                self.adapter.create_state()
                if collision_checker is not None else None
            )
            for time in timeline.times():
                qpos = timeline.get_state(time)
                try:
                    finite = all(math.isfinite(float(value)) for value in qpos)
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    issues.append(f"qpos Keyframe at {float(time):.3f} s is non-finite")
                    continue
                if collision_state is not None:
                    collision_state.set_qpos(qpos)
                    blocking = tuple(
                        collision
                        for collision in collision_checker.get_collisions(collision_state)
                        if getattr(collision, "blocking", False)
                    )
                    if blocking:
                        issues.append(
                            f"qpos Keyframe at {float(time):.3f} s has "
                            f"{len(blocking)} blocking collision(s)"
                        )
        return MotionValidationReport(not issues, tuple(issues))

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


def _vector(values, label):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise SemanticMotionError(f"{label} must contain three finite values")
    return values

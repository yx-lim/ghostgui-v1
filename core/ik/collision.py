"""MuJoCo contact inspection and collision-aware candidate IK dragging."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.math3d import quaternion_slerp
from .solver import IKSolverSettings
from .tasks import TCPOrientationTask, TCPPositionTask


@dataclass(frozen=True)
class Collision:
    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float
    kind: str
    geom1_id: int | None = None
    geom2_id: int | None = None
    body1_id: int | None = None
    body2_id: int | None = None
    body1_label: str | None = None
    body2_label: str | None = None
    blocking: bool = True

    @property
    def pair_label(self):
        return (
            f"{self.body1_label or self.body1} ↔ "
            f"{self.body2_label or self.body2}"
        )

    @property
    def diagnostic_label(self):
        penetration_mm = max(0.0, -float(self.distance)) * 1000.0
        severity = "blocking" if self.blocking else "advisory"
        return (
            f"{self.geom1} ↔ {self.geom2} "
            f"({penetration_mm:.1f} mm penetration, {severity})"
        )


def format_collision_pairs(collisions, limit=2):
    return ", ".join(item.pair_label for item in list(collisions)[:int(limit)])


def format_collision_diagnostics(collisions, limit=2):
    return ", ".join(
        item.diagnostic_label for item in list(collisions)[:int(limit)]
    )


@dataclass(frozen=True)
class CollisionPolicy:
    """Model-aware distinction between support contact and collision."""

    allowed_contact_pairs: frozenset[frozenset[str]] = frozenset()
    support_body_ids: frozenset[int] = frozenset()
    support_penetration_tolerance: float = 0.01
    allowed_body_pair_tolerances: tuple[
        tuple[frozenset[str], float], ...
    ] = ()
    blocking_penetration: float = 0.001

    @classmethod
    def for_robot_model(
        cls,
        robot_model,
        *,
        allowed_contact_pairs=None,
        support_penetration_tolerance=0.01,
    ):
        model = robot_model.mj_model
        info = getattr(robot_model, "info", None)
        support_body_ids = set()
        for logical_name, binding in getattr(
            robot_model,
            "logical_frame_bindings",
            {},
        ).items():
            if not any(
                token in logical_name.lower()
                for token in ("foot", "ankle")
            ):
                continue
            kind, object_name = binding
            object_type = (
                mujoco.mjtObj.mjOBJ_SITE
                if kind == "site"
                else mujoco.mjtObj.mjOBJ_BODY
            )
            object_id = mujoco.mj_name2id(model, object_type, object_name)
            if object_id < 0:
                continue
            body_id = (
                int(model.site_bodyid[object_id])
                if kind == "site"
                else int(object_id)
            )
            support_body_ids.add(body_id)
        return cls(
            allowed_contact_pairs=frozenset(
                frozenset(pair) for pair in (allowed_contact_pairs or ())
            ),
            allowed_body_pair_tolerances=tuple(
                (frozenset((body1, body2)), max(0.0, float(tolerance)))
                for body1, body2, tolerance in getattr(
                    info, "allowed_contact_body_pairs", ()
                )
            ),
            support_body_ids=frozenset(support_body_ids),
            support_penetration_tolerance=float(
                support_penetration_tolerance
            ),
            blocking_penetration=max(0.0, float(getattr(
                info, "collision_blocking_penetration_m", 0.001
            ))),
        )

    def allows(
        self,
        geom1,
        geom2,
        body1,
        body2,
        body1_id,
        body2_id,
        distance,
    ):
        if frozenset((geom1, geom2)) in self.allowed_contact_pairs:
            return True
        body_pair = frozenset((body1, body2))
        if any(
            body_pair == allowed_pair
            and float(distance) >= -float(penetration_tolerance)
            for allowed_pair, penetration_tolerance
            in self.allowed_body_pair_tolerances
        ):
            return True
        if 0 not in (body1_id, body2_id):
            return False
        robot_body_id = body2_id if body1_id == 0 else body1_id
        return (
            robot_body_id in self.support_body_ids
            and float(distance) >= -self.support_penetration_tolerance
        )

    def is_blocking(self, distance):
        return float(distance) <= -float(self.blocking_penetration)


class CollisionChecker:
    """Reports penetrating MuJoCo contacts with configurable allowed pairs."""

    def __init__(
        self,
        robot_model,
        allowed_contact_pairs=None,
        tolerance=0.0,
        policy=None,
    ):
        self.robot_model = robot_model
        self.model = robot_model.mj_model
        self.tolerance = float(tolerance)
        self.policy = policy or CollisionPolicy.for_robot_model(
            robot_model,
            allowed_contact_pairs=allowed_contact_pairs,
        )
        self.allowed_contact_pairs = set(self.policy.allowed_contact_pairs)

    def allow_contact(self, geom1, geom2):
        self.allowed_contact_pairs.add(frozenset((geom1, geom2)))
        self.policy = CollisionPolicy(
            allowed_contact_pairs=frozenset(self.allowed_contact_pairs),
            allowed_body_pair_tolerances=(
                self.policy.allowed_body_pair_tolerances
            ),
            support_body_ids=self.policy.support_body_ids,
            support_penetration_tolerance=(
                self.policy.support_penetration_tolerance
            ),
            blocking_penetration=self.policy.blocking_penetration,
        )

    def get_collisions(self, state):
        # mj_forward runs broad/narrow-phase collision and populates data.contact.
        state.forward_kinematics()
        collisions_by_pair = {}
        for contact_index in range(state.mj_data.ncon):
            contact = state.mj_data.contact[contact_index]
            if float(contact.dist) > self.tolerance:
                continue
            geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
            geom1 = self._name(mujoco.mjtObj.mjOBJ_GEOM, geom1_id, "geom")
            geom2 = self._name(mujoco.mjtObj.mjOBJ_GEOM, geom2_id, "geom")
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            body1 = self._name(mujoco.mjtObj.mjOBJ_BODY, body1_id, "body")
            body2 = self._name(mujoco.mjtObj.mjOBJ_BODY, body2_id, "body")
            if self.policy.allows(
                geom1,
                geom2,
                body1,
                body2,
                body1_id,
                body2_id,
                float(contact.dist),
            ):
                continue
            kind = "environment" if 0 in (body1_id, body2_id) else "self"
            collision = Collision(
                geom1,
                geom2,
                body1,
                body2,
                float(contact.dist),
                kind,
                geom1_id,
                geom2_id,
                body1_id,
                body2_id,
                self._frame_label(body1_id, body1, geom1),
                self._frame_label(body2_id, body2, geom2),
                self.policy.is_blocking(float(contact.dist)),
            )
            pair = tuple(sorted((geom1_id, geom2_id)))
            previous = collisions_by_pair.get(pair)
            if previous is None or collision.distance < previous.distance:
                collisions_by_pair[pair] = collision
        return list(collisions_by_pair.values())

    def is_state_collision_free(self, state):
        return not self.get_collisions(state)

    def get_blocking_collisions(self, state):
        return [
            collision for collision in self.get_collisions(state)
            if collision.blocking
        ]

    def _name(self, object_type, object_id, fallback):
        if object_type == mujoco.mjtObj.mjOBJ_GEOM:
            resolver = getattr(self.robot_model, "get_geom_name", None)
            if resolver is not None:
                return resolver(object_id)
        if object_type == mujoco.mjtObj.mjOBJ_BODY:
            resolver = getattr(self.robot_model, "get_body_name", None)
            if resolver is not None:
                return resolver(object_id)
        return (
            mujoco.mj_id2name(self.model, object_type, object_id)
            or f"{fallback}#{object_id}"
        )

    @staticmethod
    def _frame_label(body_id, body_name, geom_name):
        # World-owned environment geoms have no more specific body frame.
        # Retain their source name (for example ``ground``) instead of
        # replacing it with MuJoCo's synthetic ``world`` body name.
        if int(body_id) == 0 and geom_name:
            return geom_name
        return body_name


@dataclass(frozen=True)
class TrajectoryCollisionReport:
    sample_index: int
    collisions: tuple[Collision, ...]

    @property
    def blocking(self):
        return any(collision.blocking for collision in self.collisions)


def first_trajectory_collision(
    robot_model,
    qposes,
    *,
    checker=None,
    blocking_only=False,
):
    """Return the first warning or blocking collision in sampled qpos data."""
    checker = checker or CollisionChecker(robot_model)
    state = robot_model.create_state()
    for sample_index, qpos in enumerate(qposes):
        state.set_qpos(qpos)
        collisions = checker.get_collisions(state)
        if blocking_only:
            collisions = [item for item in collisions if item.blocking]
        if collisions:
            return TrajectoryCollisionReport(
                int(sample_index), tuple(collisions)
            )
    return None


def trajectory_collision_reports(robot_model, qposes, *, checker=None):
    """Find first warning and first blocking sample in one trajectory pass."""
    checker = checker or CollisionChecker(robot_model)
    state = robot_model.create_state()
    warning_report = None
    blocking_report = None
    for sample_index, qpos in enumerate(qposes):
        state.set_qpos(qpos)
        collisions = checker.get_collisions(state)
        if collisions and warning_report is None:
            warning_report = TrajectoryCollisionReport(
                int(sample_index), tuple(collisions)
            )
        blocking = tuple(item for item in collisions if item.blocking)
        if blocking and blocking_report is None:
            blocking_report = TrajectoryCollisionReport(
                int(sample_index), blocking
            )
        if warning_report is not None and blocking_report is not None:
            break
    return warning_report, blocking_report


@dataclass
class DragSolveResult:
    qpos: object
    position: object
    quaternion: object
    accepted_fraction: float
    success: bool
    status: str
    ik_error: float = 0.0
    collisions: list[Collision] = field(default_factory=list)
    near_singularity: bool = False
    min_singular_value: float = float("inf")
    condition_number: float = 0.0
    relaxed_constraints: bool = False


class CollisionAwareIKSolver:
    """Solves incrementally and reports collisions without blocking previews."""

    def __init__(
        self,
        robot_model,
        collision_checker=None,
        collision_drag_substeps=8,
        ik_tolerance=0.001,
        orientation_weight=0.25,
    ):
        self.robot_model = robot_model
        self.candidate_state = robot_model.create_state()
        self.collision_checker = collision_checker or CollisionChecker(robot_model)
        self.collision_drag_substeps = max(1, int(collision_drag_substeps))
        self.ik_tolerance = float(ik_tolerance)
        self.orientation_weight = float(orientation_weight)

    def solve_drag(
        self,
        current_qpos,
        start_position,
        start_quaternion,
        proposed_position,
        proposed_quaternion,
        *,
        object_name,
        kind=None,
        joint_weights=None,
        secondary_tasks=None,
        solver_settings=None,
        tcp_position_weight=1.0,
        tcp_orientation_weight=None,
        tcp_position_required=True,
        tcp_orientation_required=True,
    ):
        start_position = np.asarray(start_position, dtype=float)
        proposed_position = np.asarray(proposed_position, dtype=float)
        accepted_qpos = np.asarray(current_qpos, dtype=float).copy()
        accepted_position = start_position.copy()
        accepted_quaternion = np.asarray(start_quaternion, dtype=float).copy()
        accepted_fraction = 0.0
        last_error = 0.0
        near_singularity = False
        min_singular_value = float("inf")
        condition_number = 0.0
        relaxed_constraints = False
        blocked_reason = None
        settings = dict(solver_settings or {})
        orientation_weight = (
            self.orientation_weight
            if tcp_orientation_weight is None else float(tcp_orientation_weight)
        )
        if (
            not np.isfinite(tcp_position_weight)
            or not np.isfinite(orientation_weight)
            or tcp_position_weight < 0.0
            or orientation_weight < 0.0
        ):
            raise ValueError("IK task weights must be finite and nonnegative")
        ik_settings = IKSolverSettings.from_mapping(
            settings,
            position_tolerance=settings.get(
                "position_tolerance",
                self.ik_tolerance,
            ),
            orientation_tolerance=settings.get(
                "orientation_tolerance",
                0.03,
            ),
            orientation_weight=orientation_weight,
        )
        if tcp_position_weight <= 0.0 and orientation_weight <= 0.0:
            return DragSolveResult(
                accepted_qpos,
                accepted_position,
                accepted_quaternion,
                0.0,
                False,
                "IK reach limit: selected TCP tasks are disabled",
            )

        for step in range(1, self.collision_drag_substeps + 1):
            fraction = step / self.collision_drag_substeps
            candidate_position = (
                start_position + fraction * (proposed_position - start_position)
            )
            candidate_quaternion = quaternion_slerp(
                start_quaternion, proposed_quaternion, fraction
            )
            self.candidate_state.set_qpos(accepted_qpos)
            tasks = None
            if not hasattr(self.candidate_state, "solve_weighted_tasks"):
                ik_result = self.candidate_state.solve_ik(
                    object_name,
                    candidate_position,
                    candidate_quaternion,
                    kind=kind,
                    tolerance=self.ik_tolerance,
                    orientation_weight=orientation_weight,
                )
                resolved_kind = kind
                object_id = None
                free_root = None
            else:
                resolved_kind, object_id = self.candidate_state.resolve_object(
                    object_name, kind
                )
                free_root = (
                    self.robot_model.free_joint_for_body(object_id)
                    if resolved_kind == "body" else None
                )
            if (
                hasattr(self.candidate_state, "solve_weighted_tasks")
                and free_root is not None
            ):
                current_position, _ = self.candidate_state.get_body_pose(
                    object_name, resolved_kind
                )
                ik_result = self.candidate_state.solve_ik(
                    object_name,
                    (
                        candidate_position
                        if tcp_position_weight > 0.0 else current_position
                    ),
                    candidate_quaternion if orientation_weight > 0.0 else None,
                    kind=resolved_kind,
                    max_iterations=ik_settings.max_iterations,
                    tolerance=ik_settings.position_tolerance,
                    orientation_tolerance=ik_settings.orientation_tolerance,
                    orientation_weight=orientation_weight,
                    damping=ik_settings.damping,
                    step_size=ik_settings.step_size,
                    max_step=ik_settings.max_step,
                )
            elif hasattr(self.candidate_state, "solve_weighted_tasks"):
                tasks = [TCPPositionTask(
                    name="Selected TCP position",
                    weight=float(tcp_position_weight),
                    priority=2,
                    required=bool(tcp_position_required),
                    tolerance=ik_settings.position_tolerance,
                    object_name=object_name,
                    kind=resolved_kind,
                    target_position=candidate_position,
                )]
                if orientation_weight > 0.0:
                    tasks.append(TCPOrientationTask(
                        name="Selected TCP orientation",
                        weight=orientation_weight,
                        priority=2,
                        required=bool(tcp_orientation_required),
                        tolerance=ik_settings.orientation_tolerance,
                        object_name=object_name,
                        kind=resolved_kind,
                        target_quaternion=candidate_quaternion,
                    ))
                tasks.extend(secondary_tasks or [])
                ik_result = self.candidate_state.solve_weighted_tasks(
                    tasks,
                    joint_weights=joint_weights,
                    max_iterations=ik_settings.max_iterations,
                    damping=ik_settings.damping,
                    step_size=ik_settings.step_size,
                    max_step=ik_settings.max_step,
                )
            if not ik_result.success and tasks:
                required_tasks = [
                    task for task in tasks
                    if task.enabled and task.weight > 0.0 and task.required
                ]
                optional_tasks = [
                    task for task in tasks
                    if task.enabled and task.weight > 0.0 and not task.required
                ]
                if required_tasks and optional_tasks:
                    self.candidate_state.set_qpos(accepted_qpos)
                    retry_result = self.candidate_state.solve_weighted_tasks(
                        required_tasks,
                        joint_weights=joint_weights,
                        max_iterations=ik_settings.max_iterations,
                        damping=max(0.08, ik_settings.damping),
                        step_size=ik_settings.step_size,
                        max_step=ik_settings.max_step,
                    )
                    if retry_result.success:
                        ik_result = retry_result
                        relaxed_constraints = True
            last_error = ik_result.error
            if not ik_result.success:
                near_singularity = near_singularity or ik_result.near_singularity
                min_singular_value = min(
                    min_singular_value, ik_result.min_singular_value
                )
                condition_number = max(
                    condition_number, ik_result.condition_number
                )
                blocked_reason = f"IK reach limit: {ik_result.message}"
                break
            near_singularity = near_singularity or ik_result.near_singularity
            min_singular_value = min(
                min_singular_value, ik_result.min_singular_value
            )
            condition_number = max(condition_number, ik_result.condition_number)

            accepted_qpos = self.candidate_state.get_qpos()
            solved_position, solved_quaternion = self.candidate_state.get_body_pose(
                object_name, resolved_kind
            ) if hasattr(self.candidate_state, "get_body_pose") else (
                candidate_position, candidate_quaternion
            )
            accepted_position = (
                candidate_position
                if tcp_position_weight > 0.0 and tcp_position_required
                else solved_position
            )
            accepted_quaternion = (
                candidate_quaternion
                if orientation_weight > 0.0 and tcp_orientation_required
                else solved_quaternion
            )
            accepted_fraction = fraction

        self.candidate_state.set_qpos(accepted_qpos)
        preview_collisions = self.collision_checker.get_collisions(
            self.candidate_state
        )
        collision_note = ""
        if preview_collisions:
            names = format_collision_pairs(preview_collisions)
            details = format_collision_diagnostics(preview_collisions)
            collision_note = (
                f"; Collision warning: {names}; Contact geometry: {details}"
            )

        if blocked_reason is not None:
            return DragSolveResult(
                accepted_qpos,
                accepted_position,
                accepted_quaternion,
                accepted_fraction,
                accepted_fraction > 0.0,
                blocked_reason + collision_note,
                last_error,
                preview_collisions,
                near_singularity,
                min_singular_value,
                condition_number,
                relaxed_constraints,
            )
        relaxation_note = (
            "; optional constraints relaxed" if relaxed_constraints else ""
        )
        return DragSolveResult(
            accepted_qpos,
            accepted_position,
            accepted_quaternion,
            1.0,
            True,
            (
                f"{ik_result.message}{relaxation_note}{collision_note}"
                if preview_collisions
                else f"{ik_result.message}{relaxation_note}; "
                "state is collision-free"
            ),
            last_error,
            preview_collisions,
            near_singularity,
            min_singular_value,
            condition_number,
            relaxed_constraints,
        )

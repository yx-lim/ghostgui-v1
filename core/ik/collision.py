"""MuJoCo contact inspection and collision-aware candidate IK dragging."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.math3d import quaternion_slerp
from .tasks import TCPOrientationTask, TCPPositionTask


@dataclass(frozen=True)
class Collision:
    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float
    kind: str


class CollisionChecker:
    """Reports penetrating MuJoCo contacts with configurable allowed pairs."""

    def __init__(self, robot_model, allowed_contact_pairs=None, tolerance=0.0):
        self.robot_model = robot_model
        self.model = robot_model.mj_model
        self.tolerance = float(tolerance)
        self.allowed_contact_pairs = {
            frozenset(pair) for pair in (allowed_contact_pairs or [])
        }

    def allow_contact(self, geom1, geom2):
        self.allowed_contact_pairs.add(frozenset((geom1, geom2)))

    def get_collisions(self, state):
        # mj_forward runs broad/narrow-phase collision and populates data.contact.
        state.forward_kinematics()
        collisions = []
        for contact_index in range(state.mj_data.ncon):
            contact = state.mj_data.contact[contact_index]
            if float(contact.dist) > self.tolerance:
                continue
            geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
            geom1 = self._name(mujoco.mjtObj.mjOBJ_GEOM, geom1_id, "geom")
            geom2 = self._name(mujoco.mjtObj.mjOBJ_GEOM, geom2_id, "geom")
            if frozenset((geom1, geom2)) in self.allowed_contact_pairs:
                continue
            body1_id = int(self.model.geom_bodyid[geom1_id])
            body2_id = int(self.model.geom_bodyid[geom2_id])
            body1 = self._name(mujoco.mjtObj.mjOBJ_BODY, body1_id, "body")
            body2 = self._name(mujoco.mjtObj.mjOBJ_BODY, body2_id, "body")
            kind = "environment" if 0 in (body1_id, body2_id) else "self"
            collisions.append(Collision(
                geom1, geom2, body1, body2, float(contact.dist), kind
            ))
        return collisions

    def is_state_collision_free(self, state):
        return not self.get_collisions(state)

    def _name(self, object_type, object_id, fallback):
        return (
            mujoco.mj_id2name(self.model, object_type, object_id)
            or f"{fallback}#{object_id}"
        )


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


class CollisionAwareIKSolver:
    """Solves in a temporary state and accepts the furthest valid substep."""

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
    ):
        start_position = np.asarray(start_position, dtype=float)
        proposed_position = np.asarray(proposed_position, dtype=float)
        accepted_qpos = np.asarray(current_qpos, dtype=float).copy()
        accepted_position = start_position.copy()
        accepted_quaternion = np.asarray(start_quaternion, dtype=float).copy()
        accepted_fraction = 0.0
        last_error = 0.0
        blocked_collisions = []
        blocked_reason = None
        settings = dict(solver_settings or {})
        orientation_weight = (
            self.orientation_weight
            if tcp_orientation_weight is None else float(tcp_orientation_weight)
        )
        if tcp_position_weight <= 0.0 and orientation_weight <= 0.0:
            return DragSolveResult(
                accepted_qpos,
                accepted_position,
                accepted_quaternion,
                0.0,
                False,
                "IK blocked drag: selected TCP tasks are disabled",
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
                    tolerance=float(settings.get(
                        "position_tolerance", self.ik_tolerance
                    )),
                    orientation_weight=orientation_weight,
                )
            elif hasattr(self.candidate_state, "solve_weighted_tasks"):
                tasks = [TCPPositionTask(
                    name="Selected TCP position",
                    weight=float(tcp_position_weight),
                    priority=2,
                    required=True,
                    tolerance=float(settings.get(
                        "position_tolerance", self.ik_tolerance
                    )),
                    object_name=object_name,
                    kind=resolved_kind,
                    target_position=candidate_position,
                )]
                if orientation_weight > 0.0:
                    tasks.append(TCPOrientationTask(
                        name="Selected TCP orientation",
                        weight=orientation_weight,
                        priority=2,
                        required=True,
                        tolerance=float(settings.get(
                            "orientation_tolerance", 0.03
                        )),
                        object_name=object_name,
                        kind=resolved_kind,
                        target_quaternion=candidate_quaternion,
                    ))
                tasks.extend(secondary_tasks or [])
                ik_result = self.candidate_state.solve_weighted_tasks(
                    tasks,
                    joint_weights=joint_weights,
                    max_iterations=int(settings.get("max_iterations", 80)),
                    damping=float(settings.get("damping", 0.04)),
                    step_size=float(settings.get("step_size", 0.7)),
                    max_step=float(settings.get("max_step", 0.08)),
                )
            last_error = ik_result.error
            if not ik_result.success:
                blocked_reason = f"IK blocked drag: {ik_result.message}"
                break

            blocked_collisions = self.collision_checker.get_collisions(
                self.candidate_state
            )
            if blocked_collisions:
                names = ", ".join(
                    f"{item.geom1} ↔ {item.geom2}" for item in blocked_collisions[:2]
                )
                blocked_reason = f"Collision blocked drag: {names}"
                break

            accepted_qpos = self.candidate_state.get_qpos()
            solved_position, solved_quaternion = self.candidate_state.get_body_pose(
                object_name, resolved_kind
            ) if hasattr(self.candidate_state, "get_body_pose") else (
                candidate_position, candidate_quaternion
            )
            accepted_position = (
                candidate_position if tcp_position_weight > 0.0 else solved_position
            )
            accepted_quaternion = (
                candidate_quaternion if orientation_weight > 0.0 else solved_quaternion
            )
            accepted_fraction = fraction

        if blocked_reason is not None:
            return DragSolveResult(
                accepted_qpos,
                accepted_position,
                accepted_quaternion,
                accepted_fraction,
                accepted_fraction > 0.0,
                blocked_reason,
                last_error,
                blocked_collisions,
            )
        return DragSolveResult(
            accepted_qpos,
            accepted_position,
            accepted_quaternion,
            1.0,
            True,
            f"{ik_result.message}; state is collision-free",
            last_error,
            [],
        )

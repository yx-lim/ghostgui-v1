"""MuJoCo contact inspection and collision-aware candidate IK dragging."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from core.math3d import quaternion_slerp
from core.robotics import validate_trajectory_arrays
from core.trajectory import interpolate_qpos_manifold
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
    # Intended support contact needs a tiny numerical slop, not permission to
    # sink visibly into the environment.  Environment penetration beyond this
    # boundary is always blocking, independently of the more permissive
    # model-specific self-collision threshold.
    support_penetration_tolerance: float = 0.0005
    environment_penetration_tolerance: float = 0.0005
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
        support_penetration_tolerance=0.0005,
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
            environment_penetration_tolerance=float(
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

    def is_blocking(self, distance, *, kind=None):
        if kind == "environment":
            return float(distance) <= -float(
                self.environment_penetration_tolerance
            )
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
            environment_penetration_tolerance=(
                self.policy.environment_penetration_tolerance
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
                self.policy.is_blocking(float(contact.dist), kind=kind),
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
    segment_index: int | None = None
    segment_fraction: float = 0.0
    time: float | None = None

    @property
    def blocking(self):
        return any(collision.blocking for collision in self.collisions)

    @property
    def is_interior(self):
        return (
            self.segment_index is not None
            and 0.0 < float(self.segment_fraction) < 1.0
        )

    @property
    def location_label(self):
        """Human-readable location without assuming a particular UI."""
        time_note = (
            "" if self.time is None else f" at {float(self.time):.6g} s"
        )
        if self.is_interior:
            percent = 100.0 * float(self.segment_fraction)
            return (
                f"segment {self.segment_index} at {percent:.1f}%"
                f"{time_note}"
            )
        return f"sample {self.sample_index}{time_note}"


@dataclass(frozen=True)
class _AdaptiveCollisionSample:
    qpos: np.ndarray
    collisions: tuple[Collision, ...]
    centers: np.ndarray
    rotations: np.ndarray
    radii: np.ndarray


def _collision_geometry_snapshot(model, state):
    """Copy collision-geometry transforms used to bound swept movement."""
    data = getattr(state, "mj_data", None)
    if data is None:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3, 3), dtype=float),
            np.empty(0, dtype=float),
        )

    geom_xpos = getattr(data, "geom_xpos", None)
    geom_xmat = getattr(data, "geom_xmat", None)
    if geom_xpos is not None and geom_xmat is not None:
        count = int(getattr(model, "ngeom", len(geom_xpos)))
        geom_ids = np.arange(count, dtype=int)
        contype = getattr(model, "geom_contype", None)
        conaffinity = getattr(model, "geom_conaffinity", None)
        if contype is not None and conaffinity is not None:
            active = np.logical_or(
                np.asarray(contype, dtype=int) != 0,
                np.asarray(conaffinity, dtype=int) != 0,
            )
            geom_ids = geom_ids[active]
        centers = np.asarray(geom_xpos, dtype=float)[geom_ids].copy()
        rotations = np.asarray(geom_xmat, dtype=float)[geom_ids].reshape(
            (-1, 3, 3)
        ).copy()
        model_radii = getattr(model, "geom_rbound", None)
        radii = (
            np.zeros(len(geom_ids), dtype=float)
            if model_radii is None
            else np.asarray(model_radii, dtype=float)[geom_ids].copy()
        )
        return centers, rotations, radii

    # Minimal/custom state adapters may expose only body transforms.  They
    # still provide a useful translational bound; configuration-space movement
    # separately bounds rotations.
    body_xpos = getattr(data, "xpos", None)
    body_xmat = getattr(data, "xmat", None)
    if body_xpos is None or body_xmat is None:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3, 3), dtype=float),
            np.empty(0, dtype=float),
        )
    centers = np.asarray(body_xpos, dtype=float).reshape((-1, 3)).copy()
    rotations = np.asarray(body_xmat, dtype=float).reshape((-1, 3, 3)).copy()
    return centers, rotations, np.zeros(len(centers), dtype=float)


def _configuration_movement(model, start, end):
    displacement = np.zeros(int(model.nv), dtype=float)
    mujoco.mj_differentiatePos(
        model, displacement, 1.0, start.qpos, end.qpos
    )
    return (
        0.0
        if displacement.size == 0
        else float(np.max(np.abs(displacement)))
    )


def _swept_geometry_movement(start, end):
    """Conservative endpoint bound for points inside geom bounding spheres."""
    if start.centers.size == 0 or end.centers.size == 0:
        return 0.0
    if start.centers.shape != end.centers.shape:
        return float("inf")
    translations = np.linalg.norm(end.centers - start.centers, axis=1)
    # trace(R0.T @ R1) equals the element-wise Frobenius product.
    traces = np.einsum("nij,nij->n", start.rotations, end.rotations)
    cos_angles = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    radii = np.maximum(start.radii, end.radii)
    rotational_sweep = 2.0 * radii * np.sin(0.5 * angles)
    return float(np.max(translations + rotational_sweep))


def adaptive_trajectory_collision_reports(
    robot_model,
    qposes,
    *,
    times=None,
    checker=None,
    max_joint_step=0.08,
    max_body_step=0.02,
    max_depth=12,
):
    """Find first advisory and blocking contacts, including between samples.

    Each input sample is checked.  Every adjacent pair is then recursively
    subdivided on the MuJoCo configuration manifold until both the largest
    generalized-coordinate step and the largest collision-geometry sweep are
    bounded by the requested tolerances.  The return shape deliberately
    matches :func:`trajectory_collision_reports`.

    This is adaptive discrete validation, not an analytic continuous-collision
    proof.  Its safety resolution is independent of trajectory/export timing.
    """
    max_joint_step = float(max_joint_step)
    max_body_step = float(max_body_step)
    max_depth = int(max_depth)
    if not np.isfinite(max_joint_step) or max_joint_step <= 0.0:
        raise ValueError("max_joint_step must be finite and positive")
    if not np.isfinite(max_body_step) or max_body_step <= 0.0:
        raise ValueError("max_body_step must be finite and positive")
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    qposes = tuple(qposes)
    if times is None:
        times = tuple(float(index) for index in range(len(qposes)))
    normalized_times, normalized_qposes = validate_trajectory_arrays(
        times, qposes, int(robot_model.mj_model.nq)
    )
    if not normalized_qposes:
        return None, None

    checker = checker or CollisionChecker(robot_model)
    state = robot_model.create_state()
    model = robot_model.mj_model

    def evaluate(qpos):
        state.set_qpos(qpos)
        collisions = tuple(checker.get_collisions(state))
        centers, rotations, radii = _collision_geometry_snapshot(model, state)
        return _AdaptiveCollisionSample(
            np.asarray(qpos, dtype=float).copy(),
            collisions,
            centers,
            rotations,
            radii,
        )

    endpoint_samples = [evaluate(qpos) for qpos in normalized_qposes]
    ordered_locations = []
    if len(endpoint_samples) == 1:
        ordered_locations.append((
            0, None, 0.0, normalized_times[0], endpoint_samples[0]
        ))
    else:
        for segment_index in range(len(endpoint_samples) - 1):
            start = endpoint_samples[segment_index]
            end = endpoint_samples[segment_index + 1]
            samples_by_fraction = {0.0: start, 1.0: end}

            def subdivide(lo_fraction, lo_sample, hi_fraction, hi_sample, depth):
                configuration_movement = _configuration_movement(
                    model, lo_sample, hi_sample
                )
                body_movement = _swept_geometry_movement(lo_sample, hi_sample)
                if depth >= max_depth or (
                    configuration_movement <= max_joint_step
                    and body_movement <= max_body_step
                ):
                    return
                mid_fraction = 0.5 * (lo_fraction + hi_fraction)
                mid_qpos = interpolate_qpos_manifold(
                    model,
                    normalized_qposes[segment_index],
                    normalized_qposes[segment_index + 1],
                    mid_fraction,
                )
                mid_sample = evaluate(mid_qpos)
                samples_by_fraction[mid_fraction] = mid_sample
                subdivide(
                    lo_fraction, lo_sample, mid_fraction, mid_sample, depth + 1
                )
                subdivide(
                    mid_fraction, mid_sample, hi_fraction, hi_sample, depth + 1
                )

            subdivide(0.0, start, 1.0, end, 0)
            start_time = normalized_times[segment_index]
            duration = normalized_times[segment_index + 1] - start_time
            for fraction in sorted(samples_by_fraction):
                if segment_index > 0 and fraction == 0.0:
                    continue
                sample_index = (
                    segment_index + 1 if fraction == 1.0 else segment_index
                )
                ordered_locations.append((
                    sample_index,
                    segment_index,
                    fraction,
                    start_time + fraction * duration,
                    samples_by_fraction[fraction],
                ))

    warning_report = None
    blocking_report = None
    for sample_index, segment_index, fraction, time, sample in ordered_locations:
        if sample.collisions and warning_report is None:
            warning_report = TrajectoryCollisionReport(
                int(sample_index),
                sample.collisions,
                segment_index,
                float(fraction),
                float(time),
            )
        blocking = tuple(
            collision for collision in sample.collisions if collision.blocking
        )
        if blocking and blocking_report is None:
            blocking_report = TrajectoryCollisionReport(
                int(sample_index),
                blocking,
                segment_index,
                float(fraction),
                float(time),
            )
        if warning_report is not None and blocking_report is not None:
            break
    return warning_report, blocking_report


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
    """Solve incrementally while keeping the accepted preview collision-safe.

    Each drag substep is solved and collision-checked before it becomes the
    next accepted state. Advisory contacts may remain visible, but a blocking
    contact clamps the handle at the last safe substep.
    """

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
        barrier_collisions = ()
        ground_projection_offset = 0.0
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

            candidate_collisions = tuple(
                self.collision_checker.get_collisions(self.candidate_state)
            )
            blocking_collisions = tuple(
                collision
                for collision in candidate_collisions
                if collision.blocking
            )
            candidate_ground_projected = False
            if (
                blocking_collisions
                and all(
                    collision.kind == "environment"
                    for collision in blocking_collisions
                )
                and getattr(self, "robot_model", None) is not None
            ):
                # Imported lazily to keep the collision data types usable by
                # the projector without introducing an import-time cycle.
                from .motion_safety import project_qpos_above_flat_ground

                projection = project_qpos_above_flat_ground(
                    self.robot_model,
                    self.candidate_state.get_qpos(),
                    checker=self.collision_checker,
                )
                if projection.success:
                    self.candidate_state.set_qpos(projection.qpos)
                    candidate_ground_projected = projection.changed
                    ground_projection_offset = max(
                        ground_projection_offset,
                        float(projection.applied_offset),
                    )
                    candidate_collisions = tuple(
                        self.collision_checker.get_collisions(
                            self.candidate_state
                        )
                    )
                    blocking_collisions = tuple(
                        collision
                        for collision in candidate_collisions
                        if collision.blocking
                    )
            if blocking_collisions:
                barrier_collisions = blocking_collisions
                names = format_collision_pairs(blocking_collisions)
                details = format_collision_diagnostics(blocking_collisions)
                blocked_reason = (
                    "Safety barrier stopped the drag before a blocking "
                    f"collision ({names}); Contact geometry: {details}"
                )
                break

            accepted_qpos = self.candidate_state.get_qpos()
            solved_position, solved_quaternion = self.candidate_state.get_body_pose(
                object_name, resolved_kind
            ) if hasattr(self.candidate_state, "get_body_pose") else (
                candidate_position, candidate_quaternion
            )
            accepted_position = (
                solved_position
                if candidate_ground_projected
                else candidate_position
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
            barrier_note = ""
            if barrier_collisions and preview_collisions:
                barrier_note = "; last accepted state" + collision_note
            return DragSolveResult(
                accepted_qpos,
                accepted_position,
                accepted_quaternion,
                accepted_fraction,
                accepted_fraction > 0.0,
                blocked_reason + barrier_note,
                last_error,
                list(barrier_collisions or preview_collisions),
                near_singularity,
                min_singular_value,
                condition_number,
                relaxed_constraints,
            )
        relaxation_note = (
            "; optional constraints relaxed" if relaxed_constraints else ""
        )
        ground_note = (
            f"; ground barrier raised the base by up to "
            f"{ground_projection_offset * 1000.0:.1f} mm"
            if ground_projection_offset > 0.0 else ""
        )
        return DragSolveResult(
            accepted_qpos,
            accepted_position,
            accepted_quaternion,
            1.0,
            True,
            (
                f"{ik_result.message}{relaxation_note}{ground_note}{collision_note}"
                if preview_collisions
                else f"{ik_result.message}{relaxation_note}; "
                f"state is collision-free{ground_note}"
            ),
            last_error,
            preview_collisions,
            near_singularity,
            min_singular_value,
            condition_number,
            relaxed_constraints,
        )

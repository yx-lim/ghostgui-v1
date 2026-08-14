"""Deterministic motion-safety repairs built on MuJoCo collision geometry.

The editor is kinematic: assigning ``qpos`` and calling ``mj_forward`` detects
contact, but it does not apply contact forces.  This module therefore provides
an explicit, deliberately narrow projection for the one correction that is
unambiguous enough to automate: lifting a single floating-root robot out of a
flat, horizontal world plane.

Self-collision and contact with arbitrary environment geometry are not repaired
here.  Those cases require a path planner because changing one configuration
can change the authored motion's meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .collision import CollisionChecker


DEFAULT_GROUND_PENETRATION_TOLERANCE = 0.0005
DEFAULT_MAX_AUTOMATIC_GROUND_LIFT = 0.25
_HORIZONTAL_NORMAL_COSINE = 0.999


@dataclass(frozen=True)
class GroundProjectionResult:
    """Result of attempting the limited flat-ground projection."""

    success: bool
    qpos: np.ndarray
    applied_offset: float = 0.0
    penetration_depth: float = 0.0
    reason: str = ""

    @property
    def changed(self):
        return self.success and self.applied_offset > 0.0


@dataclass(frozen=True)
class _GroundContactSummary:
    deepest_distance: float | None
    unsupported_reason: str | None


def _ground_contact_summary(robot_model, state, penetration_tolerance):
    """Inspect raw contacts, including policy-allowed support-foot contact."""
    state.forward_kinematics()
    model = robot_model.mj_model
    deepest_distance = None

    for contact_index in range(int(state.mj_data.ncon)):
        contact = state.mj_data.contact[contact_index]
        distance = float(contact.dist)
        if distance >= -float(penetration_tolerance):
            continue

        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        world_owned = (body1_id == 0, body2_id == 0)
        if not any(world_owned):
            continue
        if all(world_owned):
            continue

        ground_geom_id = geom1_id if body1_id == 0 else geom2_id
        geom_type = int(model.geom_type[ground_geom_id])
        if geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE):
            ground_name = robot_model.get_geom_name(ground_geom_id)
            return _GroundContactSummary(
                deepest_distance,
                f"environment geometry {ground_name!r} is not a flat plane",
            )

        rotation = np.asarray(
            state.mj_data.geom_xmat[ground_geom_id], dtype=float
        ).reshape(3, 3)
        normal = rotation[:, 2]
        if float(normal[2]) < _HORIZONTAL_NORMAL_COSINE:
            ground_name = robot_model.get_geom_name(ground_geom_id)
            return _GroundContactSummary(
                deepest_distance,
                f"ground plane {ground_name!r} is not horizontal",
            )

        if deepest_distance is None or distance < deepest_distance:
            deepest_distance = distance

    return _GroundContactSummary(deepest_distance, None)


def project_qpos_above_flat_ground(
    robot_model,
    qpos,
    *,
    checker=None,
    penetration_tolerance=DEFAULT_GROUND_PENETRATION_TOLERANCE,
    max_lift=DEFAULT_MAX_AUTOMATIC_GROUND_LIFT,
):
    """Lift a single floating root just enough to clear a horizontal plane.

    The input array is never mutated.  A successful unchanged result means the
    pose was already above the hard-penetration tolerance.  A failed result
    keeps the original qpos and explains why automatic projection was unsafe or
    inapplicable.
    """
    qpos = np.asarray(qpos, dtype=float).copy()
    penetration_tolerance = max(0.0, float(penetration_tolerance))
    max_lift = max(0.0, float(max_lift))
    if qpos.shape != (int(robot_model.mj_model.nq),):
        return GroundProjectionResult(
            False,
            qpos,
            reason=(
                f"expected qpos width {int(robot_model.mj_model.nq)}, "
                f"found {qpos.size}"
            ),
        )
    if not np.all(np.isfinite(qpos)):
        return GroundProjectionResult(
            False, qpos, reason="qpos contains non-finite values"
        )

    checker = checker or CollisionChecker(robot_model)
    state = robot_model.create_state()
    state.set_qpos(qpos)
    get_blocking = getattr(checker, "get_blocking_collisions", None)
    if callable(get_blocking):
        blocking = get_blocking(state)
    else:
        blocking = [
            collision
            for collision in checker.get_collisions(state)
            if collision.blocking
        ]
    if any(collision.kind == "self" for collision in blocking):
        return GroundProjectionResult(
            False,
            qpos,
            reason="pose also contains a blocking self-collision",
        )

    summary = _ground_contact_summary(
        robot_model, state, penetration_tolerance
    )
    if summary.unsupported_reason is not None:
        return GroundProjectionResult(
            False, qpos, reason=summary.unsupported_reason
        )
    if summary.deepest_distance is None:
        if blocking:
            return GroundProjectionResult(
                False,
                qpos,
                reason=(
                    "blocking environment contact is not a correctable "
                    "horizontal-ground penetration"
                ),
            )
        return GroundProjectionResult(True, qpos, reason="pose clears the ground")

    free_joints = tuple(robot_model.free_joints_by_body.values())
    if len(free_joints) != 1:
        return GroundProjectionResult(
            False,
            qpos,
            penetration_depth=max(0.0, -float(summary.deepest_distance)),
            reason="automatic ground projection requires one floating root",
        )

    penetration_depth = max(0.0, -float(summary.deepest_distance))
    lift = penetration_depth + penetration_tolerance
    if lift > max_lift:
        return GroundProjectionResult(
            False,
            qpos,
            penetration_depth=penetration_depth,
            reason=(
                f"required {lift * 1000.0:.1f} mm lift exceeds the "
                f"{max_lift * 1000.0:.1f} mm automatic-repair limit"
            ),
        )

    free_joint = free_joints[0]
    projected = qpos.copy()
    projected[free_joint.qpos_address + 2] += lift
    state.set_qpos(projected)

    after = _ground_contact_summary(robot_model, state, penetration_tolerance)
    if after.unsupported_reason is not None or after.deepest_distance is not None:
        return GroundProjectionResult(
            False,
            qpos,
            penetration_depth=penetration_depth,
            reason=after.unsupported_reason or "ground projection did not clear the pose",
        )
    if callable(get_blocking):
        remaining_blocking = get_blocking(state)
    else:
        remaining_blocking = [
            collision
            for collision in checker.get_collisions(state)
            if collision.blocking
        ]
    if remaining_blocking:
        return GroundProjectionResult(
            False,
            qpos,
            penetration_depth=penetration_depth,
            reason="ground projection introduced or retained a blocking collision",
        )

    return GroundProjectionResult(
        True,
        state.get_qpos(),
        applied_offset=lift,
        penetration_depth=penetration_depth,
        reason=f"floating root raised {lift * 1000.0:.1f} mm",
    )

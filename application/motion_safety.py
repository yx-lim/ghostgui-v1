"""Explicit, reviewable repair candidates for unsafe qpos motion.

This service never edits an application document, publishes backend output, or
mutates its input arrays.  It deliberately implements only two deterministic
repairs:

* project individual poses above a flat horizontal ground plane;
* insert a lifted waypoint at an interior flat-ground penetration; and
* insert a local waypoint around an interior self-collision by perturbing one
  bounded scalar joint at a time.

The adaptive core validator remains the authority for accepting every proposed
path.  A successful changed result is therefore a candidate for user review,
not permission to silently replace authored motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ik import (
    CollisionChecker,
    adaptive_trajectory_collision_reports,
    format_collision_pairs,
    project_qpos_above_flat_ground,
)
from core.robotics import validate_trajectory_arrays
from core.trajectory import interpolate_qpos_manifold


@dataclass(frozen=True)
class SafeMotionRepairResult:
    """Outcome of constructing, but never applying, a safe-motion candidate."""

    success: bool
    qposes: tuple[np.ndarray, ...]
    times: tuple[float, ...]
    status: str
    ground_correction_count: int = 0
    detour_waypoint_count: int = 0
    blocking_report: object | None = None

    @property
    def changed(self):
        return bool(
            self.ground_correction_count or self.detour_waypoint_count
        )

    @property
    def requires_review(self):
        return self.success and self.changed


@dataclass(frozen=True)
class _DetourWaypoint:
    qpos: np.ndarray
    time: float
    ground_corrected: bool
    joint_name: str
    offset: float
    unit: str = "rad"


def _copied_qposes(qposes):
    return tuple(np.asarray(qpos, dtype=float).copy() for qpos in qposes)


def _result(
    success,
    qposes,
    times,
    status,
    *,
    ground_correction_count=0,
    detour_waypoint_count=0,
    blocking_report=None,
):
    return SafeMotionRepairResult(
        bool(success),
        _copied_qposes(qposes),
        tuple(float(time) for time in times),
        str(status),
        int(ground_correction_count),
        int(detour_waypoint_count),
        blocking_report,
    )


def _blocking_location(report):
    if report is None:
        return "an unknown path location"
    location = getattr(report, "location_label", None)
    if location:
        return str(location)
    time = getattr(report, "time", None)
    if time is not None:
        return f"{float(time):.6g} s"
    return f"sample {int(report.sample_index)}"


def _is_interior_self_collision(report):
    if report is None:
        return False
    fraction = getattr(report, "segment_fraction", None)
    segment_index = getattr(report, "segment_index", None)
    if segment_index is None or fraction is None:
        return False
    if not 0.0 < float(fraction) < 1.0:
        return False
    collisions = tuple(getattr(report, "collisions", ()))
    return bool(collisions) and all(
        collision.kind == "self" for collision in collisions
    )


def _is_interior_environment_collision(report):
    if report is None or not bool(getattr(report, "is_interior", False)):
        return False
    collisions = tuple(getattr(report, "collisions", ()))
    return bool(collisions) and all(
        collision.kind == "environment" for collision in collisions
    )


def _bounded_scalar_joints(robot_model):
    """Return editable scalar joints in a stable, model-independent order."""
    joints = []
    for name, joint in getattr(robot_model, "joints", {}).items():
        limits = getattr(joint, "limits", None)
        if limits is None:
            continue
        lo, hi = (float(limits[0]), float(limits[1]))
        if not np.isfinite((lo, hi)).all() or lo >= hi:
            continue
        joints.append((str(name), joint, lo, hi))
    return tuple(sorted(joints, key=lambda item: item[0]))


def _detour_offsets(step, maximum):
    magnitude = float(step)
    maximum = float(maximum)
    while magnitude <= maximum + 1e-12:
        # Positive is tried before negative so identical inputs always produce
        # the same review candidate.
        yield magnitude
        yield -magnitude
        magnitude += step


def _first_local_detour(
    robot_model,
    qposes,
    times,
    report,
    checker,
    *,
    detour_step,
    max_detour_offset,
):
    segment_index = int(report.segment_index)
    fraction = float(report.segment_fraction)
    start_qpos = qposes[segment_index]
    end_qpos = qposes[segment_index + 1]
    start_time = float(times[segment_index])
    end_time = float(times[segment_index + 1])
    waypoint_time = start_time + fraction * (end_time - start_time)
    if not start_time < waypoint_time < end_time:
        return None

    base_waypoint = interpolate_qpos_manifold(
        robot_model.mj_model, start_qpos, end_qpos, fraction
    )
    for joint_name, joint, lo, hi in _bounded_scalar_joints(robot_model):
        address = int(joint.qpos_address)
        authored_value = float(base_waypoint[address])
        for offset in _detour_offsets(detour_step, max_detour_offset):
            candidate_value = authored_value + offset
            if candidate_value < lo or candidate_value > hi:
                continue
            waypoint = base_waypoint.copy()
            waypoint[address] = candidate_value

            projection = project_qpos_above_flat_ground(
                robot_model, waypoint, checker=checker
            )
            if not projection.success:
                continue
            waypoint = projection.qpos

            # Validate each replacement interval independently.  This makes
            # the local search condition explicit; a collision-free waypoint
            # by itself is insufficient if either approach sweep collides.
            _left_warning, left_blocking = (
                adaptive_trajectory_collision_reports(
                    robot_model,
                    (start_qpos, waypoint),
                    times=(start_time, waypoint_time),
                    checker=checker,
                )
            )
            if left_blocking is not None:
                continue
            _right_warning, right_blocking = (
                adaptive_trajectory_collision_reports(
                    robot_model,
                    (waypoint, end_qpos),
                    times=(waypoint_time, end_time),
                    checker=checker,
                )
            )
            if right_blocking is not None:
                continue
            return _DetourWaypoint(
                waypoint.copy(),
                waypoint_time,
                bool(projection.changed),
                joint_name,
                float(waypoint[address] - authored_value),
            )
    return None


def _first_ground_waypoint(
    robot_model,
    qposes,
    times,
    report,
    checker,
):
    """Insert a lifted waypoint at an interior ground-penetration location."""
    segment_index = int(report.segment_index)
    start_qpos = qposes[segment_index]
    end_qpos = qposes[segment_index + 1]
    start_time = float(times[segment_index])
    end_time = float(times[segment_index + 1])
    free_joints = tuple(
        getattr(robot_model, "free_joints_by_body", {}).values()
    )
    if len(free_joints) != 1:
        return None
    z_address = int(free_joints[0].qpos_address) + 2
    fractions = []
    for fraction in (
        0.5,
        float(report.segment_fraction),
        0.25,
        0.75,
    ):
        if 0.0 < fraction < 1.0 and all(
            abs(fraction - existing) > 1e-9 for existing in fractions
        ):
            fractions.append(fraction)

    for fraction in fractions:
        waypoint_time = start_time + fraction * (end_time - start_time)
        base_waypoint = interpolate_qpos_manifold(
            robot_model.mj_model, start_qpos, end_qpos, fraction
        )
        projection = project_qpos_above_flat_ground(
            robot_model, base_waypoint, checker=checker
        )
        if not projection.success or not projection.changed:
            continue

        # Extra clearance can be necessary because both replacement interval
        # sweeps must clear the barrier, not only the selected waypoint.
        for extra_clearance in np.arange(0.0, 0.2501, 0.01):
            total_lift = (
                float(projection.applied_offset) + float(extra_clearance)
            )
            if total_lift > 0.25 + 1e-12:
                break
            waypoint = projection.qpos.copy()
            waypoint[z_address] += float(extra_clearance)
            _left_warning, left_blocking = (
                adaptive_trajectory_collision_reports(
                    robot_model,
                    (start_qpos, waypoint),
                    times=(start_time, waypoint_time),
                    checker=checker,
                )
            )
            if left_blocking is not None:
                continue
            _right_warning, right_blocking = (
                adaptive_trajectory_collision_reports(
                    robot_model,
                    (waypoint, end_qpos),
                    times=(waypoint_time, end_time),
                    checker=checker,
                )
            )
            if right_blocking is not None:
                continue
            return _DetourWaypoint(
                waypoint,
                waypoint_time,
                True,
                "floating root z",
                total_lift,
                "m",
            )
    return None


def propose_safe_motion_repair(
    robot_model,
    qposes,
    times,
    *,
    checker=None,
    max_detour_waypoints=4,
    detour_step=0.1,
    max_detour_offset=0.4,
):
    """Build an explicit safe-motion candidate without applying it.

    Ground penetration is handled first, per sampled pose. The complete path
    is then checked adaptively. An interior flat-ground contact may receive a
    lifted waypoint; an interior, self-only contact may receive a waypoint
    found by a deterministic coordinate stencil of bounded scalar Joint
    Angles. Both replacement intervals and the final complete path must pass
    adaptive validation.

    Invalid array/time inputs raise ``ValueError``.  Expected repair failures
    are returned as ``success=False`` with untouched copies of the original
    motion and an actionable status.
    """
    max_detour_waypoints = int(max_detour_waypoints)
    detour_step = float(detour_step)
    max_detour_offset = float(max_detour_offset)
    if max_detour_waypoints < 0:
        raise ValueError("max_detour_waypoints cannot be negative")
    if not np.isfinite(detour_step) or detour_step <= 0.0:
        raise ValueError("detour_step must be finite and positive")
    if (
        not np.isfinite(max_detour_offset)
        or max_detour_offset < detour_step
    ):
        raise ValueError(
            "max_detour_offset must be finite and at least detour_step"
        )

    normalized_times, normalized_qposes = validate_trajectory_arrays(
        times, qposes, int(robot_model.mj_model.nq)
    )
    original_qposes = _copied_qposes(normalized_qposes)
    original_times = tuple(normalized_times)
    for sample_index, qpos in enumerate(original_qposes):
        for joint_name, joint, lo, hi in _bounded_scalar_joints(robot_model):
            value = float(qpos[int(joint.qpos_address)])
            if value < lo - 1e-9 or value > hi + 1e-9:
                raise ValueError(
                    f"motion qpos row {sample_index + 1} has Joint Angle "
                    f"{joint_name}={value:.6g} outside "
                    f"[{lo:.6g}, {hi:.6g}]"
                )
    if not original_qposes:
        return _result(
            True,
            original_qposes,
            original_times,
            "Motion is empty; no repair candidate was needed.",
        )

    checker = checker or CollisionChecker(robot_model)
    candidate_qposes = []
    ground_correction_count = 0
    for sample_index, qpos in enumerate(original_qposes):
        projection = project_qpos_above_flat_ground(
            robot_model, qpos, checker=checker
        )
        if not projection.success:
            _warning, blocking = adaptive_trajectory_collision_reports(
                robot_model,
                original_qposes,
                times=original_times,
                checker=checker,
            )
            return _result(
                False,
                original_qposes,
                original_times,
                "No safe repair candidate: sampled pose "
                f"{sample_index} could not pass the flat-ground projection "
                f"({projection.reason}). The input motion was not modified.",
                blocking_report=blocking,
            )
        candidate_qposes.append(projection.qpos.copy())
        if projection.changed:
            ground_correction_count += 1
    candidate_times = list(original_times)

    _warning, blocking = adaptive_trajectory_collision_reports(
        robot_model,
        tuple(candidate_qposes),
        times=tuple(candidate_times),
        checker=checker,
    )
    if blocking is None:
        if ground_correction_count:
            return _result(
                True,
                candidate_qposes,
                candidate_times,
                "Review candidate: corrected flat-ground penetration in "
                f"{ground_correction_count} sampled pose(s), and adaptive "
                "validation found no blocking collision. The input motion "
                "was not modified or published.",
                ground_correction_count=ground_correction_count,
            )
        return _result(
            True,
            candidate_qposes,
            candidate_times,
            "Motion already passes adaptive blocking-collision validation; "
            "no repair candidate was needed.",
        )

    detour_waypoint_count = 0
    detour_notes = []
    while blocking is not None:
        is_self_detour = _is_interior_self_collision(blocking)
        is_ground_detour = _is_interior_environment_collision(blocking)
        if not (is_self_detour or is_ground_detour):
            kinds = sorted({
                collision.kind for collision in blocking.collisions
            })
            kind_label = "/".join(kinds) or "unknown"
            return _result(
                False,
                original_qposes,
                original_times,
                "No safe repair candidate: blocking "
                f"{kind_label} contact at {_blocking_location(blocking)} "
                f"({format_collision_pairs(blocking.collisions)}). Local "
                "detours are limited to interval-interior self-collisions "
                "and flat-ground penetration; "
                "the input motion was not modified.",
                blocking_report=blocking,
            )
        if detour_waypoint_count >= max_detour_waypoints:
            return _result(
                False,
                original_qposes,
                original_times,
                "No safe repair candidate: the deterministic local search "
                f"reached its {max_detour_waypoints}-waypoint limit while a "
                f"self-collision remained at {_blocking_location(blocking)}. "
                "The input motion was not modified.",
                blocking_report=blocking,
            )

        if is_ground_detour:
            detour = _first_ground_waypoint(
                robot_model,
                candidate_qposes,
                candidate_times,
                blocking,
                checker,
            )
        else:
            detour = _first_local_detour(
                robot_model,
                candidate_qposes,
                candidate_times,
                blocking,
                checker,
                detour_step=detour_step,
                max_detour_offset=max_detour_offset,
            )
        if detour is None:
            return _result(
                False,
                original_qposes,
                original_times,
                "No safe repair candidate: the bounded local search could "
                f"not reroute the collision at "
                f"{_blocking_location(blocking)}. Add or edit a Keyframe; "
                "the input motion was not modified.",
                blocking_report=blocking,
            )

        insert_index = int(blocking.segment_index) + 1
        candidate_qposes.insert(insert_index, detour.qpos.copy())
        candidate_times.insert(insert_index, float(detour.time))
        detour_waypoint_count += 1
        if detour.ground_corrected:
            ground_correction_count += 1
        detour_notes.append(
            f"{detour.joint_name} {detour.offset:+.3f} {detour.unit}"
        )

        # The full candidate is rechecked after every insertion.  Passing the
        # two local intervals does not excuse a pre-existing collision in some
        # other segment.
        _warning, blocking = adaptive_trajectory_collision_reports(
            robot_model,
            tuple(candidate_qposes),
            times=tuple(candidate_times),
            checker=checker,
        )

    correction_note = (
        f" and corrected {ground_correction_count} flat-ground pose(s)"
        if ground_correction_count else ""
    )
    return _result(
        True,
        candidate_qposes,
        candidate_times,
        "Review candidate: inserted "
        f"{detour_waypoint_count} local detour waypoint(s){correction_note}; "
        f"perturbations: {', '.join(detour_notes)}. Both replacement "
        "intervals and the complete path pass adaptive blocking-collision "
        "validation. The input motion was not modified or published.",
        ground_correction_count=ground_correction_count,
        detour_waypoint_count=detour_waypoint_count,
    )


__all__ = ["SafeMotionRepairResult", "propose_safe_motion_repair"]

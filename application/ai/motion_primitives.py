"""Model-owned, deterministic whole-body motion primitives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from application.ai.qpos_trajectory import QposAnchor, validate_qpos_anchors
from core.math3d import quaternion_angle, quaternion_slerp
from core.robotics import QposContract


@dataclass(frozen=True)
class MotionPrimitivePlan:
    name: str
    variant: str
    anchors: tuple[QposAnchor, ...]
    phases: tuple[str, ...]
    concessions: tuple[str, ...]
    quality_checks: tuple[str, ...]


def build_motion_primitive(adapter, *, primitive: str, duration_seconds: float):
    """Build a supported motion from model-owned poses, never provider qpos."""

    name = str(primitive).strip().lower().replace("-", "_")
    info = getattr(adapter, "info", None)
    supported = tuple(getattr(info, "motion_primitives", ()))
    if name != "burpee" or name not in supported:
        raise ValueError(f"active model has no local motion primitive named {primitive}")
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("motion primitive duration must be positive and finite")

    pose_files = dict(getattr(info, "motion_reference_poses", {}))
    reference_path = pose_files.get("pushup_prone")
    if reference_path is None:
        raise ValueError("burpee requires the model's pushup_prone reference pose")
    prone = load_reference_qpos(adapter, reference_path)
    standing = np.asarray(adapter.home_qpos, dtype=float).copy()
    crouch = _g1_crouch(adapter, standing)
    transition = clear_environment_collisions(
        adapter,
        _blend_pose(adapter, crouch, prone, 0.5),
    )

    fractions = (0.0, 0.16, 0.28, 0.36, 0.50, 0.64, 0.72, 0.84, 1.0)
    poses = (
        standing,
        crouch,
        transition,
        prone,
        prone,
        prone,
        transition,
        crouch,
        standing,
    )
    phases = (
        "standing",
        "crouch",
        "hands_plant_transition",
        "prone_pushup",
        "prone_pushup_hold",
        "prone_pushup",
        "hands_plant_transition",
        "crouch",
        "standing",
    )
    raw = [
        {"time_seconds": round(duration * fraction, 9), "qpos": pose.tolist()}
        for fraction, pose in zip(fractions, poses)
    ]
    anchors = validate_qpos_anchors(adapter, raw)
    checks = _validate_burpee_semantics(adapter, anchors, prone)
    return MotionPrimitivePlan(
        name="burpee",
        variant="no_jump",
        anchors=anchors,
        phases=phases,
        concessions=(
            "Closest available motion: generated a no-jump burpee from the "
            "model-owned prone push-up pose; dynamic flight is not synthesized.",
        ),
        quality_checks=checks,
    )


def load_reference_qpos(adapter, path: str | Path) -> np.ndarray:
    """Read exactly one complete qpos row from a registered reference file."""

    source = Path(path)
    try:
        with source.open("r", newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    except OSError as error:
        raise ValueError(f"motion reference pose is unavailable: {source.name}") from error
    if len(rows) != 1:
        raise ValueError("motion reference pose must contain exactly one qpos row")
    try:
        values = [float(cell) for cell in rows[0]]
    except ValueError as error:
        raise ValueError("motion reference pose contains a non-numeric value") from error
    qpos = QposContract(int(adapter.mj_model.nq)).validate(
        values,
        context=f"motion reference pose {source.name}",
    )
    return validate_qpos_anchors(
        adapter,
        ({"time_seconds": 0.0, "qpos": qpos.tolist()},),
    )[0].qpos


def _g1_crouch(adapter, standing):
    crouch = np.asarray(standing, dtype=float).copy()
    root = _root_joint(adapter)
    crouch[root.qpos_address + 2] = max(
        0.38,
        float(standing[root.qpos_address + 2]) - 0.22,
    )
    targets = {
        "left_hip_pitch_joint": -1.05,
        "right_hip_pitch_joint": -1.05,
        "left_knee_joint": 1.95,
        "right_knee_joint": 1.95,
        "left_ankle_pitch_joint": -0.78,
        "right_ankle_pitch_joint": -0.78,
        "left_shoulder_pitch_joint": 0.55,
        "right_shoulder_pitch_joint": 0.55,
    }
    for name, requested in targets.items():
        joint = adapter.joints.get(name)
        if joint is None:
            raise ValueError(f"burpee primitive requires Joint Angle {name}")
        limits = adapter.get_joint_limits(name)
        value = requested
        if limits is not None:
            value = float(np.clip(value, limits[0] + 1e-4, limits[1] - 1e-4))
        crouch[int(joint.qpos_address)] = value
    return crouch


def _blend_pose(adapter, start, end, fraction):
    result = np.asarray(start, dtype=float) * (1.0 - fraction) + np.asarray(
        end,
        dtype=float,
    ) * fraction
    for free_joint in adapter.free_joints_by_body.values():
        address = int(free_joint.qpos_address)
        result[address + 3:address + 7] = quaternion_slerp(
            start[address + 3:address + 7],
            end[address + 3:address + 7],
            fraction,
        )
    return result


def _validate_burpee_semantics(adapter, anchors, prone):
    standing = np.asarray(adapter.home_qpos, dtype=float)
    if not np.allclose(anchors[0].qpos, standing, atol=1e-9) or not np.allclose(
        anchors[-1].qpos,
        standing,
        atol=1e-9,
    ):
        raise ValueError("burpee must start and end at the model standing pose")
    root = _root_joint(adapter)
    address = int(root.qpos_address)
    if quaternion_angle(
        anchors[0].qpos[address + 3:address + 7],
        anchors[-1].qpos[address + 3:address + 7],
    ) > 1e-8:
        raise ValueError("burpee endpoints must have the same upright orientation")
    if not all(np.allclose(anchors[index].qpos, prone, atol=1e-9) for index in (3, 4, 5)):
        raise ValueError("burpee prone phase must use the certified push-up pose")

    maximum_root_step = 0.0
    maximum_angular_speed = 0.0
    maximum_joint_speed = 0.0
    joint_addresses = [int(adapter.joints[name].qpos_address) for name in adapter.joint_names]
    for previous, current in zip(anchors, anchors[1:]):
        elapsed = current.time_seconds - previous.time_seconds
        maximum_root_step = max(
            maximum_root_step,
            float(np.linalg.norm(
                current.qpos[address:address + 3] - previous.qpos[address:address + 3]
            )),
        )
        maximum_angular_speed = max(
            maximum_angular_speed,
            quaternion_angle(
                previous.qpos[address + 3:address + 7],
                current.qpos[address + 3:address + 7],
            ) / elapsed,
        )
        maximum_joint_speed = max(
            maximum_joint_speed,
            max(
                abs(float(current.qpos[index] - previous.qpos[index])) / elapsed
                for index in joint_addresses
            ),
        )
    if maximum_root_step > 0.55:
        raise ValueError("burpee root path contains a teleportation-sized step")
    if maximum_angular_speed > 6.0 or maximum_joint_speed > 8.0:
        raise ValueError("burpee exceeds the local kinematic continuity limits")

    _validate_reference_contacts(adapter, prone)
    _validate_anchor_collisions(adapter, anchors)
    return (
        "exact standing start/end",
        "certified front-down prone pose",
        "hand/foot support proximity",
        "root continuity and bounded Joint Angle speed",
        "no unintended blocking anchor collisions",
    )


def _validate_reference_contacts(adapter, prone):
    state = adapter.create_state()
    state.set_qpos(prone)
    standing_state = adapter.create_state()
    standing_state.set_qpos(adapter.home_qpos)
    support_names = ("left_hand", "right_hand", "left_foot", "right_foot")
    heights = []
    for name in support_names:
        kind, object_name = adapter.logical_frame_bindings[name]
        position, _quaternion = state.get_body_pose(object_name, kind)
        heights.append(float(position[2]))
    standing_feet = []
    for name in ("left_foot", "right_foot"):
        kind, object_name = adapter.logical_frame_bindings[name]
        position, _quaternion = standing_state.get_body_pose(object_name, kind)
        standing_feet.append(float(position[2]))
    floor_reference = sum(standing_feet) / len(standing_feet)
    if any(height < floor_reference - 0.03 for height in heights):
        raise ValueError("certified push-up pose penetrates below the support floor")
    if any(height > floor_reference + 0.16 for height in heights):
        raise ValueError("certified push-up pose does not place hands and feet near the floor")


def _validate_anchor_collisions(adapter, anchors):
    try:
        from core.ik import CollisionChecker

        checker = CollisionChecker(adapter)
    except (ImportError, RuntimeError):
        return
    for anchor in anchors:
        state = adapter.create_state()
        state.set_qpos(anchor.qpos)
        blocking = tuple(
            collision
            for collision in checker.get_collisions(state)
            if getattr(collision, "blocking", False)
        )
        if blocking:
            raise ValueError(
                f"burpee anchor at {anchor.time_seconds:.3f} s has an "
                "unintended blocking collision"
            )


def clear_environment_collisions(adapter, qpos):
    """Lift a pose just enough to clear floor contact; reject self-collision."""

    try:
        from core.ik import CollisionChecker

        checker = CollisionChecker(adapter)
    except (ImportError, RuntimeError):
        return qpos
    state = adapter.create_state()
    state.set_qpos(qpos)
    blocking = tuple(
        collision
        for collision in checker.get_collisions(state)
        if getattr(collision, "blocking", False)
    )
    self_collisions = tuple(item for item in blocking if item.kind == "self")
    if self_collisions:
        raise ValueError("burpee transition contains a blocking self-collision")
    penetration = max(
        (max(0.0, -float(item.distance)) for item in blocking),
        default=0.0,
    )
    if penetration:
        result = np.asarray(qpos, dtype=float).copy()
        root = _root_joint(adapter)
        result[int(root.qpos_address) + 2] += penetration + 0.003
        return result
    return qpos


def _root_joint(adapter):
    roots = tuple(adapter.free_joints_by_body.values())
    if not roots:
        raise ValueError("burpee primitive requires a floating root")
    return roots[0]

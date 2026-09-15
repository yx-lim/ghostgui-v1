"""Model-aware validation for provider-authored sparse qpos anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.robotics import QposContract


MIN_QUATERNION_NORM = 1e-8


@dataclass(frozen=True)
class QposAnchor:
    time_seconds: float
    qpos: np.ndarray


def validate_qpos_anchors(adapter, keyframes) -> tuple[QposAnchor, ...]:
    """Validate complete model states and normalize floating-root quaternions."""

    width = int(adapter.mj_model.nq)
    contract = QposContract(width)
    anchors = []
    for index, keyframe in enumerate(keyframes):
        time = float(keyframe["time_seconds"])
        qpos = contract.validate(
            keyframe["qpos"],
            context=f"qpos Keyframe {index + 1}",
        )
        _normalize_free_joint_quaternions(adapter, qpos, index=index)
        _validate_joint_limits(adapter, qpos, time_seconds=time)
        anchors.append(QposAnchor(time, qpos))
    return tuple(anchors)


def normalize_qpos_quaternions(adapter, values, *, context="qpos") -> np.ndarray:
    """Return one complete qpos with every free-joint quaternion normalized."""

    qpos = QposContract(int(adapter.mj_model.nq)).validate(values, context=context)
    _normalize_free_joint_quaternions(adapter, qpos, index=None)
    return qpos


def _normalize_free_joint_quaternions(adapter, qpos, *, index):
    width = len(qpos)
    for free_joint in adapter.free_joints_by_body.values():
        address = int(free_joint.qpos_address)
        if address < 0 or address + 7 > width:
            raise ValueError("floating-root qpos layout is invalid")
        quaternion = np.asarray(qpos[address + 3:address + 7], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm < MIN_QUATERNION_NORM:
            label = "qpos" if index is None else f"qpos Keyframe {index + 1}"
            raise ValueError(f"{label} has an invalid floating-root quaternion")
        qpos[address + 3:address + 7] = quaternion / norm


def _validate_joint_limits(adapter, qpos, *, time_seconds):
    plain_name = getattr(adapter, "plain_name", lambda value: value)
    for name in adapter.joint_names:
        limits = adapter.get_joint_limits(name)
        if limits is None:
            continue
        joint = adapter.joints.get(plain_name(name))
        if joint is None:
            raise ValueError(f"active model is missing Joint Angle metadata for {name}")
        address = int(joint.qpos_address)
        if address < 0 or address >= len(qpos):
            raise ValueError(f"Joint Angle qpos layout is invalid for {name}")
        value = float(qpos[address])
        if value < float(limits[0]) - 1e-9 or value > float(limits[1]) + 1e-9:
            raise ValueError(
                f"Joint Angle {name} at {time_seconds:.3f} s is outside its model limits"
            )

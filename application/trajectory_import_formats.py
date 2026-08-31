"""Application services for target-specific trajectory import formats."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from application.csv_io import LoadedTrajectory
from application.trajectory_export_formats import (
    G1_JOINT_ORDER,
    MJLAB_OUTPUT_WIDTH,
    mjlab_compatibility_error,
)
from core.robotics import validate_trajectory_arrays


def read_dsms_trajectory(
    directory,
    expected_qpos_count,
    *,
    expected_dof=None,
):
    """Read a DSMS reference folder as a canonical timed qpos trajectory."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("expected a DSMS reference folder")

    time_path = path / "time.csv"
    if not time_path.is_file():
        raise ValueError("DSMS folder does not contain time.csv")

    qpos_paths = sorted(path.glob("qpos_*dof.csv"))
    if not qpos_paths:
        raise ValueError("DSMS folder does not contain qpos_<dof>dof.csv")
    if len(qpos_paths) != 1:
        raise ValueError(
            "DSMS folder must contain exactly one qpos_<dof>dof.csv file"
        )
    qpos_path = qpos_paths[0]
    if expected_dof is not None:
        expected_name = f"qpos_{int(expected_dof)}dof.csv"
        if qpos_path.name != expected_name:
            raise ValueError(
                f"expected {expected_name} for the active model, found "
                f"{qpos_path.name}"
            )

    try:
        times = np.loadtxt(time_path, delimiter=",", ndmin=1)
        qposes = np.loadtxt(qpos_path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read DSMS numeric CSV data: {exc}") from exc

    expected = int(expected_qpos_count)
    if qposes.shape[1] != expected:
        raise ValueError(
            f"expected {expected} qpos values per DSMS row, "
            f"found {qposes.shape[1]}"
        )
    if len(times) != len(qposes):
        raise ValueError(
            f"DSMS sample count mismatch: time.csv has {len(times)} rows, "
            f"but {qpos_path.name} has {len(qposes)} rows"
        )

    normalized_times, normalized_qposes = validate_trajectory_arrays(
        times,
        qposes,
        expected,
    )
    if not normalized_times:
        raise ValueError("DSMS trajectory contains no samples")
    return LoadedTrajectory(
        path=path,
        times=normalized_times,
        qposes=normalized_qposes,
        source_format="dsms",
    )


def read_mjlab_trajectory(csv_path, adapter, sample_interval):
    """Convert a headerless G1 mjlab CSV into canonical MuJoCo qpos rows."""
    error = mjlab_compatibility_error(adapter)
    if error:
        raise ValueError(error)

    dt = float(sample_interval)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("mjlab sample interval must be positive and finite")

    path = Path(csv_path).expanduser().resolve()
    try:
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read mjlab numeric CSV data: {exc}") from exc
    if rows.size == 0:
        raise ValueError("mjlab trajectory contains no samples")
    if rows.shape[1] != MJLAB_OUTPUT_WIDTH:
        raise ValueError(
            f"expected {MJLAB_OUTPUT_WIDTH} mjlab values per row, "
            f"found {rows.shape[1]}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("mjlab trajectory contains a non-finite value")

    expected = int(adapter.mj_model.nq)
    free_joint = next(iter(adapter.free_joints_by_body.values()))
    base_address = int(free_joint.qpos_address)
    qposes = []
    for row_number, row in enumerate(rows, start=1):
        qpos = np.zeros(expected, dtype=float)
        qpos[base_address:base_address + 3] = row[:3]

        qx, qy, qz, qw = map(float, row[3:7])
        quaternion = np.asarray((qw, qx, qy, qz), dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError(
                f"mjlab row {row_number} contains a zero-length base quaternion"
            )
        qpos[base_address + 3:base_address + 7] = quaternion / norm

        for index, joint_name in enumerate(G1_JOINT_ORDER):
            address = int(adapter.joints[joint_name].qpos_address)
            qpos[address] = float(row[7 + index])
        qposes.append(qpos)

    times = tuple(index * dt for index in range(len(qposes)))
    normalized_times, normalized_qposes = validate_trajectory_arrays(
        times,
        qposes,
        expected,
    )
    return LoadedTrajectory(
        path=path,
        times=normalized_times,
        qposes=normalized_qposes,
        source_format="mjlab",
    )

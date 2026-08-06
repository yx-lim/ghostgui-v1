"""Application services for target-specific trajectory export formats."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from application.csv_io import TrajectoryExport
from application.paths import atomic_text_writer
from core.models import RobotStateTimeline
from core.robotics import validate_trajectory_arrays


G1_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

MJLAB_INPUT_WIDTH = 1 + 7 + len(G1_JOINT_ORDER)
MJLAB_OUTPUT_WIDTH = 7 + len(G1_JOINT_ORDER)


@dataclass(frozen=True)
class DSMSPreparedTrajectory:
    times: np.ndarray
    qposes: np.ndarray
    median_dt: float | None
    is_uniform: bool


@dataclass(frozen=True)
class TrajectoryFormatExportResult:
    paths: tuple[Path, ...]
    sample_count: int
    input_fps: float | None = None


def resample_trajectory_export(export, robot_model, dt):
    """Sample a time/qpos export uniformly on the MuJoCo position manifold."""
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("export interval must be a positive finite value")

    times, qposes = validate_trajectory_arrays(
        export.times,
        export.qposes,
        export.expected_qpos_count,
    )
    if not times:
        raise ValueError("trajectory contains no samples")
    if len(times) == 1:
        return TrajectoryExport(
            expected_qpos_count=export.expected_qpos_count,
            times=times,
            qposes=qposes,
            source_name=export.source_name,
            preview_active=export.preview_active,
        )

    timeline = RobotStateTimeline(
        robot_model,
        initial_time=times[0],
        initial_qpos=qposes[0],
    )
    for time, qpos in zip(times[1:], qposes[1:]):
        timeline.set_state(time, qpos)

    start_time = float(times[0])
    end_time = float(times[-1])
    step_count = int(math.floor(((end_time - start_time) / dt) + 1e-9))
    sampled_times = tuple(
        round(start_time + index * dt, 10)
        for index in range(step_count + 1)
        if start_time + index * dt <= end_time + 1e-9
    )
    sampled_qposes = tuple(
        timeline.sample_state(time) for time in sampled_times
    )
    return TrajectoryExport(
        expected_qpos_count=export.expected_qpos_count,
        times=sampled_times,
        qposes=sampled_qposes,
        source_name=export.source_name,
        preview_active=export.preview_active,
    )


def prepare_dsms_arrays(
    times,
    qposes,
    *,
    expected_qpos_count,
    allow_nonuniform_time=False,
    normalize_quaternion=True,
    base_qpos_address=0,
):
    """Validate and prepare time/qpos arrays for a DSMS reference folder."""
    expected = int(expected_qpos_count)
    normalized_times, normalized_qposes = validate_trajectory_arrays(
        times,
        qposes,
        expected,
    )
    if not normalized_times:
        raise ValueError("trajectory contains no samples")

    time_array = np.asarray(normalized_times, dtype=float)
    qpos_array = np.asarray(normalized_qposes, dtype=float)
    median_dt = None
    is_uniform = True

    if len(time_array) > 1:
        dt = np.diff(time_array)
        if np.any(dt <= 0.0):
            index = int(np.flatnonzero(dt <= 0.0)[0])
            raise ValueError(
                "Time must be strictly increasing. "
                f"Rows {index + 1} and {index + 2} have a non-positive interval."
            )
        median_dt = float(np.median(dt))
        is_uniform = bool(
            np.allclose(dt, median_dt, rtol=1e-4, atol=1e-9)
        )
        if not is_uniform and not allow_nonuniform_time:
            maximum_error = float(np.max(np.abs(dt - median_dt)))
            raise ValueError(
                "The timestamps are not uniformly sampled. "
                "Generate the trajectory using Export interval before DSMS "
                f"export. Median dt={median_dt:.9g} s; maximum deviation="
                f"{maximum_error:.9g} s. The standalone converter can accept "
                "the timestamps with --allow-nonuniform-time."
            )

    if normalize_quaternion:
        address = int(base_qpos_address)
        quaternion_start = address + 3
        quaternion_end = address + 7
        if address < 0 or quaternion_end > expected:
            raise ValueError(
                "base quaternion normalization requires seven free-root qpos "
                "values"
            )
        quaternion = qpos_array[:, quaternion_start:quaternion_end]
        norms = np.linalg.norm(quaternion, axis=1)
        if np.any(norms < 1e-12):
            index = int(np.flatnonzero(norms < 1e-12)[0])
            raise ValueError(
                f"Frame {index + 1} contains a zero-length base quaternion."
            )
        qpos_array[:, quaternion_start:quaternion_end] = (
            quaternion / norms[:, None]
        )

    return DSMSPreparedTrajectory(
        times=time_array,
        qposes=qpos_array,
        median_dt=median_dt,
        is_uniform=is_uniform,
    )


def write_dsms_files(output_dir, times, qposes, dof):
    """Write the two headerless files in a DSMS reference folder."""
    dof = int(dof)
    if dof <= 0:
        raise ValueError("DSMS joint DoF must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qpos_path = output_dir / f"qpos_{dof}dof.csv"
    time_path = output_dir / "time.csv"

    with atomic_text_writer(qpos_path, newline="") as handle:
        np.savetxt(handle, np.asarray(qposes), delimiter=",", fmt="%.18e")
    with atomic_text_writer(time_path, newline="") as handle:
        np.savetxt(handle, np.asarray(times), delimiter=",", fmt="%.9f")
    return qpos_path, time_path


def export_dsms_trajectory(
    output_dir,
    export,
    *,
    dof,
    base_qpos_address=None,
):
    prepared = prepare_dsms_arrays(
        export.times,
        export.qposes,
        expected_qpos_count=export.expected_qpos_count,
        normalize_quaternion=base_qpos_address is not None,
        base_qpos_address=base_qpos_address or 0,
    )
    paths = write_dsms_files(
        output_dir,
        prepared.times,
        prepared.qposes,
        dof,
    )
    return TrajectoryFormatExportResult(
        paths=tuple(paths),
        sample_count=len(prepared.times),
        input_fps=(
            None
            if prepared.median_dt is None
            else 1.0 / prepared.median_dt
        ),
    )


def convert_mjlab_rows(
    ghostgui_rows: Sequence[Sequence[float]],
    normalize_quaternions: bool,
):
    """Convert canonical time/xyz/wxyz/joints rows into mjlab rows."""
    output_rows: list[list[float]] = []
    non_unit_quaternion_count = 0

    for row_number, row in enumerate(ghostgui_rows, start=1):
        if len(row) != MJLAB_INPUT_WIDTH:
            raise ValueError(
                f"Row {row_number}: expected {MJLAB_INPUT_WIDTH} canonical "
                f"GhostGUI values, found {len(row)}."
            )
        if not all(math.isfinite(float(value)) for value in row):
            raise ValueError(f"Row {row_number}: contains a non-finite value.")

        base_xyz = list(map(float, row[1:4]))
        qw, qx, qy, qz = map(float, row[4:8])
        quaternion_xyzw = [qx, qy, qz, qw]
        norm = math.sqrt(
            sum(component * component for component in quaternion_xyzw)
        )
        if norm < 1e-12:
            raise ValueError(f"Row {row_number}: quaternion has zero magnitude.")
        if abs(norm - 1.0) > 1e-3:
            non_unit_quaternion_count += 1
        if normalize_quaternions:
            quaternion_xyzw = [component / norm for component in quaternion_xyzw]

        joints = list(map(float, row[8:]))
        output_row = base_xyz + quaternion_xyzw + joints
        if len(output_row) != MJLAB_OUTPUT_WIDTH:
            raise RuntimeError("Internal mjlab conversion width check failed.")
        output_rows.append(output_row)

    return output_rows, non_unit_quaternion_count


def write_mjlab_rows(path, rows):
    """Write headerless mjlab input rows using the CLI's numeric format."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path = path.resolve()
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            writer.writerow([f"{float(value):.10g}" for value in row])
    return path


def mjlab_compatibility_error(adapter):
    """Return why an adapter cannot produce the G1 29-DoF mjlab format."""
    expected_nq = 7 + len(G1_JOINT_ORDER)
    actual_nq = int(adapter.mj_model.nq)
    if actual_nq != expected_nq:
        return (
            f"mjlab export requires G1 29-DoF with nq={expected_nq}; "
            f"the active model has nq={actual_nq}"
        )
    available = set(adapter.joints)
    missing = [name for name in G1_JOINT_ORDER if name not in available]
    extra = [name for name in adapter.actuated_joints if name not in G1_JOINT_ORDER]
    if missing or extra:
        return "mjlab export requires the Unitree G1 29-DoF joint contract"
    free_joints = tuple(adapter.free_joints_by_body.values())
    if len(free_joints) != 1:
        return "mjlab export requires exactly one floating-base free joint"
    return None


def _uniform_input_fps(times, *, single_sample_dt=None):
    if len(times) == 1:
        if single_sample_dt is None:
            raise ValueError(
                "mjlab export needs two samples to infer input frequency"
            )
        dt = float(single_sample_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("mjlab input interval must be positive and finite")
        return 1.0 / dt

    deltas = np.diff(np.asarray(times, dtype=float))
    if np.any(deltas <= 0.0):
        raise ValueError("mjlab trajectory times must be strictly increasing")
    median_dt = float(np.median(deltas))
    max_relative_jitter = float(
        np.max(np.abs(deltas - median_dt) / median_dt)
    )
    if max_relative_jitter > 0.02:
        raise ValueError(
            "The timestamps are not uniformly sampled. Generate the trajectory "
            "using Export interval before mjlab export. Maximum relative "
            f"deviation={max_relative_jitter:.2%}."
        )
    return 1.0 / median_dt


def export_mjlab_trajectory(
    csv_path,
    export,
    adapter,
    *,
    single_sample_dt=None,
    normalize_quaternions=True,
):
    error = mjlab_compatibility_error(adapter)
    if error:
        raise ValueError(error)
    times, qposes = validate_trajectory_arrays(
        export.times,
        export.qposes,
        export.expected_qpos_count,
    )
    if not times:
        raise ValueError("trajectory contains no samples")
    input_fps = _uniform_input_fps(
        times,
        single_sample_dt=single_sample_dt,
    )

    free_joint = next(iter(adapter.free_joints_by_body.values()))
    base_address = int(free_joint.qpos_address)
    canonical_rows = []
    for time_value, qpos in zip(times, qposes):
        base_qpos = qpos[base_address:base_address + 7]
        if base_qpos.shape != (7,):
            raise ValueError("mjlab export could not read floating-base qpos")
        joints = [
            float(qpos[adapter.joints[name].qpos_address])
            for name in G1_JOINT_ORDER
        ]
        canonical_rows.append(
            [float(time_value)] + list(map(float, base_qpos)) + joints
        )

    rows, _non_unit_count = convert_mjlab_rows(
        canonical_rows,
        normalize_quaternions=normalize_quaternions,
    )
    path = write_mjlab_rows(csv_path, rows)
    return TrajectoryFormatExportResult(
        paths=(path,),
        sample_count=len(rows),
        input_fps=input_fps,
    )

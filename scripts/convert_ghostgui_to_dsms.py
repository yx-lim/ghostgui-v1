#!/usr/bin/env python3
"""
Convert a GhostGUI trajectory CSV into the reference-folder format expected by
shooting-for-contact / DSMS.

GhostGUI trajectory input:
    time,qpos_0,qpos_1,...,qpos_(nq-1)

DSMS output folder:
    qpos_<dof>dof.csv   # qpos only, no header
    time.csv            # time only, no header

For Unitree G1 29-DoF:
    nq = 36
    input columns = 1 + 36 = 37
    output pose file = qpos_29dof.csv

python scripts/convert_ghostgui_to_dsms.py \
  csv/trajectory/experiment/exp_stand_50f.csv \
  csv/exp_stand_50f \
  --dof 29 \
  --nq 36 \
  --allow-nonuniform-time
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _row_is_numeric(row: list[str]) -> bool:
    """Return True when every field in a CSV row is numeric."""
    if not row:
        return False

    try:
        for value in row:
            if value.strip() == "":
                return False
            float(value)
    except ValueError:
        return False

    return True


def _detect_header(path: Path, delimiter: str) -> bool:
    """Detect whether the first non-empty CSV row contains text labels."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                return not _row_is_numeric(row)

    raise ValueError(f"{path} is empty.")


def _load_csv(path: Path, delimiter: str) -> tuple[np.ndarray, bool]:
    """Load a numeric CSV, automatically skipping one header row if present."""
    has_header = _detect_header(path, delimiter)

    try:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=1 if has_header else 0,
            dtype=float,
            ndmin=2,
        )
    except ValueError as error:
        raise ValueError(
            f"Could not parse {path} as a numeric CSV"
            f"{' after its header row' if has_header else ''}: {error}"
        ) from error

    if data.size == 0 or data.shape[0] == 0:
        raise ValueError(f"{path} contains no trajectory samples.")

    return data, has_header


def convert(
    input_csv: Path,
    output_dir: Path,
    dof: int,
    nq: int,
    delimiter: str,
    allow_nonuniform_time: bool,
    normalize_quaternion: bool,
) -> tuple[Path, Path]:
    """Split a GhostGUI time-plus-qpos CSV into DSMS qpos and time files."""
    if dof <= 0:
        raise ValueError("--dof must be positive.")
    if nq <= 0:
        raise ValueError("--nq must be positive.")

    data, had_header = _load_csv(input_csv, delimiter)

    expected_columns = nq + 1
    if data.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns: one time column plus "
            f"{nq} qpos columns. Found {data.shape[1]} columns."
        )

    if not np.all(np.isfinite(data)):
        bad = np.argwhere(~np.isfinite(data))[0]
        raise ValueError(
            f"Input contains NaN or infinity at data row {bad[0] + 1}, "
            f"column {bad[1] + 1}."
        )

    time = data[:, 0].copy()
    qpos = data[:, 1:].copy()

    if np.any(time < 0):
        index = int(np.flatnonzero(time < 0)[0])
        raise ValueError(
            f"Time must be non-negative; row {index + 1} has {time[index]}."
        )

    median_dt = None
    is_uniform = True

    if len(time) > 1:
        dt = np.diff(time)

        if np.any(dt <= 0):
            index = int(np.flatnonzero(dt <= 0)[0])
            raise ValueError(
                "Time must be strictly increasing. "
                f"Rows {index + 1} and {index + 2} have a non-positive interval."
            )

        median_dt = float(np.median(dt))
        is_uniform = np.allclose(dt, median_dt, rtol=1e-4, atol=1e-9)

        if not is_uniform and not allow_nonuniform_time:
            maximum_error = float(np.max(np.abs(dt - median_dt)))
            raise ValueError(
                "The timestamps are not uniformly sampled. "
                "shooting-for-contact currently reduces time.csv to the median "
                f"sample interval. Median dt={median_dt:.9g} s; maximum deviation="
                f"{maximum_error:.9g} s. Resample the trajectory first, or pass "
                "--allow-nonuniform-time to split it without resampling."
            )

    # Floating-base MuJoCo qpos:
    # x, y, z, qw, qx, qy, qz, joint_0, ...
    if normalize_quaternion:
        if nq < 7:
            raise ValueError(
                "Quaternion normalization requires at least seven qpos columns."
            )

        quaternion = qpos[:, 3:7]
        norms = np.linalg.norm(quaternion, axis=1)

        if np.any(norms < 1e-12):
            index = int(np.flatnonzero(norms < 1e-12)[0])
            raise ValueError(
                f"Frame {index + 1} contains a zero-length base quaternion."
            )

        qpos[:, 3:7] = quaternion / norms[:, None]

    output_dir.mkdir(parents=True, exist_ok=True)
    qpos_path = output_dir / f"qpos_{dof}dof.csv"
    time_path = output_dir / "time.csv"

    # Both shooting-for-contact files are headerless.
    np.savetxt(qpos_path, qpos, delimiter=",", fmt="%.18e")
    np.savetxt(time_path, time, delimiter=",", fmt="%.9f")

    print(f"Input:       {input_csv}")
    print(f"Header:      {'detected and removed' if had_header else 'none'}")
    print(f"Frames:      {len(time)}")
    print(f"Input shape: {data.shape}")
    print(f"Qpos shape:  {qpos.shape}")

    if median_dt is not None:
        print(f"Median dt:   {median_dt:.9g} s")
        print(f"Uniform:     {'yes' if is_uniform else 'no (accepted by override)'}")

    print(f"Wrote:       {qpos_path}")
    print(f"Wrote:       {time_path}")

    return qpos_path, time_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split a GhostGUI time-plus-qpos trajectory CSV into the two "
            "headerless CSV files expected by shooting-for-contact."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the GhostGUI trajectory CSV.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination DSMS trajectory folder.",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=29,
        help="Joint DoF used in the output filename. Default: 29.",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=36,
        help="Expected MuJoCo qpos width. Default: 36 for G1 29-DoF.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Input and output delimiter. Default: comma.",
    )
    parser.add_argument(
        "--allow-nonuniform-time",
        action="store_true",
        help=(
            "Allow irregular timestamps. DSMS still interprets the reference "
            "using the median interval, so resampling is normally preferable."
        ),
    )
    parser.add_argument(
        "--no-normalize-quaternion",
        action="store_true",
        help="Do not normalize floating-base quaternion columns qw,qx,qy,qz.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        convert(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            dof=args.dof,
            nq=args.nq,
            delimiter=args.delimiter,
            allow_nonuniform_time=args.allow_nonuniform_time,
            normalize_quaternion=not args.no_normalize_quaternion,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

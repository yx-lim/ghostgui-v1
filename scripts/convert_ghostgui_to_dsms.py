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
  --speed 0.5 \
  --allow-nonuniform-time
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.trajectory_export_formats import (
    prepare_dsms_arrays,
    write_dsms_files,
)


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
    motion_speed: float = 1.0,
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

    prepared = prepare_dsms_arrays(
        data[:, 0],
        data[:, 1:],
        expected_qpos_count=nq,
        allow_nonuniform_time=allow_nonuniform_time,
        normalize_quaternion=normalize_quaternion,
        motion_speed=motion_speed,
    )
    time = prepared.times
    qpos = prepared.qposes
    median_dt = prepared.median_dt
    is_uniform = prepared.is_uniform
    qpos_path, time_path = write_dsms_files(output_dir, time, qpos, dof)

    print(f"Input:       {input_csv}")
    print(f"Header:      {'detected and removed' if had_header else 'none'}")
    print(f"Frames:      {len(time)}")
    print(f"Input shape: {data.shape}")
    print(f"Qpos shape:  {qpos.shape}")

    print(f"Motion speed: {prepared.motion_speed:.9g}x")
    print(f"Time scale:   {1.0 / prepared.motion_speed:.9g}x")
    print(f"Duration:     {prepared.source_duration:.9g} s -> "
          f"{prepared.output_duration:.9g} s")

    if median_dt is not None:
        print(f"Source dt:   {prepared.source_median_dt:.9g} s")
        print(f"Output dt:   {median_dt:.9g} s")
        print(f"Frequency:   {1.0 / prepared.source_median_dt:.9g} Hz -> "
              f"{1.0 / median_dt:.9g} Hz")
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
        "--speed",
        type=float,
        default=1.0,
        help=(
            "DSMS motion-speed multiplier. Timestamps are divided by this "
            "value while qpos samples remain unchanged. Default: 1.0."
        ),
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
            motion_speed=args.speed,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

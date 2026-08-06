#!/usr/bin/env python3
"""
Convert a GhostGUI G1 trajectory CSV into mjlab's CSV input format.

Basic usage:
    python scripts/ghostgui_to_mjlab.py input.csv output.csv
    python scripts/ghostgui_to_mjlab.py csv/trajectory/g1-frontflip-traj.csv csv/trajectory/g1-frontflip-mjlab.csv

GhostGUI canonical format:
    time,
    base_x, base_y, base_z,
    base_qw, base_qx, base_qy, base_qz,
    29 G1 joint positions

mjlab CSV input format:
    base_x, base_y, base_z,
    base_qx, base_qy, base_qz, base_qw,
    29 G1 joint positions

The script automatically estimates the input FPS from the GhostGUI time column.
The output CSV is headerless because mjlab reads it with numpy.loadtxt().

Optionally, the script can launch mjlab's csv_to_npz converter. The mjlab
output frequency is fixed at 50 Hz.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.trajectory_export_formats import (
    G1_JOINT_ORDER,
    convert_mjlab_rows,
    write_mjlab_rows,
)


OUTPUT_FPS = 50.0
DEFAULT_MAX_DT_JITTER = 0.02  # 2% relative deviation from median dt.

GHOSTGUI_BASE_COLUMNS: tuple[str, ...] = (
    "time",
    "base_x",
    "base_y",
    "base_z",
    "base_qw",
    "base_qx",
    "base_qy",
    "base_qz",
)

EXPECTED_GHOSTGUI_COLUMNS: tuple[str, ...] = GHOSTGUI_BASE_COLUMNS + G1_JOINT_ORDER
EXPECTED_INPUT_WIDTH = len(EXPECTED_GHOSTGUI_COLUMNS)  # 37
EXPECTED_OUTPUT_WIDTH = 3 + 4 + len(G1_JOINT_ORDER)    # 36


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GhostGUI G1 trajectory CSV to mjlab format. "
            "Input FPS is inferred from the time column."
        )
    )
    parser.add_argument("input_csv", type=Path, help="GhostGUI trajectory CSV.")
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Headerless mjlab-format CSV to create.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=None,
        help=(
            "Optional manual FPS override. Normally omitted because FPS is "
            "estimated from the time column."
        ),
    )
    parser.add_argument(
        "--max-dt-jitter",
        type=float,
        default=DEFAULT_MAX_DT_JITTER,
        help=(
            "Maximum allowed relative frame-spacing deviation from median dt. "
            f"Default: {DEFAULT_MAX_DT_JITTER:.0%}.".replace("%", "%%")
        ),
    )
    parser.add_argument(
        "--allow-nonuniform-time",
        action="store_true",
        help=(
            "Allow nonuniform timestamps and use the median dt to estimate FPS. "
            "Use cautiously because mjlab's converter assumes a fixed input FPS."
        ),
    )
    parser.add_argument(
        "--normalize-quaternions",
        action="store_true",
        help="Normalize each output quaternion after converting wxyz to xyzw.",
    )
    parser.add_argument(
        "--run-converter",
        action="store_true",
        help="Run `uv run -m mjlab.scripts.csv_to_npz` after adapting the CSV.",
    )
    parser.add_argument(
        "--mjlab-dir",
        type=Path,
        help=(
            "Path to the cloned mjlab repository. Required with --run-converter "
            "unless the current directory is the mjlab repository."
        ),
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Motion artifact name passed to mjlab. "
            "Default: output CSV filename without extension."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to mjlab's converter. Default: cuda:0.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Ask mjlab's converter to render the converted reference motion.",
    )
    return parser


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _normalize_header_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_csv(path: Path) -> tuple[list[str] | None, list[list[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        raw_rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(file)
            if row and any(cell.strip() for cell in row)
        ]

    if not raw_rows:
        raise ValueError(f"Input CSV is empty: {path}")

    has_header = not all(_is_float(cell) for cell in raw_rows[0])
    header = (
        [_normalize_header_name(cell) for cell in raw_rows[0]]
        if has_header
        else None
    )
    data_rows = raw_rows[1:] if has_header else raw_rows

    numeric_rows: list[list[float]] = []
    expected_width = len(header) if header is not None else EXPECTED_INPUT_WIDTH

    for line_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != expected_width:
            raise ValueError(
                f"Line {line_number}: expected {expected_width} columns, "
                f"found {len(row)}."
            )

        try:
            numeric_row = [float(cell) for cell in row]
        except ValueError as exc:
            raise ValueError(
                f"Line {line_number}: contains a non-numeric value."
            ) from exc

        if not all(math.isfinite(value) for value in numeric_row):
            raise ValueError(
                f"Line {line_number}: contains NaN or infinite values."
            )

        numeric_rows.append(numeric_row)

    if not numeric_rows:
        raise ValueError("Input CSV contains a header but no trajectory rows.")

    return header, numeric_rows


def reorder_headered_rows(
    header: Sequence[str], rows: Sequence[Sequence[float]]
) -> list[list[float]]:
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate column names found: " + ", ".join(duplicates)
        )

    index = {name: i for i, name in enumerate(header)}
    missing = [name for name in EXPECTED_GHOSTGUI_COLUMNS if name not in index]
    if missing:
        raise ValueError(
            "Missing required GhostGUI columns:\n  " + "\n  ".join(missing)
        )

    return [
        [float(row[index[name]]) for name in EXPECTED_GHOSTGUI_COLUMNS]
        for row in rows
    ]


def validate_headerless_rows(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    for row_number, row in enumerate(rows, start=1):
        if len(row) != EXPECTED_INPUT_WIDTH:
            raise ValueError(
                f"Row {row_number}: headerless GhostGUI input must contain "
                f"{EXPECTED_INPUT_WIDTH} columns in the documented order; "
                f"found {len(row)}."
            )
    return [list(map(float, row)) for row in rows]


def detect_input_fps(
    rows: Sequence[Sequence[float]],
    *,
    manual_fps: float | None,
    max_dt_jitter: float,
    allow_nonuniform_time: bool,
) -> tuple[float, float, float]:
    """
    Return (input_fps, median_dt, max_relative_jitter).

    mjlab's converter accepts one fixed input FPS, so strongly nonuniform
    timestamps are rejected unless --allow-nonuniform-time is supplied.
    """
    if manual_fps is not None:
        if manual_fps <= 0 or not math.isfinite(manual_fps):
            raise ValueError("--input-fps must be a positive finite number.")

    if max_dt_jitter < 0 or not math.isfinite(max_dt_jitter):
        raise ValueError("--max-dt-jitter must be a non-negative finite number.")

    if len(rows) < 2:
        if manual_fps is None:
            raise ValueError(
                "At least two frames are required to infer input FPS. "
                "Provide --input-fps for a single-frame trajectory."
            )
        return manual_fps, 1.0 / manual_fps, 0.0

    times = [float(row[0]) for row in rows]
    deltas = [b - a for a, b in zip(times, times[1:])]

    if any(dt <= 0 for dt in deltas):
        raise ValueError("Time values must be strictly increasing.")

    median_dt = statistics.median(deltas)
    detected_fps = 1.0 / median_dt
    max_relative_jitter = max(
        abs(dt - median_dt) / median_dt for dt in deltas
    )

    if max_relative_jitter > max_dt_jitter:
        message = (
            f"Time spacing is not sufficiently uniform. Median dt={median_dt:.9g} s "
            f"({detected_fps:.6g} Hz), maximum relative deviation="
            f"{max_relative_jitter:.2%}, allowed={max_dt_jitter:.2%}."
        )
        if not allow_nonuniform_time:
            raise ValueError(
                message
                + " Resample the GhostGUI trajectory to uniform timestamps, "
                "or pass --allow-nonuniform-time to use the median dt."
            )
        print(f"[WARN] {message}", file=sys.stderr)

    if manual_fps is not None:
        manual_dt = 1.0 / manual_fps
        relative_difference = abs(manual_dt - median_dt) / median_dt
        if relative_difference > 0.02:
            print(
                (
                    f"[WARN] Auto-detected FPS is {detected_fps:.6g} Hz, but "
                    f"--input-fps={manual_fps:.6g} was supplied. "
                    "The manual override will be used."
                ),
                file=sys.stderr,
            )
        return manual_fps, median_dt, max_relative_jitter

    return detected_fps, median_dt, max_relative_jitter


def convert_rows(
    ghostgui_rows: Sequence[Sequence[float]],
    normalize_quaternions: bool,
) -> tuple[list[list[float]], int]:
    return convert_mjlab_rows(ghostgui_rows, normalize_quaternions)


def write_headerless_csv(path: Path, rows: Sequence[Sequence[float]]) -> None:
    write_mjlab_rows(path, rows)


def resolve_mjlab_dir(requested: Path | None) -> Path:
    candidate = (requested or Path.cwd()).resolve()
    pyproject = candidate / "pyproject.toml"
    package_dir = candidate / "src" / "mjlab"

    if not pyproject.exists() or not package_dir.exists():
        raise ValueError(
            f"{candidate} does not look like the mjlab repository. "
            "Pass its path using --mjlab-dir."
        )

    return candidate


def run_mjlab_converter(
    *,
    mjlab_dir: Path,
    adapted_csv: Path,
    input_fps: float,
    output_name: str,
    device: str,
    render: bool,
) -> None:
    command = [
        "uv",
        "run",
        "-m",
        "mjlab.scripts.csv_to_npz",
        "--input-file",
        str(adapted_csv.resolve()),
        "--output-name",
        output_name,
        "--input-fps",
        f"{input_fps:.12g}",
        "--output-fps",
        f"{OUTPUT_FPS:g}",
        "--device",
        device,
        "--render",
        "True" if render else "False",
    ]

    print("\nRunning mjlab converter:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=mjlab_dir, check=True)


def main() -> int:
    args = build_parser().parse_args()

    header, rows = read_csv(args.input_csv)
    canonical_rows = (
        reorder_headered_rows(header, rows)
        if header is not None
        else validate_headerless_rows(rows)
    )

    input_fps, median_dt, max_jitter = detect_input_fps(
        canonical_rows,
        manual_fps=args.input_fps,
        max_dt_jitter=args.max_dt_jitter,
        allow_nonuniform_time=args.allow_nonuniform_time,
    )

    output_rows, non_unit_count = convert_rows(
        canonical_rows,
        normalize_quaternions=args.normalize_quaternions,
    )
    write_headerless_csv(args.output_csv, output_rows)

    print(f"Converted {len(output_rows)} frames.")
    print(f"Input:  {args.input_csv.resolve()}")
    print(f"Output: {args.output_csv.resolve()}")
    print(f"Detected median dt: {median_dt:.9g} s")
    print(f"Input frequency supplied to mjlab: {input_fps:.9g} Hz")
    print(f"Maximum timestamp jitter: {max_jitter:.3%}")
    print(f"mjlab output frequency: {OUTPUT_FPS:g} Hz")
    print("Output layout: xyz + quaternion xyzw + 29 G1 joints")
    print("Output CSV is headerless.")

    if non_unit_count:
        action = "were normalized" if args.normalize_quaternions else "were not changed"
        print(
            f"[WARN] {non_unit_count} quaternion(s) differed from unit length "
            f"by more than 1e-3 and {action}.",
            file=sys.stderr,
        )

    output_name = args.output_name or args.output_csv.stem

    if args.run_converter:
        mjlab_dir = resolve_mjlab_dir(args.mjlab_dir)
        run_mjlab_converter(
            mjlab_dir=mjlab_dir,
            adapted_csv=args.output_csv,
            input_fps=input_fps,
            output_name=output_name,
            device=args.device,
            render=args.render,
        )
    else:
        print("\nNext command from inside the mjlab repository:")
        print(
            "  uv run -m mjlab.scripts.csv_to_npz"
            f" --input-file {args.output_csv.resolve()}"
            f" --output-name {output_name}"
            f" --input-fps {input_fps:.12g}"
            f" --output-fps {OUTPUT_FPS:g}"
            " --render True"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)

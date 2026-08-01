"""Compatibility wrapper for the installed MuJoCo viewer process.

Run directly from a source checkout with::

    python3 scripts/view_g1_mujoco.py

New integrations should run ``python -m application.mujoco_viewer_process``.
"""

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.mujoco_viewer_process import (  # noqa: E402,F401
    BASE_COLUMNS,
    DEFAULT_CSV_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RAW_QPOS_KEY,
    TrajectoryPlayer,
    default_playback_cache_path,
    load_trajectory_csv,
    main,
    parse_args,
    read_stdin_commands,
)


__all__ = [
    "BASE_COLUMNS",
    "DEFAULT_CSV_PATH",
    "MODEL_PATH",
    "PROJECT_ROOT",
    "RAW_QPOS_KEY",
    "TrajectoryPlayer",
    "default_playback_cache_path",
    "load_trajectory_csv",
    "main",
    "parse_args",
    "read_stdin_commands",
]


if __name__ == "__main__":
    main()

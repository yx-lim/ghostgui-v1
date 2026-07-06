"""Shared trajectory CSV schema, import, and export helpers."""

import csv
from pathlib import Path


BASE_COLUMNS = (
    "base_x", "base_y", "base_z",
    "base_qw", "base_qx", "base_qy", "base_qz",
)


def load_trajectory_csv(csv_path):
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append({
                key: float(value)
                for key, value in row.items()
                if key is not None and value not in (None, "")
            })
    return path, rows


def export_configurations_csv(configurations, joint_names, csv_path):
    if not configurations:
        raise RuntimeError("No solved trajectory to export.")
    header = ["time", *BASE_COLUMNS, *joint_names]
    with open(csv_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for configuration in configurations:
            writer.writerow([
                configuration.time,
                *(getattr(configuration, name) for name in BASE_COLUMNS),
                *configuration.joint_positions,
            ])

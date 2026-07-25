"""Project path helpers shared by application and GUI layers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "csv"
QPOS_CSV_DIR = CSV_DIR / "qpos"
TRAJECTORY_CSV_DIR = CSV_DIR / "trajectory"


def csv_file_path(filename):
    return CSV_DIR / filename


def prepare_csv_save_path(csv_path):
    path = Path(csv_path).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    if not path.is_absolute():
        path = CSV_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()

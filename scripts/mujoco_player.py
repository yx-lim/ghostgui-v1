"""
Standalone model-agnostic MuJoCo trajectory player.

Run directly:
    python3 scripts/mujoco_player.py

The GUI launches this script as a subprocess and sends simple stdin commands:
    load /path/to/trajectory.csv
    play
    pause
    seek 42
    speed 1.5
    refresh
"""

from pathlib import Path
import argparse
import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.csv_io import BASE_COLUMNS, load_trajectory_csv


MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"
DEFAULT_CSV_PATH = PROJECT_ROOT / "pelvis_base_trajectory_uniform_dt.csv"


def read_stdin_commands(command_queue):
    for line in sys.stdin:
        line = line.strip()
        if line:
            command_queue.put(line)


class TrajectoryPlayer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.rows = []
        self.csv_path = None
        self.index = 0
        self.playing = False
        self.speed = 1.0
        self.current_time = 0.0
        self.last_status_wall_time = 0.0

        self.joint_qpos_by_column = self.build_joint_qpos_map()

    def build_joint_qpos_map(self):
        mapping = {}

        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if not name:
                continue

            qpos_address = int(self.model.jnt_qposadr[joint_id])
            mapping[name] = qpos_address

            if name.startswith("robot/"):
                mapping[name[len("robot/"):]] = qpos_address

        return mapping

    def load_csv(self, csv_path):
        path, rows = load_trajectory_csv(csv_path)
        self.csv_path = path
        self.rows = rows
        self.index = 0
        self.playing = False
        self.current_time = rows[0]["time"] if rows else 0.0
        self.apply_current_row()
        self.print_status(force=True)

    def row_count(self):
        return len(self.rows)

    def duration(self):
        if not self.rows:
            return 0.0
        return self.rows[-1].get("time", 0.0)

    def apply_current_row(self):
        if not self.rows:
            return

        row = self.rows[self.index]

        free_joints = [
            joint_id for joint_id in range(self.model.njnt)
            if int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if free_joints and all(column in row for column in BASE_COLUMNS):
            address = int(self.model.jnt_qposadr[free_joints[0]])
            self.data.qpos[address:address + 7] = [row[name] for name in BASE_COLUMNS]

        for column, qpos_address in self.joint_qpos_by_column.items():
            if column in row:
                self.data.qpos[qpos_address] = row[column]

        mujoco.mj_forward(self.model, self.data)

    def seek_index(self, index):
        if not self.rows:
            self.index = 0
            self.current_time = 0.0
            return

        self.index = max(0, min(len(self.rows) - 1, int(index)))
        self.current_time = self.rows[self.index].get("time", 0.0)
        self.apply_current_row()
        self.print_status(force=True)

    def seek_time(self, target_time):
        if not self.rows:
            return

        closest_index = min(
            range(len(self.rows)),
            key=lambda i: abs(self.rows[i].get("time", 0.0) - target_time),
        )
        self.seek_index(closest_index)

    def step_playback(self, elapsed_seconds):
        if not self.playing or not self.rows:
            return

        self.current_time += elapsed_seconds * self.speed

        if self.current_time >= self.duration():
            self.current_time = self.duration()
            self.playing = False

        closest_index = self.index
        while (
            closest_index + 1 < len(self.rows)
            and self.rows[closest_index + 1].get("time", 0.0) <= self.current_time
        ):
            closest_index += 1

        if closest_index != self.index:
            self.index = closest_index
            self.apply_current_row()

    def handle_command(self, command):
        parts = command.split(maxsplit=1)
        action = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        try:
            if action == "load":
                self.load_csv(argument)
            elif action == "play":
                if self.rows and self.index >= len(self.rows) - 1:
                    self.seek_index(0)
                self.playing = True
                self.print_status(force=True)
            elif action == "pause":
                self.playing = False
                self.print_status(force=True)
            elif action == "seek":
                self.seek_index(int(float(argument)))
            elif action == "seek_time":
                self.seek_time(float(argument))
            elif action == "speed":
                self.speed = max(0.01, float(argument))
                self.print_status(force=True)
            elif action == "refresh":
                self.apply_current_row()
                self.print_status(force=True)
            else:
                print(f"WARNING unknown command: {command}", flush=True)
        except Exception as exc:
            print(f"ERROR command failed: {command} | {exc}", flush=True)

    def print_status(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_status_wall_time < 0.10:
            return

        self.last_status_wall_time = now
        row_time = 0.0
        if self.rows:
            row_time = self.rows[self.index].get("time", 0.0)

        print(
            "STATE "
            f"index={self.index} "
            f"count={len(self.rows)} "
            f"time={row_time:.6f} "
            f"duration={self.duration():.6f} "
            f"playing={int(self.playing)} "
            f"speed={self.speed:.3f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional trajectory CSV to load on startup.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="MJCF or MuJoCo-compatible URDF to load.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    if not model_path.exists():
        print(f"Could not find model: {model_path}", flush=True)
        sys.exit(1)

    print(f"Loading model: {model_path}", flush=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.mj_forward(model, data)

    print("Model loaded.", flush=True)
    print(f"nq: {model.nq}", flush=True)
    print(f"nv: {model.nv}", flush=True)
    print(f"nu: {model.nu}", flush=True)
    print(f"njnt: {model.njnt}", flush=True)
    print(f"nbody: {model.nbody}", flush=True)

    command_queue = queue.Queue()
    command_thread = threading.Thread(
        target=read_stdin_commands,
        args=(command_queue,),
        daemon=True,
    )
    command_thread.start()

    player = TrajectoryPlayer(model, data)
    startup_csv = args.csv

    if startup_csv is None and DEFAULT_CSV_PATH.exists():
        startup_csv = str(DEFAULT_CSV_PATH)

    if startup_csv is not None:
        try:
            player.load_csv(startup_csv)
            print(f"Loaded trajectory CSV: {startup_csv}", flush=True)
        except Exception as exc:
            print(f"WARNING could not load CSV: {exc}", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_wall_time = time.monotonic()

        while viewer.is_running():
            now = time.monotonic()
            elapsed = now - last_wall_time
            last_wall_time = now

            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                player.handle_command(command)

            player.step_playback(elapsed)
            player.print_status()

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()

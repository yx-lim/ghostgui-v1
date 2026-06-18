"""
backend_interface.py

Purpose:
    Connect GUI trajectory x(t) to backend q(t).

Current behavior:
    - Uses C++ robot_backend if available
    - Falls back to Python version if C++ backend is not compiled
    - Can export q(t) CSV
"""

from dataclasses import dataclass
import csv


try:
    import robot_backend

    CPP_BACKEND_AVAILABLE = True

except ImportError:
    robot_backend = None
    CPP_BACKEND_AVAILABLE = False


LAB_JOINT_NAMES = [
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
]


DEFAULT_JOINT_POSITIONS = [0.0 for _ in LAB_JOINT_NAMES]


@dataclass
class PythonRobotConfiguration:
    time: float = 0.0

    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.9

    base_qw: float = 1.0
    base_qx: float = 0.0
    base_qy: float = 0.0
    base_qz: float = 0.0

    joint_names: list = None
    joint_positions: list = None

    ik_error: float = 0.0
    success: bool = True
    status: str = "Python fallback"


def rpy_to_quaternion(roll, pitch, yaw):
    import math

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


class PythonTrajectoryBackend:
    def __init__(self):
        self.joint_names = LAB_JOINT_NAMES
        self.default_joint_positions = DEFAULT_JOINT_POSITIONS
        self.last_solution = []

    def solve_trajectory(self, trajectory):
        self.last_solution = []

        sorted_frames = sorted(trajectory.frames, key=lambda f: f.time)

        for frame in sorted_frames:
            q = PythonRobotConfiguration(
                time=frame.time,
                joint_names=self.joint_names,
                joint_positions=self.default_joint_positions.copy(),
            )

            if frame.frame_name in ["pelvis", "base", "root"]:
                q.base_x = frame.x
                q.base_y = frame.y
                q.base_z = frame.z

                (
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ) = rpy_to_quaternion(
                    frame.roll,
                    frame.pitch,
                    frame.yaw,
                )

                q.status = (
                    f"Python fallback: mapped {frame.frame_name} "
                    f"target to base pose at t={frame.time:.2f}s"
                )

            else:
                q.status = (
                    f"Python fallback: ignored non-pelvis frame "
                    f"{frame.frame_name}"
                )

            self.last_solution.append(q)

        return self.last_solution

    def export_last_solution_csv(self, csv_path):
        if not self.last_solution:
            raise RuntimeError("No solved trajectory to export.")

        header = [
            "time",
            "base_x",
            "base_y",
            "base_z",
            "base_qw",
            "base_qx",
            "base_qy",
            "base_qz",
        ] + self.joint_names

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for q in self.last_solution:
                row = [
                    q.time,
                    q.base_x,
                    q.base_y,
                    q.base_z,
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ] + q.joint_positions

                writer.writerow(row)


class BackendInterface:
    def __init__(self):
        if CPP_BACKEND_AVAILABLE:
            self.backend = robot_backend.RobotBackend()
            self.backend.set_joint_names(LAB_JOINT_NAMES)
            self.backend.set_default_joint_positions(DEFAULT_JOINT_POSITIONS)
            self.using_cpp_backend = True
        else:
            self.backend = PythonTrajectoryBackend()
            self.using_cpp_backend = False

    def backend_name(self):
        if self.using_cpp_backend:
            return "C++ pelvis-target to base-pose backend"
        return "Python pelvis-target fallback backend"

    def solve_trajectory(self, trajectory):
        return self.backend.solve_trajectory(trajectory)

    def export_last_solution_csv(self, csv_path):
        self.backend.export_last_solution_csv(csv_path)
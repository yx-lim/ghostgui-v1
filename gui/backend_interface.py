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
import math


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


JOINT_INDEX = {
    name: index
    for index, name in enumerate(LAB_JOINT_NAMES)
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


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


def target_point(target, default_y=0.0):
    y = target.y

    if abs(y) < 1e-9:
        y = default_y

    return target.x, y, target.z


def two_link_flexion(distance, upper_len, lower_len):
    d = clamp(distance, 1e-6, upper_len + lower_len - 1e-6)
    cos_angle = (
        upper_len * upper_len
        + lower_len * lower_len
        - d * d
    ) / (2.0 * upper_len * lower_len)
    return math.pi - math.acos(clamp(cos_angle, -1.0, 1.0))


class PythonTrajectoryBackend:
    def __init__(self):
        self.joint_names = LAB_JOINT_NAMES
        self.default_joint_positions = DEFAULT_JOINT_POSITIONS
        self.last_solution = []

    def make_default_configuration(self, time=0.0):
        return PythonRobotConfiguration(
            time=time,
            joint_names=self.joint_names,
            joint_positions=self.default_joint_positions.copy(),
        )

    def copy_configuration_at_time(self, q_prev, time):
        return PythonRobotConfiguration(
            time=time,
            base_x=q_prev.base_x,
            base_y=q_prev.base_y,
            base_z=q_prev.base_z,
            base_qw=q_prev.base_qw,
            base_qx=q_prev.base_qx,
            base_qy=q_prev.base_qy,
            base_qz=q_prev.base_qz,
            joint_names=self.joint_names,
            joint_positions=q_prev.joint_positions.copy(),
            ik_error=q_prev.ik_error,
            success=q_prev.success,
            status=q_prev.status,
        )

    def default_targets_for_configuration(self, q):
        return {
            "torso": (q.base_x, q.base_y, q.base_z + 0.35),
            "left_foot": (q.base_x - 0.10, q.base_y + 0.08, 0.0),
            "right_foot": (q.base_x + 0.10, q.base_y - 0.08, 0.0),
            "left_hand": (q.base_x - 0.35, q.base_y + 0.18, q.base_z + 0.05),
            "right_hand": (q.base_x + 0.35, q.base_y - 0.18, q.base_z + 0.05),
        }

    def estimate_joint_positions(self, q, active_targets):
        """
        First-pass task-target to 29-joint estimate.

        This is a lightweight analytic approximation for CSV generation, not a
        replacement for full humanoid IK. It keeps every named joint column
        populated from pelvis, torso, hand, and foot targets.
        """
        joints = [0.0 for _ in self.joint_names]
        defaults = self.default_targets_for_configuration(q)

        def set_joint(name, value):
            index = JOINT_INDEX.get(name)
            if index is not None:
                joints[index] = clamp(value, -2.8, 2.8)

        torso_target = active_targets.get("torso")
        if torso_target is not None:
            tx, ty, tz = target_point(torso_target, default_y=q.base_y)
            dx = tx - q.base_x
            dy = ty - q.base_y
            dz = max(0.05, tz - q.base_z)

            set_joint("waist_yaw_joint", torso_target.yaw)
            set_joint("waist_roll_joint", torso_target.roll + math.atan2(dy, dz))
            set_joint("waist_pitch_joint", torso_target.pitch + math.atan2(dx, dz))

        self.estimate_leg_joints(
            side="left",
            q=q,
            target=active_targets.get("left_foot"),
            default_point=defaults["left_foot"],
            set_joint=set_joint,
        )
        self.estimate_leg_joints(
            side="right",
            q=q,
            target=active_targets.get("right_foot"),
            default_point=defaults["right_foot"],
            set_joint=set_joint,
        )
        self.estimate_arm_joints(
            side="left",
            q=q,
            target=active_targets.get("left_hand"),
            default_point=defaults["left_hand"],
            set_joint=set_joint,
        )
        self.estimate_arm_joints(
            side="right",
            q=q,
            target=active_targets.get("right_hand"),
            default_point=defaults["right_hand"],
            set_joint=set_joint,
        )

        return joints

    def estimate_leg_joints(self, side, q, target, default_point, set_joint):
        sign = 1.0 if side == "left" else -1.0
        hip_y = q.base_y + sign * 0.08
        hip_x = q.base_x
        hip_z = q.base_z - 0.10

        if target is None:
            foot_x, foot_y, foot_z = default_point
        else:
            foot_x, foot_y, foot_z = target_point(target, default_y=hip_y)

        dx = foot_x - hip_x
        dy = foot_y - hip_y
        dz = foot_z - hip_z

        upper_len = 0.42
        lower_len = 0.42
        distance = math.sqrt(dx * dx + dz * dz)
        knee = two_link_flexion(distance, upper_len, lower_len)
        hip_pitch = math.atan2(dx, max(1e-6, -dz)) - 0.5 * knee
        ankle_pitch = -hip_pitch - knee

        set_joint(f"{side}_hip_pitch_joint", hip_pitch)
        set_joint(f"{side}_hip_roll_joint", sign * math.atan2(dy, max(0.10, abs(dz))))
        set_joint(f"{side}_hip_yaw_joint", target.yaw if target is not None else 0.0)
        set_joint(f"{side}_knee_joint", knee)
        set_joint(f"{side}_ankle_pitch_joint", ankle_pitch)
        set_joint(f"{side}_ankle_roll_joint", -sign * math.atan2(dy, max(0.10, abs(dz))))

    def estimate_arm_joints(self, side, q, target, default_point, set_joint):
        sign = 1.0 if side == "left" else -1.0
        shoulder_y = q.base_y + sign * 0.18
        shoulder_x = q.base_x
        shoulder_z = q.base_z + 0.35

        if target is None:
            hand_x, hand_y, hand_z = default_point
            roll = pitch = yaw = 0.0
        else:
            hand_x, hand_y, hand_z = target_point(target, default_y=shoulder_y)
            roll = target.roll
            pitch = target.pitch
            yaw = target.yaw

        dx = hand_x - shoulder_x
        dy = hand_y - shoulder_y
        dz = hand_z - shoulder_z

        upper_len = 0.28
        lower_len = 0.28
        planar_distance = math.sqrt(dx * dx + dz * dz)
        elbow = two_link_flexion(planar_distance, upper_len, lower_len)
        shoulder_pitch = -math.atan2(dz, dx if abs(dx) > 1e-6 else 1e-6)

        if side == "right":
            shoulder_pitch = math.atan2(dz, -dx if abs(dx) > 1e-6 else 1e-6)

        set_joint(f"{side}_shoulder_pitch_joint", shoulder_pitch)
        set_joint(f"{side}_shoulder_roll_joint", sign * math.atan2(dy, max(0.10, planar_distance)))
        set_joint(f"{side}_shoulder_yaw_joint", yaw + math.atan2(dy, max(0.10, abs(dx))))
        set_joint(f"{side}_elbow_joint", elbow)
        set_joint(f"{side}_wrist_roll_joint", roll)
        set_joint(f"{side}_wrist_pitch_joint", pitch)
        set_joint(f"{side}_wrist_yaw_joint", yaw)

    def solve_trajectory(self, trajectory):
        if getattr(trajectory, "samples", None) is not None:
            return self.solve_grouped_trajectory(trajectory.samples)

        self.last_solution = []
        active_targets = {}

        sorted_frames = sorted(trajectory.frames, key=lambda f: f.time)

        for frame in sorted_frames:
            q = self.make_default_configuration(time=frame.time)
            active_targets[frame.frame_name] = frame

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
                    f"Python fallback: estimated joints from "
                    f"{frame.frame_name} target"
                )

            q.joint_positions = self.estimate_joint_positions(q, active_targets)
            self.last_solution.append(q)

        return self.last_solution

    def solve_grouped_trajectory(self, samples):
        self.last_solution = []
        q_prev = self.make_default_configuration()
        active_targets = {}

        for sample in samples:
            targets = sample["targets"]
            active_targets.update(targets)
            pelvis_target = None

            for name in ["pelvis", "base", "root"]:
                if name in targets:
                    pelvis_target = targets[name]
                    break

            q = self.copy_configuration_at_time(q_prev, sample["time"])

            ignored_names = sorted(
                name
                for name in targets.keys()
                if name not in ["pelvis", "base", "root"]
            )

            if pelvis_target is not None:
                q.base_x = pelvis_target.x
                q.base_y = pelvis_target.y
                q.base_z = pelvis_target.z

                (
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ) = rpy_to_quaternion(
                    pelvis_target.roll,
                    pelvis_target.pitch,
                    pelvis_target.yaw,
                )

                q.status = (
                    f"Python fallback: mapped {pelvis_target.frame_name} "
                    f"target to base pose at t={sample['time']:.2f}s"
                )
            else:
                q.status = (
                    "Python fallback: no pelvis/base/root target at "
                    f"t={sample['time']:.2f}s; held previous base pose"
                )

            if ignored_names:
                q.status += (
                    "; estimated joint targets from: "
                    + ", ".join(ignored_names)
                )

            q.joint_positions = self.estimate_joint_positions(q, active_targets)
            self.last_solution.append(q)
            q_prev = q

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
        self.grouped_fallback_backend = PythonTrajectoryBackend()
        self.last_backend = None

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
        if getattr(trajectory, "samples", None) is not None and self.using_cpp_backend:
            self.last_backend = self.grouped_fallback_backend
            return self.grouped_fallback_backend.solve_trajectory(trajectory)

        self.last_backend = self.backend
        return self.backend.solve_trajectory(trajectory)

    def export_last_solution_csv(self, csv_path):
        backend = self.last_backend or self.backend
        backend.export_last_solution_csv(csv_path)

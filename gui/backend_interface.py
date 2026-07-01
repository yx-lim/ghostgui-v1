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
from pathlib import Path

try:
    import mujoco
    import numpy as np

    MUJOCO_IK_AVAILABLE = True
except ImportError:
    mujoco = None
    np = None
    MUJOCO_IK_AVAILABLE = False


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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


IK_TASKS = {
    "torso": ("body", "robot/torso_link", 0.6),
    "left_foot": ("site", "robot/left_foot", 1.0),
    "right_foot": ("site", "robot/right_foot", 1.0),
    "left_hand": ("site", "robot/left_palm", 0.8),
    "right_hand": ("site", "robot/right_palm", 0.8),
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


class MujocoIKBackend(PythonTrajectoryBackend):
    """
    Position-only MuJoCo IK backend.

    Pelvis/base/root targets set the floating base directly. Other active
    target frames are solved against real MuJoCo bodies/sites using position
    Jacobians and damped least squares.
    """

    def __init__(self, model_path=MODEL_PATH, mj_model=None):
        super().__init__()

        if not MUJOCO_IK_AVAILABLE:
            raise RuntimeError("mujoco or numpy is not available")

        self.model_path = Path(model_path)

        if mj_model is None and not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.model = mj_model
        if self.model is None:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.joint_qpos_addresses = {}
        self.joint_dof_addresses = {}
        self.joint_limits = {}
        self.task_bindings = {}

        self.build_joint_maps()
        self.build_task_bindings()
        self.reset_model_state()

    def reset_model_state(self):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

    def build_joint_maps(self):
        for joint_id in range(self.model.njnt):
            mujoco_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if not mujoco_name:
                continue

            plain_name = mujoco_name
            if plain_name.startswith("robot/"):
                plain_name = plain_name[len("robot/"):]

            if plain_name not in JOINT_INDEX:
                continue

            self.joint_qpos_addresses[plain_name] = int(
                self.model.jnt_qposadr[joint_id]
            )
            self.joint_dof_addresses[plain_name] = int(
                self.model.jnt_dofadr[joint_id]
            )

            if self.model.jnt_limited[joint_id]:
                lo, hi = self.model.jnt_range[joint_id]
                self.joint_limits[plain_name] = (float(lo), float(hi))

    def build_task_bindings(self):
        for frame_name, (kind, mujoco_name, weight) in IK_TASKS.items():
            if kind == "site":
                object_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    mujoco_name,
                )
            else:
                object_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    mujoco_name,
                )

            if object_id >= 0:
                self.task_bindings[frame_name] = (
                    kind,
                    object_id,
                    float(weight),
                )

    def backend_label(self):
        return "MuJoCo position IK backend"

    def qpos_to_configuration(self, time=0.0, status="MuJoCo IK"):
        q = PythonRobotConfiguration(
            time=time,
            base_x=float(self.data.qpos[0]),
            base_y=float(self.data.qpos[1]),
            base_z=float(self.data.qpos[2]),
            base_qw=float(self.data.qpos[3]),
            base_qx=float(self.data.qpos[4]),
            base_qy=float(self.data.qpos[5]),
            base_qz=float(self.data.qpos[6]),
            joint_names=self.joint_names,
            joint_positions=[],
            status=status,
        )

        for joint_name in self.joint_names:
            qpos_address = self.joint_qpos_addresses.get(joint_name)
            if qpos_address is None:
                q.joint_positions.append(0.0)
            else:
                q.joint_positions.append(float(self.data.qpos[qpos_address]))

        return q

    def set_base_from_target(self, target):
        self.data.qpos[0] = target.x
        self.data.qpos[1] = target.y
        self.data.qpos[2] = target.z
        (
            self.data.qpos[3],
            self.data.qpos[4],
            self.data.qpos[5],
            self.data.qpos[6],
        ) = rpy_to_quaternion(target.roll, target.pitch, target.yaw)

    def clamp_joint_limits(self):
        for joint_name, qpos_address in self.joint_qpos_addresses.items():
            limits = self.joint_limits.get(joint_name)
            if limits is None:
                continue
            lo, hi = limits
            self.data.qpos[qpos_address] = clamp(
                float(self.data.qpos[qpos_address]),
                lo,
                hi,
            )

    def active_ik_tasks(self, active_targets):
        tasks = []

        for frame_name, target in active_targets.items():
            binding = self.task_bindings.get(frame_name)
            if binding is None:
                continue

            kind, object_id, weight = binding
            desired = np.array([target.x, target.y, target.z], dtype=float)
            tasks.append((frame_name, kind, object_id, weight, desired))

        return tasks

    def current_task_position(self, kind, object_id):
        if kind == "site":
            return self.data.site_xpos[object_id].copy()
        return self.data.xpos[object_id].copy()

    def task_jacobian(self, kind, object_id):
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        if kind == "site":
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, object_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, object_id)

        dof_addresses = [
            self.joint_dof_addresses[name]
            for name in self.joint_names
            if name in self.joint_dof_addresses
        ]
        return jacp[:, dof_addresses]

    def solve_position_ik(
        self,
        active_targets,
        max_iterations=80,
        tolerance=0.005,
        damping=0.04,
        step_size=0.7,
        max_step=0.08,
    ):
        tasks = self.active_ik_tasks(active_targets)

        if not tasks:
            mujoco.mj_forward(self.model, self.data)
            return 0.0, True, 0

        joint_names = [
            name
            for name in self.joint_names
            if name in self.joint_qpos_addresses
            and name in self.joint_dof_addresses
        ]

        final_error = 0.0

        for iteration in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)

            error_blocks = []
            jacobian_blocks = []

            for _, kind, object_id, weight, desired in tasks:
                current = self.current_task_position(kind, object_id)
                error_blocks.append((desired - current) * weight)
                jacobian_blocks.append(
                    self.task_jacobian(kind, object_id) * weight
                )

            error = np.concatenate(error_blocks)
            jacobian = np.vstack(jacobian_blocks)
            final_error = float(np.linalg.norm(error))

            if final_error < tolerance:
                return final_error, True, iteration

            lhs = jacobian.T @ jacobian
            lhs += (damping * damping) * np.eye(lhs.shape[0])
            rhs = jacobian.T @ error

            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            dq = np.clip(dq, -max_step, max_step)

            for index, joint_name in enumerate(joint_names):
                qpos_address = self.joint_qpos_addresses[joint_name]
                self.data.qpos[qpos_address] += step_size * dq[index]

            self.clamp_joint_limits()

        mujoco.mj_forward(self.model, self.data)
        return final_error, final_error < tolerance * 2.0, max_iterations

    def solve_trajectory(self, trajectory):
        if getattr(trajectory, "samples", None) is not None:
            return self.solve_grouped_trajectory(trajectory.samples)

        samples = [
            {
                "time": frame.time,
                "targets": {frame.frame_name: frame},
            }
            for frame in sorted(trajectory.frames, key=lambda f: f.time)
        ]
        return self.solve_grouped_trajectory(samples)

    def solve_grouped_trajectory(self, samples):
        self.last_solution = []
        self.reset_model_state()
        active_targets = {}

        for sample in samples:
            targets = sample["targets"]
            active_targets.update(targets)
            pelvis_target = None

            for name in ["pelvis", "base", "root"]:
                if name in targets:
                    pelvis_target = targets[name]
                    break

            if pelvis_target is not None:
                self.set_base_from_target(pelvis_target)

            error, success, iterations = self.solve_position_ik(active_targets)
            q = self.qpos_to_configuration(
                time=sample["time"],
                status=(
                    "MuJoCo position IK: "
                    f"error={error:.4f}, iterations={iterations}"
                ),
            )
            q.ik_error = error
            q.success = success

            if pelvis_target is None:
                q.status += "; held previous base pose"

            self.last_solution.append(q)

        return self.last_solution


class BackendInterface:
    def __init__(self, mj_model=None):
        self.grouped_fallback_backend = PythonTrajectoryBackend()
        self.ik_backend = None
        self.ik_error = None
        self.last_backend = None

        if MUJOCO_IK_AVAILABLE:
            try:
                self.backend = MujocoIKBackend(mj_model=mj_model)
                self.ik_backend = self.backend
                self.using_cpp_backend = False
                self.using_mujoco_ik_backend = True
            except Exception as exc:
                self.ik_error = str(exc)
                self.backend = PythonTrajectoryBackend()
                self.using_cpp_backend = False
                self.using_mujoco_ik_backend = False
        elif CPP_BACKEND_AVAILABLE:
            self.backend = robot_backend.RobotBackend()
            self.backend.set_joint_names(LAB_JOINT_NAMES)
            self.backend.set_default_joint_positions(DEFAULT_JOINT_POSITIONS)
            self.using_cpp_backend = True
            self.using_mujoco_ik_backend = False
        else:
            self.backend = PythonTrajectoryBackend()
            self.using_cpp_backend = False
            self.using_mujoco_ik_backend = False

    def backend_name(self):
        if self.using_mujoco_ik_backend:
            return "MuJoCo position IK backend"
        if self.using_cpp_backend:
            return "C++ pelvis-target to base-pose backend"
        if self.ik_error:
            return f"Python fallback backend (MuJoCo IK unavailable: {self.ik_error})"
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

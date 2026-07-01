"""
trajectory.py

Purpose:
    Stores target reference-frame keyframes.

Updated project idea:
    The GUI does not directly control the robot.
    The GUI creates a trajectory array of target reference frames.

Each TargetFrame says:
    At this time, this robot frame should be at this pose.

Updated:
    - Position is still linearly interpolated.
    - Orientation is converted from roll/pitch/yaw to quaternion.
    - Quaternion orientation is interpolated using SLERP.
    - Result is converted back to roll/pitch/yaw for the existing backend/export format.
"""

from dataclasses import dataclass, asdict
import json
import math


# ============================================================
# Quaternion helpers
# Convention:
#     Quaternion is [w, x, y, z]
#     Roll  = rotation around x-axis
#     Pitch = rotation around y-axis
#     Yaw   = rotation around z-axis
#     Euler convention is standard robotics RPY:
#         R = Rz(yaw) * Ry(pitch) * Rx(roll)
# ============================================================

def normalize_quat(q):
    """
    Normalize quaternion q = [w, x, y, z].
    """
    norm = math.sqrt(sum(v * v for v in q))

    if norm < 1e-12:
        raise ValueError("Quaternion norm is too close to zero.")

    return tuple(v / norm for v in q)


def rpy_to_quat(roll, pitch, yaw):
    """
    Convert roll/pitch/yaw to quaternion [w, x, y, z].
    Angles are in radians.
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return normalize_quat((w, x, y, z))


def quat_to_rpy(q):
    """
    Convert quaternion [w, x, y, z] back to roll/pitch/yaw.
    Angles are returned in radians.
    """
    w, x, y, z = normalize_quat(q)

    # Roll, x-axis rotation
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch, y-axis rotation
    sinp = 2.0 * (w * y - z * x)

    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw, z-axis rotation
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def slerp(q0, q1, alpha):
    """
    Spherical linear interpolation between two quaternions.

    q0, q1:
        Quaternions in [w, x, y, z] format.

    alpha:
        Interpolation value from 0.0 to 1.0.
    """
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)

    dot = sum(a * b for a, b in zip(q0, q1))

    # q and -q represent the same rotation.
    # If dot < 0, flip q1 so interpolation takes the shorter path.
    if dot < 0.0:
        q1 = tuple(-v for v in q1)
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    # If orientations are almost identical, use normalized lerp.
    # This avoids numerical issues when sin(theta) is tiny.
    if dot > 0.9995:
        q = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(q0, q1))
        return normalize_quat(q)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = tuple(s0 * a + s1 * b for a, b in zip(q0, q1))
    return normalize_quat(q)


# ============================================================
# Data classes
# ============================================================

DEFAULT_TRACK_NAMES = [
    "pelvis",
    "torso",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
]


@dataclass
class TargetFrame:
    time: float = 0.0
    phase: str = "crouch"
    frame_name: str = "pelvis"

    # Position of target reference frame
    x: float = 0.0
    y: float = 0.0
    z: float = 0.9

    # Orientation of target reference frame, radians
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return TargetFrame(**data)


class Trajectory:
    def __init__(self):
        # A trajectory is no longer one mixed path. It is multiple named
        # timelines, one per robot frame/body part.
        self.tracks = {name: [] for name in DEFAULT_TRACK_NAMES}

    @property
    def frames(self):
        """
        Compatibility view for existing viewers and table code.

        The internal source of truth is self.tracks; this property returns all
        keyframes flattened and sorted for display/drawing only.
        """
        return self.all_frames()

    def ensure_track(self, frame_name):
        if frame_name not in self.tracks:
            self.tracks[frame_name] = []

    def all_frames(self):
        frames = []

        for track in self.tracks.values():
            frames.extend(track)

        return sorted(frames, key=lambda f: (f.time, f.frame_name))

    def _sort_track(self, frame_name):
        self.tracks[frame_name].sort(key=lambda f: f.time)

    def _locate_flat_index(self, index):
        frames = self.all_frames()

        if index < 0 or index >= len(frames):
            return None, None

        frame = frames[index]
        track = self.tracks.get(frame.frame_name, [])

        for track_index, candidate in enumerate(track):
            if candidate is frame:
                return frame.frame_name, track_index

        return None, None

    def index_of_frame(self, frame):
        for index, candidate in enumerate(self.all_frames()):
            if candidate is frame:
                return index

        return -1

    def add_frame(self, frame):
        """
        Add one keyframe to the matching frame-name track.
        """
        self.ensure_track(frame.frame_name)
        self.tracks[frame.frame_name].append(frame)
        self._sort_track(frame.frame_name)
        return self.index_of_frame(frame)

    def upsert_frame(self, frame, tolerance=1e-6):
        """Insert or replace one logical target keyframe at the same time."""
        self.ensure_track(frame.frame_name)
        track = self.tracks[frame.frame_name]
        for index, existing in enumerate(track):
            if abs(existing.time - frame.time) <= tolerance:
                track[index] = frame
                self._sort_track(frame.frame_name)
                return self.index_of_frame(frame)
        return self.add_frame(frame)

    def update_frame(self, index, frame):
        """
        Replace an existing keyframe selected from the flattened display.

        If the frame name changed, the keyframe moves to the new named track.
        """
        old_frame_name, track_index = self._locate_flat_index(index)

        if old_frame_name is None:
            return

        del self.tracks[old_frame_name][track_index]
        self.add_frame(frame)

    def delete_frame(self, index):
        """
        Delete a keyframe selected from the flattened display.
        """
        frame_name, track_index = self._locate_flat_index(index)

        if frame_name is None:
            return

        del self.tracks[frame_name][track_index]

    def clear(self):
        for track in self.tracks.values():
            track.clear()

    def as_list(self):
        """
        Flattened serialization for compatibility with older saved files.
        """
        return [frame.to_dict() for frame in self.all_frames()]

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.as_list(), f, indent=4)

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.clear()

        for item in data:
            self.add_frame(TargetFrame.from_dict(item))

    # ============================================================
    # Uniform-dt sampling
    # ============================================================

    def sample_uniform_dt(self, dt=0.01):
        """
        Backward-compatible flattened sampling.

        Prefer sample_tracks_uniform_dt() for whole-body trajectories. This
        method samples each named track independently, then flattens the result
        for older code that expects a list of TargetFrame objects.
        """
        sampled_frames = []

        for sample in self.sample_tracks_uniform_dt(dt=dt):
            sampled_frames.extend(sample["targets"].values())

        return sorted(sampled_frames, key=lambda f: (f.time, f.frame_name))

    def sample_tracks_uniform_dt(self, dt=0.01):
        """
        Sample each robot frame/body-part track independently.

        Pelvis keyframes interpolate only with pelvis keyframes, left_foot
        only with left_foot, and so on. The returned list is grouped by sampled
        time:

            {"time": t, "targets": {"pelvis": TargetFrame(...), ...}}
        """
        non_empty_tracks = {
            name: sorted(track, key=lambda f: f.time)
            for name, track in self.tracks.items()
            if track
        }

        if not non_empty_tracks:
            return []

        t_start = min(track[0].time for track in non_empty_tracks.values())
        t_end = max(track[-1].time for track in non_empty_tracks.values())
        num_steps = int(round((t_end - t_start) / dt))
        samples = []

        for k in range(num_steps + 1):
            t = t_start + k * dt
            t = round(t, 10)

            if t > t_end:
                t = t_end

            targets = {}

            for frame_name, track in non_empty_tracks.items():
                if t < track[0].time - 1e-9 or t > track[-1].time + 1e-9:
                    continue

                f0, f1 = self.find_surrounding_keyframes(track, t)
                targets[frame_name] = self.interpolate_frames(f0, f1, t)

            samples.append({
                "time": t,
                "targets": targets,
            })

        return samples

    def targets_at_time(self, t):
        """
        Return interpolated targets grouped by frame name at one time.

        This is the single-time version of sample_tracks_uniform_dt(). Each
        robot frame/body part is interpolated only within its own named track;
        pelvis never interpolates with hands, feet, torso, etc.
        """
        targets = {}

        for frame_name, track in self.tracks.items():
            if not track:
                continue

            sorted_track = sorted(track, key=lambda f: f.time)

            if t < sorted_track[0].time - 1e-9:
                continue

            if t > sorted_track[-1].time + 1e-9:
                continue

            f0, f1 = self.find_surrounding_keyframes(sorted_track, t)
            targets[frame_name] = self.interpolate_frames(f0, f1, t)

        return targets

    def find_surrounding_keyframes(self, sorted_frames, t):
        """
        Find keyframes f0 and f1 such that:

            f0.time <= t <= f1.time
        """

        if t <= sorted_frames[0].time:
            return sorted_frames[0], sorted_frames[0]

        if t >= sorted_frames[-1].time:
            return sorted_frames[-1], sorted_frames[-1]

        for i in range(len(sorted_frames) - 1):
            f0 = sorted_frames[i]
            f1 = sorted_frames[i + 1]

            if f0.time <= t <= f1.time:
                return f0, f1

        return sorted_frames[-1], sorted_frames[-1]

    def interpolate_frames(self, f0, f1, t):
        """
        Interpolate between two TargetFrames.

        Position:
            Linear interpolation.

        Orientation:
            roll/pitch/yaw -> quaternion
            SLERP quaternion
            quaternion -> roll/pitch/yaw
        """

        if abs(f1.time - f0.time) < 1e-9:
            alpha = 0.0
        else:
            alpha = (t - f0.time) / (f1.time - f0.time)

        alpha = max(0.0, min(1.0, alpha))

        def lerp(a, b):
            return a + alpha * (b - a)

        # Position interpolation
        x = lerp(f0.x, f1.x)
        y = lerp(f0.y, f1.y)
        z = lerp(f0.z, f1.z)

        # Orientation interpolation using SLERP
        q0 = rpy_to_quat(f0.roll, f0.pitch, f0.yaw)
        q1 = rpy_to_quat(f1.roll, f1.pitch, f1.yaw)
        q_interp = slerp(q0, q1, alpha)

        roll, pitch, yaw = quat_to_rpy(q_interp)

        return TargetFrame(
            time=t,

            # For phase/frame_name, use previous keyframe.
            # Later, you can make this phase-aware.
            phase=f0.phase,
            frame_name=f0.frame_name,

            x=x,
            y=y,
            z=z,

            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )


class SampledTrajectory:
    def __init__(self, frames=None, samples=None):
        self.samples = samples
        self.frames = frames if frames is not None else self.flatten_samples(samples)

    def flatten_samples(self, samples):
        if samples is None:
            return []

        frames = []

        for sample in samples:
            frames.extend(sample["targets"].values())

        return sorted(frames, key=lambda f: (f.time, f.frame_name))

    def as_list(self):
        return [frame.to_dict() for frame in self.frames]

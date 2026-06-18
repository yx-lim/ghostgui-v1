"""
trajectory.py

Purpose:
    Stores target reference-frame keyframes.

Updated project idea:
    The GUI does not directly control the robot.
    The GUI creates a trajectory array of target reference frames.

Each TargetFrame says:
    At this time, this robot frame should be at this pose.
"""

from dataclasses import dataclass, asdict
import json
import math


@dataclass
class TargetFrame:
    time: float = 0.0
    phase: str = "crouch"
    frame_name: str = "pelvis"

    # Position of target reference frame
    x: float = 0.0
    y: float = 0.0
    z: float = 0.9

    # Orientation of target reference frame
    # For the 2D prototype, yaw is most important.
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
        self.frames = []

    def add_frame(self, frame):
        """
        Add one keyframe and keep trajectory sorted by time.
        """
        self.frames.append(frame)
        self.frames.sort(key=lambda f: f.time)
        return self.frames.index(frame)

    def update_frame(self, index, frame):
        """
        Replace an existing keyframe.
        """
        if index < 0 or index >= len(self.frames):
            return

        self.frames[index] = frame
        self.frames.sort(key=lambda f: f.time)

    def delete_frame(self, index):
        """
        Delete a keyframe.
        """
        if index < 0 or index >= len(self.frames):
            return

        del self.frames[index]

    def clear(self):
        self.frames.clear()

    def as_list(self):
        """
        This is the trajectory array that gets passed to the backend.
        """
        return [frame.to_dict() for frame in self.frames]

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.as_list(), f, indent=4)

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.frames = [TargetFrame.from_dict(item) for item in data]
        self.frames.sort(key=lambda f: f.time)

    # ============================================================
    # New: uniform-dt sampling
    # ============================================================

    def sample_uniform_dt(self, dt=0.01):
        """
        Convert sparse GUI keyframes into a uniformly sampled trajectory.

        Example:
            keyframes at t = 0.0, 0.7, 1.4
            dt = 0.01

        Output:
            sampled frames at t = 0.00, 0.01, 0.02, ..., 1.40

        Current version:
            - assumes one frame type, e.g. pelvis
            - linear interpolation for x, y, z, roll, pitch, yaw
            - phase and frame_name are copied from the previous keyframe
        """

        if len(self.frames) == 0:
            return []

        if len(self.frames) == 1:
            return [self.frames[0]]

        sorted_frames = sorted(self.frames, key=lambda f: f.time)

        t_start = sorted_frames[0].time
        t_end = sorted_frames[-1].time

        sampled_frames = []

        # Number of samples on a uniform grid.
        num_steps = int(round((t_end - t_start) / dt))

        for k in range(num_steps + 1):
            t = t_start + k * dt

            # Avoid floating point drift in printed CSV.
            t = round(t, 10)

            # Clamp last sample exactly to final keyframe time.
            if t > t_end:
                t = t_end

            f0, f1 = self.find_surrounding_keyframes(sorted_frames, t)

            sampled = self.interpolate_frames(f0, f1, t)
            sampled_frames.append(sampled)

        return sampled_frames

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
        Linear interpolation between two TargetFrames.
        """

        if abs(f1.time - f0.time) < 1e-9:
            alpha = 0.0
        else:
            alpha = (t - f0.time) / (f1.time - f0.time)

        def lerp(a, b):
            return a + alpha * (b - a)

        return TargetFrame(
            time=t,

            # For phase/frame_name, use previous keyframe.
            # Later, you can make this phase-aware.
            phase=f0.phase,
            frame_name=f0.frame_name,

            x=lerp(f0.x, f1.x),
            y=lerp(f0.y, f1.y),
            z=lerp(f0.z, f1.z),

            roll=lerp(f0.roll, f1.roll),
            pitch=lerp(f0.pitch, f1.pitch),
            yaw=lerp(f0.yaw, f1.yaw),
        )

class SampledTrajectory:
    def __init__(self, frames):
        self.frames = frames

    def as_list(self):
        return [frame.to_dict() for frame in self.frames]        
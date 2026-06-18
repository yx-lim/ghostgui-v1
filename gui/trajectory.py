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
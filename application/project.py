"""Serializable project-level state shared by the editors."""

from __future__ import annotations

from dataclasses import dataclass

from core.trajectory.model import Trajectory


@dataclass
class ProjectDocument:
    """Keep target and qpos timelines related without conflating their meaning."""

    model_key: str
    target_trajectory: Trajectory
    robot_state_timeline: object | None = None


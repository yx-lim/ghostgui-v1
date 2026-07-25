"""Plain-Python model session cache helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotModelSession:
    adapter: object
    backend: object
    reference: object
    viewer_3d: object
    trajectory: object
    active_index: int = -1


def remember_current_session(model_sessions, model_key, trajectory, active_index):
    current = model_sessions.get(model_key)
    if current is not None:
        current.trajectory = trajectory
        current.active_index = active_index


def activated_session_state(model_key, session):
    return {
        "model_key": model_key,
        "robot_model_3d": session.adapter,
        "robot_model_error": session.adapter.load_warning,
        "backend_interface": session.backend,
        "model_reference": session.reference,
        "viewer_3d": session.viewer_3d,
        "trajectory": session.trajectory,
        "active_index": session.active_index,
    }

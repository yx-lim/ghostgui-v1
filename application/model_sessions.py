"""Plain-Python model session cache helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotModelSession:
    adapter: object
    backend: object
    reference: object
    viewer_3d: object
    viewer_2d_skeleton: object
    trajectory: object
    active_index: int = -1
    actor_id: str | None = None
    model_key: str | None = None
    selected_frame: str | None = None


def session_key(actor_id, model_key):
    return str(actor_id), str(model_key)


def remember_current_session(model_sessions, key, trajectory, active_index):
    current = model_sessions.get(key)
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
        "viewer_2d_stickman": session.viewer_2d_skeleton,
        "trajectory": session.trajectory,
        "active_index": session.active_index,
    }

"""Committed/preview robot state and qpos timeline ownership."""

from core.models.model import RobotStateTimeline


class PreviewTimelineController:
    """Own state transitions; widgets remain responsible for visual feedback."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.committed_state = robot_model.create_state() if robot_model else None
        self.preview_state = robot_model.create_state() if robot_model else None
        self.current_time = 0.0
        self.preview_active = False
        self.timeline = (
            RobotStateTimeline(
                robot_model, initial_qpos=self.committed_state.get_qpos()
            )
            if robot_model else None
        )

    def begin_preview(self):
        if self.committed_state is None or self.preview_active:
            return False
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = True
        return True

    def accept_preview(self):
        if not self.preview_active:
            return False
        self.committed_state.set_qpos(self.preview_state.get_qpos())
        self.timeline.set_state(self.current_time, self.committed_state.get_qpos())
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        return True

    def cancel_preview(self):
        if self.committed_state is None:
            return False
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        was_active = self.preview_active
        self.preview_active = False
        return was_active

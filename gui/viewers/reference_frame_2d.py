"""
viewer_2d.py

Purpose:
    Side-view 2D editor for the target reference frame and trajectory.

Behavior:
    - Draws a ground line
    - Draws the target reference frame as a red marker
    - Allows dragging the target frame in X/Z
    - Draws the stored trajectory keyframes
"""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPen, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene

from ..trajectory_colors import qt_color_for_frame

TRAJECTORY_LINE_DT = 0.02


class RobotCanvas(QGraphicsView):
    target_dragged = Signal(float, float)

    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setMinimumSize(650, 500)
        self.setSceneRect(-325, -250, 650, 500)

        self.scale_pixels_per_meter = 180

        self.target_x = 0.0
        self.target_z = 0.9
        self.target_yaw = 0.0

        self.dragging_target = False

    # ============================================================
    # Coordinate transforms
    # ============================================================

    def world_to_screen(self, x, z):
        sx = x * self.scale_pixels_per_meter
        sy = -z * self.scale_pixels_per_meter + 180
        return sx, sy

    def screen_to_world(self, sx, sy):
        x = sx / self.scale_pixels_per_meter
        z = -(sy - 180) / self.scale_pixels_per_meter
        return x, z

    # ============================================================
    # Main draw function
    # ============================================================

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        show_trajectory_lines=True,
        trajectory_smoothing=0.0,
    ):
        self.scene.clear()

        if active_frame is not None:
            self.target_x = active_frame.x
            self.target_z = active_frame.z
            self.target_yaw = active_frame.yaw

        self.draw_ground()
        self.draw_trajectory(
            trajectory,
            show_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
        )
        self.draw_target_frame()
        self.draw_legend()

    def draw_ground(self):
        ground_y = self.world_to_screen(0.0, 0.0)[1]
        self.scene.addLine(-325, ground_y, 325, ground_y, QPen(Qt.GlobalColor.black, 2))

    def draw_trajectory(
        self,
        trajectory,
        show_lines=True,
        trajectory_smoothing=0.0,
    ):
        """
        Draw stored trajectory keyframes and sampled connecting lines.
        """

        if len(trajectory.frames) == 0:
            return

        if show_lines:
            self.draw_sampled_trajectory_lines(trajectory, trajectory_smoothing)

        for frame in trajectory.frames:
            x, y = self.world_to_screen(frame.x, frame.z)
            color = qt_color_for_frame(frame.frame_name)
            pen_point = QPen(color, 2)
            brush_point = QBrush(color)

            self.scene.addEllipse(
                x - 5,
                y - 5,
                10,
                10,
                pen_point,
                brush_point,
            )

            self.scene.addText(f"{frame.time:.1f}s").setPos(x + 6, y - 18)

    def draw_sampled_trajectory_lines(self, trajectory, trajectory_smoothing):
        samples = trajectory.sample_tracks_uniform_dt(
            dt=TRAJECTORY_LINE_DT,
            smoothing=trajectory_smoothing,
        )
        previous_by_frame = {}

        for sample in samples:
            for frame_name, target in sample["targets"].items():
                previous = previous_by_frame.get(frame_name)
                if previous is not None:
                    px, py = self.world_to_screen(previous.x, previous.z)
                    x, y = self.world_to_screen(target.x, target.z)
                    self.scene.addLine(
                        px,
                        py,
                        x,
                        y,
                        QPen(qt_color_for_frame(frame_name), 2),
                    )
                previous_by_frame[frame_name] = target

    def draw_target_frame(self):
        """
        Draw the currently edited target reference frame.

        Red dot = origin of target frame.
        Red arrow = x-axis direction of target frame.
        """

        x, y = self.world_to_screen(self.target_x, self.target_z)

        pen_target = QPen(Qt.GlobalColor.red, 3)
        brush_target = QBrush(Qt.GlobalColor.red)

        # Origin marker
        self.scene.addEllipse(
            x - 9,
            y - 9,
            18,
            18,
            pen_target,
            brush_target,
        )

        # Local x-axis arrow
        arrow_len = 45
        x2 = x + arrow_len * math.cos(self.target_yaw)
        y2 = y - arrow_len * math.sin(self.target_yaw)

        self.scene.addLine(x, y, x2, y2, pen_target)
        self.scene.addText("target frame").setPos(x + 12, y + 12)

    def draw_legend(self):
        self.scene.addText("Red = currently edited target reference frame").setPos(-315, -240)
        self.scene.addText("Colored = stored per-frame keyframes").setPos(-315, -215)

    # ============================================================
    # Mouse dragging
    # ============================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())

            target_sx, target_sy = self.world_to_screen(self.target_x, self.target_z)

            dx = scene_pos.x() - target_sx
            dy = scene_pos.y() - target_sy

            distance = math.sqrt(dx * dx + dy * dy)

            if distance < 25:
                self.dragging_target = True
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_target:
            scene_pos = self.mapToScene(event.position().toPoint())

            x, z = self.screen_to_world(scene_pos.x(), scene_pos.y())

            # Keep target above ground.
            z = max(0.0, z)

            self.target_x = x
            self.target_z = z

            self.target_dragged.emit(x, z)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_target = False
        super().mouseReleaseEvent(event)

"""
viewer_3d.py

Purpose:
    OpenGL 3D viewer/editor for the target reference frame and trajectory.

The editor shares the same small contract as the 2D side view:
    - update_scene(trajectory, active_frame)
    - target_dragged(x, z)
"""

import math

from OpenGL import GL
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .trajectory_colors import gl_color_for_frame


class RobotCanvas3D(QOpenGLWidget):
    target_dragged = Signal(float, float)

    def __init__(self):
        super().__init__()

        self.setMinimumSize(650, 500)
        self.setMouseTracking(True)

        self.trajectory = None
        self.show_trajectory_lines = True

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.9
        self.target_yaw = 0.0

        self.camera_distance = 5.0
        self.camera_yaw = 38.0
        self.camera_pitch = 24.0

        self.dragging_target = False
        self.rotating_camera = False
        self.last_mouse_pos = None

        self._model_view = QMatrix4x4()
        self._projection = QMatrix4x4()
        self._viewport = QRect(0, 0, 1, 1)

    # ============================================================
    # Scene API
    # ============================================================

    def update_scene(self, trajectory, active_frame=None, show_trajectory_lines=True):
        self.trajectory = trajectory
        self.show_trajectory_lines = show_trajectory_lines

        if active_frame is not None:
            self.target_x = active_frame.x
            self.target_y = active_frame.y
            self.target_z = active_frame.z
            self.target_yaw = active_frame.yaw

        self.update()

    # ============================================================
    # OpenGL lifecycle
    # ============================================================

    def initializeGL(self):
        GL.glClearColor(0.08, 0.09, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(8.0)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.configure_camera()

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(self.matrix_values(self._projection))

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(self.matrix_values(self._model_view))

        self.draw_ground_grid()
        self.draw_world_axes()
        self.draw_trajectory()
        self.draw_target_frame()

    def configure_camera(self):
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height

        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)

        eye = QVector3D(
            self.camera_distance * math.cos(pitch) * math.sin(yaw),
            -self.camera_distance * math.cos(pitch) * math.cos(yaw),
            self.camera_distance * math.sin(pitch) + 1.1,
        )
        center = QVector3D(0.0, 0.0, 0.75)
        up = QVector3D(0.0, 0.0, 1.0)

        self._projection = QMatrix4x4()
        self._projection.perspective(45.0, aspect, 0.05, 100.0)

        self._model_view = QMatrix4x4()
        self._model_view.lookAt(eye, center, up)

        self._viewport = QRect(0, 0, width, height)

    def matrix_values(self, matrix):
        data = matrix.data()
        return [data[i] for i in range(16)]

    # ============================================================
    # Drawing helpers
    # ============================================================

    def draw_ground_grid(self):
        GL.glLineWidth(1.0)
        GL.glColor3f(0.30, 0.33, 0.35)

        GL.glBegin(GL.GL_LINES)
        for i in range(-10, 11):
            v = i * 0.25
            GL.glVertex3f(-2.5, v, 0.0)
            GL.glVertex3f(2.5, v, 0.0)
            GL.glVertex3f(v, -2.5, 0.0)
            GL.glVertex3f(v, 2.5, 0.0)
        GL.glEnd()

        GL.glLineWidth(2.0)
        GL.glColor3f(0.55, 0.58, 0.60)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(-2.5, 0.0, 0.0)
        GL.glVertex3f(2.5, 0.0, 0.0)
        GL.glVertex3f(0.0, -2.5, 0.0)
        GL.glVertex3f(0.0, 2.5, 0.0)
        GL.glEnd()

    def draw_world_axes(self):
        GL.glLineWidth(3.0)
        GL.glBegin(GL.GL_LINES)

        GL.glColor3f(0.90, 0.15, 0.12)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.6, 0.0, 0.02)

        GL.glColor3f(0.20, 0.75, 0.25)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.6, 0.02)

        GL.glColor3f(0.20, 0.45, 0.95)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.0, 0.6)

        GL.glEnd()

    def draw_trajectory(self):
        if self.trajectory is None or len(self.trajectory.frames) == 0:
            return

        if self.show_trajectory_lines:
            frames_by_name = {}
            for frame in self.trajectory.frames:
                frames_by_name.setdefault(frame.frame_name, []).append(frame)

            GL.glLineWidth(2.0)

            for frame_name, frames in frames_by_name.items():
                if len(frames) < 2:
                    continue

                GL.glColor3f(*gl_color_for_frame(frame_name))
                GL.glBegin(GL.GL_LINE_STRIP)
                for frame in sorted(frames, key=lambda f: f.time):
                    GL.glVertex3f(frame.x, frame.y, frame.z)
                GL.glEnd()

        GL.glPointSize(7.0)
        GL.glBegin(GL.GL_POINTS)
        for frame in self.trajectory.frames:
            GL.glColor3f(*gl_color_for_frame(frame.frame_name))
            GL.glVertex3f(frame.x, frame.y, frame.z)
        GL.glEnd()

    def draw_target_frame(self):
        x = self.target_x
        y = self.target_y
        z = self.target_z

        GL.glPointSize(12.0)
        GL.glColor3f(1.0, 0.12, 0.08)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(x, y, z)
        GL.glEnd()

        arrow_len = 0.35
        yaw = self.target_yaw

        GL.glLineWidth(4.0)
        GL.glColor3f(1.0, 0.12, 0.08)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(x, y, z)
        GL.glVertex3f(
            x + arrow_len * math.cos(yaw),
            y + arrow_len * math.sin(yaw),
            z,
        )
        GL.glEnd()

        GL.glLineWidth(2.0)
        GL.glColor3f(1.0, 0.35, 0.30)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(x, y, 0.0)
        GL.glVertex3f(x, y, z)
        GL.glEnd()

    # ============================================================
    # Mouse editing
    # ============================================================

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton:
            target_sx, target_sy = self.project_point(
                self.target_x,
                self.target_y,
                self.target_z,
            )
            dx = event.position().x() - target_sx
            dy = event.position().y() - target_sy
            if math.sqrt(dx * dx + dy * dy) < 32:
                self.dragging_target = True
                return

        if event.button() == Qt.MouseButton.RightButton:
            self.rotating_camera = True
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_target:
            x, z = self.screen_to_edit_plane(event.position().x(), event.position().y())

            self.target_x = x
            self.target_z = max(0.0, z)
            self.target_dragged.emit(self.target_x, self.target_z)
            self.update()
            return

        if self.rotating_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.camera_yaw += delta.x() * 0.4
            self.camera_pitch = max(-5.0, min(80.0, self.camera_pitch + delta.y() * 0.3))
            self.last_mouse_pos = event.position()
            self.update()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_target = False
        self.rotating_camera = False
        self.last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.camera_distance = max(1.8, min(10.0, self.camera_distance - steps * 0.35))
        self.update()

    # ============================================================
    # Projection helpers
    # ============================================================

    def project_point(self, x, y, z):
        self.configure_camera()
        point = QVector3D(x, y, z)
        screen = point.project(self._model_view, self._projection, self._viewport)
        return screen.x(), self.height() - screen.y()

    def screen_to_edit_plane(self, sx, sy):
        """
        Convert a screen point to the Y=0 edit plane.

        The reference-frame editor currently exposes X/Z controls, so the 3D
        editor drags the target within that same side-view plane.
        """

        self.configure_camera()

        near = QVector3D(sx, self.height() - sy, 0.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )
        far = QVector3D(sx, self.height() - sy, 1.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )

        direction = far - near
        if abs(direction.y()) < 1e-6:
            return self.target_x, self.target_z

        t = -near.y() / direction.y()
        point = near + direction * t
        return point.x(), point.z()

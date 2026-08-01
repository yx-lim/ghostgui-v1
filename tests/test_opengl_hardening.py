"""Desktop OpenGL compatibility and high-DPI rendering contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from gui.viewers.opengl_compat import (
    compatibility_context_failure,
    desktop_compatibility_format,
)
from gui.viewers.robot_canvas_3d import RobotCanvas3D


class _Context:
    def __init__(self, surface_format, *, valid=True, gles=False):
        self._format = surface_format
        self._valid = valid
        self._gles = gles

    def format(self):
        return self._format

    def isValid(self):
        return self._valid

    def isOpenGLES(self):
        return self._gles


def _actual_format(
    version=(2, 1),
    profile=QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile,
    depth=24,
    deprecated=False,
):
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(*version)
    surface_format.setProfile(profile)
    surface_format.setDepthBufferSize(depth)
    surface_format.setOption(
        QSurfaceFormat.FormatOption.DeprecatedFunctions,
        on=deprecated,
    )
    return surface_format


class OpenGLFormatTests(unittest.TestCase):
    def test_requested_format_is_desktop_21_compatibility_with_depth(self):
        surface_format = desktop_compatibility_format()

        self.assertEqual(
            surface_format.renderableType(),
            QSurfaceFormat.RenderableType.OpenGL,
        )
        self.assertEqual(
            (surface_format.majorVersion(), surface_format.minorVersion()),
            (2, 1),
        )
        self.assertEqual(
            surface_format.profile(),
            QSurfaceFormat.OpenGLContextProfile.NoProfile,
        )
        self.assertEqual(surface_format.depthBufferSize(), 24)
        self.assertTrue(
            surface_format.testOption(
                QSurfaceFormat.FormatOption.DeprecatedFunctions
            )
        )

    def test_realized_context_rejects_es_core_profile_and_missing_depth(self):
        self.assertIn(
            "OpenGL ES",
            compatibility_context_failure(
                _Context(_actual_format(), gles=True)
            ),
        )
        self.assertIn(
            "core-only",
            compatibility_context_failure(
                _Context(
                    _actual_format(
                        (4, 5),
                        QSurfaceFormat.OpenGLContextProfile.CoreProfile,
                    )
                )
            ),
        )
        self.assertIn(
            "no depth buffer",
            compatibility_context_failure(
                _Context(_actual_format(depth=0))
            ),
        )
        self.assertIsNone(
            compatibility_context_failure(
                _Context(_actual_format((4, 5)))
            )
        )
        self.assertIn(
            "core-only",
            compatibility_context_failure(
                _Context(
                    _actual_format(
                        (4, 1),
                        QSurfaceFormat.OpenGLContextProfile.NoProfile,
                    )
                )
            ),
        )
        self.assertIn(
            "deprecated fixed-function",
            compatibility_context_failure(
                _Context(_actual_format((3, 1)))
            ),
        )
        self.assertIsNone(
            compatibility_context_failure(
                _Context(_actual_format((3, 1), deprecated=True))
            )
        )


class OpenGLCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_canvas_requests_compatible_format_for_direct_construction(self):
        canvas = RobotCanvas3D()
        try:
            surface_format = canvas.format()
            self.assertEqual(
                surface_format.renderableType(),
                QSurfaceFormat.RenderableType.OpenGL,
            )
            self.assertEqual(
                (surface_format.majorVersion(), surface_format.minorVersion()),
                (2, 1),
            )
            self.assertEqual(
                surface_format.profile(),
                QSurfaceFormat.OpenGLContextProfile.NoProfile,
            )
            self.assertGreaterEqual(surface_format.depthBufferSize(), 24)
            self.assertEqual(surface_format.alphaBufferSize(), 0)
        finally:
            canvas.shutdown()

    def test_rendering_failure_is_signaled_and_visible_once(self):
        canvas = RobotCanvas3D()
        failures = []
        canvas.rendering_failed.connect(failures.append)
        try:
            canvas._report_rendering_failure("core-only context")
            canvas._report_rendering_failure("core-only context")

            self.assertEqual(len(failures), 1)
            self.assertIn("desktop OpenGL 2.1", failures[0])
            self.assertFalse(canvas._rendering_failure_label.isHidden())
            self.assertEqual(canvas._rendering_failure_label.text(), failures[0])
        finally:
            canvas.shutdown()

    def test_viewport_and_presentation_sizes_scale_and_clamp(self):
        canvas = RobotCanvas3D()
        canvas._max_viewport_dimensions = (500, 300)
        canvas._line_width_range = (1.0, 4.0)
        canvas._point_size_range = (2.0, 12.0)
        try:
            with patch.object(canvas, "devicePixelRatioF", return_value=2.0):
                self.assertEqual(canvas._physical_viewport_size(400, 200), (500, 300))
                with (
                    patch(
                        "gui.viewers.robot_canvas_3d.GL.glViewport"
                    ) as viewport,
                    patch(
                        "gui.viewers.robot_canvas_3d.GL.glLineWidth"
                    ) as line_width,
                    patch(
                        "gui.viewers.robot_canvas_3d.GL.glPointSize"
                    ) as point_size,
                ):
                    canvas.resizeGL(400, 200)
                    canvas._set_line_width(3.0)
                    canvas._set_point_size(7.0)

            viewport.assert_called_once_with(0, 0, 500, 300)
            line_width.assert_called_once_with(4.0)
            point_size.assert_called_once_with(12.0)
        finally:
            canvas.shutdown()

    def test_each_paint_reasserts_fixed_function_state(self):
        canvas = RobotCanvas3D()
        try:
            canvas.transform_gizmo_visible = False
            with (
                patch.object(canvas, "_reset_fixed_function_state") as reset,
                patch.object(canvas, "configure_camera"),
                patch.object(canvas, "draw_ground_grid"),
                patch.object(canvas, "draw_world_axes"),
                patch.object(canvas, "draw_robot"),
                patch.object(canvas, "draw_trajectory_ghosts"),
                patch.object(canvas, "draw_preview_robot"),
                patch.object(canvas, "draw_trajectory"),
                patch.object(canvas, "draw_selected_target_marker"),
                patch("gui.viewers.robot_canvas_3d.GL"),
            ):
                canvas.paintGL()
                canvas.paintGL()

            self.assertEqual(reset.call_count, 2)
            self.assertIsNone(canvas._rendering_failure)
        finally:
            canvas.shutdown()

    def test_cleanup_drops_handles_when_driver_deletion_fails(self):
        canvas = RobotCanvas3D()
        canvas._geom_lists = [7]
        canvas._mesh_display_lists = {1: 7}
        canvas._quadric = object()
        try:
            with (
                patch(
                    "gui.viewers.robot_canvas_3d.GL.glDeleteLists",
                    side_effect=RuntimeError("context lost"),
                ),
                patch(
                    "gui.viewers.robot_canvas_3d.GLU.gluDeleteQuadric",
                    side_effect=RuntimeError("context lost"),
                ),
            ):
                canvas.cleanup_gl_resources(context_current=True)

            self.assertEqual(canvas._geom_lists, [])
            self.assertEqual(canvas._mesh_display_lists, {})
            self.assertIsNone(canvas._quadric)
        finally:
            canvas.shutdown()


if __name__ == "__main__":
    unittest.main()

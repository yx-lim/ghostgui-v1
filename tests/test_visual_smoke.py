"""Software-OpenGL visual smoke and stable layout regression checks."""

from __future__ import annotations

import os
import unittest

from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.viewers.opengl_compat import configure_default_surface_format
from gui.main_window import RobotGuiMainWindow


VISUAL_TESTS_ENABLED = os.environ.get("GHOSTGUI_VISUAL_TESTS") == "1"


@unittest.skipUnless(
    VISUAL_TESTS_ENABLED,
    "set GHOSTGUI_VISUAL_TESTS=1 and run under Xvfb",
)
class MainWindowVisualSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_default_surface_format()
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_renders_and_matches_structural_baseline(self):
        self.assertNotEqual(QGuiApplication.platformName(), "offscreen")
        window = RobotGuiMainWindow("g1")
        try:
            window.resize(1200, 800)
            window.show()
            for _ in range(250):
                self.app.processEvents()
                canvas = window.viewer_3d.canvas
                if (
                    canvas.isValid()
                    and canvas._geometry_build_count > 0
                    and canvas._rendering_failure is None
                ):
                    break
                QTest.qWait(20)

            self.assertTrue(window.isVisible())
            self.assertTrue(window.viewer_3d.canvas.isValid())
            self.assertIsNone(window.viewer_3d.canvas._rendering_failure)
            self.assertGreater(window.viewer_3d.canvas._geometry_build_count, 0)
            self.assertEqual(
                [window.viewer_tabs.tabText(index) for index in range(2)],
                ["3D Pose", "Simulation"],
            )
            self.assertEqual(
                [
                    section.title
                    for section in window.left_sidebar_content.sections
                ],
                ["Target", "Editing Mode", "Planning"],
            )
            self.assertEqual(
                [
                    section.title
                    for section in window.right_sidebar_content.sections
                ],
                ["Status", "IK / Constraints"],
            )
            self.assertEqual(
                [
                    action.text()
                    for action in (
                        window.preview_action,
                        window.slice_action,
                        window.generate_action,
                        window.playback_action,
                        window.reset_action,
                        window.clear_action,
                        window.move_action,
                        window.rotate_action,
                    )
                ],
                [
                    "Preview Path",
                    "Commit Keyframe",
                    "Generate",
                    "Play",
                    "Reset",
                    "Clear",
                    "Move",
                    "Rotate",
                ],
            )

            image = window.grab().toImage()
            self.assertFalse(image.isNull())
            self.assertGreaterEqual(image.width(), 1100)
            self.assertGreaterEqual(image.height(), 700)
            colors = {
                image.pixelColor(x, y).rgba()
                for x in range(0, image.width(), 30)
                for y in range(0, image.height(), 30)
            }
            self.assertGreater(
                len(colors),
                12,
                "captured window is visually blank or failed to composite",
            )
            self.assertEqual(image.pixelColor(0, 0).alpha(), 255)

            canvas_image = window.viewer_3d.canvas.grabFramebuffer()
            self.assertFalse(canvas_image.isNull())
            canvas_colors = {
                canvas_image.pixelColor(x, y).rgba()
                for x in range(0, canvas_image.width(), 24)
                for y in range(0, canvas_image.height(), 24)
            }
            self.assertGreater(
                len(canvas_colors),
                8,
                "3D canvas framebuffer is blank or failed to render geometry",
            )
        finally:
            window.current_project = None
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()

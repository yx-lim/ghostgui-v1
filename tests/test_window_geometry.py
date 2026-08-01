"""Cross-platform window sizing contracts."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QSize

from gui.window_geometry import bounded_window_size


class WindowGeometryTests(unittest.TestCase):
    def test_preferred_size_is_bounded_by_screen_margin(self):
        self.assertEqual(
            bounded_window_size(QSize(840, 560), QSize(640, 480), margin=24),
            QSize(592, 432),
        )

    def test_small_or_invalid_dimensions_remain_positive(self):
        self.assertEqual(
            bounded_window_size(QSize(0, -10), QSize(20, 10), margin=24),
            QSize(1, 1),
        )

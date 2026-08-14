import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider

from gui.widgets.timeline import TimesliceSlider


def _report(time, *, blocking):
    return SimpleNamespace(time=time, blocking=blocking)


class TimelineSafetyMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.slider = TimesliceSlider(Qt.Orientation.Horizontal)
        self.slider.resize(420, 30)
        # The displayed timeline covers 1.00 s through 9.00 s.
        self.slider.setRange(100, 900)
        self.slider.setValue(100)
        self.slider.show()
        QApplication.processEvents()

    def tearDown(self):
        self.slider.close()

    def _render(self):
        return self.slider.grab().toImage()

    def test_reports_map_to_range_and_paint_amber_and_red_ticks(self):
        advisory = _report(3.0, blocking=False)
        blocking = _report(7.0, blocking=True)

        self.slider.set_safety_reports((advisory, blocking))
        QApplication.processEvents()

        self.assertEqual(self.slider.safety_reports, (advisory, blocking))
        self.assertEqual(
            [(marker.time, marker.blocking) for marker in self.slider.safety_markers],
            [(3.0, False), (7.0, True)],
        )

        option = QStyleOptionSlider()
        self.slider.initStyleOption(option)
        groove = self.slider._groove_rect()
        handle = self.slider._handle_rect()
        span = max(1, groove.width() - handle.width())

        def expected_x(time):
            offset = QStyle.sliderPositionFromValue(
                self.slider.minimum(),
                self.slider.maximum(),
                int(round(time * 100.0)),
                span,
                option.upsideDown,
            )
            return groove.x() + handle.width() // 2 + offset

        advisory_x = self.slider._time_to_pixel(advisory.time)
        blocking_x = self.slider._time_to_pixel(blocking.time)
        self.assertEqual(advisory_x, expected_x(advisory.time))
        self.assertEqual(blocking_x, expected_x(blocking.time))

        image = self._render()
        advisory_rect = self.slider._safety_marker_rect(advisory_x)
        blocking_rect = self.slider._safety_marker_rect(blocking_x)
        self.assertEqual(
            image.pixelColor(advisory_rect.center()),
            self.slider.ADVISORY_MARKER_COLOR,
        )
        self.assertEqual(
            image.pixelColor(blocking_rect.center()),
            self.slider.BLOCKING_MARKER_COLOR,
        )

    def test_safety_ticks_do_not_replace_existing_keyframe_ticks(self):
        self.slider.set_defined_times((5.0,))
        self.slider.set_safety_reports((_report(5.0, blocking=True),))
        QApplication.processEvents()

        x = self.slider._time_to_pixel(5.0)
        safety_rect = self.slider._safety_marker_rect(x)
        keyframe_rect = self.slider._marker_rect(x, current=False)
        self.assertFalse(safety_rect.intersects(keyframe_rect))

        image = self._render()
        self.assertEqual(
            image.pixelColor(safety_rect.center()),
            self.slider.BLOCKING_MARKER_COLOR,
        )
        self.assertEqual(
            image.pixelColor(keyframe_rect.center()),
            QColor(21, 116, 214),
        )

    def test_same_time_blocking_report_wins_and_none_clears_markers(self):
        advisory = _report(4.0, blocking=False)
        blocking = _report(4.0, blocking=True)
        missing_time = SimpleNamespace(blocking=True)
        nonfinite_time = _report(float("nan"), blocking=True)

        self.slider.set_safety_reports(
            (None, advisory, missing_time, nonfinite_time, blocking)
        )

        self.assertEqual(self.slider.safety_reports, (advisory, blocking))
        self.assertEqual(len(self.slider.safety_markers), 1)
        self.assertTrue(self.slider.safety_markers[0].blocking)

        self.slider.set_safety_reports(None)

        self.assertEqual(self.slider.safety_reports, ())
        self.assertEqual(self.slider.safety_markers, ())


if __name__ == "__main__":
    unittest.main()

"""Timeline widgets for trajectory editing."""

from dataclasses import dataclass
import math

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


@dataclass(frozen=True)
class TimelineSafetyMarker:
    """Paint-ready safety state derived from a structured collision report."""

    time: float
    blocking: bool


class TimesliceSlider(QSlider):
    """Horizontal time slider with markers for accepted logical slices."""

    marker_activated = Signal(float)
    time_activated = Signal(float)
    WHEEL_STEP_RAW = 2
    WHEEL_NOTCH_ANGLE = 120
    MARKER_WIDTH = 2
    MARKER_HEIGHT = 6
    CURRENT_MARKER_WIDTH = 2
    CURRENT_MARKER_HEIGHT = 10
    SAFETY_MARKER_WIDTH = 3
    SAFETY_MARKER_HEIGHT = 8
    ADVISORY_MARKER_COLOR = QColor(245, 158, 11)
    BLOCKING_MARKER_COLOR = QColor(220, 38, 38)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.defined_times = set()
        self.safety_reports = ()
        self.safety_markers = ()
        self.marker_snap_pixels = 10
        self._wheel_angle_remainder = 0

    def set_defined_times(self, times):
        self.defined_times = {round(float(time), 6) for time in times}
        self.update()

    def set_safety_reports(self, reports):
        """Show advisory/blocking ticks for reports carrying absolute time.

        ``reports`` may contain ``None`` so callers can pass the natural
        ``(warning_report, blocking_report)`` result from motion validation.
        Reports without a finite ``time`` are ignored because a sample index
        alone cannot be mapped reliably after trajectory resampling.
        """
        if reports is None:
            reports = ()
        try:
            reports = tuple(reports)
        except TypeError:
            reports = (reports,)

        valid_reports = []
        markers_by_time = {}
        for report in reports:
            if report is None:
                continue
            try:
                time = float(report.time)
            except (AttributeError, TypeError, ValueError):
                continue
            if not math.isfinite(time):
                continue

            blocking = bool(getattr(report, "blocking", False))
            time_key = round(time, 6)
            previous = markers_by_time.get(time_key)
            markers_by_time[time_key] = TimelineSafetyMarker(
                time=time,
                # A blocking report wins when advisory and blocking reports
                # describe the same instant.
                blocking=blocking or bool(previous and previous.blocking),
            )
            valid_reports.append(report)

        self.safety_reports = tuple(valid_reports)
        self.safety_markers = tuple(
            markers_by_time[key] for key in sorted(markers_by_time)
        )
        self.update()

    def snap_to_nearest_defined_time(self, time, tolerance=0.06):
        nearest = self._nearest_defined_time(time)
        if nearest is None or abs(nearest - float(time)) > tolerance:
            return False
        raw_value = self._time_to_raw(nearest)
        self.setValue(raw_value)
        self.marker_activated.emit(nearest)
        return True

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.orientation() != Qt.Orientation.Horizontal:
            return

        painter = QPainter(self)
        current_raw = self.value()

        for marker in self.safety_markers:
            x = self._time_to_pixel(marker.time)
            color = (
                self.BLOCKING_MARKER_COLOR
                if marker.blocking
                else self.ADVISORY_MARKER_COLOR
            )
            painter.fillRect(self._safety_marker_rect(x), color)

        for time in sorted(self.defined_times):
            x = self._time_to_pixel(time)
            raw_value = self._time_to_raw(time)
            current = abs(raw_value - current_raw) <= 1
            color = QColor(21, 116, 214) if not current else QColor(15, 158, 255)
            painter.fillRect(self._marker_rect(x, current), color)

    def _safety_marker_rect(self, x):
        """Place safety ticks above Keyframe ticks around the groove."""
        groove = self._groove_rect()
        bottom = min(self.height() - 1, groove.top() + 2)
        top = max(0, bottom - self.SAFETY_MARKER_HEIGHT + 1)
        return QRect(
            int(x) - self.SAFETY_MARKER_WIDTH // 2,
            top,
            self.SAFETY_MARKER_WIDTH,
            max(1, bottom - top + 1),
        )

    def _marker_rect(self, x, current):
        width = (
            self.CURRENT_MARKER_WIDTH
            if current
            else self.MARKER_WIDTH
        )
        height = (
            self.CURRENT_MARKER_HEIGHT
            if current
            else self.MARKER_HEIGHT
        )
        top = min(
            self._groove_rect().bottom() + 3,
            max(0, self.height() - height),
        )
        return QRect(int(x) - width // 2, top, width, height)

    def mousePressEvent(self, event):
        if (
            self.orientation() == Qt.Orientation.Horizontal
            and event.button() == Qt.MouseButton.LeftButton
        ):
            click_pos = event.position().toPoint()
            if self._handle_rect().contains(click_pos):
                super().mousePressEvent(event)
                return
            if self.activate_time_at_pixel(event.position().x()):
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        previous = self.value()
        self._wheel_angle_remainder += event.angleDelta().y()
        notches = int(self._wheel_angle_remainder / self.WHEEL_NOTCH_ANGLE)
        if notches:
            self._wheel_angle_remainder -= notches * self.WHEEL_NOTCH_ANGLE
            self.setValue(previous + notches * self.WHEEL_STEP_RAW)
        if self.value() != previous:
            self.time_activated.emit(self.value() / 100.0)
        event.accept()

    def activate_time_at_pixel(self, x):
        if self.defined_times:
            nearest = self._nearest_time_by_pixel(x)
            if nearest is not None:
                marker_x = self._time_to_pixel(nearest)
                if abs(marker_x - x) <= self.marker_snap_pixels:
                    self.setValue(self._time_to_raw(nearest))
                    self.marker_activated.emit(nearest)
                    return True

        raw_value = self._pixel_to_raw(x)
        self.setValue(raw_value)
        self.time_activated.emit(raw_value / 100.0)
        return True

    def _nearest_defined_time(self, time):
        if not self.defined_times:
            return None
        return min(self.defined_times, key=lambda candidate: abs(candidate - time))

    def _nearest_time_by_pixel(self, x):
        if not self.defined_times:
            return None
        return min(
            self.defined_times,
            key=lambda candidate: abs(self._time_to_pixel(candidate) - x),
        )

    def _time_to_raw(self, time):
        raw_value = int(round(float(time) * 100.0))
        return max(self.minimum(), min(self.maximum(), raw_value))

    def _groove_rect(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

    def _handle_rect(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )

    def _time_to_pixel(self, time):
        groove = self._groove_rect()
        handle = self._handle_rect()
        span = max(1, groove.width() - handle.width())
        raw_value = self._time_to_raw(time)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        offset = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            raw_value,
            span,
            option.upsideDown,
        )
        return groove.x() + handle.width() // 2 + offset

    def _pixel_to_raw(self, x):
        groove = self._groove_rect()
        handle = self._handle_rect()
        span = max(1, groove.width() - handle.width())
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        start = groove.x() + handle.width() // 2
        offset = int(round(float(x) - start))
        offset = max(0, min(span, offset))
        return QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            offset,
            span,
            option.upsideDown,
        )

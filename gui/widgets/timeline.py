"""Timeline widgets for trajectory editing."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


class TimesliceSlider(QSlider):
    """Horizontal time slider with markers for accepted logical slices."""

    marker_activated = Signal(float)
    time_activated = Signal(float)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.defined_times = set()
        self.marker_snap_pixels = 10

    def set_defined_times(self, times):
        self.defined_times = {round(float(time), 6) for time in times}
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
        if self.orientation() != Qt.Orientation.Horizontal or not self.defined_times:
            return

        painter = QPainter(self)
        current_raw = self.value()
        for time in sorted(self.defined_times):
            x = self._time_to_pixel(time)
            raw_value = self._time_to_raw(time)
            current = abs(raw_value - current_raw) <= 1
            color = QColor(21, 116, 214) if not current else QColor(15, 158, 255)
            painter.setPen(QPen(color, 3 if current else 2))
            groove = self._groove_rect()
            top = groove.bottom() + 3
            bottom = min(self.height() - 2, top + (8 if current else 6))
            painter.drawLine(x, top, x, bottom)

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
        super().wheelEvent(event)
        if self.value() != previous:
            self.time_activated.emit(self.value() / 100.0)

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

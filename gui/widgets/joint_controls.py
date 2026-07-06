"""Joint angle and IK-influence slider widgets."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class JointControl(QWidget):
    value_changed = Signal(str, float)

    def __init__(self, name, limits, value):
        super().__init__()
        self.name = name
        self.lo, self.hi = limits if limits is not None else (-3.14159, 3.14159)
        self._syncing = False
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 2000)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(65)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(name), 1)
        layout.addWidget(self.slider, 2)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def _to_value(self, raw):
        return self.lo + (self.hi - self.lo) * raw / 2000.0

    def _to_raw(self, value):
        if self.hi <= self.lo:
            return 0
        return round((value - self.lo) * 2000.0 / (self.hi - self.lo))

    def _changed(self, raw):
        value = self._to_value(raw)
        self.value_label.setText(f"{value:+.3f} rad")
        if not self._syncing:
            self.value_changed.emit(self.name, value)

    def set_value(self, value):
        self._syncing = True
        self.slider.setValue(max(0, min(2000, self._to_raw(value))))
        self._syncing = False
        self.value_label.setText(f"{float(value):+.3f} rad")


class IKInfluenceControl(QWidget):
    value_changed = Signal(str, float)

    def __init__(self, name, value=1.0):
        super().__init__()
        self.name = name
        self._syncing = False
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 300)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(38)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(name), 1)
        layout.addWidget(self.slider, 2)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def _changed(self, raw):
        value = raw / 100.0
        self.value_label.setText(f"{value:.2f}")
        if not self._syncing:
            self.value_changed.emit(self.name, value)

    def set_value(self, value):
        self._syncing = True
        self.slider.setValue(round(max(0.0, min(3.0, float(value))) * 100.0))
        self._syncing = False
        self.value_label.setText(f"{float(value):.2f}")

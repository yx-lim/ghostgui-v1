"""Small dialogs for non-destructive Keyframe timeline retiming."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from application.timeline_editing import snap_time


class TimelineEditDialog(QDialog):
    """Collect one or more timeline values with optional export-grid snapping."""

    def __init__(
        self,
        title,
        description,
        fields,
        *,
        export_interval,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(390)
        self._export_interval = float(export_interval)
        self._inputs = {}

        layout = QVBoxLayout(self)
        message = QLabel(description)
        message.setWordWrap(True)
        layout.addWidget(message)

        form = QFormLayout()
        for key, label, value, minimum, maximum in fields:
            editor = QDoubleSpinBox()
            editor.setRange(float(minimum), float(maximum))
            editor.setDecimals(2)
            editor.setSingleStep(max(0.01, min(self._export_interval, 10.0)))
            editor.setSuffix(" s")
            editor.setValue(float(value))
            form.addRow(label, editor)
            self._inputs[key] = editor
        layout.addLayout(form)

        self.snap_checkbox = QCheckBox("Snap to Export interval")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip(
            "Align edited times with the sampling grid used by Generate and export."
        )
        layout.addWidget(self.snap_checkbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self, name):
        value = float(self._inputs[name].value())
        if self.snap_checkbox.isChecked():
            value = snap_time(value, self._export_interval)
        return value

    def snap_enabled(self):
        return self.snap_checkbox.isChecked()


class ScaleTimeRangeDialog(QDialog):
    """Collect a timeline range and an actual motion-speed multiplier."""

    def __init__(
        self,
        *,
        start_time,
        end_time,
        entire_start,
        entire_end,
        export_interval,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Scale Time Range")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._export_interval = float(export_interval)
        self._entire_bounds = (float(entire_start), float(entire_end))

        layout = QVBoxLayout(self)
        message = QLabel(
            "Change actual motion speed by scaling Keyframe timestamps around "
            "the range start. Values above 1× are faster; values below 1× "
            "are slower."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        self.entire_motion_checkbox = QCheckBox("Scale entire motion")
        self.entire_motion_checkbox.setChecked(True)
        layout.addWidget(self.entire_motion_checkbox)

        form = QFormLayout()
        self.start_input = self._time_input(start_time)
        self.end_input = self._time_input(end_time)
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.10, 4.00)
        self.speed_input.setDecimals(2)
        self.speed_input.setSingleStep(0.25)
        self.speed_input.setValue(2.00)
        self.speed_input.setSuffix("×")
        self.speed_input.setToolTip(
            "2× halves the duration; 0.5× doubles the duration."
        )
        form.addRow("Range start", self.start_input)
        form.addRow("Range end", self.end_input)
        form.addRow("Motion speed", self.speed_input)
        layout.addLayout(form)

        self.snap_checkbox = QCheckBox("Snap resulting Keyframes to Export interval")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip(
            "Align every scaled Keyframe with the sampling grid used by "
            "Generate and export."
        )
        layout.addWidget(self.snap_checkbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.entire_motion_checkbox.toggled.connect(
            self._sync_range_enabled
        )
        self._sync_range_enabled(True)

    def _time_input(self, value):
        editor = QDoubleSpinBox()
        editor.setRange(0.0, 120.0)
        editor.setDecimals(2)
        editor.setSingleStep(max(0.01, min(self._export_interval, 10.0)))
        editor.setSuffix(" s")
        editor.setValue(float(value))
        return editor

    def _sync_range_enabled(self, entire_motion):
        self.start_input.setEnabled(not entire_motion)
        self.end_input.setEnabled(not entire_motion)

    def range_values(self):
        if self.entire_motion_checkbox.isChecked():
            start_time, end_time = self._entire_bounds
        else:
            start_time = float(self.start_input.value())
            end_time = float(self.end_input.value())
        return start_time, end_time

    def speed(self):
        return float(self.speed_input.value())

    def snap_interval(self):
        if self.snap_checkbox.isChecked():
            return self._export_interval
        return None


class MotionRangeDialog(QDialog):
    """Collect an entire-motion or custom Keyframe range."""

    def __init__(
        self,
        title,
        description,
        *,
        entire_label,
        entire_start,
        entire_end,
        export_interval,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self._entire_bounds = (float(entire_start), float(entire_end))
        self._export_interval = float(export_interval)

        self.root_layout = QVBoxLayout(self)
        message = QLabel(description)
        message.setWordWrap(True)
        self.root_layout.addWidget(message)

        self.entire_motion_checkbox = QCheckBox(entire_label)
        self.entire_motion_checkbox.setChecked(True)
        self.root_layout.addWidget(self.entire_motion_checkbox)

        self.form = QFormLayout()
        self.start_input = self._time_input(entire_start)
        self.end_input = self._time_input(entire_end)
        self.form.addRow("Range start", self.start_input)
        self.form.addRow("Range end", self.end_input)
        self.root_layout.addLayout(self.form)

        self.snap_checkbox = QCheckBox("Snap custom range to Export interval")
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip(
            "Align custom range bounds with the sampling grid used by "
            "Generate and export. Entire-motion bounds are preserved exactly."
        )
        self.root_layout.addWidget(self.snap_checkbox)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.root_layout.addWidget(self.buttons)

        self.entire_motion_checkbox.toggled.connect(
            self._sync_range_enabled
        )
        self._sync_range_enabled(True)

    def _time_input(self, value):
        editor = QDoubleSpinBox()
        editor.setRange(0.0, 120.0)
        editor.setDecimals(2)
        editor.setSingleStep(max(0.01, min(self._export_interval, 10.0)))
        editor.setSuffix(" s")
        editor.setValue(float(value))
        return editor

    def _sync_range_enabled(self, entire_motion):
        self.start_input.setEnabled(not entire_motion)
        self.end_input.setEnabled(not entire_motion)
        self.snap_checkbox.setEnabled(not entire_motion)

    def range_values(self):
        if self.entire_motion_checkbox.isChecked():
            return self._entire_bounds
        start_time = float(self.start_input.value())
        end_time = float(self.end_input.value())
        if self.snap_checkbox.isChecked():
            start_time = snap_time(start_time, self._export_interval)
            end_time = snap_time(end_time, self._export_interval)
        return start_time, end_time


class RepeatMotionDialog(MotionRangeDialog):
    """Collect a source range and periodic repetition settings."""

    def __init__(
        self,
        *,
        entire_start,
        entire_end,
        export_interval,
        parent=None,
    ):
        super().__init__(
            "Repeat Motion",
            "Append copies of a committed Keyframe range. Forward repeats the "
            "same time order and requires matching start/end poses; Ping-pong "
            "alternates reversed and forward copies for an A-to-B motion.",
            entire_label="Repeat entire motion",
            entire_start=entire_start,
            entire_end=entire_end,
            export_interval=export_interval,
            parent=parent,
        )
        self.additional_copies_input = QSpinBox()
        self.additional_copies_input.setRange(1, 100)
        self.additional_copies_input.setValue(1)
        self.pattern_box = QComboBox()
        self.pattern_box.addItems(("Forward", "Ping-pong"))
        self.pattern_box.setCurrentText("Ping-pong")
        self.form.addRow("Additional copies", self.additional_copies_input)
        self.form.addRow("Pattern", self.pattern_box)

    def additional_copies(self):
        return int(self.additional_copies_input.value())

    def ping_pong(self):
        return self.pattern_box.currentText() == "Ping-pong"


def insert_time_dialog(parent, *, at_time, default_duration, export_interval):
    default_duration = max(
        float(export_interval),
        snap_time(default_duration, export_interval),
    )
    return TimelineEditDialog(
        "Insert Time",
        f"Open a held interval at the current time ({at_time:.2f} s) and shift "
        "all later Keyframes to the right.",
        (
            (
                "duration",
                "Duration",
                default_duration,
                0.01,
                120.0,
            ),
        ),
        export_interval=export_interval,
        parent=parent,
    )


def shift_motion_dialog(parent, *, default_offset, export_interval):
    default_offset = max(
        float(export_interval),
        snap_time(default_offset, export_interval),
    )
    return TimelineEditDialog(
        "Shift Entire Motion",
        "Move every logical target and robot-pose Keyframe by the same offset. "
        "A negative offset is allowed only when no Keyframe would cross 0 s.",
        (
            ("offset", "Time offset", default_offset, -120.0, 120.0),
        ),
        export_interval=export_interval,
        parent=parent,
    )


def move_range_dialog(
    parent,
    *,
    start_time,
    end_time,
    destination_start,
    export_interval,
):
    return TimelineEditDialog(
        "Move Time Range",
        "Move all logical target and robot-pose Keyframes in the inclusive "
        "range. Existing destination Keyframes cause the edit to be rejected.",
        (
            ("start", "Range start", start_time, 0.0, 120.0),
            ("end", "Range end", end_time, 0.0, 120.0),
            (
                "destination",
                "Destination start",
                destination_start,
                0.0,
                120.0,
            ),
        ),
        export_interval=export_interval,
        parent=parent,
    )


def scale_range_dialog(
    parent,
    *,
    start_time,
    end_time,
    entire_start,
    entire_end,
    export_interval,
):
    return ScaleTimeRangeDialog(
        start_time=start_time,
        end_time=end_time,
        entire_start=entire_start,
        entire_end=entire_end,
        export_interval=export_interval,
        parent=parent,
    )


def copy_motion_range_dialog(
    parent,
    *,
    entire_start,
    entire_end,
    export_interval,
):
    return MotionRangeDialog(
        "Copy Motion Range",
        "Copy committed logical target Keyframes and poses, including Joint "
        "Angles. The Orange preview and generated motion are not copied.",
        entire_label="Copy entire motion",
        entire_start=entire_start,
        entire_end=entire_end,
        export_interval=export_interval,
        parent=parent,
    )


def repeat_motion_dialog(
    parent,
    *,
    entire_start,
    entire_end,
    export_interval,
):
    return RepeatMotionDialog(
        entire_start=entire_start,
        entire_end=entire_end,
        export_interval=export_interval,
        parent=parent,
    )

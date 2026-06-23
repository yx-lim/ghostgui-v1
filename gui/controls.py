"""
controls.py

Purpose:
    GUI panel for editing target reference-frame keyframes.

This replaces the older "robot control" sliders.

The GUI now edits:
    - selected robot frame: pelvis, left foot, right foot, torso
    - time
    - phase
    - target frame position
    - target frame yaw
    - trajectory keyframe table
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
)

from .trajectory import TargetFrame


class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name, min_value, max_value, initial_value, scale=100):
        super().__init__()

        self.name = name
        self.scale = scale
        self._syncing = False

        layout = QVBoxLayout()
        value_row = QHBoxLayout()

        self.label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.input = QDoubleSpinBox()

        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(initial_value)

        self.input.setDecimals(2)
        self.input.setRange(min_value / self.scale, max_value / self.scale)
        self.input.setSingleStep(1 / self.scale)
        self.input.setValue(initial_value / self.scale)

        self.slider.valueChanged.connect(self.on_slider_changed)
        self.input.valueChanged.connect(self.on_input_changed)

        layout.addWidget(self.label)
        value_row.addWidget(self.slider, stretch=1)
        value_row.addWidget(self.input)
        layout.addLayout(value_row)

        self.setLayout(layout)
        self.update_label()

    def value(self):
        return self.slider.value() / self.scale

    def set_value(self, value):
        self.slider.setValue(round(value * self.scale))

    def on_slider_changed(self, raw_value):
        if self._syncing:
            return

        self._syncing = True
        value = raw_value / self.scale
        self.input.setValue(value)
        self._syncing = False

        self.update_label()
        self.value_changed.emit(value)

    def on_input_changed(self, value):
        if self._syncing:
            return

        self._syncing = True
        self.slider.setValue(round(value * self.scale))
        self._syncing = False

        self.update_label()
        self.value_changed.emit(self.value())

    def update_label(self):
        self.label.setText(f"{self.name}: {self.value():.2f}")


class TrajectoryControlPanel(QGroupBox):
    """
    Left-side panel for creating trajectory keyframes.

    Signals:
        pose_changed:
            emitted when sliders change, so the viewer can update live.

        add_keyframe_clicked:
            emitted when user wants to add the current target frame.

        update_keyframe_clicked:
            emitted when user wants to overwrite selected keyframe.

        delete_keyframe_clicked:
            emitted when user wants to delete selected keyframe.

        generate_clicked:
            emitted when user wants to send trajectory to backend.
    """

    pose_changed = Signal(float, float, float)
    add_keyframe_clicked = Signal()
    update_keyframe_clicked = Signal()
    delete_keyframe_clicked = Signal()
    generate_clicked = Signal()
    keyframe_selected = Signal(int)
    frame_name_changed = Signal(str)

    def __init__(self):
        super().__init__("Reference Frame Trajectory Editor")
        self._suppress_pose_changed = False
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout()

        # --------------------------------------------------------
        # Select which robot frame this target refers to
        # --------------------------------------------------------
        layout.addWidget(QLabel("Target robot frame"))

        self.frame_box = QComboBox()
        self.frame_box.addItems([
            "pelvis",
            "torso",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ])
        self.frame_box.currentTextChanged.connect(self.frame_name_changed.emit)
        layout.addWidget(self.frame_box)

        # --------------------------------------------------------
        # Motion phase
        # --------------------------------------------------------
        layout.addWidget(QLabel("Motion phase"))

        self.phase_box = QComboBox()
        self.phase_box.addItems([
            "crouch",
            "launch",
            "flight",
            "landing",
        ])
        layout.addWidget(self.phase_box)

        # --------------------------------------------------------
        # Sliders for target reference-frame pose
        # --------------------------------------------------------
        self.time_slider = LabeledSlider(
            "Time [s]",
            min_value=0,
            max_value=500,
            initial_value=0,
            scale=100,
        )

        self.x_slider = LabeledSlider(
            "Target X [m]",
            min_value=-200,
            max_value=200,
            initial_value=0,
            scale=100,
        )

        self.z_slider = LabeledSlider(
            "Target Z [m]",
            min_value=0,
            max_value=200,
            initial_value=90,
            scale=100,
        )

        self.yaw_slider = LabeledSlider(
            "Target yaw [rad]",
            min_value=-314,
            max_value=314,
            initial_value=0,
            scale=100,
        )

        layout.addWidget(self.time_slider)
        layout.addWidget(self.x_slider)
        layout.addWidget(self.z_slider)
        layout.addWidget(self.yaw_slider)

        # Whenever pose controls change, update the viewer target marker.
        self.x_slider.value_changed.connect(self.emit_pose_changed)
        self.z_slider.value_changed.connect(self.emit_pose_changed)
        self.yaw_slider.value_changed.connect(self.emit_pose_changed)

        # --------------------------------------------------------
        # Keyframe buttons
        # --------------------------------------------------------
        button_row = QHBoxLayout()

        self.add_button = QPushButton("Add Keyframe")
        self.update_button = QPushButton("Update")
        self.delete_button = QPushButton("Delete")

        button_row.addWidget(self.add_button)
        button_row.addWidget(self.update_button)
        button_row.addWidget(self.delete_button)

        layout.addLayout(button_row)

        self.generate_button = QPushButton("Generate / Simulate Trajectory")
        layout.addWidget(self.generate_button)

        self.add_button.clicked.connect(self.add_keyframe_clicked.emit)
        self.update_button.clicked.connect(self.update_keyframe_clicked.emit)
        self.delete_button.clicked.connect(self.delete_keyframe_clicked.emit)
        self.generate_button.clicked.connect(self.generate_clicked.emit)

        # --------------------------------------------------------
        # Keyframe table
        # --------------------------------------------------------
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "time",
            "phase",
            "frame",
            "x",
            "z",
            "yaw",
        ])
        self.table.cellClicked.connect(self.on_table_cell_clicked)

        layout.addWidget(QLabel("Trajectory keyframes"))
        layout.addWidget(self.table)

        self.setLayout(layout)

    def emit_pose_changed(self):
        if self._suppress_pose_changed:
            return

        self.pose_changed.emit(
            self.x_slider.value(),
            self.z_slider.value(),
            self.yaw_slider.value(),
        )

    def current_frame(self):
        """
        Convert GUI values into a TargetFrame object.
        """

        return TargetFrame(
            time=self.time_slider.value(),
            phase=self.phase_box.currentText(),
            frame_name=self.frame_box.currentText(),
            x=self.x_slider.value(),
            y=0.0,
            z=self.z_slider.value(),
            roll=0.0,
            pitch=0.0,
            yaw=self.yaw_slider.value(),
        )

    def set_from_frame(self, frame):
        """
        Load a keyframe into the editor.
        """

        self._suppress_pose_changed = True

        try:
            self.time_slider.set_value(frame.time)
            self.x_slider.set_value(frame.x)
            self.z_slider.set_value(frame.z)
            self.yaw_slider.set_value(frame.yaw)

            self.phase_box.setCurrentText(frame.phase)
            previous_block_state = self.frame_box.blockSignals(True)
            self.frame_box.setCurrentText(frame.frame_name)
            self.frame_box.blockSignals(previous_block_state)
        finally:
            self._suppress_pose_changed = False

        self.emit_pose_changed()

    def set_position_from_viewer(self, x, z, emit_pose_changed=True):
        """
        Called when user drags target frame in the viewer.
        """

        self._suppress_pose_changed = True

        try:
            self.x_slider.set_value(x)
            self.z_slider.set_value(z)
        finally:
            self._suppress_pose_changed = False

        if emit_pose_changed:
            self.emit_pose_changed()

    def refresh_table(self, trajectory):
        """
        Display trajectory array in table.
        """

        self.table.setRowCount(len(trajectory.frames))

        for row, frame in enumerate(trajectory.frames):
            values = [
                f"{frame.time:.2f}",
                frame.phase,
                frame.frame_name,
                f"{frame.x:.2f}",
                f"{frame.z:.2f}",
                f"{frame.yaw:.2f}",
            ]

            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def selected_row(self):
        return self.table.currentRow()

    def on_table_cell_clicked(self, row, col):
        self.keyframe_selected.emit(row)

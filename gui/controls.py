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
    - target frame roll/pitch/yaw
    - trajectory keyframe table
"""

import math

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
    QStackedWidget,
    QHBoxLayout,
    QCheckBox,
    QSizePolicy,
)

from .trajectory import TargetFrame


class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name, min_value, max_value, initial_value, scale=100):
        super().__init__()

        self.name = name
        self.scale = scale
        self.decimals = max(2, int(math.ceil(math.log10(max(1, scale)))))
        self._syncing = False

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(4)

        self.label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.input = QDoubleSpinBox()

        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setValue(initial_value)

        self.input.setDecimals(self.decimals)
        self.input.setRange(min_value / self.scale, max_value / self.scale)
        self.input.setSingleStep(1 / self.scale)
        self.input.setValue(initial_value / self.scale)
        self.input.setMinimumWidth(52)
        self.input.setMaximumWidth(64)

        self.slider.valueChanged.connect(self.on_slider_changed)
        self.input.valueChanged.connect(self.on_input_changed)

        self.label.setWordWrap(True)
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
        self.label.setText(f"{self.name}: {self.value():.{self.decimals}f}")


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

    pose_changed = Signal(float, float, float, float, float, float)
    add_keyframe_clicked = Signal()
    update_keyframe_clicked = Signal()
    delete_keyframe_clicked = Signal()
    generate_clicked = Signal()
    keyframe_selected = Signal(int)
    frame_name_changed = Signal(str)
    trajectory_lines_changed = Signal(bool)
    time_changed = Signal(float)
    model_changed = Signal(str)
    open_model_clicked = Signal()
    choose_mesh_folder_clicked = Signal()

    def __init__(self, model_registry=None, model_key="g1", frame_names=None):
        super().__init__("Reference Frame Trajectory Editor")
        self._suppress_pose_changed = False
        self.model_registry = model_registry or {}
        self.model_key = model_key
        self.frame_names = list(frame_names or [
            "pelvis", "torso", "left_foot", "right_foot",
            "left_hand", "right_hand",
        ])
        self.build_ui()

    def build_ui(self):
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(root_layout)

        self.robot_panel, robot_layout = self._make_section_panel()
        self.target_panel, self.target_layout = self._make_section_panel()
        self.transform_panel, transform_layout = self._make_section_panel()
        self.preview_ik_panel, self.preview_ik_layout = self._make_section_panel()
        self.trajectory_panel, self.trajectory_layout = self._make_section_panel()
        self.view_panel, self.view_layout = self._make_section_panel()

        if self.model_registry:
            robot_layout.addWidget(QLabel("Robot model"))
            self.model_box = QComboBox()
            for key, info in self.model_registry.items():
                self.model_box.addItem(info.display_name, key)
            selected = self.model_box.findData(self.model_key)
            if selected >= 0:
                self.model_box.setCurrentIndex(selected)
            self.model_box.currentIndexChanged.connect(
                lambda index: self.model_changed.emit(self.model_box.itemData(index))
            )
            robot_layout.addWidget(self.model_box)
            self.open_model_button = QPushButton("Open Model")
            self.open_model_button.clicked.connect(self.open_model_clicked.emit)
            robot_layout.addWidget(self.open_model_button)
            self.choose_mesh_folder_button = QPushButton("Mesh Folder (.stl)")
            self.choose_mesh_folder_button.clicked.connect(
                self.choose_mesh_folder_clicked.emit
            )
            robot_layout.addWidget(self.choose_mesh_folder_button)

        # --------------------------------------------------------
        # Select which robot frame this target refers to
        # --------------------------------------------------------
        self.target_layout.addWidget(QLabel("Target robot frame"))

        self.frame_box = QComboBox()
        self.frame_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.frame_box.setMinimumContentsLength(10)
        self.frame_box.addItems(self.frame_names)
        # A hand is the most useful default for the 3D transform gizmo. The
        # user can still select pelvis/feet exactly as before.
        preferred = "left_hand" if "left_hand" in self.frame_names else self.frame_names[0]
        self.frame_box.setCurrentText(preferred)
        self.frame_box.currentTextChanged.connect(self.frame_name_changed.emit)
        self.target_layout.addWidget(self.frame_box)

        # --------------------------------------------------------
        # Motion phase
        # --------------------------------------------------------
        self.trajectory_layout.addWidget(QLabel("Motion phase"))

        self.phase_box = QComboBox()
        self.phase_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.phase_box.setMinimumContentsLength(8)
        self.phase_box.addItems([
            "crouch",
            "launch",
            "flight",
            "landing",
        ])
        self.trajectory_layout.addWidget(self.phase_box)

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
            min_value=-2000,
            max_value=2000,
            initial_value=0,
            scale=1000,
        )

        self.y_slider = LabeledSlider(
            "Target Y [m]",
            min_value=-1000,
            max_value=1000,
            initial_value=0,
            scale=1000,
        )

        self.z_slider = LabeledSlider(
            "Target Z [m]",
            min_value=0,
            max_value=2000,
            initial_value=900,
            scale=1000,
        )

        self.roll_slider = LabeledSlider(
            "Target roll [rad]",
            min_value=-314,
            max_value=314,
            initial_value=0,
            scale=100,
        )

        self.pitch_slider = LabeledSlider(
            "Target pitch [rad]",
            min_value=-157,
            max_value=157,
            initial_value=0,
            scale=100,
        )

        self.yaw_slider = LabeledSlider(
            "Target yaw [rad]",
            min_value=-314,
            max_value=314,
            initial_value=0,
            scale=100,
        )

        self.trajectory_layout.addWidget(self.time_slider)
        transform_layout.addWidget(self.x_slider)
        transform_layout.addWidget(self.y_slider)
        transform_layout.addWidget(self.z_slider)
        transform_layout.addWidget(self.roll_slider)
        transform_layout.addWidget(self.pitch_slider)
        transform_layout.addWidget(self.yaw_slider)

        # Whenever pose controls change, update the viewer target marker.
        self.x_slider.value_changed.connect(self.emit_pose_changed)
        self.y_slider.value_changed.connect(self.emit_pose_changed)
        self.z_slider.value_changed.connect(self.emit_pose_changed)
        self.roll_slider.value_changed.connect(self.emit_pose_changed)
        self.pitch_slider.value_changed.connect(self.emit_pose_changed)
        self.yaw_slider.value_changed.connect(self.emit_pose_changed)
        # Commit timeline selection once per interaction. Connecting the raw
        # valueChanged signal would create a qpos keyframe at every intermediate
        # slider tick while scrubbing from (for example) 0.0 to 0.2 seconds.
        self.time_slider.slider.sliderReleased.connect(
            lambda: self.emit_time_changed(self.time_slider.value())
        )
        self.time_slider.input.editingFinished.connect(
            lambda: self.emit_time_changed(self.time_slider.value())
        )

        # --------------------------------------------------------
        # Trajectory display options
        # --------------------------------------------------------
        self.show_lines_box = QCheckBox("Show trajectory lines")
        self.show_lines_box.setChecked(True)
        self.show_lines_box.toggled.connect(self.trajectory_lines_changed.emit)
        self.view_layout.addWidget(self.show_lines_box)

        # --------------------------------------------------------
        # Keyframe buttons
        # --------------------------------------------------------
        button_row = QVBoxLayout()

        self.add_button = QPushButton("Add Keyframe")
        self.update_button = QPushButton("Update")
        self.delete_button = QPushButton("Delete")

        button_row.addWidget(self.add_button)
        button_row.addWidget(self.update_button)
        button_row.addWidget(self.delete_button)

        self.trajectory_layout.addLayout(button_row)

        self.generate_button = QPushButton("Generate / Simulate")
        self.trajectory_layout.addWidget(self.generate_button)

        self.add_button.clicked.connect(self.add_keyframe_clicked.emit)
        self.update_button.clicked.connect(self.update_keyframe_clicked.emit)
        self.delete_button.clicked.connect(self.delete_keyframe_clicked.emit)
        self.generate_button.clicked.connect(self.generate_clicked.emit)

        # --------------------------------------------------------
        # Keyframe table
        # --------------------------------------------------------
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "time",
            "phase",
            "frame",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ])
        self.table.cellClicked.connect(self.on_table_cell_clicked)

        self.trajectory_layout.addWidget(QLabel("Trajectory keyframes"))
        self.trajectory_layout.addWidget(self.table)

        self.robot_context_stack = self._make_context_stack()
        self.target_context_stack = self._make_context_stack()
        self.trajectory_context_stack = self._make_context_stack()
        self.preview_ik_context_stack = self._make_context_stack()
        self.robot_view_context_stack = self._make_context_stack()
        robot_layout.addWidget(self.robot_context_stack)
        self.target_layout.addWidget(self.target_context_stack)
        self.trajectory_layout.addWidget(self.trajectory_context_stack)
        self.preview_ik_layout.addWidget(self.preview_ik_context_stack)
        robot_layout.addWidget(self.robot_view_context_stack)

    def _make_section_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        return panel, layout

    def _make_context_stack(self):
        stack = QStackedWidget()
        stack.empty_widget = QWidget()
        stack.addWidget(stack.empty_widget)
        stack.setVisible(False)
        stack.setMinimumWidth(0)
        stack.setMaximumWidth(220)
        stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return stack

    def _set_context_widget(self, stack, widget):
        if widget is None:
            widget = stack.empty_widget
        if stack.indexOf(widget) < 0:
            stack.addWidget(widget)
        stack.setCurrentWidget(widget)
        stack.setVisible(widget is not stack.empty_widget)

    def set_selection_context_widget(self, widget):
        self._set_context_widget(self.target_context_stack, widget)

    def set_robot_context_widget(self, widget):
        self._set_context_widget(self.robot_context_stack, widget)

    def set_trajectory_context_widget(self, widget):
        self._set_context_widget(self.trajectory_context_stack, widget)

    def set_display_context_widget(self, widget):
        self._set_context_widget(self.robot_view_context_stack, widget)

    def set_preview_ik_context_widget(self, widget):
        self._set_context_widget(self.preview_ik_context_stack, widget)

    def selection_context_widget(self):
        return self.target_context_stack.currentWidget()

    def robot_context_widget(self):
        return self.robot_context_stack.currentWidget()

    def trajectory_context_widget(self):
        return self.trajectory_context_stack.currentWidget()

    def display_context_widget(self):
        return self.robot_view_context_stack.currentWidget()

    def preview_ik_context_widget(self):
        return self.preview_ik_context_stack.currentWidget()

    def workflow_sections(self):
        return [
            ("Robot", self.robot_panel, True),
            ("Trajectory", self.trajectory_panel, True),
        ]

    def inspector_sections(self):
        return [
            ("Target", self.target_panel, True),
            ("Transform", self.transform_panel, True),
            ("Preview / IK", self.preview_ik_panel, True),
        ]

    def set_frame_names(self, frame_names, preferred=None):
        """Replace target choices without emitting an intermediate selection."""
        names = list(frame_names)
        if not names:
            return
        old = self.frame_box.currentText()
        self.frame_box.blockSignals(True)
        self.frame_box.clear()
        self.frame_box.addItems(names)
        choice = preferred if preferred in names else old if old in names else names[0]
        self.frame_box.setCurrentText(choice)
        self.frame_box.blockSignals(False)
        self.frame_name_changed.emit(choice)

    def add_model(self, key, display_name, select=True):
        if not hasattr(self, "model_box"):
            return
        index = self.model_box.findData(key)
        if index < 0:
            self.model_box.addItem(display_name, key)
            index = self.model_box.findData(key)
        if select and index >= 0:
            self.model_box.setCurrentIndex(index)

    def emit_pose_changed(self):
        if self._suppress_pose_changed:
            return

        self.pose_changed.emit(
            self.x_slider.value(),
            self.y_slider.value(),
            self.z_slider.value(),
            self.roll_slider.value(),
            self.pitch_slider.value(),
            self.yaw_slider.value(),
        )

    def emit_time_changed(self, value):
        if not self._suppress_pose_changed:
            self.time_changed.emit(float(value))

    def current_frame(self):
        """
        Convert GUI values into a TargetFrame object.
        """

        return TargetFrame(
            time=self.time_slider.value(),
            phase=self.phase_box.currentText(),
            frame_name=self.frame_box.currentText(),
            x=self.x_slider.value(),
            y=self.y_slider.value(),
            z=self.z_slider.value(),
            roll=self.roll_slider.value(),
            pitch=self.pitch_slider.value(),
            yaw=self.yaw_slider.value(),
        )

    def show_trajectory_lines(self):
        return self.show_lines_box.isChecked()

    def set_from_frame(self, frame):
        """
        Load a keyframe into the editor.
        """

        self._suppress_pose_changed = True

        try:
            self.time_slider.set_value(frame.time)
            self.x_slider.set_value(frame.x)
            self.y_slider.set_value(frame.y)
            self.z_slider.set_value(frame.z)
            self.roll_slider.set_value(frame.roll)
            self.pitch_slider.set_value(frame.pitch)
            self.yaw_slider.set_value(frame.yaw)

            self.phase_box.setCurrentText(frame.phase)
            previous_block_state = self.frame_box.blockSignals(True)
            self.frame_box.setCurrentText(frame.frame_name)
            self.frame_box.blockSignals(previous_block_state)
        finally:
            self._suppress_pose_changed = False

        self.time_changed.emit(self.time_slider.value())
        self.emit_pose_changed()

    def set_position_from_viewer(self, x, z, emit_pose_changed=True):
        """
        Called when user drags target frame in the viewer.
        """

        self.set_position_values(x=x, z=z, emit_pose_changed=emit_pose_changed)

    def set_position_values(
        self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None,
        emit_pose_changed=True,
    ):
        """
        Set target position controls, optionally preserving untouched axes.
        """

        self._suppress_pose_changed = True

        try:
            if x is not None:
                self.x_slider.set_value(x)
            if y is not None:
                self.y_slider.set_value(y)
            if z is not None:
                self.z_slider.set_value(z)
            if roll is not None:
                self.roll_slider.set_value(roll)
            if pitch is not None:
                self.pitch_slider.set_value(pitch)
            if yaw is not None:
                self.yaw_slider.set_value(yaw)
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
                f"{frame.y:.2f}",
                f"{frame.z:.2f}",
                f"{frame.roll:.2f}",
                f"{frame.pitch:.2f}",
                f"{frame.yaw:.2f}",
            ]

            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def selected_row(self):
        return self.table.currentRow()

    def on_table_cell_clicked(self, row, col):
        self.keyframe_selected.emit(row)

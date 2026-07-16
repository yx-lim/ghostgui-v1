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
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
)

from core.trajectory import TargetFrame


class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name, min_value, max_value, initial_value, scale=100):
        super().__init__()

        self.name = name
        self.scale = scale
        self.decimals = max(2, int(math.ceil(math.log10(max(1, scale)))))
        self._syncing = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

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

        self.label.setText(self.name)
        self.label.setMinimumWidth(64)
        self.label.setMaximumWidth(86)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.input)

        self.setLayout(layout)

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

        self.value_changed.emit(value)

    def on_input_changed(self, value):
        if self._syncing:
            return

        self._syncing = True
        self.slider.setValue(round(value * self.scale))
        self._syncing = False

        self.value_changed.emit(self.value())


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

        clear_trajectory_clicked:
            emitted when user wants to delete every trajectory keyframe.

        generate_clicked:
            emitted when user wants to send trajectory to backend.
    """

    pose_changed = Signal(float, float, float, float, float, float)
    add_keyframe_clicked = Signal()
    update_keyframe_clicked = Signal()
    delete_keyframe_clicked = Signal()
    clear_trajectory_clicked = Signal()
    generate_clicked = Signal()
    keyframe_selected = Signal(int)
    frame_name_changed = Signal(str)
    trajectory_lines_changed = Signal(bool)
    time_changed = Signal(float)
    model_changed = Signal(str)
    open_model_clicked = Signal()
    choose_mesh_folder_clicked = Signal()
    exposed_frames_changed = Signal(object)

    def __init__(self, model_registry=None, model_key="g1", frame_names=None):
        super().__init__("Reference Frame Trajectory Editor")
        self._suppress_pose_changed = False
        self._syncing_frame_selection = False
        self.current_target_key = None
        self.target_frame_entries = []
        self.target_frame_keys = []
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
            self.open_model_button = QPushButton("Upload Model")
            self.open_model_button.clicked.connect(self.open_model_clicked.emit)
            robot_layout.addWidget(self.open_model_button)
            self.choose_mesh_folder_button = QPushButton("Mesh Folder (.stl)")
            self.choose_mesh_folder_button.clicked.connect(
                self.choose_mesh_folder_clicked.emit
            )
            robot_layout.addWidget(self.choose_mesh_folder_button)

        self.frame_box = QComboBox()
        self.frame_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.frame_box.setMinimumContentsLength(10)
        self.frame_box.addItems(self.frame_names)
        preferred = "left_hand" if "left_hand" in self.frame_names else self.frame_names[0]
        self.frame_box.setCurrentText(preferred)
        self.current_target_key = preferred
        self.frame_box.currentTextChanged.connect(self._on_frame_box_text_changed)

        # --------------------------------------------------------
        # Select and expose logical robot frames for viewer picking.
        # --------------------------------------------------------
        self.target_layout.addWidget(QLabel("Exposed robot frames"))
        self.exposed_frame_list = QListWidget()
        self.exposed_frame_list.setMaximumHeight(138)
        self.exposed_frame_list.setMaximumWidth(212)
        self.exposed_frame_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.exposed_frame_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        self.exposed_frame_list.itemChanged.connect(
            self._on_exposed_frame_item_changed
        )
        self.exposed_frame_list.currentItemChanged.connect(
            self._on_exposed_frame_current_changed
        )
        self.set_exposed_frame_entries(self.frame_names, current_key=preferred)
        self.target_layout.addWidget(self.exposed_frame_list)

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
            "X [m]",
            min_value=-2000,
            max_value=2000,
            initial_value=0,
            scale=1000,
        )

        self.y_slider = LabeledSlider(
            "Y [m]",
            min_value=-1000,
            max_value=1000,
            initial_value=0,
            scale=1000,
        )

        self.z_slider = LabeledSlider(
            "Z [m]",
            min_value=0,
            max_value=2000,
            initial_value=900,
            scale=1000,
        )

        self.roll_slider = LabeledSlider(
            "Roll [rad]",
            min_value=-314,
            max_value=314,
            initial_value=0,
            scale=100,
        )

        self.pitch_slider = LabeledSlider(
            "Pitch [rad]",
            min_value=-157,
            max_value=157,
            initial_value=0,
            scale=100,
        )

        self.yaw_slider = LabeledSlider(
            "Yaw [rad]",
            min_value=-314,
            max_value=314,
            initial_value=0,
            scale=100,
        )

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

        self.corner_smoothing_slider = LabeledSlider(
            "Smoothing [%]",
            min_value=0,
            max_value=100,
            initial_value=0,
            scale=1,
        )
        self.corner_smoothing_slider.input.setMaximumWidth(52)
        self.corner_smoothing_slider.setMaximumWidth(212)

        # --------------------------------------------------------
        # Keyframe buttons
        # --------------------------------------------------------
        self.add_button = QPushButton("Add Keyframe")
        self.update_button = QPushButton("Update")
        self.delete_button = QPushButton("Delete")
        self.clear_button = QPushButton("Clear Trajectory")

        self.generate_button = QPushButton("Generate / Simulate")

        self.add_button.clicked.connect(self.add_keyframe_clicked.emit)
        self.update_button.clicked.connect(self.update_keyframe_clicked.emit)
        self.delete_button.clicked.connect(self.delete_keyframe_clicked.emit)
        self.clear_button.clicked.connect(self.clear_trajectory_clicked.emit)
        self.generate_button.clicked.connect(self.generate_clicked.emit)

        # --------------------------------------------------------
        # Keyframe table
        # --------------------------------------------------------
        self.table = QTableWidget()
        self.table.setMinimumHeight(96)
        self.table.setMaximumHeight(180)
        self.table.setMaximumWidth(212)
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
            ("Target", self.target_panel, True),
            ("Pose", self.transform_panel, True),
            ("Trajectory", self.trajectory_panel, True),
            ("Advanced IK", self.preview_ik_panel, False),
        ]

    def inspector_sections(self):
        return []

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
        self.set_exposed_frame_entries(names, current_key=choice)
        self.frame_name_changed.emit(choice)
        self.exposed_frames_changed.emit(self.exposed_frame_names())

    def set_exposed_frame_entries(self, entries, checked_keys=None, current_key=None):
        normalized = []
        for entry in entries:
            if isinstance(entry, dict):
                key = entry.get("key")
                label = entry.get("label", key)
                checked = bool(entry.get("checked", False))
            elif isinstance(entry, (tuple, list)):
                key = entry[0]
                label = entry[1] if len(entry) > 1 else key
                checked = bool(entry[2]) if len(entry) > 2 else key in self.frame_names
            else:
                key = str(entry)
                label = key
                checked = key in self.frame_names
            if not key:
                continue
            normalized.append({
                "key": str(key),
                "label": str(label),
                "checked": checked,
            })
        if not normalized:
            normalized = [
                {"key": name, "label": name, "checked": True}
                for name in self.frame_names
            ]
        if checked_keys is not None:
            checked = set(checked_keys)
            for entry in normalized:
                entry["checked"] = entry["key"] in checked
        self.target_frame_entries = normalized
        self.target_frame_keys = [entry["key"] for entry in normalized]
        target_key = (
            current_key
            if current_key in self.target_frame_keys
            else self.current_target_key
            if self.current_target_key in self.target_frame_keys
            else self.frame_box.currentText()
            if self.frame_box.currentText() in self.target_frame_keys
            else self.target_frame_keys[0]
        )
        self._populate_exposed_frame_list(normalized, target_key)
        self.current_target_key = target_key

    def exposed_frame_names(self):
        names = []
        for index in range(self.exposed_frame_list.count()):
            item = self.exposed_frame_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.data(Qt.ItemDataRole.UserRole))
        return names

    def set_exposed_frame_names(self, frame_names, emit=True):
        exposed = set(frame_names)
        self._syncing_frame_selection = True
        self.exposed_frame_list.blockSignals(True)
        try:
            for index in range(self.exposed_frame_list.count()):
                item = self.exposed_frame_list.item(index)
                state = (
                    Qt.CheckState.Checked
                    if item.data(Qt.ItemDataRole.UserRole) in exposed
                    else Qt.CheckState.Unchecked
                )
                item.setCheckState(state)
        finally:
            self.exposed_frame_list.blockSignals(False)
            self._syncing_frame_selection = False
        if emit:
            self.exposed_frames_changed.emit(self.exposed_frame_names())

    def set_current_frame_name(self, frame_name, emit=True):
        if frame_name not in self.target_frame_keys:
            return False
        self._syncing_frame_selection = True
        frame_blocked = self.frame_box.blockSignals(True)
        list_blocked = self.exposed_frame_list.blockSignals(True)
        try:
            if frame_name in self.frame_names:
                self.frame_box.setCurrentText(frame_name)
            self.current_target_key = frame_name
            self._select_exposed_frame_item(frame_name)
        finally:
            self.exposed_frame_list.blockSignals(list_blocked)
            self.frame_box.blockSignals(frame_blocked)
            self._syncing_frame_selection = False
        if emit:
            self.frame_name_changed.emit(frame_name)
        return True

    def current_frame_name(self):
        return self.current_target_key or self.frame_box.currentText()

    def _populate_exposed_frame_list(self, entries, current_frame):
        self._syncing_frame_selection = True
        blocked = self.exposed_frame_list.blockSignals(True)
        try:
            self.exposed_frame_list.clear()
            for entry in entries:
                frame_name = entry["key"]
                item = QListWidgetItem(entry["label"])
                item.setData(Qt.ItemDataRole.UserRole, frame_name)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if entry["checked"]
                    else Qt.CheckState.Unchecked
                )
                self.exposed_frame_list.addItem(item)
            self._select_exposed_frame_item(current_frame)
        finally:
            self.exposed_frame_list.blockSignals(blocked)
            self._syncing_frame_selection = False

    def _select_exposed_frame_item(self, frame_name):
        for index in range(self.exposed_frame_list.count()):
            item = self.exposed_frame_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == frame_name:
                self.exposed_frame_list.setCurrentRow(index)
                return True
        return False

    def _on_frame_box_text_changed(self, frame_name):
        if self._syncing_frame_selection:
            return
        self._syncing_frame_selection = True
        blocked = self.exposed_frame_list.blockSignals(True)
        try:
            self.current_target_key = frame_name
            self._select_exposed_frame_item(frame_name)
        finally:
            self.exposed_frame_list.blockSignals(blocked)
            self._syncing_frame_selection = False
        self.frame_name_changed.emit(frame_name)

    def _on_exposed_frame_current_changed(self, current, previous):
        if self._syncing_frame_selection or current is None:
            return
        frame_name = current.data(Qt.ItemDataRole.UserRole)
        self._syncing_frame_selection = True
        blocked = self.frame_box.blockSignals(True)
        try:
            if frame_name in self.frame_names:
                self.frame_box.setCurrentText(frame_name)
            self.current_target_key = frame_name
        finally:
            self.frame_box.blockSignals(blocked)
            self._syncing_frame_selection = False
        self.frame_name_changed.emit(frame_name)

    def _on_exposed_frame_item_changed(self, item):
        if self._syncing_frame_selection:
            return
        self.exposed_frames_changed.emit(self.exposed_frame_names())

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
            frame_name=self.current_frame_name(),
            x=self.x_slider.value(),
            y=self.y_slider.value(),
            z=self.z_slider.value(),
            roll=self.roll_slider.value(),
            pitch=self.pitch_slider.value(),
            yaw=self.yaw_slider.value(),
        )

    def show_trajectory_lines(self):
        return self.show_lines_box.isChecked()

    def corner_smoothing(self):
        return max(0.0, min(1.0, self.corner_smoothing_slider.value() / 100.0))

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
            self.set_current_frame_name(frame.frame_name, emit=False)
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

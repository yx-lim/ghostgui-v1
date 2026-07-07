"""MoveIt-style live robot editor wrapped around GhostGUI's OpenGL canvas."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .robot_model_3d import (
    RobotStateTimeline,
    TrajectoryGhostRenderer,
    interpolate_qpos,
)
from .collision_checker import CollisionAwareIKSolver, CollisionChecker
from .trajectory import quat_to_rpy, rpy_to_quat
from .viewer_3d import RobotCanvas3D
from .collapsible_sidebar import CollapsibleSidebar
from .ik_tasks import (
    FootLockTask,
    JointRegularizationTask,
    PostureTask,
    RootPoseTask,
)


FRAME_BINDINGS = {
    "pelvis": ("body", "robot/pelvis"),
    "torso": ("body", "robot/torso_link"),
    "left_foot": ("site", "robot/left_foot"),
    "right_foot": ("site", "robot/right_foot"),
    "left_hand": ("site", "robot/left_palm"),
    "right_hand": ("site", "robot/right_palm"),
}
REVERSE_BINDINGS = {value: key for key, value in FRAME_BINDINGS.items()}


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


class RobotViewer3D(QWidget):
    """Live FK/IK/trajectory viewer. It preserves the old canvas contract."""

    target_dragged = Signal(float, float)
    target_pose_dragged = Signal(float, float, float, float, float, float)
    target_pose_drag_finished = Signal(float, float, float, float, float, float)
    target_frame_changed = Signal(str)
    preview_cancelled = Signal()

    def __init__(self, robot_model=None, error=None):
        super().__init__()
        self.robot_model = robot_model
        self.frame_bindings = (
            dict(robot_model.logical_frame_bindings)
            if robot_model is not None and hasattr(robot_model, "logical_frame_bindings")
            else dict(FRAME_BINDINGS)
        )
        self.reverse_bindings = {value: key for key, value in self.frame_bindings.items()}
        self.committed_state = robot_model.create_state() if robot_model else None
        self.preview_state = robot_model.create_state() if robot_model else None
        # Compatibility alias: external readers historically used robot_state.
        # It now always means the timeline-backed committed state.
        self.robot_state = self.committed_state
        self.preview_active = False
        self.current_time = 0.0
        self.state_timeline = (
            RobotStateTimeline(robot_model, initial_qpos=self.robot_state.get_qpos())
            if robot_model else None
        )
        self.ghost_renderer = TrajectoryGhostRenderer(robot_model) if robot_model else None
        self.collision_checker = CollisionChecker(robot_model) if robot_model else None
        self.collision_solver = (
            CollisionAwareIKSolver(
                robot_model,
                self.collision_checker,
                orientation_weight=(
                    0.0 if getattr(robot_model, "model_type", None) == "quadruped"
                    else 0.25
                ),
            )
            if robot_model else None
        )
        self.robot_trajectory = []
        self.ghost_trajectory = []
        self.joint_controls = {}
        self.ik_influence_controls = {}
        self.ik_joint_weights = {
            name: 1.0 for name in (
                robot_model.get_joint_names() if robot_model else []
            )
        }
        self.preview_reference_qpos = None
        self.foot_lock_targets = {}
        self.root_lock_target = None
        self._last_ik_status = None
        self._syncing_target = False
        self.canvas = RobotCanvas3D()
        self.canvas.geometry_progress.connect(self._on_geometry_progress)
        self.canvas.target_dragged.connect(self.target_dragged.emit)
        self.canvas.target_transform_dragged.connect(self._on_transform_moved)
        self.canvas.transform_drag_finished.connect(
            self._on_transform_drag_finished
        )
        self.canvas.body_double_clicked.connect(self._on_body_double_clicked)
        self.last_valid_target_position = None
        self.last_valid_target_quaternion = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._advance_frame)
        self._build_ui(error)
        if self.robot_state:
            self.canvas.set_robot_states(
                self.committed_state, self.preview_state, self.ghost_renderer
            )
            self._set_target_to_selected_pose()

    def _build_ui(self, error):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        controls = QWidget()
        controls.setMaximumWidth(390)
        panel = QVBoxLayout(controls)
        model_text = str(self.robot_model.model_path) if self.robot_model else "Unavailable"
        panel.addWidget(QLabel(f"Model: {model_text}"))
        self.status_label = QLabel(error or "Robot model loaded; FK ready.")
        self.status_label.setWordWrap(True)
        panel.addWidget(self.status_label)
        self.timeline_state_label = QLabel("3D state time: 0.00 s")
        panel.addWidget(self.timeline_state_label)

        self.model_colors_box = QCheckBox("Use model colors")
        self.model_colors_box.setChecked(True)
        self.model_colors_box.toggled.connect(self.canvas.set_use_model_colors)
        panel.addWidget(self.model_colors_box)
        if self.robot_model:
            texture_warnings = self.robot_model.get_visual_texture_warnings()
            if texture_warnings:
                warning = QLabel("; ".join(texture_warnings))
                warning.setWordWrap(True)
                panel.addWidget(warning)

        self.reset_button = QPushButton("Reset 3D Pose")
        self.reset_button.clicked.connect(self.reset_robot_pose)
        panel.addWidget(self.reset_button)

        pose_file_row = QHBoxLayout()
        self.load_qpos_button = QPushButton("Load qpos CSV")
        self.save_qpos_button = QPushButton("Save qpos CSV")
        self.load_qpos_button.clicked.connect(self.choose_qpos_csv)
        self.save_qpos_button.clicked.connect(self.choose_qpos_save_path)
        pose_file_row.addWidget(self.load_qpos_button)
        pose_file_row.addWidget(self.save_qpos_button)
        panel.addLayout(pose_file_row)

        target_group = QGroupBox("End-effector transform gizmo")
        target_layout = QFormLayout(target_group)
        self.target_box = QComboBox()
        if self.robot_model:
            for frame_name, (kind, name) in self.frame_bindings.items():
                try:
                    self.robot_state.resolve_object(name, kind)
                except KeyError:
                    continue
                self.target_box.addItem(frame_name.replace("_", " "), (kind, name))
            known = {self.target_box.itemData(i) for i in range(self.target_box.count())}
            for kind, names in (("site", self.robot_model.site_names),
                                ("body", self.robot_model.body_names)):
                for name in names:
                    if (kind, name) not in known and name != "world":
                        self.target_box.addItem(f"{kind}: {name}", (kind, name))
        self.target_box.currentIndexChanged.connect(self._target_selected)
        self.collision_substeps = QSpinBox()
        self.collision_substeps.setRange(1, 32)
        self.collision_substeps.setValue(8)
        self.collision_substeps.valueChanged.connect(
            self._set_collision_substeps
        )
        target_layout.addRow("Target", self.target_box)
        target_layout.addRow(QLabel(
            "Drag arrows to translate; drag rings to rotate (world axes)."
        ))
        target_layout.addRow("Collision substeps", self.collision_substeps)
        self.root_pose_label = QLabel()
        target_layout.addRow("Root pose", self.root_pose_label)
        panel.addWidget(target_group)

        preview_group = QGroupBox("Preview workflow")
        preview_layout = QHBoxLayout(preview_group)
        self.plan_preview_button = QPushButton("Plan Preview")
        self.accept_preview_button = QPushButton("Accept Preview")
        self.cancel_preview_button = QPushButton("Cancel Preview")
        self.preview_alpha = QDoubleSpinBox()
        self.preview_alpha.setRange(0.1, 1.0)
        self.preview_alpha.setSingleStep(0.05)
        self.preview_alpha.setValue(0.65)
        self.preview_alpha.valueChanged.connect(self.canvas.set_preview_alpha)
        self.plan_preview_button.clicked.connect(self.plan_preview)
        self.accept_preview_button.clicked.connect(self.accept_preview)
        self.cancel_preview_button.clicked.connect(self.cancel_preview)
        preview_layout.addWidget(self.plan_preview_button)
        preview_layout.addWidget(self.accept_preview_button)
        preview_layout.addWidget(self.cancel_preview_button)
        preview_layout.addWidget(QLabel("Alpha"))
        preview_layout.addWidget(self.preview_alpha)
        panel.addWidget(preview_group)

        trajectory_group = QGroupBox("Trajectory / ghosts")
        trajectory_layout = QFormLayout(trajectory_group)
        self.generate_button = QPushButton("Generate demo trajectory")
        self.generate_button.clicked.connect(self.generate_demo_trajectory)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.set_trajectory_frame)
        self.show_ghosts = QCheckBox("Show trajectory ghosts")
        self.show_ghosts.toggled.connect(self._update_ghost_options)
        self.ghost_stride = QSpinBox()
        self.ghost_stride.setRange(1, 100)
        self.ghost_stride.setValue(8)
        self.ghost_stride.valueChanged.connect(self._rebuild_ghosts)
        self.ghost_alpha = QDoubleSpinBox()
        self.ghost_alpha.setRange(0.02, 0.8)
        self.ghost_alpha.setSingleStep(0.05)
        self.ghost_alpha.setValue(0.16)
        self.ghost_alpha.valueChanged.connect(self._update_ghost_options)
        trajectory_layout.addRow(self.generate_button)
        trajectory_layout.addRow(self.play_button, self.show_ghosts)
        trajectory_layout.addRow("Frame", self.frame_slider)
        trajectory_layout.addRow("Ghost stride", self.ghost_stride)
        trajectory_layout.addRow("Ghost alpha", self.ghost_alpha)
        panel.addWidget(trajectory_group)

        editor_tabs = QTabWidget()
        joint_group = QGroupBox("Controllable joints")
        joint_layout = QVBoxLayout(joint_group)
        if self.robot_state:
            for name in self.robot_state.get_joint_names():
                control = JointControl(
                    name,
                    self.robot_state.get_joint_limits(name),
                    self.robot_state.get_joint_value(name),
                )
                control.value_changed.connect(self._joint_changed)
                self.joint_controls[name] = control
                joint_layout.addWidget(control)
        else:
            joint_layout.addWidget(QLabel("Joint controls unavailable."))
        joint_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(joint_group)
        editor_tabs.addTab(scroll, "Joint angles")
        editor_tabs.addTab(self._build_ik_settings_widget(), "IK controls")
        panel.addWidget(editor_tabs, 1)
        self.controls_sidebar = CollapsibleSidebar(
            "3D",
            controls,
            side="right",
            minimum_expanded_width=320,
            maximum_expanded_width=430,
        )
        self.viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer_splitter.setChildrenCollapsible(False)
        self.viewer_splitter.setHandleWidth(4)
        self.viewer_splitter.addWidget(self.canvas)
        self.viewer_splitter.addWidget(self.controls_sidebar)
        self.viewer_splitter.setStretchFactor(0, 1)
        self.viewer_splitter.setStretchFactor(1, 0)
        self.viewer_splitter.setSizes([900, 380])
        self.viewer_splitter.splitterMoved.connect(
            lambda position, index: self.controls_sidebar.remember_width(
                self.viewer_splitter.sizes()[1]
            )
        )
        root.addWidget(self.viewer_splitter)

        enabled = self.robot_state is not None
        self.reset_button.setEnabled(enabled)
        self.load_qpos_button.setEnabled(enabled)
        self.save_qpos_button.setEnabled(enabled)
        target_group.setEnabled(enabled)
        trajectory_group.setEnabled(enabled)
        preview_group.setEnabled(enabled)

    def _build_ik_settings_widget(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        solver_group = QGroupBox("Solver settings")
        solver_layout = QFormLayout(solver_group)
        self.ik_damping = QDoubleSpinBox()
        self.ik_damping.setRange(0.001, 1.0)
        self.ik_damping.setDecimals(4)
        self.ik_damping.setValue(0.04)
        self.ik_max_iterations = QSpinBox()
        self.ik_max_iterations.setRange(1, 300)
        self.ik_max_iterations.setValue(80)
        self.ik_step_size = QDoubleSpinBox()
        self.ik_step_size.setRange(0.01, 1.0)
        self.ik_step_size.setValue(0.7)
        self.ik_max_step = QDoubleSpinBox()
        self.ik_max_step.setRange(0.001, 0.5)
        self.ik_max_step.setDecimals(3)
        self.ik_max_step.setValue(0.08)
        self.ik_position_tolerance = QDoubleSpinBox()
        self.ik_position_tolerance.setRange(0.0001, 0.1)
        self.ik_position_tolerance.setDecimals(4)
        self.ik_position_tolerance.setValue(0.005)
        self.ik_orientation_tolerance = QDoubleSpinBox()
        self.ik_orientation_tolerance.setRange(0.001, 1.0)
        self.ik_orientation_tolerance.setDecimals(3)
        self.ik_orientation_tolerance.setValue(0.03)
        solver_layout.addRow("Damping", self.ik_damping)
        solver_layout.addRow("Max iterations", self.ik_max_iterations)
        solver_layout.addRow("Step size", self.ik_step_size)
        solver_layout.addRow("Max joint step", self.ik_max_step)
        solver_layout.addRow("Position tolerance", self.ik_position_tolerance)
        solver_layout.addRow("Orientation tolerance", self.ik_orientation_tolerance)
        layout.addWidget(solver_group)

        task_group = QGroupBox("Weighted tasks (priority metadata; weighted v1)")
        task_layout = QFormLayout(task_group)
        self.ik_task_controls = {}
        defaults = {
            "tcp_position": (True, 1.0),
            "tcp_orientation": (
                self.robot_model.model_type != "quadruped", 0.25
            ),
            # Secondary pose objectives are opt-in. Enabling them by default
            # makes an ordinary TCP drag settle at a weighted compromise and
            # look like an artificial range limit.
            "posture": (False, 0.05),
            "foot_lock": (False, 0.5),
            "root_orientation": (True, 0.1),
            "regularization": (False, 0.01),
        }
        labels = {
            "tcp_position": "TCP position",
            "tcp_orientation": "TCP orientation",
            "posture": "Posture preservation",
            "foot_lock": "Foot lock",
            "root_orientation": "Root/base upright",
            "regularization": "Joint regularization",
        }
        for key, (enabled, weight) in defaults.items():
            checkbox = QCheckBox()
            checkbox.setChecked(enabled)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setDecimals(3)
            spin.setValue(weight)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(spin)
            self.ik_task_controls[key] = (checkbox, spin)
            task_layout.addRow(labels[key], row)
        layout.addWidget(task_group)

        influence_group = QGroupBox("Per-joint IK influence (0=locked, 1=normal)")
        influence_layout = QVBoxLayout(influence_group)
        preset_row = QHBoxLayout()
        self.ik_preset_box = QComboBox()
        presets = ["All joints normal", "Root locked", "Selected limb only", "Feet planted"]
        if self.robot_model.model_type == "humanoid":
            presets.extend(("Upper body only", "Legs only"))
        elif self.robot_model.model_type == "quadruped":
            presets.append("Quadruped legs only")
        self.ik_preset_box.addItems(presets)
        apply_preset = QPushButton("Apply")
        apply_preset.clicked.connect(self.apply_ik_preset)
        preset_row.addWidget(self.ik_preset_box, 1)
        preset_row.addWidget(apply_preset)
        influence_layout.addLayout(preset_row)
        for name in self.robot_model.get_joint_names():
            control = IKInfluenceControl(name, 1.0)
            control.value_changed.connect(self._ik_influence_changed)
            self.ik_influence_controls[name] = control
            influence_layout.addWidget(control)
        influence_layout.addStretch()
        layout.addWidget(influence_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _on_geometry_progress(self, complete, total):
        if total <= 0:
            return
        if complete < total:
            self.status_label.setText(
                f"Preparing 3D geometry… {complete}/{total}"
            )
        else:
            self.status_label.setText("3D geometry ready.")

    def update_scene(self, trajectory, active_frame=None, show_trajectory_lines=True):
        # The live gizmo owns its quaternion while editing. Ordinary status
        # refreshes must not reset an in-progress ring rotation from controls.
        self.canvas.update_scene(trajectory, None, show_trajectory_lines)
        if active_frame is not None:
            binding = self.frame_bindings.get(active_frame.frame_name)
            if binding is not None:
                self.select_target(*binding, emit=False)

    def select_target(self, kind, name, emit=False):
        for index in range(self.target_box.count()):
            if self.target_box.itemData(index) == (kind, name):
                self._syncing_target = not emit
                self.target_box.setCurrentIndex(index)
                self._syncing_target = False
                return True
        return False

    def _selected_target(self):
        data = self.target_box.currentData()
        return tuple(data) if data else (None, None)

    def _target_selected(self):
        if not self.robot_state:
            return
        kind, name = self._selected_target()
        self._set_target_to_selected_pose()
        frame_name = self.reverse_bindings.get((kind, name))
        if frame_name and not self._syncing_target:
            self.target_frame_changed.emit(frame_name)

    def _set_target_to_selected_pose(self):
        kind, name = self._selected_target()
        if not name:
            return
        try:
            state = self.preview_state if self.preview_active else self.committed_state
            position, quaternion = state.get_body_pose(name, kind)
        except KeyError as exc:
            self.status_label.setText(str(exc))
            return
        self.canvas.set_target_pose(position, quaternion)
        self.last_valid_target_position = position.copy()
        self.last_valid_target_quaternion = quaternion.copy()
        self._update_root_pose_label()

    def _joint_changed(self, name, value):
        self.begin_preview()
        self.preview_state.set_joint_value(name, value)
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Preview FK: {name} = {value:+.3f} rad; Accept Preview to commit"
        )

    def _sync_joint_controls(self):
        state = self.preview_state if self.preview_active else self.committed_state
        for name, control in self.joint_controls.items():
            control.set_value(state.get_joint_value(name))

    def _ik_influence_changed(self, name, value):
        self.ik_joint_weights[name] = float(value)
        state = "locked" if value <= 1e-9 else f"influence {value:.2f}"
        self.status_label.setText(f"IK joint {name}: {state}")

    def _set_all_ik_influences(self, selector):
        for name, control in self.ik_influence_controls.items():
            value = float(selector(name))
            self.ik_joint_weights[name] = value
            control.set_value(value)

    def apply_ik_preset(self):
        preset = self.ik_preset_box.currentText()
        if preset == "All joints normal":
            self._set_all_ik_influences(lambda name: 1.0)
        elif preset == "Root locked":
            # Floating roots are already excluded from limb IK. Keep actuated
            # joints normal and make that hard-lock explicit in status.
            self._set_all_ik_influences(lambda name: 1.0)
        elif preset == "Upper body only":
            tokens = ("waist", "shoulder", "elbow", "wrist", "arm")
            self._set_all_ik_influences(
                lambda name: 1.0 if any(token in name.lower() for token in tokens) else 0.0
            )
        elif preset in ("Legs only", "Quadruped legs only"):
            tokens = ("hip", "thigh", "knee", "calf", "ankle", "leg")
            self._set_all_ik_influences(
                lambda name: 1.0 if any(token in name.lower() for token in tokens) else 0.0
            )
        elif preset == "Selected limb only":
            kind, object_name = self._selected_target()
            logical = self.reverse_bindings.get((kind, object_name), "")
            lower = logical.lower()
            side = next(
                (token for token in ("left", "right", "fl", "fr", "rl", "rr")
                 if lower.startswith(token)),
                "",
            )
            limb_tokens = (
                ("shoulder", "elbow", "wrist", "arm")
                if "hand" in lower else
                ("hip", "thigh", "knee", "calf", "ankle", "leg")
            )
            self._set_all_ik_influences(lambda name: 1.0 if (
                (not side or name.lower().startswith(side))
                and any(token in name.lower() for token in limb_tokens)
            ) else 0.0)
        elif preset == "Feet planted":
            self._set_all_ik_influences(lambda name: 1.0)
            checkbox, spin = self.ik_task_controls["foot_lock"]
            checkbox.setChecked(True)
            spin.setValue(max(1.0, spin.value()))
        self.status_label.setText(f"Applied IK preset: {preset}")

    def _task_setting(self, name):
        checkbox, spin = self.ik_task_controls[name]
        return checkbox.isChecked(), float(spin.value())

    def _solver_settings(self):
        return {
            "damping": self.ik_damping.value(),
            "max_iterations": self.ik_max_iterations.value(),
            "step_size": self.ik_step_size.value(),
            "max_step": self.ik_max_step.value(),
            "position_tolerance": self.ik_position_tolerance.value(),
            "orientation_tolerance": self.ik_orientation_tolerance.value(),
        }

    def _capture_secondary_targets(self):
        self.preview_reference_qpos = self.committed_state.get_qpos()
        self.foot_lock_targets = {}
        for logical, (kind, name) in self.frame_bindings.items():
            if "foot" not in logical.lower():
                continue
            position, quaternion = self.committed_state.get_body_pose(name, kind)
            self.foot_lock_targets[logical] = (
                kind, name, position.copy(), quaternion.copy()
            )
        root_name = self.robot_model.root_body
        self.root_lock_target = (
            self.committed_state.get_body_pose(root_name, "body")
            if root_name else None
        )

    def _secondary_ik_tasks(self):
        tasks = []
        posture_enabled, posture_weight = self._task_setting("posture")
        if posture_enabled and posture_weight > 0.0:
            tasks.append(PostureTask(
                name="Posture preservation", weight=posture_weight,
                priority=3, enabled=True, required=False, tolerance=0.2,
                reference_qpos=self.preview_reference_qpos,
            ))
        regularization_enabled, regularization_weight = self._task_setting(
            "regularization"
        )
        if regularization_enabled and regularization_weight > 0.0:
            tasks.append(JointRegularizationTask(
                name="Joint regularization", weight=regularization_weight,
                priority=3, enabled=True, required=False, tolerance=0.3,
                reference_qpos=self.robot_model.home_qpos,
            ))
        selected = self.reverse_bindings.get(self._selected_target())
        foot_enabled, foot_weight = self._task_setting("foot_lock")
        if foot_enabled and foot_weight > 0.0:
            for logical, (kind, name, position, _) in self.foot_lock_targets.items():
                if logical == selected:
                    continue
                tasks.append(FootLockTask(
                    name=f"Lock {logical}", weight=foot_weight,
                    priority=1, enabled=True, required=False, tolerance=0.005,
                    object_name=name, kind=kind, target_position=position,
                ))
        root_enabled, root_weight = self._task_setting("root_orientation")
        if (
            root_enabled and root_weight > 0.0
            and self.root_lock_target is not None and self.robot_model.root_body
        ):
            _, quaternion = self.root_lock_target
            tasks.append(RootPoseTask(
                name="Root/base upright", weight=root_weight,
                priority=1, enabled=True, required=False, tolerance=0.05,
                object_name=self.robot_model.root_body, kind="body",
                target_quaternion=quaternion,
            ))
        return tasks

    def begin_preview(self):
        if not self.robot_state or self.preview_active:
            return
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self._capture_secondary_targets()
        self.preview_active = True
        self.canvas.set_preview_visible(True)
        self._update_root_pose_label()

    def _on_transform_moved(self, position, quaternion):
        kind, name = self._selected_target()
        if not name:
            self.status_label.setText("Target pose moved; no body/site is selected.")
            return
        self.begin_preview()
        if self.last_valid_target_position is None:
            current_position, current_quaternion = self.preview_state.get_body_pose(
                name, kind
            )
            self.last_valid_target_position = current_position
            self.last_valid_target_quaternion = current_quaternion

        secondary_tasks = self._secondary_ik_tasks()
        tcp_position_enabled, tcp_position_weight = self._task_setting(
            "tcp_position"
        )
        tcp_orientation_enabled, tcp_orientation_weight = self._task_setting(
            "tcp_orientation"
        )
        selected_task_count = int(
            tcp_position_enabled and tcp_position_weight > 0.0
        ) + int(tcp_orientation_enabled and tcp_orientation_weight > 0.0)
        _, selected_object_id = self.preview_state.resolve_object(name, kind)
        is_free_root = (
            kind == "body"
            and self.robot_model.free_joint_for_body(selected_object_id) is not None
        )
        result = self.collision_solver.solve_drag(
            self.preview_state.get_qpos(),
            self.last_valid_target_position,
            self.last_valid_target_quaternion,
            position,
            quaternion,
            object_name=name,
            kind=kind,
            joint_weights=self.ik_joint_weights,
            secondary_tasks=secondary_tasks,
            solver_settings=self._solver_settings(),
            tcp_position_weight=(
                tcp_position_weight if tcp_position_enabled else 0.0
            ),
            tcp_orientation_weight=(
                tcp_orientation_weight if tcp_orientation_enabled else 0.0
            ),
        )
        if result.success:
            self.preview_state.set_qpos(result.qpos)
            self.last_valid_target_position = result.position.copy()
            self.last_valid_target_quaternion = result.quaternion.copy()

        # Whether fully accepted, clamped, or rejected, snap the handle back to
        # the last collision-free pose rather than displaying an invalid target.
        self.canvas.set_target_pose(
            self.last_valid_target_position,
            self.last_valid_target_quaternion,
        )
        self._sync_joint_controls()
        logical_frame = self.reverse_bindings.get((kind, name), name)
        model_name = getattr(
            self.robot_model, "model_name", self.robot_model.model_path.stem
        )
        active_task_count = selected_task_count + (
            0 if is_free_root else len(secondary_tasks)
        )
        self._last_ik_status = (
            f"{'TCP free translate; ' if self.canvas.gizmo.state.name == 'DRAG_TRANSLATE_FREE' else ''}"
            f"{result.status}; accepted={result.accepted_fraction:.0%}; "
            f"IK error={result.ik_error:.4f}; tasks={active_task_count}; "
            f"frame={logical_frame}; model={model_name}; preview not committed"
        )
        self.status_label.setText(self._last_ik_status)
        roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
        self.target_pose_dragged.emit(
            *map(float, self.last_valid_target_position), roll, pitch, yaw
        )

    def _on_gizmo_moved(self, x, y, z):
        """Compatibility shim for older callers/tests using position only."""
        self._on_transform_moved(
            (x, y, z), self.canvas.gizmo.quaternion.copy()
        )

    def _set_collision_substeps(self, count):
        if self.collision_solver:
            self.collision_solver.collision_drag_substeps = int(count)

    def _on_transform_drag_finished(self):
        if self.last_valid_target_position is not None:
            detail = self._last_ik_status or "Preview pose updated"
            self.status_label.setText(
                f"{detail}; ready to Plan, Accept, or Cancel"
            )

    def plan_preview(self):
        if not self.preview_active:
            self.status_label.setText("No preview changes to plan.")
            return
        start = self.committed_state.get_qpos()
        goal = self.preview_state.get_qpos()
        planned = [
            self.state_timeline._interpolate(start, goal, alpha)
            for alpha in np.linspace(0.0, 1.0, 40)
        ]
        self.robot_trajectory = planned
        self.ghost_trajectory = list(planned)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, len(planned) - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.show_ghosts.setChecked(True)
        self._rebuild_ghosts()
        self.status_label.setText(
            "Planned committed-to-preview path; no timeline state was changed."
        )

    def accept_preview(self):
        if not self.preview_active:
            self.status_label.setText("No preview changes to accept.")
            return
        self.committed_state.set_qpos(self.preview_state.get_qpos())
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_visible(False)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Accepted preview into committed keyframe at t={self.current_time:.2f} s"
        )
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )

    def cancel_preview(self):
        if not self.robot_state:
            return
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_visible(False)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText("Preview discarded; committed state is unchanged.")
        self.preview_cancelled.emit()

    def _on_body_double_clicked(self, body_name):
        logical = (
            self.robot_model.logical_frame_for_body(body_name)
            if hasattr(self.robot_model, "logical_frame_for_body") else None
        )
        binding = self.frame_bindings.get(logical) if logical else None
        if binding is None or not self.select_target(*binding, emit=False):
            self.status_label.setText(
                f"Body {body_name!r} has no nearby editable trajectory frame."
            )
            return
        self.target_frame_changed.emit(logical)
        self.status_label.setText(
            f"Selected {logical} from double-clicked body {body_name}."
        )

    def get_current_time(self):
        return self.current_time

    def get_current_keyframe(self):
        if not self.state_timeline:
            return None
        return self.state_timeline.get_state(self.current_time)

    def ensure_keyframe_at_current_time(self):
        if not self.state_timeline:
            return None
        return self.state_timeline.ensure_state(
            self.current_time,
            fallback_qpos=self.robot_state.get_qpos(),
        )

    def set_robot_state_for_current_time(self, qpos):
        if not self.robot_state:
            return
        self.committed_state.set_qpos(qpos)
        self.preview_state.set_qpos(qpos)
        self.preview_active = False
        self.canvas.set_preview_visible(False)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.canvas.update()

    def update_current_keyframe_from_robot_state(self, refresh_ghosts=True):
        if not self.state_timeline or not self.robot_state:
            return
        self.state_timeline.set_state(
            self.current_time, self.committed_state.get_qpos()
        )
        self._update_timeline_label()
        if refresh_ghosts:
            self._refresh_timeline_trajectory()

    def set_current_time(self, time):
        if not self.state_timeline:
            return
        self.current_time = self.state_timeline.time_key(time)
        qpos = self.ensure_keyframe_at_current_time()
        self.set_robot_state_for_current_time(qpos)
        self._refresh_timeline_trajectory()
        self._update_timeline_label()
        self.status_label.setText(
            f"Loaded editable 3D robot state at t={self.current_time:.2f} s"
        )

    def reset_robot_pose(self):
        """Reset only the active 3D timeline frame to the model home pose."""
        if not self.robot_state:
            return
        # Reset is an immediate one-shot action, never a playback mode. If the
        # timer is active, pause first so its next tick cannot overwrite the
        # just-reset editing frame, and cancel any in-progress gizmo gesture.
        was_playing = self.play_timer.isActive()
        self.pause_playback()
        self.canvas.cancel_transform_drag()
        self.committed_state.reset_to_default()
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_visible(False)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Reset 3D pose once at t={self.current_time:.2f} s to model home qpos"
            + ("; playback paused" if was_playing else "")
        )
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )

    def choose_qpos_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load robot qpos",
            str(Path.cwd()),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                self.load_qpos_csv(path)
            except (OSError, ValueError) as exc:
                self.status_label.setText(f"Could not load qpos CSV: {exc}")

    def choose_qpos_save_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save robot qpos",
            str(Path.cwd() / "updated_qpos.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                self.save_qpos_csv(path)
            except OSError as exc:
                self.status_label.setText(f"Could not save qpos CSV: {exc}")

    def load_qpos_csv(self, csv_path):
        """Load one headerless MuJoCo qpos row into the active keyframe."""
        path = Path(csv_path).expanduser().resolve()
        with path.open("r", newline="") as handle:
            rows = [
                row
                for row in csv.reader(handle)
                if any(cell.strip() for cell in row)
            ]
        if not rows:
            raise ValueError("the file is empty")
        try:
            qpos = np.asarray([float(cell.strip()) for cell in rows[0]], dtype=float)
        except ValueError as exc:
            raise ValueError(
                "expected a headerless row containing only qpos numbers"
            ) from exc
        expected = int(self.robot_model.mj_model.nq)
        if qpos.shape != (expected,):
            raise ValueError(
                f"expected {expected} qpos values for this model, found {qpos.size}"
            )

        self.pause_playback()
        self.canvas.cancel_transform_drag()
        self.set_robot_state_for_current_time(qpos)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self.status_label.setText(
            f"Loaded {expected}-value qpos from {path.name} at "
            f"t={self.current_time:.2f} s"
        )

    def save_qpos_csv(self, csv_path):
        """Save the committed active keyframe as one headerless qpos row."""
        path = Path(csv_path).expanduser()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        path = path.resolve()
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                f"{value:.18e}" for value in self.committed_state.get_qpos()
            )
        preview_note = (
            "; unaccepted preview was not saved" if self.preview_active else ""
        )
        self.status_label.setText(f"Saved committed qpos to {path}{preview_note}")
        return path

    def _refresh_timeline_trajectory(self):
        if not self.state_timeline:
            return
        qposes = self.state_timeline.qpos_trajectory()
        # Editor timeline changes update ghost poses only. They must not replace
        # the explicit playback list: doing so made a reset keyframe appear to
        # fire again whenever the short editor timeline looped.
        self.ghost_trajectory = qposes
        self._rebuild_ghosts()

    def _update_timeline_label(self):
        count = len(self.state_timeline.states) if self.state_timeline else 0
        self.timeline_state_label.setText(
            f"3D state time: {self.current_time:.2f} s ({count} keyframes)"
        )

    def _update_root_pose_label(self):
        if not self.robot_state:
            self.root_pose_label.setText("unavailable")
            return
        free_joints = list(self.robot_model.free_joints_by_body.values())
        if not free_joints:
            self.root_pose_label.setText("fixed root")
            return
        address = free_joints[0].qpos_address
        state = self.preview_state if self.preview_active else self.committed_state
        x, y, z = state.mj_data.qpos[address:address + 3]
        suffix = " (preview)" if self.preview_active else " (committed)"
        self.root_pose_label.setText(f"{x:+.3f}, {y:+.3f}, {z:+.3f} m{suffix}")

    def set_robot_trajectory(self, qposes):
        if not self.robot_state:
            return
        valid = []
        for qpos in qposes:
            try:
                if len(qpos) == self.robot_state.mj_model.nq:
                    valid.append(qpos.copy())
            except (TypeError, AttributeError):
                continue
        self.robot_trajectory = valid
        self.ghost_trajectory = list(valid)
        self.frame_slider.setRange(0, max(0, len(valid) - 1))
        self.frame_slider.setValue(0)
        self._rebuild_ghosts()
        if valid:
            self.set_trajectory_frame(0)
            self.status_label.setText(f"Loaded {len(valid)} robot trajectory states.")

    def load_backend_states(self, states):
        if not self.robot_state:
            return
        qposes = []
        for configuration in states:
            if getattr(configuration, "qpos", None) is not None:
                qposes.append(configuration.qpos.copy())
                continue
            qpos = self.robot_state.get_qpos()
            if len(qpos) >= 7:
                qpos[:7] = [configuration.base_x, configuration.base_y, configuration.base_z,
                            configuration.base_qw, configuration.base_qx,
                            configuration.base_qy, configuration.base_qz]
            for name, value in zip(configuration.joint_names, configuration.joint_positions):
                joint = self.robot_model.joints.get(name)
                if joint:
                    qpos[joint.qpos_address] = value
            qposes.append(qpos)
        self.set_robot_trajectory(qposes)

    def generate_demo_trajectory(self):
        start = self.robot_state.get_qpos()
        target = start.copy()
        names = self.robot_state.get_joint_names()
        preferred = "left_shoulder_pitch_joint"
        name = preferred if preferred in names else names[0]
        joint = self.robot_model.joints[name]
        lo, hi = joint.limits or (-1.0, 1.0)
        target[joint.qpos_address] = max(lo, min(hi, start[joint.qpos_address] + 0.35))
        self.set_robot_trajectory(interpolate_qpos(start, target, 60))

    def _rebuild_ghosts(self):
        if self.ghost_renderer:
            self.ghost_renderer.update(
                self.ghost_trajectory, self.ghost_stride.value()
            )
        self._update_ghost_options()

    def _update_ghost_options(self):
        self.canvas.set_ghost_options(self.show_ghosts.isChecked(), self.ghost_alpha.value())

    def set_trajectory_frame(self, index):
        if not self.robot_trajectory:
            return
        index = max(0, min(len(self.robot_trajectory) - 1, int(index)))
        self.committed_state.set_qpos(self.robot_trajectory[index])
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_visible(False)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause_playback()
        elif self.robot_trajectory:
            self.play_timer.start()
            self.play_button.setText("Pause")

    def pause_playback(self):
        self.play_timer.stop()
        self.play_button.setText("Play")

    def _advance_frame(self):
        if not self.robot_trajectory:
            self.toggle_playback()
            return
        next_frame = (self.frame_slider.value() + 1) % len(self.robot_trajectory)
        self.frame_slider.setValue(next_frame)

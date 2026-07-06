"""Compact weighted-IK settings and per-joint influence panel."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.joint_controls import IKInfluenceControl


class IKControlPanel(QScrollArea):
    influence_changed = Signal(str, float)
    preset_requested = Signal()

    def __init__(self, robot_model):
        super().__init__()
        self.robot_model = robot_model
        self.task_controls = {}
        self.influence_controls = {}
        self._build_ui()

    def _build_ui(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        solver_group = QGroupBox("Solver settings")
        solver_layout = QFormLayout(solver_group)
        self.damping = QDoubleSpinBox()
        self.damping.setRange(0.001, 1.0)
        self.damping.setDecimals(4)
        self.damping.setValue(0.04)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 300)
        self.max_iterations.setValue(80)
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.01, 1.0)
        self.step_size.setValue(0.7)
        self.max_step = QDoubleSpinBox()
        self.max_step.setRange(0.001, 0.5)
        self.max_step.setDecimals(3)
        self.max_step.setValue(0.08)
        self.position_tolerance = QDoubleSpinBox()
        self.position_tolerance.setRange(0.0001, 0.1)
        self.position_tolerance.setDecimals(4)
        self.position_tolerance.setValue(0.005)
        self.orientation_tolerance = QDoubleSpinBox()
        self.orientation_tolerance.setRange(0.001, 1.0)
        self.orientation_tolerance.setDecimals(3)
        self.orientation_tolerance.setValue(0.03)
        solver_layout.addRow("Damping", self.damping)
        solver_layout.addRow("Max iterations", self.max_iterations)
        solver_layout.addRow("Step size", self.step_size)
        solver_layout.addRow("Max joint step", self.max_step)
        solver_layout.addRow("Position tolerance", self.position_tolerance)
        solver_layout.addRow("Orientation tolerance", self.orientation_tolerance)
        layout.addWidget(solver_group)

        task_group = QGroupBox("Weighted tasks (priority metadata; weighted v1)")
        task_layout = QFormLayout(task_group)
        defaults = {
            "tcp_position": (True, 1.0),
            "tcp_orientation": (
                self.robot_model.model_type != "quadruped", 0.25
            ),
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
            self.task_controls[key] = (checkbox, spin)
            task_layout.addRow(labels[key], row)
        layout.addWidget(task_group)

        influence_group = QGroupBox("Per-joint IK influence (0=locked, 1=normal)")
        influence_layout = QVBoxLayout(influence_group)
        preset_row = QHBoxLayout()
        self.preset_box = QComboBox()
        presets = [
            "All joints normal", "Root locked", "Selected limb only", "Feet planted"
        ]
        if self.robot_model.model_type == "humanoid":
            presets.extend(("Upper body only", "Legs only"))
        elif self.robot_model.model_type == "quadruped":
            presets.append("Quadruped legs only")
        self.preset_box.addItems(presets)
        apply_preset = QPushButton("Apply")
        apply_preset.clicked.connect(self.preset_requested.emit)
        preset_row.addWidget(self.preset_box, 1)
        preset_row.addWidget(apply_preset)
        influence_layout.addLayout(preset_row)
        for name in self.robot_model.get_joint_names():
            control = IKInfluenceControl(name, 1.0)
            control.value_changed.connect(self.influence_changed.emit)
            self.influence_controls[name] = control
            influence_layout.addWidget(control)
        influence_layout.addStretch()
        layout.addWidget(influence_group)
        layout.addStretch()
        self.setWidgetResizable(True)
        self.setWidget(content)

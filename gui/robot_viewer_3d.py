"""MoveIt-style live robot editor wrapped around GhostGUI's OpenGL canvas."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from application.paths import (
    QPOS_CSV_DIR,
    TRAJECTORY_CSV_DIR,
)
from application.background_jobs import SerializedBackgroundJobs
from application.playback import PlaybackClock
from application.trajectory_export_formats import (
    export_dsms_trajectory,
    export_mjlab_trajectory,
    mjlab_compatibility_error,
    resample_trajectory_export,
)
from application.csv_io import (
    TrajectoryExport,
    read_qpos_csv,
    read_trajectory_csv,
    write_qpos_csv,
    write_trajectory_csv,
)
from core.models import (
    RobotStateTimeline,
    TrajectoryGhostRenderer,
    interpolate_qpos,
)
from core.ik import (
    adaptive_trajectory_collision_reports,
    CollisionAwareIKSolver,
    CollisionChecker,
    project_qpos_above_flat_ground,
    format_collision_diagnostics,
    format_collision_pairs,
)
from core.robotics import validate_trajectory_arrays
from core.trajectory import quat_to_rpy, rpy_to_quat
from gui.file_selection import SynchronousFileSelectionStage
from gui.viewers import ik_panels
from gui.viewers.robot_canvas_3d import RobotCanvas3D
from core.ik import (
    FootLockTask,
    JointRegularizationTask,
    PostureTask,
    RootPoseTask,
)
from .widgets.compact import compact_combo as _compact_combo
from .widgets.compact import compact_spinbox as _compact_spinbox
from .widgets.joint_controls import JointControl
from .widgets.status import StatusValueLabel
from .widgets.timeline import TimesliceSlider


FRAME_BINDINGS = {
    "pelvis": ("body", "robot/pelvis"),
    "torso": ("body", "robot/torso_link"),
    "left_foot": ("site", "robot/left_foot"),
    "right_foot": ("site", "robot/right_foot"),
    "left_hand": ("site", "robot/left_palm"),
    "right_hand": ("site", "robot/right_palm"),
}
REVERSE_BINDINGS = {value: key for key, value in FRAME_BINDINGS.items()}


@dataclass(frozen=True)
class PreviewPathValidation:
    ok: bool
    message: str
    failed_index: int | None = None
    collision_indices: tuple[int, ...] = ()
    blocking_collision_indices: tuple[int, ...] = ()
    collisions: tuple = ()


class RobotViewer3D(QWidget):
    """Live FK/IK/trajectory viewer. It preserves the old canvas contract."""

    target_dragged = Signal(float, float)
    target_pose_dragged = Signal(float, float, float, float, float, float)
    target_pose_drag_finished = Signal(float, float, float, float, float, float)
    target_frame_changed = Signal(str)
    preview_cancelled = Signal()
    trajectory_csv_loaded = Signal(str)
    generate_requested = Signal()
    clear_trajectory_requested = Signal()
    timeslice_time_changed = Signal(float)
    timeslice_preview_time_changed = Signal(float)
    timeline_duration_changed = Signal(float)
    accept_timeslice_requested = Signal()
    delete_timeslice_requested = Signal()
    history_action_finished = Signal(str)
    playback_state_changed = Signal(bool)
    safe_motion_accepted = Signal(object, object, str, object)

    @property
    def _playback_last_tick(self):
        """Compatibility view of the extracted playback clock."""
        return self.playback_clock.last_tick

    @_playback_last_tick.setter
    def _playback_last_tick(self, value):
        self.playback_clock.last_tick = value

    def __init__(
        self,
        robot_model=None,
        error=None,
        background_jobs=None,
        file_selection_stage=None,
    ):
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
        self.playback_state = robot_model.create_state() if robot_model else None
        self.preview_active = False
        self.current_time = 0.0
        self.display_time = 0.0
        self.timeline_duration = 5.0
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
        self.robot_trajectory_times = []
        self.robot_trajectory_warning_report = None
        self.robot_trajectory_blocking_report = None
        self._last_export_collision_warning = None
        self.quarantined_motion_qposes = ()
        self.quarantined_motion_times = ()
        self.quarantined_motion_report = None
        self.quarantined_motion_source = None
        self.quarantined_motion_source_path = None
        self.safe_motion_candidate_qposes = ()
        self.safe_motion_candidate_times = ()
        self._prompt_trajectory_import_dt_on_load = False
        self._background_trajectory_postprocess_requested = False
        self.csv_file_selection_stage = (
            file_selection_stage
            if file_selection_stage is not None
            else SynchronousFileSelectionStage(self)
        )
        self._owns_file_selection_stage = file_selection_stage is None
        self.background_jobs = (
            background_jobs
            if background_jobs is not None
            else SerializedBackgroundJobs(self)
        )
        self._owns_background_jobs = background_jobs is None
        self._shutdown = False
        self.ghost_trajectory = []
        self.ghost_collision_flags = []
        self.ghost_source = None
        self.joint_controls = {}
        self.ik_influence_controls = {}
        self.ik_joint_weights = (
            robot_model.default_ik_joint_weights()
            if robot_model is not None
            and hasattr(robot_model, "default_ik_joint_weights")
            else {
                name: 1.0 for name in (
                    robot_model.get_joint_names() if robot_model else []
                )
            }
        )
        self.active_ik_weight_preset = "All joints normal"
        self.preview_reference_qpos = None
        self.foot_lock_targets = {}
        self.root_lock_target = None
        self._last_ik_status = None
        self._syncing_target = False
        self.playback_clock = PlaybackClock()
        self._resume_playback_after_scrub = False
        self._pending_scrub_preview_time = None
        self.canvas = RobotCanvas3D()
        self.canvas.geometry_progress.connect(self._on_geometry_progress)
        self.canvas.rendering_failed.connect(self._on_rendering_failed)
        self.canvas.target_dragged.connect(self.target_dragged.emit)
        self.canvas.target_transform_dragged.connect(self._on_transform_moved)
        self.canvas.transform_drag_finished.connect(
            self._on_transform_drag_finished
        )
        self.canvas.transform_drag_cancel_requested.connect(
            self._on_transform_cancel_requested
        )
        self.canvas.gizmo_mode_changed.connect(self._on_gizmo_mode_changed)
        self.canvas.body_double_clicked.connect(self._on_body_double_clicked)
        self.last_valid_target_position = None
        self.last_valid_target_quaternion = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._advance_playback)
        self.scrub_preview_timer = QTimer(self)
        self.scrub_preview_timer.setSingleShot(True)
        self.scrub_preview_timer.setInterval(16)
        self.scrub_preview_timer.timeout.connect(
            self._flush_pending_scrub_preview
        )
        self._build_ui(error)
        if self.robot_state:
            self.canvas.set_robot_states(
                self.committed_state, self.preview_state, self.ghost_renderer
            )
            self._set_target_to_selected_pose()

    def shutdown(self):
        """Stop timers, selectors, jobs, and GL resources idempotently."""
        if self._shutdown:
            return
        self._shutdown = True
        self.play_timer.stop()
        self.scrub_preview_timer.stop()
        self.playback_clock.stop()
        self._pending_scrub_preview_time = None
        if self._owns_file_selection_stage:
            self.csv_file_selection_stage.cancel()
        if self._owns_background_jobs:
            self.background_jobs.shutdown()
        self.canvas.shutdown()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def _build_ui(self, error):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar_controls = QWidget()

        self.robot_context_panel = QWidget()
        model_layout = QVBoxLayout(self.robot_context_panel)
        model_layout.setContentsMargins(6, 6, 6, 6)
        self.status_label = StatusValueLabel(error or "Robot model loaded; FK ready.")
        self.status_label.setWordWrap(True)
        self.timeline_state_label = StatusValueLabel("3D state time: 0.00 s")

        self.model_colors_box = QCheckBox("Use model colors")
        self.model_colors_box.setChecked(True)
        self.model_colors_box.toggled.connect(self.canvas.set_use_model_colors)
        display_panel = QWidget()
        self.display_layout = QVBoxLayout(display_panel)
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_layout.addWidget(self.model_colors_box)
        self.show_ghosts = QCheckBox("Show playback poses")
        self.show_ghosts.toggled.connect(self._sync_playback_pose_ghosts)
        self.display_layout.addWidget(self.show_ghosts)
        self.display_context_panel = display_panel
        if self.robot_model:
            texture_warnings = self.robot_model.get_visual_texture_warnings()
            if texture_warnings:
                warning = QLabel("; ".join(texture_warnings))
                warning.setWordWrap(True)
                self.display_layout.addWidget(warning)

        self.trajectory_context_panel = QWidget()
        trajectory_context_layout = QVBoxLayout(self.trajectory_context_panel)
        trajectory_context_layout.setContentsMargins(6, 6, 6, 6)
        self.load_qpos_button = QPushButton("Load")
        self.load_trajectory_button = QPushButton("Load")
        self.save_qpos_button = QPushButton("Save")
        self.save_trajectory_button = QPushButton("Save")
        for button in (
            self.load_qpos_button,
            self.load_trajectory_button,
            self.save_qpos_button,
            self.save_trajectory_button,
        ):
            button.setMinimumWidth(0)
            button.setMaximumWidth(104)
        self.trajectory_import_dt = QDoubleSpinBox()
        self.trajectory_import_dt.setDecimals(2)
        self.trajectory_import_dt.setRange(0.01, 10.0)
        self.trajectory_import_dt.setSingleStep(0.01)
        self.trajectory_import_dt.setValue(0.10)
        self.trajectory_import_dt.setSuffix(" s")
        self.trajectory_import_dt.setMaximumWidth(76)
        self.trajectory_import_dt.setVisible(False)
        self.load_qpos_button.clicked.connect(self.choose_qpos_csv)
        self.load_trajectory_button.clicked.connect(self.choose_trajectory_csv)
        self.save_qpos_button.clicked.connect(self.choose_qpos_save_path)
        self.save_trajectory_button.clicked.connect(
            self.choose_trajectory_save_path
        )
        self.trajectory_csv_group = QGroupBox("Trajectory CSV")
        trajectory_csv_layout = QHBoxLayout(self.trajectory_csv_group)
        trajectory_csv_layout.setContentsMargins(6, 6, 6, 6)
        trajectory_csv_layout.setSpacing(4)
        trajectory_csv_layout.addWidget(self.load_trajectory_button)
        trajectory_csv_layout.addWidget(self.save_trajectory_button)
        self.trajectory_csv_group.setVisible(False)
        self.qpos_csv_group = QGroupBox("Qpos CSV")
        qpos_csv_layout = QHBoxLayout(self.qpos_csv_group)
        qpos_csv_layout.setContentsMargins(6, 6, 6, 6)
        qpos_csv_layout.setSpacing(4)
        qpos_csv_layout.addWidget(self.load_qpos_button)
        qpos_csv_layout.addWidget(self.save_qpos_button)
        self.qpos_csv_group.setVisible(False)

        self.target_context_panel = QWidget()
        target_layout = QVBoxLayout(self.target_context_panel)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(4)
        # A late-populated QStackedWidget can otherwise retain the empty
        # page's zero-height size under QMacStyle. Propagate the native label
        # and combo minimum sizes to the page and its stack.
        target_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.advanced_target_label = QLabel("Advanced target")
        self.target_box = QComboBox()
        _compact_combo(self.target_box, minimum_chars=12)
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
        target_layout.addWidget(self.advanced_target_label)
        target_layout.addWidget(self.target_box)
        self.root_pose_label = StatusValueLabel()
        self.root_pose_label.setWordWrap(True)

        self.preview_ik_context_panel = QWidget()
        preview_ik_layout = QVBoxLayout(self.preview_ik_context_panel)
        preview_ik_layout.setContentsMargins(4, 4, 4, 4)
        preview_ik_layout.setSpacing(4)

        self.timeslice_context_panel = QWidget()
        self.timeslice_context_layout = QFormLayout(self.timeslice_context_panel)
        self.timeslice_context_layout.setContentsMargins(6, 6, 6, 6)
        self.timeslice_context_layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.collision_substeps = QSpinBox()
        _compact_spinbox(self.collision_substeps)
        self.collision_substeps.setRange(1, 32)
        self.collision_substeps.setValue(8)
        self.collision_substeps.valueChanged.connect(
            self._set_collision_substeps
        )
        self.collision_substeps_label = QLabel("Collision substeps")
        self.timeslice_context_layout.addRow(
            self.collision_substeps_label, self.collision_substeps
        )
        self.playback_speed = QDoubleSpinBox()
        _compact_spinbox(self.playback_speed)
        self.playback_speed.setRange(0.10, 4.00)
        self.playback_speed.setDecimals(2)
        self.playback_speed.setSingleStep(0.25)
        self.playback_speed.setValue(1.00)
        self.playback_speed.setSuffix("×")
        self.playback_speed_label = QLabel("Playback speed")
        self.timeslice_context_layout.addRow(
            self.playback_speed_label, self.playback_speed
        )
        self.generate_button = QPushButton("Demo trajectory")
        self.generate_button.clicked.connect(self.generate_demo_trajectory)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.ghost_stride = QSpinBox()
        _compact_spinbox(self.ghost_stride)
        self.ghost_stride.setRange(1, 100)
        self.ghost_stride.setValue(8)
        self.ghost_stride.valueChanged.connect(self._rebuild_ghosts)
        self.ghost_alpha = QDoubleSpinBox()
        _compact_spinbox(self.ghost_alpha)
        self.ghost_alpha.setRange(0.02, 0.8)
        self.ghost_alpha.setSingleStep(0.05)
        self.ghost_alpha.setValue(0.16)
        self.ghost_alpha.valueChanged.connect(self._update_ghost_options)
        self.preview_alpha = QDoubleSpinBox()
        _compact_spinbox(self.preview_alpha)
        self.preview_alpha.setRange(0.1, 1.0)
        self.preview_alpha.setSingleStep(0.05)
        self.preview_alpha.setValue(0.65)
        self.preview_alpha.valueChanged.connect(self.canvas.set_preview_alpha)
        self.ghost_stride_label = QLabel("Playback spacing")
        self.ghost_alpha_label = QLabel("Playback opacity")
        self.preview_alpha_label = QLabel("Preview opacity")
        self.timeslice_context_layout.addRow(
            self.ghost_stride_label, self.ghost_stride
        )
        self.timeslice_context_layout.addRow(
            self.ghost_alpha_label, self.ghost_alpha
        )
        self.timeslice_context_layout.addRow(
            self.preview_alpha_label, self.preview_alpha
        )

        editor_tabs = QTabWidget()
        editor_tabs.setObjectName("ikEditorTabs")
        editor_tabs.setMinimumWidth(0)
        self.ik_editor_tabs = editor_tabs
        joint_group = QWidget()
        joint_group.setObjectName("ikEditorTabContent")
        joint_layout = QVBoxLayout(joint_group)
        joint_layout.setContentsMargins(6, 6, 6, 6)
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
        scroll.setObjectName("jointEditorScroll")
        scroll.viewport().setObjectName("ikEditorViewport")
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(0)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(joint_group)
        self.joint_angles_page = scroll
        editor_tabs.addTab(self._build_ik_tasks_widget(), "Tasks")
        editor_tabs.addTab(self._build_joint_weights_widget(), "Weights")
        editor_tabs.addTab(self._build_solver_widget(), "Solver")
        editor_tabs.setCurrentIndex(0)
        preview_ik_layout.addWidget(editor_tabs)
        root.addWidget(self._build_canvas_workspace(), stretch=1)
        root.addWidget(self._build_timeslice_editor())

        enabled = self.robot_state is not None
        self.load_qpos_button.setEnabled(enabled)
        self.load_trajectory_button.setEnabled(enabled)
        self.trajectory_import_dt.setEnabled(enabled)
        self.save_qpos_button.setEnabled(enabled)
        self.save_trajectory_button.setEnabled(enabled)
        self.target_context_panel.setEnabled(enabled)
        self.timeslice_context_panel.setEnabled(enabled)
        self.preview_ik_context_panel.setEnabled(enabled)
        self.timeslice_editor.setEnabled(enabled)

    def _build_canvas_workspace(self):
        self.canvas_workspace = QWidget()
        layout = QGridLayout(self.canvas_workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 0, 0)
        return self.canvas_workspace

    def _build_timeslice_editor(self):
        self.timeslice_editor = QWidget()
        self.timeslice_layout = QVBoxLayout(self.timeslice_editor)
        self.timeslice_layout.setContentsMargins(8, 4, 8, 4)
        self.timeslice_layout.setSpacing(3)

        self.timeslice_label = QLabel("Time")

        self.timeslice_slider = TimesliceSlider(Qt.Orientation.Horizontal)
        self.timeslice_slider.setRange(0, 500)
        self.timeslice_slider.setValue(0)
        self.timeslice_slider.marker_activated.connect(
            self._emit_timeslice_marker_time
        )
        self.timeslice_slider.time_activated.connect(
            self._emit_timeslice_marker_time
        )
        self.timeslice_slider.sliderMoved.connect(
            self._preview_timeslice_slider_value
        )
        self.timeslice_slider.sliderPressed.connect(
            self._begin_timeslice_scrub
        )
        self.timeslice_slider.sliderReleased.connect(
            self._emit_timeslice_slider_time
        )

        self.timeslice_time_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_time_input, width=72)
        self.timeslice_time_input.setRange(0.0, self.timeline_duration)
        self.timeslice_time_input.setDecimals(2)
        self.timeslice_time_input.setSingleStep(0.01)
        self.timeslice_time_input.setSuffix(" s")
        self.timeslice_time_input.editingFinished.connect(
            self._emit_timeslice_input_time
        )
        self.timeslice_frame_readout = QLabel("Frame —")
        self.timeslice_frame_readout.setMinimumWidth(82)
        self.timeslice_frame_readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.timeslice_frame_readout.setAccessibleName("Playback frame")

        self.delete_timeslice_button = QPushButton("Delete Keyframe")
        self.delete_timeslice_button.clicked.connect(self.delete_timeslice)
        self.jump_to_unsafe_button = QPushButton("Jump to unsafe time")
        self.jump_to_unsafe_button.clicked.connect(
            self.jump_to_motion_safety_issue
        )
        self.try_safe_reroute_button = QPushButton("Try Safe Reroute")
        self.try_safe_reroute_button.clicked.connect(self.try_safe_reroute)
        self.accept_safe_motion_button = QPushButton("Accept Safe Motion")
        self.accept_safe_motion_button.clicked.connect(self.accept_safe_motion)
        self.revert_safe_motion_button = QPushButton("Revert")
        self.revert_safe_motion_button.clicked.connect(self.revert_safe_motion)
        for button in (
            self.jump_to_unsafe_button,
            self.try_safe_reroute_button,
            self.accept_safe_motion_button,
            self.revert_safe_motion_button,
        ):
            button.setVisible(False)
        self.timeslice_step_label = QLabel("Keyframe interval")
        self.timeslice_step_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_step_input, width=72)
        self.timeslice_step_input.setRange(0.01, 5.0)
        self.timeslice_step_input.setDecimals(2)
        self.timeslice_step_input.setSingleStep(0.01)
        self.timeslice_step_input.setValue(0.10)
        self.timeslice_step_input.setSuffix(" s")

        self.timeslice_duration_label = QLabel("Max time")
        self.timeslice_duration_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_duration_input, width=72)
        self.timeslice_duration_input.setRange(0.10, 120.0)
        self.timeslice_duration_input.setDecimals(2)
        self.timeslice_duration_input.setSingleStep(0.10)
        self.timeslice_duration_input.setValue(self.timeline_duration)
        self.timeslice_duration_input.setSuffix(" s")
        self.timeslice_duration_input.valueChanged.connect(
            self._on_timeslice_duration_changed
        )

        self.export_dt_label = QLabel("Export interval")
        self.export_dt_input = QDoubleSpinBox()
        self.export_dt_input.setObjectName("exportIntervalSpinBox")
        _compact_spinbox(self.export_dt_input, width=72)
        self.export_dt_input.setRange(0.01, 10.0)
        self.export_dt_input.setDecimals(2)
        self.export_dt_input.setSingleStep(0.01)
        self.export_dt_input.setValue(0.01)
        self.export_dt_input.setSuffix(" s")
        self.export_dt_input.setToolTip(
            "Choose the uniform time interval used by Generate and trajectory "
            "export."
        )
        self.dsms_motion_speed_label = QLabel("DSMS motion speed")
        self.dsms_motion_speed_input = QDoubleSpinBox()
        self.dsms_motion_speed_input.setObjectName("dsmsMotionSpeedSpinBox")
        _compact_spinbox(self.dsms_motion_speed_input, width=72)
        self.dsms_motion_speed_input.setRange(0.10, 4.00)
        self.dsms_motion_speed_input.setDecimals(2)
        self.dsms_motion_speed_input.setSingleStep(0.25)
        self.dsms_motion_speed_input.setValue(1.00)
        self.dsms_motion_speed_input.setSuffix("×")
        self.dsms_motion_speed_input.setToolTip(
            "Scale actual DSMS reference timing without changing qpos. "
            "For example, 0.5× doubles the exported duration. This does not "
            "change visual playback speed or other trajectory formats."
        )

        self.timeslice_time_row = QHBoxLayout()
        self.timeslice_time_row.setContentsMargins(0, 0, 0, 0)
        self.timeslice_time_row.setSpacing(8)
        self.timeslice_time_row.addWidget(self.timeslice_label)
        self.timeslice_time_row.addWidget(self.timeslice_slider, stretch=1)
        self.timeslice_time_row.addWidget(self.timeslice_time_input)
        self.timeslice_time_row.addWidget(self.timeslice_frame_readout)

        self.timeslice_context_layout.addRow(
            self.timeslice_step_label, self.timeslice_step_input
        )
        self.timeslice_context_layout.addRow(
            self.timeslice_duration_label, self.timeslice_duration_input
        )
        self.timeslice_context_layout.addRow(
            self.export_dt_label, self.export_dt_input
        )
        self.timeslice_context_layout.addRow(
            self.dsms_motion_speed_label, self.dsms_motion_speed_input
        )

        self.timeslice_timeline_group = QGroupBox("Timeline")
        self.timeslice_timeline_layout = QHBoxLayout(self.timeslice_timeline_group)
        self.timeslice_timeline_layout.setContentsMargins(6, 6, 6, 6)
        self.timeslice_timeline_layout.setSpacing(8)
        self.timeslice_scrubber_layout = QVBoxLayout()
        self.timeslice_scrubber_layout.setContentsMargins(0, 0, 0, 0)
        self.timeslice_scrubber_layout.setSpacing(3)
        self.timeslice_scrubber_layout.addLayout(self.timeslice_time_row)
        self.motion_safety_action_row = QHBoxLayout()
        self.motion_safety_action_row.setContentsMargins(0, 0, 0, 0)
        self.motion_safety_action_row.setSpacing(4)
        self.motion_safety_action_row.addWidget(self.jump_to_unsafe_button)
        self.motion_safety_action_row.addWidget(self.try_safe_reroute_button)
        self.motion_safety_action_row.addWidget(self.accept_safe_motion_button)
        self.motion_safety_action_row.addWidget(self.revert_safe_motion_button)
        self.timeslice_action_row = QVBoxLayout()
        self.timeslice_action_row.setContentsMargins(0, 0, 0, 0)
        self.timeslice_action_row.setSpacing(4)
        self.timeslice_action_row.addWidget(self.delete_timeslice_button)
        self.timeslice_timeline_layout.addLayout(
            self.timeslice_scrubber_layout, stretch=1
        )
        self.timeslice_timeline_layout.addLayout(self.timeslice_action_row)
        self.timeslice_timeline_layout.addLayout(
            self.motion_safety_action_row
        )

        self.timeslice_layout.addWidget(self.timeslice_timeline_group)
        return self.timeslice_editor

    def set_defined_timeslices(self, times):
        self.timeslice_slider.set_defined_times(times)

    def _motion_report_text(self, report):
        if report is None:
            return "unknown motion location"
        location = getattr(report, "location_label", None)
        if not location:
            time = getattr(report, "time", None)
            if time is not None:
                location = f"t={float(time):.3f} s"
            else:
                location = f"sample {getattr(report, 'sample_index', '?')}"
        collisions = tuple(getattr(report, "collisions", ()) or ())
        if not collisions:
            return str(location)
        return (
            f"{location}: {format_collision_pairs(collisions)}; "
            f"Contact geometry: {format_collision_diagnostics(collisions)}"
        )

    def _set_motion_safety_controls(self):
        report = self.quarantined_motion_report
        has_quarantine = bool(self.quarantined_motion_qposes)
        has_candidate = bool(self.safe_motion_candidate_qposes)
        self.jump_to_unsafe_button.setVisible(report is not None)
        self.try_safe_reroute_button.setVisible(has_quarantine)
        self.accept_safe_motion_button.setVisible(has_candidate)
        self.revert_safe_motion_button.setVisible(
            has_quarantine or has_candidate
        )
        self.timeslice_slider.set_safety_reports(
            (report,) if report is not None else ()
        )

    def _clear_motion_safety_review(self):
        self.quarantined_motion_qposes = ()
        self.quarantined_motion_times = ()
        self.quarantined_motion_report = None
        self.quarantined_motion_source = None
        self.quarantined_motion_source_path = None
        self.safe_motion_candidate_qposes = ()
        self.safe_motion_candidate_times = ()
        self._clear_ghost_overlay(source="safety_candidate")
        self._set_motion_safety_controls()

    def quarantine_motion(
        self,
        qposes,
        times,
        *,
        report=None,
        source="motion",
        source_path=None,
    ):
        """Keep unsafe motion inspectable without publishing it to playback."""
        if not self.robot_state:
            return False
        try:
            normalized_times, normalized_qposes = validate_trajectory_arrays(
                times, qposes, int(self.robot_model.mj_model.nq)
            )
        except ValueError as error:
            self.status_label.setText(
                f"Could not quarantine invalid {source}: {error}"
            )
            return False
        if not normalized_qposes:
            return False

        self.pause_playback()
        self.quarantined_motion_qposes = tuple(
            qpos.copy() for qpos in normalized_qposes
        )
        self.quarantined_motion_times = tuple(normalized_times)
        self.quarantined_motion_report = report
        self.quarantined_motion_source = str(source)
        self.quarantined_motion_source_path = (
            None if source_path is None else str(source_path)
        )
        self.safe_motion_candidate_qposes = ()
        self.safe_motion_candidate_times = ()
        self.ghost_trajectory = [
            qpos.copy() for qpos in self.quarantined_motion_qposes
        ]
        self.ghost_collision_flags = [
            False for _qpos in self.quarantined_motion_qposes
        ]
        segment_index = getattr(report, "segment_index", None)
        segment_fraction = (
            float(getattr(report, "segment_fraction", 0.0))
            if report is not None else 0.0
        )
        if (
            segment_index is not None
            and 0.0 < segment_fraction < 1.0
            and 0 <= int(segment_index) < len(self.ghost_trajectory) - 1
        ):
            segment_index = int(segment_index)
            collision_qpos = self.state_timeline._interpolate(
                self.quarantined_motion_qposes[segment_index],
                self.quarantined_motion_qposes[segment_index + 1],
                segment_fraction,
            )
            self.ghost_trajectory.insert(segment_index + 1, collision_qpos)
            self.ghost_collision_flags.insert(segment_index + 1, True)
        elif report is not None:
            sample_index = max(
                0,
                min(
                    int(getattr(report, "sample_index", 0)),
                    len(self.ghost_collision_flags) - 1,
                ),
            )
            self.ghost_collision_flags[sample_index] = True
        else:
            self.ghost_collision_flags = [
                True for _qpos in self.ghost_trajectory
            ]
        self.ghost_source = "safety_candidate"
        self._rebuild_ghosts()
        self._set_motion_safety_controls()
        self.status_label.setText(
            f"Quarantined unsafe {source}; playback and export still use the "
            f"last accepted safe motion. First issue: "
            f"{self._motion_report_text(report)}. Use Jump to unsafe time, "
            "Try Safe Reroute, or Revert."
        )
        return True

    def jump_to_motion_safety_issue(self):
        report = self.quarantined_motion_report
        if report is None:
            return False
        time = getattr(report, "time", None)
        if time is None:
            index = int(getattr(report, "sample_index", 0))
            if not self.quarantined_motion_times:
                return False
            index = max(0, min(index, len(self.quarantined_motion_times) - 1))
            time = self.quarantined_motion_times[index]
        time = float(time)
        qposes = self.quarantined_motion_qposes
        times = self.quarantined_motion_times
        if time <= times[0]:
            qpos = qposes[0]
        elif time >= times[-1]:
            qpos = qposes[-1]
        else:
            upper = bisect_right(times, time)
            lower = upper - 1
            duration = times[upper] - times[lower]
            fraction = (
                0.0 if duration <= 0.0
                else (time - times[lower]) / duration
            )
            qpos = self.state_timeline._interpolate(
                qposes[lower], qposes[upper], fraction
            )
        self.display_time = max(
            0.0, min(time, self.timeline_duration)
        )
        self._set_timeslice_widgets(self.display_time)
        self.playback_state.set_qpos(qpos)
        self.canvas.set_robot_states(
            self.playback_state, self.preview_state, self.ghost_renderer
        )
        self.canvas.set_preview_visible(False)
        self.canvas.update()
        self.timeslice_preview_time_changed.emit(self.display_time)
        self.status_label.setText(
            f"Unsafe motion location: {self._motion_report_text(report)}"
        )
        return True

    def try_safe_reroute(self):
        if not self.quarantined_motion_qposes:
            self.status_label.setText("No quarantined motion is available to reroute.")
            return False
        from application.motion_safety import propose_safe_motion_repair

        result = propose_safe_motion_repair(
            self.robot_model,
            self.quarantined_motion_qposes,
            self.quarantined_motion_times,
            checker=self.collision_checker,
        )
        if not result.success:
            if result.blocking_report is not None:
                self.quarantined_motion_report = result.blocking_report
            self._set_motion_safety_controls()
            self.status_label.setText(
                f"Safe reroute was not found: {result.status}. "
                "The quarantined motion remains unchanged."
            )
            return False

        self.safe_motion_candidate_qposes = tuple(
            qpos.copy() for qpos in result.qposes
        )
        self.safe_motion_candidate_times = tuple(result.times)
        self.ghost_trajectory = [
            qpos.copy() for qpos in self.safe_motion_candidate_qposes
        ]
        self.ghost_collision_flags = [
            False for _qpos in self.safe_motion_candidate_qposes
        ]
        self.ghost_source = "safety_candidate"
        self._rebuild_ghosts()
        self._set_motion_safety_controls()
        self.status_label.setText(
            f"{result.status} Review the proposed Preview Path, then choose "
            "Accept Safe Motion or Revert."
        )
        return True

    def accept_safe_motion(self):
        if not self.safe_motion_candidate_qposes:
            self.status_label.setText("No reviewed safe-motion candidate is ready.")
            return False
        qposes = tuple(qpos.copy() for qpos in self.safe_motion_candidate_qposes)
        times = tuple(self.safe_motion_candidate_times)
        source = self.quarantined_motion_source or "motion"
        source_path = self.quarantined_motion_source_path
        self._clear_motion_safety_review()
        if not self.set_robot_trajectory(qposes, times=times):
            return False

        if source == "import" and self.state_timeline and qposes:
            self.state_timeline.reset(times[0], qposes[0])
            for time, qpos in zip(times[1:], qposes[1:]):
                self.state_timeline.set_state(time, qpos)
            self.set_defined_timeslices(times)
            if source_path is not None:
                self.trajectory_csv_loaded.emit(source_path)

        self.safe_motion_accepted.emit(qposes, times, source, source_path)
        self.status_label.setText(
            f"Accepted {len(qposes)} adaptively validated safe motion states "
            f"from {source}."
        )
        self.history_action_finished.emit("Accept safe motion")
        return True

    def revert_safe_motion(self):
        if not (
            self.quarantined_motion_qposes or self.safe_motion_candidate_qposes
        ):
            return False
        self._clear_motion_safety_review()
        self._sync_playback_pose_ghosts()
        self.timeslice_slider.set_safety_reports(
            (self.robot_trajectory_warning_report,)
        )
        self.status_label.setText(
            "Discarded the quarantined/repaired candidate; the last accepted "
            "motion is unchanged."
        )
        return True

    def set_smoothing_widget(self, widget):
        if widget is None:
            return
        if (
            widget.parent() is self.timeslice_context_panel
            and getattr(self, "smoothing_widget", None) is widget
        ):
            return
        if widget.parent() is not self.timeslice_context_panel:
            widget.setParent(self.timeslice_context_panel)
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            widget.sizePolicy().verticalPolicy(),
        )
        self.smoothing_widget = widget
        self.timeslice_context_layout.insertRow(0, widget)

    def set_trajectory_display_widgets(self, keyframes_widget, lines_widget):
        widgets = [
            widget for widget in (keyframes_widget, lines_widget)
            if widget is not None
        ]
        if not widgets:
            return
        for widget in widgets:
            if widget.parent() is not self.display_context_panel:
                widget.setParent(self.display_context_panel)
        insert_index = 1
        if keyframes_widget is not None:
            self.display_layout.insertWidget(insert_index, keyframes_widget)
            insert_index += 1
        if lines_widget is not None:
            self.display_layout.insertWidget(insert_index, lines_widget)

    def set_trajectory_lines_widget(self, widget):
        self.set_trajectory_display_widgets(None, widget)

    def _set_timeslice_widgets(self, time):
        raw_time = int(round(float(time) * 100.0))
        raw_time = max(self.timeslice_slider.minimum(), raw_time)
        raw_time = min(self.timeslice_slider.maximum(), raw_time)
        was_blocked = self.timeslice_slider.blockSignals(True)
        self.timeslice_slider.setValue(raw_time)
        self.timeslice_slider.blockSignals(was_blocked)
        was_blocked = self.timeslice_time_input.blockSignals(True)
        self.timeslice_time_input.setValue(raw_time / 100.0)
        self.timeslice_time_input.blockSignals(was_blocked)

    def _preview_timeslice_slider_value(self, raw_value):
        time = raw_value / 100.0
        if not self.scrub_preview_timer.isActive():
            self.preview_trajectory_time(time, emit_time_signal=True)
            self.scrub_preview_timer.start()
            return

        self._pending_scrub_preview_time = time
        was_blocked = self.timeslice_time_input.blockSignals(True)
        self.timeslice_time_input.setValue(time)
        self.timeslice_time_input.blockSignals(was_blocked)

    def _flush_pending_scrub_preview(self):
        time = self._pending_scrub_preview_time
        self._pending_scrub_preview_time = None
        if time is None:
            return
        self.preview_trajectory_time(time, emit_time_signal=True)
        self.scrub_preview_timer.start()

    def _begin_timeslice_scrub(self):
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self._resume_playback_after_scrub = self.play_timer.isActive()
        if self._resume_playback_after_scrub:
            self.pause_playback(commit_time=False)

    def _emit_timeslice_slider_time(self):
        time = self.timeslice_slider.value() / 100.0
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self.preview_trajectory_time(time, emit_time_signal=True)
        self.timeslice_time_changed.emit(time)
        resume_playback = self._resume_playback_after_scrub
        self._resume_playback_after_scrub = False
        if resume_playback:
            self.start_playback()

    def _emit_timeslice_marker_time(self, time):
        was_playing = self.play_timer.isActive()
        if was_playing:
            self.pause_playback(commit_time=False)
        self._set_timeslice_widgets(time)
        self.timeslice_time_changed.emit(float(time))
        if was_playing:
            self.start_playback()

    def _emit_timeslice_input_time(self):
        time = self.timeslice_time_input.value()
        was_playing = self.play_timer.isActive()
        if was_playing:
            self.pause_playback(commit_time=False)
        self._set_timeslice_widgets(time)
        self.timeslice_time_changed.emit(time)
        if was_playing:
            self.start_playback()

    def timeslice_step(self):
        return float(self.timeslice_step_input.value())

    def export_dt(self):
        return float(self.export_dt_input.value())

    def set_export_dt(self, export_dt):
        self.export_dt_input.setValue(float(export_dt))

    def dsms_motion_speed(self):
        return float(self.dsms_motion_speed_input.value())

    def set_dsms_motion_speed(self, motion_speed):
        self.dsms_motion_speed_input.setValue(float(motion_speed))

    def _on_timeslice_duration_changed(self, duration):
        self.set_timeline_duration(duration)

    def set_timeline_duration(self, duration, emit_signal=True):
        duration = max(0.10, float(duration))
        self.timeline_duration = duration
        raw_max = int(round(duration * 100.0))

        was_blocked = self.timeslice_duration_input.blockSignals(True)
        self.timeslice_duration_input.setValue(duration)
        self.timeslice_duration_input.blockSignals(was_blocked)

        self.timeslice_slider.setMaximum(raw_max)
        self.timeslice_time_input.setMaximum(duration)
        self.timeslice_step_input.setMaximum(duration)
        if self.current_time > duration:
            if self.state_timeline:
                self.current_time = self.state_timeline.time_key(duration)
            else:
                self.current_time = duration
        self.display_time = min(self.display_time, duration)
        self._set_timeslice_widgets(self.display_time)
        self._update_frame_readout(self.display_time)
        if emit_signal:
            self.timeline_duration_changed.emit(duration)

    def next_timeslice_time(self, time=None):
        base_time = self.current_time if time is None else float(time)
        raw_time = int(round((base_time + self.timeslice_step()) * 100.0))
        raw_time = max(self.timeslice_slider.minimum(), raw_time)
        raw_time = min(self.timeslice_slider.maximum(), raw_time)
        return raw_time / 100.0

    def accept_timeslice(self):
        # Playback and live scrubbing intentionally move display_time without
        # creating an editable state. Commit Keyframe must first choose one authoritative
        # time so its qpos state and logical-target snapshot cannot be written
        # at two different positions on the timeline.
        slice_time = max(0.0, min(float(self.display_time), self.timeline_duration))
        if self.state_timeline is not None:
            slice_time = self.state_timeline.time_key(slice_time)

        if self.play_timer.isActive():
            self.pause_playback(commit_time=False)

        if self.preview_active:
            # Preserve the active preview while moving it to the visible time.
            # Calling set_current_time() here would discard that preview.
            self.current_time = slice_time
            self.display_time = slice_time
            self._set_timeslice_widgets(slice_time)
            self.timeslice_preview_time_changed.emit(slice_time)
            if not self.accept_preview(emit_pose_finished=False):
                return
        elif abs(slice_time - self.current_time) > 1e-9:
            # With no edit preview to preserve, load the visible trajectory pose
            # as the committed state before taking the logical-target snapshot.
            self.timeslice_time_changed.emit(slice_time)

        candidate_qpos = self.committed_state.get_qpos()
        limit_error = self._joint_limit_violation(candidate_qpos)
        if limit_error is not None:
            self.status_label.setText(
                f"Cannot Commit Keyframe: {limit_error}."
            )
            return
        projection = project_qpos_above_flat_ground(
            self.robot_model,
            candidate_qpos,
            checker=self.collision_checker,
        )
        if not projection.success:
            self.status_label.setText(
                "Cannot Commit Keyframe: safety barrier rejected the "
                f"committed pose ({projection.reason})."
            )
            return
        if projection.changed:
            self.set_robot_state_for_current_time(projection.qpos)
            self.update_current_keyframe_from_robot_state(
                refresh_ghosts=True
            )
            candidate_qpos = projection.qpos

        (
            interval_warning,
            interval_blocking,
            candidate_times,
            candidate_qposes,
        ) = self._validate_candidate_keyframe_intervals(
            candidate_qpos, slice_time
        )
        if interval_blocking is not None:
            self.quarantine_motion(
                candidate_qposes,
                candidate_times,
                report=interval_blocking,
                source="Keyframe interval",
            )
            self.status_label.setText(
                "Cannot Commit Keyframe: adaptive between-Keyframe validation "
                f"found {self._motion_report_text(interval_blocking)}."
            )
            return
        self.timeslice_slider.set_safety_reports((interval_warning,))
        self.accept_timeslice_requested.emit()

    def delete_timeslice(self):
        self.delete_timeslice_requested.emit()

    def sidebar_context_widget(self):
        return self.sidebar_controls

    def robot_context_widget(self):
        return self.robot_context_panel

    def target_context_widget(self):
        return self.target_context_panel

    def trajectory_context_widget(self):
        return None

    def consume_trajectory_import_dt_prompt_request(self):
        requested = bool(
            getattr(self, "_prompt_trajectory_import_dt_on_load", False)
        )
        self._prompt_trajectory_import_dt_on_load = False
        return requested

    def consume_background_trajectory_postprocess_request(self):
        requested = bool(self._background_trajectory_postprocess_requested)
        self._background_trajectory_postprocess_requested = False
        return requested

    def timeslice_context_widget(self):
        return self.timeslice_context_panel

    def display_context_widget(self):
        return self.display_context_panel

    def preview_ik_context_widget(self):
        return self.preview_ik_context_panel

    def joint_editor_widget(self):
        return self.joint_angles_page

    def _make_ik_scroll_area(self, content):
        return ik_panels.make_ik_scroll_area(content)

    def _build_solver_widget(self):
        return ik_panels.build_solver_widget(self)

    def _build_ik_tasks_widget(self):
        return ik_panels.build_ik_tasks_widget(self)

    def _build_joint_weights_widget(self):
        return ik_panels.build_joint_weights_widget(self)

    def _on_geometry_progress(self, complete, total):
        if total <= 0:
            return
        if complete < total:
            self.status_label.setText(
                f"Preparing 3D geometry… {complete}/{total}"
            )
        else:
            self.status_label.setText("3D geometry ready.")

    def _on_rendering_failed(self, message):
        """Mirror persistent canvas failures into the shared Status panel."""
        self.status_label.setText(str(message))

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        show_trajectory_lines=True,
        trajectory_smoothing=0.0,
        show_keyframes=True,
    ):
        # The live gizmo owns its quaternion while editing. Ordinary status
        # refreshes must not reset an in-progress ring rotation from controls.
        self.canvas.update_scene(
            trajectory,
            None,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
            show_keyframes=show_keyframes,
        )
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
                self._set_canvas_selected_target(kind, name)
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
        self._sync_live_ik_weight_preset()
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
        self._set_canvas_selected_target(kind, name)
        self.canvas.set_target_pose(position, quaternion)
        self.last_valid_target_position = position.copy()
        self.last_valid_target_quaternion = quaternion.copy()
        self._update_root_pose_label()

    def _set_canvas_selected_target(self, kind, name):
        if not name or self.committed_state is None:
            self.canvas.set_selected_target()
            return
        try:
            resolved_kind, object_id = self.committed_state.resolve_object(name, kind)
        except KeyError:
            self.canvas.set_selected_target()
            return
        if resolved_kind == "site":
            owner_body_id = int(self.robot_model.mj_model.site_bodyid[object_id])
        else:
            owner_body_id = int(object_id)
        self.canvas.set_selected_target(resolved_kind, name, owner_body_id)

    def preview_target_pose(self, frame_name, position, quaternion):
        binding = self.frame_bindings.get(frame_name)
        if binding is None:
            self.status_label.setText(
                f"Frame {frame_name!r} has no editable 3D body/site target."
            )
            return False
        if not self.select_target(*binding, emit=False):
            self.status_label.setText(
                f"Frame {frame_name!r} is not selectable in the 3D viewer."
            )
            return False
        if self.last_valid_target_position is None:
            self._set_target_to_selected_pose()
        self._on_transform_moved(position, quaternion)
        return True

    def _joint_changed(self, name, value):
        self.begin_preview()
        self._use_editor_canvas_states()
        self.canvas.set_preview_visible(True)
        previous_qpos = self.preview_state.get_qpos()
        self.preview_state.set_joint_value(name, value)
        requested_collisions = (
            self.collision_checker.get_collisions(self.preview_state)
            if self.collision_checker else []
        )
        projection = project_qpos_above_flat_ground(
            self.robot_model,
            self.preview_state.get_qpos(),
            checker=self.collision_checker,
        )
        if not projection.success:
            self.preview_state.set_qpos(previous_qpos)
            self._set_target_to_selected_pose()
            self._update_preview_collisions(requested_collisions)
            self._sync_joint_controls()
            self.status_label.setText(
                f"Safety barrier stopped Joint Angles edit for {name}: "
                f"{projection.reason}."
            )
            return
        if projection.changed:
            self.preview_state.set_qpos(projection.qpos)
        self._set_target_to_selected_pose()
        collisions = self._update_preview_collisions()
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_dragged.emit(
                *map(float, self.last_valid_target_position),
                roll,
                pitch,
                yaw,
            )
        if projection.changed:
            self.status_label.setText(
                f"Preview FK: {name} = {value:+.3f} rad; "
                f"ground barrier auto-corrected the pose by raising the base "
                f"{projection.applied_offset * 1000.0:.1f} mm; "
                "use Commit Keyframe to save"
            )
        elif collisions:
            names = format_collision_pairs(collisions)
            details = format_collision_diagnostics(collisions)
            self.status_label.setText(
                f"Collision warning: {names}; "
                f"Contact geometry: {details}; "
                f"Preview FK: {name} = {value:+.3f} rad; "
                "adjust the pose before Commit Keyframe"
            )
        else:
            self.status_label.setText(
                f"Preview FK: {name} = {value:+.3f} rad; "
                "use Commit Keyframe to save"
            )

    def _update_preview_collisions(self, collisions=None):
        if collisions is None:
            collisions = (
                self.collision_checker.get_collisions(self.preview_state)
                if self.collision_checker and self.preview_state else []
            )
        collisions = list(collisions)
        self.canvas.set_preview_collisions(collisions)
        return collisions

    def _sync_joint_controls(self, state=None):
        if state is None:
            state = self.preview_state if self.preview_active else self.committed_state
        for name, control in self.joint_controls.items():
            control.set_value(state.get_joint_value(name))

    def _ik_influence_changed(self, name, value):
        self.ik_joint_weights[name] = float(value)
        self.active_ik_weight_preset = "Custom"
        if hasattr(self, "ik_preset_box"):
            self.ik_preset_box.setCurrentText("Custom")
        state = "locked" if value <= 1e-9 else f"influence {value:.2f}"
        self.status_label.setText(
            f"IK joint {name}: {state}; joint-weight preset is now Custom"
        )

    def _set_all_ik_influences(self, selector):
        for name, control in self.ik_influence_controls.items():
            value = float(selector(name))
            self.ik_joint_weights[name] = value
            control.set_value(value)

    def _apply_selected_limb_weights(self):
        kind, object_name = self._selected_target()
        logical = self.reverse_bindings.get((kind, object_name), "")
        frame_reference = logical or object_name or ""
        chain = set(
            self.robot_model.limb_joint_chain_for_frame(frame_reference)
            if frame_reference and hasattr(
                self.robot_model, "limb_joint_chain_for_frame"
            ) else ()
        )
        passive = set(getattr(self.robot_model, "passive_joints", ()))
        self._set_all_ik_influences(
            lambda name: 1.0
            if name in chain and name not in passive else 0.0
        )
        return frame_reference, len(chain - passive)

    def _sync_live_ik_weight_preset(self):
        if self.active_ik_weight_preset != "Selected limb only":
            return False
        logical, enabled_count = self._apply_selected_limb_weights()
        selected = logical or "non-limb target"
        self.status_label.setText(
            f"Selected limb only synced to {selected}: "
            f"{enabled_count} joint weights enabled"
        )
        return True

    def apply_ik_preset(self):
        preset = self.ik_preset_box.currentText()
        if preset == "Custom":
            self.active_ik_weight_preset = preset
            self.status_label.setText(
                "Custom joint weights active; target changes will not "
                "overwrite them"
            )
            return
        if preset == "All joints normal":
            defaults = (
                self.robot_model.default_ik_joint_weights()
                if hasattr(self.robot_model, "default_ik_joint_weights")
                else {}
            )
            self._set_all_ik_influences(
                lambda name: defaults.get(name, 1.0)
            )
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
            self._apply_selected_limb_weights()
        elif preset == "Feet planted":
            self._set_all_ik_influences(lambda name: 1.0)
            checkbox, spin = self.ik_task_controls["foot_lock"]
            checkbox.setChecked(True)
            spin.setValue(max(1.0, spin.value()))
        self.active_ik_weight_preset = preset
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
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(True)
        self._update_root_pose_label()

    def _on_transform_moved(self, position, quaternion):
        kind, name = self._selected_target()
        if not name:
            self.status_label.setText("Target pose moved; no body/site is selected.")
            return
        self.begin_preview()
        self._use_editor_canvas_states()
        self.canvas.set_preview_visible(True)
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
        state_name = self.canvas.gizmo.state.name
        rotation_drag = "ROTATE" in state_name
        if not rotation_drag and "TRANSLATE" not in state_name:
            position_changed = not np.allclose(
                position, self.last_valid_target_position, atol=1e-9, rtol=0.0
            )
            quaternion_changed = not np.allclose(
                quaternion,
                self.last_valid_target_quaternion,
                atol=1e-9,
                rtol=0.0,
            )
            rotation_drag = quaternion_changed and not position_changed
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
            # Translation keeps orientation enabled as a best-effort task;
            # rotation requires both orientation and the held TCP position.
            tcp_position_required=True,
            tcp_orientation_required=rotation_drag,
        )
        if result.success:
            projection = project_qpos_above_flat_ground(
                self.robot_model,
                result.qpos,
                checker=self.collision_checker,
            )
            if projection.success:
                result.qpos = projection.qpos
                self.preview_state.set_qpos(result.qpos)
                if projection.changed:
                    result.position, result.quaternion = (
                        self.preview_state.get_body_pose(name, kind)
                    )
                    result.status += (
                        "; final ground barrier correction "
                        f"{projection.applied_offset * 1000.0:.1f} mm"
                    )
                self.last_valid_target_position = result.position.copy()
                self.last_valid_target_quaternion = result.quaternion.copy()
            else:
                result.success = False
                result.status = (
                    "Safety barrier retained the last safe drag pose: "
                    f"{projection.reason}"
                )

        # Blocking contacts clamp at the last safe substep. IK failures can
        # likewise leave the handle at the last solvable substep.
        self.canvas.set_target_pose(
            self.last_valid_target_position,
            self.last_valid_target_quaternion,
        )
        self._update_preview_collisions(
            result.collisions
            if result.success or result.collisions
            else None
        )
        self._sync_joint_controls()
        logical_frame = self.reverse_bindings.get((kind, name), name)
        model_name = getattr(
            self.robot_model, "model_name", self.robot_model.model_path.stem
        )
        active_task_count = selected_task_count + (
            0 if is_free_root else len(secondary_tasks)
        )
        singularity_note = ""
        if result.near_singularity:
            singularity_note = (
                f"; near singularity sigma_min={result.min_singular_value:.2e}, "
                f"cond={result.condition_number:.1e}"
            )
        drag_status = self.canvas.gizmo.drag_status()
        drag_prefix = f"{drag_status}; " if drag_status else ""
        self._last_ik_status = (
            drag_prefix +
            f"{'TCP free translate; ' if self.canvas.gizmo.state.name == 'DRAG_TRANSLATE_FREE' else ''}"
            f"{result.status}; accepted={result.accepted_fraction:.0%}; "
            f"IK error={result.ik_error:.4f}; tasks={active_task_count}; "
            f"frame={logical_frame}; model={model_name}{singularity_note}; "
            "preview not committed"
        )
        self.status_label.setText(self._last_ik_status)
        roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
        self.target_pose_dragged.emit(
            *map(float, self.last_valid_target_position), roll, pitch, yaw
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

    def _on_transform_cancel_requested(self):
        if self.preview_active:
            self.cancel_preview()
        else:
            self.status_label.setText("Transform edit cancelled.")

    def _on_gizmo_mode_changed(self, mode):
        self.status_label.setText(f"Transform gizmo mode: {mode}.")

    def _joint_limit_violation(self, qpos):
        for joint in self.robot_model.joints.values():
            if joint.limits is None:
                continue
            value = float(qpos[joint.qpos_address])
            lo, hi = joint.limits
            if value < lo - 1e-9 or value > hi + 1e-9:
                return (
                    f"{joint.name} outside limits "
                    f"[{lo:.3f}, {hi:.3f}] at {value:.3f}"
                )
        return None

    def _build_validated_preview_path(self, start, goal, samples=40):
        planned = []
        collision_indices = []
        blocking_collision_indices = []
        first_collisions = ()
        candidate = self.robot_model.create_state()
        for index, alpha in enumerate(np.linspace(0.0, 1.0, int(samples))):
            qpos = self.state_timeline._interpolate(start, goal, alpha)
            qpos = np.asarray(qpos, dtype=float)
            if not np.all(np.isfinite(qpos)):
                return (
                    PreviewPathValidation(
                        False, f"non-finite qpos at path sample {index}", index
                    ),
                    [],
                )
            limit_error = self._joint_limit_violation(qpos)
            if limit_error:
                return (
                    PreviewPathValidation(
                        False, f"{limit_error} at path sample {index}", index
                    ),
                    [],
                )
            candidate.set_qpos(qpos)
            planned.append(qpos)
            collisions = (
                self.collision_checker.get_collisions(candidate)
                if self.collision_checker else []
            )
            if collisions:
                collision_indices.append(index)
                if any(item.blocking for item in collisions):
                    blocking_collision_indices.append(index)
                if not first_collisions:
                    first_collisions = tuple(collisions)
        adaptive_warning = None
        adaptive_blocking = None
        if self.collision_checker is not None:
            adaptive_warning, adaptive_blocking = (
                adaptive_trajectory_collision_reports(
                    self.robot_model,
                    (start, goal),
                    times=(0.0, 1.0),
                    checker=self.collision_checker,
                )
            )
        adaptive_report = adaptive_blocking or adaptive_warning
        if adaptive_report is not None:
            fraction = float(
                getattr(adaptive_report, "segment_fraction", 0.0)
            )
            report_index = max(
                0,
                min(len(planned) - 1, int(round(fraction * (len(planned) - 1)))),
            )
            if report_index not in collision_indices:
                collision_indices.append(report_index)
            if adaptive_report.blocking and report_index not in blocking_collision_indices:
                blocking_collision_indices.append(report_index)
            if not first_collisions:
                first_collisions = tuple(adaptive_report.collisions)
        if collision_indices:
            names = format_collision_pairs(first_collisions)
            details = format_collision_diagnostics(first_collisions)
            barrier_note = (
                " A blocking contact was found between poses, so Commit "
                "Keyframe will remain blocked; use Try Safe Reroute."
                if blocking_collision_indices else
                " Contacts are advisory, so the path may still be committed."
            )
            report_note = (
                f" Adaptive first issue: "
                f"{self._motion_report_text(adaptive_report)}."
                if adaptive_report is not None else ""
            )
            message = (
                f"Preview Path contains collision warnings at "
                f"{len(collision_indices)} of {len(planned)} samples; "
                f"first at sample {collision_indices[0]}: {names}; "
                f"contact geometry: {details}. Red poses mark the contacts."
                f"{report_note}{barrier_note}"
            )
            return (
                PreviewPathValidation(
                    True,
                    message,
                    collision_indices[0],
                    tuple(collision_indices),
                    tuple(blocking_collision_indices),
                    first_collisions,
                ),
                planned,
            )
        return PreviewPathValidation(True, "Preview Path is valid."), planned

    def plan_preview(self):
        if not self.preview_active:
            self.status_label.setText("No preview changes to validate.")
            return
        start = self.committed_state.get_qpos()
        goal = self.preview_state.get_qpos()
        validation, planned = self._build_validated_preview_path(start, goal)
        if not validation.ok:
            self._clear_ghost_overlay(source="preview_path")
            self.status_label.setText(
                f"Cannot preview path: {validation.message}."
            )
            return
        self.ghost_trajectory = [qpos.copy() for qpos in planned]
        collision_indices = set(validation.collision_indices)
        self.ghost_collision_flags = [
            index in collision_indices for index in range(len(planned))
        ]
        self.ghost_source = "preview_path"
        self._rebuild_ghosts()
        self.status_label.setText(
            validation.message
            if validation.collision_indices
            else "Preview Path is valid; no keyframe was changed."
        )
        self.history_action_finished.emit("Preview path")

    def _candidate_timeline_motion(self, qpos, time):
        time = self.state_timeline.time_key(time)
        states = [
            (float(existing_time), self.state_timeline.get_state(existing_time))
            for existing_time in self.state_timeline.times()
            if self.state_timeline.time_key(existing_time) != time
        ]
        states.append((float(time), np.asarray(qpos, dtype=float).copy()))
        states.sort(key=lambda item: item[0])
        return (
            tuple(item[0] for item in states),
            tuple(item[1] for item in states),
        )

    def _validate_candidate_keyframe_intervals(self, qpos, time):
        """Validate only the intervals whose endpoint this commit replaces."""
        if self.state_timeline is None or self.collision_checker is None:
            return None, None, (), ()
        times, qposes = self._candidate_timeline_motion(qpos, time)
        candidate_key = self.state_timeline.time_key(time)
        candidate_index = next(
            index
            for index, value in enumerate(times)
            if self.state_timeline.time_key(value) == candidate_key
        )
        pair_starts = []
        if candidate_index > 0:
            pair_starts.append(candidate_index - 1)
        if candidate_index + 1 < len(times):
            pair_starts.append(candidate_index)

        first_warning = None
        first_blocking = None
        for pair_start in pair_starts:
            warning, blocking = adaptive_trajectory_collision_reports(
                self.robot_model,
                qposes[pair_start:pair_start + 2],
                times=times[pair_start:pair_start + 2],
                checker=self.collision_checker,
            )
            if first_warning is None and warning is not None:
                first_warning = warning
            if first_blocking is None and blocking is not None:
                first_blocking = blocking
        return first_warning, first_blocking, times, qposes

    def accept_preview(self, *, emit_pose_finished=True):
        if not self.preview_active:
            self.status_label.setText("No preview changes to commit.")
            return False
        preview_qpos = self.preview_state.get_qpos()
        if not np.all(np.isfinite(preview_qpos)):
            self.status_label.setText(
                "Cannot commit keyframe: preview pose contains non-finite qpos values."
            )
            return False
        limit_error = self._joint_limit_violation(preview_qpos)
        if limit_error is not None:
            self.status_label.setText(
                f"Cannot commit Keyframe: {limit_error}."
            )
            return False
        projection = project_qpos_above_flat_ground(
            self.robot_model,
            preview_qpos,
            checker=self.collision_checker,
        )
        if not projection.success:
            collisions = (
                self.collision_checker.get_collisions(self.preview_state)
                if self.collision_checker else []
            )
            self.canvas.set_preview_collisions(collisions)
            self.status_label.setText(
                "Cannot commit Keyframe: safety barrier could not correct the "
                f"preview ({projection.reason})."
            )
            return False
        if projection.changed:
            preview_qpos = projection.qpos
            self.preview_state.set_qpos(preview_qpos)

        collisions = (
            self.collision_checker.get_collisions(self.preview_state)
            if self.collision_checker else []
        )
        blocking_collisions = [item for item in collisions if item.blocking]
        if blocking_collisions:
            self.canvas.set_preview_collisions(collisions)
            names = format_collision_pairs(blocking_collisions)
            details = format_collision_diagnostics(blocking_collisions)
            self.status_label.setText(
                f"Cannot commit keyframe: blocking collision ({names}); "
                f"contact geometry: {details}."
            )
            return False
        (
            interval_warning,
            interval_blocking,
            candidate_times,
            candidate_qposes,
        ) = self._validate_candidate_keyframe_intervals(
            preview_qpos, self.current_time
        )
        if interval_blocking is not None:
            self.quarantine_motion(
                candidate_qposes,
                candidate_times,
                report=interval_blocking,
                source="Keyframe interval",
            )
            self.canvas.set_preview_collisions(collisions)
            self.status_label.setText(
                "Cannot commit Keyframe: the endpoint is safe, but adaptive "
                "validation found a blocking contact "
                f"{self._motion_report_text(interval_blocking)}. "
                "Use Try Safe Reroute or adjust the surrounding Keyframes."
            )
            return False
        self.timeslice_slider.set_safety_reports((interval_warning,))
        self.committed_state.set_qpos(preview_qpos)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        if projection.changed:
            self.status_label.setText(
                f"Committed Keyframe at t={self.current_time:.2f} s after the "
                f"ground barrier raised the base "
                f"{projection.applied_offset * 1000.0:.1f} mm"
            )
        elif collisions or interval_warning is not None:
            warning_collisions = (
                collisions
                if collisions else interval_warning.collisions
            )
            names = format_collision_pairs(warning_collisions)
            details = format_collision_diagnostics(warning_collisions)
            self.status_label.setText(
                f"Committed keyframe at t={self.current_time:.2f} s; "
                f"Collision warning: {names}; Contact geometry: {details}"
            )
        else:
            self.status_label.setText(
                f"Committed keyframe at t={self.current_time:.2f} s"
            )
        if emit_pose_finished and self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )
        return True

    def cancel_preview(self):
        if not self.robot_state:
            return
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
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

    def _trajectory_sample(self, time):
        """Return an interpolated playback qpos and its nearest frame index."""
        if not self.robot_trajectory:
            return None, None

        times = self.robot_trajectory_times
        if len(times) != len(self.robot_trajectory):
            times = [float(index) for index in range(len(self.robot_trajectory))]

        time = float(time)
        if len(times) == 1 or time <= times[0]:
            return self.robot_trajectory[0].copy(), 0
        if time >= times[-1]:
            last = len(self.robot_trajectory) - 1
            return self.robot_trajectory[last].copy(), last

        upper = bisect_right(times, time)
        lower = max(0, upper - 1)
        upper = min(len(times) - 1, upper)
        lower_time = times[lower]
        upper_time = times[upper]
        if upper_time <= lower_time:
            return self.robot_trajectory[upper].copy(), upper

        fraction = (time - lower_time) / (upper_time - lower_time)
        qpos = self.state_timeline._interpolate(
            self.robot_trajectory[lower],
            self.robot_trajectory[upper],
            fraction,
        )
        nearest = (
            lower
            if time - lower_time <= upper_time - time
            else upper
        )
        return qpos, nearest

    def _update_frame_readout(self, time=None, frame_index=None):
        if not self.robot_trajectory:
            self.timeslice_frame_readout.setText("Frame —")
            return
        if frame_index is None:
            _qpos, frame_index = self._trajectory_sample(
                self.display_time if time is None else time
            )
        self.timeslice_frame_readout.setText(
            f"Frame {int(frame_index) + 1} / {len(self.robot_trajectory)}"
        )

    def _set_canvas_target_from_state(self, state):
        kind, name = self._selected_target()
        if not name:
            return
        try:
            position, quaternion = state.get_body_pose(name, kind)
        except KeyError:
            return
        self.canvas.set_target_pose(position, quaternion)

    def _emit_selected_target_pose(self):
        if (
            self.last_valid_target_position is None
            or self.last_valid_target_quaternion is None
        ):
            return
        roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
        self.target_pose_dragged.emit(
            *map(float, self.last_valid_target_position),
            roll,
            pitch,
            yaw,
        )

    def _use_editor_canvas_states(self):
        if not self.robot_state:
            return
        self.canvas.set_robot_states(
            self.committed_state, self.preview_state, self.ghost_renderer
        )
        self.canvas.set_preview_visible(self.preview_active)

    def preview_trajectory_time(self, time, emit_time_signal=False):
        """Display a timeline pose without creating an editable qpos state."""
        if not self.robot_state:
            return None
        time = max(0.0, min(float(time), self.timeline_duration))
        self.display_time = time
        self._set_timeslice_widgets(time)

        qpos, frame_index = self._trajectory_sample(time)
        if qpos is None and self.state_timeline:
            qpos = self.state_timeline.sample_state(
                time,
                fallback_qpos=self.committed_state.get_qpos(),
            )

        if qpos is not None and self.playback_state is not None:
            self.playback_state.set_qpos(qpos)
            self.canvas.set_robot_states(
                self.playback_state, self.preview_state, self.ghost_renderer
            )
            self.canvas.set_preview_visible(False)
            self._sync_joint_controls(state=self.playback_state)
            self._set_canvas_target_from_state(self.playback_state)
            self.canvas.update()

        self._update_frame_readout(time, frame_index)
        self._update_timeline_label(time)
        if emit_time_signal:
            self.timeslice_preview_time_changed.emit(time)
        return None if qpos is None else qpos.copy()

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
        self._use_editor_canvas_states()
        self.committed_state.set_qpos(qpos)
        self.preview_state.set_qpos(qpos)
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
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
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        time = max(0.0, min(float(time), self.timeline_duration))
        self.current_time = self.state_timeline.time_key(time)
        self.display_time = self.current_time
        self._set_timeslice_widgets(self.current_time)
        qpos, _frame_index = self._trajectory_sample(self.current_time)
        if qpos is not None:
            # Releasing the unified scrubber commits exactly the pose that was
            # shown during live trajectory sampling.
            self.state_timeline.set_state(self.current_time, qpos)
        else:
            qpos = self.ensure_keyframe_at_current_time()
        self.set_robot_state_for_current_time(qpos)
        self._emit_selected_target_pose()
        self._update_frame_readout(self.current_time)
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
        self.pause_playback(commit_time=was_playing)
        self.canvas.cancel_transform_drag()
        self.committed_state.reset_to_default()
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Reset 3D pose once at t={self.current_time:.2f} s to model home qpos"
            + ("; playback paused" if was_playing else "")
        )
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self._pending_history_action_description = "Reset 3D pose"
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )

    def choose_qpos_csv(self):
        self._open_csv_file_dialog(
            title="Load robot qpos",
            directory=QPOS_CSV_DIR,
            selected=self._load_selected_qpos_csv,
        )

    def choose_trajectory_csv(self, prompt_import_dt=False):
        self._open_csv_file_dialog(
            title="Load robot trajectory",
            directory=TRAJECTORY_CSV_DIR,
            selected=lambda path: self._load_selected_trajectory_csv(
                path, prompt_import_dt
            ),
        )

    def choose_qpos_save_path(self):
        self._open_csv_file_dialog(
            title="Save robot qpos",
            directory=QPOS_CSV_DIR,
            selected=self._save_selected_qpos_csv,
            save=True,
            filename="updated_qpos.csv",
        )

    def choose_trajectory_save_path(self):
        self._open_csv_file_dialog(
            title="Export MuJoCo trajectory",
            directory=TRAJECTORY_CSV_DIR,
            selected=self._save_selected_trajectory_csv,
            save=True,
            filename="robot_trajectory.csv",
        )

    def choose_dsms_trajectory_output_dir(self):
        self._open_csv_file_dialog(
            title="Export DSMS trajectory folder",
            directory=TRAJECTORY_CSV_DIR,
            selected=self._save_selected_dsms_trajectory,
            directory_mode=True,
            name_filter="Folders (*)",
        )

    def choose_mjlab_trajectory_save_path(self):
        error = self.mjlab_export_compatibility_error()
        if error:
            self.status_label.setText(f"Could not export mjlab trajectory: {error}")
            return
        self._open_csv_file_dialog(
            title="Export mjlab trajectory",
            directory=TRAJECTORY_CSV_DIR,
            selected=self._save_selected_mjlab_trajectory,
            save=True,
            filename="robot_trajectory_mjlab.csv",
        )

    def mjlab_export_compatibility_error(self):
        if self.robot_model is None:
            return "no robot model is loaded"
        return mjlab_compatibility_error(self.robot_model)

    def _open_csv_file_dialog(
        self,
        *,
        title,
        directory,
        selected,
        save=False,
        filename=None,
        directory_mode=False,
        name_filter="CSV files (*.csv);;All files (*)",
    ):
        if self.background_jobs.is_busy():
            self.status_label.setText(
                "Wait for the current import or export to finish."
            )
            return
        if self.csv_file_selection_stage.is_active():
            self.status_label.setText("A file selector is already open.")
            return

        self.csv_file_selection_stage.select_file(
            mode="directory" if directory_mode else "save" if save else "open",
            title=title,
            directory=directory,
            name_filter=name_filter,
            filename=filename,
            selected=selected,
            failed=lambda message: self.status_label.setText(
                f"Could not open file selector: {message}"
            ),
        )

    def _load_selected_qpos_csv(self, path):
        expected = int(self.robot_model.mj_model.nq)
        self.status_label.setText(f"Loading qpos from {Path(path).name}...")
        self._submit_csv_job(
            "load qpos CSV",
            lambda: read_qpos_csv(path, expected),
            self._apply_loaded_qpos,
            lambda error: self.status_label.setText(
                f"Could not load qpos CSV: {error}"
            ),
        )

    def _load_selected_trajectory_csv(self, path, prompt_import_dt=False):
        expected = int(self.robot_model.mj_model.nq)
        self.status_label.setText(
            f"Loading trajectory from {Path(path).name}..."
        )
        self._submit_csv_job(
            "load trajectory CSV",
            lambda: read_trajectory_csv(path, expected),
            lambda loaded: self._apply_loaded_trajectory(
                loaded,
                prompt_import_dt=prompt_import_dt,
                background_postprocess=True,
            ),
            lambda error: self._trajectory_csv_load_failed(error),
        )

    def _save_selected_qpos_csv(self, path):
        qpos = self.committed_state.get_qpos().copy()
        preview_active = bool(self.preview_active)
        self.status_label.setText(f"Saving qpos to {Path(path).name}...")
        self._submit_csv_job(
            "save qpos CSV",
            lambda: write_qpos_csv(path, qpos),
            lambda saved_path: self._show_qpos_saved(
                saved_path, preview_active
            ),
            lambda error: self.status_label.setText(
                f"Could not save qpos CSV: {error}"
            ),
        )

    def _save_selected_trajectory_csv(self, path):
        try:
            export = self._trajectory_export_snapshot()
        except ValueError as error:
            self.status_label.setText(
                f"Could not save trajectory CSV: {error}"
            )
            return
        self.status_label.setText(
            f"Saving trajectory to {Path(path).name}..."
        )
        self._submit_csv_job(
            "save trajectory CSV",
            lambda: write_trajectory_csv(path, export),
            lambda saved_path: self._show_trajectory_saved(
                saved_path, export
            ),
            lambda error: self.status_label.setText(
                f"Could not save trajectory CSV: {error}"
            ),
        )

    def _save_selected_dsms_trajectory(self, output_dir):
        try:
            export = self._trajectory_export_snapshot(
                sample_dt=self.export_dt()
            )
        except ValueError as error:
            self.status_label.setText(
                f"Could not export DSMS trajectory: {error}"
            )
            return
        dof = len(self.robot_model.actuated_joints)
        free_joints = tuple(self.robot_model.free_joints_by_body.values())
        base_qpos_address = (
            free_joints[0].qpos_address if len(free_joints) == 1 else None
        )
        motion_speed = self.dsms_motion_speed()
        self.status_label.setText(
            f"Exporting DSMS trajectory to {Path(output_dir).name}..."
        )
        self._submit_csv_job(
            "export DSMS trajectory",
            lambda: export_dsms_trajectory(
                output_dir,
                export,
                dof=dof,
                base_qpos_address=base_qpos_address,
                motion_speed=motion_speed,
            ),
            lambda result: self._show_format_trajectory_saved(
                "DSMS", result, export
            ),
            lambda error: self.status_label.setText(
                f"Could not export DSMS trajectory: {error}"
            ),
        )

    def _save_selected_mjlab_trajectory(self, path):
        try:
            export = self._trajectory_export_snapshot(
                sample_dt=self.export_dt()
            )
        except ValueError as error:
            self.status_label.setText(
                f"Could not export mjlab trajectory: {error}"
            )
            return
        self.status_label.setText(
            f"Exporting mjlab trajectory to {Path(path).name}..."
        )
        self._submit_csv_job(
            "export mjlab trajectory",
            lambda: export_mjlab_trajectory(
                path,
                export,
                self.robot_model,
                single_sample_dt=self.export_dt(),
            ),
            lambda result: self._show_format_trajectory_saved(
                "mjlab", result, export
            ),
            lambda error: self.status_label.setText(
                f"Could not export mjlab trajectory: {error}"
            ),
        )

    def _show_format_trajectory_saved(self, format_name, result, export):
        if len(result.paths) == 1:
            destination = str(result.paths[0])
        else:
            destination = (
                f"{result.paths[0].parent} "
                f"({', '.join(path.name for path in result.paths)})"
            )
        if result.motion_speed is not None:
            speed_note = f"; motion speed {result.motion_speed:.6g}×"
            duration_note = (
                ""
                if result.source_duration is None
                or result.output_duration is None
                else f"; duration {result.source_duration:.6g} s → "
                f"{result.output_duration:.6g} s"
            )
            fps_note = (
                ""
                if result.source_fps is None or result.input_fps is None
                else f"; reference frequency {result.source_fps:.6g} Hz → "
                f"{result.input_fps:.6g} Hz"
            )
        else:
            speed_note = ""
            duration_note = ""
            fps_note = (
                ""
                if result.input_fps is None
                else f"; input frequency {result.input_fps:.6g} Hz"
            )
        preview_note = (
            "; unaccepted preview was not saved"
            if export.preview_active
            else ""
        )
        collision_note = ""
        if self._last_export_collision_warning is not None:
            report = self._last_export_collision_warning
            collision_note = (
                f"; Collision warning {report.location_label}: "
                f"{format_collision_pairs(report.collisions)}"
            )
        self.status_label.setText(
            f"Saved {result.sample_count} {format_name} trajectory samples "
            f"to {destination}{speed_note}{duration_note}{fps_note}"
            f"{preview_note}{collision_note}"
        )

    def _submit_csv_job(self, name, work, succeeded, failed):
        if self.background_jobs.is_busy():
            self.status_label.setText(
                "Wait for the current import or export to finish."
            )
            return False
        submitted = self.background_jobs.submit(
            name,
            work,
            succeeded,
            failed,
        )
        if not submitted:
            self.status_label.setText(f"Could not start {name}.")
        return submitted

    def load_qpos_csv(self, csv_path):
        """Load one headerless MuJoCo qpos row into the active keyframe."""
        expected = int(self.robot_model.mj_model.nq)
        loaded = read_qpos_csv(csv_path, expected)
        self._apply_loaded_qpos(loaded)

    def _apply_loaded_qpos(self, loaded):
        self.pause_playback()
        self.canvas.cancel_transform_drag()
        limit_error = self._joint_limit_violation(loaded.qpos)
        if limit_error is not None:
            self.status_label.setText(
                f"Could not load qpos from {loaded.path.name}: {limit_error}."
            )
            return False
        projection = project_qpos_above_flat_ground(
            self.robot_model,
            loaded.qpos,
            checker=self.collision_checker,
        )
        if not projection.success:
            self.status_label.setText(
                f"Could not load qpos from {loaded.path.name}: safety barrier "
                f"rejected the pose ({projection.reason})."
            )
            return False
        self.set_robot_state_for_current_time(projection.qpos)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        correction = (
            f"; ground barrier raised the base "
            f"{projection.applied_offset * 1000.0:.1f} mm"
            if projection.changed else ""
        )
        self.status_label.setText(
            f"Loaded {loaded.qpos.size}-value qpos from {loaded.path.name} at "
            f"t={self.current_time:.2f} s{correction}"
        )
        self.history_action_finished.emit("Load qpos")
        return True

    def load_trajectory_csv(self, csv_path):
        """Load headerless time,qpos rows into playback and editable qpos states."""
        expected = int(self.robot_model.mj_model.nq)
        loaded = read_trajectory_csv(csv_path, expected)
        self._apply_loaded_trajectory(loaded)

    def _apply_loaded_trajectory(
        self,
        loaded,
        prompt_import_dt=None,
        background_postprocess=False,
    ):
        qposes = list(loaded.qposes)
        times = list(loaded.times)
        if prompt_import_dt is not None:
            self._prompt_trajectory_import_dt_on_load = bool(prompt_import_dt)
        self.pause_playback()
        self.canvas.cancel_transform_drag()
        if not self.set_robot_trajectory(
            qposes,
            times=times,
            quarantine_source="import",
            quarantine_source_path=loaded.path,
        ):
            self._background_trajectory_postprocess_requested = False
            return False
        if self.state_timeline and qposes:
            self.state_timeline.reset(times[0], qposes[0])
            for time, qpos in zip(times[1:], qposes[1:]):
                self.state_timeline.set_state(time, qpos)
            self.set_defined_timeslices(times)
            self.current_time = self.state_timeline.time_key(times[0])
            self.display_time = self.current_time
            self._set_timeslice_widgets(self.current_time)
            self._update_frame_readout(self.current_time)
            self._update_timeline_label()
        duration = times[-1] if times else 0.0
        self.status_label.setText(
            f"Loaded {len(qposes)} timed qpos trajectory states from "
            f"{loaded.path.name} ({duration:.3f} s)"
        )
        self._background_trajectory_postprocess_requested = bool(
            background_postprocess
        )
        self.trajectory_csv_loaded.emit(str(loaded.path))
        return True

    def _trajectory_csv_load_failed(self, error):
        self._prompt_trajectory_import_dt_on_load = False
        self.status_label.setText(f"Could not load trajectory CSV: {error}")

    def save_qpos_csv(self, csv_path):
        """Save the committed active keyframe as one headerless qpos row."""
        preview_active = bool(self.preview_active)
        path = write_qpos_csv(csv_path, self.committed_state.get_qpos())
        self._show_qpos_saved(path, preview_active)
        return path

    def _show_qpos_saved(self, path, preview_active):
        preview_note = (
            "; unaccepted preview was not saved" if preview_active else ""
        )
        self.status_label.setText(f"Saved committed qpos to {path}{preview_note}")

    def save_trajectory_csv(self, csv_path):
        """Save generated trajectory rows, falling back to editable timeline rows."""
        export = self._trajectory_export_snapshot()
        path = write_trajectory_csv(csv_path, export)
        self._show_trajectory_saved(path, export)
        return path

    def _trajectory_export_snapshot(self, sample_dt=None):
        if not self.state_timeline:
            raise ValueError("no robot timeline is available")

        expected = int(self.robot_model.mj_model.nq)
        source_name = "generated trajectory"

        if self.robot_trajectory:
            qposes = tuple(qpos.copy() for qpos in self.robot_trajectory)
            times = tuple(float(time) for time in self.robot_trajectory_times)
            if len(times) != len(qposes):
                times = tuple(float(index) for index in range(len(qposes)))
        else:
            self.state_timeline.set_state(
                self.current_time, self.committed_state.get_qpos()
            )
            times = tuple(float(time) for time in self.state_timeline.times())
            qposes = tuple(
                self.state_timeline.get_state(time_value)
                for time_value in times
            )
            source_name = "editable timeline"

        export = TrajectoryExport(
            expected_qpos_count=expected,
            times=times,
            qposes=qposes,
            source_name=source_name,
            preview_active=bool(self.preview_active),
        )
        if sample_dt is not None:
            export = resample_trajectory_export(
                export,
                self.robot_model,
                sample_dt,
            )
            times = export.times
            qposes = export.qposes

        for index, qpos in enumerate(qposes):
            limit_error = self._joint_limit_violation(qpos)
            if limit_error is not None:
                raise ValueError(
                    f"{source_name} qpos row {index + 1} has {limit_error}"
                )

        # Always validate the exact export candidate. This catches collisions
        # between rows even when endpoints are safe, and remains independent
        # of the user-selected Export interval.
        warning_report, blocking_report = (
            adaptive_trajectory_collision_reports(
                self.robot_model,
                qposes,
                times=times,
                checker=self.collision_checker,
            )
            if qposes else (None, None)
        )
        if blocking_report is not None:
            names = format_collision_pairs(blocking_report.collisions)
            details = format_collision_diagnostics(blocking_report.collisions)
            raise ValueError(
                f"{source_name} has a blocking collision "
                f"{blocking_report.location_label}: {names}; "
                f"contact geometry: {details}"
            )
        self._last_export_collision_warning = warning_report

        return export

    def _show_trajectory_saved(self, path, export):
        preview_note = (
            "; unaccepted preview was not saved"
            if export.preview_active
            else ""
        )
        collision_note = ""
        if self._last_export_collision_warning is not None:
            report = self._last_export_collision_warning
            collision_note = (
                f"; Collision warning {report.location_label}: "
                f"{format_collision_pairs(report.collisions)}"
            )
        self.status_label.setText(
            f"Saved {len(export.qposes)} timed qpos states from "
            f"{export.source_name} "
            f"to {path}{preview_note}{collision_note}"
        )

    def robot_trajectory_collision_status(self):
        report = (
            self.robot_trajectory_blocking_report
            or self.robot_trajectory_warning_report
        )
        if report is None:
            return ""
        severity = "Blocking collision" if report.blocking else "Collision warning"
        return (
            f"{severity} {getattr(report, 'location_label', '')}: "
            f"{format_collision_pairs(report.collisions)}; "
            f"Contact geometry: {format_collision_diagnostics(report.collisions)}"
        )

    def _refresh_timeline_trajectory(self):
        # Timeline keyframes no longer publish ghost poses. Scrubbing and
        # accepting states should update the committed robot, not silently swap
        # the overlay from preview/playback into keyframe ghosts.
        self._update_timeline_label()

    def _update_timeline_label(self, display_time=None):
        count = len(self.state_timeline.states) if self.state_timeline else 0
        time = self.current_time if display_time is None else float(display_time)
        self.timeline_state_label.setText(
            f"3D state time: {time:.2f} s ({count} keyframes)"
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

    def set_robot_trajectory(
        self,
        qposes,
        times=None,
        activate_first_frame=True,
        *,
        quarantine_source="motion",
        quarantine_source_path=None,
    ):
        """Publish motion only after endpoint and between-frame validation."""
        if not self.robot_state:
            return False
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        qposes = tuple(qposes)
        times = (
            tuple(float(index) for index in range(len(qposes)))
            if times is None else tuple(times)
        )
        try:
            valid_times, valid = validate_trajectory_arrays(
                times, qposes, int(self.robot_state.mj_model.nq)
            )
        except ValueError as error:
            self.status_label.setText(
                f"Could not publish {quarantine_source}: {error}"
            )
            return False
        for index, qpos in enumerate(valid):
            limit_error = self._joint_limit_violation(qpos)
            if limit_error is not None:
                self.status_label.setText(
                    f"Could not publish {quarantine_source}: qpos row "
                    f"{index + 1} has {limit_error}."
                )
                return False

        warning_report, blocking_report = (
            adaptive_trajectory_collision_reports(
                self.robot_model,
                valid,
                times=valid_times,
                checker=self.collision_checker,
            )
            if valid else (None, None)
        )
        if blocking_report is not None:
            self.quarantine_motion(
                valid,
                valid_times,
                report=blocking_report,
                source=quarantine_source,
                source_path=quarantine_source_path,
            )
            return False

        self._clear_motion_safety_review()
        self.robot_trajectory = valid
        self.robot_trajectory_times = valid_times
        self.robot_trajectory_warning_report = warning_report
        self.robot_trajectory_blocking_report = None
        self.timeslice_slider.set_safety_reports((warning_report,))
        if valid_times:
            self.set_timeline_duration(max(self.timeline_duration, max(valid_times)))
        self._clear_ghost_overlay(source="preview_path")
        self._sync_playback_pose_ghosts()
        if valid and activate_first_frame:
            self.set_current_time(valid_times[0])
            self.status_label.setText(f"Loaded {len(valid)} robot trajectory states.")
        else:
            self._update_frame_readout(self.display_time)
        return True

    def clear_robot_trajectory(self):
        self.pause_playback()
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self.robot_trajectory = []
        self.robot_trajectory_times = []
        self.robot_trajectory_warning_report = None
        self.robot_trajectory_blocking_report = None
        self._clear_motion_safety_review()
        self.timeslice_slider.set_safety_reports(())
        self.ghost_trajectory = []
        self.ghost_collision_flags = []
        self.ghost_source = None
        self.display_time = self.current_time
        self._set_timeslice_widgets(self.display_time)
        self._update_frame_readout()
        self._use_editor_canvas_states()
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        if self.ghost_renderer:
            self.ghost_renderer.clear()
        self._update_ghost_options()

    def clear_editable_timeline(self, keep_current_pose=True, reset_time=None):
        if not self.state_timeline or not self.robot_state:
            return

        qpos = (
            self.committed_state.get_qpos()
            if keep_current_pose
            else self.robot_model.home_qpos
        )
        if reset_time is not None:
            self.current_time = self.state_timeline.time_key(reset_time)
            self.display_time = self.current_time
            self._set_timeslice_widgets(self.current_time)
        self.state_timeline.reset(self.current_time, qpos)
        self.set_robot_state_for_current_time(qpos)
        self._update_timeline_label()

    def load_backend_states(self, states):
        if not self.robot_state:
            return
        qposes = []
        times = []
        for configuration in states:
            times.append(float(getattr(configuration, "time", len(times))))
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
        return self.set_robot_trajectory(
            qposes,
            times=times,
            quarantine_source="generated",
        )

    def generate_demo_trajectory(self):
        start = self.robot_state.get_qpos()
        target = start.copy()
        names = self.robot_state.get_joint_names()
        preferred = "left_shoulder_pitch_joint"
        name = preferred if preferred in names else names[0]
        joint = self.robot_model.joints[name]
        lo, hi = joint.limits or (-1.0, 1.0)
        target[joint.qpos_address] = max(lo, min(hi, start[joint.qpos_address] + 0.35))
        qposes = interpolate_qpos(start, target, 60)
        frame_period = self.play_timer.interval() / 1000.0
        times = [index * frame_period for index in range(len(qposes))]
        if self.set_robot_trajectory(
            qposes,
            times=times,
            quarantine_source="demo",
        ):
            self.history_action_finished.emit("Demo trajectory")

    def _rebuild_ghosts(self):
        if self.ghost_renderer:
            self.ghost_renderer.update(
                self.ghost_trajectory,
                self.ghost_stride.value(),
                self.ghost_collision_flags,
            )
        self._update_ghost_options()

    def _clear_ghost_overlay(self, source=None):
        if source is not None and self.ghost_source != source:
            return
        self.ghost_trajectory = []
        self.ghost_collision_flags = []
        self.ghost_source = None
        if self.ghost_renderer:
            self.ghost_renderer.clear()
        self._update_ghost_options()

    def _sync_playback_pose_ghosts(self):
        if self.ghost_source == "safety_candidate":
            self._update_ghost_options()
            return
        if self.show_ghosts.isChecked() and self.robot_trajectory:
            self.ghost_trajectory = [qpos.copy() for qpos in self.robot_trajectory]
            self.ghost_collision_flags = [False] * len(self.ghost_trajectory)
            self.ghost_source = "playback"
            self._rebuild_ghosts()
            return
        if self.ghost_source == "playback":
            self.ghost_trajectory = []
            self.ghost_collision_flags = []
            self.ghost_source = None
            if self.ghost_renderer:
                self.ghost_renderer.clear()
        self._update_ghost_options()

    def _update_ghost_options(self):
        visible = bool(
            self.ghost_trajectory and (
                self.ghost_source in ("preview_path", "safety_candidate")
                or (
                    self.ghost_source == "playback"
                    and self.show_ghosts.isChecked()
                )
            )
        )
        self.canvas.set_ghost_options(visible, self.ghost_alpha.value())

    def set_trajectory_frame(self, index):
        """Compatibility entry point that selects a trajectory frame by time."""
        if not self.robot_trajectory:
            return
        index = max(0, min(len(self.robot_trajectory) - 1, int(index)))
        self.set_current_time(self.robot_trajectory_times[index])

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause_playback(commit_time=True)
        else:
            self.start_playback()

    def start_playback(self):
        if not self.robot_trajectory:
            return
        for index, qpos in enumerate(self.robot_trajectory):
            limit_error = self._joint_limit_violation(qpos)
            if limit_error is not None:
                self.status_label.setText(
                    f"Playback blocked: qpos row {index + 1} has "
                    f"{limit_error}."
                )
                return
        warning_report, blocking_report = (
            adaptive_trajectory_collision_reports(
                self.robot_model,
                self.robot_trajectory,
                times=self.robot_trajectory_times,
                checker=self.collision_checker,
            )
        )
        self.robot_trajectory_warning_report = warning_report
        self.robot_trajectory_blocking_report = blocking_report
        self.timeslice_slider.set_safety_reports(
            (warning_report, blocking_report)
        )
        if blocking_report is not None:
            self.quarantine_motion(
                self.robot_trajectory,
                self.robot_trajectory_times,
                report=blocking_report,
                source="playback revalidation",
            )
            self.status_label.setText(
                "Playback blocked because the active motion no longer passes "
                f"the safety gate: {self._motion_report_text(blocking_report)}"
            )
            return
        start_time = self.robot_trajectory_times[0]
        end_time = self.robot_trajectory_times[-1]
        if end_time <= start_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
            return
        if self.display_time < start_time or self.display_time >= end_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
        self.playback_clock.start()
        self.play_timer.start()
        self._set_playback_button_text("Pause")
        self.playback_state_changed.emit(True)

    def pause_playback(self, commit_time=False):
        self.play_timer.stop()
        self.playback_clock.stop()
        self._set_playback_button_text("Play")
        self.playback_state_changed.emit(False)
        if commit_time and abs(self.display_time - self.current_time) > 1e-9:
            self.timeslice_time_changed.emit(self.display_time)

    def _set_playback_button_text(self, text):
        self.play_button.setText(text)

    def _advance_playback(self, elapsed=None):
        if not self.robot_trajectory:
            self.pause_playback()
            return
        start_time = self.robot_trajectory_times[0]
        end_time = self.robot_trajectory_times[-1]
        if end_time <= start_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
            self.pause_playback(commit_time=True)
            return

        elapsed = self.playback_clock.elapsed(
            self.play_timer.interval() / 1000.0,
            supplied=elapsed,
        )
        next_time = self.playback_clock.advance(
            self.display_time,
            start_time,
            end_time,
            elapsed,
            self.playback_speed.value(),
        )
        self.preview_trajectory_time(next_time, emit_time_signal=True)

    def _advance_frame(self):
        """Backward-compatible alias for elapsed-time playback advancement."""
        self._advance_playback()

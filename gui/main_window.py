"""
main_window.py

Purpose:
    Main GUI window for reference-frame trajectory editing.

Updated project flow:
    1. User drags/edits target frame
    2. User adds keyframe
    3. GUI stores trajectory array
    4. User clicks Generate / Simulate
    5. Backend maps robot to each target frame
"""

from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)

from core.trajectory import (
    TargetFrame,
    Trajectory,
    quat_to_rpy,
    rpy_to_quat,
)
from application import model_sessions, timeslice_service, trajectory_generation
from .controls import TrajectoryControlPanel
from gui.viewers.reference_frame_2d import RobotCanvas
from .robot_viewer_3d import RobotViewer3D
from gui.viewers.skeleton_2d import Stickman2DViewer
from gui.viewers.mujoco_player import Mujoco3DViewerPanel
from application.backend_interface import BackendInterface
from core.models import MujocoReferenceFrames
from .app_sidebars import AppLeftSidebar, AppRightSidebar
from core.models import MuJoCoRobotAdapter, ROBOT_MODELS
from application.model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)


LEFT_SIDEBAR_WIDTH = 250
RIGHT_SIDEBAR_WIDTH = 270
INITIAL_RENDER_PROGRESS_DELAY_MS = 500
MAX_HISTORY_DEPTH = 100


@dataclass(frozen=True)
class GuiHistorySnapshot:
    trajectory_frames: tuple
    trajectory_track_names: tuple
    active_index: int
    control_frame: dict
    selected_row: int
    current_time: float
    timeline_states: tuple
    committed_qpos: object
    preview_qpos: object
    preview_active: bool
    robot_trajectory: tuple
    robot_trajectory_times: tuple
    ghost_trajectory: tuple
    frame_slider_value: int
    show_ghosts: bool


@dataclass(frozen=True)
class GuiHistoryEntry:
    description: str
    snapshot: GuiHistorySnapshot


class RenderProgressOverlay(QWidget):
    """Viewer-local overlay for robot model rendering progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("renderProgressOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._allow_close = False
        if parent is not None:
            parent.installEventFilter(self)

        self.setStyleSheet(
            """
            QWidget#renderProgressOverlay {
                background: rgba(17, 24, 39, 132);
            }
            QWidget#renderProgressCard {
                background: #f3f5f8;
                border: 1px solid #9aa5b1;
                border-radius: 6px;
            }
            QLabel#renderTitle {
                color: #1f2933;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#renderDetail {
                color: #52606d;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #9aa5b1;
                border-radius: 4px;
                min-height: 16px;
                text-align: center;
                background: #e4e7eb;
            }
            QProgressBar::chunk {
                background: #2f80ed;
                border-radius: 3px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.card = QWidget(self)
        self.card.setObjectName("renderProgressCard")
        self.card.setFixedWidth(420)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(30, 24, 30, 24)
        card_layout.setSpacing(10)

        self.title_label = QLabel("Rendering robot model")
        self.title_label.setObjectName("renderTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)

        self.detail_label = QLabel("Preparing 3D geometry...")
        self.detail_label.setObjectName("renderDetail")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(300)
        self.progress_bar.setMaximumWidth(360)

        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.detail_label)
        card_layout.addWidget(
            self.progress_bar,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.card.setFixedHeight(self.card.sizeHint().height())
        layout.addStretch(1)
        layout.addWidget(self.card, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.hide()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self.update_geometry()
        return super().eventFilter(watched, event)

    def update_geometry(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.setGeometry(parent.rect())
        if not self.isHidden():
            self.raise_()

    def set_message(self, title, detail, progress=None):
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        if progress is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(progress))))

    def show_rendering(self, title, detail, progress=None):
        self._allow_close = False
        self.set_message(title, detail, progress)
        self.update_geometry()
        self.show()
        self.raise_()

    def finish(self):
        self._allow_close = True
        self.hide()

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
        else:
            event.ignore()


class ModelLoadThread(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, model_key, model_info=None, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.model_info = model_info

    def run(self):
        try:
            adapter = MuJoCoRobotAdapter(self.model_info or self.model_key)
        except Exception as exc:
            self.failed.emit(self.model_key, str(exc))
            return
        self.loaded.emit(self.model_key, adapter)


class RobotGuiMainWindow(QMainWindow):
    def __init__(self, model_key="g1"):
        super().__init__()

        self.setWindowTitle("Reference Frame Trajectory GUI")

        # --------------------------------------------------------
        # Core data
        # --------------------------------------------------------
        self.trajectory = Trajectory()
        self.active_index = -1

        # One immutable MuJoCo model is shared by FK, IK, and rendering. Each
        # subsystem owns its own MjData so live UI and batch solves stay isolated.
        self.model_library_root = default_model_library_root()
        self.model_registry = dict(ROBOT_MODELS)
        for info in discover_imported_models(self.model_library_root).values():
            self.register_model_info(info)
        self.import_mesh_folder = None
        self.model_key = model_key
        self.robot_model_3d = None
        self.robot_model_error = None
        try:
            self.robot_model_3d = MuJoCoRobotAdapter(
                self.model_registry.get(model_key, model_key)
            )
            self.robot_model_error = self.robot_model_3d.load_warning
        except Exception as exc:
            self.robot_model_error = str(exc)
            if model_key != "g1":
                failed_model_key = model_key
                model_key = "g1"
                self.model_key = model_key
                try:
                    self.robot_model_3d = MuJoCoRobotAdapter(
                        self.model_registry.get(model_key, model_key)
                    )
                    fallback_warning = self.robot_model_3d.load_warning
                    self.robot_model_error = (
                        f"Could not load {failed_model_key}: {exc}\n"
                        "Loaded Unitree G1 instead."
                    )
                    if fallback_warning:
                        self.robot_model_error += f"\n{fallback_warning}"
                except Exception as fallback_exc:
                    self.robot_model_error = (
                        f"Could not load {failed_model_key}: {exc}\n"
                        f"Fallback g1 also failed: {fallback_exc}"
                    )
        if self.robot_model_3d is not None:
            self.setWindowTitle(
                f"Reference Frame Trajectory GUI — {self.robot_model_3d.model_name}"
            )

        shared_mj_model = (
            self.robot_model_3d.mj_model if self.robot_model_3d else None
        )
        self.backend_interface = BackendInterface(
            mj_model=shared_mj_model, adapter=self.robot_model_3d
        )
        self.model_reference = MujocoReferenceFrames(
            mj_model=shared_mj_model, adapter=self.robot_model_3d
        )

        # GUI widgets
        frame_names = (
            self.robot_model_3d.trajectory_frames
            if self.robot_model_3d else ["pelvis"]
        )
        self.controls = TrajectoryControlPanel(
            self.model_registry, model_key=model_key, frame_names=frame_names
        )
        self.viewer_2d = RobotCanvas()
        self.viewer_3d = RobotViewer3D(
            robot_model=self.robot_model_3d,
            error=self.robot_model_error,
        )
        self.viewer_2d_stickman = Stickman2DViewer(self.robot_model_3d)
        self.viewer_3d_mujoco = Mujoco3DViewerPanel(self.robot_model_3d)
        self.model_sessions = {
                model_key: model_sessions.RobotModelSession(
                adapter=self.robot_model_3d,
                backend=self.backend_interface,
                reference=self.model_reference,
                viewer_3d=self.viewer_3d,
                viewer_2d_skeleton=self.viewer_2d_stickman,
                trajectory=self.trajectory,
                active_index=self.active_index,
            )
        }
        self.model_loaders = {}
        self.render_progress_overlay = None
        self.render_progress_viewer = None
        self.render_progress_restore_widget = None
        self.pending_initial_render_progress = None
        self.undo_stack = []
        self.redo_stack = []
        self._history_restoring = False
        self._last_history_snapshot = None
        self.viewer_tabs = self.build_viewer_tabs()
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.status_panel = self.build_status_panel()
        self.left_sidebar_content = AppLeftSidebar(self.controls, self.viewer_tabs)
        self.right_sidebar_content = AppRightSidebar(self.status_panel)
        self.left_sidebar = self.left_sidebar_content
        self.right_sidebar = self.right_sidebar_content
        self.left_sidebar.setMinimumWidth(LEFT_SIDEBAR_WIDTH)
        self.left_sidebar.setMaximumWidth(LEFT_SIDEBAR_WIDTH)
        self.right_sidebar.setMinimumWidth(RIGHT_SIDEBAR_WIDTH)
        self.right_sidebar.setMaximumWidth(RIGHT_SIDEBAR_WIDTH)

        self.connect_signals()
        self.install_history_shortcuts()
        self.controls.corner_smoothing_slider.value_changed.connect(
            lambda _value: self.refresh_display()
        )
        self.set_current_frame_to_model_reference(
            self.controls.frame_box.currentText(),
            emit_pose_changed=False,
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------
        # Persistent splitter children resize in place. The 3D viewer is never
        # recreated, so resizing sidebars retains its OpenGL context.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.addWidget(self.left_sidebar)
        self.main_splitter.addWidget(self.viewer_tabs)
        self.main_splitter.addWidget(self.right_sidebar)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([200, 900, 260])
        self.setCentralWidget(self.main_splitter)
        self.render_progress_overlay = RenderProgressOverlay(self.viewer_3d_stack)

        # Initial view
        if self.robot_model_3d is not None:
            self.pending_initial_render_progress = (
                f"Rendering {self.robot_model_3d.model_name}",
                "Preparing the 3D model for rendering...",
            )
        self.update_editor_context()
        self.refresh_display()
        self._refresh_history_baseline()

    def register_model_info(self, info):
        if info.key not in self.model_registry:
            self.model_registry[info.key] = info
            return info
        key = info.key
        index = 2
        while key in self.model_registry:
            key = f"{info.key}-{index}"
            index += 1
        info = replace(info, key=key)
        self.model_registry[info.key] = info
        return info

    # ============================================================
    # Build right status/debug panel
    # ============================================================

    def build_status_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(244)
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.backend_label = QLabel()
        self.backend_label.setWordWrap(True)
        self.viewer_status_label = QLabel()
        self.viewer_status_label.setWordWrap(True)
        self.viewer_time_label = QLabel()
        self.viewer_time_label.setWordWrap(True)
        self.viewer_root_pose_label = QLabel()
        self.viewer_root_pose_label.setWordWrap(True)
        self.model_source_label = QLabel(self.model_source_text(self.robot_model_3d))
        self.model_source_label.setWordWrap(True)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumWidth(0)
        self.status_text.setMaximumWidth(236)
        self.status_text.setMinimumHeight(240)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        layout.addWidget(self.backend_label)
        layout.addWidget(self.viewer_status_label)
        layout.addWidget(self.viewer_time_label)
        layout.addWidget(self.viewer_root_pose_label)
        layout.addWidget(self.model_source_label)
        layout.addWidget(self.status_text)

        panel.setLayout(layout)
        return panel

    def build_viewer_tabs(self):
        tabs = QTabWidget()
        self.viewer_3d_stack = QStackedWidget()
        self.viewer_3d_stack.addWidget(self.viewer_3d)
        self.viewer_2d_skeleton_stack = QStackedWidget()
        self.viewer_2d_skeleton_stack.addWidget(self.viewer_2d_stickman)
        tabs.addTab(self.viewer_3d_stack, "3D Pose")
        tabs.addTab(self.viewer_2d, "2D Side View")
        tabs.addTab(self.viewer_2d_skeleton_stack, "2D Skeleton")
        tabs.addTab(self.viewer_3d_mujoco, "Simulation")
        tabs.currentChanged.connect(self.update_editor_context)
        tabs.setCurrentIndex(0)
        tabs.tabBar().hide()
        return tabs

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_render_progress_overlay_geometry()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(
            INITIAL_RENDER_PROGRESS_DELAY_MS,
            self.prepare_pending_initial_render_progress,
        )

    def prepare_pending_initial_render_progress(self):
        if (
            not self.isVisible()
            or self.viewer_3d_stack.width() <= 1
            or self.viewer_3d_stack.height() <= 1
        ):
            QTimer.singleShot(50, self.prepare_pending_initial_render_progress)
            return
        if QApplication.platformName().lower() != "offscreen":
            self.raise_()
            self.activateWindow()
        QTimer.singleShot(0, self.show_pending_initial_render_progress)

    def update_render_progress_overlay_geometry(self):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.update_geometry()

    def present_render_progress_overlay(self):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.update_geometry()
        self.render_progress_overlay.show()
        self.render_progress_overlay.raise_()

    def show_pending_initial_render_progress(self):
        if self.pending_initial_render_progress is None:
            self.request_active_model_render()
            return
        title, detail = self.pending_initial_render_progress
        self.pending_initial_render_progress = None
        self.begin_render_progress(title, detail, viewer=self.viewer_3d)
        self.request_active_model_render()

    def begin_render_progress(self, title, detail="", viewer=None, progress=None):
        if self.render_progress_overlay is None:
            return
        self.render_progress_viewer = viewer
        self.render_progress_overlay.show_rendering(title, detail, progress)
        self.present_render_progress_overlay()
        QTimer.singleShot(0, self.present_render_progress_overlay)

    def update_render_progress(self, title, detail="", progress=None):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.show_rendering(title, detail, progress)
        self.present_render_progress_overlay()

    def finish_render_progress(self):
        if self.render_progress_overlay is not None:
            self.render_progress_overlay.finish()
        self.render_progress_viewer = None
        if self.render_progress_restore_widget is not None:
            restore_widget = self.render_progress_restore_widget
            self.render_progress_restore_widget = None
            if self.viewer_tabs.indexOf(restore_widget) >= 0:
                self.viewer_tabs.setCurrentWidget(restore_widget)
        else:
            self.render_progress_restore_widget = None

    def active_3d_geometry_ready(self):
        viewer = self.viewer_3d
        canvas = viewer.canvas
        if viewer.robot_state is None:
            return True
        if not canvas.isValid():
            return False
        if canvas._geometry_queue:
            return False
        return canvas._geometry_build_count > 0

    def request_active_model_render(self):
        if (
            self.render_progress_overlay is None
            or self.render_progress_overlay.isHidden()
        ):
            return
        self.render_progress_viewer = self.viewer_3d
        if self.active_3d_geometry_ready():
            QTimer.singleShot(0, self.finish_render_progress)
            return
        if not self.isVisible():
            return
        if self.viewer_tabs.currentWidget() is not self.viewer_3d_stack:
            self.render_progress_restore_widget = self.viewer_tabs.currentWidget()
            self.viewer_tabs.setCurrentWidget(self.viewer_3d_stack)
        self.viewer_3d.canvas.update()

    def on_viewer_geometry_progress(self, viewer, complete, total):
        if viewer is not self.viewer_3d:
            return
        if (
            self.render_progress_overlay is not None
            and not self.render_progress_overlay.isHidden()
            and self.render_progress_viewer not in (None, viewer)
        ):
            return
        if total <= 0:
            self.finish_render_progress()
            return

        progress = round((float(complete) / float(total)) * 100.0)
        model_name = (
            self.robot_model_3d.model_name
            if self.robot_model_3d is not None
            else "robot model"
        )
        self.update_render_progress(
            f"Rendering {model_name}",
            f"Building 3D geometry {complete}/{total}...",
            progress=progress,
        )
        self.render_progress_viewer = viewer
        if complete >= total:
            self.finish_render_progress()

    def model_source_text(self, adapter):
        path = getattr(adapter, "model_path", None)
        if path is None:
            return "Model source: unavailable"
        return f"Model source: {path}"

    def update_editor_context(self, index=None):
        active = self.viewer_tabs.currentWidget()
        if active is self.viewer_3d_stack:
            self.controls.set_robot_context_widget(
                self.viewer_3d.robot_context_widget()
            )
            self.controls.set_selection_context_widget(
                self.viewer_3d.selection_context_widget()
            )
            self.controls.set_trajectory_context_widget(
                self.viewer_3d.trajectory_context_widget()
            )
            self.controls.set_display_context_widget(
                self.viewer_3d.display_context_widget()
            )
            self.viewer_3d.set_trajectory_lines_widget(
                self.controls.show_lines_box
            )
            self.controls.set_preview_ik_context_widget(
                self.viewer_3d.preview_ik_context_widget()
            )
            self.sync_viewer_status_panel()
        else:
            self.controls.set_robot_context_widget(None)
            self.controls.set_selection_context_widget(None)
            self.controls.set_trajectory_context_widget(None)
            self.controls.set_display_context_widget(None)
            self.controls.set_preview_ik_context_widget(None)

    # ============================================================
    # Signal connections
    # ============================================================

    def connect_signals(self):
        self.controls.model_changed.connect(self.on_model_changed)
        self.controls.open_model_clicked.connect(self.on_open_model_file)
        self.controls.choose_mesh_folder_clicked.connect(self.on_choose_mesh_folder)
        self.controls.pose_changed.connect(self.on_pose_changed)
        self.controls.add_keyframe_clicked.connect(self.on_add_keyframe)
        self.controls.update_keyframe_clicked.connect(self.on_update_keyframe)
        self.controls.delete_keyframe_clicked.connect(self.on_delete_keyframe)
        self.controls.clear_trajectory_clicked.connect(self.on_clear_trajectory)
        self.controls.generate_clicked.connect(self.on_generate_trajectory)
        self.controls.keyframe_selected.connect(self.on_keyframe_selected)
        self.controls.frame_name_changed.connect(self.on_frame_name_changed)
        self.controls.trajectory_lines_changed.connect(
            self.on_trajectory_lines_changed
        )
        self.controls.time_changed.connect(self.on_time_changed)

        self.viewer_2d.target_dragged.connect(self.on_target_dragged)
        self.connect_model_viewer_signals(self.viewer_3d, self.viewer_2d_stickman)

    def install_history_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_action)

        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.redo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.redo_shortcut.activated.connect(self.redo_last_action)

    def connect_model_viewer_signals(self, viewer_3d, viewer_2d_skeleton):
        viewer_3d.target_dragged.connect(self.on_target_dragged)
        viewer_3d.target_pose_dragged.connect(self.on_target_pose_dragged)
        viewer_3d.target_pose_drag_finished.connect(
            self.on_target_pose_drag_finished
        )
        viewer_3d.target_frame_changed.connect(self.on_3d_target_frame_changed)
        viewer_3d.preview_cancelled.connect(self.on_preview_cancelled)
        viewer_3d.trajectory_csv_loaded.connect(self.on_trajectory_csv_loaded)
        viewer_3d.generate_requested.connect(self.on_generate_trajectory)
        viewer_3d.clear_trajectory_requested.connect(self.on_clear_trajectory)
        viewer_3d.canvas.geometry_progress.connect(
            lambda complete, total, viewer=viewer_3d: (
                self.on_viewer_geometry_progress(viewer, complete, total)
            )
        )
        viewer_3d.timeslice_time_changed.connect(
            self.on_viewer_timeslice_time_changed
        )
        viewer_3d.accept_timeslice_requested.connect(
            self.on_accept_timeslice_requested
        )
        viewer_3d.delete_timeslice_requested.connect(
            self.on_delete_timeslice_requested
        )
        viewer_3d.history_action_finished.connect(
            self.on_viewer_history_action_finished
        )
        viewer_3d.status_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_status_changed(
                viewer, text
            )
        )
        viewer_3d.timeline_state_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_time_changed(
                viewer, text
            )
        )
        viewer_3d.root_pose_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_root_pose_changed(
                viewer, text
            )
        )
        viewer_2d_skeleton.target_dragged.connect(self.on_target_dragged)

    def capture_history_snapshot(self):
        viewer = self.viewer_3d
        timeline_states = ()
        if viewer.state_timeline is not None:
            timeline_states = tuple(
                (time, viewer.state_timeline.get_state(time))
                for time in viewer.state_timeline.times()
            )

        committed_qpos = (
            None if viewer.committed_state is None
            else viewer.committed_state.get_qpos()
        )
        preview_qpos = (
            None if viewer.preview_state is None
            else viewer.preview_state.get_qpos()
        )

        return GuiHistorySnapshot(
            trajectory_frames=tuple(
                frame.to_dict() for frame in self.trajectory.frames
            ),
            trajectory_track_names=tuple(self.trajectory.tracks.keys()),
            active_index=int(self.active_index),
            control_frame=self.controls.current_frame().to_dict(),
            selected_row=int(self.controls.selected_row()),
            current_time=float(viewer.get_current_time()),
            timeline_states=timeline_states,
            committed_qpos=committed_qpos,
            preview_qpos=preview_qpos,
            preview_active=bool(viewer.preview_active),
            robot_trajectory=tuple(
                qpos.copy() for qpos in viewer.robot_trajectory
            ),
            robot_trajectory_times=tuple(float(t) for t in viewer.robot_trajectory_times),
            ghost_trajectory=tuple(qpos.copy() for qpos in viewer.ghost_trajectory),
            frame_slider_value=int(viewer.frame_slider.value()),
            show_ghosts=bool(viewer.show_ghosts.isChecked()),
        )

    def _restore_control_frame(self, frame):
        controls = self.controls
        controls._suppress_pose_changed = True
        try:
            controls.time_slider.set_value(frame.time)
            controls.x_slider.set_value(frame.x)
            controls.y_slider.set_value(frame.y)
            controls.z_slider.set_value(frame.z)
            controls.roll_slider.set_value(frame.roll)
            controls.pitch_slider.set_value(frame.pitch)
            controls.yaw_slider.set_value(frame.yaw)
            controls.phase_box.setCurrentText(frame.phase)
            blocked = controls.frame_box.blockSignals(True)
            controls.frame_box.setCurrentText(frame.frame_name)
            controls.frame_box.blockSignals(blocked)
        finally:
            controls._suppress_pose_changed = False

    def restore_history_snapshot(self, snapshot):
        viewer = self.viewer_3d
        self._history_restoring = True
        try:
            self.trajectory.tracks = {
                name: [] for name in snapshot.trajectory_track_names
            }
            for frame_data in snapshot.trajectory_frames:
                self.trajectory.add_frame(TargetFrame.from_dict(dict(frame_data)))
            self.active_index = snapshot.active_index

            if viewer.robot_state is not None:
                viewer.pause_playback()
                viewer.canvas.cancel_transform_drag()
                if snapshot.robot_trajectory:
                    viewer.set_robot_trajectory(
                        snapshot.robot_trajectory,
                        times=snapshot.robot_trajectory_times,
                        activate_first_frame=False,
                    )
                else:
                    viewer.clear_robot_trajectory()
                viewer.frame_slider.setValue(snapshot.frame_slider_value)
                viewer.ghost_trajectory = [
                    qpos.copy() for qpos in snapshot.ghost_trajectory
                ]
                viewer._rebuild_ghosts()

                if viewer.state_timeline is not None:
                    viewer.state_timeline.states.clear()
                    for time, qpos in snapshot.timeline_states:
                        viewer.state_timeline.set_state(time, qpos)
                    if not snapshot.timeline_states and snapshot.committed_qpos is not None:
                        viewer.state_timeline.set_state(
                            snapshot.current_time, snapshot.committed_qpos
                        )
                    viewer.current_time = viewer.state_timeline.time_key(
                        snapshot.current_time
                    )
                    viewer._set_timeslice_widgets(viewer.current_time)
                    viewer._update_timeline_label()

                if snapshot.committed_qpos is not None:
                    viewer.committed_state.set_qpos(snapshot.committed_qpos)
                if snapshot.preview_qpos is not None:
                    viewer.preview_state.set_qpos(snapshot.preview_qpos)
                viewer.preview_active = snapshot.preview_active
                viewer.canvas.set_preview_visible(snapshot.preview_active)
                viewer.show_ghosts.setChecked(snapshot.show_ghosts)
                viewer._sync_joint_controls()
                viewer._set_target_to_selected_pose()

            control_frame = TargetFrame.from_dict(dict(snapshot.control_frame))
            self._restore_control_frame(control_frame)
            self.refresh_display(apply_stickman_frame=False)
            if 0 <= snapshot.selected_row < self.controls.table.rowCount():
                self.controls.table.setCurrentCell(snapshot.selected_row, 0)
            else:
                self.controls.table.clearSelection()
        finally:
            self._history_restoring = False

    def _refresh_history_baseline(self):
        if self._history_restoring:
            return
        self._last_history_snapshot = self.capture_history_snapshot()

    def record_history_action(self, description):
        if self._history_restoring:
            return False
        before = self._last_history_snapshot or self.capture_history_snapshot()
        after = self.capture_history_snapshot()
        self.undo_stack.append(GuiHistoryEntry(description, before))
        if len(self.undo_stack) > MAX_HISTORY_DEPTH:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._last_history_snapshot = after
        self.statusBar().showMessage(f"{description}; Ctrl+Z can undo.", 3000)
        return True

    def undo_last_action(self):
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo.", 2000)
            return
        current = self.capture_history_snapshot()
        entry = self.undo_stack.pop()
        self.redo_stack.append(GuiHistoryEntry(entry.description, current))
        self.restore_history_snapshot(entry.snapshot)
        self._last_history_snapshot = entry.snapshot
        self.statusBar().showMessage(f"Undid {entry.description}.", 3000)

    def redo_last_action(self):
        if not self.redo_stack:
            self.statusBar().showMessage("Nothing to redo.", 2000)
            return
        current = self.capture_history_snapshot()
        entry = self.redo_stack.pop()
        self.undo_stack.append(GuiHistoryEntry(entry.description, current))
        self.restore_history_snapshot(entry.snapshot)
        self._last_history_snapshot = entry.snapshot
        self.statusBar().showMessage(f"Redid {entry.description}.", 3000)

    def on_viewer_history_action_finished(self, description):
        self.record_history_action(description)

    def on_viewer_status_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_status_label.setText(f"Status: {text}")

    def on_viewer_time_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_time_label.setText(text.replace("3D state time:", "Time:"))

    def on_viewer_root_pose_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_root_pose_label.setText(f"Root: {text}")

    def sync_viewer_status_panel(self):
        self.on_viewer_status_changed(
            self.viewer_3d, self.viewer_3d.status_label.text()
        )
        self.on_viewer_time_changed(
            self.viewer_3d, self.viewer_3d.timeline_state_label.text()
        )
        self.on_viewer_root_pose_changed(
            self.viewer_3d, self.viewer_3d.root_pose_label.text()
        )

    def on_model_changed(self, model_key):
        """Swap model-owned widgets while retaining the surrounding app."""
        if model_key == self.model_key:
            return
        model_info = self.model_registry.get(model_key)
        if model_info is None:
            self.status_text.setText(f"Unknown robot model: {model_key}")
            return
        cached = self.model_sessions.get(model_key)
        if cached is not None:
            self.activate_model_session(model_key, cached)
            return
        if model_key in self.model_loaders:
            return
        self.begin_render_progress(
            f"Loading {model_info.display_name}",
            "Loading robot model data...",
        )
        self.controls.model_box.setEnabled(False)
        self.statusBar().showMessage(f"Loading {model_info.display_name}…")
        loader = ModelLoadThread(model_key, model_info, self)
        loader.loaded.connect(self.on_model_loaded)
        loader.failed.connect(self.on_model_load_failed)
        loader.finished.connect(loader.deleteLater)
        self.model_loaders[model_key] = loader
        loader.start()

    def on_model_loaded(self, model_key, adapter):
        self.model_loaders.pop(model_key, None)
        backend = BackendInterface(mj_model=adapter.mj_model, adapter=adapter)
        reference = MujocoReferenceFrames(adapter=adapter)
        viewer_3d = RobotViewer3D(adapter, adapter.load_warning)
        viewer_2d_skeleton = Stickman2DViewer(adapter)
        self.connect_model_viewer_signals(viewer_3d, viewer_2d_skeleton)
        self.viewer_3d_stack.addWidget(viewer_3d)
        self.viewer_2d_skeleton_stack.addWidget(viewer_2d_skeleton)
        session = model_sessions.RobotModelSession(
            adapter, backend, reference, viewer_3d, viewer_2d_skeleton,
            Trajectory(), -1,
        )
        self.model_sessions[model_key] = session
        self.finish_model_loading_ui()
        self.activate_model_session(model_key, session)

    def on_model_load_failed(self, model_key, error):
        self.model_loaders.pop(model_key, None)
        self.finish_model_loading_ui()
        self.finish_render_progress()
        self.status_text.setText(f"Could not load {model_key}: {error}")
        index = self.controls.model_box.findData(self.model_key)
        self.controls.model_box.blockSignals(True)
        self.controls.model_box.setCurrentIndex(index)
        self.controls.model_box.blockSignals(False)

    def on_open_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open robot model",
            str(Path.home()),
            "Robot model files (*.urdf *.xml)",
        )
        if not path:
            return
        self.import_model_file(path)

    def on_choose_mesh_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Mesh Folder (.stl)",
            str(Path.home()),
        )
        if not path:
            return
        self.import_mesh_folder = Path(path).expanduser().resolve()
        self.status_text.append(f"Mesh folder: {self.import_mesh_folder}")

    def _prompt_for_mesh_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Mesh Folder (.stl)",
            str(Path.home()),
        )
        if not path:
            return None
        self.import_mesh_folder = Path(path).expanduser().resolve()
        return self.import_mesh_folder

    def import_model_file(self, path):
        mesh_roots = [self.import_mesh_folder] if self.import_mesh_folder else []
        self.begin_render_progress(
            "Importing robot model",
            f"Copying and preparing {Path(path).name}...",
        )
        QApplication.processEvents()
        try:
            try:
                info = import_robot_model(
                    path, self.model_library_root, mesh_roots=mesh_roots
                )
            except RuntimeError:
                self.finish_render_progress()
                mesh_folder = self._prompt_for_mesh_folder()
                if mesh_folder is None:
                    raise
                self.begin_render_progress(
                    "Importing robot model",
                    f"Copying and preparing {Path(path).name}...",
                )
                QApplication.processEvents()
                info = import_robot_model(
                    path, self.model_library_root, mesh_roots=[mesh_folder]
                )
            info = self.register_model_info(info)
        except Exception as exc:
            self.finish_render_progress()
            message = f"Could not import model: {exc}"
            self.status_text.setText(message)
            QMessageBox.warning(self, "Import model failed", message)
            return
        self.status_text.append(
            f"Imported {info.display_name} to {info.model_path}"
        )
        self.controls.add_model(info.key, info.display_name, select=True)

    def finish_model_loading_ui(self):
        self.controls.model_box.setEnabled(True)
        self.statusBar().clearMessage()

    def activate_model_session(self, model_key, session):
        model_sessions.remember_current_session(
            self.model_sessions, self.model_key, self.trajectory, self.active_index
        )
        for name, value in model_sessions.activated_session_state(
            model_key, session
        ).items():
            setattr(self, name, value)
        self.viewer_3d_stack.setCurrentWidget(self.viewer_3d)
        self.viewer_2d_skeleton_stack.setCurrentWidget(self.viewer_2d_stickman)
        self.viewer_3d_mujoco.set_model_adapter(session.adapter)
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.model_source_label.setText(self.model_source_text(session.adapter))
        self.begin_render_progress(
            f"Rendering {session.adapter.model_name}",
            "Preparing the 3D model geometry...",
            viewer=self.viewer_3d,
        )
        self.controls.set_frame_names(session.adapter.trajectory_frames)
        self.setWindowTitle(
            f"Reference Frame Trajectory GUI — {session.adapter.model_name}"
        )
        self.update_editor_context()
        self.refresh_display(apply_stickman_frame=False)
        self.request_active_model_render()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_history_baseline()

    def on_trajectory_csv_loaded(self, csv_path):
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)
        count = self.import_loaded_robot_trajectory_as_keyframes()
        import_dt = self.viewer_3d.trajectory_import_dt.value()
        self.status_text.setText(
            f"Loaded trajectory CSV: {csv_path}\n"
            f"Imported {count} editable target-frame keyframes from FK "
            f"at {import_dt:.2f} s intervals."
        )
        self.record_history_action("Load trajectory")

    # ============================================================
    # GUI interaction callbacks
    # ============================================================

    def on_pose_changed(self, x, y, z, roll, pitch, yaw):
        """
        Called when sliders change.

        If a keyframe is selected, we only preview the target frame.
        The committed keyframe is overwritten only when the preview is accepted.
        """

        frame_name = self.controls.frame_box.currentText()
        self.viewer_3d.preview_target_pose(
            frame_name,
            (x, y, z),
            rpy_to_quat(roll, pitch, yaw),
        )
        self.refresh_display()

    def on_time_changed(self, time):
        """Load or create the editable qpos keyframe for this GUI time."""
        frame_name = self.controls.frame_box.currentText()
        target = self.trajectory.targets_at_time(time).get(frame_name)
        if target is not None:
            self.controls.set_position_values(
                x=target.x,
                y=target.y,
                z=target.z,
                roll=target.roll,
                pitch=target.pitch,
                yaw=target.yaw,
                emit_pose_changed=False,
            )
        self.refresh_display()
        self.viewer_3d.set_current_time(time)
        self._refresh_history_baseline()

    def on_viewer_timeslice_time_changed(self, time):
        """Keep the sidebar time editor in sync with the viewer-bottom scrubber."""
        self.controls.time_slider.set_value(time)
        self.on_time_changed(time)

    def on_accept_timeslice_requested(self):
        if self.viewer_3d.preview_active and not self.viewer_3d.accept_preview():
            return

        count = self.define_timeslice_from_committed_pose()
        time = self.viewer_3d.get_current_time()
        if count <= 0:
            message = f"No logical target frames were available at t={time:.2f} s."
            self.viewer_3d.status_label.setText(message)
            self.status_text.setText(message)
            return

        self.refresh_display()
        next_time = self.viewer_3d.next_timeslice_time(time)
        if abs(next_time - time) > 1e-9:
            self.on_viewer_timeslice_time_changed(next_time)
            advance_note = f" advanced to t={next_time:.2f} s."
        else:
            advance_note = " already at the timeline end."
        message = (
            f"Accepted slice at t={time:.2f} s; captured {count} logical targets "
            f"from the committed solved pose;{advance_note}"
        )
        self.viewer_3d.status_label.setText(message)
        self.status_text.setText(message)
        self.record_history_action("Accept slice")

    def on_delete_timeslice_requested(self):
        time = self.viewer_3d.get_current_time()
        deleted_targets = self.delete_timeslice_at_time(time)
        deleted_qpos = False
        if self.viewer_3d.state_timeline is not None:
            deleted_qpos = self.viewer_3d.state_timeline.delete_state(time)

        if deleted_targets <= 0 and not deleted_qpos:
            message = f"No defined slice found at t={time:.2f} s."
            self.viewer_3d.status_label.setText(message)
            self.status_text.setText(message)
            return

        self.active_index = -1
        next_time = time
        if deleted_qpos and self.viewer_3d.state_timeline is not None:
            remaining_times = self.viewer_3d.state_timeline.times()
            if remaining_times:
                next_time = min(
                    remaining_times, key=lambda candidate: abs(candidate - time)
                )
        self.controls.time_slider.set_value(next_time)
        self.viewer_3d.set_current_time(next_time)
        self.refresh_display()
        parts = []
        if deleted_targets > 0:
            parts.append(f"{deleted_targets} target frames")
        if deleted_qpos:
            parts.append("robot qpos")
        detail = " and ".join(parts)
        message = f"Deleted slice at t={time:.2f} s ({detail})."
        self.viewer_3d.status_label.setText(message)
        self.status_text.setText(message)
        self.record_history_action("Delete slice")

    def define_timeslice_from_committed_pose(self):
        """Snapshot every editable logical target from the committed MuJoCo pose."""
        result = timeslice_service.define_timeslice_from_committed_pose(
            self.trajectory,
            self.viewer_3d.committed_state,
            time=self.viewer_3d.get_current_time(),
            phase=self.controls.phase_box.currentText(),
            selected_frame_name=self.controls.frame_box.currentText(),
            frame_names=self.editable_logical_frame_names(),
            frame_bindings=self.viewer_3d.frame_bindings,
        )
        self.active_index = result.active_index
        if result.selected_frame is not None:
            selected_frame = result.selected_frame
            self.controls.set_position_values(
                x=selected_frame.x,
                y=selected_frame.y,
                z=selected_frame.z,
                roll=selected_frame.roll,
                pitch=selected_frame.pitch,
                yaw=selected_frame.yaw,
                emit_pose_changed=False,
            )
        return result.count

    def editable_logical_frame_names(self):
        return timeslice_service.editable_logical_frame_names(
            self.robot_model_3d, self.controls, self.viewer_3d
        )

    def import_loaded_robot_trajectory_as_keyframes(self):
        """Convert loaded qpos playback rows into editable logical target frames."""
        qposes = list(getattr(self.viewer_3d, "robot_trajectory", []))
        times = list(getattr(self.viewer_3d, "robot_trajectory_times", []))
        if not qposes:
            return 0

        state = self.viewer_3d.committed_state
        if state is None:
            return 0

        self.trajectory.clear()
        self.active_index = -1
        phase = self.controls.phase_box.currentText()
        selected_frame_name = self.controls.frame_box.currentText()
        selected_frame = None
        selected_index = -1
        last_index = -1
        count = 0

        frame_names = self.editable_logical_frame_names()
        import_samples = self.selected_loaded_trajectory_import_samples(times, qposes)
        for time, qpos in import_samples:
            state.set_qpos(qpos)
            for frame_name in frame_names:
                binding = self.viewer_3d.frame_bindings.get(frame_name)
                if binding is None:
                    continue
                kind, object_name = binding
                try:
                    position, quaternion = state.get_body_pose(object_name, kind)
                except KeyError:
                    continue
                roll, pitch, yaw = quat_to_rpy(quaternion)
                frame = TargetFrame(
                    time=float(time),
                    phase=phase,
                    frame_name=frame_name,
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                )
                last_index = self.trajectory.upsert_frame(frame)
                if (
                    selected_frame is None
                    and frame_name == selected_frame_name
                    and abs(float(time) - self.viewer_3d.get_current_time()) <= 1e-6
                ):
                    selected_frame = frame
                    selected_index = last_index
                count += 1

        if qposes:
            state.set_qpos(qposes[0])
            self.viewer_3d.preview_state.set_qpos(qposes[0])
            self.viewer_3d.preview_active = False
            self.viewer_3d.canvas.set_preview_visible(False)
            self.controls.time_slider.set_value(float(times[0]))

        self.active_index = selected_index if selected_index >= 0 else last_index
        if selected_frame is not None:
            self.controls.set_position_values(
                x=selected_frame.x,
                y=selected_frame.y,
                z=selected_frame.z,
                roll=selected_frame.roll,
                pitch=selected_frame.pitch,
                yaw=selected_frame.yaw,
                emit_pose_changed=False,
            )
        imported_times = [time for time, _ in import_samples]
        self.viewer_3d.set_defined_timeslices(imported_times)
        self.refresh_display()
        return count

    def selected_loaded_trajectory_import_samples(self, times, qposes):
        return timeslice_service.selected_loaded_trajectory_import_samples(
            times, qposes, self.viewer_3d.trajectory_import_dt.value()
        )

    def delete_timeslice_at_time(self, time, tolerance=1e-6):
        return timeslice_service.delete_timeslice_at_time(
            self.trajectory, time, tolerance
        )

    def on_target_dragged(self, x, z):
        """
        Called when user drags the red reference frame in the viewer.

        This updates the sliders, so the GUI stays consistent.
        """

        self.controls.set_position_from_viewer(x, z)

    def on_target_pose_dragged(
        self, x, y, z, roll=None, pitch=None, yaw=None
    ):
        """Sync controls without repainting every viewer on every mouse event."""
        # A full refresh here previously scheduled the 2D, 3D, stickman, table,
        # and status panel repeatedly during one drag. The live canvas already
        # updated its transforms; refresh the rest once on mouse release.
        self.controls.set_position_values(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )

    def on_target_pose_drag_finished(
        self, x, y, z, roll=None, pitch=None, yaw=None
    ):
        self.controls.set_position_values(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )
        # A completed 3D edit is an intentional keyframe edit. Upsert the
        # selected logical target at the active time while RobotViewer3D stores
        # the corresponding accepted qpos in its time-keyed state timeline.
        frame = self.controls.current_frame()
        self.active_index = self.trajectory.upsert_frame(frame)
        self.refresh_display()
        description = (
            getattr(self.viewer_3d, "_pending_history_action_description", None)
            or "Accept pose"
        )
        self.viewer_3d._pending_history_action_description = None
        self.record_history_action(description)

    def on_3d_target_frame_changed(self, frame_name):
        """Map common 3D body/site selections back to the 2D frame concept."""
        self.controls.frame_box.blockSignals(True)
        self.controls.frame_box.setCurrentText(frame_name)
        self.controls.frame_box.blockSignals(False)
        binding = self.robot_model_3d.resolve_logical_frame(frame_name)
        if binding is not None:
            kind, name = binding
            state = (
                self.viewer_3d.preview_state
                if self.viewer_3d.preview_active
                else self.viewer_3d.committed_state
            )
            position, quaternion = state.get_body_pose(name, kind)
            roll, pitch, yaw = quat_to_rpy(quaternion)
            self.controls.set_position_values(
                x=float(position[0]), y=float(position[1]), z=float(position[2]),
                roll=roll, pitch=pitch, yaw=yaw,
                emit_pose_changed=False,
            )
        self.refresh_display(apply_stickman_frame=False)

    def on_preview_cancelled(self):
        kind, name = self.viewer_3d._selected_target()
        if not name:
            return
        position, quaternion = self.viewer_3d.committed_state.get_body_pose(
            name, kind
        )
        roll, pitch, yaw = quat_to_rpy(quaternion)
        self.controls.set_position_values(
            x=float(position[0]), y=float(position[1]), z=float(position[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )
        self.refresh_display(apply_stickman_frame=False)

    def on_trajectory_lines_changed(self, checked):
        self.refresh_display()

    def on_add_keyframe(self):
        """
        Add the currently edited target frame to the trajectory array.
        """

        frame = self.controls.current_frame()
        self.active_index = self.trajectory.add_frame(frame)

        self.refresh_display()
        self.record_history_action("Add keyframe")

    def on_update_keyframe(self):
        """
        Replace selected keyframe with current editor values.
        """

        row = self.controls.selected_row()

        if row < 0:
            self.status_text.append("No keyframe selected to update.")
            return

        frame = self.controls.current_frame()
        self.trajectory.update_frame(row, frame)

        self.active_index = row
        self.refresh_display()
        self.record_history_action("Update keyframe")

    def on_delete_keyframe(self):
        """
        Delete selected keyframe.
        """

        row = self.controls.selected_row()

        if row < 0:
            self.status_text.append("No keyframe selected to delete.")
            return

        self.trajectory.delete_frame(row)
        self.active_index = -1

        self.refresh_display()
        self.record_history_action("Delete keyframe")

    def on_clear_trajectory(self):
        keyframe_count = len(self.trajectory.frames)
        if keyframe_count == 0:
            self.status_text.append("Trajectory is already empty.")
            return

        response = QMessageBox.question(
            self,
            "Clear trajectory",
            f"Delete all {keyframe_count} trajectory keyframes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            self.status_text.append("Clear trajectory cancelled.")
            return

        self.trajectory.clear()
        self.active_index = -1
        self.viewer_3d.clear_robot_trajectory()
        self.viewer_3d.clear_editable_timeline(
            keep_current_pose=True,
            reset_time=0.0,
        )
        self.controls.time_slider.set_value(0.0)
        self.viewer_3d.set_defined_timeslices([])
        self.refresh_display()
        message = (
            f"Cleared {keyframe_count} trajectory keyframes; "
            "current robot pose was left unchanged."
        )
        self.viewer_3d.status_label.setText(message)
        self.status_text.setText(message)
        self.record_history_action("Clear trajectory")

    def on_keyframe_selected(self, row):
        """
        Load selected keyframe into the editor.
        """

        if row < 0 or row >= len(self.trajectory.frames):
            return

        self.active_index = row
        frame = self.trajectory.frames[row]

        self.controls.set_from_frame(frame)
        self.refresh_display()
        self._refresh_history_baseline()

    def on_frame_name_changed(self, frame_name):
        """
        Called when user changes the selected target robot frame.

        Example:
            pelvis -> left_foot

        The target controls should jump to the current real MuJoCo body/site
        position. The simplified stickman is still drawn as a 2D helper, but
        target-frame defaults come from the actual robot model.
        """

        if self.set_current_frame_to_model_reference(
            frame_name,
            emit_pose_changed=False,
        ):
            self.refresh_display(apply_stickman_frame=False)
            self._refresh_history_baseline()
            return

        x, z = self.viewer_2d_stickman.get_body_point(frame_name)

        self.controls.set_position_from_viewer(
            x,
            z,
            emit_pose_changed=False,
        )

        self.refresh_display(apply_stickman_frame=False)
        self._refresh_history_baseline()

    def set_current_frame_to_model_reference(
        self,
        frame_name,
        emit_pose_changed=True,
    ):
        pose = self.model_reference.pose_for_frame(frame_name)

        if pose is None:
            return False

        position, quaternion = pose
        x, y, z = position
        roll, pitch, yaw = quat_to_rpy(quaternion)
        self.controls.set_position_values(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            emit_pose_changed=emit_pose_changed,
        )
        return True

    def on_generate_trajectory(self):
        if len(self.trajectory.frames) == 0:
            self.status_text.setText("Trajectory is empty. Add keyframes first.")
            return

        result = trajectory_generation.generate_trajectory_status(
            self.trajectory,
            self.backend_interface,
            smoothing=self.controls.corner_smoothing(),
        )
        self.viewer_3d.load_backend_states(result.result_states)
        self.viewer_3d_mujoco.set_trajectory_csv(result.csv_path)
        self.status_text.setText(result.status_text)
        self.record_history_action("Generate trajectory")

    # ============================================================
    # Display update
    # ============================================================

    def refresh_display(self, apply_stickman_frame=True):
        """
        Refresh viewer, table, and status text.
        """

        active_frame = self.controls.current_frame()
        show_trajectory_lines = self.controls.show_trajectory_lines()
        trajectory_smoothing = self.controls.corner_smoothing()

        self.viewer_2d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
        )
        self.viewer_3d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
        )
        self.viewer_2d_stickman.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            apply_active_frame=apply_stickman_frame,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
        )

        self.controls.refresh_table(self.trajectory)
        self.viewer_3d.set_defined_timeslices(
            sorted({frame.time for frame in self.trajectory.frames})
        )

        self.backend_label.setText(
            f"Backend: {self.backend_interface.backend_name()}"
        )
        self.sync_viewer_status_panel()

        summary = []
        summary.append(f"Number of keyframes: {len(self.trajectory.frames)}")
        summary.append("")
        summary.append("Current edited target frame:")
        summary.append(f"time: {active_frame.time:.2f} s")
        summary.append(f"phase: {active_frame.phase}")
        summary.append(f"frame: {active_frame.frame_name}")
        summary.append(
            f"position: x={active_frame.x:.2f}, "
            f"y={active_frame.y:.2f}, "
            f"z={active_frame.z:.2f}"
        )
        summary.append(
            f"orientation: roll={active_frame.roll:.2f}, "
            f"pitch={active_frame.pitch:.2f}, "
            f"yaw={active_frame.yaw:.2f}"
        )

        self.status_text.setText("\n".join(summary))

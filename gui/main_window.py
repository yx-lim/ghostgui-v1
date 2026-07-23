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

import copy
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
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
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QMenu,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidgetAction,
)

from core.trajectory import (
    TargetFrame,
    Trajectory,
    quat_to_rpy,
    rpy_to_quat,
)
from core.scene import Scene, SceneRuntime, Transform, TransformKeyframe
from application import model_sessions, timeslice_service, trajectory_generation
from application.model_resources import ModelResourcePool
from application.project_manager import (
    GhostGUIProject,
    available_default_project_root_from_name,
    forget_recent_project,
    load_project_browser_previews,
    load_recent_projects,
    remember_recent_project,
)
from .controls import TrajectoryControlPanel
from gui.viewers.reference_frame_2d import RobotCanvas
from .robot_viewer_3d import RobotViewer3D
from gui.viewers.skeleton_2d import Stickman2DViewer
from gui.viewers.mujoco_player import Mujoco3DViewerPanel
from application.backend_interface import BackendInterface
from core.models import MujocoReferenceFrames
from .app_sidebars import AppLeftSidebar, AppRightSidebar
from .help import HelpCenterDialog
from .project_browser import ProjectBrowserDialog
from .theme import ensure_application_theme
from .tutorial import TutorialManager
from core.models import MuJoCoRobotAdapter, ROBOT_MODELS, RobotStateTimeline
from application.model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)
from application.scene_object_importer import SUPPORTED_OBJECT_MESH_EXTENSIONS


LEFT_SIDEBAR_WIDTH = 250
RIGHT_SIDEBAR_WIDTH = 270
INITIAL_RENDER_PROGRESS_DELAY_MS = 500
PROJECT_AUTOSAVE_INTERVAL_MS = 30000
MAX_HISTORY_DEPTH = 100
MODEL_IO_LOCK = threading.Lock()


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
    ghost_source: str | None
    frame_slider_value: int
    show_ghosts: bool
    timeline_duration: float
    scene_state: dict


@dataclass(frozen=True)
class GuiHistoryEntry:
    description: str
    snapshot: GuiHistorySnapshot


class ObjectMeshImportDialog(QDialog):
    def __init__(self, mesh_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Mesh Object")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit(Path(mesh_path).stem)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 1000.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(1.0)

        form.addRow("Name", self.name_edit)
        form.addRow("Scale", self.scale_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def object_name(self):
        return self.name_edit.text().strip() or "Object"

    def uniform_scale(self):
        value = float(self.scale_spin.value())
        return [value, value, value]


class RenderProgressOverlay(QWidget):
    """Viewer-local overlay for robot model rendering progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("renderProgressOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._allow_close = False
        if parent is not None:
            parent.installEventFilter(self)

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
            with MODEL_IO_LOCK:
                adapter = MuJoCoRobotAdapter(self.model_info or self.model_key)
        except Exception as exc:
            self.failed.emit(self.model_key, str(exc))
            return
        self.loaded.emit(self.model_key, adapter)


class ModelImportThread(QThread):
    imported = Signal(object)
    failed = Signal(str, bool)

    def __init__(self, source_path, library_root, mesh_roots=None, parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.library_root = library_root
        self.mesh_roots = list(mesh_roots or [])

    def run(self):
        try:
            with MODEL_IO_LOCK:
                info = import_robot_model(
                    self.source_path,
                    self.library_root,
                    mesh_roots=self.mesh_roots,
                )
        except Exception as exc:
            self.failed.emit(str(exc), isinstance(exc, RuntimeError))
            return
        self.imported.emit(info)


class RobotGuiMainWindow(QMainWindow):
    def __init__(self, model_key="g1"):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            ensure_application_theme(app)

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
        self._scene_runtime_model_path_cache = {}
        self.model_resources = ModelResourcePool()
        self._scene_robot_state_cache = {}
        self._scene_robot_timeline_cache = {}
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
            self.robot_model_3d = self.model_resources.register(self.robot_model_3d)
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
        self.scene = Scene.single_robot(
            self.model_key,
            model_name=(
                self.robot_model_3d.model_name
                if self.robot_model_3d is not None
                else self.model_key
            ),
            trajectory=self.trajectory,
        )
        self.editor_robot_actor_id = self.scene.active_robot_id()
        self.scene.metadata["editor_robot_actor_id"] = self.editor_robot_actor_id
        self.scene_runtime = SceneRuntime(self.scene)

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
        # Release B owns exactly one OpenGL widget. Actor/model sessions retain
        # editor and timeline state, then bind that state to this canvas when
        # activated.
        self.shared_scene_canvas = self.viewer_3d.canvas
        self.viewer_2d_stickman = Stickman2DViewer(self.robot_model_3d)
        self.viewer_3d_mujoco = Mujoco3DViewerPanel(self.robot_model_3d)
        initial_session = model_sessions.RobotModelSession(
                adapter=self.robot_model_3d,
                backend=self.backend_interface,
                reference=self.model_reference,
                viewer_3d=self.viewer_3d,
                viewer_2d_skeleton=self.viewer_2d_stickman,
                trajectory=self.trajectory,
                active_index=self.active_index,
                actor_id=self.editor_robot_actor_id,
                model_key=self.model_key,
                selected_frame=self.controls.frame_box.currentText(),
            )
        self.model_sessions = {
            model_sessions.session_key(self.editor_robot_actor_id, self.model_key):
                initial_session
        }
        self.model_loaders = {}
        self._model_loader_actor_ids = {}
        self._pending_scene_robot_loads = {}
        self._pending_added_robot_ids = set()
        self.model_importer = None
        self._model_import_retry_prompted = False
        self._close_confirmed = False
        self._close_when_workers_finish = False
        self._background_workers = set()
        self.current_project = None
        self._pending_project_restore = None
        self._pending_project_restore_autosave = False
        self._pending_project_restore_source = "direct"
        self.project_dirty = False
        self._suppress_project_dirty = False
        self.render_progress_overlay = None
        self.render_progress_viewer = None
        self.render_progress_restore_widget = None
        self.pending_initial_render_progress = None
        self.undo_stack = []
        self.redo_stack = []
        self._history_restoring = False
        self._syncing_editor_selection = False
        self._last_history_snapshot = None
        self.viewer_tabs = self.build_viewer_tabs()
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.project_panel = self.build_project_panel()
        self.help_dialog = None
        self.help_button = self.build_help_button()
        self.app_toolbar = self.build_app_toolbar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.app_toolbar)
        self.refresh_recent_projects()
        self.status_panel = self.build_status_panel()
        self.scene_panel = self.build_scene_panel()
        self.left_sidebar_content = AppLeftSidebar(
            self.controls,
            include_view=False,
        )
        inspector_sections = [("Scene", self.scene_panel, True)]
        inspector_sections.extend(self.controls.inspector_sections())
        self.right_sidebar_content = AppRightSidebar(
            self.status_panel, inspector_sections
        )
        self.left_sidebar = self.left_sidebar_content
        self.right_sidebar = self.right_sidebar_content
        self.left_sidebar.setMinimumWidth(LEFT_SIDEBAR_WIDTH)
        self.left_sidebar.setMaximumWidth(LEFT_SIDEBAR_WIDTH)
        self.right_sidebar.setMinimumWidth(RIGHT_SIDEBAR_WIDTH)
        self.right_sidebar.setMaximumWidth(RIGHT_SIDEBAR_WIDTH)

        self.connect_signals()
        self.install_history_shortcuts()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(PROJECT_AUTOSAVE_INTERVAL_MS)
        self.autosave_timer.timeout.connect(self.on_autosave_timer)
        self.autosave_timer.start()
        self.controls.corner_smoothing_slider.value_changed.connect(
            lambda _value: self.refresh_display()
        )
        self.controls.corner_smoothing_slider.value_changed.connect(
            lambda _value: self.mark_project_dirty("Smoothing")
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
        self.tutorial_manager = TutorialManager(self)

        # Initial view
        if self.robot_model_3d is not None:
            self.pending_initial_render_progress = (
                f"Rendering {self.robot_model_3d.model_name}",
                "Preparing the 3D model for rendering...",
            )
        self.update_editor_context()
        self.refresh_scene_tree()
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
    # Build project controls
    # ============================================================

    def build_project_panel(self):
        panel = QWidget()
        panel.setObjectName("projectMenuPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.project_name_label = QLabel("No project open")
        self.project_name_label.setObjectName("projectNameLabel")
        self.project_name_label.setMinimumWidth(180)
        self.project_name_label.setMaximumWidth(240)
        self.project_name_label.setWordWrap(True)
        layout.addWidget(self.project_name_label)

        self.recent_projects_box = QComboBox()
        self.recent_projects_box.setObjectName("recentProjectsCombo")
        self.recent_projects_box.setMinimumWidth(180)
        self.recent_projects_box.setMaximumWidth(240)
        self.recent_projects_box.setToolTip("Open a recently used GhostGUI project.")
        self.recent_projects_box.activated.connect(self.on_recent_project_selected)
        layout.addWidget(self.recent_projects_box)
        return panel

    def build_toolbar_dropdown(self, text, object_name):
        button = QToolButton()
        button.setObjectName(object_name)
        button.setText(text)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(text, button)
        button.setMenu(menu)
        return button, menu

    def build_robot_menu_panel(self):
        panel = QWidget()
        panel.setObjectName("robotMenuPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.controls.model_box.setMinimumWidth(180)
        self.controls.model_box.setMaximumWidth(240)
        layout.addWidget(self.controls.model_box)
        return panel

    def build_app_toolbar(self):
        toolbar = QToolBar("App", self)
        toolbar.setObjectName("appToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.project_toolbar_button, self.project_toolbar_menu = (
            self.build_toolbar_dropdown("Project", "projectToolbarButton")
        )
        self.new_project_action = self.project_toolbar_menu.addAction("New")
        self.new_project_action.setObjectName("newProjectAction")
        self.new_project_action.setToolTip("Create a GhostGUI project folder.")
        self.new_project_action.triggered.connect(
            lambda checked=False: self.on_new_project()
        )
        self.open_project_action = self.project_toolbar_menu.addAction("Open")
        self.open_project_action.setObjectName("openProjectAction")
        self.open_project_action.setToolTip("Show GhostGUI project previews.")
        self.open_project_action.triggered.connect(
            lambda checked=False: self.on_open_project()
        )
        self.save_project_action = self.project_toolbar_menu.addAction("Save")
        self.save_project_action.setObjectName("saveProjectAction")
        self.save_project_action.setToolTip("Save the current GhostGUI project.")
        self.save_project_action.triggered.connect(
            lambda checked=False: self.on_save_project()
        )
        self.project_toolbar_menu.addSeparator()
        self.project_panel_action = QWidgetAction(self.project_toolbar_menu)
        self.project_panel_action.setDefaultWidget(self.project_panel)
        self.project_toolbar_menu.addAction(self.project_panel_action)
        self.project_toolbar_button.setToolTip("No GhostGUI project is open.")
        toolbar.addWidget(self.project_toolbar_button)
        toolbar.addSeparator()

        if hasattr(self.controls, "model_box"):
            self.robot_toolbar_button, self.robot_toolbar_menu = (
                self.build_toolbar_dropdown("Robot", "robotToolbarButton")
            )
            self.robot_menu_panel = self.build_robot_menu_panel()
            self.robot_panel_action = QWidgetAction(self.robot_toolbar_menu)
            self.robot_panel_action.setDefaultWidget(self.robot_menu_panel)
            self.robot_toolbar_menu.addAction(self.robot_panel_action)
            toolbar.addWidget(self.robot_toolbar_button)
            toolbar.addSeparator()

        self.import_toolbar_button, self.import_toolbar_menu = (
            self.build_toolbar_dropdown("Import", "importToolbarButton")
        )
        if hasattr(self.controls, "import_action_box"):
            self.import_actions = {}
            for label, action_key in (
                ("Model", "model"),
                ("Qpos", "qpos"),
                ("Trajectory", "trajectory"),
            ):
                action = self.import_toolbar_menu.addAction(label)
                action.setObjectName(f"import{label}Action")
                action.setData(action_key)
                action.triggered.connect(
                    lambda checked=False, key=action_key: (
                        self.on_setup_import_requested(key)
                    )
                )
                self.import_actions[action_key] = action
            toolbar.addWidget(self.import_toolbar_button)

        self.export_toolbar_button, self.export_toolbar_menu = (
            self.build_toolbar_dropdown("Export", "exportToolbarButton")
        )
        if hasattr(self.controls, "export_action_box"):
            self.export_actions = {}
            for label, action_key in (
                ("Qpos", "qpos"),
                ("Trajectory", "trajectory"),
            ):
                action = self.export_toolbar_menu.addAction(label)
                action.setObjectName(f"export{label}Action")
                action.setData(action_key)
                action.triggered.connect(
                    lambda checked=False, key=action_key: (
                        self.on_setup_export_requested(key)
                    )
                )
                self.export_actions[action_key] = action
            toolbar.addWidget(self.export_toolbar_button)
            toolbar.addSeparator()

        self.view_toolbar_button, self.view_toolbar_menu = (
            self.build_toolbar_dropdown("View", "viewToolbarButton")
        )
        self.view_action_group = QActionGroup(self)
        self.view_action_group.setExclusive(True)
        self.view_actions = []

        for index in range(self.viewer_tabs.count()):
            action = QAction(self.viewer_tabs.tabText(index), self)
            action.setCheckable(True)
            action.setData(index)
            action.triggered.connect(
                lambda checked=False, tab_index=index: (
                    self.viewer_tabs.setCurrentIndex(tab_index)
                )
            )
            self.view_action_group.addAction(action)
            self.view_toolbar_menu.addAction(action)
            self.view_actions.append(action)

        self.view_toolbar_menu.addSeparator()
        self.controls.view_panel.setObjectName("toolbarViewPanel")
        self.view_panel_action = QWidgetAction(self.view_toolbar_menu)
        self.view_panel_action.setDefaultWidget(self.controls.view_panel)
        self.view_toolbar_menu.addAction(self.view_panel_action)
        self.viewer_tabs.currentChanged.connect(self.sync_view_toolbar_actions)
        self.sync_view_toolbar_actions(self.viewer_tabs.currentIndex())

        toolbar.addWidget(self.view_toolbar_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.help_button)
        return toolbar

    def sync_view_toolbar_actions(self, active_index):
        for index, action in enumerate(getattr(self, "view_actions", [])):
            action.setChecked(index == active_index)

    # ============================================================
    # Build right status/debug panel
    # ============================================================

    def build_status_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(244)
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.backend_label = QLabel()
        self.backend_label.setWordWrap(True)
        self.viewer_status_label = QLabel()
        self.viewer_status_label.setWordWrap(True)
        self.status_frame_label = QLabel("-")
        self.status_frame_label.setWordWrap(True)
        self.status_ik_label = QLabel("-")
        self.status_ik_label.setWordWrap(True)
        self.status_move_label = QLabel("-")
        self.status_move_label.setWordWrap(True)
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
        self.status_text.setMinimumHeight(120)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        def add_summary_row(name, value_widget):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            label = QLabel(name)
            label.setMinimumWidth(42)
            row.addWidget(label)
            row.addWidget(value_widget, stretch=1)
            layout.addLayout(row)

        add_summary_row("State", self.viewer_status_label)
        add_summary_row("Frame", self.status_frame_label)
        add_summary_row("IK", self.status_ik_label)
        add_summary_row("Move", self.status_move_label)
        add_summary_row("Root", self.viewer_root_pose_label)

        self.status_details_button = QToolButton()
        self.status_details_button.setText("Details")
        self.status_details_button.setCheckable(True)
        self.status_details_button.setChecked(False)
        self.status_details_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.status_details_button.setArrowType(Qt.ArrowType.RightArrow)

        self.status_details_panel = QWidget()
        details_layout = QVBoxLayout(self.status_details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        details_layout.addWidget(self.status_text)
        self.status_details_panel.setVisible(False)

        def set_details_visible(visible):
            self.status_details_panel.setVisible(visible)
            self.status_details_button.setArrowType(
                Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
            )

        self.status_details_button.toggled.connect(set_details_visible)
        layout.addWidget(self.status_details_button)
        layout.addWidget(self.status_details_panel)

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
        tabs.currentChanged.connect(lambda _index: self.mark_project_dirty("Active view"))
        tabs.setCurrentIndex(0)
        tabs.tabBar().hide()
        return tabs

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_render_progress_overlay_geometry()
        self.position_help_button()

    def showEvent(self, event):
        super().showEvent(event)
        self.position_help_button()
        QTimer.singleShot(
            INITIAL_RENDER_PROGRESS_DELAY_MS,
            self.prepare_pending_initial_render_progress,
        )

    def closeEvent(self, event):
        if (
            not self._close_confirmed
            and not self.confirm_project_transition("close GhostGUI")
        ):
            event.ignore()
            return
        self._close_confirmed = True
        workers = [worker for worker in self._background_workers if worker.isRunning()]
        if workers:
            event.ignore()
            self._close_when_workers_finish = True
            self.statusBar().showMessage(
                "Finishing background model work before closing…"
            )
            return
        super().closeEvent(event)

    def _close_after_workers_finish(self):
        if not self._close_when_workers_finish:
            return
        workers = [worker for worker in self._background_workers if worker.isRunning()]
        if workers:
            return
        self._close_when_workers_finish = False
        QTimer.singleShot(0, self.close)

    def _background_worker_finished(self, worker):
        self._background_workers.discard(worker)
        self._close_after_workers_finish()

    def on_autosave_timer(self):
        try:
            self.autosave_current_project(show_status=False, reason="timer")
        except (OSError, ValueError) as exc:
            self.status_text.append(f"Project autosave failed: {exc}")

    def current_model_display_name(self):
        if self.robot_model_3d is not None:
            return self.robot_model_3d.model_name
        model_info = self.model_registry.get(self.model_key)
        return getattr(model_info, "display_name", self.model_key)

    def current_robot_session_key(self):
        return model_sessions.session_key(self.editor_robot_actor_id, self.model_key)

    def current_robot_session(self):
        return self.model_sessions.get(self.current_robot_session_key())

    def sync_current_robot_session(self):
        actor = self.scene.actors.get(getattr(self, "editor_robot_actor_id", None))
        if actor is None or actor.kind != "robot":
            return None
        session = self.current_robot_session()
        if session is not None:
            session.trajectory = self.trajectory
            session.active_index = self.active_index
            session.selected_frame = self.controls.frame_box.currentText()
        actor.metadata["selected_frame"] = self.controls.frame_box.currentText()
        self.scene.tracks.set_robot_trajectory(actor.id, self.trajectory)
        viewer = self.viewer_3d
        if viewer.state_timeline is not None:
            if viewer.committed_state is not None:
                viewer.state_timeline.set_state(
                    viewer.get_current_time(),
                    viewer.committed_state.get_qpos(),
                )
            self.scene.tracks.set_robot_qpos_timeline(
                actor.id,
                viewer.state_timeline,
            )
        return actor

    def rebind_current_robot_session(self, actor, reset_sessions=False):
        session = model_sessions.RobotModelSession(
            adapter=self.robot_model_3d,
            backend=self.backend_interface,
            reference=self.model_reference,
            viewer_3d=self.viewer_3d,
            viewer_2d_skeleton=self.viewer_2d_stickman,
            trajectory=self.trajectory,
            active_index=self.active_index,
            actor_id=actor.id,
            model_key=self.model_key,
            selected_frame=actor.metadata.get("selected_frame"),
        )
        if reset_sessions:
            seen_sessions = set()
            for existing in self.model_sessions.values():
                if id(existing) in seen_sessions:
                    continue
                seen_sessions.add(id(existing))
                if existing.viewer_3d is not self.viewer_3d:
                    self.viewer_3d_stack.removeWidget(existing.viewer_3d)
                    existing.viewer_3d.deleteLater()
                if existing.viewer_2d_skeleton is not self.viewer_2d_stickman:
                    self.viewer_2d_skeleton_stack.removeWidget(
                        existing.viewer_2d_skeleton
                    )
                    existing.viewer_2d_skeleton.deleteLater()
            self.model_sessions.clear()
            self._scene_robot_state_cache.clear()
            self._scene_robot_timeline_cache.clear()
        self.model_sessions[
            model_sessions.session_key(actor.id, self.model_key)
        ] = session
        self.scene.tracks.load_robot_qpos_timeline(
            actor.id,
            self.viewer_3d.state_timeline,
        )
        return session

    def ensure_scene_active_robot(self):
        if not hasattr(self, "scene") or self.scene is None:
            self.scene = Scene.single_robot(
                self.model_key,
                model_name=self.current_model_display_name(),
                trajectory=self.trajectory,
            )
        actor = self.scene.actors.get(
            getattr(self, "editor_robot_actor_id", None)
            or self.scene.metadata.get("editor_robot_actor_id")
        )
        if actor is None or actor.kind != "robot":
            robots = self.scene.actors.robots()
            actor = robots[0] if robots else None
        if actor is None:
            actor = self.scene.add_robot(
                self.model_key,
                model_name=self.current_model_display_name(),
            )
        self.editor_robot_actor_id = actor.id
        self.scene.metadata["editor_robot_actor_id"] = actor.id
        if self.scene.selection.actor_id not in self.scene.actors.actors:
            self.scene.selection.actor_id = actor.id
        return actor

    def set_scene_active_robot_model(self, model_key, model_name=None):
        actor = self.ensure_scene_active_robot()
        actor.name = model_name or model_key
        actor.model_reference = {
            "type": "robot_model",
            "model_key": str(model_key),
            "model_name": str(model_name or model_key),
        }
        self.scene.metadata["active_model_key"] = str(model_key)
        self.scene_runtime = SceneRuntime(self.scene)
        return actor

    def sync_scene_from_current_robot(self):
        actor = self.set_scene_active_robot_model(
            self.model_key,
            self.current_model_display_name(),
        )
        self.sync_current_robot_session()
        viewer = self.viewer_3d
        self.scene.timeline.current_time = float(viewer.get_current_time())
        self.scene.timeline.duration = float(viewer.timeline_duration)
        if self.scene.selection.actor_id not in self.scene.actors.actors:
            self.scene.selection.actor_id = actor.id
        if self.scene.selection.actor_id == actor.id:
            self.scene.selection.frame_id = self.controls.frame_box.currentText()
        self.scene_runtime = SceneRuntime(self.scene)
        return self.scene

    def capture_project_scene(self):
        return self.sync_scene_from_current_robot().to_dict()

    def restore_scene(self, scene_data, trajectory_data=None, workspace=None):
        try:
            scene = (
                Scene.from_dict(scene_data)
                if isinstance(scene_data, dict)
                else Scene.from_legacy(
                    self.model_key,
                    model_name=self.current_model_display_name(),
                    trajectory_data=trajectory_data,
                    workspace=workspace,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Project scene data is invalid: {exc}") from exc

        if scene.active_robot_id() is None:
            scene.add_robot(self.model_key, model_name=self.current_model_display_name())
        self.editor_robot_actor_id = (
            scene.metadata.get("editor_robot_actor_id")
            or scene.active_robot_id()
        )
        active_robot_id = self.ensure_scene_editor_robot_id(scene)
        if (
            active_robot_id is not None
            and not scene.tracks.robot_targets.get(active_robot_id)
            and trajectory_data is not None
        ):
            scene.tracks.set_robot_tracks_from_dict(active_robot_id, trajectory_data)
        self.scene = scene
        self.scene_runtime = SceneRuntime(scene)
        self.trajectory = scene.tracks.robot_trajectory(active_robot_id)
        self.set_scene_active_robot_model(
            self.model_key,
            self.current_model_display_name(),
        )
        actor = self.scene.actors.require(active_robot_id)
        self.rebind_current_robot_session(actor, reset_sessions=True)
        return scene

    def ensure_scene_editor_robot_id(self, scene):
        actor = scene.actors.get(getattr(self, "editor_robot_actor_id", None))
        if actor is None or actor.kind != "robot":
            robots = scene.actors.robots()
            actor = robots[0] if robots else None
        if actor is None:
            actor = scene.add_robot(
                self.model_key,
                model_name=self.current_model_display_name(),
            )
        self.editor_robot_actor_id = actor.id
        scene.metadata["editor_robot_actor_id"] = actor.id
        return actor.id

    def scene_project_model_key(self, project, autosave=False):
        try:
            scene_data = project.read_scene_dict(autosave=autosave)
            scene = Scene.from_dict(scene_data)
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return None
        actor_id = scene.metadata.get("editor_robot_actor_id")
        actor = scene.actors.get(actor_id) if actor_id else scene.active_robot()
        if actor is None:
            return None
        return actor.model_reference.get("model_key")

    def add_scene_object(
        self,
        name="Object",
        shape="box",
        size=None,
        color=None,
        transform=None,
    ):
        transform = transform or Transform.identity()
        actor = self.scene.add_object(
            name=name,
            shape=shape,
            size=size,
            color=color,
            transform=transform,
        )
        self.scene.select_actor(actor.id)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Add object")
        return actor

    def add_scene_mesh_object(
        self,
        name="Object",
        model_reference=None,
        transform=None,
    ):
        reference = dict(model_reference or {})
        actor = self.scene.add_mesh_object(
            name=name,
            asset_path=reference.get("asset_path"),
            mesh_format=reference.get("mesh_format"),
            scale=reference.get("scale"),
            color=reference.get("rgba"),
            transform=transform or Transform.identity(),
            metadata={
                key: reference[key]
                for key in ("source_name",)
                if key in reference
            },
        )
        self.scene.select_actor(actor.id)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Import mesh object")
        return actor

    def add_scene_robot(self, model_key=None, model_name=None):
        model_key = model_key or self.model_key
        model_name = model_name or self.current_model_display_name()
        offset_index = max(1, len(self.scene.actors.robots()))
        actor = self.scene.add_robot(model_key, model_name=model_name)
        actor.world_transform = Transform(
            position=(0.80 * float(offset_index), 0.0, 0.0)
        )
        self._pending_added_robot_ids.add(actor.id)
        model_info = self.model_registry.get(model_key)
        adapter = self.model_resources.get(model_info=model_info)
        if adapter is not None:
            self.create_robot_actor_session(actor, adapter, model_key=model_key)
            self._pending_added_robot_ids.discard(actor.id)
            self.select_scene_actor(actor.id)
            self.record_history_action("Add robot actor")
            return actor

        self.scene.select_actor(actor.id)
        self._pending_scene_robot_loads[actor.id] = str(model_key)
        self.refresh_display(apply_stickman_frame=False)
        self.request_scene_robot_load(actor)
        return actor

    def duplicate_scene_actor(self, actor_id):
        if actor_id == self.editor_robot_actor_id:
            self.sync_current_robot_session()
        actor = self.scene.duplicate_actor(actor_id)
        self.select_scene_actor(actor.id)
        self.record_history_action("Duplicate actor")
        return actor

    def delete_scene_actor(self, actor_id):
        actor = self.scene.actors.require(actor_id)
        if (
            actor.kind == "robot"
            and actor.id == self.scene.active_robot_id()
            and len(self.scene.actors.robots()) <= 1
        ):
            raise ValueError("Cannot delete the only active robot actor.")
        deleting_editor = actor.id == self.editor_robot_actor_id
        if deleting_editor:
            self.sync_current_robot_session()
        actor = self.scene.delete_actor(actor_id)
        self._pending_scene_robot_loads.pop(actor.id, None)
        self._pending_added_robot_ids.discard(actor.id)
        if deleting_editor:
            replacement = self.scene.active_robot()
            if replacement is not None:
                self.activate_scene_robot_actor(replacement.id)
        self.discard_robot_actor_sessions(actor.id)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Delete actor")
        return actor

    def discard_robot_actor_sessions(self, actor_id):
        actor_id = str(actor_id)
        for key, session in list(self.model_sessions.items()):
            if not isinstance(key, tuple) or key[0] != actor_id:
                continue
            self.model_sessions.pop(key, None)
            if session.viewer_3d is not self.viewer_3d:
                self.viewer_3d_stack.removeWidget(session.viewer_3d)
                session.viewer_3d.deleteLater()
            if session.viewer_2d_skeleton is not self.viewer_2d_stickman:
                self.viewer_2d_skeleton_stack.removeWidget(
                    session.viewer_2d_skeleton
                )
                session.viewer_2d_skeleton.deleteLater()
        for key in list(self._scene_robot_state_cache):
            if key[0] == actor_id:
                self._scene_robot_state_cache.pop(key, None)
                self._scene_robot_timeline_cache.pop(key, None)

    def set_scene_actor_visibility(self, actor_id, visible):
        actor = self.scene.set_actor_visibility(actor_id, visible)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Set actor visibility")
        return actor

    def set_scene_actor_locked(self, actor_id, locked):
        actor = self.scene.set_actor_locked(actor_id, locked)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Set actor lock")
        return actor

    def set_scene_object_transform_keyframe(self, actor_id, time, transform):
        keyframe = self.scene.set_object_transform_keyframe(actor_id, time, transform)
        self.refresh_display(apply_stickman_frame=False)
        self.record_history_action("Set object keyframe")
        return keyframe

    def attach_scene_frames(
        self,
        source_actor_id,
        source_frame_id,
        target_actor_id,
        target_frame_id,
    ):
        constraint = self.scene.attach(
            source_actor_id,
            source_frame_id,
            target_actor_id,
            target_frame_id,
        )
        self.record_history_action("Attach actors")
        return constraint

    def scene_robot_model_path(self, actor):
        adapter = self.scene_robot_adapter(actor)
        if adapter is None:
            reference = actor.model_reference or {}
            model_key = reference.get("model_key")
            model_info = self.model_registry.get(model_key)
            model_path = reference.get("model_path") or getattr(
                model_info, "model_path", None
            )
            if model_path is None:
                raise ValueError(f"Robot actor {actor.name!r} is still loading.")
            return str(model_path)
        return {
            "runtime_model_path": adapter.runtime_model_path,
            "logical_frame_bindings": dict(adapter.logical_frame_bindings),
        }

    def scene_robot_adapter(self, actor):
        reference = actor.model_reference or {}
        model_key = reference.get("model_key")
        model_info = self.model_registry.get(model_key)
        if model_info is None:
            model_path = reference.get("model_path")
            if not model_path:
                return None
            return self.model_resources.get(model_path=model_path)
        return self.model_resources.get(model_info=model_info)

    def robot_actor_model_key(self, actor):
        reference = actor.model_reference or {}
        return str(reference.get("model_key") or self.model_key)

    def ensure_robot_actor_session(self, actor):
        model_key = self.robot_actor_model_key(actor)
        key = model_sessions.session_key(actor.id, model_key)
        session = self.model_sessions.get(key)
        if session is not None:
            return session

        adapter = self.scene_robot_adapter(actor)
        if adapter is None:
            self.request_scene_robot_load(actor)
            return None
        return self.create_robot_actor_session(actor, adapter, model_key=model_key)

    def create_robot_actor_session(
        self,
        actor,
        adapter,
        model_key=None,
        trajectory=None,
        load_scene_timeline=True,
    ):
        model_key = str(model_key or self.robot_actor_model_key(actor))
        key = model_sessions.session_key(actor.id, model_key)
        existing = self.model_sessions.get(key)
        if existing is not None:
            return existing
        backend = BackendInterface(mj_model=adapter.mj_model, adapter=adapter)
        reference = MujocoReferenceFrames(adapter=adapter)
        viewer_3d = RobotViewer3D(
            adapter,
            adapter.load_warning,
            create_canvas=False,
        )
        detail_index = viewer_3d.secondary_detail_box.findData(
            self.shared_scene_canvas.secondary_robot_detail
        )
        if detail_index >= 0:
            viewer_3d.secondary_detail_box.setCurrentIndex(detail_index)
        viewer_2d_skeleton = Stickman2DViewer(adapter)
        self.connect_model_viewer_signals(viewer_3d, viewer_2d_skeleton)
        self.viewer_3d_stack.addWidget(viewer_3d)
        self.viewer_2d_skeleton_stack.addWidget(viewer_2d_skeleton)
        trajectory = trajectory or self.scene.tracks.robot_trajectory(actor.id)
        session = model_sessions.RobotModelSession(
            adapter=adapter,
            backend=backend,
            reference=reference,
            viewer_3d=viewer_3d,
            viewer_2d_skeleton=viewer_2d_skeleton,
            trajectory=trajectory,
            active_index=-1,
            actor_id=actor.id,
            model_key=model_key,
            selected_frame=actor.metadata.get("selected_frame"),
        )
        if load_scene_timeline:
            self.scene.tracks.load_robot_qpos_timeline(
                actor.id,
                viewer_3d.state_timeline,
            )
        current_time = float(self.scene.timeline.current_time)
        if viewer_3d.state_timeline is not None:
            qpos = viewer_3d.state_timeline.ensure_state(current_time)
            viewer_3d.current_time = viewer_3d.state_timeline.time_key(current_time)
            viewer_3d.set_robot_state_for_current_time(qpos)
        self.model_sessions[key] = session
        return session

    def request_scene_robot_load(self, actor):
        if actor is None or actor.kind != "robot":
            return None
        if self.scene.actors.get(actor.id) is None:
            return None
        model_key = self.robot_actor_model_key(actor)
        model_info = self.model_registry.get(model_key)
        if model_info is None:
            self.on_model_load_failed(
                model_key,
                f"Robot actor {actor.name!r} has no registered model.",
                actor_ids={actor.id},
            )
            return None

        adapter = self.model_resources.get(model_info=model_info)
        if adapter is not None:
            session = self.create_robot_actor_session(
                actor, adapter, model_key=model_key
            )
            self._pending_scene_robot_loads.pop(actor.id, None)
            return session

        self._pending_scene_robot_loads[actor.id] = model_key
        if model_key in self.model_loaders:
            self._model_loader_actor_ids.setdefault(model_key, set()).add(actor.id)
            return self.model_loaders[model_key]

        self.begin_render_progress(
            f"Loading {model_info.display_name}",
            "Loading robot model data in the background...",
        )
        self.statusBar().showMessage(f"Loading {model_info.display_name}…")
        loader = ModelLoadThread(model_key, model_info, self)
        loader.loaded.connect(self.on_model_loaded)
        loader.failed.connect(self.on_model_load_failed)
        self._background_workers.add(loader)
        loader.finished.connect(
            lambda loader=loader: self._background_worker_finished(loader)
        )
        loader.finished.connect(loader.deleteLater)
        self.model_loaders[model_key] = loader
        self._model_loader_actor_ids[model_key] = {actor.id}
        loader.start()
        return loader

    def scene_robot_render_states(self):
        states = {}
        time = float(self.scene.timeline.current_time)
        for actor in self.scene.actors.robots():
            if actor.id == self.editor_robot_actor_id or not actor.visible:
                continue
            key = model_sessions.session_key(
                actor.id,
                self.robot_actor_model_key(actor),
            )
            session = self.model_sessions.get(key)
            if session is not None and session.viewer_3d.committed_state is not None:
                timeline = session.viewer_3d.state_timeline
                if timeline is not None:
                    session.viewer_3d.committed_state.set_qpos(
                        timeline.sample_state(time)
                    )
                states[actor.id] = session.viewer_3d.committed_state
                continue
            cache_key = (actor.id, self.robot_actor_model_key(actor))
            state = self._scene_robot_state_cache.get(cache_key)
            render_timeline = self._scene_robot_timeline_cache.get(cache_key)
            if state is None:
                adapter = self.scene_robot_adapter(actor)
                if adapter is None:
                    self.request_scene_robot_load(actor)
                    continue
                state = adapter.create_state()
                render_timeline = RobotStateTimeline(adapter)
                self.scene.tracks.load_robot_qpos_timeline(
                    actor.id,
                    render_timeline,
                )
                self._scene_robot_state_cache[cache_key] = state
                self._scene_robot_timeline_cache[cache_key] = render_timeline
            if render_timeline is not None:
                state.set_qpos(render_timeline.sample_state(time))
            states[actor.id] = state
        return states

    def scene_extra_robot_adapters(self):
        adapters = {}
        for actor in self.scene.actors.robots():
            if actor.id == self.editor_robot_actor_id:
                continue
            if not actor.visible:
                continue
            try:
                adapter = self.scene_robot_adapter(actor)
            except Exception as exc:
                self.status_text.append(
                    f"Could not render robot actor {actor.name}: {exc}"
                )
                continue
            if adapter is None:
                self.request_scene_robot_load(actor)
                continue
            if adapter is not None:
                adapters[actor.id] = adapter
        return adapters

    def scene_edit_actor_id(self):
        actor = self.scene.actors.get(self.scene.selection.actor_id)
        if actor is None or actor.kind != "object" or actor.locked:
            return None
        return actor.id

    def scene_edit_target(self):
        actor = self.scene.actors.get(self.scene.selection.actor_id)
        if actor is None or actor.locked or not actor.visible:
            return None
        if actor.kind == "object":
            return {"kind": "object", "actor_id": actor.id}
        if actor.kind == "robot" and actor.id == self.editor_robot_actor_id:
            frame_id = (
                self.scene.selection.frame_id
                or actor.metadata.get("selected_frame")
                or self.controls.frame_box.currentText()
            )
            return {
                "kind": "robot",
                "actor_id": actor.id,
                "frame_id": frame_id,
            }
        return None

    def build_composed_scene_model(self, time=None):
        project_root = (
            None if self.current_project is None
            else self.current_project.root_dir
        )
        return self.scene_runtime.compile_model(
            project_root=project_root,
            robot_model_resolver=self.scene_robot_model_path,
            time=time,
        )

    def available_scene_robot_choices(self):
        choices = []
        for key, info in sorted(
            self.model_registry.items(),
            key=lambda item: getattr(item[1], "display_name", item[0]).lower(),
        ):
            display_name = getattr(info, "display_name", key)
            choices.append((f"{display_name} ({key})", key, display_name))
        return choices

    def build_scene_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.scene_tree = QTreeWidget()
        self.scene_tree.setObjectName("sceneTree")
        self.scene_tree.setHeaderHidden(True)
        self.scene_tree.setColumnCount(1)
        self.scene_tree.setMinimumHeight(120)
        self.scene_tree.setMaximumHeight(190)
        self.scene_tree.itemSelectionChanged.connect(
            self.on_scene_tree_selection_changed
        )
        layout.addWidget(self.scene_tree)

        add_button_row = QHBoxLayout()
        add_button_row.setContentsMargins(0, 0, 0, 0)
        add_button_row.setSpacing(4)
        self.add_object_button = QPushButton("Add")
        self.import_object_button = QPushButton("Import")
        self.add_robot_button = QPushButton("Robot")
        self.duplicate_actor_button = QPushButton("Duplicate")
        self.delete_actor_button = QPushButton("Delete")
        for button in (
            self.add_object_button,
            self.import_object_button,
            self.add_robot_button,
            self.duplicate_actor_button,
            self.delete_actor_button,
        ):
            button.setMinimumWidth(0)
        self.add_object_button.setToolTip("Add a movable box object.")
        self.import_object_button.setToolTip("Import a mesh object into the project.")
        self.add_robot_button.setToolTip("Add another robot actor to the scene.")
        self.duplicate_actor_button.setToolTip("Duplicate the selected actor.")
        self.delete_actor_button.setToolTip("Delete the selected object or extra robot.")
        self.add_object_button.clicked.connect(self.on_add_scene_object_clicked)
        self.import_object_button.clicked.connect(self.on_import_scene_object_clicked)
        self.add_robot_button.clicked.connect(self.on_add_scene_robot_clicked)
        self.duplicate_actor_button.clicked.connect(
            self.on_duplicate_scene_actor_clicked
        )
        self.delete_actor_button.clicked.connect(self.on_delete_scene_actor_clicked)
        add_button_row.addWidget(self.add_object_button)
        add_button_row.addWidget(self.import_object_button)
        add_button_row.addWidget(self.add_robot_button)
        layout.addLayout(add_button_row)

        actor_button_row = QHBoxLayout()
        actor_button_row.setContentsMargins(0, 0, 0, 0)
        actor_button_row.setSpacing(4)
        actor_button_row.addWidget(self.duplicate_actor_button)
        actor_button_row.addWidget(self.delete_actor_button)
        layout.addLayout(actor_button_row)

        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(8)
        self.scene_visible_box = QCheckBox("Visible")
        self.scene_locked_box = QCheckBox("Locked")
        self.scene_visible_box.toggled.connect(self.on_scene_visible_toggled)
        self.scene_locked_box.toggled.connect(self.on_scene_locked_toggled)
        toggle_row.addWidget(self.scene_visible_box)
        toggle_row.addWidget(self.scene_locked_box)
        layout.addLayout(toggle_row)
        self._syncing_scene_panel = False
        return panel

    def selected_scene_actor_id(self):
        item = getattr(self, "scene_tree", None).currentItem()
        if item is None:
            return self.scene.selection.actor_id
        actor_id = item.data(0, Qt.ItemDataRole.UserRole)
        return str(actor_id) if actor_id else None

    def refresh_scene_tree(self):
        if not hasattr(self, "scene_tree"):
            return
        self._syncing_scene_panel = True
        try:
            tree = self.scene_tree
            tree_blocked = tree.blockSignals(True)
            tree.clear()
            selected_item = None
            for actor in self.scene.actors:
                label = actor.name
                if actor.id in self._pending_scene_robot_loads:
                    label = f"{label} (loading)"
                if not actor.visible:
                    label = f"{label} (hidden)"
                if actor.locked:
                    label = f"{label} (locked)"
                item = QTreeWidgetItem([label])
                item.setData(0, Qt.ItemDataRole.UserRole, actor.id)
                item.setToolTip(0, f"{actor.kind}: {actor.id}")
                tree.addTopLevelItem(item)
                if actor.id == self.scene.selection.actor_id:
                    selected_item = item
            if selected_item is not None:
                tree.setCurrentItem(selected_item)
            tree.blockSignals(tree_blocked)
            self.refresh_scene_actor_controls()
        finally:
            self._syncing_scene_panel = False

    def refresh_scene_actor_controls(self):
        actor = self.scene.actors.get(self.scene.selection.actor_id)
        enabled = actor is not None
        for widget in (
            self.duplicate_actor_button,
            self.delete_actor_button,
        ):
            widget.setEnabled(enabled)
        self.scene_visible_box.setEnabled(enabled)
        self.scene_locked_box.setEnabled(enabled)
        visible_blocked = self.scene_visible_box.blockSignals(True)
        locked_blocked = self.scene_locked_box.blockSignals(True)
        if actor is None:
            try:
                self.scene_visible_box.setChecked(False)
                self.scene_locked_box.setChecked(False)
            finally:
                self.scene_visible_box.blockSignals(visible_blocked)
                self.scene_locked_box.blockSignals(locked_blocked)
            return
        try:
            self.scene_visible_box.setChecked(actor.visible)
            self.scene_locked_box.setChecked(actor.locked)
        finally:
            self.scene_visible_box.blockSignals(visible_blocked)
            self.scene_locked_box.blockSignals(locked_blocked)

        robot_editing = bool(
            actor is not None
            and actor.kind == "robot"
            and actor.id == self.editor_robot_actor_id
            and actor.visible
            and not actor.locked
        )
        for panel in (
            self.controls.robot_panel,
            self.controls.target_panel,
            self.controls.preview_ik_panel,
            self.controls.trajectory_panel,
        ):
            panel.setEnabled(robot_editing)
        self.viewer_3d.set_editing_enabled(robot_editing)

    def activate_scene_robot_actor(self, actor_id, frame_id=None):
        actor = self.scene.actors.require(actor_id)
        if actor.kind != "robot":
            raise ValueError(f"Actor {actor_id!r} is not a robot.")

        if actor.id != self.editor_robot_actor_id:
            self.sync_current_robot_session()
        session = self.ensure_robot_actor_session(actor)
        if session is None:
            self.scene.select_actor(actor.id, frame_id=frame_id)
            self.refresh_scene_tree()
            self.statusBar().showMessage(f"Loading {actor.name}…")
            return None
        previous_viewer = self.viewer_3d
        self.editor_robot_actor_id = actor.id
        self.scene.metadata["editor_robot_actor_id"] = actor.id

        for name, value in model_sessions.activated_session_state(
            self.robot_actor_model_key(actor),
            session,
        ).items():
            setattr(self, name, value)

        available_frames = list(session.adapter.trajectory_frames)
        preferred_frame = (
            frame_id
            or actor.metadata.get("selected_frame")
            or session.selected_frame
            or self.controls.frame_box.currentText()
        )
        if preferred_frame not in available_frames:
            preferred_frame = available_frames[0] if available_frames else None
        actor.metadata["selected_frame"] = preferred_frame
        session.selected_frame = preferred_frame
        self.scene.select_actor(actor.id, frame_id=preferred_frame)

        self._syncing_editor_selection = True
        try:
            model_index = self.controls.model_box.findData(self.model_key)
            if model_index >= 0:
                blocked = self.controls.model_box.blockSignals(True)
                self.controls.model_box.setCurrentIndex(model_index)
                self.controls.model_box.blockSignals(blocked)
            if available_frames:
                blocked = self.controls.frame_box.blockSignals(True)
                self.controls.frame_box.clear()
                self.controls.frame_box.addItems(available_frames)
                self.controls.frame_box.setCurrentText(preferred_frame)
                self.controls.frame_box.blockSignals(blocked)
        finally:
            self._syncing_editor_selection = False

        self.viewer_3d_stack.setCurrentWidget(self.viewer_3d)
        if previous_viewer is not self.viewer_3d:
            previous_viewer.detach_canvas()
        self.viewer_3d.attach_canvas(
            self.shared_scene_canvas,
            scene=self.scene,
            active_robot_actor_id=actor.id,
            scene_edit_target=self.scene_edit_target(),
        )
        self.viewer_2d_skeleton_stack.setCurrentWidget(self.viewer_2d_stickman)
        self.viewer_3d_mujoco.set_model_adapter(session.adapter)
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.set_editor_timeline_duration(self.scene.timeline.duration)
        if self.viewer_3d.state_timeline is not None:
            self.viewer_3d.set_current_time(self.scene.timeline.current_time)
        self.model_source_label.setText(self.model_source_text(session.adapter))
        self.scene.metadata["active_model_key"] = self.model_key
        self.update_project_chrome()
        self.update_editor_context()

        binding = self.viewer_3d.frame_bindings.get(preferred_frame)
        if binding is not None:
            self.viewer_3d.select_target(*binding, emit=False)
            self.viewer_3d._set_target_to_selected_pose()
            self.on_3d_target_frame_changed(preferred_frame)
        else:
            self.refresh_display(apply_stickman_frame=False)
        return session

    def select_active_robot_frame(self, frame_id=None):
        actor = self.scene.actors.require(self.editor_robot_actor_id)
        if actor.kind != "robot":
            raise ValueError(f"Actor {actor.id!r} is not a robot.")

        session = self.current_robot_session()
        available_frames = list(self.robot_model_3d.trajectory_frames)
        preferred_frame = (
            frame_id
            or actor.metadata.get("selected_frame")
            or (session.selected_frame if session is not None else None)
            or self.controls.frame_box.currentText()
        )
        if preferred_frame not in available_frames:
            preferred_frame = available_frames[0] if available_frames else None

        actor.metadata["selected_frame"] = preferred_frame
        if session is not None:
            session.selected_frame = preferred_frame
        self.scene.select_actor(actor.id, frame_id=preferred_frame)

        if preferred_frame is not None:
            blocked = self.controls.frame_box.blockSignals(True)
            try:
                self.controls.frame_box.setCurrentText(preferred_frame)
            finally:
                self.controls.frame_box.blockSignals(blocked)
            self.set_current_frame_to_active_robot_pose(
                preferred_frame,
                emit_pose_changed=False,
            )

        self.update_editor_context()
        self.refresh_display(apply_stickman_frame=False)
        return preferred_frame

    def select_scene_actor(self, actor_id, frame_id=None):
        actor = self.scene.actors.require(actor_id)
        if actor.kind == "robot":
            if frame_id is None and self.scene.selection.actor_id == actor.id:
                frame_id = self.scene.selection.frame_id
            if actor.id == self.editor_robot_actor_id:
                self.select_active_robot_frame(frame_id)
            else:
                self.activate_scene_robot_actor(actor.id, frame_id=frame_id)
        else:
            self.sync_current_robot_session()
            self.scene.select_actor(actor.id)
            self.refresh_scene_actor_controls()
            self.refresh_display(apply_stickman_frame=False)
        self.refresh_scene_actor_controls()
        return actor

    def on_scene_tree_selection_changed(self):
        if self._syncing_scene_panel:
            return
        actor_id = self.selected_scene_actor_id()
        if actor_id is None or actor_id not in self.scene.actors.actors:
            return
        actor = self.select_scene_actor(actor_id)
        self.statusBar().showMessage(f"Selected {actor.kind}: {actor.name}", 2000)
        self.mark_project_dirty("Scene selection")
        self._refresh_history_baseline()

    def on_add_scene_object_clicked(self):
        index = len(self.scene.actors.objects()) + 1
        self.add_scene_object(
            name=f"Object {index}",
            shape="box",
            size=[0.24, 0.18, 0.16],
            transform=Transform(position=(0.45, 0.0, 0.12)),
        )

    def on_import_scene_object_clicked(self):
        if self.current_project is None:
            QMessageBox.warning(
                self,
                "Import Mesh Object",
                "Save or create a project before importing mesh assets.",
            )
            return
        extensions = " ".join(
            f"*{suffix}" for suffix in sorted(SUPPORTED_OBJECT_MESH_EXTENSIONS)
        )
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Import Mesh Object",
            str(self.import_mesh_folder or self.current_project.root_dir),
            f"Mesh objects ({extensions});;All files (*)",
        )
        if not path:
            return
        self.import_mesh_folder = str(Path(path).expanduser().parent)
        dialog = ObjectMeshImportDialog(path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            imported = self.current_project.import_object_mesh(path)
            reference = imported.model_reference(scale=dialog.uniform_scale())
            actor = self.add_scene_mesh_object(
                name=dialog.object_name(),
                model_reference=reference,
                transform=Transform(position=(0.45, 0.0, 0.12)),
            )
            self.statusBar().showMessage(
                f"Imported mesh object: {actor.name}",
                3000,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import Mesh Object", str(exc))

    def on_add_scene_robot_clicked(self):
        choices = self.available_scene_robot_choices()
        if not choices:
            QMessageBox.warning(
                self,
                "Add Robot Actor",
                "No robot models are available.",
            )
            return
        labels = [label for label, _key, _display_name in choices]
        current_index = 0
        for index, (_label, key, _display_name) in enumerate(choices):
            if key == self.model_key:
                current_index = index
                break
        label, accepted = QInputDialog.getItem(
            self,
            "Add Robot Actor",
            "Robot model",
            labels,
            current_index,
            False,
        )
        if not accepted:
            return
        choice_by_label = {
            label: (key, display_name)
            for label, key, display_name in choices
        }
        model_key, model_name = choice_by_label[label]
        self.add_scene_robot(model_key=model_key, model_name=model_name)

    def on_duplicate_scene_actor_clicked(self):
        actor_id = self.selected_scene_actor_id()
        if actor_id is None:
            return
        try:
            self.duplicate_scene_actor(actor_id)
        except (KeyError, ValueError) as exc:
            self.status_text.append(str(exc))

    def on_delete_scene_actor_clicked(self):
        actor_id = self.selected_scene_actor_id()
        if actor_id is None:
            return
        try:
            self.delete_scene_actor(actor_id)
        except (KeyError, ValueError) as exc:
            self.status_text.append(str(exc))

    def on_scene_visible_toggled(self, checked):
        if self._syncing_scene_panel:
            return
        actor_id = self.selected_scene_actor_id()
        if actor_id is not None:
            self.set_scene_actor_visibility(actor_id, checked)

    def on_scene_locked_toggled(self, checked):
        if self._syncing_scene_panel:
            return
        actor_id = self.selected_scene_actor_id()
        if actor_id is not None:
            self.set_scene_actor_locked(actor_id, checked)

    def project_transition_reason(self, action):
        clean = "".join(
            character.lower() if character.isalnum() else "_"
            for character in str(action)
        )
        clean = "_".join(part for part in clean.split("_") if part)
        return f"before_{clean or 'transition'}"

    def update_project_chrome(self):
        title = "Reference Frame Trajectory GUI"
        if self.current_project is not None:
            title = f"{title} - {self.current_project.project_name}"
        elif self.robot_model_3d is not None:
            title = f"{title} - {self.robot_model_3d.model_name}"
        if self.project_dirty:
            title = f"{title} *"
        self.setWindowTitle(title)

    def set_project_dirty(self, dirty=True):
        self.project_dirty = bool(dirty and self.current_project is not None)
        self.update_project_panel()

    def mark_project_dirty(self, reason=None):
        if self._suppress_project_dirty or self.current_project is None:
            return
        if not self.project_dirty:
            self.set_project_dirty(True)

    def handle_project_transition_choice(self, choice, action):
        if not self.project_dirty or self.current_project is None:
            return True

        reason = self.project_transition_reason(action)
        project = self.current_project
        if choice == "save":
            return self.save_current_project(reason=reason)
        if choice == "autosave":
            if not self.autosave_current_project(
                show_status=False,
                capture_snapshot=True,
                reason=reason,
            ):
                return False
            self.set_project_dirty(False)
            return True
        if choice == "discard":
            project.clear_autosave()
            project.append_session_event(
                "project_dirty_discarded",
                self.project_session_details(
                    {
                        "action": action,
                        "reason": reason,
                    }
                ),
            )
            self.set_project_dirty(False)
            return True
        if choice == "cancel":
            project.append_session_event(
                "project_transition_cancelled",
                self.project_session_details(
                    {
                        "action": action,
                        "reason": reason,
                    }
                ),
            )
        return False

    def confirm_project_transition(self, action):
        if not self.project_dirty or self.current_project is None:
            return True

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Unsaved Project Changes")
        dialog.setText(f"Save changes before you {action}?")
        dialog.setInformativeText(
            "Save updates the project files. Autosave stores recovery state "
            "for this project. Discard leaves the last saved project unchanged."
        )
        save_button = dialog.addButton(
            "Save",
            QMessageBox.ButtonRole.AcceptRole,
        )
        autosave_button = dialog.addButton(
            "Autosave",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = dialog.addButton(
            "Discard",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(save_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is save_button:
            choice = "save"
        elif clicked is autosave_button:
            choice = "autosave"
        elif clicked is discard_button:
            choice = "discard"
        else:
            choice = "cancel"
        return self.handle_project_transition_choice(choice, action)

    def project_session_details(self, extra=None):
        viewer = self.viewer_3d
        active_view = ""
        if 0 <= self.viewer_tabs.currentIndex() < self.viewer_tabs.count():
            active_view = self.viewer_tabs.tabText(self.viewer_tabs.currentIndex())
        details = {
            "model_key": self.model_key,
            "model_name": self.current_model_display_name(),
            "frame_count": len(self.trajectory.frames),
            "actor_count": len(getattr(self.scene.actors, "actors", {})),
            "active_actor_id": self.scene.selection.actor_id,
            "current_time": float(viewer.get_current_time()),
            "selected_frame": self.controls.frame_box.currentText(),
            "active_view": active_view,
        }
        if extra:
            details.update(extra)
        return details

    def update_project_panel(self):
        if self.current_project is None:
            self.project_name_label.setText("No project open")
            self.project_name_label.setToolTip("No GhostGUI project is open.")
            project_button = getattr(self, "project_toolbar_button", None)
            if project_button is not None:
                project_button.setToolTip("No GhostGUI project is open.")
            self.update_project_chrome()
            return
        name = self.current_project.project_name
        if self.project_dirty:
            name = f"{name} *"
        lines = [str(self.current_project.root_dir)]
        if self.project_dirty:
            lines.append("Unsaved changes")
        self.project_name_label.setText(name)
        self.project_name_label.setToolTip("\n".join(lines))
        project_button = getattr(self, "project_toolbar_button", None)
        if project_button is not None:
            project_button.setToolTip("\n".join(lines))
        self.update_project_chrome()

    def recent_project_display_name(self, entry):
        project_path = Path(entry["path"])
        project_name = entry.get("project_name") or project_path.stem
        folder_name = project_path.name
        if folder_name.endswith(".ghostgui"):
            folder_name = folder_name[: -len(".ghostgui")]
        if folder_name and folder_name != project_name:
            return f"{project_name} ({folder_name})"
        return project_name

    def refresh_recent_projects(self):
        if not hasattr(self, "recent_projects_box"):
            return
        entries = load_recent_projects()
        box = self.recent_projects_box
        was_blocked = box.blockSignals(True)
        try:
            box.clear()
            box.addItem("Recent projects...", "")
            for entry in entries:
                path = entry["path"]
                box.addItem(self.recent_project_display_name(entry), path)
                index = box.count() - 1
                box.setItemData(index, path, Qt.ItemDataRole.ToolTipRole)
            box.setCurrentIndex(0)
            box.setEnabled(bool(entries))
        finally:
            box.blockSignals(was_blocked)

    def remember_current_project(self):
        if self.current_project is None:
            return
        try:
            remember_recent_project(self.current_project)
            self.refresh_recent_projects()
        except OSError as exc:
            if hasattr(self, "status_text"):
                self.status_text.append(f"Could not update recent projects: {exc}")

    def on_recent_project_selected(self, index):
        path = self.recent_projects_box.itemData(index)
        self.recent_projects_box.setCurrentIndex(0)
        if not path:
            return
        self.open_project_path(path, source="recent_projects")

    def build_project_browser_dialog(self):
        return ProjectBrowserDialog(load_project_browser_previews(), parent=self)

    def on_open_project(self):
        dialog = self.build_project_browser_dialog()
        if dialog.exec():
            if dialog.selected_project_path:
                self.open_project_path(
                    dialog.selected_project_path,
                    source="project_browser",
                )
            elif dialog.browse_requested:
                self.on_browse_project_folder()
        self.refresh_recent_projects()

    def on_project_browser(self):
        self.on_open_project()

    def on_new_project(self):
        if not self.confirm_project_transition("create a new project"):
            return
        default_name = (
            self.current_project.project_name
            if self.current_project is not None
            else f"{self.model_key}_project"
        )
        name, accepted = QInputDialog.getText(
            self,
            "New GhostGUI Project",
            "Project name",
            text=default_name,
        )
        if not accepted or not name.strip():
            return
        project_root = available_default_project_root_from_name(name)
        try:
            self.create_project_at(
                project_root,
                name.strip(),
                reset_workspace=True,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Create project failed", str(exc))

    def on_browse_project_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Open GhostGUI Project",
            str(Path.home()),
        )
        if path:
            self.open_project_path(path, source="folder_dialog")

    def on_save_project(self):
        if self.current_project is None:
            default_name = f"{self.model_key}_project"
            name, accepted = QInputDialog.getText(
                self,
                "Save GhostGUI Project",
                "Project name",
                text=default_name,
            )
            if not accepted or not name.strip():
                return
            project_root = available_default_project_root_from_name(name)
            try:
                self.create_project_at(project_root, name.strip())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Save project failed", str(exc))
            return
        try:
            self.save_current_project()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save project failed", str(exc))

    def reset_workspace_for_new_project(self):
        was_suppressing_dirty = self._suppress_project_dirty
        self._suppress_project_dirty = True
        try:
            viewer = self.viewer_3d
            viewer.pause_playback()
            viewer.canvas.cancel_transform_drag()
            self.trajectory.clear()
            self.active_index = -1
            self.scene = Scene.single_robot(
                self.model_key,
                model_name=self.current_model_display_name(),
                trajectory=self.trajectory,
            )
            self.editor_robot_actor_id = self.scene.active_robot_id()
            self.scene.metadata["editor_robot_actor_id"] = self.editor_robot_actor_id
            self.scene_runtime = SceneRuntime(self.scene)
            self.rebind_current_robot_session(
                self.scene.actors.require(self.editor_robot_actor_id),
                reset_sessions=True,
            )
            viewer.clear_robot_trajectory()
            viewer.clear_editable_timeline(keep_current_pose=False, reset_time=0.0)
            viewer.preview_active = False
            viewer.clear_pinned_frame_targets()
            viewer.canvas.set_preview_visible(False)
            viewer.canvas.camera_distance = 5.0
            viewer.canvas.camera_yaw = 38.0
            viewer.canvas.camera_pitch = 24.0
            viewer.canvas.camera_center = np.asarray([0.0, 0.0, 0.75], dtype=float)
            viewer.canvas.update()
            self.set_editor_timeline_duration(5.0)
            self.controls.time_slider.set_value(0.0)
            self.controls.show_keyframes_box.setChecked(True)
            self.controls.show_lines_box.setChecked(True)
            self.controls.corner_smoothing_slider.set_value(0.0)
            viewer.show_ghosts.setChecked(False)
            viewer.frame_slider.setValue(0)
            if self.controls.table.currentRow() >= 0:
                self.controls.table.clearSelection()
            self.set_current_frame_to_model_reference(
                self.controls.frame_box.currentText(),
                emit_pose_changed=False,
            )
            self.refresh_display()
            self._refresh_history_baseline()
        finally:
            self._suppress_project_dirty = was_suppressing_dirty

    def create_project_at(self, project_root, project_name, reset_workspace=False):
        if reset_workspace:
            self.reset_workspace_for_new_project()
        project = GhostGUIProject.create(
            project_root,
            project_name,
            self.model_key,
            self.current_model_display_name(),
        )
        self.current_project = project
        self.save_current_project(reason="initial_create")
        return project

    def save_current_project(
        self,
        show_status=True,
        capture_snapshot=True,
        reason="manual",
    ):
        project = self.current_project
        if project is None:
            return False

        viewer = self.viewer_3d
        if viewer.state_timeline is not None and viewer.committed_state is not None:
            viewer.state_timeline.set_state(
                viewer.get_current_time(),
                viewer.committed_state.get_qpos(),
            )

        project.update_robot(self.model_key, self.current_model_display_name())
        project.write_scene(self.capture_project_scene())
        project.write_trajectory(self.trajectory)
        if viewer.state_timeline is not None:
            project.save_qpos_timeline(viewer.state_timeline)
        project.write_workspace(self.capture_project_workspace())
        snapshot_saved = False
        if capture_snapshot:
            snapshot_saved = bool(self.save_project_snapshot(project))
        project.save_metadata()
        project.clear_autosave()
        project.append_session_event(
            "project_saved",
            self.project_session_details(
                {
                    "reason": reason,
                    "snapshot_saved": snapshot_saved,
                }
            ),
        )
        self.set_project_dirty(False)
        self.remember_current_project()
        if show_status:
            message = f"Saved project: {project.root_dir}"
            self.status_text.setText(message)
            self.statusBar().showMessage(message, 3000)
        return True

    def autosave_current_project(
        self,
        show_status=True,
        capture_snapshot=False,
        reason="manual",
    ):
        project = self.current_project
        if project is None:
            return False

        viewer = self.viewer_3d
        if viewer.state_timeline is None or viewer.committed_state is None:
            return False

        viewer.state_timeline.set_state(
            viewer.get_current_time(),
            viewer.committed_state.get_qpos(),
        )
        project.write_autosave(
            self.trajectory,
            viewer.state_timeline,
            self.capture_project_workspace(),
            self.model_key,
            self.current_model_display_name(),
            scene=self.capture_project_scene(),
        )
        snapshot_saved = False
        if capture_snapshot:
            snapshot_saved = bool(self.save_project_snapshot(project))
        project.append_session_event(
            "project_autosaved",
            self.project_session_details(
                {
                    "reason": reason,
                    "snapshot_saved": snapshot_saved,
                }
            ),
        )
        if show_status:
            message = f"Autosaved project recovery state: {project.root_dir}"
            self.statusBar().showMessage(message, 3000)
        return True

    def save_project_snapshot(self, project):
        try:
            path = project.paths.last_snapshot
            path.parent.mkdir(parents=True, exist_ok=True)
            pixmap = self.grab()
            if not pixmap.isNull():
                return pixmap.save(str(path))
        except Exception:
            return False
        return False

    def capture_project_workspace(self):
        viewer = self.viewer_3d
        canvas = viewer.canvas
        selected_kind, selected_name = viewer._selected_target()
        return {
            "schema_version": 1,
            "scene_selection": self.scene.selection.to_dict(),
            "active_index": int(self.active_index),
            "active_view_index": int(self.viewer_tabs.currentIndex()),
            "current_time": float(viewer.get_current_time()),
            "selected_frame": self.controls.frame_box.currentText(),
            "selected_row": int(self.controls.selected_row()),
            "timeline_duration": float(viewer.timeline_duration),
            "control_frame": self.controls.current_frame().to_dict(),
            "target_selection": {
                "kind": selected_kind,
                "name": selected_name,
            },
            "camera": {
                "yaw": float(canvas.camera_yaw),
                "pitch": float(canvas.camera_pitch),
                "distance": float(canvas.camera_distance),
                "center": [float(value) for value in canvas.camera_center],
            },
            "display": {
                "show_keyframes": bool(self.controls.show_keyframes()),
                "show_trajectory_lines": bool(self.controls.show_trajectory_lines()),
                "smoothing": float(self.controls.corner_smoothing()),
                "show_playback_ghosts": bool(viewer.show_ghosts.isChecked()),
                "secondary_robot_detail": str(
                    viewer.secondary_detail_box.currentData() or "full"
                ),
                "frame_slider_value": int(viewer.frame_slider.value()),
                "trajectory_import_dt": float(viewer.trajectory_import_dt.value()),
            },
        }

    def open_project_path(self, path, source="direct"):
        try:
            project = GhostGUIProject.open(path)
        except (OSError, ValueError) as exc:
            try:
                forget_recent_project(path)
                self.refresh_recent_projects()
            except OSError:
                pass
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False

        if not self.confirm_project_transition(f"open {project.project_name}"):
            return False

        restore_autosave = self.should_restore_project_autosave(project)
        model_key = self.project_restore_model_key(project, restore_autosave)

        if model_key != self.model_key:
            if model_key not in self.model_registry:
                message = f"Project robot model is not available: {model_key}"
                QMessageBox.warning(self, "Open project failed", message)
                return False
            self._pending_project_restore = project
            self._pending_project_restore_autosave = restore_autosave
            self._pending_project_restore_source = source
            self.on_model_changed(model_key)
            self.restore_pending_project_if_ready()
            return True

        return self.restore_project(
            project,
            autosave=restore_autosave,
            source=source,
        )

    def should_restore_project_autosave(self, project):
        if not project.is_autosave_newer():
            return False
        response = QMessageBox.question(
            self,
            "Restore project autosave?",
            (
                "A newer autosave exists for this project. "
                "Restore the autosaved workspace?\n\n"
                "Choose No to discard the autosave and open the last saved project."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if response == QMessageBox.StandardButton.Yes:
            return True
        project.clear_autosave()
        project.append_session_event(
            "project_autosave_discarded",
            {
                "model_key": project.model_key,
                "source": "open_project",
            },
        )
        return False

    def project_restore_model_key(self, project, autosave=False):
        scene_model_key = self.scene_project_model_key(project, autosave=autosave)
        if scene_model_key:
            return scene_model_key
        if not autosave:
            return project.model_key
        try:
            return project.autosave_model_key()
        except (OSError, TypeError, ValueError):
            return project.model_key

    def restore_pending_project_if_ready(self):
        project = self._pending_project_restore
        autosave = self._pending_project_restore_autosave
        source = self._pending_project_restore_source
        if project is None:
            return False
        model_key = self.project_restore_model_key(project, autosave)
        if model_key != self.model_key:
            return False
        self._pending_project_restore = None
        self._pending_project_restore_autosave = False
        self._pending_project_restore_source = "direct"
        return self.restore_project(project, autosave=autosave, source=source)

    def restore_project(self, project, autosave=False, source="direct"):
        try:
            trajectory_data = project.read_trajectory_dict(autosave=autosave)
            workspace = project.read_workspace(autosave=autosave)
            try:
                scene_data = project.read_scene_dict(autosave=autosave)
            except FileNotFoundError:
                scene_data = None
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False
        if not isinstance(workspace, dict):
            QMessageBox.warning(
                self,
                "Open project failed",
                "Project workspace data must be a JSON object.",
            )
            return False

        viewer = self.viewer_3d
        viewer.pause_playback()
        viewer.canvas.cancel_transform_drag()
        viewer.clear_robot_trajectory()
        try:
            self.restore_scene(scene_data, trajectory_data, workspace)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False

        if viewer.state_timeline is not None:
            actor_tracks = self.scene.tracks.joint_tracks.get(
                self.editor_robot_actor_id,
                {},
            )
            has_scene_qpos = bool(
                isinstance(actor_tracks, dict) and actor_tracks.get("qpos")
            )
            qpos_path = (
                project.autosave_paths.qpos_timeline
                if autosave else project.paths.qpos_timeline
            )
            if qpos_path.exists() and not has_scene_qpos:
                try:
                    project.load_qpos_timeline(
                        viewer.state_timeline,
                        autosave=autosave,
                    )
                except ValueError as exc:
                    QMessageBox.warning(self, "Open project failed", str(exc))
                    return False
                self.scene.tracks.set_robot_qpos_timeline(
                    self.editor_robot_actor_id,
                    viewer.state_timeline,
                )
            elif not has_scene_qpos:
                viewer.clear_editable_timeline(keep_current_pose=True, reset_time=0.0)

        timeline_duration = workspace.get("timeline_duration")
        if timeline_duration is not None:
            self.set_editor_timeline_duration(float(timeline_duration))

        frame_data = workspace.get("control_frame")
        if frame_data:
            try:
                self._restore_control_frame(TargetFrame.from_dict(dict(frame_data)))
            except (TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Open project failed", str(exc))
                return False

        was_suppressing_dirty = self._suppress_project_dirty
        self._suppress_project_dirty = True
        self.current_project = project
        try:
            self.restore_project_workspace(workspace)
        finally:
            self._suppress_project_dirty = was_suppressing_dirty

        current_time = float(workspace.get("current_time", 0.0))
        self.scene.timeline.current_time = current_time
        viewer = self.viewer_3d
        viewer.set_current_time(current_time)
        self.controls._suppress_pose_changed = True
        try:
            self.controls.time_slider.set_value(current_time)
        finally:
            self.controls._suppress_pose_changed = False

        self.active_index = int(workspace.get("active_index", -1))
        self.refresh_display(apply_stickman_frame=False)
        selected_row = int(workspace.get("selected_row", -1))
        if 0 <= selected_row < self.controls.table.rowCount():
            self.controls.table.setCurrentCell(selected_row, 0)
        else:
            self.controls.table.clearSelection()

        self.set_project_dirty(False)
        self.remember_current_project()
        project.append_session_event(
            "project_opened",
            self.project_session_details(
                {
                    "autosave": bool(autosave),
                    "source": source,
                }
            ),
        )
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_history_baseline()
        source = "autosaved project" if autosave else "project"
        message = f"Opened {source}: {project.root_dir}"
        self.status_text.setText(message)
        self.statusBar().showMessage(message, 3000)
        return True

    def restore_project_workspace(self, workspace):
        display = workspace.get("display", {})
        if "show_keyframes" in display:
            self.controls.show_keyframes_box.setChecked(bool(display["show_keyframes"]))
        if "show_trajectory_lines" in display:
            self.controls.show_lines_box.setChecked(
                bool(display["show_trajectory_lines"])
            )
        if "smoothing" in display:
            self.controls.corner_smoothing_slider.set_value(float(display["smoothing"]))
        if "trajectory_import_dt" in display:
            self.viewer_3d.trajectory_import_dt.setValue(
                float(display["trajectory_import_dt"])
            )
        if "show_playback_ghosts" in display:
            self.viewer_3d.show_ghosts.setChecked(
                bool(display["show_playback_ghosts"])
            )
        if "secondary_robot_detail" in display:
            detail_index = self.viewer_3d.secondary_detail_box.findData(
                str(display["secondary_robot_detail"])
            )
            if detail_index >= 0:
                self.viewer_3d.secondary_detail_box.setCurrentIndex(detail_index)
        if "frame_slider_value" in display:
            self.viewer_3d.frame_slider.setValue(int(display["frame_slider_value"]))

        target = workspace.get("target_selection", {})
        scene_selection = workspace.get("scene_selection", {})
        scene_actor_id = scene_selection.get("actor_id")
        if scene_actor_id in self.scene.actors.actors:
            self.select_scene_actor(
                scene_actor_id,
                frame_id=scene_selection.get("frame_id"),
            )
        target_kind = target.get("kind")
        target_name = target.get("name")
        if target_kind and target_name:
            self.viewer_3d.select_target(target_kind, target_name, emit=False)
        else:
            selected_frame = workspace.get("selected_frame")
            binding = self.viewer_3d.frame_bindings.get(selected_frame)
            if binding is not None:
                self.viewer_3d.select_target(*binding, emit=False)

        camera = workspace.get("camera", {})
        canvas = self.viewer_3d.canvas
        if "yaw" in camera:
            canvas.camera_yaw = float(camera["yaw"])
        if "pitch" in camera:
            canvas.camera_pitch = float(camera["pitch"])
        if "distance" in camera:
            canvas.camera_distance = float(camera["distance"])
        center = camera.get("center")
        if isinstance(center, list) and len(center) == 3:
            canvas.camera_center = np.asarray(center, dtype=float)
        canvas.update()

        active_view_index = int(workspace.get("active_view_index", 0))
        if 0 <= active_view_index < self.viewer_tabs.count():
            self.viewer_tabs.setCurrentIndex(active_view_index)

    def build_help_button(self):
        button = QToolButton()
        button.setObjectName("helpButton")
        button.setText("Help")
        button.setToolTip("Open GhostGUI help center")
        button.setMinimumWidth(54)
        button.setFixedHeight(28)
        button.clicked.connect(self.show_help_center)
        return button

    def position_help_button(self):
        button = getattr(self, "help_button", None)
        if button is None:
            return
        if button.parentWidget() is not self:
            return
        margin = 10
        x = max(margin, self.width() - button.width() - margin)
        y = margin
        button.move(x, y)
        button.raise_()

    def show_help_center(self):
        if self.help_dialog is None:
            self.help_dialog = HelpCenterDialog(self)
            self.help_dialog.start_tutorial_requested.connect(
                self.start_first_motion_tutorial
            )
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def start_first_motion_tutorial(self):
        if self.help_dialog is not None:
            self.help_dialog.hide()
        self.tutorial_manager.start_first_motion()

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
            self.controls.set_timeslice_context_widget(
                self.viewer_3d.timeslice_context_widget()
            )
            self.controls.set_display_context_widget(
                self.viewer_3d.display_context_widget()
            )
            self.viewer_3d.set_trajectory_display_widgets(
                self.controls.show_keyframes_box,
                self.controls.show_lines_box,
            )
            self.controls.set_preview_ik_context_widget(
                self.viewer_3d.preview_ik_context_widget()
            )
            self.sync_viewer_status_panel()
        else:
            self.controls.set_robot_context_widget(None)
            self.controls.set_selection_context_widget(None)
            self.controls.set_trajectory_context_widget(None)
            self.controls.set_timeslice_context_widget(None)
            self.controls.set_display_context_widget(None)
            self.controls.set_preview_ik_context_widget(None)

    # ============================================================
    # Signal connections
    # ============================================================

    def connect_signals(self):
        self.controls.model_changed.connect(self.on_model_changed)
        self.controls.open_model_clicked.connect(self.on_open_model_file)
        self.controls.choose_mesh_folder_clicked.connect(self.on_choose_mesh_folder)
        self.controls.setup_import_requested.connect(self.on_setup_import_requested)
        self.controls.setup_export_requested.connect(self.on_setup_export_requested)
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
        self.controls.keyframes_visibility_changed.connect(
            self.on_trajectory_display_changed
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
        viewer_3d.scene_actor_transform_dragged.connect(
            self.on_scene_actor_transform_dragged
        )
        viewer_3d.scene_actor_transform_drag_finished.connect(
            self.on_scene_actor_transform_drag_finished
        )
        viewer_3d.scene_robot_body_double_clicked.connect(
            self.on_scene_robot_body_double_clicked
        )
        viewer_3d.target_frame_changed.connect(self.on_3d_target_frame_changed)
        viewer_3d.preview_cancelled.connect(self.on_preview_cancelled)
        viewer_3d.trajectory_csv_loaded.connect(self.on_trajectory_csv_loaded)
        viewer_3d.generate_requested.connect(self.on_generate_trajectory)
        viewer_3d.clear_trajectory_requested.connect(self.on_clear_trajectory)
        viewer_3d.geometry_progress.connect(
            lambda complete, total, viewer=viewer_3d: (
                self.on_viewer_geometry_progress(viewer, complete, total)
            )
        )
        viewer_3d.camera_changed.connect(
            lambda: self.mark_project_dirty("Camera")
        )
        viewer_3d.show_ghosts.toggled.connect(
            lambda _checked: self.mark_project_dirty("Display settings")
        )
        viewer_3d.secondary_detail_box.currentIndexChanged.connect(
            lambda _index, viewer=viewer_3d: (
                self.on_secondary_robot_detail_changed(viewer)
            )
        )
        viewer_3d.frame_slider.valueChanged.connect(
            lambda _value: self.mark_project_dirty("Playback frame")
        )
        viewer_3d.trajectory_import_dt.valueChanged.connect(
            lambda _value: self.mark_project_dirty("Import time step")
        )
        viewer_3d.timeslice_time_changed.connect(
            self.on_viewer_timeslice_time_changed
        )
        viewer_3d.timeline_duration_changed.connect(
            self.on_viewer_timeline_duration_changed
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

    def on_secondary_robot_detail_changed(self, source_viewer):
        """Keep the secondary-detail choice scene-wide across actor editors."""
        mode = source_viewer.secondary_detail_box.currentData() or "full"
        seen = set()
        for session in self.model_sessions.values():
            viewer = session.viewer_3d
            if id(viewer) in seen:
                continue
            seen.add(id(viewer))
            index = viewer.secondary_detail_box.findData(mode)
            if index < 0 or index == viewer.secondary_detail_box.currentIndex():
                continue
            blocked = viewer.secondary_detail_box.blockSignals(True)
            viewer.secondary_detail_box.setCurrentIndex(index)
            viewer.secondary_detail_box.blockSignals(blocked)
        self.shared_scene_canvas.set_secondary_robot_detail(mode)
        self.mark_project_dirty("Display settings")

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
            ghost_source=viewer.ghost_source,
            frame_slider_value=int(viewer.frame_slider.value()),
            show_ghosts=bool(viewer.show_ghosts.isChecked()),
            timeline_duration=float(viewer.timeline_duration),
            scene_state=copy.deepcopy(self.capture_project_scene()),
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
        self._history_restoring = True
        try:
            self.scene = Scene.from_dict(snapshot.scene_state)
            restored_selection = self.scene.selection.to_dict()
            target_actor_id = self.scene.metadata.get(
                "editor_robot_actor_id"
            ) or self.scene.active_robot_id()
            self.scene_runtime = SceneRuntime(self.scene)
            for key, session in list(self.model_sessions.items()):
                actor_id = key[0] if isinstance(key, tuple) else None
                actor = self.scene.actors.get(actor_id)
                if (
                    actor is None
                    or actor.kind != "robot"
                    or self.robot_actor_model_key(actor) != str(key[1])
                ):
                    self.discard_robot_actor_sessions(actor_id)
                    continue
                session.trajectory = self.scene.tracks.robot_trajectory(actor_id)
                session.selected_frame = actor.metadata.get("selected_frame")
                timeline = session.viewer_3d.state_timeline
                if timeline is not None:
                    if not self.scene.tracks.load_robot_qpos_timeline(
                        actor_id,
                        timeline,
                    ):
                        timeline.reset()

            target_actor = self.scene.actors.get(target_actor_id)
            if target_actor is not None:
                self.editor_robot_actor_id = target_actor.id
                self.activate_scene_robot_actor(
                    target_actor.id,
                    frame_id=self.scene.selection.frame_id,
                )
            viewer = self.viewer_3d
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
                self.set_editor_timeline_duration(snapshot.timeline_duration)
                show_ghosts_blocked = viewer.show_ghosts.blockSignals(True)
                quick_ghosts_blocked = viewer.quick_show_ghosts.blockSignals(True)
                try:
                    viewer.show_ghosts.setChecked(snapshot.show_ghosts)
                    viewer.quick_show_ghosts.setChecked(snapshot.show_ghosts)
                finally:
                    viewer.show_ghosts.blockSignals(show_ghosts_blocked)
                    viewer.quick_show_ghosts.blockSignals(quick_ghosts_blocked)
                viewer.ghost_trajectory = [
                    qpos.copy() for qpos in snapshot.ghost_trajectory
                ]
                viewer.ghost_source = snapshot.ghost_source
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
                viewer.clear_pinned_frame_targets()
                viewer.canvas.set_preview_visible(snapshot.preview_active)
                viewer._sync_joint_controls()
                viewer._set_target_to_selected_pose()

            control_frame = TargetFrame.from_dict(dict(snapshot.control_frame))
            self._restore_control_frame(control_frame)
            selected_actor_id = restored_selection.get("actor_id")
            if selected_actor_id in self.scene.actors.actors:
                self.scene.select_actor(
                    selected_actor_id,
                    frame_id=restored_selection.get("frame_id"),
                    track_id=restored_selection.get("track_id"),
                )
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
        self.mark_project_dirty(description)
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
        self.mark_project_dirty(f"Undo {entry.description}")

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
        self.mark_project_dirty(f"Redo {entry.description}")

    def on_viewer_history_action_finished(self, description):
        self.record_history_action(description)

    def on_viewer_status_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            status = self.format_viewer_status(text)
            self.viewer_status_label.setText(status["state"])
            self.status_frame_label.setText(status["frame"])
            self.status_ik_label.setText(status["ik"])
            self.status_move_label.setText(status["move"])
            self.status_text.setText(status["detail"])

    def format_viewer_status(self, text):
        detail = self.format_status_detail(text)
        parts = [part.strip() for part in str(text).split(";") if part.strip()]
        if not self.is_verbose_ik_status(parts):
            return {
                "state": str(text).splitlines()[0],
                "frame": "-",
                "ik": "-",
                "move": "-",
                "detail": detail,
            }

        frame = self.status_field(parts, "frame") or "selected frame"
        frame = frame.replace("_", " ")
        accepted = self.status_field(parts, "accepted")
        ik_error = self.status_field(parts, "IK error")
        lower_text = str(text).lower()
        state = "Preview"

        if "collision blocked" in lower_text:
            state = "Blocked: collision"
        elif "ik blocked" in lower_text:
            state = "Blocked: IK"

        return {
            "state": state,
            "frame": frame,
            "ik": f"{ik_error} m" if ik_error else "-",
            "move": accepted or "-",
            "detail": detail,
        }

    def is_verbose_ik_status(self, parts):
        fields = {"accepted", "IK error", "frame"}
        found = {
            part.split("=", 1)[0].strip()
            for part in parts
            if "=" in part
        }
        return fields.issubset(found)

    def status_field(self, parts, name):
        prefix = f"{name}="
        for part in parts:
            if part.startswith(prefix):
                return part[len(prefix):].strip()
        return None

    def format_status_detail(self, text):
        parts = [part.strip() for part in str(text).split(";") if part.strip()]
        if len(parts) <= 1:
            return str(text)
        return "\n".join(parts)

    def on_viewer_time_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_time_label.setText(text.replace("3D state time:", "Time:"))

    def on_viewer_root_pose_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_root_pose_label.setText(text)

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
        actor = self.scene.actors.get(self.scene.selection.actor_id)
        if actor is None or actor.kind != "robot":
            return
        model_info = self.model_registry.get(model_key)
        if model_info is None:
            self.status_text.setText(f"Unknown robot model: {model_key}")
            return
        actor_id = actor.id
        cached = self.model_sessions.get(
            model_sessions.session_key(actor_id, model_key)
        )
        if cached is not None:
            self.activate_model_session(model_key, cached, actor_id=actor_id)
            return
        pooled_adapter = self.model_resources.get(model_info=model_info)
        if pooled_adapter is not None:
            session = self.create_robot_actor_session(
                actor,
                pooled_adapter,
                model_key=model_key,
                trajectory=Trajectory(),
                load_scene_timeline=False,
            )
            self.activate_model_session(model_key, session, actor_id=actor_id)
            return
        if model_key in self.model_loaders:
            self._model_loader_actor_ids.setdefault(model_key, set()).add(actor_id)
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
        self._background_workers.add(loader)
        loader.finished.connect(
            lambda loader=loader: self._background_worker_finished(loader)
        )
        loader.finished.connect(loader.deleteLater)
        self.model_loaders[model_key] = loader
        self._model_loader_actor_ids[model_key] = {actor_id}
        loader.start()

    def on_model_loaded(self, model_key, adapter):
        self.model_loaders.pop(model_key, None)
        adapter = self.model_resources.register(adapter)
        actor_ids = self._model_loader_actor_ids.pop(
            model_key,
            {self.editor_robot_actor_id},
        )
        created_sessions = {}
        added_actor_ids = set()
        for actor_id in actor_ids:
            actor = self.scene.actors.get(actor_id)
            if actor is None or actor.kind != "robot":
                continue
            is_added_actor = actor_id in self._pending_added_robot_ids
            trajectory = (
                self.scene.tracks.robot_trajectory(actor_id)
                if is_added_actor
                else Trajectory()
            )
            session = self.create_robot_actor_session(
                actor,
                adapter,
                model_key=model_key,
                trajectory=trajectory,
                load_scene_timeline=is_added_actor,
            )
            actor.name = adapter.model_name
            actor.model_reference = {
                "type": "robot_model",
                "model_key": str(model_key),
                "model_name": str(adapter.model_name),
            }
            self.scene.tracks.set_robot_trajectory(actor_id, session.trajectory)
            self._pending_scene_robot_loads.pop(actor_id, None)
            if is_added_actor:
                added_actor_ids.add(actor_id)
                self._pending_added_robot_ids.discard(actor_id)
            created_sessions[actor_id] = session
        self.finish_model_loading_ui()
        selected_actor_id = self.scene.selection.actor_id
        session = created_sessions.get(selected_actor_id)
        if session is not None:
            if selected_actor_id in added_actor_ids:
                self.activate_scene_robot_actor(selected_actor_id)
            else:
                self.activate_model_session(
                    model_key,
                    session,
                    actor_id=selected_actor_id,
                )
        else:
            self.refresh_display(apply_stickman_frame=False)
        if added_actor_ids:
            self.record_history_action("Add robot actor")

    def on_model_load_failed(self, model_key, error, actor_ids=None):
        self.model_loaders.pop(model_key, None)
        actor_ids = set(
            actor_ids
            or self._model_loader_actor_ids.pop(model_key, None)
            or ()
        )
        removed_added_actor = False
        for actor_id in actor_ids:
            self._pending_scene_robot_loads.pop(actor_id, None)
            if actor_id not in self._pending_added_robot_ids:
                continue
            self._pending_added_robot_ids.discard(actor_id)
            if self.scene.actors.get(actor_id) is not None:
                self.scene.delete_actor(actor_id)
                self.discard_robot_actor_sessions(actor_id)
                removed_added_actor = True
        if (
            self._pending_project_restore is not None
            and self.project_restore_model_key(
                self._pending_project_restore,
                self._pending_project_restore_autosave,
            ) == model_key
        ):
            self._pending_project_restore = None
            self._pending_project_restore_autosave = False
            self._pending_project_restore_source = "direct"
        self.finish_model_loading_ui()
        self.finish_render_progress()
        self.status_text.setText(f"Could not load {model_key}: {error}")
        index = self.controls.model_box.findData(self.model_key)
        self.controls.model_box.blockSignals(True)
        self.controls.model_box.setCurrentIndex(index)
        self.controls.model_box.blockSignals(False)
        if removed_added_actor:
            self.refresh_display(apply_stickman_frame=False)

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

    def on_setup_import_requested(self, action):
        if action == "model":
            self.on_open_model_file()
        elif action == "qpos":
            self.viewer_3d.choose_qpos_csv()
        elif action == "trajectory":
            self.viewer_3d.choose_trajectory_csv(prompt_import_dt=True)

    def on_setup_export_requested(self, action):
        if action == "qpos":
            self.viewer_3d.choose_qpos_save_path()
        elif action == "trajectory":
            self.viewer_3d.choose_trajectory_save_path()

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
        if self.model_importer is not None and self.model_importer.isRunning():
            self.statusBar().showMessage("A robot model import is already running.")
            return self.model_importer
        self._model_import_retry_prompted = False
        mesh_roots = [self.import_mesh_folder] if self.import_mesh_folder else []
        return self._start_model_import(path, mesh_roots)

    def _start_model_import(self, path, mesh_roots):
        self.begin_render_progress(
            "Importing robot model",
            f"Copying and preparing {Path(path).name} in the background...",
        )
        importer = ModelImportThread(
            path,
            self.model_library_root,
            mesh_roots=mesh_roots,
            parent=self,
        )
        importer.imported.connect(self.on_model_imported)
        importer.failed.connect(self.on_model_import_failed)
        self._background_workers.add(importer)
        importer.finished.connect(
            lambda importer=importer: self._background_worker_finished(importer)
        )
        importer.finished.connect(importer.deleteLater)
        self.model_importer = importer
        importer.start()
        return importer

    def on_model_imported(self, info):
        self.model_importer = None
        self.finish_render_progress()
        info = self.register_model_info(info)
        self.status_text.append(
            f"Imported {info.display_name} to {info.model_path}"
        )
        self.controls.add_model(info.key, info.display_name, select=True)

    def on_model_import_failed(self, error, can_retry_with_mesh_folder=False):
        importer = self.model_importer
        source_path = None if importer is None else importer.source_path
        self.model_importer = None
        self.finish_render_progress()
        if can_retry_with_mesh_folder and not self._model_import_retry_prompted:
            self._model_import_retry_prompted = True
            mesh_folder = self._prompt_for_mesh_folder()
            if mesh_folder is not None and source_path is not None:
                self._start_model_import(source_path, [mesh_folder])
                return
        message = f"Could not import model: {error}"
        self.status_text.setText(message)
        QMessageBox.warning(self, "Import model failed", message)

    def finish_model_loading_ui(self):
        self.controls.model_box.setEnabled(True)
        self.statusBar().clearMessage()

    def activate_model_session(self, model_key, session, actor_id=None):
        previous_model_key = self.model_key
        previous_viewer = self.viewer_3d
        actor_id = str(actor_id or self.editor_robot_actor_id)
        restoring_project = (
            self._pending_project_restore is not None
            and self.project_restore_model_key(
                self._pending_project_restore,
                self._pending_project_restore_autosave,
            ) == model_key
        )
        self.sync_current_robot_session()
        model_sessions.remember_current_session(
            self.model_sessions,
            self.current_robot_session_key(),
            self.trajectory,
            self.active_index,
        )
        self.editor_robot_actor_id = actor_id
        self.scene.metadata["editor_robot_actor_id"] = actor_id
        for name, value in model_sessions.activated_session_state(
            model_key, session
        ).items():
            setattr(self, name, value)
        self.set_scene_active_robot_model(model_key, self.current_model_display_name())
        actor = self.scene.actors.require(actor_id)
        actor.metadata["selected_frame"] = session.selected_frame
        self.scene.select_actor(actor_id, frame_id=session.selected_frame)
        self.viewer_3d_stack.setCurrentWidget(self.viewer_3d)
        if previous_viewer is not self.viewer_3d:
            previous_viewer.detach_canvas()
        active_actor = self.scene.actors.get(actor_id)
        self.viewer_3d.attach_canvas(
            self.shared_scene_canvas,
            scene=self.scene,
            active_robot_actor_id=actor_id,
            scene_edit_target=(
                self.scene_edit_target()
                if active_actor is not None
                else None
            ),
        )
        self.viewer_2d_skeleton_stack.setCurrentWidget(self.viewer_2d_stickman)
        self.viewer_3d_mujoco.set_model_adapter(session.adapter)
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.set_editor_timeline_duration(self.viewer_3d.timeline_duration)
        self.model_source_label.setText(self.model_source_text(session.adapter))
        self.begin_render_progress(
            f"Rendering {session.adapter.model_name}",
            "Preparing the 3D model geometry...",
            viewer=self.viewer_3d,
        )
        self.controls.set_frame_names(session.adapter.trajectory_frames)
        self.update_project_chrome()
        self.update_editor_context()
        self.refresh_display(apply_stickman_frame=False)
        self.request_active_model_render()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_history_baseline()
        if previous_model_key != model_key and not restoring_project:
            self.mark_project_dirty("Change robot model")
        self.restore_pending_project_if_ready()

    def on_trajectory_csv_loaded(self, csv_path):
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)
        if self.viewer_3d.consume_trajectory_import_dt_prompt_request():
            self.prompt_trajectory_import_dt()
        count = self.import_loaded_robot_trajectory_as_keyframes()
        import_dt = self.viewer_3d.trajectory_import_dt.value()
        self.status_text.setText(
            f"Loaded trajectory CSV: {csv_path}\n"
            f"Imported {count} editable target-frame keyframes from FK "
            f"at {import_dt:.2f} s intervals."
        )
        self.record_history_action("Load trajectory")

    def prompt_trajectory_import_dt(self):
        current = self.viewer_3d.trajectory_import_dt.value()
        value, accepted = QInputDialog.getDouble(
            self,
            "Import trajectory time step",
            "Editable keyframe interval [s]",
            current,
            0.01,
            10.0,
            2,
        )
        if accepted:
            self.viewer_3d.trajectory_import_dt.setValue(value)
            self.mark_project_dirty("Import time step")

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
        self.scene.timeline.current_time = float(time)
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
        self.viewer_3d.set_current_time(time)
        self.refresh_display()
        self._refresh_history_baseline()

    def on_viewer_timeslice_time_changed(self, time):
        """Keep the sidebar time editor in sync with the viewer-bottom scrubber."""
        self.scene.timeline.current_time = float(time)
        self.controls.time_slider.set_value(time)
        self.refresh_display()
        self.on_time_changed(time)

    def on_viewer_timeline_duration_changed(self, duration):
        self.scene.timeline.duration = float(duration)
        self.set_sidebar_timeline_duration(duration)
        self.mark_project_dirty("Timeline duration")

    def set_sidebar_timeline_duration(self, duration):
        self.controls.time_slider.set_range(0, int(round(float(duration) * 100.0)))

    def set_editor_timeline_duration(self, duration):
        self.viewer_3d.set_timeline_duration(duration, emit_signal=False)
        self.set_sidebar_timeline_duration(duration)

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
            self.viewer_3d.clear_pinned_frame_targets()
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

    def on_scene_actor_transform_dragged(self, actor_id, position, quaternion):
        self.apply_scene_actor_transform(actor_id, position, quaternion)
        actor = self.scene.actors.get(actor_id)
        if actor is not None:
            self.statusBar().showMessage(f"Moving object: {actor.name}", 500)

    def on_scene_actor_transform_drag_finished(self, actor_id, position, quaternion):
        actor = self.apply_scene_actor_transform(actor_id, position, quaternion)
        self.refresh_display(apply_stickman_frame=False)
        if actor is not None:
            self.record_history_action(f"Move object {actor.name}")

    def apply_scene_actor_transform(self, actor_id, position, quaternion):
        actor = self.scene.actors.get(actor_id)
        if actor is None or actor.kind != "object" or actor.locked:
            return None
        transform = Transform(
            position=tuple(float(value) for value in position),
            quaternion=tuple(float(value) for value in quaternion),
        )
        actor.world_transform = transform
        track = self.scene.tracks.object_transforms.get(actor.id, [])
        if track:
            self.scene.tracks.add_object_transform_keyframe(
                actor.id,
                TransformKeyframe(
                    self.scene.timeline.current_time,
                    transform,
                ),
            )
        self.viewer_3d.canvas.update()
        return actor

    def on_3d_target_frame_changed(self, frame_name):
        """Map common 3D body/site selections back to the 2D frame concept."""
        actor = self.scene.actors.get(self.editor_robot_actor_id)
        if actor is not None:
            actor.metadata["selected_frame"] = frame_name
            self.scene.select_actor(actor.id, frame_id=frame_name)
            session = self.current_robot_session()
            if session is not None:
                session.selected_frame = frame_name
        self.controls.frame_box.blockSignals(True)
        self.controls.frame_box.setCurrentText(frame_name)
        self.controls.frame_box.blockSignals(False)
        self.set_current_frame_to_active_robot_pose(
            frame_name,
            emit_pose_changed=False,
        )
        self.refresh_display(apply_stickman_frame=False)

    def on_scene_robot_body_double_clicked(self, actor_id, body_name):
        actor = self.scene.actors.get(actor_id)
        if actor is None or actor.kind != "robot" or not actor.visible:
            return
        try:
            adapter = self.scene_robot_adapter(actor)
        except Exception as exc:
            self.status_text.append(f"Could not select robot {actor.name}: {exc}")
            return
        logical = (
            adapter.logical_frame_for_body(body_name)
            if hasattr(adapter, "logical_frame_for_body")
            else None
        )
        self.select_scene_actor(actor.id, frame_id=logical)
        self.refresh_scene_tree()
        if logical is None:
            self.statusBar().showMessage(
                f"Selected robot {actor.name}; {body_name} has no editable frame.",
                3000,
            )
            return
        self.statusBar().showMessage(
            f"Selected {actor.name}: {logical}",
            2000,
        )

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
        self.on_trajectory_display_changed(checked)

    def on_trajectory_display_changed(self, checked):
        self.refresh_display()
        self.mark_project_dirty("Display settings")

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

        The target controls should jump to the active actor's current MuJoCo
        body/site pose without changing its committed or preview state.
        """

        if self._syncing_editor_selection:
            return
        actor = self.scene.actors.get(self.editor_robot_actor_id)
        if actor is not None:
            actor.metadata["selected_frame"] = frame_name
            self.scene.select_actor(actor.id, frame_id=frame_name)
            session = self.current_robot_session()
            if session is not None:
                session.selected_frame = frame_name

        if self.set_current_frame_to_active_robot_pose(
            frame_name,
            emit_pose_changed=False,
        ):
            self.refresh_display(apply_stickman_frame=False)
            self._refresh_history_baseline()
            return

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

    def set_current_frame_to_active_robot_pose(
        self,
        frame_name,
        emit_pose_changed=True,
    ):
        viewer = self.viewer_3d
        binding = viewer.frame_bindings.get(frame_name)
        state = (
            viewer.preview_state
            if viewer.preview_active
            else viewer.committed_state
        )
        if binding is None or state is None:
            return False

        kind, name = binding
        try:
            position, quaternion = state.get_body_pose(name, kind)
        except KeyError:
            return False

        viewer.select_target(kind, name, emit=False)
        viewer._set_target_to_selected_pose()
        roll, pitch, yaw = quat_to_rpy(quaternion)
        self.controls.set_position_values(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            emit_pose_changed=emit_pose_changed,
        )
        return True

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
        show_keyframes = self.controls.show_keyframes()
        show_trajectory_lines = self.controls.show_trajectory_lines()
        trajectory_smoothing = self.controls.corner_smoothing()

        self.viewer_2d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
            show_keyframes=show_keyframes,
        )
        self.viewer_3d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            scene=self.scene,
            scene_asset_root=(
                None if self.current_project is None
                else self.current_project.root_dir
            ),
            scene_edit_actor_id=self.scene_edit_actor_id(),
            scene_robot_adapters=self.scene_extra_robot_adapters(),
            scene_edit_target=self.scene_edit_target(),
            active_robot_actor_id=self.editor_robot_actor_id,
            scene_robot_states=self.scene_robot_render_states(),
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
            show_keyframes=show_keyframes,
        )
        self.viewer_2d_stickman.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            apply_active_frame=apply_stickman_frame,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
            show_keyframes=show_keyframes,
        )

        self.controls.refresh_table(self.trajectory)
        self.refresh_scene_tree()
        self.viewer_3d.set_defined_timeslices(
            sorted({frame.time for frame in self.trajectory.frames})
        )

        self.backend_label.setText(
            f"Backend: {self.backend_interface.backend_name()}"
        )
        self.sync_viewer_status_panel()

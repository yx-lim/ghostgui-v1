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

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QProgressDialog,
    QFileDialog,
    QMessageBox,
)

from .trajectory import (
    TargetFrame,
    Trajectory,
    SampledTrajectory,
    quat_to_rpy,
    rpy_to_quat,
)
from .controls import TrajectoryControlPanel
from .viewer_2d import RobotCanvas
from .robot_viewer_3d import RobotViewer3D
from .viewer_2d_stickman import Stickman2DViewer
from .viewer_3d_mujoco import Mujoco3DViewerPanel
from .backend_interface import BackendInterface
from .model_reference import MujocoReferenceFrames
from .app_sidebars import AppLeftSidebar, AppRightSidebar
from .robot_model_adapter import MuJoCoRobotAdapter
from .robot_model_registry import ROBOT_MODELS
from .model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)


LEFT_SIDEBAR_WIDTH = 250
RIGHT_SIDEBAR_WIDTH = 270


@dataclass
class RobotModelSession:
    adapter: object
    backend: object
    reference: object
    viewer_3d: object
    viewer_2d_skeleton: object
    trajectory: object
    active_index: int = -1


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
            model_key: RobotModelSession(
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
        self.model_loading_dialog = None
        self.viewer_tabs = self.build_viewer_tabs()
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

        # Initial view
        self.update_editor_context()
        self.refresh_display()

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
        layout.addWidget(self.status_text)

        panel.setLayout(layout)
        return panel

    def build_viewer_tabs(self):
        tabs = QTabWidget()
        self.viewer_3d_stack = QStackedWidget()
        self.viewer_3d_stack.addWidget(self.viewer_3d)
        self.viewer_2d_skeleton_stack = QStackedWidget()
        self.viewer_2d_skeleton_stack.addWidget(self.viewer_2d_stickman)
        tabs.addTab(self.viewer_2d, "2D Side View")
        tabs.addTab(self.viewer_3d_stack, "3D Pose")
        tabs.addTab(self.viewer_2d_skeleton_stack, "2D Skeleton")
        tabs.addTab(self.viewer_3d_mujoco, "Simulation")
        tabs.currentChanged.connect(self.update_editor_context)
        return tabs

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
        viewer_3d.timeslice_time_changed.connect(
            self.on_viewer_timeslice_time_changed
        )
        viewer_3d.accept_timeslice_requested.connect(
            self.on_accept_timeslice_requested
        )
        viewer_3d.delete_timeslice_requested.connect(
            self.on_delete_timeslice_requested
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
        self.controls.model_box.setEnabled(False)
        self.statusBar().showMessage(f"Loading {model_info.display_name}…")
        self.model_loading_dialog = QProgressDialog(
            f"Loading {model_info.display_name}…",
            None, 0, 0, self,
        )
        self.model_loading_dialog.setWindowTitle("Loading robot model")
        self.model_loading_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.model_loading_dialog.setMinimumDuration(0)
        self.model_loading_dialog.show()
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
        session = RobotModelSession(
            adapter, backend, reference, viewer_3d, viewer_2d_skeleton,
            Trajectory(), -1,
        )
        self.model_sessions[model_key] = session
        self.finish_model_loading_ui()
        self.activate_model_session(model_key, session)

    def on_model_load_failed(self, model_key, error):
        self.model_loaders.pop(model_key, None)
        self.finish_model_loading_ui()
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
        try:
            try:
                info = import_robot_model(
                    path, self.model_library_root, mesh_roots=mesh_roots
                )
            except RuntimeError:
                mesh_folder = self._prompt_for_mesh_folder()
                if mesh_folder is None:
                    raise
                info = import_robot_model(
                    path, self.model_library_root, mesh_roots=[mesh_folder]
                )
            info = self.register_model_info(info)
        except Exception as exc:
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
        if self.model_loading_dialog is not None:
            self.model_loading_dialog.close()
            self.model_loading_dialog.deleteLater()
            self.model_loading_dialog = None

    def activate_model_session(self, model_key, session):
        current = self.model_sessions.get(self.model_key)
        if current is not None:
            current.trajectory = self.trajectory
            current.active_index = self.active_index
        self.model_key = model_key
        self.robot_model_3d = session.adapter
        self.robot_model_error = session.adapter.load_warning
        self.backend_interface = session.backend
        self.model_reference = session.reference
        self.viewer_3d = session.viewer_3d
        self.viewer_2d_stickman = session.viewer_2d_skeleton
        self.trajectory = session.trajectory
        self.active_index = session.active_index
        self.viewer_3d_stack.setCurrentWidget(self.viewer_3d)
        self.viewer_2d_skeleton_stack.setCurrentWidget(self.viewer_2d_stickman)
        self.viewer_3d_mujoco.set_model_adapter(session.adapter)
        self.controls.set_frame_names(session.adapter.trajectory_frames)
        self.setWindowTitle(
            f"Reference Frame Trajectory GUI — {session.adapter.model_name}"
        )
        self.update_editor_context()
        self.refresh_display(apply_stickman_frame=False)

    def on_trajectory_csv_loaded(self, csv_path):
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)
        count = self.import_loaded_robot_trajectory_as_keyframes()
        import_dt = self.viewer_3d.trajectory_import_dt.value()
        self.status_text.setText(
            f"Loaded trajectory CSV: {csv_path}\n"
            f"Imported {count} editable target-frame keyframes from FK "
            f"at {import_dt:.2f} s intervals."
        )

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
        message = (
            f"Accepted slice at t={time:.2f} s; captured {count} logical targets "
            "from the committed solved pose."
        )
        self.viewer_3d.status_label.setText(message)
        self.status_text.setText(message)

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

    def define_timeslice_from_committed_pose(self):
        """Snapshot every editable logical target from the committed MuJoCo pose."""
        state = self.viewer_3d.committed_state
        if state is None:
            return 0

        time = self.viewer_3d.get_current_time()
        phase = self.controls.phase_box.currentText()
        selected_frame_name = self.controls.frame_box.currentText()
        selected_frame = None
        selected_index = -1
        last_index = -1
        count = 0

        frame_names = self.editable_logical_frame_names()

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
                time=time,
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
            if frame_name == selected_frame_name:
                selected_frame = frame
                selected_index = last_index
            count += 1

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
        return count

    def editable_logical_frame_names(self):
        frame_names = []
        for name in getattr(self.robot_model_3d, "trajectory_frames", []):
            if name not in frame_names:
                frame_names.append(name)
        for name in self.controls.frame_names:
            if name not in frame_names:
                frame_names.append(name)
        for name in self.viewer_3d.frame_bindings:
            if name not in frame_names:
                frame_names.append(name)
        return frame_names

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
        if not times or not qposes:
            return []

        interval = max(0.0, float(self.viewer_3d.trajectory_import_dt.value()))
        samples = []
        last_import_time = None
        for time, qpos in zip(times, qposes):
            time = float(time)
            if (
                last_import_time is None
                or interval <= 1e-9
                or time >= last_import_time + interval - 1e-9
            ):
                samples.append((time, qpos))
                last_import_time = time

        final_time = float(times[-1])
        if abs(samples[-1][0] - final_time) > 1e-9:
            samples.append((final_time, qposes[-1]))

        return samples

    def delete_timeslice_at_time(self, time, tolerance=1e-6):
        count = 0
        for track in self.trajectory.tracks.values():
            kept = []
            for frame in track:
                if abs(frame.time - time) <= tolerance:
                    count += 1
                else:
                    kept.append(frame)
            track[:] = kept
        return count

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
        self.viewer_3d.clear_editable_timeline(keep_current_pose=True)
        self.viewer_3d.set_defined_timeslices([])
        self.refresh_display()
        message = (
            f"Cleared {keyframe_count} trajectory keyframes; "
            "current robot pose was left unchanged."
        )
        self.viewer_3d.status_label.setText(message)
        self.status_text.setText(message)

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
            return

        x, z = self.viewer_2d_stickman.get_body_point(frame_name)

        self.controls.set_position_from_viewer(
            x,
            z,
            emit_pose_changed=False,
        )

        self.refresh_display(apply_stickman_frame=False)

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

        export_dt = 0.01
        smoothing = self.controls.corner_smoothing()

        sampled_tracks = self.trajectory.sample_tracks_uniform_dt(
            dt=export_dt,
            smoothing=smoothing,
        )
        sampled_trajectory = SampledTrajectory(samples=sampled_tracks)

        result_states = self.backend_interface.solve_trajectory(sampled_trajectory)
        self.viewer_3d.load_backend_states(result_states)

        csv_path = "pelvis_base_trajectory_uniform_dt.csv"
        self.backend_interface.export_last_solution_csv(csv_path)
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)

        lines = []
        lines.append("Generated uniformly sampled per-frame target tracks.")
        lines.append(f"Backend: {self.backend_interface.last_backend_name()}")
        lines.append(f"Export dt: {export_dt:.4f} s")
        lines.append(f"Corner smoothing: {smoothing * 100.0:.0f}%")
        lines.append(f"Number of GUI keyframes: {len(self.trajectory.frames)}")
        lines.append(f"Number of sampled time steps: {len(sampled_tracks)}")
        lines.append(f"Number of backend states: {len(result_states)}")
        if result_states:
            max_ik_error = max(state.ik_error for state in result_states)
            lines.append(f"Max IK position error: {max_ik_error:.4f} m")
            max_orientation_error = max(
                state.orientation_error for state in result_states
            )
            lines.append(
                "Max IK orientation error: "
                f"{max_orientation_error:.4f} rad"
            )
        lines.append(f"Exported CSV to: {csv_path}")
        lines.append("")
        lines.append("First few sampled time groups:")
        lines.append("")

        for sample in sampled_tracks[:10]:
            frame_names = ", ".join(sorted(sample["targets"].keys()))
            lines.append(f"t={sample['time']:.3f}s | targets={frame_names}")

        self.status_text.setText("\n".join(lines))

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

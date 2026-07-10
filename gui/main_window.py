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
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QGroupBox,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QProgressDialog,
    QFileDialog,
    QMessageBox,
)

from .trajectory import Trajectory, SampledTrajectory, quat_to_rpy, rpy_to_quat
from .controls import TrajectoryControlPanel
from .viewer_2d import RobotCanvas
from .robot_viewer_3d import RobotViewer3D
from .viewer_2d_stickman import Stickman2DViewer
from .viewer_3d_mujoco import Mujoco3DViewerPanel
from .backend_interface import BackendInterface
from .model_reference import MujocoReferenceFrames
from .collapsible_sidebar import CollapsibleSidebar
from .robot_model_adapter import MuJoCoRobotAdapter
from .robot_model_registry import ROBOT_MODELS
from .model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)


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
        self.left_sidebar = CollapsibleSidebar(
            "Frames",
            self.controls,
            side="left",
            minimum_expanded_width=300,
            maximum_expanded_width=560,
        )
        self.right_sidebar = CollapsibleSidebar(
            "Status",
            self.status_panel,
            side="right",
            minimum_expanded_width=350,
            maximum_expanded_width=520,
        )

        self.connect_signals()
        self.set_current_frame_to_model_reference(
            self.controls.frame_box.currentText(),
            emit_pose_changed=False,
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------
        # Persistent splitter children resize/hide in place. The 3D viewer is
        # never recreated, so collapsing a sidebar retains its OpenGL context.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        self.main_splitter.addWidget(self.left_sidebar)
        self.main_splitter.addWidget(self.viewer_tabs)
        self.main_splitter.addWidget(self.right_sidebar)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([380, 900, 390])
        self.main_splitter.splitterMoved.connect(self.remember_sidebar_widths)
        self.setCentralWidget(self.main_splitter)

        self.toggle_left_shortcut = QShortcut(QKeySequence("Ctrl+["), self)
        self.toggle_left_shortcut.activated.connect(
            self.left_sidebar.toggle_collapsed
        )
        self.toggle_right_shortcut = QShortcut(QKeySequence("Ctrl+]"), self)
        self.toggle_right_shortcut.activated.connect(
            self.right_sidebar.toggle_collapsed
        )

        # Initial view
        self.refresh_display()

    def remember_sidebar_widths(self, position=None, index=None):
        sizes = self.main_splitter.sizes()
        if len(sizes) == 3:
            self.left_sidebar.remember_width(sizes[0])
            self.right_sidebar.remember_width(sizes[2])

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
        panel = QGroupBox("Trajectory / Backend Status")
        layout = QVBoxLayout()

        self.backend_label = QLabel()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumWidth(330)

        layout.addWidget(self.backend_label)
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
        tabs.addTab(self.viewer_3d_stack, "3D View")
        tabs.addTab(self.viewer_2d_skeleton_stack, "2D Skeleton")
        tabs.addTab(self.viewer_3d_mujoco, "3D MuJoCo")
        return tabs

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
        viewer_2d_skeleton.target_dragged.connect(self.on_target_dragged)

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
        self.refresh_display(apply_stickman_frame=False)

    def on_trajectory_csv_loaded(self, csv_path):
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)
        self.status_text.setText(f"Loaded trajectory CSV: {csv_path}")

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

        sampled_tracks = self.trajectory.sample_tracks_uniform_dt(dt=export_dt)
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

        self.viewer_2d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            show_trajectory_lines=show_trajectory_lines,
        )
        self.viewer_3d.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            show_trajectory_lines=show_trajectory_lines,
        )
        self.viewer_2d_stickman.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
            apply_active_frame=apply_stickman_frame,
            show_trajectory_lines=show_trajectory_lines,
        )

        self.controls.refresh_table(self.trajectory)

        self.backend_label.setText(
            f"Backend: {self.backend_interface.backend_name()}"
        )

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

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

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QGroupBox,
    QTabWidget,
)

from .trajectory import Trajectory
from .trajectory import SampledTrajectory
from .controls import TrajectoryControlPanel
from .viewer_2d import RobotCanvas
from .viewer_3d import RobotCanvas3D
from .viewer_2d_stickman import Stickman2DViewer
from .viewer_3d_mujoco import Mujoco3DViewerPanel
from .backend_interface import BackendInterface
from .model_reference import MujocoReferenceFrames


class RobotGuiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Reference Frame Trajectory GUI")

        # --------------------------------------------------------
        # Core data
        # --------------------------------------------------------
        self.trajectory = Trajectory()
        self.active_index = -1

        # Backend
        self.backend_interface = BackendInterface()
        self.model_reference = MujocoReferenceFrames()

        # GUI widgets
        self.controls = TrajectoryControlPanel()
        self.viewer_2d = RobotCanvas()
        self.viewer_3d = RobotCanvas3D()
        self.viewer_2d_stickman = Stickman2DViewer()
        self.viewer_3d_mujoco = Mujoco3DViewerPanel()
        self.viewer_tabs = self.build_viewer_tabs()
        self.status_panel = self.build_status_panel()

        self.connect_signals()
        self.set_current_frame_to_model_reference("pelvis", emit_pose_changed=False)

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------
        central = QWidget()
        layout = QHBoxLayout()

        layout.addWidget(self.controls)
        layout.addWidget(self.viewer_tabs, stretch=1)
        layout.addWidget(self.status_panel)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # Initial view
        self.refresh_display()

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
        tabs.addTab(self.viewer_2d, "2D Side View")
        tabs.addTab(self.viewer_3d, "3D View")
        tabs.addTab(self.viewer_2d_stickman, "2D Stickman")
        tabs.addTab(self.viewer_3d_mujoco, "3D MuJoCo")
        return tabs

    # ============================================================
    # Signal connections
    # ============================================================

    def connect_signals(self):
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

        self.viewer_2d.target_dragged.connect(self.on_target_dragged)
        self.viewer_3d.target_dragged.connect(self.on_target_dragged)
        self.viewer_2d_stickman.target_dragged.connect(self.on_target_dragged)

    # ============================================================
    # GUI interaction callbacks
    # ============================================================

    def on_pose_changed(self, x, y, z, yaw):
        """
        Called when sliders change.

        If a keyframe is selected, we only preview the target frame.
        The actual keyframe is overwritten only when user clicks Update.
        """

        self.refresh_display()

    def on_target_dragged(self, x, z):
        """
        Called when user drags the red reference frame in the viewer.

        This updates the sliders, so the GUI stays consistent.
        """

        self.controls.set_position_from_viewer(x, z)

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
        position = self.model_reference.position_for_frame(frame_name)

        if position is None:
            return False

        x, y, z = position
        self.controls.set_position_values(
            x=x,
            y=y,
            z=z,
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

        csv_path = "pelvis_base_trajectory_uniform_dt.csv"
        self.backend_interface.export_last_solution_csv(csv_path)
        self.viewer_3d_mujoco.set_trajectory_csv(csv_path)

        lines = []
        lines.append("Generated uniformly sampled per-frame target tracks.")
        lines.append(f"Export dt: {export_dt:.4f} s")
        lines.append(f"Number of GUI keyframes: {len(self.trajectory.frames)}")
        lines.append(f"Number of sampled time steps: {len(sampled_tracks)}")
        lines.append(f"Number of backend states: {len(result_states)}")
        if result_states:
            max_ik_error = max(state.ik_error for state in result_states)
            lines.append(f"Max IK position error: {max_ik_error:.4f} m")
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

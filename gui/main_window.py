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
)

from .trajectory import Trajectory
from .trajectory import SampledTrajectory
from .controls import TrajectoryControlPanel
from .viewer import RobotCanvas
from .backend_interface import BackendInterface


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

        # GUI widgets
        self.controls = TrajectoryControlPanel()
        self.canvas = RobotCanvas()
        self.status_panel = self.build_status_panel()

        self.connect_signals()

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------
        central = QWidget()
        layout = QHBoxLayout()

        layout.addWidget(self.controls)
        layout.addWidget(self.canvas, stretch=1)
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

        self.canvas.target_dragged.connect(self.on_target_dragged)

    # ============================================================
    # GUI interaction callbacks
    # ============================================================

    def on_pose_changed(self, x, z, yaw):
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

    def on_generate_trajectory(self):
        if len(self.trajectory.frames) == 0:
            self.status_text.setText("Trajectory is empty. Add keyframes first.")
            return

        export_dt = 0.01

        sampled_frames = self.trajectory.sample_uniform_dt(dt=export_dt)
        sampled_trajectory = SampledTrajectory(sampled_frames)

        result_states = self.backend_interface.solve_trajectory(sampled_trajectory)

        csv_path = "pelvis_base_trajectory_uniform_dt.csv"
        self.backend_interface.export_last_solution_csv(csv_path)

        lines = []
        lines.append("Generated uniformly sampled q(t) trajectory.")
        lines.append(f"Export dt: {export_dt:.4f} s")
        lines.append(f"Number of GUI keyframes: {len(self.trajectory.frames)}")
        lines.append(f"Number of exported samples: {len(sampled_frames)}")
        lines.append(f"Exported CSV to: {csv_path}")
        lines.append("")
        lines.append("First few sampled frames:")
        lines.append("")

        for frame in sampled_frames[:10]:
            lines.append(
                f"t={frame.time:.3f}s | "
                f"frame={frame.frame_name} | "
                f"x={frame.x:.3f}, y={frame.y:.3f}, z={frame.z:.3f}"
            )

        self.status_text.setText("\n".join(lines))

    # ============================================================
    # Display update
    # ============================================================

    def refresh_display(self):
        """
        Refresh viewer, table, and status text.
        """

        active_frame = self.controls.current_frame()

        self.canvas.update_scene(
            trajectory=self.trajectory,
            active_frame=active_frame,
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
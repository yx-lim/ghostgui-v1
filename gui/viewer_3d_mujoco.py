"""
3D viewer panel.

For now, this launches the MuJoCo viewer in a separate process.

Reason:
    Embedding MuJoCo directly inside PySide6 is possible,
    but launching it separately is much simpler and more stable
    for the first version.
"""

from pathlib import Path
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
)


class Mujoco3DViewerPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.process = None

        self.project_root = Path(__file__).resolve().parents[1]
        self.viewer_script = self.project_root / "scripts" / "view_g1_mujoco.py"
        self.model_path = self.project_root / "models" / "g1_29dof.xml"

        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout()

        self.title = QLabel("3D MuJoCo Viewer: G1 29-DoF")
        self.model_label = QLabel(f"Model: {self.model_path}")

        self.open_button = QPushButton("Open G1 MuJoCo Viewer")
        self.close_button = QPushButton("Close MuJoCo Viewer")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.open_button.clicked.connect(self.open_viewer)
        self.close_button.clicked.connect(self.close_viewer)

        layout.addWidget(self.title)
        layout.addWidget(self.model_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.close_button)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def open_viewer(self):
        if not self.model_path.exists():
            self.log_box.append(f"Model file not found: {self.model_path}")
            return

        if not self.viewer_script.exists():
            self.log_box.append(f"Viewer script not found: {self.viewer_script}")
            return

        if self.process is not None:
            self.log_box.append("MuJoCo viewer is already running.")
            return

        self.process = QProcess(self)

        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.on_process_finished)

        self.log_box.append("Launching MuJoCo viewer...")

        self.process.start(
            sys.executable,
            [str(self.viewer_script)],
        )

    def close_viewer(self):
        if self.process is None:
            self.log_box.append("No MuJoCo viewer process is running.")
            return

        self.log_box.append("Closing MuJoCo viewer...")
        self.process.terminate()

    def read_stdout(self):
        if self.process is None:
            return

        output = bytes(self.process.readAllStandardOutput()).decode("utf-8")
        self.log_box.append(output.strip())

    def read_stderr(self):
        if self.process is None:
            return

        output = bytes(self.process.readAllStandardError()).decode("utf-8")
        self.log_box.append(output.strip())

    def on_process_finished(self):
        self.log_box.append("MuJoCo viewer closed.")
        self.process = None
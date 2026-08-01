"""Focused installed-runtime contracts for projects and MuJoCo Simulation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from application import paths
from application import mujoco_viewer_process
from application.project_manager import ghostgui_projects_dir
from gui.viewers import mujoco_player
from scripts import view_g1_mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InstalledProjectPathTests(unittest.TestCase):
    def test_platform_user_data_default_matches_native_convention(self):
        with patch.dict(
            os.environ,
            {"GHOSTGUI_USER_DATA_DIR": ""},
            clear=False,
        ):
            actual = paths.ghostgui_user_data_dir()

        if sys.platform == "darwin":
            expected = Path.home() / "Library" / "Application Support" / "GhostGUI"
        elif sys.platform.startswith("win") and os.environ.get("LOCALAPPDATA"):
            expected = Path(os.environ["LOCALAPPDATA"]) / "GhostGUI"
        elif os.environ.get("XDG_DATA_HOME"):
            expected = Path(os.environ["XDG_DATA_HOME"]) / "ghostgui"
        else:
            expected = Path.home() / ".local" / "share" / "ghostgui"
        self.assertEqual(actual, expected)

    def test_checkout_default_remains_checkout_projects_folder(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            paths,
            "is_source_checkout",
            return_value=True,
        ):
            self.assertEqual(
                ghostgui_projects_dir(),
                paths.PROJECT_ROOT / "projects",
            )

    def test_installed_default_uses_writable_data_projects_folder(self):
        user_data_root = Path("/runtime/user-data")
        with patch.dict(os.environ, {}, clear=True), patch.object(
            paths,
            "is_source_checkout",
            return_value=False,
        ), patch.object(
            paths,
            "ghostgui_user_data_dir",
            return_value=user_data_root,
        ):
            self.assertEqual(
                ghostgui_projects_dir(),
                user_data_root / "projects",
            )

    def test_projects_directory_override_has_highest_precedence(self):
        override = "/runtime/explicit-projects"
        with patch.dict(
            os.environ,
            {"GHOSTGUI_PROJECTS_DIR": override},
            clear=True,
        ):
            self.assertEqual(ghostgui_projects_dir(), Path(override))


class _Signal:
    def connect(self, callback):
        self.callback = callback


class _Process:
    def __init__(self, parent):
        self.parent = parent
        self.readyReadStandardOutput = _Signal()
        self.readyReadStandardError = _Signal()
        self.finished = _Signal()
        self.executable = None
        self.arguments = None

    def start(self, executable, arguments):
        self.executable = executable
        self.arguments = list(arguments)


class InstalledViewerProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_legacy_script_reexports_installed_viewer_api(self):
        self.assertIs(
            view_g1_mujoco.TrajectoryPlayer,
            mujoco_viewer_process.TrajectoryPlayer,
        )
        self.assertIs(
            view_g1_mujoco.load_trajectory_csv,
            mujoco_viewer_process.load_trajectory_csv,
        )

    def test_gui_launches_packaged_viewer_module(self):
        panel = mujoco_player.Mujoco3DViewerPanel()
        try:
            with patch.object(
                panel,
                "_ensure_trajectory_file",
                return_value=False,
            ), patch.object(
                panel,
                "_viewer_python_executable",
                return_value="/runtime/python",
            ), patch.object(mujoco_player, "QProcess", _Process):
                panel.open_viewer()

            self.assertEqual(panel.process.executable, "/runtime/python")
            self.assertEqual(
                panel.process.arguments,
                [
                    "-m",
                    "application.mujoco_viewer_process",
                    "--model",
                    str(panel.model_path),
                ],
            )
        finally:
            panel.process = None
            panel.deleteLater()


class MacInstallerTests(unittest.TestCase):
    def test_missing_mjpython_warns_without_failing_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts_dir = root / "scripts"
            fake_bin = root / "fake-bin"
            venv_bin = root / ".venv" / "bin"
            scripts_dir.mkdir()
            fake_bin.mkdir()
            venv_bin.mkdir(parents=True)
            shutil.copy2(
                PROJECT_ROOT / "scripts" / "install_macos.sh",
                scripts_dir / "install_macos.sh",
            )
            (venv_bin / "activate").write_text("# test environment\n")
            self._write_executable(
                fake_bin / "uname",
                "#!/bin/sh\necho x86_64\n",
            )
            python_stub = (
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then\n"
                "  echo 'Python 3.13.0'\n"
                "elif [ \"$1\" = \"-c\" ]; then\n"
                "  echo x86_64\n"
                "fi\n"
            )
            self._write_executable(fake_bin / "python3", python_stub)
            self._write_executable(fake_bin / "python", python_stub)
            environment = dict(os.environ)
            environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"

            result = subprocess.run(
                ["bash", str(scripts_dir / "install_macos.sh")],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WARNING: mjpython was not found", result.stdout)
        self.assertIn("Simulation: unavailable", result.stdout)
        self.assertIn("GhostGUI installed successfully", result.stdout)

    @staticmethod
    def _write_executable(path, content):
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()

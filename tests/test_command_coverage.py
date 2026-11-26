"""Ratchet: every named GUI control must map to a registry command or stay in the baseline."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from application.commands import default_registry
from application.commands.coverage import coverage, load_baseline, ratchet_errors, write_baseline
from gui.commands import COMMAND_PROPERTY, collect_controls
from gui.main_window import RobotGuiMainWindow

BASELINE = Path(__file__).resolve().parent / "fixtures" / "command_coverage_baseline.json"


class CommandCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        environment = patch.dict(os.environ, {
            "GHOSTGUI_CONFIG_DIR": str(root / "config"), "GHOSTGUI_CACHE_DIR": str(root / "cache"),
            "GHOSTGUI_PROJECTS_DIR": str(root / "projects"), "GHOSTGUI_USER_DATA_DIR": str(root / "data"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.window = RobotGuiMainWindow()
        self.addCleanup(self._close)

    def _close(self):
        self.window.agent_bridge_controller.shutdown()
        self.window.set_project_dirty(False)
        self.window.close()
        self.app.processEvents()

    def test_ratchet(self):
        controls = collect_controls(self.window)
        report = coverage(controls, default_registry())
        if os.environ.get("GHOSTGUI_UPDATE_COVERAGE_BASELINE") == "1":
            write_baseline(BASELINE, report.unbound)
        self.assertEqual(ratchet_errors(report, load_baseline(BASELINE)), [])

    def test_bound_controls_are_tagged(self):
        controls = collect_controls(self.window)
        for name, (spec, _args) in default_registry().bindings().items():
            if name in controls:
                self.assertEqual(controls[name].property(COMMAND_PROPERTY), spec.id)


if __name__ == "__main__":
    unittest.main()

import ast
import unittest
from pathlib import Path

from application.preview_timeline import PreviewTimelineController
from application.project import ProjectDocument
from core.models.adapter import MuJoCoRobotAdapter
from core.trajectory.model import Trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_core_does_not_import_gui_or_qt(self):
        violations = []
        for path in (PROJECT_ROOT / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "gui" or name.startswith(("gui.", "PySide6")) for name in names):
                    violations.append((path.name, names))
        self.assertEqual(violations, [])

    def test_project_keeps_target_and_robot_timelines_distinct(self):
        adapter = MuJoCoRobotAdapter("go2")
        preview = PreviewTimelineController(adapter)
        targets = Trajectory(adapter.trajectory_frames)
        project = ProjectDocument("go2", targets, preview.timeline)
        self.assertIs(project.target_trajectory, targets)
        self.assertIs(project.robot_state_timeline, preview.timeline)

    def test_old_model_imports_remain_compatible(self):
        from gui.robot_model_adapter import MuJoCoRobotAdapter as LegacyAdapter
        from gui.robot_model_3d import RobotState3D as LegacyState
        from core.models.model import RobotState3D

        self.assertIs(LegacyAdapter, MuJoCoRobotAdapter)
        self.assertIs(LegacyState, RobotState3D)


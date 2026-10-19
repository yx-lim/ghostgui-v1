"""Current GUI orchestration contracts for the upcoming service extraction."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from application.csv_io import read_qpos_csv, read_trajectory_csv
from core.trajectory import rpy_to_quat
from gui.main_window import RobotGuiMainWindow


class GuiWorkflowCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {
            "GHOSTGUI_CONFIG_DIR": str(self.root / "config"),
            "GHOSTGUI_CACHE_DIR": str(self.root / "cache"),
            "GHOSTGUI_PROJECTS_DIR": str(self.root / "projects"),
            "GHOSTGUI_USER_DATA_DIR": str(self.root / "data"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.window = RobotGuiMainWindow(model_key="g1")
        self.addCleanup(self.window.close)
        self.viewer = self.window.viewer_3d

    def test_commit_at_end_captures_committed_fk_and_is_one_undoable_action(self):
        window, viewer = self.window, self.viewer
        end_time = viewer.timeline_duration
        window.on_viewer_timeslice_time_changed(end_time)
        window.trajectory.clear()
        window._refresh_history_baseline()
        before = window.capture_history_snapshot()
        depth = len(window.undo_stack)
        committed = viewer.committed_state.get_qpos()
        viewer.preview_state.set_qpos(committed)
        viewer.preview_state.set_joint_value("right_wrist_yaw_joint", 0.2)
        # The capture boundary must use committed FK even if an unrelated
        # preview state contains a different pose.
        expected = {
            name: viewer.committed_state.get_body_pose(object_name, kind)
            for name in window.editable_logical_frame_names()
            for kind, object_name in (viewer.frame_bindings[name],)
        }
        window.on_accept_timeslice_requested()
        frames = window.trajectory.frames
        self.assertEqual({frame.frame_name for frame in frames}, set(expected))
        self.assertEqual(len(frames), len(expected))
        for frame in frames:
            self.assertEqual(frame.time, end_time)
            position, quaternion = expected[frame.frame_name]
            np.testing.assert_allclose([frame.x, frame.y, frame.z], position, atol=1e-12)
            actual_quaternion = rpy_to_quat(frame.roll, frame.pitch, frame.yaw)
            self.assertAlmostEqual(abs(np.dot(actual_quaternion, quaternion)), 1.0)
        np.testing.assert_array_equal(viewer.state_timeline.get_state(end_time), committed)
        self.assertAlmostEqual(viewer.current_time, end_time)
        self.assertEqual(len(window.undo_stack), depth + 1)
        accepted_frames = window.trajectory.to_project_dict()
        window.undo_last_action()
        self.assertEqual(
            [frame.to_dict() for frame in window.trajectory.frames],
            list(before.trajectory_frames),
        )
        self.assertEqual(viewer.state_timeline.times(), [time for time, _ in before.timeline_states])
        for time, qpos in before.timeline_states:
            np.testing.assert_array_equal(viewer.state_timeline.get_state(time), qpos)
        window.redo_last_action()
        self.assertEqual(window.trajectory.to_project_dict(), accepted_frames)
        np.testing.assert_array_equal(viewer.state_timeline.get_state(end_time), committed)

    def test_commit_with_auto_advance_is_reversed_by_one_undo(self):
        window, viewer = self.window, self.viewer
        window.trajectory.clear()
        window._refresh_history_baseline()
        window.on_accept_timeslice_requested()
        self.assertAlmostEqual(viewer.current_time, viewer.timeslice_step())
        window.undo_last_action()
        self.assertEqual(window.trajectory.frames, [])

    def test_one_hundred_scrubs_do_not_author_motion_dirty_or_replace_history(self):
        window, viewer = self.window, self.viewer
        home = viewer.committed_state.get_qpos()
        end = home.copy()
        end[-1] += 0.05
        viewer.state_timeline.set_state(1.0, end)
        window.set_project_dirty(False)
        window._refresh_history_baseline()
        before = viewer.state_timeline.qpos_trajectory()
        revision = window.document.revision
        baseline = window.history.baseline
        dirty = window.document.dirty
        depth = len(window.undo_stack)
        for index in range(100):
            window.on_viewer_timeslice_time_changed(index / 100)
        self.assertEqual(viewer.state_timeline.times(), [0.0, 1.0])
        np.testing.assert_array_equal(viewer.state_timeline.qpos_trajectory(), before)
        self.assertEqual(window.document.revision, revision)
        self.assertEqual(window.document.dirty, dirty)
        self.assertEqual(len(window.undo_stack), depth)
        self.assertIs(window.history.baseline, baseline)
        project = window.create_project_at(self.root / "navigation.ghostgui", "Navigation")
        self.assertEqual(viewer.state_timeline.times(), [0.0, 1.0])
        window.autosave_current_project(capture_snapshot=False)
        for path in (project.paths.qpos_timeline, project.autosave_paths.qpos_timeline):
            with np.load(path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["times"], [0.0, 1.0])

    def test_commit_invalidates_generated_playback_and_requires_explicit_export_source(self):
        window, viewer = self.window, self.viewer
        home = viewer.committed_state.get_qpos()
        viewer.set_robot_trajectory([home, home], times=(0.0, 1.0), activate_first_frame=False)
        self.assertTrue(viewer.generated_motion_fresh())
        window.on_accept_timeslice_requested()
        self.assertFalse(viewer.generated_motion_fresh())
        self.assertEqual(viewer._trajectory_sample(0.5), (None, None))
        with self.assertRaisesRegex(ValueError, "stale"):
            viewer._trajectory_export_snapshot()
        viewer.export_source_box.setCurrentIndex(viewer.export_source_box.findData("editable"))
        export = viewer._trajectory_export_snapshot()
        self.assertEqual(export.source_name, "editable timeline")

    def test_export_selects_generated_or_committed_motion_and_never_preview(self):
        viewer = self.viewer
        committed = viewer.committed_state.get_qpos()
        later = committed.copy()
        later[-1] += 0.05
        generated = committed.copy()
        generated[-1] += 0.1
        viewer.state_timeline.set_state(0.0, committed)
        viewer.state_timeline.set_state(1.0, later)
        viewer.preview_state.set_qpos(committed)
        viewer.preview_state.set_joint_value("right_wrist_yaw_joint", 0.3)
        viewer.preview_active = True
        viewer.set_robot_trajectory([generated, generated], times=(2.0, 3.0), activate_first_frame=False)
        timeline_before = viewer.state_timeline.qpos_trajectory()
        selected = viewer._trajectory_export_snapshot()
        self.assertEqual(selected.source_name, "generated trajectory")
        self.assertEqual(selected.times, (2.0, 3.0))
        self.assertTrue(selected.preview_active)
        np.testing.assert_array_equal(selected.qposes, [generated, generated])
        generated_path = viewer.save_trajectory_csv(self.root / "generated.csv")
        loaded = read_trajectory_csv(generated_path, committed.size)
        np.testing.assert_array_equal(loaded.qposes, [generated, generated])
        pose_path = viewer.save_qpos_csv(self.root / "pose.csv")
        np.testing.assert_array_equal(read_qpos_csv(pose_path, committed.size).qpos, committed)
        np.testing.assert_array_equal(viewer.state_timeline.qpos_trajectory(), timeline_before)
        # Snapshot arrays must not alias mutable playback state.
        selected.qposes[0][-1] = -1.0
        np.testing.assert_array_equal(viewer.robot_trajectory[0], generated)
        viewer.clear_robot_trajectory()
        editable = viewer._trajectory_export_snapshot()
        self.assertEqual(editable.source_name, "editable timeline")
        self.assertEqual(editable.times, (0.0, 1.0))
        np.testing.assert_array_equal(editable.qposes, [committed, later])

    def test_failed_generate_preserves_prior_fresh_playback_identity(self):
        self.window.define_timeslice_from_committed_pose()
        home = self.viewer.committed_state.get_qpos()
        self.viewer.set_robot_trajectory([home, home], times=(0.0, 1.0), activate_first_frame=False)
        self.assertTrue(self.viewer.generated_motion_fresh())
        with patch("gui.main_window.trajectory_generation.generate_trajectory_status",
                   side_effect=ValueError("test generation failure")):
            self.window.on_generate_trajectory()
        self.assertTrue(self.viewer.generated_motion_fresh())
        self.assertEqual(self.viewer._trajectory_export_snapshot().source_name, "generated trajectory")

    def test_workspace_capture_restore_preserves_legacy_editor_settings(self):
        window, viewer = self.window, self.viewer
        original = window.capture_project_workspace()
        required = {
            "schema_version", "ai_motion_metadata", "active_index", "active_view_index",
            "active_view", "current_time", "selected_frame", "selected_row",
            "timeline_duration", "control_frame", "target_selection", "camera", "display",
        }
        self.assertTrue(required.issubset(original))
        self.assertEqual(original["schema_version"], 1)
        desired = dict(original)
        desired["camera"] = {"yaw": 21.0, "pitch": 32.0, "distance": 4.0, "center": [0.1, 0.2, 0.3]}
        desired["display"] = {
            "show_keyframes": False, "show_trajectory_lines": False,
            "smoothing": 0.25, "export_dt": 0.02, "dsms_motion_speed": 0.5,
            "show_playback_ghosts": True, "trajectory_import_dt": 0.2,
        }
        window.restore_project_workspace(desired)
        captured = window.capture_project_workspace()
        self.assertEqual(captured["camera"], desired["camera"])
        self.assertEqual(captured["display"], desired["display"])
        self.assertEqual(captured["target_selection"], original["target_selection"])
        project = window.create_project_at(self.root / "settings.ghostgui", "Settings")
        self.assertEqual(project.read_workspace(), captured)
        window.restore_project_workspace(original)
        viewer.set_export_dt(0.1)
        with patch.object(window, "confirm_project_transition", return_value=True):
            reopened = window.open_project_path(project.root_dir)
        self.assertTrue(reopened)
        restored = window.capture_project_workspace()
        self.assertEqual(restored["camera"], desired["camera"])
        self.assertEqual(restored["display"], desired["display"])


if __name__ == "__main__":
    unittest.main()

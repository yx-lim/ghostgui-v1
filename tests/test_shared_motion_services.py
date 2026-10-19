"""Correctness and parity contracts at the shared GUI/headless service boundary."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from application.backend_interface import BackendInterface
from application.csv_io import TrajectoryExport
from application.editor_controller import EditorController
from application.keyframe_service import commit_keyframe
from application.project_document import ProjectDocument
from application.trajectory_export_service import GeneratedMotion, build_trajectory_export, motion_source_identity
from application.trajectory_generation import generate_trajectory_status
from application.workspace_state import merge_workspace_payload
from core.models import MuJoCoRobotAdapter, RobotStateTimeline


class SharedMotionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = MuJoCoRobotAdapter("g1")

    def setUp(self):
        self.timeline = RobotStateTimeline(self.adapter)
        self.document = ProjectDocument("g1", qpos_timeline=self.timeline)
        self.controller = EditorController(self.document)

    def test_commit_captures_both_representations_as_one_revision(self):
        qpos = self.adapter.home_qpos.copy()
        qpos[-1] += 0.04
        result = commit_keyframe(self.controller, self.adapter, 0.4, qpos=qpos,
                                 phase="hold", selected_frame="right_hand")
        self.assertTrue(result.changed)
        self.assertEqual(self.document.revision, 1)
        self.assertEqual({frame.frame_name for frame in self.document.frames_at_time(0.4)},
                         set(self.adapter.trajectory_frames))
        self.assertEqual(self.document.trajectory.frames[self.document.active_index].frame_name, "right_hand")
        np.testing.assert_array_equal(self.timeline.get_state(0.4), qpos)
        self.assertFalse(commit_keyframe(self.controller, self.adapter, 0.4, qpos=qpos,
                                        phase="hold", selected_frame="right_hand").changed)
        self.assertEqual(self.document.revision, 1)

    def test_invalid_commit_leaves_both_representations_unchanged(self):
        before = self.document.trajectory.to_project_dict()
        for time, qpos in ((0.5, [1, 2]), (float("nan"), self.adapter.home_qpos),
                           (6.0, self.adapter.home_qpos)):
            with self.subTest(time=time), self.assertRaises(ValueError):
                commit_keyframe(self.controller, self.adapter, time, qpos=qpos)
        self.assertEqual(self.document.trajectory.to_project_dict(), before)
        self.assertEqual(self.timeline.times(), [0.0])
        self.assertEqual(self.document.revision, 0)

    def test_export_is_read_only_and_explicit_editable_source_escapes_stale_generated(self):
        home = self.adapter.home_qpos.copy()
        identity = motion_source_identity(self.adapter, self.timeline, document=self.document)
        generated = GeneratedMotion(TrajectoryExport(36, (0.0, 1.0), (home, home), "generated", False), identity)
        before_times = self.timeline.times()
        result = build_trajectory_export(self.adapter, self.timeline, generated, current_identity=identity)
        self.assertEqual(result.export.source_name, "generated trajectory")
        changed = home.copy()
        changed[-1] += 0.03
        self.timeline.set_state(0.5, changed)
        current = motion_source_identity(self.adapter, self.timeline, document=self.document)
        with self.assertRaisesRegex(ValueError, "stale"):
            build_trajectory_export(self.adapter, self.timeline, generated, current_identity=current)
        result = build_trajectory_export(self.adapter, self.timeline, generated,
                                         source="editable", current_identity=current)
        self.assertEqual(result.export.times, (0.0, 0.5))
        self.assertEqual(self.timeline.times(), before_times + [0.5])
        result.export.qposes[0][-1] = 99
        np.testing.assert_array_equal(self.timeline.get_state(0.0), home)

    def test_generation_uses_detached_inputs_and_rejects_late_result_before_writing(self):
        for time in (0.0, 1.0):
            commit_keyframe(self.controller, self.adapter, time, qpos=self.adapter.home_qpos)
        backend = BackendInterface(mj_model=self.adapter.mj_model, adapter=self.adapter)
        solve = backend.solve_trajectory

        def edit_during_solve(samples):
            # Solver receives the old snapshot; an independent edit changes the live source.
            changed = self.adapter.home_qpos.copy()
            changed[-1] += 0.02
            self.timeline.set_state(0.5, changed)
            return solve(samples)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "late.csv"
            with patch.object(backend, "solve_trajectory", side_effect=edit_during_solve), \
                 patch.object(backend, "export_last_solution_csv") as write:
                with self.assertRaisesRegex(ValueError, "changed during Generate"):
                    generate_trajectory_status(self.document.trajectory, backend,
                                               smoothing=0.0, export_dt=0.25,
                                               csv_path=path, state_timeline=self.timeline)
                write.assert_not_called()
            self.assertFalse(path.exists())
            self.assertFalse(backend.has_last_solution())

    def test_workspace_merge_preserves_extensions_and_does_not_alias_inputs(self):
        original = {"camera": {"yaw": 12, "extension": [1]}, "unknown": {"nested": [2]}}
        updates = {"camera": {"yaw": 21}, "display": {"export_dt": 0.02}}
        merged = merge_workspace_payload(original, updates)
        self.assertEqual(merged["camera"], {"yaw": 21, "extension": [1]})
        merged["unknown"]["nested"].append(3)
        merged["display"]["export_dt"] = 0.04
        self.assertEqual(original["unknown"], {"nested": [2]})
        self.assertEqual(updates["display"]["export_dt"], 0.02)


if __name__ == "__main__":
    unittest.main()

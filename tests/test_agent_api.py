"""Offline contract and end-to-end coverage for the shared command runtime."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
from application.agent_api.sandbox import OutputSandbox
from application.agent_api.workspace import HeadlessWorkspace
from application.ai.motion_state import ReplaceMotionState, capture_motion_state
from application.ai.schemas import EditAuthor
from application.trajectory_import_formats import read_mjlab_trajectory
from core.models.registry import RobotModelInfo


class AgentAPIContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = HeadlessWorkspace(self.root, approval="auto_accept")
        self.runtime = ToolRuntime(build_command_catalog(self.workspace), self.workspace)
        self.call("new_project", {"name": "motion", "duration_s": 0.6})

    def call(self, name, arguments=None):
        outcome = self.runtime.call(name, arguments or {})
        self.assertTrue(outcome.ok, outcome.payload)
        return outcome.payload

    def author(self):
        home = self.workspace.adapter.home_qpos.copy()
        middle = home.copy()
        middle[-1] += 0.05
        return self.call("author_keyframes", {
            "duration_s": 0.6,
            "keyframes": [{"time_seconds": time, "qpos": pose.tolist()}
                          for time, pose in ((0, home), (0.2, middle), (0.4, middle), (0.6, home))],
        })

    def test_end_to_end_repeat_generate_accept_save_reopen_mjlab(self):
        self.author()
        self.call("validate_motion")
        self.call("repeat_motion", {"start_s": 0, "end_s": 0.6, "additional_copies": 1})
        generated = self.call("generate_trajectory", {"dt": 0.02})
        report = self.call("validate_motion")
        self.assertTrue(report["valid"])
        self.assertEqual(report["source"], "generated")
        self.assertEqual(generated["samples"], 61)
        self.call("accept_candidate")
        self.assertEqual(len(self.workspace.history.undo_entries), 1)
        self.assertAlmostEqual(self.workspace.committed.timeline_duration, 1.2)
        self.call("save_project")
        result = self.call("export_trajectory", {"format": "mjlab_csv", "file_name": "motion"})
        loaded = read_mjlab_trajectory(result["paths"][0], self.workspace.adapter, sample_interval=0.02)
        self.assertEqual(len(loaded.qposes), 61)
        np.testing.assert_allclose(loaded.qposes, self.workspace.generated.export.qposes, atol=1e-9)
        before = self.workspace.entity_values(self.workspace.committed)
        self.call("open_project", {"name": "motion"})
        self.assertEqual(self.workspace.entity_values(self.workspace.committed), before)
        self.assertIsNone(self.workspace.generated)

    def test_accept_is_one_history_transition_and_can_be_reversed(self):
        before = self.workspace.entity_values(self.workspace.committed)
        self.author()
        candidate = self.workspace.entity_values(self.workspace.document())
        self.call("validate_motion")
        self.call("accept_candidate")
        current = (capture_motion_state(self.workspace.committed), self.workspace.metadata_store.snapshot())
        transition = self.workspace.history.undo(current)
        motion, metadata = transition.target
        self.workspace.controller.execute(ReplaceMotionState(motion))
        self.workspace.metadata_store.replace(metadata)
        self.assertEqual(self.workspace.entity_values(self.workspace.committed), before)
        redo = self.workspace.history.redo((capture_motion_state(self.workspace.committed), metadata))
        self.workspace.controller.execute(ReplaceMotionState(redo.target[0]))
        self.assertEqual(self.workspace.entity_values(self.workspace.committed), candidate)

    def test_review_policy_requires_trusted_approval(self):
        self.workspace.approval = "review"
        before = self.workspace.identity(self.workspace.committed)
        self.author()
        self.call("validate_motion")
        self.assertEqual(self.call("accept_candidate")["status"], "pending_user_review")
        self.assertEqual(self.workspace.identity(self.workspace.committed), before)
        self.assertFalse(self.runtime.call("accept_candidate", {"user_approved": True}).ok)
        result = self.workspace.accept_candidate(user_approved=True)
        self.assertEqual(result["status"], "accepted")

    def test_malformed_inputs_and_unknown_tools_do_not_create_candidate(self):
        before = self.workspace.identity(self.workspace.committed)
        for name, args in (
            ("run_python", {"code": "pass"}),
            ("author_keyframes", {"keyframes": [], "approval": "auto_accept"}),
            ("set_joint_angles", {"time_s": float("nan"), "values": []}),
            ("new_project", {"name": "../outside"}),
            ("commit_keyframe", {"time_s": 0.2, "expected_revision": 999}),
        ):
            with self.subTest(name=name):
                self.assertFalse(self.runtime.call(name, args).ok)
                self.assertIsNone(self.workspace.candidate())
                self.assertEqual(self.workspace.identity(self.workspace.committed), before)

    def test_failed_second_operation_restores_entire_call_and_provenance(self):
        self.author()
        session = self.workspace.candidate()
        before = self.workspace.entity_values(session.working_document)
        metadata = session.metadata.snapshot()
        result = self.runtime.call("apply_edit_spec", {"spec_json": json.dumps({
            "mode": "edit", "summary": "test rollback", "operations": [
                {"type": "set_joint_target", "arguments": json.dumps({
                    "joint": "right_wrist_yaw_joint", "time_seconds": 0.2, "angle_rad": 0.07,
                })},
                {"type": "set_joint_target", "arguments": json.dumps({
                    "joint": "missing_joint", "time_seconds": 0.2, "angle_rad": 0.07,
                })},
            ],
        })})
        self.assertFalse(result.ok)
        self.assertEqual(self.workspace.entity_values(session.working_document), before)
        self.assertEqual(session.metadata.snapshot(), metadata)

    def test_user_owned_and_protected_edits_reject_atomically(self):
        reference = self.workspace.metadata.reference_for_qpos_keyframe(0.0)
        for author, protected in ((EditAuthor.USER, False), (EditAuthor.AI, True)):
            with self.subTest(author=author, protected=protected):
                self.workspace.metadata_store.record(reference, author)
                self.workspace.metadata_store.set_protected(reference, protected)
                before = self.workspace.identity(self.workspace.committed)
                home = self.workspace.adapter.home_qpos.tolist()
                changed = list(home)
                changed[-1] += 0.02
                result = self.runtime.call("author_keyframes", {"duration_s": 0.6, "keyframes": [
                    {"time_seconds": 0.0, "qpos": changed}, {"time_seconds": 0.6, "qpos": home},
                ]})
                self.assertFalse(result.ok)
                self.assertIsNone(self.workspace.candidate())
                self.assertEqual(self.workspace.identity(self.workspace.committed), before)

    def test_stale_revision_and_direct_committed_edit_reject(self):
        self.author()
        before = self.workspace.entity_values(self.workspace.document())
        revision = self.workspace.document().revision
        result = self.runtime.call("commit_keyframe", {"time_s": 0.3, "expected_revision": revision - 1})
        self.assertFalse(result.ok)
        self.assertEqual(result.payload["code"], "stale_revision")
        self.assertEqual(self.workspace.entity_values(self.workspace.document()), before)
        self.call("validate_motion")
        # Legacy writers may bypass ProjectDocument.mark_changed. Content identity must still guard Accept.
        self.workspace.committed.qpos_timeline.states[0.0][-1] += 0.02
        self.assertFalse(self.runtime.call("accept_candidate", {}).ok)

    def test_candidate_change_invalidates_validation_and_generated_path(self):
        self.author()
        self.call("generate_trajectory")
        self.call("validate_motion")
        self.call("set_joint_angles", {"time_s": 0.2, "values": [
            {"joint": "right_wrist_yaw_joint", "angle_rad": 0.06},
        ]})
        self.assertIsNone(self.workspace.generated)
        self.assertFalse(self.runtime.call("accept_candidate", {}).ok)
        self.call("validate_motion")
        self.call("accept_candidate")

    def test_checkpoint_restoration_requires_revalidation_and_is_session_owned(self):
        first = self.author()
        self.call("commit_keyframe", {"time_s": 0.3})
        self.call("undo", {"checkpoint_id": first["checkpoint_id"]})
        self.assertEqual(len(self.workspace.document().trajectory.frames), 0)
        self.assertFalse(self.runtime.call("accept_candidate", {}).ok)
        self.call("discard_candidate")
        self.assertFalse(self.runtime.call("undo", {"checkpoint_id": first["checkpoint_id"]}).ok)

    def test_repeated_default_undo_walks_back_candidate_operations(self):
        self.author()
        before_commit = self.workspace.entity_values(self.workspace.document())
        self.call("commit_keyframe", {"time_s": 0.3})
        self.call("undo")
        self.assertEqual(self.workspace.entity_values(self.workspace.document()), before_commit)
        self.call("undo")
        self.assertEqual(self.workspace.document().trajectory.frames, [])
        self.assertFalse(self.runtime.call("undo", {}).ok)

    def test_no_overwrite_and_failed_open_preserve_current_workspace(self):
        identity = self.workspace.identity(self.workspace.committed)
        metadata_bytes = self.workspace.project.project_file.read_bytes()
        self.assertFalse(self.runtime.call("new_project", {"name": "motion"}).ok)
        self.assertFalse(self.runtime.call("open_project", {"name": "missing"}).ok)
        self.assertEqual(self.workspace.identity(self.workspace.committed), identity)
        self.assertEqual(self.workspace.project.project_file.read_bytes(), metadata_bytes)

    def test_sandbox_rejects_escapes_absolute_names_and_symlinks(self):
        sandbox = OutputSandbox(self.root / "exports")
        sandbox.root.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (sandbox.root / "escape.csv").symlink_to(outside / "motion.csv")
        for name in ("../bad", "/tmp/bad", "C:\\bad", "nested/file", "..", "escape.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sandbox.resolve_output(name, suffix=".csv")
        self.assertEqual(list(outside.iterdir()), [])

    def test_call_result_and_sample_budgets_and_cancellation(self):
        runtime = ToolRuntime(self.runtime.catalog, self.workspace, max_calls=1)
        self.assertTrue(runtime.call("get_status", {}).ok)
        self.assertFalse(runtime.call("get_status", {}).ok)
        runtime = ToolRuntime(self.runtime.catalog, self.workspace, max_result_bytes=1)
        self.assertFalse(runtime.call("get_robot_summary", {}).ok)
        self.author()
        before = self.workspace.identity()
        runtime = ToolRuntime(self.runtime.catalog, self.workspace,
                              cancellation_token=SimpleNamespace(cancellation_requested=True))
        self.assertFalse(runtime.call("commit_keyframe", {"time_s": 0.3}).ok)
        self.assertEqual(self.workspace.identity(), before)
        self.workspace.max_samples = 2
        self.assertFalse(self.runtime.call("generate_trajectory", {"dt": 0.02}).ok)

    def test_unknown_workspace_fields_survive_headless_save(self):
        self.workspace.workspace_payload["custom_extension"] = {"untouched": [1, 2]}
        self.workspace.workspace_payload["camera"] = {"yaw": 12.0}
        self.call("save_project")
        self.call("open_project", {"name": "motion"})
        self.assertEqual(self.workspace.workspace_payload["custom_extension"], {"untouched": [1, 2]})
        self.assertEqual(self.workspace.workspace_payload["camera"], {"yaw": 12.0})

    def test_all_inspection_tools_are_read_only(self):
        before = self.workspace.identity(self.workspace.committed)
        for name, args in (("list_models", {}), ("get_status", {}), ("get_robot_summary", {}),
                           ("list_keyframes", {"source": "committed"}), ("get_pose", {"time_s": 0.2})):
            with self.subTest(tool=name):
                self.call(name, args)
                self.assertIsNone(self.workspace.candidate())
                self.assertEqual(self.workspace.identity(self.workspace.committed), before)
        self.assertEqual(len(self.runtime.catalog.registry.definitions()), 23)
        self.assertNotIn("render_views", self.runtime.catalog.meta)

    def test_named_ik_scale_delete_and_checkpoint_tools(self):
        self.author()
        state = self.workspace.adapter.create_state()
        state.set_qpos(self.workspace.document().qpos_timeline.sample_state(0.2))
        kind, name = self.workspace.adapter.logical_frame_bindings["right_hand"]
        position, _ = state.get_body_pose(name, kind)
        position[0] += 0.005
        self.call("solve_ik", {"logical_frame": "right_hand", "time_s": 0.2,
                               "position_m": position.tolist()})
        self.call("commit_keyframe", {"time_s": 0.3})
        self.call("delete_keyframes", {"times": [0.3]})
        self.assertEqual(self.workspace.document().frames_at_time(0.3), ())
        self.assertNotIn(0.3, self.workspace.document().qpos_timeline.times())
        self.call("edit_timeline", {"start_s": 0.0, "end_s": 0.6, "scale": 2.0})
        self.assertAlmostEqual(self.workspace.document().timeline_duration, 1.2)
        self.assertTrue(self.call("validate_motion")["valid"])

    def test_incompatible_model_export_and_existing_file_leave_outputs_untouched(self):
        self.call("load_model", {"model_key": "go2"})
        self.call("new_project", {"name": "quadruped", "model_key": "go2"})
        result = self.runtime.call("export_trajectory", {"format": "mjlab_csv", "file_name": "bad"})
        self.assertFalse(result.ok)
        self.assertIn("G1", result.payload["message"])
        self.assertFalse((self.workspace.project.root_dir / "exports" / "bad.csv").exists())
        exported = self.call("export_trajectory", {"format": "qpos_csv", "file_name": "pose"})
        path = Path(exported["paths"][0])
        content = path.read_bytes()
        self.assertFalse(self.runtime.call("export_trajectory", {
            "format": "qpos_csv", "file_name": "pose",
        }).ok)
        self.assertEqual(path.read_bytes(), content)

    def test_failed_edit_keeps_last_validated_generated_candidate_usable(self):
        self.author()
        self.call("generate_trajectory")
        self.call("validate_motion")
        before = self.workspace.entity_values(self.workspace.document())
        result = self.runtime.call("set_joint_angles", {"time_s": 0.2, "values": [
            {"joint": "right_wrist_yaw_joint", "angle_rad": 0.07},
            {"joint": "missing_joint", "angle_rad": 0.02},
        ]})
        self.assertFalse(result.ok)
        self.assertEqual(self.workspace.entity_values(self.workspace.document()), before)
        self.assertIsNotNone(self.workspace.generated)
        self.assertEqual(self.workspace.generated.source, self.workspace.identity())
        self.call("accept_candidate")

    def test_late_protection_and_unvalidated_direct_candidate_changes_reject_accept(self):
        self.author()
        self.call("validate_motion")
        reference = self.workspace.metadata.reference_for_qpos_keyframe(0.2)
        self.workspace.metadata_store.set_protected(reference, True)
        self.assertFalse(self.runtime.call("accept_candidate", {}).ok)
        self.workspace.metadata_store.set_protected(reference, False)
        self.workspace.document().qpos_timeline.states[0.2][-1] += 0.005
        self.assertFalse(self.runtime.call("accept_candidate", {}).ok)

    def test_intermediate_collision_blocks_validation_and_accept(self):
        model_path = self.root / "obstacle_arm.xml"
        model_path.write_text(
            '<mujoco><compiler angle="radian"/><worldbody>'
            '<geom name="obstacle" type="sphere" size="0.08" pos="1 0 0.4"/>'
            '<body name="base"><body name="arm" pos="0 0 0.4">'
            '<joint name="yaw" axis="0 0 1" range="-2 2"/>'
            '<body name="tip" pos="1 0 0"><geom name="tool_geom" '
            'type="sphere" size="0.08"/></body></body></body>'
            '</worldbody><keyframe><key name="home" qpos="-1.5707963267948966"/>'
            '</keyframe></mujoco>', encoding="utf-8",
        )
        info = RobotModelInfo(
            key="obstacle_arm", display_name="Obstacle arm", model_type="generic",
            model_path=model_path, root_body_candidates=("base",),
            logical_frames={"base": ("base",), "tool": ("tip",)},
            floating_base=False,
        )
        workspace = HeadlessWorkspace(self.root / "obstacle-projects", model="obstacle_arm",
                                      models={"obstacle_arm": info}, approval="auto_accept")
        runtime = ToolRuntime(build_command_catalog(workspace), workspace)
        self.assertTrue(runtime.call("new_project", {"name": "sweep", "duration_s": 0.6}).ok)
        self.assertTrue(runtime.call("author_keyframes", {"duration_s": 0.6, "keyframes": [
            {"time_seconds": 0.0, "qpos": [-math.pi / 2]},
            {"time_seconds": 0.6, "qpos": [math.pi / 2]},
        ]}).ok)
        report = runtime.call("validate_motion", {"dt": 0.02})
        self.assertTrue(report.ok, report.payload)
        self.assertFalse(report.payload["valid"])
        self.assertTrue(any("blocking collision" in issue for issue in report.payload["issues"]))
        self.assertFalse(runtime.call("accept_candidate", {}).ok)

    def test_headless_import_and_execution_block_all_gui_rendering_imports(self):
        source = '''
import importlib.abc, sys, tempfile
class NoUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gui', 'PySide6', 'PySide2', 'PyQt5', 'PyQt6', 'OpenGL'}:
            raise AssertionError('headless import attempted: ' + fullname)
        if any(fullname == root or fullname.startswith(root + '.')
               for root in ('anthropic', 'google.genai', 'keyring', 'mcp')):
            raise AssertionError('optional SDK import attempted: ' + fullname)
sys.meta_path.insert(0, NoUI())
from application.agent_api.workspace import HeadlessWorkspace
from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
with tempfile.TemporaryDirectory() as root:
    workspace = HeadlessWorkspace(root)
    runtime = ToolRuntime(build_command_catalog(workspace), workspace)
    assert runtime.call('get_robot_summary', {}).ok
'''
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

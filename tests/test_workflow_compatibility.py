"""Legacy file and motion contracts to retain during service extraction.

All inputs are synthetic and all files are temporary. Expected wire bytes and
sample values are independent of the writers/samplers under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from application.backend_interface import BackendInterface, FallbackPolicy
from application.csv_io import (
    TrajectoryExport,
    read_qpos_csv,
    read_trajectory_csv,
    write_qpos_csv,
    write_trajectory_csv,
)
from application.project_manager import GhostGUIProject
from application.timeslice_service import capture_timeslice_from_committed_pose
from application.trajectory_export_formats import (
    export_dsms_trajectory,
    export_mjlab_trajectory,
    resample_trajectory_export,
)
from application.trajectory_generation import generate_trajectory_status
from core.models import MuJoCoRobotAdapter, RobotStateTimeline
from core.models.registry import RobotModelInfo
from core.trajectory import Trajectory


# This is the historical consumer order, deliberately independent of the
# production export constant and of a model's compiled qpos addresses.
LEGACY_G1_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class LegacyExportBytesTests(unittest.TestCase):
    def test_pose_and_mujoco_csv_keep_headerless_precision_and_newlines(self):
        source = TrajectoryExport(
            expected_qpos_count=2,
            times=(2.0, 2.25),
            qposes=(np.array([1.0, -0.5]), np.array([0.0, 2.0])),
            source_name="editable timeline",
            preview_active=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            pose_path = write_qpos_csv(Path(directory) / "pose", source.qposes[0])
            motion_path = write_trajectory_csv(Path(directory) / "motion", source)
            self.assertEqual(pose_path.name, "pose.csv")
            self.assertEqual(motion_path.name, "motion.csv")
            self.assertEqual(
                pose_path.read_bytes(),
                b"1.000000000000000000e+00,-5.000000000000000000e-01\r\n",
            )
            self.assertEqual(
                motion_path.read_bytes(),
                b"2.000000,1.000000000000000000e+00,-5.000000000000000000e-01\r\n"
                b"2.250000,0.000000000000000000e+00,2.000000000000000000e+00\r\n",
            )
            np.testing.assert_array_equal(read_qpos_csv(pose_path, 2).qpos, [1, -0.5])
            loaded = read_trajectory_csv(motion_path, 2)
            self.assertEqual(loaded.times, (2.0, 2.25))
            np.testing.assert_array_equal(loaded.qposes, source.qposes)

    def test_dsms_keeps_filenames_and_scales_only_elapsed_time(self):
        source = TrajectoryExport(
            2, (2.0, 2.25),
            (np.array([1.0, -0.5]), np.array([0.0, 2.0])),
            "editable timeline", False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = export_dsms_trajectory(directory, source, dof=2, motion_speed=0.5)
            qpos_path, time_path = result.paths
            self.assertEqual([path.name for path in result.paths], [
                "qpos_2dof.csv", "time.csv",
            ])
            self.assertEqual(
                qpos_path.read_bytes(),
                b"1.000000000000000000e+00,-5.000000000000000000e-01\n"
                b"0.000000000000000000e+00,2.000000000000000000e+00\n",
            )
            self.assertEqual(time_path.read_bytes(), b"2.000000000\n2.500000000\n")
        self.assertEqual(source.times, (2.0, 2.25))
        np.testing.assert_array_equal(source.qposes, [[1, -0.5], [0, 2]])
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.source_fps, 4.0)
        self.assertEqual(result.input_fps, 2.0)

    def test_mjlab_keeps_named_order_xyzw_and_exact_consumer_bytes(self):
        # Simulate a G1 compiled in the opposite joint order. Output must use
        # semantic names, not the qpos layout or actuated_joints iteration order.
        names = tuple(reversed(LEGACY_G1_JOINT_ORDER))
        adapter = SimpleNamespace(
            mj_model=SimpleNamespace(nq=36),
            actuated_joints=names,
            joints={name: SimpleNamespace(qpos_address=7 + i) for i, name in enumerate(names)},
            free_joints_by_body={1: SimpleNamespace(qpos_address=0)},
        )
        qpos = np.array([1, 2, 3, 3, 0, 0, 4] + list(range(29, 0, -1)), dtype=float)
        source = TrajectoryExport(36, (2.0,), (qpos,), "generated trajectory", False)
        with tempfile.TemporaryDirectory() as directory:
            result = export_mjlab_trajectory(
                Path(directory) / "g1.csv", source, adapter, single_sample_dt=0.02,
            )
            self.assertEqual(
                result.paths[0].read_bytes(),
                b"1,2,3,0,0,0.8,0.6,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
                b"16,17,18,19,20,21,22,23,24,25,26,27,28,29\n",
            )
        self.assertEqual(result.input_fps, 50.0)
        np.testing.assert_array_equal(qpos[3:7], [3, 0, 0, 4])


class ImportedMotionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        source = Path(directory.name) / "synthetic_arm.xml"
        source.write_text(
            '<mujoco model="synthetic_arm"><compiler angle="radian"/>'
            '<worldbody><body name="base"><freejoint/>'
            '<geom type="sphere" size="0.1" contype="0" conaffinity="0"/>'
            '<body name="arm" pos="0 0 0.2">'
            '<joint name="hinge" axis="0 0 1" range="-2 2"/>'
            '<geom type="sphere" size="0.05" contype="0" conaffinity="0"/>'
            '<site name="tip" pos="1 0 0"/>'
            '</body></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        cls.adapter = MuJoCoRobotAdapter(RobotModelInfo(
            key="synthetic_arm", display_name="Synthetic arm", model_type="generic",
            model_path=source, root_body_candidates=("base",),
            logical_frames={"base": ("base",), "tool": ("tip",)},
            end_effector_frames=("tool",), floating_base=True,
        ))

    def test_capture_and_generation_keep_imported_model_anchors_and_csv_width(self):
        adapter = self.adapter
        state = adapter.create_state()
        qpos = np.array([0, 0, 1, 1, 0, 0, 0, 0.5], dtype=float)
        state.set_qpos(qpos)
        frames = capture_timeslice_from_committed_pose(
            state, time=2.0, phase="hold", frame_names=adapter.trajectory_frames,
            frame_bindings=adapter.logical_frame_bindings,
        )
        self.assertEqual({frame.frame_name for frame in frames}, {"base", "tool"})
        tool = next(frame for frame in frames if frame.frame_name == "tool")
        np.testing.assert_allclose([tool.x, tool.y, tool.z], [math.cos(0.5), math.sin(0.5), 1.2])
        np.testing.assert_allclose([tool.roll, tool.pitch, tool.yaw], [0, 0, 0.5], atol=1e-14)
        trajectory = Trajectory()
        for time in (2.0, 2.5):
            for frame in capture_timeslice_from_committed_pose(
                state, time=time, phase="hold", frame_names=adapter.trajectory_frames,
                frame_bindings=adapter.logical_frame_bindings,
            ):
                trajectory.add_frame(frame)
        before = trajectory.to_project_dict()
        timeline = RobotStateTimeline(adapter, initial_time=2.0, initial_qpos=qpos)
        timeline.set_state(2.5, qpos)
        backend = BackendInterface(
            mj_model=adapter.mj_model, adapter=adapter, fallback_policy=FallbackPolicy.ERROR,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = generate_trajectory_status(
                trajectory, backend, smoothing=0.0, export_dt=0.25,
                csv_path=Path(directory) / "generated.csv", state_timeline=timeline,
            )
            loaded = read_trajectory_csv(result.csv_path, 8)
        self.assertEqual(loaded.times, (2.0, 2.25, 2.5))
        np.testing.assert_allclose(loaded.qposes, np.tile(qpos, (3, 1)), atol=1e-12)
        np.testing.assert_array_equal(result.result_states[0].qpos, qpos)
        np.testing.assert_array_equal(result.result_states[-1].qpos, qpos)
        self.assertEqual(trajectory.to_project_dict(), before)
        self.assertEqual(timeline.times(), [2.0, 2.5])

    def test_resampling_preserves_nonzero_time_and_free_root_rotation(self):
        start = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=float)
        end = np.array([2, 0, 1, 0, 0, 0, 1, 1], dtype=float)
        source = TrajectoryExport(8, (2.0, 3.0), (start, end), "editable timeline", True)
        sampled = resample_trajectory_export(source, self.adapter, 0.25)
        self.assertEqual(sampled.times, (2.0, 2.25, 2.5, 2.75, 3.0))
        np.testing.assert_allclose(
            sampled.qposes[2], [1, 0, 1, math.sqrt(0.5), 0, 0, math.sqrt(0.5), 0.5],
            atol=1e-14,
        )
        np.testing.assert_allclose(np.linalg.norm(np.array(sampled.qposes)[:, 3:7], axis=1), 1)
        np.testing.assert_array_equal(sampled.qposes[0], start)
        np.testing.assert_array_equal(sampled.qposes[-1], end)
        self.assertEqual(sampled.source_name, "editable timeline")
        self.assertTrue(sampled.preview_active)
        np.testing.assert_array_equal(source.qposes, [start, end])

    def test_copied_v1_v2_projects_preserve_sparse_targets_and_dense_qpos(self):
        for schema in (1, 2):
            for times in ((0.0, 1.0), (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)):
                with self.subTest(schema=schema, times=times), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    original = root / "original.ghostgui"
                    (original / "data").mkdir(parents=True)
                    (original / "workspace").mkdir()
                    metadata = {
                        "schema_version": schema, "project_name": "Legacy imported arm",
                        "created_at": "2025-01-01T00:00:00Z", "modified_at": "2025-01-01T00:00:00Z",
                        "application": {"name": "GhostGUI", "project_format": f"ghostgui.project.v{schema}"},
                        "robot": {"model_key": "synthetic_arm", "model_name": "Synthetic arm"},
                        "files": {
                            "target_trajectory": "data/target_frames.json",
                            "qpos_timeline": "data/qpos_timeline.npz",
                            "workspace": "workspace/workspace.json",
                            "last_snapshot": "snapshots/last_workspace.png",
                            "session_log": "metadata/session_log.jsonl",
                        },
                    }
                    frame = {"time": 1.0, "phase": "hold", "frame_name": "tool",
                             "x": 0.25, "y": -0.5, "z": 1.0,
                             "roll": 0.1, "pitch": -0.2, "yaw": 0.3}
                    targets = {"schema_version": 1, "tracks": {"tool": [frame]}}
                    workspace = {
                        "schema_version": 1, "current_time": 0.5, "timeline_duration": 3.0,
                        "display": {"export_dt": 0.02, "dsms_motion_speed": 0.5},
                        "camera": {"yaw": 21.0, "pitch": 32.0, "distance": 4.0, "center": [1, 2, 3]},
                        "custom_extension": {"keep": ["opaque", 7]},
                    }
                    for name, payload in (
                        ("ghostgui_project.json", metadata), ("data/target_frames.json", targets),
                        ("workspace/workspace.json", workspace),
                    ):
                        (original / name).write_text(json.dumps(payload), encoding="utf-8")
                    rows = np.array([[time, 0, 1, 1, 0, 0, 0, time / 2] for time in times], dtype=float)
                    np.savez(original / "data/qpos_timeline.npz", schema_version=[1], times=times, qpos=rows)
                    original_bytes = {
                        path.relative_to(original): path.read_bytes()
                        for path in original.rglob("*") if path.is_file()
                    }
                    copied = root / "working.ghostgui"
                    shutil.copytree(original, copied)
                    project = GhostGUIProject.open(copied)
                    self.assertEqual(project.source_schema_version, schema)
                    trajectory = Trajectory()
                    trajectory.load_project_dict(project.read_trajectory_dict())
                    timeline = RobotStateTimeline(self.adapter)
                    project.load_qpos_timeline(timeline)
                    np.testing.assert_array_equal(timeline.qpos_trajectory(), rows)
                    project.save_bundle(trajectory, timeline, project.read_workspace())
                    reopened = GhostGUIProject.open(copied)
                    restored = RobotStateTimeline(self.adapter)
                    reopened.load_qpos_timeline(restored)
                    self.assertEqual(reopened.source_schema_version, 2)
                    self.assertEqual(reopened.model_key, "synthetic_arm")
                    self.assertEqual(reopened.read_workspace(), workspace)
                    self.assertEqual(reopened.read_trajectory_dict()["tracks"]["tool"], [frame])
                    self.assertEqual(restored.times(), list(times))
                    np.testing.assert_array_equal(restored.qpos_trajectory(), rows)
                    for relative, content in original_bytes.items():
                        self.assertEqual((original / relative).read_bytes(), content)

    def test_all_bundled_models_keep_their_compiled_qpos_export_width(self):
        for model in ("g1", "go2", "h2", "z1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                adapter = MuJoCoRobotAdapter(model)
                qpos = adapter.home_qpos.copy()
                source = TrajectoryExport(
                    adapter.mj_model.nq, (0.0, 0.25), (qpos, qpos),
                    "editable timeline", False,
                )
                path = write_trajectory_csv(Path(directory) / "motion.csv", source)
                loaded = read_trajectory_csv(path, adapter.mj_model.nq)
                self.assertEqual(loaded.times, (0.0, 0.25))
                np.testing.assert_array_equal(loaded.qposes, [qpos, qpos])
                pose_path = write_qpos_csv(Path(directory) / "pose.csv", qpos)
                np.testing.assert_array_equal(read_qpos_csv(pose_path, adapter.mj_model.nq).qpos, qpos)


if __name__ == "__main__":
    unittest.main()

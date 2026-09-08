"""Focused tests for automatic sampled robot and motion context."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from application.ai.context import (
    EditorSelectionContext,
    MotionAssistantContextBuilder,
)
from application.project_document import ProjectDocument


class FakeTimeline:
    def __init__(self):
        self.states = {
            0.0: np.array([0, 0, 0.8, 1, 0, 0, 0, 0.0, 0.2], dtype=float),
            1.0: np.array([0, 0, 1.0, 1, 0, 0, 0, 1.0, 0.6], dtype=float),
        }
        self.sampled_times = []

    def times(self):
        return sorted(self.states)

    def sample_state(self, time, fallback_qpos=None):
        self.sampled_times.append(float(time))
        if time <= 0.0:
            return self.states[0.0].copy()
        if time >= 1.0:
            return self.states[1.0].copy()
        return self.states[0.0] + (self.states[1.0] - self.states[0.0]) * time


class FakeState:
    def __init__(self, adapter):
        self.adapter = adapter
        self.qpos = adapter.home_qpos.copy()

    def set_qpos(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float).copy()

    def get_joint_value(self, name):
        return float(self.qpos[self.adapter.joints[name].qpos_address])

    def get_body_pose(self, object_name, kind=None):
        offsets = {
            "pelvis_body": (0.0, 0.0, 0.0),
            "torso_body": (0.0, 0.0, 0.4),
            "left_hand_site": (0.2, 0.2, 0.3),
            "right_hand_site": (0.2, -0.2, 0.3),
            "left_foot_site": (0.0, 0.1, -0.8),
            "right_foot_site": (0.0, -0.1, -0.8),
        }
        return (
            self.qpos[:3] + np.asarray(offsets[object_name]),
            self.qpos[3:7].copy(),
        )


class FakeAdapter:
    def __init__(self):
        self.mj_model = SimpleNamespace(nq=9)
        self.home_qpos = np.array([0, 0, 0.8, 1, 0, 0, 0, 0, 0], dtype=float)
        self.free_joints_by_body = {0: SimpleNamespace(qpos_address=0)}
        self.joint_names = ("right_shoulder", "left_knee")
        self.joints = {
            "right_shoulder": SimpleNamespace(qpos_address=7),
            "left_knee": SimpleNamespace(qpos_address=8),
        }
        self.joint_groups = {
            "right_arm": ("right_shoulder",),
            "left_leg": ("left_knee",),
        }
        self.logical_frame_bindings = {
            "pelvis": ("body", "pelvis_body"),
            "torso": ("body", "torso_body"),
            "left_hand": ("site", "left_hand_site"),
            "right_hand": ("site", "right_hand_site"),
            "left_foot": ("site", "left_foot_site"),
            "right_foot": ("site", "right_foot_site"),
        }
        self.end_effectors = {
            name: self.logical_frame_bindings[name]
            for name in ("left_hand", "right_hand", "left_foot", "right_foot")
        }

    def create_state(self):
        return FakeState(self)


def _document():
    return ProjectDocument(
        "g1",
        current_time=0.5,
        timeline_duration=1.0,
        qpos_timeline=FakeTimeline(),
    )


class MotionAssistantContextBuilderTests(unittest.TestCase):
    def test_current_state_includes_named_joints_fk_and_root_conventions(self):
        payload = MotionAssistantContextBuilder(FakeAdapter()).build(
            _document()
        ).to_dict()

        current = payload["current_state"]
        self.assertEqual(current["time_seconds"], 0.5)
        self.assertEqual(current["root"]["position_m"], [0.0, 0.0, 0.9])
        self.assertEqual(current["joint_angles"], {
            "right_shoulder": 0.5,
            "left_knee": 0.4,
        })
        self.assertEqual(
            current["end_effectors"]["right_hand"]["position_m"],
            [0.2, -0.2, 1.2],
        )
        self.assertEqual(payload["robot"]["qpos_layout"]["width"], 9)
        self.assertEqual(
            payload["robot"]["qpos_layout"]["quaternion_convention"],
            "wxyz",
        )

    def test_between_keyframes_uses_timeline_sampler_without_insertion(self):
        document = _document()
        original_times = tuple(document.qpos_timeline.times())
        payload = MotionAssistantContextBuilder(FakeAdapter()).build(
            document,
            selection=EditorSelectionContext(time_interval=(0.5, 0.5)),
        ).to_dict()

        sample = payload["motion"]["numerical_samples"][0]
        self.assertEqual(sample["time_seconds"], 0.5)
        self.assertEqual(sample["joint_angles"]["right_shoulder"], 0.5)
        self.assertIn(0.5, document.qpos_timeline.sampled_times)
        self.assertEqual(tuple(document.qpos_timeline.times()), original_times)

    def test_selected_end_effector_prioritizes_related_joint_group(self):
        payload = MotionAssistantContextBuilder(FakeAdapter()).build(
            _document(),
            selection=EditorSelectionContext(end_effector="right_hand"),
        ).to_dict()

        samples = payload["motion"]["numerical_samples"]
        self.assertLessEqual(len(samples), 12)
        self.assertGreaterEqual(len(samples), 8)
        self.assertEqual(
            set(samples[0]["joint_angles"]),
            {"right_shoulder"},
        )
        self.assertEqual(
            set(payload["current_state"]["joint_angles"]),
            {"right_shoulder", "left_knee"},
        )

    def test_sample_limit_is_hard_bounded(self):
        payload = MotionAssistantContextBuilder(
            FakeAdapter(),
            max_numerical_samples=8,
        ).build(_document()).to_dict()

        self.assertLessEqual(payload["motion"]["numerical_sample_count"], 8)
        with self.assertRaisesRegex(ValueError, "8 to 20"):
            MotionAssistantContextBuilder(FakeAdapter(), max_numerical_samples=21)


if __name__ == "__main__":
    unittest.main()

"""Tests for compact, authoritative GhostGUI AI context."""

from __future__ import annotations

import json
import unittest

from application.ai.context import (
    ContextBuilder,
    EditorSelectionContext,
    RobotCapabilityContext,
)
from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.schemas import EditAuthor
from application.editor_commands import UpdateKeyframe
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class FakeTimeline:
    def __init__(self):
        self.states = {0.0: [123456.0, 654321.0], 2.0: [42.0, 43.0]}

    def times(self):
        return sorted(self.states)

    def get_state(self, time):
        return self.states[time].copy()

    def set_state(self, time, value):
        self.states[round(float(time), 6)] = list(value)


def _document():
    document = ProjectDocument(
        "g1",
        current_time=1.0,
        timeline_duration=4.0,
        qpos_timeline=FakeTimeline(),
    )
    for frame in (
        TargetFrame(time=0.0, frame_name="pelvis", z=0.9),
        TargetFrame(time=1.0, frame_name="torso", pitch=0.1),
        TargetFrame(time=2.0, frame_name="right_hand", x=0.4),
    ):
        document.trajectory.add_frame(frame)
    document.active_index = 1
    return document


class ContextBuilderTests(unittest.TestCase):
    def test_builds_compact_context_without_raw_qpos(self):
        context = ContextBuilder().build(
            _document(),
            motion_name="landing",
            selection=EditorSelectionContext(
                time_interval=(0.8, 1.4),
                logical_frame="torso",
                end_effector="right_hand",
                edit_mode="move",
                camera_view="front",
            ),
            robot_capabilities=RobotCapabilityContext(
                logical_frames=("pelvis", "torso", "right_hand"),
                end_effectors=("right_hand",),
                joints=("waist_pitch",),
                joint_groups=(("waist", ("waist_pitch",)),),
            ),
        )

        payload = context.to_dict()
        self.assertEqual(payload["robot"]["model_key"], "g1")
        self.assertEqual(payload["motion"]["name"], "landing")
        self.assertEqual(payload["motion"]["qpos_keyframe_count"], 2)
        self.assertEqual(payload["selection"]["logical_frame"], "torso")
        self.assertEqual(
            payload["selection"]["active_keyframe"]["time_seconds"],
            1.0,
        )
        serialized = json.dumps(payload)
        self.assertNotIn("123456", serialized)
        self.assertNotIn("654321", serialized)
        self.assertNotIn("qpos_values", serialized)

    def test_session_context_uses_manually_modified_working_copy(self):
        document = _document()
        session = AIEditSession(document)
        session.apply_ai(
            UpdateKeyframe(
                1,
                TargetFrame(time=1.0, frame_name="torso", pitch=0.2),
            )
        )
        session.apply_manual(
            UpdateKeyframe(
                1,
                TargetFrame(time=1.0, frame_name="torso", pitch=0.35),
            )
        )

        payload = ContextBuilder().build_for_session(session).to_dict()

        self.assertTrue(payload["motion"]["working_copy"])
        self.assertEqual(
            payload["selection"]["active_keyframe"]["orientation_rpy_rad"][1],
            0.35,
        )
        self.assertEqual(
            [item["author"] for item in payload["recent_edits"]],
            ["ai", "user"],
        )
        self.assertEqual(document.trajectory.frames[1].pitch, 0.1)

    def test_user_and_protected_metadata_are_resolved_through_service(self):
        document = _document()
        store = InMemoryMotionMetadataStore()
        service = MotionMetadataService(store, TimestampMotionIdentityResolver())
        torso = document.trajectory.frames[1]
        reference = service.reference_for_keyframe(torso)
        store.record(reference, author=EditAuthor.USER)
        store.set_protected(reference, True)

        constraints = ContextBuilder().build(
            document,
            metadata=service,
        ).to_dict()["constraints"]

        self.assertEqual(
            constraints["keyframes"],
            [{
                "logical_frame": "torso",
                "time_seconds": 1.0,
                "author": "user",
                "protected": True,
            }],
        )

    def test_time_summary_is_bounded_and_keeps_endpoints(self):
        document = ProjectDocument("g1", timeline_duration=20.0)
        for time in range(10):
            document.trajectory.add_frame(
                TargetFrame(time=float(time), frame_name="pelvis")
            )

        summary = ContextBuilder(max_times=4).build(document).to_dict()["motion"][
            "keyframe_times"
        ]

        self.assertEqual(summary["total_count"], 10)
        self.assertTrue(summary["truncated"])
        self.assertEqual(len(summary["values"]), 4)
        self.assertEqual(summary["values"][0], 0.0)
        self.assertEqual(summary["values"][-1], 9.0)

    def test_selection_interval_validation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "start"):
            EditorSelectionContext(time_interval=(2.0, 1.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            EditorSelectionContext(time_interval=(0.0, float("nan")))
        with self.assertRaisesRegex(ValueError, "motion duration"):
            ContextBuilder().build(
                _document(),
                selection=EditorSelectionContext(time_interval=(3.0, 5.0)),
            )

    def test_prompt_text_is_deterministic_json(self):
        context = ContextBuilder().build(_document())
        prefix, encoded = context.to_prompt_text().split("\n", 1)
        self.assertEqual(prefix, "GhostGUI editor context:")
        self.assertEqual(json.loads(encoded), context.to_dict())


if __name__ == "__main__":
    unittest.main()

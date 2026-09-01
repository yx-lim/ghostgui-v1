"""Tests for the first strict semantic motion-tool allowlist."""

from __future__ import annotations

import json
import unittest

from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_services import (
    LogicalFrameSolveResult,
    MotionValidationReport,
)
from application.ai.schemas import EditAuthor
from application.ai.semantic_tools import (
    SemanticToolContext,
    build_semantic_tool_registry,
)
from application.ai.errors import ToolExecutionError, ToolValidationError
from application.editor_commands import UpdateKeyframe
from application.editor_controller import EditorController
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class FakeTimeline:
    def __init__(self, states=()):
        self.states = {}
        for time, value in states:
            self.set_state(time, value)

    def set_state(self, time, value):
        self.states[round(float(time), 6)] = list(value)

    def get_state(self, time):
        value = self.states.get(round(float(time), 6))
        return None if value is None else value.copy()

    def times(self):
        return sorted(self.states)

    def sample_state(self, time):
        key = round(float(time), 6)
        if key in self.states:
            return self.states[key].copy()
        lower = max((value for value in self.states if value < key), default=None)
        upper = min((value for value in self.states if value > key), default=None)
        if lower is None:
            return self.states[upper].copy()
        if upper is None:
            return self.states[lower].copy()
        fraction = (key - lower) / (upper - lower)
        return [
            start + fraction * (end - start)
            for start, end in zip(self.states[lower], self.states[upper])
        ]


class FakeMotionService:
    logical_frames = ("pelvis", "torso", "right_hand")
    end_effectors = ("right_hand",)
    joint_names = ("waist_pitch", "right_shoulder")
    joint_groups = {"upper_body": ("waist_pitch", "right_shoulder")}

    def __init__(self):
        self.solve_calls = []

    def solve_logical_frame_target(
        self,
        document,
        *,
        logical_frame,
        time_seconds,
        position_m,
        orientation_rpy_rad,
        mode,
        protected_logical_frames,
    ):
        self.solve_calls.append({
            "logical_frame": logical_frame,
            "mode": mode,
            "protected": protected_logical_frames,
        })
        existing = document.trajectory.targets_at_time(time_seconds).get(logical_frame)
        start = (0.0, 0.0, 0.0) if existing is None else (existing.x, existing.y, existing.z)
        position = tuple(float(value) for value in position_m)
        if mode == "delta":
            position = tuple(start[index] + position[index] for index in range(3))
        orientation = orientation_rpy_rad or (0.0, 0.0, 0.0)
        return LogicalFrameSolveResult(
            frame=TargetFrame(
                time=time_seconds,
                phase="ai_edit",
                frame_name=logical_frame,
                x=position[0],
                y=position[1],
                z=position[2],
                roll=orientation[0],
                pitch=orientation[1],
                yaw=orientation[2],
            ),
            qpos=[position[2], float(time_seconds)],
            status="fake IK solved",
        )

    def set_joint_angles(
        self,
        document,
        *,
        time_seconds,
        values,
        protected_logical_frames,
    ):
        qpos = document.qpos_timeline.sample_state(time_seconds)
        for index, name in enumerate(self.joint_names):
            if name in values:
                qpos[index] = float(values[name])
        return qpos

    def ensure_qpos_keyframe(self, document, *, time_seconds):
        return document.qpos_timeline.sample_state(time_seconds)

    def validate_motion(self, document):
        return MotionValidationReport(True, ())


def _document():
    document = ProjectDocument(
        "g1",
        timeline_duration=4.0,
        qpos_timeline=FakeTimeline(((0.0, [0.0, 0.0]), (2.0, [0.2, 0.3]))),
    )
    for frame in (
        TargetFrame(time=0.0, frame_name="pelvis", z=0.9),
        TargetFrame(time=0.0, frame_name="right_hand", x=0.3, z=1.0),
        TargetFrame(time=2.0, frame_name="pelvis", z=0.8),
    ):
        document.trajectory.add_frame(frame)
    return document


def _setup():
    committed = _document()
    store = InMemoryMotionMetadataStore()
    resolver = TimestampMotionIdentityResolver()
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(
        session=session,
        metadata=MotionMetadataService(store, resolver),
        motion_name="test motion",
    )
    motion = FakeMotionService()
    registry = build_semantic_tool_registry(motion)
    return committed, session, context, motion, registry


class SemanticToolTests(unittest.TestCase):
    def test_registry_exposes_only_bounded_semantic_tools(self):
        _committed, _session, _context, _motion, registry = _setup()
        names = {definition.name for definition in registry.definitions()}
        self.assertEqual(names, {
            "ensure_keyframe",
            "inspect_motion",
            "move_end_effector",
            "protect_keyframe",
            "retime_segment",
            "set_joint_angle",
            "set_joint_group_angles",
            "set_logical_frame_target",
            "validate_motion",
        })
        self.assertNotIn("run_code", names)
        self.assertNotIn("set_qpos_trajectory", names)

    def test_logical_frame_target_uses_solver_and_only_changes_working_copy(self):
        committed, session, context, motion, registry = _setup()

        result = registry.execute(
            "set_logical_frame_target",
            {
                "logical_frame": "pelvis",
                "time_seconds": 0.0,
                "position_m": [0.0, 0.0, 0.7],
                "orientation_rpy_rad": [0.0, 0.0, 0.0],
                "mode": "absolute",
            },
            context=context,
        )

        self.assertEqual(result["status"], "fake IK solved")
        self.assertEqual(motion.solve_calls[0]["logical_frame"], "pelvis")
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.7)
        self.assertEqual(session.working_document.qpos_timeline.get_state(0.0), [0.7, 0.0])
        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        self.assertEqual(committed.qpos_timeline.get_state(0.0), [0.0, 0.0])

    def test_end_effector_move_is_relative_and_schema_rejects_unknown_names(self):
        _committed, session, context, motion, registry = _setup()
        result = registry.execute(
            "move_end_effector",
            {
                "end_effector": "right_hand",
                "time_seconds": 0.0,
                "delta_m": [0.0, 0.0, 0.1],
            },
            context=context,
        )
        hand = next(
            frame for frame in session.working_document.trajectory.frames
            if frame.frame_name == "right_hand"
        )
        self.assertAlmostEqual(hand.z, 1.1)
        self.assertEqual(motion.solve_calls[-1]["mode"], "delta")
        with self.assertRaises(ToolValidationError):
            registry.execute(
                "move_end_effector",
                {
                    "end_effector": "nonexistent_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, 0.0, 0.1],
                },
                context=context,
            )

    def test_joint_angle_and_group_tools_use_named_joint_service(self):
        _committed, session, context, _motion, registry = _setup()
        registry.execute(
            "set_joint_angle",
            {"joint": "waist_pitch", "time_seconds": 0.0, "angle_rad": 0.4},
            context=context,
        )
        self.assertEqual(session.working_document.qpos_timeline.get_state(0.0), [0.4, 0.0])
        registry.execute(
            "set_joint_group_angles",
            {
                "joint_group": "upper_body",
                "time_seconds": 0.0,
                "angles_rad": [0.2, 0.6],
            },
            context=context,
        )
        self.assertEqual(session.working_document.qpos_timeline.get_state(0.0), [0.2, 0.6])
        with self.assertRaisesRegex(ToolExecutionError, "requires 2"):
            registry.execute(
                "set_joint_group_angles",
                {
                    "joint_group": "upper_body",
                    "time_seconds": 0.0,
                    "angles_rad": [0.2],
                },
                context=context,
            )

    def test_protected_keyframe_is_passed_to_ik_and_cannot_be_modified(self):
        _committed, session, context, motion, registry = _setup()
        registry.execute(
            "set_logical_frame_target",
            {
                "logical_frame": "pelvis",
                "time_seconds": 0.0,
                "position_m": [0.0, 0.0, 0.85],
                "mode": "absolute",
            },
            context=context,
        )
        registry.execute(
            "protect_keyframe",
            {"logical_frame": "pelvis", "time_seconds": 0.0, "protected": True},
            context=context,
        )
        with self.assertRaisesRegex(ToolExecutionError, "protected"):
            registry.execute(
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.6],
                    "mode": "absolute",
                },
                context=context,
            )
        self.assertIn("pelvis", motion.solve_calls[-1]["protected"])
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.85)

    def test_protection_only_session_can_be_accepted_once(self):
        committed, session, context, _motion, registry = _setup()
        result = registry.execute(
            "protect_keyframe",
            {"logical_frame": "pelvis", "time_seconds": 0.0, "protected": True},
            context=context,
        )

        accepted = session.accept(EditorController(committed))

        frame = next(
            item for item in committed.trajectory.frames
            if item.frame_name == "pelvis" and item.time == 0.0
        )
        reference = context.metadata.reference_for_keyframe(frame)
        self.assertTrue(result["changed"])
        self.assertTrue(accepted.changed)
        self.assertEqual(committed.revision, 1)
        self.assertTrue(context.metadata.store.get(reference).protected)

    def test_retime_migrates_metadata_without_ai_timestamp_identity(self):
        _committed, session, context, _motion, registry = _setup()
        metadata = context.working_metadata
        before = next(
            frame for frame in session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 2.0
        )
        before_ref = metadata.reference_for_keyframe(before)
        session.metadata.record(before_ref, EditAuthor.AI)

        registry.execute(
            "retime_segment",
            {
                "start_time_seconds": 0.0,
                "end_time_seconds": 2.0,
                "speed": 2.0,
            },
            context=context,
        )

        after = next(
            frame for frame in session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 1.0
        )
        after_ref = metadata.reference_for_keyframe(after)
        self.assertIsNone(session.metadata.get(before_ref))
        self.assertEqual(session.metadata.get(after_ref).author, EditAuthor.AI)

    def test_inspect_and_validate_are_read_only_and_contain_no_qpos_values(self):
        _committed, session, context, _motion, registry = _setup()
        revision = session.working_document.revision
        inspected = registry.execute("inspect_motion", {}, context=context)
        validated = registry.execute("validate_motion", {}, context=context)
        serialized = json.dumps(inspected)

        self.assertEqual(inspected["motion"]["name"], "test motion")
        self.assertEqual(validated, {"valid": True, "issues": []})
        self.assertNotIn("qpos_values", serialized)
        self.assertEqual(session.working_document.revision, revision)


if __name__ == "__main__":
    unittest.main()

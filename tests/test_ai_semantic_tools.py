"""Tests for the first strict semantic motion-tool allowlist."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import numpy as np

from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_services import (
    GhostGUIMotionService,
    JointAngleEditResult,
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
from application.timeslice_service import capture_timeslice_from_committed_pose
from core.models import MuJoCoRobotAdapter, RobotStateTimeline
from core.trajectory import TargetFrame, rpy_to_quat


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
        return JointAngleEditResult(qpos)

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


def _mark_existing_motion_ai_owned(session, resolver):
    metadata = MotionMetadataService(session.metadata, resolver)
    for frame in session.working_document.trajectory.frames:
        session.metadata.record(
            metadata.reference_for_keyframe(frame),
            EditAuthor.AI,
        )
    timeline = session.working_document.qpos_timeline
    if timeline is not None:
        for time in timeline.times():
            session.metadata.record(
                metadata.reference_for_qpos_keyframe(time),
                EditAuthor.AI,
            )


def _setup():
    committed = _document()
    store = InMemoryMotionMetadataStore()
    resolver = TimestampMotionIdentityResolver()
    session = AIEditSession(committed, metadata_store=store)
    _mark_existing_motion_ai_owned(session, resolver)
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
            {"logical_frame": "pelvis", "time_seconds": 0.0},
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
            {"logical_frame": "pelvis", "time_seconds": 0.0},
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

    def test_protection_tool_rejects_ai_unprotect_argument(self):
        _committed, _session, context, _motion, registry = _setup()

        with self.assertRaisesRegex(ToolValidationError, "unknown property"):
            registry.execute(
                "protect_keyframe",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "protected": False,
                },
                context=context,
            )

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
        self.assertEqual(validated, {
            "valid": True,
            "issues": [],
            "scope": "structural_kinematic",
            "dynamic_feasibility_assessed": False,
        })
        self.assertNotIn("qpos_values", serialized)
        self.assertEqual(session.working_document.revision, revision)


class GhostGUIMotionServiceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MuJoCoRobotAdapter("g1")
        timeline = RobotStateTimeline(self.adapter)
        document = ProjectDocument(
            "g1",
            timeline_duration=2.0,
            qpos_timeline=timeline,
        )
        state = self.adapter.create_state()
        state.set_qpos(self.adapter.home_qpos)
        for frame in capture_timeslice_from_committed_pose(
            state,
            time=0.0,
            phase="test",
            frame_names=self.adapter.trajectory_frames,
            frame_bindings=self.adapter.logical_frame_bindings,
        ):
            document.trajectory.add_frame(frame)

        store = InMemoryMotionMetadataStore()
        resolver = TimestampMotionIdentityResolver()
        self.session = AIEditSession(document, metadata_store=store)
        self.context = SemanticToolContext(
            session=self.session,
            metadata=MotionMetadataService(store, resolver),
        )
        _mark_existing_motion_ai_owned(self.session, resolver)
        self.motion = GhostGUIMotionService(self.adapter)
        self.registry = build_semantic_tool_registry(self.motion)
        self.committed = document

    def assert_logical_frames_match_fk(self, frame_names):
        working = self.session.working_document
        state = self.adapter.create_state()
        state.set_qpos(working.qpos_timeline.get_state(0.0))
        targets = working.trajectory.targets_at_time(0.0)
        for frame_name in frame_names:
            with self.subTest(frame=frame_name):
                kind, object_name = self.adapter.logical_frame_bindings[
                    frame_name
                ]
                position, quaternion = state.get_body_pose(object_name, kind)
                target = targets[frame_name]
                np.testing.assert_allclose(
                    [target.x, target.y, target.z],
                    position,
                    atol=1e-9,
                )
                expected_quaternion = np.asarray(
                    rpy_to_quat(target.roll, target.pitch, target.yaw)
                )
                self.assertAlmostEqual(
                    abs(float(np.dot(expected_quaternion, quaternion))),
                    1.0,
                    places=8,
                )

    def test_joint_angle_updates_qpos_and_fk_targets_atomically(self):
        joint = "right_elbow_joint"
        before_qpos = self.committed.qpos_timeline.get_state(0.0)
        before_hand = self.committed.trajectory.targets_at_time(0.0)[
            "right_hand"
        ].to_dict()
        angle = self.adapter.create_state().get_joint_value(joint) + 0.05

        result = self.registry.execute(
            "set_joint_angle",
            {"joint": joint, "time_seconds": 0.0, "angle_rad": angle},
            context=self.context,
        )

        self.assertIn("right_hand", result["updated_logical_frames"])
        self.assertEqual(self.session.working_document.revision, 1)
        self.assertEqual(len(self.session.edits), 1)
        self.assert_logical_frames_match_fk(result["updated_logical_frames"])
        working_state = self.adapter.create_state()
        working_state.set_qpos(
            self.session.working_document.qpos_timeline.get_state(0.0)
        )
        self.assertAlmostEqual(working_state.get_joint_value(joint), angle)
        np.testing.assert_allclose(
            self.committed.qpos_timeline.get_state(0.0),
            before_qpos,
        )
        self.assertEqual(
            self.committed.trajectory.targets_at_time(0.0)[
                "right_hand"
            ].to_dict(),
            before_hand,
        )

    def test_joint_group_uses_the_same_atomic_fk_synchronization(self):
        joints = ("right_shoulder_pitch_joint", "right_elbow_joint")
        self.adapter.joint_groups = {"right_arm": joints}
        self.motion = GhostGUIMotionService(self.adapter)
        self.registry = build_semantic_tool_registry(self.motion)
        state = self.adapter.create_state()
        angles = [state.get_joint_value(name) + 0.03 for name in joints]

        result = self.registry.execute(
            "set_joint_group_angles",
            {
                "joint_group": "right_arm",
                "time_seconds": 0.0,
                "angles_rad": angles,
            },
            context=self.context,
        )

        self.assertIn("right_hand", result["updated_logical_frames"])
        self.assertEqual(self.session.working_document.revision, 1)
        self.assertEqual(len(self.session.edits), 1)
        self.assert_logical_frames_match_fk(result["updated_logical_frames"])
        working_state = self.adapter.create_state()
        working_state.set_qpos(
            self.session.working_document.qpos_timeline.get_state(0.0)
        )
        for name, angle in zip(joints, angles):
            self.assertAlmostEqual(working_state.get_joint_value(name), angle)

    def test_user_owned_affected_target_rejects_the_whole_joint_edit(self):
        working_metadata = self.context.working_metadata
        right_hand = self.session.working_document.trajectory.targets_at_time(
            0.0
        )["right_hand"]
        reference = working_metadata.reference_for_keyframe(right_hand)
        self.session.metadata.record(reference, EditAuthor.USER)
        before_qpos = self.session.working_document.qpos_timeline.get_state(0.0)
        before_hand = right_hand.to_dict()
        angle = (
            self.adapter.create_state().get_joint_value("right_elbow_joint")
            + 0.05
        )

        with self.assertRaisesRegex(ToolExecutionError, "user-authored"):
            self.registry.execute(
                "set_joint_angle",
                {
                    "joint": "right_elbow_joint",
                    "time_seconds": 0.0,
                    "angle_rad": angle,
                },
                context=self.context,
            )

        np.testing.assert_allclose(
            self.session.working_document.qpos_timeline.get_state(0.0),
            before_qpos,
        )
        self.assertEqual(
            self.session.working_document.trajectory.targets_at_time(0.0)[
                "right_hand"
            ].to_dict(),
            before_hand,
        )
        self.assertEqual(self.session.working_document.revision, 0)
        self.assertFalse(self.session.has_changes)

    def test_structural_kinematic_validation_accepts_consistent_motion(self):
        report = self.motion.validate_motion(self.session.working_document)

        self.assertTrue(report.valid, report.issues)
        self.assertEqual(report.issues, ())

    def test_validation_checks_model_qpos_limits_and_all_time_contracts(self):
        document = self.session.working_document
        width = self.adapter.mj_model.nq
        timeline = document.qpos_timeline
        timeline.states[0.0] = np.zeros(width - 1)
        non_finite = self.adapter.home_qpos.copy()
        non_finite[0] = np.nan
        timeline.states[0.5] = non_finite
        outside_limit = self.adapter.home_qpos.copy()
        joint = self.adapter.joints["right_elbow_joint"]
        outside_limit[joint.qpos_address] = joint.limits[1] + 0.5
        timeline.states[1.0] = outside_limit
        timeline.states[3.0] = self.adapter.home_qpos.copy()
        timeline.robot_model = SimpleNamespace(
            info=SimpleNamespace(key="go2")
        )
        document.model_key = "go2"
        document.current_time = 3.0
        document.trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="unknown_frame",
        ))
        negative_time = document.trajectory.frames[0]
        negative_time.time = -0.1

        report = self.motion.validate_motion(document)
        message = "\n".join(report.issues)

        self.assertFalse(report.valid)
        self.assertIn("does not match active model", message)
        self.assertIn("qpos timeline model go2 does not match", message)
        self.assertIn(f"must contain {width} values", message)
        self.assertIn("non-finite value", message)
        self.assertIn("outside its model limits", message)
        self.assertIn("qpos Keyframe at 3.000 s exceeds", message)
        self.assertIn("Current motion time exceeds", message)
        self.assertIn("Unknown logical frame unknown_frame", message)
        self.assertIn("has a negative time", message)

    def test_validation_checks_duration_fk_and_blocking_collisions(self):
        document = self.session.working_document
        right_hand = next(
            frame
            for frame in document.trajectory.frames
            if frame.frame_name == "right_hand"
        )
        right_hand.x += 0.1
        blocking_collision = SimpleNamespace(blocking=True)
        solver = SimpleNamespace(
            collision_checker=SimpleNamespace(
                get_collisions=lambda _state: (blocking_collision,)
            )
        )
        motion = GhostGUIMotionService(
            self.adapter,
            collision_solver=solver,
        )

        report = motion.validate_motion(document)
        message = "\n".join(report.issues)

        self.assertFalse(report.valid)
        self.assertIn("does not match qpos forward kinematics", message)
        self.assertIn("1 blocking collision", message)

        document.timeline_duration = float("nan")
        duration_report = motion.validate_motion(document)
        self.assertIn(
            "Motion duration must be positive and finite",
            duration_report.issues,
        )


if __name__ == "__main__":
    unittest.main()

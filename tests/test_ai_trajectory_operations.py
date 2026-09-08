"""Tests for deterministic local compact trajectory operations."""

from __future__ import annotations

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
    JointAngleEditResult,
    LogicalFrameSolveResult,
    MotionValidationReport,
)
from application.ai.trajectory_edit_spec import (
    TrajectoryEditMode,
    TrajectoryEditSpec,
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import (
    TrajectoryExecutionContext,
    TrajectorySpecExecutor,
)
from application.ai.trajectory_operations import build_trajectory_operation_handlers
from application.editor_controller import EditorController
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class Timeline:
    def __init__(self):
        self.states = {
            0.0: np.array([1, 2, 0.8, 1, 0, 0, 0, 0.1], dtype=float),
            1.0: np.array([2, 3, 0.9, 1, 0, 0, 0, 0.2], dtype=float),
        }

    def times(self):
        return sorted(self.states)

    def get_state(self, time):
        return self.states[float(time)].copy()

    def set_state(self, time, qpos):
        self.states[float(time)] = np.asarray(qpos, dtype=float).copy()

    def sample_state(self, time):
        time = float(time)
        if time in self.states:
            return self.states[time].copy()
        lower = max(value for value in self.states if value < time)
        upper = min(value for value in self.states if value > time)
        fraction = (time - lower) / (upper - lower)
        return self.states[lower] + (self.states[upper] - self.states[lower]) * fraction


def _setup(*, protected=False):
    document = ProjectDocument(
        "g1",
        timeline_duration=1.0,
        qpos_timeline=Timeline(),
    )
    for time, z in ((0.0, 0.8), (1.0, 0.9)):
        document.trajectory.add_frame(
            TargetFrame(time=time, frame_name="pelvis", x=time, y=0.2, z=z)
        )
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    metadata.seed_document_as_user_owned(document)
    if protected:
        store.set_protected(
            metadata.reference_for_qpos_keyframe(0.0),
            True,
        )
    session = AIEditSession(document, metadata_store=store)
    motion = SimpleNamespace(
        adapter=SimpleNamespace(
            free_joints_by_body={0: SimpleNamespace(qpos_address=0)},
        ),
        validate_motion=lambda _document: MotionValidationReport(True),
    )
    handlers = build_trajectory_operation_handlers(motion, metadata)
    executor = TrajectorySpecExecutor(handlers, motion.validate_motion)
    spec = TrajectoryEditSpec(
        TrajectoryEditMode.EDIT,
        "Raise the whole robot by five centimetres.",
        (TrajectoryOperation(
            TrajectoryOperationType.ROOT_OFFSET,
            {"start_time": 0.0, "end_time": 1.0, "translation_m": [0, 0, 0.05]},
        ),),
    )
    return document, session, executor, spec


class RootOffsetTests(unittest.TestCase):
    def test_root_offset_changes_only_root_xyz_and_matching_logical_positions(self):
        committed, session, executor, spec = _setup()
        before = {
            time: committed.qpos_timeline.get_state(time)
            for time in committed.qpos_timeline.times()
        }

        executor.execute(spec, context=TrajectoryExecutionContext(session, object()))

        self.assertTrue(session.can_accept)
        for time in (0.0, 1.0):
            after = session.working_document.qpos_timeline.get_state(time)
            np.testing.assert_allclose(after[:2], before[time][:2])
            self.assertAlmostEqual(after[2], before[time][2] + 0.05)
            np.testing.assert_array_equal(after[3:], before[time][3:])
        self.assertEqual(
            [round(frame.z, 6) for frame in session.working_document.trajectory.frames],
            [0.85, 0.95],
        )
        np.testing.assert_array_equal(
            committed.qpos_timeline.get_state(0.0),
            before[0.0],
        )

        session.accept(EditorController(committed))
        self.assertAlmostEqual(committed.qpos_timeline.get_state(0.0)[2], 0.85)

    def test_protected_root_keyframe_rejects_entire_operation(self):
        committed, session, executor, spec = _setup(protected=True)
        before = committed.qpos_timeline.get_state(0.0)

        with self.assertRaisesRegex(Exception, "protected"):
            executor.execute(spec, context=TrajectoryExecutionContext(session, object()))

        np.testing.assert_array_equal(
            session.working_document.qpos_timeline.get_state(0.0),
            before,
        )
        self.assertFalse(session.has_changes)


class _HoldState:
    def __init__(self, adapter):
        self.adapter = adapter
        self.qpos = np.zeros(9)

    def set_qpos(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float).copy()

    def get_joint_value(self, name):
        return float(self.qpos[self.adapter.joints[name].qpos_address])


class _HoldMotion:
    def __init__(self):
        self.adapter = SimpleNamespace(
            create_state=lambda: _HoldState(self.adapter),
            joints={
                "right_shoulder": SimpleNamespace(qpos_address=7),
                "left_knee": SimpleNamespace(qpos_address=8),
            },
            free_joints_by_body={0: SimpleNamespace(qpos_address=0)},
            logical_frame_bindings={},
        )
        self.joint_names = ("right_shoulder", "left_knee")
        self.joint_groups = {"right_arm": ("right_shoulder",)}

    def set_joint_angles(
        self,
        document,
        *,
        time_seconds,
        values,
        protected_logical_frames,
    ):
        qpos = document.qpos_timeline.sample_state(time_seconds)
        for name, value in values.items():
            qpos[self.adapter.joints[name].qpos_address] = value
        return JointAngleEditResult(qpos)

    def validate_motion(self, _document):
        return MotionValidationReport(True)


class HoldPoseTests(unittest.TestCase):
    def test_joint_group_hold_samples_source_and_preserves_unrelated_values(self):
        committed = ProjectDocument(
            "g1",
            timeline_duration=1.0,
            qpos_timeline=Timeline(),
        )
        committed.qpos_timeline.states[0.0] = np.array(
            [1, 2, 0.8, 1, 0, 0, 0, 0.1, 0.2],
            dtype=float,
        )
        committed.qpos_timeline.states[1.0] = np.array(
            [2, 3, 0.9, 1, 0, 0, 0, 0.9, 0.6],
            dtype=float,
        )
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        motion = _HoldMotion()
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        spec = TrajectoryEditSpec(
            TrajectoryEditMode.EDIT,
            "Hold the right arm.",
            (TrajectoryOperation(
                TrajectoryOperationType.HOLD_POSE,
                {
                    "source_time": 0.5,
                    "start_time": 0.5,
                    "end_time": 1.0,
                    "body_scope": "joint_group",
                    "body_name": "right_arm",
                },
            ),),
        )
        unrelated = {
            time: committed.qpos_timeline.get_state(time)[8]
            for time in committed.qpos_timeline.times()
        }

        executor.execute(spec, context=TrajectoryExecutionContext(session, object()))

        before_interval = session.working_document.qpos_timeline.get_state(0.0)
        self.assertAlmostEqual(before_interval[7], 0.1)
        self.assertAlmostEqual(before_interval[8], unrelated[0.0])
        end = session.working_document.qpos_timeline.get_state(1.0)
        self.assertAlmostEqual(end[7], 0.5)
        self.assertAlmostEqual(end[8], unrelated[1.0])
        inserted = session.working_document.qpos_timeline.get_state(0.5)
        self.assertAlmostEqual(inserted[7], 0.5)
        self.assertAlmostEqual(inserted[8], 0.4)
        self.assertTrue(session.can_accept)


class RetimeIntervalTests(unittest.TestCase):
    def test_duration_scale_reuses_atomic_timeline_retiming(self):
        committed = ProjectDocument(
            "g1",
            timeline_duration=1.0,
            qpos_timeline=Timeline(),
        )
        committed.trajectory.add_frame(
            TargetFrame(time=0.0, frame_name="pelvis", z=0.8)
        )
        committed.trajectory.add_frame(
            TargetFrame(time=1.0, frame_name="pelvis", z=0.9)
        )
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        motion = _HoldMotion()
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        spec = TrajectoryEditSpec(
            TrajectoryEditMode.EDIT,
            "Make the interval 50 percent slower.",
            (TrajectoryOperation(
                TrajectoryOperationType.RETIME_INTERVAL,
                {"start_time": 0.0, "end_time": 1.0, "scale": 1.5},
            ),),
        )

        result = executor.execute(
            spec,
            context=TrajectoryExecutionContext(session, object()),
        )

        self.assertEqual(
            session.working_document.qpos_timeline.times(),
            [0.0, 1.5],
        )
        self.assertEqual(
            [frame.time for frame in session.working_document.trajectory.frames],
            [0.0, 1.5],
        )
        self.assertAlmostEqual(session.working_document.timeline_duration, 1.5)
        self.assertAlmostEqual(
            result.operations[0].output["duration_scale"],
            1.5,
        )
        self.assertTrue(session.can_accept)


class JointTargetTests(unittest.TestCase):
    def _execute(self, operation):
        committed = ProjectDocument(
            "g1",
            timeline_duration=1.0,
            qpos_timeline=Timeline(),
        )
        committed.qpos_timeline.states[0.0] = np.array(
            [1, 2, 0.8, 1, 0, 0, 0, 0.1, 0.2], dtype=float
        )
        committed.qpos_timeline.states[1.0] = np.array(
            [2, 3, 0.9, 1, 0, 0, 0, 0.9, 0.6], dtype=float
        )
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        motion = _HoldMotion()
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.EDIT,
                "Set explicit Joint Angles.",
                (operation,),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )
        return committed, session

    def test_named_joint_target_uses_interpolated_posture_and_preserves_others(self):
        committed, session = self._execute(TrajectoryOperation(
            TrajectoryOperationType.SET_JOINT_TARGET,
            {
                "joint": "right_shoulder",
                "time_seconds": 0.5,
                "angle_rad": 0.25,
            },
        ))

        qpos = session.working_document.qpos_timeline.get_state(0.5)
        self.assertAlmostEqual(qpos[7], 0.25)
        self.assertAlmostEqual(qpos[8], 0.4)
        self.assertIsNone(committed.qpos_timeline.states.get(0.5))
        self.assertTrue(session.can_accept)

    def test_joint_group_target_rejects_names_outside_registered_group(self):
        with self.assertRaisesRegex(Exception, "not part of group"):
            self._execute(TrajectoryOperation(
                TrajectoryOperationType.SET_JOINT_GROUP_TARGET,
                {
                    "joint_group": "right_arm",
                    "time_seconds": 0.5,
                    "joint_angles_rad": [
                        {"joint": "left_knee", "angle_rad": 0.3},
                    ],
                },
            ))


class _EndEffectorState:
    def __init__(self):
        self.qpos = None

    def set_qpos(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)

    def get_body_pose(self, object_name, kind):
        del kind
        offset = 0.1 if object_name == "right_hand_site" else -0.1
        return np.array([self.qpos[0], offset, self.qpos[2]]), np.array([1, 0, 0, 0])


class _EndEffectorMotion:
    end_effectors = ("right_hand", "left_hand")
    joint_names = ()
    joint_groups = {}

    def __init__(self):
        self.calls = []
        self.adapter = SimpleNamespace(
            create_state=lambda: _EndEffectorState(),
            logical_frame_bindings={
                "right_hand": ("site", "right_hand_site"),
                "left_hand": ("site", "left_hand_site"),
            },
        )

    def solve_logical_frame_target(self, document, **arguments):
        self.calls.append(arguments)
        qpos = document.qpos_timeline.sample_state(arguments["time_seconds"])
        position = np.asarray(arguments["position_m"], dtype=float)
        if arguments["mode"] == "delta":
            position = position + np.array([qpos[0], 0.1, qpos[2]])
        qpos = np.asarray(qpos, dtype=float).copy()
        qpos[0:3] = position
        return LogicalFrameSolveResult(
            TargetFrame(
                time=arguments["time_seconds"],
                frame_name=arguments["logical_frame"],
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            qpos,
            "solved",
        )

    def validate_motion(self, _document):
        return MotionValidationReport(True)


class EndEffectorOperationTests(unittest.TestCase):
    def _execute(self, operation):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        motion = _EndEffectorMotion()
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.EDIT,
                "Apply an End Effector target.",
                (operation,),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )
        return committed, session, motion

    def test_relative_end_effector_target_uses_existing_ik_for_interval(self):
        committed, session, motion = self._execute(TrajectoryOperation(
            TrajectoryOperationType.SET_END_EFFECTOR_TARGET,
            {
                "end_effector": "right_hand",
                "start_time": 0.0,
                "end_time": 1.0,
                "mode": "relative",
                "position_m": [0, 0, 0.1],
            },
        ))

        self.assertEqual([call["mode"] for call in motion.calls], ["delta", "delta"])
        self.assertAlmostEqual(session.working_document.qpos_timeline.get_state(0.0)[2], 0.9)
        self.assertAlmostEqual(committed.qpos_timeline.get_state(0.0)[2], 0.8)
        self.assertTrue(session.can_accept)

    def test_lock_uses_fk_source_pose_as_absolute_target(self):
        _committed, _session, motion = self._execute(TrajectoryOperation(
            TrajectoryOperationType.LOCK_END_EFFECTOR,
            {
                "end_effectors": ["right_hand"],
                "source_time": 0.5,
                "start_time": 0.0,
                "end_time": 1.0,
            },
        ))

        self.assertEqual([call["mode"] for call in motion.calls], ["absolute", "absolute"])
        for call in motion.calls:
            np.testing.assert_allclose(call["position_m"], [1.5, 0.1, 0.85])


class _GenerationMotion(_EndEffectorMotion):
    logical_frames = ("torso", "right_hand", "left_hand")
    joint_names = ("right_shoulder",)

    def __init__(self):
        super().__init__()
        self.adapter.free_joints_by_body = {0: SimpleNamespace(qpos_address=0)}
        self.adapter.trajectory_frames = self.logical_frames
        self.adapter.logical_frame_bindings["torso"] = ("body", "torso_site")

    def set_joint_angles(self, document, **arguments):
        qpos = document.qpos_timeline.sample_state(arguments["time_seconds"])
        qpos[7] = arguments["values"]["right_shoulder"]
        return JointAngleEditResult(qpos)


class SparseGenerationTests(unittest.TestCase):
    def test_sparse_targets_are_solved_then_interpolated_to_dense_qpos(self):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        motion = _GenerationMotion()
        executor = TrajectorySpecExecutor(
            build_trajectory_operation_handlers(motion, metadata),
            motion.validate_motion,
        )
        keyframes = []
        for time, z, shoulder in (
            (0.0, 0.8, 0.0),
            (0.25, 0.6, 0.3),
            (0.5, 0.7, 0.6),
            (1.0, 0.9, 0.1),
        ):
            keyframes.append({
                "time_seconds": time,
                "root_position_m": [0.0, 0.0, z],
                "torso_rpy_rad": None,
                "end_effector_targets": [],
                "joint_targets": [
                    {"joint": "right_shoulder", "angle_rad": shoulder},
                ],
            })
        executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.GENERATE,
                "Generate a short motion.",
                (TrajectoryOperation(
                    TrajectoryOperationType.SPARSE_KEYFRAMES,
                    {"duration_seconds": 1.0, "keyframes": keyframes},
                ),),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )

        generated = session.working_document
        self.assertEqual(len(generated.qpos_timeline.times()), 101)
        self.assertAlmostEqual(generated.qpos_timeline.get_state(0.25)[2], 0.6)
        self.assertAlmostEqual(generated.qpos_timeline.get_state(0.25)[7], 0.3)
        self.assertEqual(generated.timeline_duration, 1.0)
        self.assertTrue(session.can_accept)
        self.assertEqual(committed.qpos_timeline.times(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

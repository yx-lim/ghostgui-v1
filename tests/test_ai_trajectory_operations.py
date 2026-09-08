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
from application.ai.motion_services import MotionValidationReport
from application.ai.motion_services import JointAngleEditResult
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


if __name__ == "__main__":
    unittest.main()

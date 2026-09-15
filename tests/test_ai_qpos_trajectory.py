"""Behavioral coverage for qpos-native Motion Assistant trajectories."""

from __future__ import annotations

from unittest import mock
import unittest

import numpy as np

from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_services import GhostGUIMotionService
from application.ai.qpos_trajectory import validate_qpos_anchors
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
from application.timeslice_service import capture_timeslice_from_committed_pose
from core.models import MuJoCoRobotAdapter, RobotStateTimeline


def _setup(duration=2.0):
    adapter = MuJoCoRobotAdapter("g1")
    timeline = RobotStateTimeline(adapter)
    for time in sorted({
        value for value in (0.5, 1.0, 1.5, duration) if value <= duration
    }):
        timeline.set_state(time, adapter.home_qpos)
    document = ProjectDocument(
        "g1",
        timeline_duration=duration,
        qpos_timeline=timeline,
    )
    for time in (0.0, 1.0, duration):
        state = adapter.create_state()
        state.set_qpos(timeline.sample_state(time))
        for frame in capture_timeslice_from_committed_pose(
            state,
            time=time,
            phase="source",
            frame_names=adapter.trajectory_frames,
            frame_bindings=adapter.logical_frame_bindings,
        ):
            document.trajectory.add_frame(frame)
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    metadata.seed_document_as_user_owned(document)
    session = AIEditSession(document, metadata_store=store)
    motion = GhostGUIMotionService(adapter)
    executor = TrajectorySpecExecutor(
        build_trajectory_operation_handlers(motion, metadata),
        motion.validate_motion,
    )
    return adapter, document, session, motion, executor


def _operation(mode, duration, start, end, anchors):
    return TrajectoryOperation(
        TrajectoryOperationType.QPOS_KEYFRAMES,
        {
            "mode": mode,
            "duration_seconds": duration,
            "start_time": start,
            "end_time": end,
            "keyframes": [
                {"time_seconds": time, "qpos": qpos.tolist()}
                for time, qpos in anchors
            ],
        },
    )


def _execute(executor, session, operation):
    mode = (
        TrajectoryEditMode.GENERATE
        if operation.arguments["mode"] == "replace"
        else TrajectoryEditMode.EDIT
    )
    return executor.execute(
        TrajectoryEditSpec(mode, "Generate full robot states.", (operation,)),
        context=TrajectoryExecutionContext(session, object()),
    )


class QposNativeTrajectoryTests(unittest.TestCase):
    def test_g1_replace_bypasses_ik_interpolates_and_rebuilds_fk(self):
        adapter, committed, session, motion, executor = _setup(duration=1.0)
        self.assertEqual(adapter.mj_model.nq, 36)
        start = adapter.home_qpos.copy()
        middle = start.copy()
        middle[adapter.joints["left_knee_joint"].qpos_address] += 0.1
        middle[3:7] *= 2.0
        end = start.copy()
        operation = _operation(
            "replace",
            1.0,
            0.0,
            1.0,
            ((0.0, start), (0.333, middle), (1.0, end)),
        )
        checker = motion.collision_solver.collision_checker

        with (
            mock.patch.object(
                motion.collision_solver,
                "solve_drag",
                side_effect=AssertionError("qpos generation must not invoke IK"),
            ),
            mock.patch.object(
                checker,
                "get_collisions",
                wraps=checker.get_collisions,
            ) as collision_check,
        ):
            result = _execute(executor, session, operation)

        generated = session.working_document
        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertTrue(session.can_accept)
        self.assertGreater(collision_check.call_count, 0)
        self.assertIn(0.333, generated.qpos_timeline.times())
        self.assertGreater(len(generated.qpos_timeline.times()), 101)
        self.assertAlmostEqual(
            generated.qpos_timeline.get_state(0.333)[
                adapter.joints["left_knee_joint"].qpos_address
            ],
            middle[adapter.joints["left_knee_joint"].qpos_address],
        )
        for time in generated.qpos_timeline.times():
            self.assertAlmostEqual(
                np.linalg.norm(generated.qpos_timeline.get_state(time)[3:7]),
                1.0,
                places=8,
            )
        self.assertEqual(
            len(generated.trajectory.frames),
            3 * len(adapter.trajectory_frames),
        )
        self.assertEqual(committed.qpos_timeline.times(), [0.0, 0.5, 1.0])

        session.reject()
        self.assertFalse(session.can_accept)
        self.assertEqual(committed.qpos_timeline.times(), [0.0, 0.5, 1.0])

    def test_patch_preserves_outside_motion_and_exact_boundary_states(self):
        adapter, committed, session, _motion, executor = _setup()
        outside_before = {
            time: committed.qpos_timeline.get_state(time)
            for time in (0.0, 2.0)
        }
        outside_frames_before = {
            (frame.time, frame.frame_name): frame.to_dict()
            for frame in committed.trajectory.frames
            if frame.time in (0.0, 2.0)
        }
        boundary_before = {
            time: committed.qpos_timeline.get_state(time)
            for time in (0.5, 1.5)
        }
        revised = adapter.home_qpos.copy()
        knee_address = adapter.joints["right_knee_joint"].qpos_address
        revised[knee_address] += 0.15
        operation = _operation(
            "patch",
            2.0,
            0.5,
            1.5,
            ((1.0, revised),),
        )

        result = _execute(executor, session, operation)
        patched = session.working_document

        self.assertTrue(result.validation.valid, result.validation.issues)
        for time, expected in outside_before.items():
            np.testing.assert_array_equal(
                patched.qpos_timeline.get_state(time),
                expected,
            )
        for time, expected in boundary_before.items():
            np.testing.assert_array_equal(
                patched.qpos_timeline.get_state(time),
                expected,
            )
        self.assertAlmostEqual(
            patched.qpos_timeline.get_state(1.0)[knee_address],
            revised[knee_address],
        )
        outside_frames_after = {
            (frame.time, frame.frame_name): frame.to_dict()
            for frame in patched.trajectory.frames
            if frame.time in (0.0, 2.0)
        }
        self.assertEqual(outside_frames_after, outside_frames_before)
        self.assertEqual(
            sorted({frame.time for frame in patched.trajectory.frames}),
            [0.0, 0.5, 1.0, 1.5, 2.0],
        )
        self.assertTrue(session.can_accept)

        session.accept(EditorController(committed))
        self.assertAlmostEqual(
            committed.qpos_timeline.get_state(1.0)[knee_address],
            revised[knee_address],
        )

    def test_malformed_model_states_fail_and_restore_the_working_copy(self):
        for case in ("wrong width", "zero quaternion", "joint limit"):
            with self.subTest(case=case):
                adapter, committed, session, _motion, executor = _setup(duration=1.0)
                invalid = adapter.home_qpos.copy()
                if case == "wrong width":
                    invalid = invalid[:-1]
                elif case == "zero quaternion":
                    invalid[3:7] = 0.0
                else:
                    joint = adapter.joints["left_knee_joint"]
                    invalid[joint.qpos_address] = joint.limits[1] + 0.5
                operation = _operation(
                    "replace",
                    1.0,
                    0.0,
                    1.0,
                    ((0.0, invalid), (1.0, invalid)),
                )
                before = committed.qpos_timeline.get_state(0.0)

                with self.assertRaises(Exception):
                    _execute(executor, session, operation)

                np.testing.assert_array_equal(
                    session.working_document.qpos_timeline.get_state(0.0),
                    before,
                )
                self.assertFalse(session.has_changes)

    def test_qpos_validation_uses_each_active_models_width(self):
        g1 = MuJoCoRobotAdapter("g1")
        go2 = MuJoCoRobotAdapter("go2")

        self.assertNotEqual(g1.mj_model.nq, go2.mj_model.nq)
        anchors = validate_qpos_anchors(
            go2,
            [{"time_seconds": 0.0, "qpos": go2.home_qpos.tolist()}],
        )

        self.assertEqual(len(anchors[0].qpos), go2.mj_model.nq)


if __name__ == "__main__":
    unittest.main()

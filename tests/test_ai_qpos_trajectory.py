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
from application.ai.schemas import EditAuthor
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
from core.math3d import quaternion_angle
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
    for time in sorted({0.0, 1.0, duration}):
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
    def test_g1_root_offset_rebuilds_logical_frames_from_shifted_qpos(self):
        _adapter, _committed, session, _motion, executor = _setup(duration=2.0)
        operation = TrajectoryOperation(
            TrajectoryOperationType.ROOT_OFFSET,
            {
                "start_time": 0.0,
                "end_time": 2.0,
                "translation_m": [0.0, 0.0, 0.05],
            },
        )

        result = executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.EDIT,
                "Move the entire robot five centimetres higher.",
                (operation,),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )

        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertTrue(session.can_accept)
        self.assertEqual(
            result.operations[0].output["affected_qpos_keyframes"],
            len(session.working_document.qpos_timeline.times()),
        )

    def test_qpos_replace_cannot_bypass_protected_logical_keyframe(self):
        adapter, committed, session, _motion, executor = _setup(duration=1.0)
        metadata = MotionMetadataService(
            session.metadata,
            TimestampMotionIdentityResolver(),
        )
        left_foot = committed.trajectory.targets_at_time(0.0)["left_foot"]
        session.metadata.set_protected(
            metadata.reference_for_keyframe(left_foot),
            True,
        )
        start = adapter.home_qpos.copy()
        end = start.copy()
        end[adapter.joints["left_knee_joint"].qpos_address] += 0.05
        operation = _operation(
            "replace",
            1.0,
            0.0,
            1.0,
            ((0.0, start), (1.0, end)),
        )

        with self.assertRaisesRegex(
            Exception,
            r"protected content: left_foot Keyframe at t=0\.00 s",
        ):
            _execute(executor, session, operation)

        self.assertFalse(session.has_changes)

    def test_g1_direct_edit_of_protected_foot_reports_name_and_time(self):
        _adapter, committed, session, _motion, executor = _setup(duration=1.0)
        metadata = MotionMetadataService(
            session.metadata,
            TimestampMotionIdentityResolver(),
        )
        left_foot = committed.trajectory.targets_at_time(0.0)["left_foot"]
        session.metadata.set_protected(
            metadata.reference_for_keyframe(left_foot),
            True,
        )
        operation = TrajectoryOperation(
            TrajectoryOperationType.SET_END_EFFECTOR_TARGET,
            {
                "end_effector": "left_foot",
                "start_time": 0.0,
                "end_time": 0.0,
                "mode": "relative",
                "position_m": [0.0, 0.0, 0.05],
            },
        )

        with self.assertRaisesRegex(
            Exception,
            r"protected left_foot at t=0\.00 s cannot be targeted",
        ):
            executor.execute(
                TrajectoryEditSpec(
                    TrajectoryEditMode.EDIT,
                    "Move the protected left foot.",
                    (operation,),
                ),
                context=TrajectoryExecutionContext(session, object()),
            )

        self.assertFalse(session.has_changes)

    def test_g1_deeper_squat_preserves_protected_planted_feet(self):
        adapter, committed, session, _motion, executor = _setup(duration=1.0)
        metadata = MotionMetadataService(
            session.metadata,
            TimestampMotionIdentityResolver(),
        )
        protected_frames = {
            (frame.frame_name, frame.time): frame.to_dict()
            for frame in committed.trajectory.frames
            if frame.frame_name in {"left_foot", "right_foot"}
        }
        for frame in committed.trajectory.frames:
            if frame.frame_name in {"left_foot", "right_foot"}:
                session.metadata.set_protected(
                    metadata.reference_for_keyframe(frame),
                    True,
                )

        before_poses = {}
        before_pelvis_z = {}
        for time in (0.0, 1.0):
            state = adapter.create_state()
            state.set_qpos(committed.qpos_timeline.sample_state(time))
            pelvis_kind, pelvis_object = adapter.logical_frame_bindings["pelvis"]
            before_pelvis_z[time] = float(
                state.get_body_pose(
                    pelvis_object,
                    pelvis_kind,
                )[0][2]
            )
            for name in ("left_foot", "right_foot"):
                kind, object_name = adapter.logical_frame_bindings[name]
                before_poses[(name, time)] = state.get_body_pose(
                    object_name,
                    kind,
                )

        operation = TrajectoryOperation(
            TrajectoryOperationType.SET_LOGICAL_FRAME_TARGET,
            {
                "logical_frame": "pelvis",
                "start_time": 0.0,
                "end_time": 1.0,
                "mode": "relative",
                "position_m": [0.0, 0.0, -0.03],
                "orientation_rpy_rad": None,
                "locked_end_effectors": ["left_foot", "right_foot"],
            },
        )

        result = executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.EDIT,
                "Make the squat deeper while keeping both feet planted.",
                (operation,),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )

        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertTrue(session.can_accept)
        self.assertEqual(
            result.operations[0].output["locked_end_effectors"],
            ["left_foot", "right_foot"],
        )
        after_targets = {
            (frame.frame_name, frame.time): frame.to_dict()
            for frame in session.working_document.trajectory.frames
            if frame.frame_name in {"left_foot", "right_foot"}
        }
        self.assertEqual(after_targets, protected_frames)

        for time in (0.0, 1.0):
            state = adapter.create_state()
            state.set_qpos(
                session.working_document.qpos_timeline.sample_state(time)
            )
            pelvis_kind, pelvis_object = adapter.logical_frame_bindings["pelvis"]
            pelvis_position, _pelvis_quaternion = state.get_body_pose(
                pelvis_object,
                pelvis_kind,
            )
            self.assertAlmostEqual(
                float(pelvis_position[2]),
                before_pelvis_z[time] - 0.03,
                places=3,
            )
            for name in ("left_foot", "right_foot"):
                kind, object_name = adapter.logical_frame_bindings[name]
                position, quaternion = state.get_body_pose(object_name, kind)
                expected_position, expected_quaternion = before_poses[(name, time)]
                self.assertLessEqual(
                    float(np.linalg.norm(position - expected_position)),
                    0.001,
                )
                self.assertLessEqual(
                    quaternion_angle(quaternion, expected_quaternion),
                    0.01,
                )
                frame = next(
                    item
                    for item in session.working_document.trajectory.frames
                    if item.frame_name == name and item.time == time
                )
                edit_metadata = session.metadata.get(
                    metadata.reference_for_keyframe(frame)
                )
                self.assertEqual(edit_metadata.author, EditAuthor.USER)
                self.assertTrue(edit_metadata.protected)

    def test_g1_repeat_motion_copies_complete_local_qpos_without_provider_reauthoring(self):
        adapter, committed, session, _motion, executor = _setup(duration=1.0)
        self.assertEqual(adapter.mj_model.nq, 36)
        source_times = tuple(committed.qpos_timeline.times())
        source_states = {
            time: committed.qpos_timeline.get_state(time)
            for time in source_times
        }
        operation = TrajectoryOperation(
            TrajectoryOperationType.REPEAT_MOTION,
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "additional_copies": 1,
                "ping_pong": False,
            },
        )

        result = executor.execute(
            TrajectoryEditSpec(
                TrajectoryEditMode.EDIT,
                "Repeat the first second once.",
                (operation,),
            ),
            context=TrajectoryExecutionContext(session, object()),
        )

        repeated = session.working_document
        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertTrue(session.can_accept)
        self.assertEqual(repeated.timeline_duration, 2.0)
        for time in repeated.qpos_timeline.times():
            self.assertEqual(
                repeated.qpos_timeline.get_state(time).shape,
                (adapter.mj_model.nq,),
            )
        for source_time, expected in source_states.items():
            np.testing.assert_array_equal(
                repeated.qpos_timeline.get_state(source_time + 1.0),
                expected,
            )
        self.assertEqual(committed.timeline_duration, 1.0)
        self.assertEqual(tuple(committed.qpos_timeline.times()), source_times)

        metadata = MotionMetadataService(
            session.metadata,
            TimestampMotionIdentityResolver(),
        )
        self.assertEqual(
            session.metadata.get(
                metadata.reference_for_qpos_keyframe(source_times[0])
            ).author,
            EditAuthor.USER,
        )
        self.assertEqual(
            session.metadata.get(
                metadata.reference_for_qpos_keyframe(1.5)
            ).author,
            EditAuthor.AI,
        )
        self.assertEqual(
            result.operations[0].output["inserted_qpos_keyframes"],
            len(source_times) - 1,
        )

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

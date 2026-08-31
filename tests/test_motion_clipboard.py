"""Contracts for committed Motion Clip capture, paste, and repetition."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from application.editor_controller import EditorController
from application.motion_clipboard import (
    MotionClip,
    capture_motion_clip,
    plan_paste_motion,
    plan_repeat_motion,
)
from application.project_document import ProjectDocument
from application.timeline_editing import ApplyTimelineEditPlan, TimelineEditError
from core.models import MuJoCoRobotAdapter, RobotStateTimeline
from core.trajectory import TargetFrame, quat_to_rpy, rpy_to_quat


class FakeStateTimeline:
    def __init__(self, states=()):
        self.states = {}
        for time, qpos in states:
            self.set_state(time, qpos)

    @staticmethod
    def time_key(time):
        return round(float(time), 6)

    def set_state(self, time, qpos):
        self.states[self.time_key(time)] = np.asarray(qpos, dtype=float).copy()

    def get_state(self, time):
        value = self.states.get(self.time_key(time))
        return None if value is None else value.copy()

    def times(self):
        return sorted(self.states)

    def sample_state(self, time):
        key = self.time_key(time)
        exact = self.states.get(key)
        if exact is not None:
            return exact.copy()
        lower = max((item for item in self.states if item < key), default=None)
        upper = min((item for item in self.states if item > key), default=None)
        if lower is None and upper is None:
            raise ValueError("timeline is empty")
        if lower is None:
            return self.states[upper].copy()
        if upper is None:
            return self.states[lower].copy()
        fraction = (key - lower) / (upper - lower)
        return self.states[lower] + fraction * (
            self.states[upper] - self.states[lower]
        )


def target(time, x, frame_name="tool", phase="motion"):
    return TargetFrame(
        time=time,
        phase=phase,
        frame_name=frame_name,
        x=x,
        y=x + 1.0,
        z=x + 2.0,
        roll=x * 0.1,
        pitch=x * 0.2,
        yaw=x * 0.3,
    )


def motion_document(times=(0.0, 1.0), values=None, *, model_key="test"):
    values = tuple(times) if values is None else tuple(values)
    timeline = FakeStateTimeline(
        (time, np.array([value, value + 10.0]))
        for time, value in zip(times, values)
    )
    document = ProjectDocument(
        model_key,
        timeline_duration=2.0,
        qpos_timeline=timeline,
    )
    for time, value in zip(times, values):
        document.trajectory.add_frame(target(time, value))
    return document


class MotionClipboardTests(unittest.TestCase):
    def test_capture_materializes_relative_boundaries_and_is_detached(self):
        document = motion_document((0.0, 1.0, 2.0))

        clip = capture_motion_clip(document, 0.5, 1.5)

        self.assertEqual(clip.model_key, "test")
        self.assertEqual(clip.duration, 1.0)
        self.assertEqual(clip.qpos_width, 2)
        self.assertEqual(
            [frame.time for frame in clip.frames],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(
            [frame.x for frame in clip.frames],
            [0.5, 1.0, 1.5],
        )
        self.assertEqual(
            [time for time, _qpos in clip.states],
            [0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(clip.states[0][1], [0.5, 10.5])
        np.testing.assert_allclose(clip.states[-1][1], [1.5, 11.5])

        document.trajectory.frames[1].x = 99.0
        document.qpos_timeline.states[1.0][0] = 99.0
        self.assertEqual(clip.frames[1].x, 1.0)
        np.testing.assert_allclose(clip.states[1][1], [1.0, 11.0])

    def test_capture_preserves_exact_boundary_phase(self):
        document = motion_document((0.0, 1.0, 2.0))
        for frame, phase in zip(
            document.trajectory.frames,
            ("approach", "contact", "recover"),
        ):
            frame.phase = phase

        clip = capture_motion_clip(document, 1.0, 2.0)

        self.assertEqual(
            [(frame.time, frame.phase) for frame in clip.frames],
            [(0.0, "contact"), (1.0, "recover")],
        )

    def test_capture_uses_sampled_qpos_fk_for_in_between_boundaries(self):
        adapter = MuJoCoRobotAdapter("g1")
        qpos_timeline = RobotStateTimeline(adapter)
        start_qpos = adapter.home_qpos.copy()
        end_qpos = start_qpos.copy()
        shoulder = adapter.joints["right_shoulder_pitch_joint"]
        end_qpos[shoulder.qpos_address] += 0.7
        qpos_timeline.set_state(0.0, start_qpos)
        qpos_timeline.set_state(1.0, end_qpos)
        document = ProjectDocument(
            "g1",
            timeline_duration=1.0,
            qpos_timeline=qpos_timeline,
        )

        kind, object_name = adapter.resolve_logical_frame("right_hand")
        for time, qpos in ((0.0, start_qpos), (1.0, end_qpos)):
            state = adapter.create_state()
            state.set_qpos(qpos)
            position, quaternion = state.get_body_pose(object_name, kind=kind)
            roll, pitch, yaw = quat_to_rpy(quaternion)
            document.trajectory.add_frame(TargetFrame(
                time=time,
                phase="motion",
                frame_name="right_hand",
                x=position[0],
                y=position[1],
                z=position[2],
                roll=roll,
                pitch=pitch,
                yaw=yaw,
            ))

        clip = capture_motion_clip(document, 0.25, 0.75)

        self.assertEqual(len(clip.frames), 2)
        self.assertEqual(len(clip.states), 2)
        for frame, (time, qpos) in zip(clip.frames, clip.states):
            self.assertEqual(frame.time, time)
            state = adapter.create_state()
            state.set_qpos(qpos)
            position, quaternion = state.get_body_pose(object_name, kind=kind)
            np.testing.assert_allclose(
                (frame.x, frame.y, frame.z),
                position,
                atol=1e-10,
            )
            frame_quaternion = rpy_to_quat(
                frame.roll,
                frame.pitch,
                frame.yaw,
            )
            self.assertAlmostEqual(
                abs(float(np.dot(frame_quaternion, quaternion))),
                1.0,
                places=10,
            )

    def test_reverse_paste_coalesces_shared_end_and_applies_once(self):
        document = motion_document()
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 1.0, reverse=True)

        self.assertEqual(plan.operation, "paste_motion_reversed")
        self.assertEqual(
            [(frame.time, frame.x) for frame in plan.frames],
            [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(plan.states[-1][1], [0.0, 10.0])
        self.assertEqual(plan.inserted_frame_count, 1)
        self.assertEqual(plan.inserted_state_count, 1)
        self.assertEqual(plan.current_time, 2.0)

        result = EditorController(document).execute(ApplyTimelineEditPlan(plan))
        self.assertTrue(result.changed)
        self.assertEqual(document.revision, 1)
        self.assertEqual(document.qpos_timeline.times(), [0.0, 1.0, 2.0])

    def test_forward_paste_accepts_a_closed_loop_seam(self):
        document = motion_document(
            (0.0, 0.5, 1.0),
            values=(0.0, 1.0, 0.0),
        )
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 1.0)

        self.assertEqual(
            [(frame.time, frame.x) for frame in plan.frames],
            [
                (0.0, 0.0),
                (0.5, 1.0),
                (1.0, 0.0),
                (1.5, 1.0),
                (2.0, 0.0),
            ],
        )
        self.assertEqual(plan.current_time, 2.0)

    def test_forward_paste_rejects_a_nonclosed_shared_seam(self):
        document = motion_document()
        original_frames = [
            frame.to_dict() for frame in document.trajectory.frames
        ]
        original_states = {
            time: qpos.copy()
            for time, qpos in document.qpos_timeline.states.items()
        }
        clip = capture_motion_clip(document, 0.0, 1.0)

        with self.assertRaisesRegex(TimelineEditError, "t=1.00 s"):
            plan_paste_motion(document, clip, 1.0)

        self.assertEqual(
            [frame.to_dict() for frame in document.trajectory.frames],
            original_frames,
        )
        for time, qpos in original_states.items():
            np.testing.assert_allclose(
                document.qpos_timeline.states[time], qpos
            )

    def test_paste_rejects_existing_interpolated_motion_without_mutation(self):
        document = motion_document((0.0, 1.0, 2.0))
        clip = capture_motion_clip(document, 0.0, 1.0)
        original_frames = [
            frame.to_dict() for frame in document.trajectory.frames
        ]
        original_states = {
            time: qpos.copy()
            for time, qpos in document.qpos_timeline.states.items()
        }

        with self.assertRaisesRegex(TimelineEditError, "overlaps existing"):
            plan_paste_motion(document, clip, 0.5, reverse=True)

        self.assertEqual(
            [frame.to_dict() for frame in document.trajectory.frames],
            original_frames,
        )
        self.assertEqual(
            set(document.qpos_timeline.states),
            set(original_states),
        )
        for time, qpos in original_states.items():
            np.testing.assert_allclose(
                document.qpos_timeline.states[time], qpos
            )

    def test_ping_pong_repeat_alternates_reversed_and_forward_copies(self):
        document = motion_document()
        document.timeline_duration = 1.0
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_repeat_motion(
            document,
            clip,
            1.0,
            3,
            ping_pong=True,
        )

        self.assertEqual(
            [(frame.time, frame.x) for frame in plan.frames],
            [
                (0.0, 0.0),
                (1.0, 1.0),
                (2.0, 0.0),
                (3.0, 1.0),
                (4.0, 0.0),
            ],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(plan.timeline_duration, 4.0)
        self.assertEqual(plan.current_time, 4.0)
        self.assertEqual(plan.operation, "repeat_motion_ping_pong")

    def test_forward_repeat_rejects_an_open_seam_without_mutation(self):
        document = motion_document()
        clip = capture_motion_clip(document, 0.0, 1.0)

        with self.assertRaisesRegex(TimelineEditError, "conflicts"):
            plan_repeat_motion(document, clip, 1.0, 2)

        self.assertEqual(
            [(frame.time, frame.x) for frame in document.trajectory.frames],
            [(0.0, 0.0), (1.0, 1.0)],
        )
        self.assertEqual(document.qpos_timeline.times(), [0.0, 1.0])

    def test_model_width_count_and_maximum_time_are_preflighted(self):
        document = motion_document()
        clip = capture_motion_clip(document, 0.0, 1.0)

        document.model_key = "other"
        with self.assertRaisesRegex(TimelineEditError, "different robot"):
            plan_paste_motion(document, clip, 2.0)
        document.model_key = "test"

        bad_width = MotionClip(
            model_key="test",
            duration=clip.duration,
            frames=clip.frames,
            states=((0.0, np.zeros(3)), (1.0, np.ones(3))),
            qpos_width=3,
        )
        with self.assertRaisesRegex(TimelineEditError, "Joint Angles"):
            plan_paste_motion(document, bad_width, 2.0)

        with self.assertRaisesRegex(TimelineEditError, "positive integer"):
            plan_repeat_motion(document, clip, 2.0, 0)

        with self.assertRaisesRegex(TimelineEditError, "3.00 s limit"):
            plan_paste_motion(document, clip, 2.5, maximum_time=3.0)

    def test_stateful_clip_requires_a_destination_qpos_timeline(self):
        source = motion_document()
        clip = capture_motion_clip(source, 0.0, 1.0)
        destination = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=None,
        )
        destination.trajectory.add_frame(target(0.0, 0.0))
        destination.trajectory.add_frame(target(1.0, 1.0))
        original_frames = [
            frame.to_dict() for frame in destination.trajectory.frames
        ]

        with self.assertRaisesRegex(TimelineEditError, "robot-pose timeline"):
            plan_paste_motion(destination, clip, 1.0, reverse=True)

        self.assertEqual(
            [frame.to_dict() for frame in destination.trajectory.frames],
            original_frames,
        )

    def test_euler_equivalent_forward_seam_coalesces(self):
        document = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=None,
        )
        document.trajectory.add_frame(TargetFrame(
            time=0.0,
            phase="motion",
            frame_name="tool",
            x=0.25,
            y=-0.5,
            z=1.0,
            yaw=0.0,
        ))
        document.trajectory.add_frame(TargetFrame(
            time=1.0,
            phase="motion",
            frame_name="tool",
            x=0.25,
            y=-0.5,
            z=1.0,
            yaw=2.0 * np.pi,
        ))
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 1.0)

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(plan.inserted_frame_count, 1)

    def test_free_joint_quaternion_sign_equivalent_seam_coalesces(self):
        timeline = FakeStateTimeline()
        timeline.robot_model = SimpleNamespace(
            mj_model=SimpleNamespace(nq=7),
            free_joints_by_body={
                "root": SimpleNamespace(qpos_address=0),
            },
        )
        timeline.set_state(
            0.0,
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        timeline.set_state(
            1.0,
            np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        )
        document = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=timeline,
        )
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 1.0)

        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(plan.inserted_state_count, 1)

    def test_timestamp_overflow_raises_timeline_edit_error(self):
        document = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=None,
        )
        clip = MotionClip(
            model_key="test",
            duration=1e308,
            frames=(
                TargetFrame(time=0.0, frame_name="tool"),
                TargetFrame(time=1e308, frame_name="tool"),
            ),
            states=(),
            qpos_width=None,
        )

        with self.assertRaisesRegex(TimelineEditError, "finite"):
            plan_paste_motion(document, clip, 1e308)

    def test_target_only_clip_can_be_pasted(self):
        document = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=None,
        )
        document.trajectory.add_frame(target(0.0, 0.0))
        document.trajectory.add_frame(target(1.0, 1.0))
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 2.0, reverse=True)

        self.assertEqual(clip.states, ())
        self.assertIsNone(clip.qpos_width)
        self.assertEqual(
            [(frame.time, frame.x) for frame in plan.frames],
            [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)],
        )

    def test_qpos_only_clip_can_be_pasted(self):
        timeline = FakeStateTimeline(
            ((0.0, np.array([0.0, 10.0])), (1.0, np.array([1.0, 11.0])))
        )
        document = ProjectDocument(
            "test",
            timeline_duration=1.0,
            qpos_timeline=timeline,
        )
        clip = capture_motion_clip(document, 0.0, 1.0)

        plan = plan_paste_motion(document, clip, 1.0, reverse=True)

        self.assertEqual(clip.frames, ())
        self.assertEqual(clip.qpos_width, 2)
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(plan.states[-1][1], [0.0, 10.0])


if __name__ == "__main__":
    unittest.main()

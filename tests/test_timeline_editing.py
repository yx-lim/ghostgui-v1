"""Contracts for atomic logical-target and qpos timeline retiming."""

from __future__ import annotations

import unittest

import numpy as np

from application.editor_controller import EditorController
from application.project_document import ProjectDocument
from application.timeline_editing import (
    ApplyTimelineEditPlan,
    TimelineEditError,
    plan_insert_time,
    plan_move_time_range,
    plan_scale_time_range,
    plan_shift_entire_motion,
    snap_time,
    timeline_content_bounds,
)
from core.trajectory import TargetFrame


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
        if lower is None:
            return self.states[upper].copy()
        if upper is None:
            return self.states[lower].copy()
        fraction = (key - lower) / (upper - lower)
        return self.states[lower] + fraction * (
            self.states[upper] - self.states[lower]
        )


def target(time, x, frame_name="tool"):
    return TargetFrame(time=time, frame_name=frame_name, x=x)


def document_with_motion(times=(0.0, 2.0)):
    timeline = FakeStateTimeline(
        (time, np.array([time, time + 10.0])) for time in times
    )
    document = ProjectDocument(
        "test",
        timeline_duration=5.0,
        qpos_timeline=timeline,
    )
    for time in times:
        document.trajectory.add_frame(target(time, time))
    return document


class TimelineEditingTests(unittest.TestCase):
    def test_insert_time_holds_sampled_boundary_and_shifts_later_motion(self):
        document = document_with_motion()
        document.current_time = 1.0

        plan = plan_insert_time(document, 1.0, 1.0)

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(
            [frame.x for frame in plan.frames],
            [0.0, 1.0, 1.0, 2.0],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(plan.states[1][1], [1.0, 11.0])
        np.testing.assert_allclose(plan.states[2][1], [1.0, 11.0])
        self.assertEqual(plan.timeline_duration, 6.0)
        self.assertEqual(plan.current_time, 1.0)

        # Planning is a non-mutating preflight.
        self.assertEqual(
            [frame.time for frame in document.trajectory.frames],
            [0.0, 2.0],
        )
        self.assertEqual(document.qpos_timeline.times(), [0.0, 2.0])

    def test_apply_plan_updates_both_sources_through_one_command(self):
        document = document_with_motion()
        plan = plan_insert_time(document, 1.0, 1.0)

        result = EditorController(document).execute(
            ApplyTimelineEditPlan(plan)
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.operation, "insert_time")
        self.assertEqual(document.revision, 1)
        self.assertEqual(
            [frame.time for frame in document.trajectory.frames],
            [0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(document.qpos_timeline.times(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(document.timeline_duration, 6.0)

    def test_shift_entire_motion_translates_targets_and_qpos_together(self):
        document = document_with_motion()

        plan = plan_shift_entire_motion(document, 1.0)

        self.assertEqual([frame.time for frame in plan.frames], [1.0, 3.0])
        self.assertEqual([time for time, _qpos in plan.states], [1.0, 3.0])
        self.assertEqual(plan.timeline_duration, 6.0)
        self.assertEqual(plan.current_time, 1.0)

    def test_shift_rejects_a_negative_result_before_mutation(self):
        document = document_with_motion()

        with self.assertRaisesRegex(TimelineEditError, "before 0 s"):
            plan_shift_entire_motion(document, -0.1)

        self.assertEqual(
            [frame.time for frame in document.trajectory.frames],
            [0.0, 2.0],
        )
        self.assertEqual(document.qpos_timeline.times(), [0.0, 2.0])

    def test_move_adjacent_range_and_leave_other_keyframes_unchanged(self):
        document = document_with_motion((0.0, 1.0, 2.0))

        plan = plan_move_time_range(document, 1.0, 2.0, 2.0)

        self.assertEqual([frame.time for frame in plan.frames], [0.0, 2.0, 3.0])
        self.assertEqual([time for time, _qpos in plan.states], [0.0, 2.0, 3.0])
        self.assertEqual(plan.current_time, 2.0)
        self.assertEqual(plan.timeline_duration, 5.0)

    def test_move_rejects_destination_conflicts_before_mutation(self):
        document = document_with_motion((0.0, 1.0, 2.0, 3.0))
        original_frames = [frame.to_dict() for frame in document.trajectory.frames]
        original_states = document.qpos_timeline.times()

        with self.assertRaisesRegex(TimelineEditError, "t=3.00 s"):
            plan_move_time_range(document, 1.0, 2.0, 2.0)

        self.assertEqual(
            [frame.to_dict() for frame in document.trajectory.frames],
            original_frames,
        )
        self.assertEqual(document.qpos_timeline.times(), original_states)

    def test_move_rejects_overlapping_ranges(self):
        document = document_with_motion((0.0, 1.0, 2.0))

        with self.assertRaisesRegex(TimelineEditError, "ranges overlap"):
            plan_move_time_range(document, 0.0, 2.0, 1.0)

    def test_scale_entire_motion_changes_authoritative_keyframe_times(self):
        document = document_with_motion((0.0, 1.0, 2.0))

        plan = plan_scale_time_range(document, 0.0, 2.0, 2.0)

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 0.5, 1.0],
        )
        np.testing.assert_allclose(plan.states[1][1], [1.0, 11.0])
        self.assertEqual(plan.current_time, 0.0)
        self.assertEqual(plan.operation, "scale_time_range")

    def test_scale_slower_expands_timeline_when_needed(self):
        document = document_with_motion((0.0, 1.0, 2.0))
        document.timeline_duration = 2.0

        plan = plan_scale_time_range(document, 0.0, 2.0, 0.5)

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.0, 2.0, 4.0],
        )
        self.assertEqual(plan.timeline_duration, 4.0)

    def test_scale_range_leaves_keyframes_outside_range_unchanged(self):
        document = document_with_motion((0.0, 1.0, 2.0, 3.0, 4.0))

        plan = plan_scale_time_range(document, 1.0, 3.0, 2.0)

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.0, 1.0, 1.5, 2.0, 4.0],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.0, 1.0, 1.5, 2.0, 4.0],
        )

    def test_scale_slower_rejects_overlap_with_later_keyframe(self):
        document = document_with_motion((0.0, 1.0, 2.0, 3.0))
        original_times = document.qpos_timeline.times()

        with self.assertRaisesRegex(TimelineEditError, "t=3.00 s"):
            plan_scale_time_range(document, 0.0, 2.0, 0.5)

        self.assertEqual(document.qpos_timeline.times(), original_times)

    def test_scale_snapping_rejects_collapsed_keyframes(self):
        document = document_with_motion((0.0, 0.1, 0.2))

        with self.assertRaisesRegex(TimelineEditError, "t=0.05 s"):
            plan_scale_time_range(
                document,
                0.0,
                0.2,
                3.0,
                snap_interval=0.05,
            )

    def test_scale_snapping_does_not_exclude_off_grid_range_boundaries(self):
        document = document_with_motion((0.03, 0.13))

        plan = plan_scale_time_range(
            document,
            0.03,
            0.13,
            2.0,
            snap_interval=0.05,
        )

        self.assertEqual(
            [frame.time for frame in plan.frames],
            [0.05, 0.1],
        )
        self.assertEqual(
            [time for time, _qpos in plan.states],
            [0.05, 0.1],
        )

    def test_scale_validates_speed_and_distinct_times(self):
        document = document_with_motion((0.0, 1.0))
        with self.assertRaisesRegex(TimelineEditError, "greater than zero"):
            plan_scale_time_range(document, 0.0, 1.0, 0.0)
        with self.assertRaisesRegex(TimelineEditError, "differ from 1.00"):
            plan_scale_time_range(document, 0.0, 1.0, 1.0)
        with self.assertRaisesRegex(TimelineEditError, "distinct Keyframe"):
            plan_scale_time_range(document, 0.0, 0.5, 2.0)
        with self.assertRaisesRegex(TimelineEditError, "does not change"):
            plan_scale_time_range(
                document,
                0.0,
                1.0,
                1.01,
                snap_interval=0.1,
            )

    def test_timeline_content_bounds_include_targets_and_qpos(self):
        document = document_with_motion((1.0, 3.0))
        document.trajectory.add_frame(target(0.5, 0.5, "camera"))

        self.assertEqual(timeline_content_bounds(document), (0.5, 3.0))

    def test_snap_time_uses_export_interval_grid(self):
        self.assertEqual(snap_time(1.0, 0.03), 0.99)
        self.assertEqual(snap_time(-0.08, 0.05), -0.1)

    def test_timeline_limit_is_preflighted(self):
        document = document_with_motion()
        document.timeline_duration = 119.5

        with self.assertRaisesRegex(TimelineEditError, "120.00 s limit"):
            plan_insert_time(document, 1.0, 1.0, maximum_time=120.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for bounded, timestamp-correlated v3.1 frame capture."""

from __future__ import annotations

import unittest

from application.ai.frame_capture import (
    EncodedFrame,
    FrameCaptureError,
    FrameSampler,
    FrameSamplingPlan,
    FrameSamplingRequest,
    capture_comparison_frames,
)
from application.ai.schemas import ImageVariant, MessageRole, ProviderMessage
from application.project_document import ProjectDocument


class _Timeline:
    def __init__(self, states):
        self.states = {float(time): list(qpos) for time, qpos in states}

    def times(self):
        return sorted(self.states)

    def sample_state(self, time_seconds):
        times = self.times()
        if not times:
            return None
        nearest = min(times, key=lambda value: abs(value - float(time_seconds)))
        return list(self.states[nearest])


class _RecordingRenderer:
    def __init__(self):
        self.calls = []

    def render_frame(self, qpos, *, time_seconds, variant):
        self.calls.append((list(qpos), time_seconds, variant))
        payload = f"{variant.value}@{time_seconds:.3f}:{qpos}".encode()
        return EncodedFrame(payload)


def _document(*, duration=5.0, states=()):
    return ProjectDocument(
        "g1",
        timeline_duration=duration,
        qpos_timeline=_Timeline(states),
    )


class FrameSamplerTests(unittest.TestCase):
    def test_default_plan_is_bounded_and_spans_the_motion(self):
        document = _document(
            states=((0.0, [0.0]), (1.0, [1.0]), (2.5, [2.5]), (5.0, [5.0]))
        )

        plan = FrameSampler().plan(document)

        self.assertGreaterEqual(len(plan.times_seconds), 4)
        self.assertLessEqual(len(plan.times_seconds), 8)
        self.assertEqual(plan.times_seconds[0], 0.0)
        self.assertEqual(plan.times_seconds[-1], 5.0)
        self.assertEqual(tuple(sorted(set(plan.times_seconds))), plan.times_seconds)

    def test_selected_interval_and_suspected_area_are_retained(self):
        document = _document(
            duration=5.0,
            states=tuple((value, [value]) for value in range(6)),
        )
        request = FrameSamplingRequest(
            selected_interval=(2.4, 3.2),
            suspected_times=(2.6, 2.8, 3.0),
        )

        plan = FrameSampler().plan(document, request)

        self.assertEqual(plan.times_seconds[0], 2.4)
        self.assertEqual(plan.times_seconds[-1], 3.2)
        self.assertTrue({2.6, 2.8, 3.0}.issubset(plan.times_seconds))
        self.assertTrue(all(2.4 <= value <= 3.2 for value in plan.times_seconds))

    def test_sampling_contract_rejects_out_of_bounds_requests(self):
        with self.assertRaisesRegex(ValueError, "4--8"):
            FrameSamplingRequest(minimum_frames=3)
        with self.assertRaisesRegex(FrameCaptureError, "exceeds"):
            FrameSampler().plan(
                _document(duration=2.0),
                FrameSamplingRequest(selected_interval=(1.0, 3.0)),
            )

    def test_many_suspected_times_still_preserve_interval_bounds(self):
        request = FrameSamplingRequest(
            selected_interval=(1.0, 4.0),
            suspected_times=tuple(1.1 + index * 0.2 for index in range(14)),
        )

        plan = FrameSampler().plan(_document(duration=5.0), request)

        self.assertEqual(len(plan.times_seconds), 8)
        self.assertEqual(plan.times_seconds[0], 1.0)
        self.assertEqual(plan.times_seconds[-1], 4.0)


class ComparisonCaptureTests(unittest.TestCase):
    def test_original_and_candidate_use_identical_explicit_timestamps(self):
        original = _document(
            states=((0.0, [0.0]), (1.0, [1.0]), (2.0, [2.0]), (3.0, [3.0]))
        )
        candidate = _document(
            states=((0.0, [10.0]), (1.0, [11.0]), (2.0, [12.0]), (3.0, [13.0]))
        )
        plan = FrameSamplingPlan((0.0, 1.0, 2.0, 3.0))
        renderer = _RecordingRenderer()

        frames = capture_comparison_frames(original, candidate, plan, renderer)
        message = ProviderMessage(MessageRole.USER, motion_frames=frames)

        self.assertEqual(len(message.motion_frames), 8)
        for pair_index in range(0, len(frames), 2):
            before, after = frames[pair_index : pair_index + 2]
            self.assertEqual(before.variant, ImageVariant.ORIGINAL)
            self.assertEqual(after.variant, ImageVariant.CANDIDATE)
            self.assertEqual(before.time_seconds, after.time_seconds)
            self.assertEqual(before.comparison_id, after.comparison_id)
            self.assertRegex(before.label, r"^frame_\d+$")

        self.assertEqual(original.qpos_timeline.states[0.0], [0.0])
        self.assertEqual(candidate.qpos_timeline.states[0.0], [10.0])
        self.assertEqual(
            [(call[1], call[2]) for call in renderer.calls[:2]],
            [(0.0, ImageVariant.ORIGINAL), (0.0, ImageVariant.CANDIDATE)],
        )

    def test_comparison_requires_compatible_sampleable_documents(self):
        plan = FrameSamplingPlan((0.0, 1.0, 2.0, 3.0))
        original = _document(states=((0.0, [0.0]),))
        other_model = ProjectDocument(
            "go2",
            qpos_timeline=_Timeline(((0.0, [0.0]),)),
        )
        with self.assertRaisesRegex(FrameCaptureError, "same robot model"):
            capture_comparison_frames(
                original,
                other_model,
                plan,
                _RecordingRenderer(),
            )


if __name__ == "__main__":
    unittest.main()

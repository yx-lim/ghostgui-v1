"""Tests for one-request compact trajectory planning."""

from __future__ import annotations

import json
import unittest

from application.ai.context import AIContext
from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.errors import ProviderCancelledError
from application.ai.providers import MockProvider, RequestCountingProvider
from application.ai.schemas import (
    ImageVariant,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderResponse,
    Usage,
)
from application.ai.trajectory_edit_spec import TrajectoryOperationType
from application.ai.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryPlannerError,
    TrajectoryPlannerLimits,
)
from application.project_document import ProjectDocument


def _response():
    return ProviderResponse(
        text=json.dumps({
            "mode": "edit",
            "summary": "Raise the robot by five centimetres.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 8.7,
                    "translation_m": [0.0, 0.0, 0.05],
                }),
            }],
        }),
        usage=Usage(120, 45),
    )


def _frame():
    return MotionFrameImage(
        data=b"small-frame",
        mime_type="image/png",
        time_seconds=1.25,
        variant=ImageVariant.ORIGINAL,
        comparison_id="frame_1",
        label="frame_1",
    )


class _Token:
    cancellation_requested = True


class TrajectoryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_contains_context_frames_schema_and_no_tools(self):
        delegate = MockProvider([_response()])
        provider = RequestCountingProvider(delegate)
        session = AIEditSession(ProjectDocument("g1"))

        result = await TrajectoryPlanner(provider).plan(
            "Move the entire robot 5 cm higher.",
            model="mock",
            context=AIContext({"robot": {"model_key": "g1"}}),
            session=session,
            motion_frames=(_frame(),),
        )

        self.assertEqual(provider.counter.counts.total, 1)
        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(result.usage, Usage(120, 45))
        self.assertEqual(
            result.spec.operations[0].operation_type,
            TrajectoryOperationType.ROOT_OFFSET,
        )
        request = delegate.requests[0]
        self.assertEqual(request.tools, ())
        self.assertIsNotNone(request.response_schema)
        self.assertEqual(request.messages[-1].motion_frames, (_frame(),))
        self.assertIn("operation_argument_contracts", request.messages[-1].text)
        self.assertNotIn("qpos_values", request.messages[-1].text)
        self.assertEqual(session.state, AIEditSessionState.READY)
        delegate.assert_exhausted()

    async def test_malformed_response_receives_exactly_one_successful_repair(self):
        delegate = MockProvider([
            ProviderResponse(text="not json", usage=Usage(10, 2)),
            _response(),
        ])
        provider = RequestCountingProvider(delegate)

        result = await TrajectoryPlanner(provider).plan(
            "Raise the robot.",
            model="mock",
            context={},
            session=AIEditSession(ProjectDocument("g1")),
        )

        self.assertEqual(provider.counter.counts.total, 2)
        self.assertEqual(result.provider_requests, 2)
        self.assertEqual(result.usage, Usage(130, 47))
        repair_text = delegate.requests[1].messages[-1].text
        self.assertIn("parser_error", repair_text)
        self.assertIn("original_instruction", repair_text)
        delegate.assert_exhausted()

    async def test_invalid_repair_stops_after_second_request(self):
        delegate = MockProvider([
            ProviderResponse(text="not json"),
            ProviderResponse(text="still not json"),
        ])
        provider = RequestCountingProvider(delegate)

        with self.assertRaisesRegex(TrajectoryPlannerError, "after one repair"):
            await TrajectoryPlanner(provider).plan(
                "Raise the robot.",
                model="mock",
                context={},
                session=AIEditSession(ProjectDocument("g1")),
            )

        self.assertEqual(provider.counter.counts.total, 2)
        delegate.assert_exhausted()

    async def test_pre_cancelled_request_consumes_no_provider_call(self):
        delegate = MockProvider([_response()])
        provider = RequestCountingProvider(delegate)

        with self.assertRaises(ProviderCancelledError):
            await TrajectoryPlanner(provider).plan(
                "Raise the robot.",
                model="mock",
                context={},
                session=AIEditSession(ProjectDocument("g1")),
                cancellation_token=_Token(),
            )

        self.assertEqual(provider.counter.counts.total, 0)

    async def test_text_only_provider_accepts_numerical_context_without_frames(self):
        provider = MockProvider(
            [_response()],
            capabilities=ProviderCapabilities(
                supports_tools=False,
                supports_vision=False,
                supports_structured_output=True,
            ),
        )

        result = await TrajectoryPlanner(provider).plan(
            "Raise the robot.",
            model="mock",
            context={"current_state": {"available": True}},
            session=AIEditSession(ProjectDocument("g1")),
        )

        self.assertEqual(result.provider_requests, 1)

    def test_workflow_limits_are_locally_bounded(self):
        with self.assertRaisesRegex(ValueError, "timeout"):
            TrajectoryPlannerLimits(request_timeout_seconds=181.0)
        with self.assertRaisesRegex(ValueError, "image"):
            TrajectoryPlannerLimits(max_images=9)

    async def test_low_randomness_is_used_only_when_provider_supports_it(self):
        capabilities = ProviderCapabilities(
            supports_tools=False,
            supports_vision=False,
            supports_structured_output=True,
            supports_temperature=True,
        )
        provider = MockProvider([_response()], capabilities=capabilities)

        await TrajectoryPlanner(provider).plan(
            "Raise the robot.",
            model="mock",
            context={},
            session=AIEditSession(ProjectDocument("g1")),
        )

        self.assertEqual(provider.requests[0].temperature, 0.0)


if __name__ == "__main__":
    unittest.main()

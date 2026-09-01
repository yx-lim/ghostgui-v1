"""Tests for the read-only v3.1 structured visual critic."""

from __future__ import annotations

import asyncio
import json
import unittest

from application.ai.errors import ProviderCancelledError, ProviderCapabilityError
from application.ai.providers import MockProvider, MockStep
from application.ai.schemas import (
    ImageVariant,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderResponse,
    StopReason,
    ToolCall,
    Usage,
)
from application.ai.visual_critique import (
    VISUAL_CRITIQUE_RESPONSE_SCHEMA,
    VisualCritic,
    VisualCritiqueError,
    parse_visual_critique,
)


def _frames():
    return tuple(
        MotionFrameImage(
            data=f"png-{index}".encode(),
            mime_type="image/png",
            time_seconds=float(index),
            variant=ImageVariant.ORIGINAL,
            comparison_id=f"frame_{index + 1}",
            label=f"frame_{index + 1}",
        )
        for index in range(4)
    )


def _response_payload():
    return {
        "summary": "The landing is visually abrupt.",
        "observations": [
            {
                "time_seconds": 2.1,
                "body_part": "right foot",
                "issue": "The foot appears to slide.",
                "severity": 0.7,
            },
            {
                "time_seconds": None,
                "body_part": None,
                "issue": "Arm movement appears asymmetric.",
                "severity": None,
            },
        ],
    }


class _Token:
    def __init__(self, cancelled=False):
        self.cancellation_requested = cancelled


class VisualCritiqueParsingTests(unittest.TestCase):
    def test_parses_small_timestamped_observation_schema(self):
        critique = parse_visual_critique(json.dumps(_response_payload()))

        self.assertEqual(critique.summary, "The landing is visually abrupt.")
        self.assertEqual(critique.observations[0].time_seconds, 2.1)
        self.assertEqual(critique.observations[0].body_part, "right foot")
        self.assertIsNone(critique.observations[1].time_seconds)

    def test_rejects_malformed_extra_and_out_of_range_values(self):
        cases = (
            "not json",
            json.dumps({"summary": "x", "observations": [], "plan": []}),
            json.dumps({
                "summary": "x",
                "observations": [{
                    "time_seconds": 1.0,
                    "body_part": None,
                    "issue": "x",
                    "severity": 1.5,
                }],
            }),
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaises(VisualCritiqueError):
                parse_visual_critique(text)


class VisualCriticTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_images_context_and_schema_without_edit_tools(self):
        response = ProviderResponse(
            text=json.dumps(_response_payload()),
            usage=Usage(120, 30),
        )
        provider = MockProvider([response])

        result = await VisualCritic(provider).run(
            "What is wrong with this motion?",
            model="mock-vision",
            motion_context={"motion": {"duration_seconds": 3.0}},
            motion_frames=_frames(),
        )

        self.assertEqual(result.critique.observations[0].time_seconds, 2.1)
        self.assertEqual(result.usage, Usage(120, 30))
        request = provider.requests[0]
        self.assertEqual(request.tools, ())
        self.assertEqual(request.response_schema, VISUAL_CRITIQUE_RESPONSE_SCHEMA)
        self.assertEqual(request.messages[-1].motion_frames, _frames())
        self.assertIn("approximate motion time", request.messages[0].text)
        self.assertNotIn("qpos", request.messages[-1].text)

    async def test_rejects_non_visual_provider_before_consuming_request(self):
        provider = MockProvider(
            [ProviderResponse(text="unused")],
            capabilities=ProviderCapabilities(
                supports_tools=True,
                supports_vision=False,
                supports_structured_output=True,
            ),
        )

        with self.assertRaises(ProviderCapabilityError):
            await VisualCritic(provider).run(
                "Critique",
                model="text-only",
                motion_context={},
                motion_frames=_frames(),
            )
        self.assertEqual(provider.remaining_steps, 1)

    async def test_rejects_tool_calls_and_malformed_structured_output(self):
        tool_provider = MockProvider([ProviderResponse(
            tool_calls=(ToolCall("edit-1", "set_joint_angle", {}),),
            stop_reason=StopReason.TOOL_CALLS,
        )])
        with self.assertRaisesRegex(VisualCritiqueError, "cannot execute"):
            await VisualCritic(tool_provider).run(
                "Critique",
                model="mock",
                motion_context={},
                motion_frames=_frames(),
            )

        malformed_provider = MockProvider([ProviderResponse(text="not json")])
        with self.assertRaisesRegex(VisualCritiqueError, "malformed"):
            await VisualCritic(malformed_provider).run(
                "Critique",
                model="mock",
                motion_context={},
                motion_frames=_frames(),
            )

    async def test_cancellation_and_timeout_are_bounded(self):
        cancelled_provider = MockProvider([ProviderResponse(text="unused")])
        with self.assertRaises(ProviderCancelledError):
            await VisualCritic(cancelled_provider).run(
                "Critique",
                model="mock",
                motion_context={},
                motion_frames=_frames(),
                cancellation_token=_Token(cancelled=True),
            )

        slow_provider = MockProvider([
            MockStep(
                response=ProviderResponse(text=json.dumps(_response_payload())),
                delay_seconds=0.05,
            )
        ])
        with self.assertRaisesRegex(VisualCritiqueError, "timed out"):
            await VisualCritic(
                slow_provider,
                request_timeout_seconds=0.001,
            ).run(
                "Critique",
                model="mock",
                motion_context={},
                motion_frames=_frames(),
            )


if __name__ == "__main__":
    unittest.main()

"""Tests for the deterministic provider and provider capability boundary."""

from __future__ import annotations

import asyncio
import unittest

from application.ai.errors import (
    ProviderCancelledError,
    ProviderCapabilityError,
    ProviderError,
)
from application.ai.providers import LLMProvider, MockProvider, MockStep
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    ToolCall,
    ToolDefinition,
)


class _Token:
    def __init__(self, cancelled=False):
        self.cancellation_requested = cancelled


def _request(**changes):
    values = {
        "model": "mock-model",
        "messages": (ProviderMessage(MessageRole.USER, text="Move the right hand"),),
    }
    values.update(changes)
    return ProviderRequest(**values)


class MockProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_replays_sequential_responses_and_records_requests(self):
        first = ProviderResponse(
            tool_calls=(
                ToolCall(
                    "call-1",
                    "move_end_effector",
                    {"target": "right_hand", "delta_z": 0.1},
                ),
            ),
            stop_reason=StopReason.TOOL_CALLS,
        )
        second = ProviderResponse(text="The hand was moved.")
        provider = MockProvider([first, second])

        self.assertIsInstance(provider, LLMProvider)
        self.assertIs(await provider.generate(_request()), first)
        self.assertIs(await provider.generate(_request()), second)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.remaining_steps, 0)
        provider.assert_exhausted()

    async def test_malformed_tool_arguments_are_preserved_for_registry_validation(self):
        response = ProviderResponse(
            tool_calls=(ToolCall("bad-1", "move_end_effector", {"delta_z": "high"}),),
            stop_reason=StopReason.TOOL_CALLS,
        )
        provider = MockProvider([response])

        result = await provider.generate(_request())

        self.assertEqual(result.tool_calls[0].arguments["delta_z"], "high")

    async def test_visual_critique_mock_records_timestamped_before_after_pair(self):
        frames = tuple(
            MotionFrameImage(
                data=variant.value.encode("ascii"),
                mime_type="image/png",
                time_seconds=2.8,
                variant=variant,
                comparison_id="pair-3",
            )
            for variant in (ImageVariant.ORIGINAL, ImageVariant.CANDIDATE)
        )
        request = _request(
            messages=(ProviderMessage(MessageRole.USER, motion_frames=frames),)
        )
        provider = MockProvider(
            [ProviderResponse(text="At about 2.8 s, the torso should be more upright.")]
        )

        response = await provider.generate(request)

        self.assertIn("2.8 s", response.text)
        recorded_frames = provider.requests[0].messages[0].motion_frames
        self.assertEqual(
            {frame.time_seconds for frame in recorded_frames},
            {2.8},
        )

    async def test_scripted_failure_is_normalized_and_exhaustion_is_explicit(self):
        provider = MockProvider([MockStep(error=RuntimeError("offline"))])
        with self.assertRaisesRegex(ProviderError, "offline"):
            await provider.generate(_request())
        with self.assertRaisesRegex(ProviderError, "exhausted"):
            await provider.generate(_request())

    async def test_cancellation_does_not_consume_a_step_before_work_starts(self):
        response = ProviderResponse(text="unused")
        provider = MockProvider([response])

        with self.assertRaises(ProviderCancelledError):
            await provider.generate(_request(), _Token(cancelled=True))

        self.assertEqual(provider.remaining_steps, 1)
        self.assertEqual(provider.requests, ())

    async def test_cancellation_is_observed_during_delayed_response(self):
        token = _Token()
        provider = MockProvider(
            [MockStep(response=ProviderResponse(text="late"), delay_seconds=0.05)]
        )
        task = asyncio.create_task(provider.generate(_request(), token))
        await asyncio.sleep(0.015)
        token.cancellation_requested = True

        with self.assertRaises(ProviderCancelledError):
            await task

    async def test_rejects_request_features_missing_from_capabilities(self):
        capabilities = ProviderCapabilities(
            supports_tools=False,
            supports_vision=False,
            supports_structured_output=False,
        )
        tool = ToolDefinition(
            "inspect_motion",
            "Inspect motion.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        cases = (
            _request(tools=(tool,)),
            _request(response_schema={"type": "object"}),
            _request(
                messages=(
                    ProviderMessage(
                        MessageRole.USER,
                        motion_frames=(
                            MotionFrameImage(
                                data=b"png",
                                mime_type="image/png",
                                time_seconds=1.0,
                                variant=ImageVariant.CANDIDATE,
                                comparison_id="frame-1",
                            ),
                        ),
                    ),
                )
            ),
        )
        for request in cases:
            with self.subTest(request=request):
                provider = MockProvider([ProviderResponse(text="unused")], capabilities=capabilities)
                with self.assertRaises(ProviderCapabilityError):
                    await provider.generate(request)
                self.assertEqual(provider.remaining_steps, 1)

    async def test_rejects_undeclared_parallel_tool_calls(self):
        capabilities = ProviderCapabilities(
            supports_tools=True,
            supports_vision=False,
            supports_parallel_tool_calls=False,
        )
        response = ProviderResponse(
            tool_calls=(
                ToolCall("one", "inspect_motion", {}),
                ToolCall("two", "validate_motion", {}),
            ),
            stop_reason=StopReason.TOOL_CALLS,
        )
        provider = MockProvider([response], capabilities=capabilities)

        with self.assertRaisesRegex(ProviderCapabilityError, "parallel"):
            await provider.generate(_request())


if __name__ == "__main__":
    unittest.main()

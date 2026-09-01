"""Tests for provider-neutral AI data contracts."""

from __future__ import annotations

import unittest

from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionEntityRef,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderMessage,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from application.ai.limits import MAX_MOTION_FRAME_BYTES


class AISchemaTests(unittest.TestCase):
    def test_motion_frame_rejects_oversized_upload(self):
        with self.assertRaisesRegex(ValueError, "upload-size"):
            MotionFrameImage(
                data=b"x" * (MAX_MOTION_FRAME_BYTES + 1),
                mime_type="image/png",
                time_seconds=0.0,
                variant=ImageVariant.ORIGINAL,
                comparison_id="frame-1",
            )

    def test_motion_entity_reference_is_opaque(self):
        reference = MotionEntityRef("keyframe-id:v1:abc123")
        self.assertEqual(reference.identifier, "keyframe-id:v1:abc123")
        self.assertFalse(hasattr(reference, "time_seconds"))
        self.assertFalse(hasattr(reference, "frame_name"))

    def test_motion_frame_requires_explicit_valid_time(self):
        frame = MotionFrameImage(
            data=b"png",
            mime_type="image/png",
            time_seconds=2.8,
            variant=ImageVariant.CANDIDATE,
            comparison_id="comparison-3",
        )
        message = ProviderMessage(MessageRole.USER, motion_frames=(frame,))
        self.assertEqual(message.motion_frames[0].time_seconds, 2.8)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            MotionFrameImage(
                data=b"png",
                mime_type="image/png",
                time_seconds=float("nan"),
                variant=ImageVariant.ORIGINAL,
                comparison_id="comparison-3",
            )

    def test_motion_frames_are_only_user_message_content(self):
        frame = MotionFrameImage(
            data=b"png",
            mime_type="image/png",
            time_seconds=1.0,
            variant=ImageVariant.ORIGINAL,
            comparison_id="pair",
        )
        with self.assertRaisesRegex(ValueError, "only valid on user"):
            ProviderMessage(MessageRole.ASSISTANT, motion_frames=(frame,))

    def test_before_after_comparison_requires_identical_times(self):
        original = MotionFrameImage(
            data=b"original",
            mime_type="image/png",
            time_seconds=2.8,
            variant=ImageVariant.ORIGINAL,
            comparison_id="pair-3",
        )
        candidate = MotionFrameImage(
            data=b"candidate",
            mime_type="image/png",
            time_seconds=2.81,
            variant=ImageVariant.CANDIDATE,
            comparison_id="pair-3",
        )
        with self.assertRaisesRegex(ValueError, "identical times"):
            ProviderMessage(
                MessageRole.USER,
                motion_frames=(original, candidate),
            )

    def test_provider_capabilities_reject_images_for_text_only_provider(self):
        with self.assertRaisesRegex(ValueError, "text-only"):
            ProviderCapabilities(
                supports_tools=True,
                supports_vision=False,
                max_images_per_request=1,
            )

    def test_tool_call_response_contract_is_consistent(self):
        call = ToolCall("call-1", "inspect_motion", {})
        response = ProviderResponse(
            tool_calls=(call,),
            stop_reason=StopReason.TOOL_CALLS,
        )
        self.assertEqual(response.tool_calls, (call,))
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            ProviderResponse(stop_reason=StopReason.TOOL_CALLS)


if __name__ == "__main__":
    unittest.main()

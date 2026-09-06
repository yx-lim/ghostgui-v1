"""Tests for one-shot visual planning and explicit visual verification."""

from __future__ import annotations

import json
import unittest

from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.errors import ProviderCancelledError, ProviderCapabilityError
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.providers import MockProvider, MockStep
from application.ai.schemas import (
    ImageVariant,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from application.ai.semantic_tools import SemanticToolContext, build_semantic_tool_registry
from application.ai.visual_refinement import (
    VISUAL_VERIFICATION_RESPONSE_SCHEMA,
    VisualMotionWorkflow,
    VisualMotionPlannerLimits,
    VisualRefinementError,
    VisualVerifier,
    parse_visual_motion_plan,
    parse_visual_verification,
    visual_motion_plan_response_schema,
)
from tests.test_ai_semantic_tools import FakeMotionService, _document


class _Token:
    def __init__(self, cancelled=False):
        self.cancellation_requested = cancelled


def _visual_plan_payload(*, operations=None):
    return {
        "observations": [{
            "time_seconds": 1.8,
            "body_part": "torso",
            "issue": "The torso pitches forward near landing.",
            "severity": 0.4,
        }],
        "operations": operations if operations is not None else [{
            "tool": "set_logical_frame_target",
            "arguments": json.dumps({
                "logical_frame": "torso",
                "time_seconds": 1.8,
                "position_m": [0.0, 0.0, 1.2],
                "orientation_rpy_rad": [0.0, 0.0, 0.0],
                "mode": "absolute",
            }),
        }],
    }


def _verification_payload():
    return {
        "summary": "The candidate is softer but still pitches forward.",
        "preferred": "candidate",
        "reasons": ["greater knee flexion", "slower pelvis descent"],
        "observations": [{
            "time_seconds": 1.8,
            "body_part": "torso",
            "issue": "The torso still pitches forward near landing.",
            "severity": 0.3,
        }],
    }


def _motion_frames():
    return tuple(
        MotionFrameImage(
            data=f"candidate-{index}".encode(),
            mime_type="image/png",
            time_seconds=time_seconds,
            variant=ImageVariant.CANDIDATE,
            comparison_id=f"frame_{index}",
            label=f"frame_{index}",
        )
        for index, time_seconds in enumerate((0.0, 1.0, 2.0, 3.0), start=1)
    )


def _comparison_frames():
    frames = []
    for index, time_seconds in enumerate((0.0, 1.0, 2.0, 3.0), start=1):
        for variant in (ImageVariant.ORIGINAL, ImageVariant.CANDIDATE):
            frames.append(MotionFrameImage(
                data=f"{variant.value}-{index}".encode(),
                mime_type="image/png",
                time_seconds=time_seconds,
                variant=variant,
                comparison_id=f"frame_{index}",
                label=f"frame_{index}",
            ))
    return tuple(frames)


def _semantic_setup():
    committed = _document()
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(session, metadata, motion_name="landing")
    tools = build_semantic_tool_registry(FakeMotionService())
    tools.execute("ensure_keyframe", {"time_seconds": 1.0}, context=context)
    return committed, session, context, tools


class VisualMotionPlanContractTests(unittest.TestCase):
    def test_schema_combines_observations_with_allowlisted_operations(self):
        _committed, _session, _context, tools = _semantic_setup()
        schema = visual_motion_plan_response_schema(tools)

        self.assertEqual(set(schema["properties"]), {"observations", "operations"})
        operation = schema["properties"]["operations"]["items"]
        names = operation["properties"]["tool"]["enum"]
        self.assertIn("set_logical_frame_target", names)
        self.assertNotIn("inspect_motion", names)
        self.assertNotIn("validate_motion", names)
        self.assertEqual(
            operation["properties"]["arguments"]["type"],
            "string",
        )

    def test_parser_returns_timed_observations_and_motion_edit_plan(self):
        observations, plan = parse_visual_motion_plan(
            json.dumps(_visual_plan_payload())
        )

        self.assertEqual(observations[0].time_seconds, 1.8)
        self.assertEqual(plan.operations[0].tool, "set_logical_frame_target")
        self.assertEqual(plan.operations[0].arguments["logical_frame"], "torso")

    def test_parser_rejects_prose_plan_and_non_string_arguments(self):
        payload = _visual_plan_payload()
        payload["plan"] = ["make the torso upright"]
        with self.assertRaisesRegex(VisualRefinementError, "invalid fields"):
            parse_visual_motion_plan(json.dumps(payload))

        payload = _visual_plan_payload()
        payload["operations"][0]["arguments"] = {"time_seconds": 1.8}
        with self.assertRaisesRegex(VisualRefinementError, "fields are invalid"):
            parse_visual_motion_plan(json.dumps(payload))


class VisualMotionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_plan_and_local_edit_use_one_provider_request(self):
        committed, session, context, tools = _semantic_setup()
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_visual_plan_payload()))
        ])
        edits_before = len(session.edits)

        result = await VisualMotionWorkflow(provider, tools).run(
            "Make the landing softer.",
            model="mock-vision",
            motion_context={"motion": {"working_copy": True}},
            motion_frames=_motion_frames(),
            semantic_context=context,
        )

        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(result.execution.changed_operations)
        self.assertGreater(len(session.edits), edits_before)
        self.assertEqual(session.state, AIEditSessionState.STAGED)
        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        request = provider.requests[0]
        self.assertEqual(request.tools, ())
        self.assertEqual(len(request.messages[-1].motion_frames), 4)
        self.assertEqual(
            set(request.response_schema["properties"]),
            {"observations", "operations"},
        )
        self.assertIn("never image indexes", request.messages[0].text)
        self.assertIn("Allowed semantic edit operations", request.messages[-1].text)

    async def test_empty_operation_plan_stops_after_one_request_without_editing(self):
        _committed, session, context, tools = _semantic_setup()
        edits_before = tuple(session.edits)
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_visual_plan_payload(operations=[])))
        ])

        result = await VisualMotionWorkflow(provider, tools).run(
            "Review the landing.",
            model="mock-vision",
            motion_context={},
            motion_frames=_motion_frames(),
            semantic_context=context,
        )

        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(tuple(session.edits), edits_before)
        self.assertFalse(result.execution.operations)

    async def test_failed_local_operation_never_starts_a_second_request(self):
        _committed, _session, context, tools = _semantic_setup()
        payload = _visual_plan_payload(operations=[{
            "tool": "set_logical_frame_target",
            "arguments": json.dumps({"logical_frame": "torso"}),
        }])
        provider = MockProvider([ProviderResponse(text=json.dumps(payload))])

        result = await VisualMotionWorkflow(provider, tools).run(
            "Make the landing softer.",
            model="mock-vision",
            motion_context={},
            motion_frames=_motion_frames(),
            semantic_context=context,
        )

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(len(result.execution.failed_operations), 1)

    async def test_capabilities_and_frame_count_are_checked_before_request(self):
        _committed, _session, context, tools = _semantic_setup()
        provider = MockProvider(
            [ProviderResponse(text="unused")],
            capabilities=ProviderCapabilities(
                supports_tools=True,
                supports_vision=False,
                supports_structured_output=True,
            ),
        )

        with self.assertRaises(ProviderCapabilityError):
            await VisualMotionWorkflow(provider, tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
            )
        self.assertEqual(provider.remaining_steps, 1)

        with self.assertRaisesRegex(VisualRefinementError, "4--8 motion frames"):
            await VisualMotionWorkflow(MockProvider([]), tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames()[:-1],
                semantic_context=context,
            )

        original = tuple(
            MotionFrameImage(
                data=frame.data,
                mime_type=frame.mime_type,
                time_seconds=frame.time_seconds,
                variant=ImageVariant.ORIGINAL,
                comparison_id=frame.comparison_id,
                label=frame.label,
            )
            for frame in _motion_frames()
        )
        with self.assertRaisesRegex(VisualRefinementError, "candidate frames"):
            await VisualMotionWorkflow(MockProvider([]), tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=original,
                semantic_context=context,
            )

    async def test_instruction_response_and_tool_call_outputs_are_bounded(self):
        _committed, session, context, tools = _semantic_setup()
        unused = MockProvider([ProviderResponse(text="unused")])
        with self.assertRaisesRegex(VisualRefinementError, "goal"):
            await VisualMotionWorkflow(unused, tools).run(
                "x" * 16_001,
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
            )
        self.assertEqual(unused.remaining_steps, 1)

        oversized = MockProvider([ProviderResponse(text="x" * 65_537)])
        with self.assertRaisesRegex(VisualRefinementError, "response"):
            await VisualMotionWorkflow(oversized, tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
            )
        self.assertEqual(session.state, AIEditSessionState.STAGED)

        tool_call = MockProvider([ProviderResponse(
            tool_calls=(ToolCall("bad", "ensure_keyframe", {"time_seconds": 1.0}),),
            stop_reason=StopReason.TOOL_CALLS,
        )])
        with self.assertRaisesRegex(VisualRefinementError, "structured plan"):
            await VisualMotionWorkflow(tool_call, tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
            )
        self.assertEqual(len(tool_call.requests), 1)

    async def test_cancellation_and_timeout_never_start_a_follow_up_request(self):
        _committed, session, context, tools = _semantic_setup()
        cancelled = MockProvider([ProviderResponse(text="unused")])
        with self.assertRaises(ProviderCancelledError):
            await VisualMotionWorkflow(cancelled, tools).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
                cancellation_token=_Token(cancelled=True),
            )
        self.assertEqual(cancelled.remaining_steps, 1)

        delayed = MockProvider([MockStep(
            response=ProviderResponse(text=json.dumps(_visual_plan_payload())),
            delay_seconds=0.05,
        )])
        with self.assertRaisesRegex(VisualRefinementError, "timed out"):
            await VisualMotionWorkflow(
                delayed,
                tools,
                limits=VisualMotionPlannerLimits(request_timeout_seconds=0.001),
            ).run(
                "Improve it.",
                model="mock",
                motion_context={},
                motion_frames=_motion_frames(),
                semantic_context=context,
            )
        self.assertEqual(len(delayed.requests), 1)
        self.assertEqual(session.state, AIEditSessionState.STAGED)


class VisualVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_verification_is_one_read_only_comparison_request(self):
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_verification_payload()))
        ])

        result = await VisualVerifier(provider).run(
            "Make the landing softer.",
            model="mock-vision",
            motion_context={"motion": {"working_copy": True}},
            comparison_frames=_comparison_frames(),
        )

        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(result.verification.preferred, ImageVariant.CANDIDATE)
        self.assertEqual(result.verification.observations[0].time_seconds, 1.8)
        request = provider.requests[0]
        self.assertEqual(request.tools, ())
        self.assertEqual(request.response_schema, VISUAL_VERIFICATION_RESPONSE_SCHEMA)
        self.assertEqual(len(request.messages[-1].motion_frames), 8)
        self.assertIn("read-only verification", request.messages[0].text)

    async def test_verification_requires_complete_identically_timed_pairs(self):
        provider = MockProvider([ProviderResponse(text="unused")])
        with self.assertRaisesRegex(VisualRefinementError, "image pairs"):
            await VisualVerifier(provider).run(
                "Verify it.",
                model="mock",
                motion_context={},
                comparison_frames=_comparison_frames()[:-1],
            )
        self.assertEqual(provider.remaining_steps, 1)

    def test_verification_parser_rejects_edit_fields(self):
        payload = _verification_payload()
        payload["operations"] = []
        with self.assertRaisesRegex(VisualRefinementError, "invalid fields"):
            parse_visual_verification(json.dumps(payload))

    async def test_verification_rejects_tool_calls(self):
        provider = MockProvider([ProviderResponse(
            tool_calls=(ToolCall("bad", "ensure_keyframe", {"time_seconds": 1.0}),),
            stop_reason=StopReason.TOOL_CALLS,
        )])
        with self.assertRaisesRegex(VisualRefinementError, "cannot execute"):
            await VisualVerifier(provider).run(
                "Verify it.",
                model="mock",
                motion_context={},
                comparison_frames=_comparison_frames(),
            )


if __name__ == "__main__":
    unittest.main()

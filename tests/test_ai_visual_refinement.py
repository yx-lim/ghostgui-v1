"""Tests for bounded visual Observe/Plan/Edit refinement."""

from __future__ import annotations

import json
import unittest

from application.ai.agent import AgentRunResult
from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.errors import ProviderCapabilityError
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.providers import MockProvider
from application.ai.schemas import (
    ImageVariant,
    MotionFrameImage,
    ProviderResponse,
    ProviderCapabilities,
    StopReason,
    ToolCall,
)
from application.ai.semantic_tools import SemanticToolContext, build_semantic_tool_registry
from application.ai.visual_refinement import (
    VISUAL_COMPARISON_RESPONSE_SCHEMA,
    VisualComparator,
    VisualRefinementAction,
    VisualRefinementError,
    VisualRefinementLimits,
    VisualRefinementProgress,
    VisualRefinementStep,
    VisualRefinementStepResult,
    parse_visual_comparison,
)
from tests.test_ai_semantic_tools import FakeMotionService, _document


def _comparison_payload(*, should_refine=True):
    return {
        "summary": "The candidate is softer but the torso still pitches forward.",
        "preferred": "candidate",
        "reasons": ["greater knee flexion", "slower pelvis descent"],
        "observations": [{
            "time_seconds": 1.8,
            "body_part": "torso",
            "issue": "The torso pitches forward near landing.",
            "severity": 0.4,
        }],
        "should_refine": should_refine,
        "plan": ["make the torso slightly more upright at 1.8 seconds"]
        if should_refine else [],
    }


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


class VisualComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_comparison_is_structured_read_only_and_timestamp_paired(self):
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_comparison_payload(should_refine=False)))
        ])

        result = await VisualComparator(provider).run(
            "Make the landing softer.",
            model="mock-vision",
            motion_context={"motion": {"working_copy": True}},
            comparison_frames=_comparison_frames(),
        )

        self.assertEqual(result.comparison.preferred, ImageVariant.CANDIDATE)
        self.assertEqual(result.comparison.observations[0].time_seconds, 1.8)
        request = provider.requests[0]
        self.assertEqual(request.tools, ())
        self.assertEqual(request.response_schema, VISUAL_COMPARISON_RESPONSE_SCHEMA)
        self.assertIn("never image", request.messages[0].text)

    async def test_comparison_requires_complete_identically_timed_pairs(self):
        provider = MockProvider([ProviderResponse(text="unused")])
        with self.assertRaisesRegex(VisualRefinementError, "image pairs"):
            await VisualComparator(provider).run(
                "Improve it",
                model="mock",
                motion_context={},
                comparison_frames=_comparison_frames()[:-1],
            )
        self.assertEqual(provider.remaining_steps, 1)


class VisualRefinementStepTests(unittest.IsolatedAsyncioTestCase):
    async def test_editing_capability_is_checked_before_visual_request(self):
        _committed, _session, context, tools = _semantic_setup()
        provider = MockProvider(
            [ProviderResponse(text="unused")],
            capabilities=ProviderCapabilities(
                supports_tools=False,
                supports_vision=True,
                supports_structured_output=True,
                max_images_per_request=16,
            ),
        )

        with self.assertRaises(ProviderCapabilityError):
            await VisualRefinementStep(provider, tools).run(
                "Improve it",
                model="mock",
                motion_context={},
                comparison_frames=_comparison_frames(),
                semantic_context=context,
            )
        self.assertEqual(provider.remaining_steps, 1)

    async def test_observe_plan_then_edit_uses_only_semantic_tools(self):
        committed, session, context, tools = _semantic_setup()
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_comparison_payload())),
            ProviderResponse(
                tool_calls=(ToolCall(
                    "upright-1",
                    "set_logical_frame_target",
                    {
                        "logical_frame": "torso",
                        "time_seconds": 1.8,
                        "position_m": [0.0, 0.0, 1.2],
                        "orientation_rpy_rad": [0.0, 0.0, 0.0],
                        "mode": "absolute",
                    },
                ),),
                stop_reason=StopReason.TOOL_CALLS,
            ),
            ProviderResponse(text="Made the torso more upright."),
        ])
        edits_before = len(session.edits)

        result = await VisualRefinementStep(provider, tools).run(
            "Make the landing softer.",
            model="mock",
            motion_context={"motion": {"working_copy": True}},
            comparison_frames=_comparison_frames(),
            semantic_context=context,
        )

        self.assertTrue(result.motion_changed)
        self.assertIsInstance(result.edit_result, AgentRunResult)
        self.assertGreater(len(session.edits), edits_before)
        self.assertEqual(session.state, AIEditSessionState.STAGED)
        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        self.assertEqual(provider.requests[0].tools, ())
        self.assertTrue(provider.requests[1].tools)
        edit_prompt = provider.requests[1].messages[-1].text
        self.assertIn("around t=1.800 s", edit_prompt)
        self.assertIn("semantic refinement plan", edit_prompt)
        self.assertNotIn("raw qpos trajectory", edit_prompt)

    async def test_assess_only_never_calls_edit_agent(self):
        _committed, session, context, tools = _semantic_setup()
        edits_before = tuple(session.edits)
        provider = MockProvider([
            ProviderResponse(text=json.dumps(_comparison_payload()))
        ])

        result = await VisualRefinementStep(provider, tools).run(
            "Make the landing softer.",
            model="mock",
            motion_context={},
            comparison_frames=_comparison_frames(),
            semantic_context=context,
            allow_edit=False,
        )

        self.assertFalse(result.motion_changed)
        self.assertIsNone(result.edit_result)
        self.assertEqual(tuple(session.edits), edits_before)
        self.assertEqual(len(provider.requests), 1)


class VisualRefinementContractTests(unittest.TestCase):
    def test_loop_allows_two_edits_then_requires_final_assessment(self):
        progress = VisualRefinementProgress(VisualRefinementLimits())
        changed = VisualRefinementStepResult(None, None, True)
        unchanged = VisualRefinementStepResult(None, None, False)

        self.assertEqual(progress.after_step(changed), VisualRefinementAction.REFINE)
        self.assertEqual(
            progress.after_step(changed),
            VisualRefinementAction.ASSESS_ONLY,
        )
        self.assertEqual(progress.after_step(unchanged), VisualRefinementAction.COMPLETE)
        self.assertEqual(progress.completed_edit_iterations, 2)
        with self.assertRaisesRegex(VisualRefinementError, "limit exceeded"):
            progress.after_step(changed)

    def test_parser_rejects_refinement_without_a_plan(self):
        payload = _comparison_payload()
        payload["plan"] = []
        with self.assertRaisesRegex(VisualRefinementError, "requires a plan"):
            parse_visual_comparison(json.dumps(payload))

    def test_refinement_bound_cannot_exceed_three(self):
        with self.assertRaisesRegex(ValueError, "1--3"):
            VisualRefinementLimits(max_edit_iterations=4)


if __name__ == "__main__":
    unittest.main()

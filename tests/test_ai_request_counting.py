"""Measured provider-request baseline for the pre-refactor AI workflows."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from application.ai.agent import GhostGUIAgent
from application.ai.edit_session import AIEditSession
from application.ai.errors import ProviderError
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.providers import (
    MockProvider,
    MockStep,
    ProviderRequestCounter,
    RequestCountingProvider,
)
from application.ai.schemas import (
    MessageRole,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from application.ai.semantic_tools import (
    SemanticToolContext,
    build_semantic_tool_registry,
)
from application.ai.visual_critique import VisualCritic
from application.ai.visual_refinement import (
    VisualRefinementAction,
    VisualRefinementLimits,
    VisualRefinementProgress,
    VisualRefinementStep,
)
from tests.test_ai_semantic_tools import FakeMotionService, _document
from tests.test_ai_visual_critique import _frames as critique_frames
from tests.test_ai_visual_critique import _response_payload as critique_payload
from tests.test_ai_visual_refinement import _comparison_frames, _comparison_payload


try:
    from gui.ai_assistant_controller import AIAssistantController
except ImportError:
    AIAssistantController = None


MEASURED_BASELINE = {
    "single semantic edit": 2,
    "four-operation generation": 2,
    "failed semantic tool plus recovery": 3,
    "critique-only": 1,
    "one visual refinement iteration": 4,
    "two visual refinement iterations": 7,
    "Test Connection": 1,
}


def _tool_response(*calls: ToolCall) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=tuple(calls),
        stop_reason=StopReason.TOOL_CALLS,
    )


def _semantic_scenario(responses):
    committed = _document()
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(session, metadata, motion_name="count baseline")
    tools = build_semantic_tool_registry(FakeMotionService())
    provider = RequestCountingProvider(MockProvider(responses))
    return provider, GhostGUIAgent(provider, tools), context


def _visual_scenario(responses):
    committed = _document()
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(session, metadata, motion_name="count baseline")
    tools = build_semantic_tool_registry(FakeMotionService())
    provider = RequestCountingProvider(MockProvider(responses))
    return provider, context, tools


def _logical_target_call(
    identifier: str,
    *,
    time_seconds: float,
    height: float = 1.2,
) -> ToolCall:
    return ToolCall(
        identifier,
        "set_logical_frame_target",
        {
            "logical_frame": "torso",
            "time_seconds": time_seconds,
            "position_m": [0.0, 0.0, height],
            "orientation_rpy_rad": [0.0, 0.0, 0.0],
            "mode": "absolute",
        },
    )


def _visual_responses(edit_iterations: int):
    responses = []
    for index in range(edit_iterations):
        responses.extend((
            ProviderResponse(text=json.dumps(_comparison_payload())),
            _tool_response(_logical_target_call(
                f"visual-edit-{index + 1}",
                time_seconds=1.8,
                height=1.2 + (0.1 * index),
            )),
            ProviderResponse(text=f"Visual edit {index + 1} complete."),
        ))
    responses.append(ProviderResponse(
        text=json.dumps(_comparison_payload(should_refine=False))
    ))
    return responses


async def _measure_visual_iterations(edit_iterations: int) -> int:
    provider, context, tools = _visual_scenario(
        _visual_responses(edit_iterations)
    )
    limits = VisualRefinementLimits(max_edit_iterations=edit_iterations)
    progress = VisualRefinementProgress(limits)
    step = VisualRefinementStep(provider, tools, limits=limits)

    for index in range(edit_iterations):
        result = await step.run(
            "Make the landing softer.",
            model="mock",
            motion_context={"iteration": index + 1},
            comparison_frames=_comparison_frames(),
            semantic_context=context,
        )
        expected = (
            VisualRefinementAction.REFINE
            if index + 1 < edit_iterations
            else VisualRefinementAction.ASSESS_ONLY
        )
        if progress.after_step(result) is not expected:
            raise AssertionError("visual refinement took an unexpected action")

    final = await step.run(
        "Make the landing softer.",
        model="mock",
        motion_context={"final_assessment": True},
        comparison_frames=_comparison_frames(),
        semantic_context=context,
        allow_edit=False,
    )
    if progress.after_step(final) is not VisualRefinementAction.COMPLETE:
        raise AssertionError("visual refinement did not complete after assessment")
    return provider.counter.counts.total


class RequestCountingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_success_failure_and_reset_at_normalized_boundary(self):
        delegate = MockProvider([
            ProviderResponse(text="OK"),
            MockStep(error=RuntimeError("offline")),
        ])
        counter = ProviderRequestCounter()
        provider = RequestCountingProvider(delegate, counter=counter)
        request = ProviderRequest(
            model="mock",
            messages=(ProviderMessage(MessageRole.USER, text="test"),),
        )

        await provider.generate(request)
        with self.assertRaises(ProviderError):
            await provider.generate(request)

        self.assertEqual(counter.counts.total, 2)
        self.assertEqual(counter.counts.succeeded, 1)
        self.assertEqual(counter.counts.failed, 1)
        self.assertEqual(counter.counts.in_flight, 0)
        counter.reset()
        self.assertEqual(counter.counts.total, 0)

    async def test_single_edit_and_parallel_four_operation_generation(self):
        provider, agent, context = _semantic_scenario([
            _tool_response(_logical_target_call("single", time_seconds=0.0)),
            ProviderResponse(text="Single edit complete."),
        ])
        result = await agent.run(
            "Make the torso upright.",
            model="mock",
            context=context,
        )

        self.assertEqual(len(result.tool_executions), 1)
        self.assertEqual(
            provider.counter.counts.total,
            MEASURED_BASELINE["single semantic edit"],
        )

        calls = tuple(
            ToolCall(
                f"operation-{index}",
                "ensure_keyframe",
                {"time_seconds": time_seconds},
            )
            for index, time_seconds in enumerate((0.4, 0.8, 1.2, 1.6), start=1)
        )
        provider, agent, context = _semantic_scenario([
            _tool_response(*calls),
            ProviderResponse(text="Four operations complete."),
        ])
        result = await agent.run(
            "Generate four Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(len(result.tool_executions), 4)
        self.assertEqual(
            provider.counter.counts.total,
            MEASURED_BASELINE["four-operation generation"],
        )

    async def test_failed_tool_recovery_and_critique_only(self):
        invalid = ToolCall(
            "invalid",
            "move_end_effector",
            {
                "end_effector": "right_hand",
                "time_seconds": 0.0,
                "delta_m": [0.0, "higher", 0.1],
            },
        )
        repaired = ToolCall(
            "repaired",
            "move_end_effector",
            {
                "end_effector": "right_hand",
                "time_seconds": 0.0,
                "delta_m": [0.0, 0.0, 0.1],
            },
        )
        provider, agent, context = _semantic_scenario([
            _tool_response(invalid),
            _tool_response(repaired),
            ProviderResponse(text="Recovered."),
        ])
        result = await agent.run(
            "Raise the hand.",
            model="mock",
            context=context,
        )

        self.assertEqual(
            [execution.succeeded for execution in result.tool_executions],
            [False, True],
        )
        self.assertEqual(
            provider.counter.counts.total,
            MEASURED_BASELINE["failed semantic tool plus recovery"],
        )

        provider = RequestCountingProvider(MockProvider([
            ProviderResponse(text=json.dumps(critique_payload()))
        ]))
        await VisualCritic(provider).run(
            "Critique this motion.",
            model="mock",
            motion_context={},
            motion_frames=critique_frames(),
        )
        self.assertEqual(
            provider.counter.counts.total,
            MEASURED_BASELINE["critique-only"],
        )

    async def test_one_and_two_visual_refinement_iterations(self):
        self.assertEqual(
            await _measure_visual_iterations(1),
            MEASURED_BASELINE["one visual refinement iteration"],
        )
        self.assertEqual(
            await _measure_visual_iterations(2),
            MEASURED_BASELINE["two visual refinement iterations"],
        )


@unittest.skipUnless(
    AIAssistantController is not None,
    "Motion Assistant dependencies unavailable",
)
class ConnectionRequestCountingTests(unittest.TestCase):
    def test_connection_probe_uses_one_normalized_provider_request(self):
        provider = RequestCountingProvider(
            MockProvider([ProviderResponse(text="OK")])
        )
        controller = AIAssistantController.__new__(AIAssistantController)
        controller._session_api_keys = {}

        with patch(
            "gui.ai_assistant_controller.GeminiProvider",
            return_value=provider,
        ):
            result = asyncio.run(controller._run_connection_test(
                "gemini",
                "mock",
                "ephemeral-test-key",
            ))

        self.assertEqual(result, "OK")
        self.assertEqual(
            provider.counter.counts.total,
            MEASURED_BASELINE["Test Connection"],
        )


if __name__ == "__main__":
    unittest.main()

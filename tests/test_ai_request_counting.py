"""Measured historical and current provider-request workflow contracts."""

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
    VisualMotionWorkflow,
    VisualVerifier,
)
from tests.test_ai_semantic_tools import FakeMotionService, _document
from tests.test_ai_visual_critique import _frames as critique_frames
from tests.test_ai_visual_critique import _response_payload as critique_payload
from tests.test_ai_visual_refinement import (
    _comparison_frames,
    _motion_frames,
    _verification_payload,
    _visual_plan_payload,
)


try:
    from gui.ai_assistant_controller import AIAssistantController
except ImportError:
    AIAssistantController = None


EXPECTED_REQUEST_COUNTS = {
    "single semantic edit": 2,
    "four-operation generation": 2,
    "failed semantic tool plus recovery": 3,
    "critique-only": 1,
    "visual refinement": 1,
    "visual refinement plus verification": 2,
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


async def _measure_visual_refinement_and_verification() -> tuple[int, int]:
    provider, context, tools = _visual_scenario([
        ProviderResponse(text=json.dumps(_visual_plan_payload())),
        ProviderResponse(text=json.dumps(_verification_payload())),
    ])
    result = await VisualMotionWorkflow(provider, tools).run(
        "Make the landing softer.",
        model="mock",
        motion_context={"motion": {"working_copy": True}},
        motion_frames=_motion_frames(),
        semantic_context=context,
    )
    if not result.execution.changed_operations:
        raise AssertionError("visual refinement did not stage its local operation")
    refinement_count = provider.counter.counts.total
    await VisualVerifier(provider).run(
        "Make the landing softer.",
        model="mock",
        motion_context={"motion": {"working_copy": True}},
        comparison_frames=_comparison_frames(),
    )
    return refinement_count, provider.counter.counts.total


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
            EXPECTED_REQUEST_COUNTS["single semantic edit"],
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
            EXPECTED_REQUEST_COUNTS["four-operation generation"],
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
            EXPECTED_REQUEST_COUNTS["failed semantic tool plus recovery"],
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
            EXPECTED_REQUEST_COUNTS["critique-only"],
        )

    async def test_visual_refinement_and_optional_verification(self):
        refinement, with_verification = (
            await _measure_visual_refinement_and_verification()
        )
        self.assertEqual(
            refinement,
            EXPECTED_REQUEST_COUNTS["visual refinement"],
        )
        self.assertEqual(
            with_verification,
            EXPECTED_REQUEST_COUNTS["visual refinement plus verification"],
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
            EXPECTED_REQUEST_COUNTS["Test Connection"],
        )


if __name__ == "__main__":
    unittest.main()

"""Mocked end-to-end tests for the bounded v3.0 agent flow."""

from __future__ import annotations

import unittest

from application.ai.agent import (
    AgentLimitError,
    AgentLimits,
    AgentTimeoutError,
    GhostGUIAgent,
)
from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.errors import ProviderError
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.providers import MockProvider, MockStep
from application.ai.schemas import (
    MessageRole,
    ProviderResponse,
    StopReason,
    ToolCall,
)
from application.ai.semantic_tools import (
    SemanticToolContext,
    build_semantic_tool_registry,
)
from application.editor_commands import UpdateKeyframe
from application.editor_controller import EditorController
from core.trajectory import TargetFrame
from tests.test_ai_semantic_tools import FakeMotionService, _document


def _tool_response(identifier, name, arguments):
    return ProviderResponse(
        tool_calls=(ToolCall(identifier, name, arguments),),
        stop_reason=StopReason.TOOL_CALLS,
    )


def _setup(responses, *, limits=None):
    committed = _document()
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(session, metadata, motion_name="agent test")
    tools = build_semantic_tool_registry(FakeMotionService())
    provider = MockProvider(responses)
    agent = GhostGUIAgent(provider, tools, limits=limits)
    return committed, store, session, context, provider, agent


class _Token:
    def __init__(self, cancelled=False):
        self.cancellation_requested = cancelled


class GhostGUIAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_default_turn_budget_can_finish_the_full_tool_budget(self):
        limits = AgentLimits()

        self.assertGreaterEqual(
            limits.max_provider_turns,
            limits.max_tool_calls + 1,
        )

    async def test_mock_tool_call_stages_validated_motion_without_committing(self):
        committed, _store, session, context, provider, agent = _setup([
            _tool_response(
                "call-1",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.7],
                    "mode": "absolute",
                },
            ),
            ProviderResponse(text="The pelvis was lowered."),
        ])

        result = await agent.run(
            "Lower the pelvis.",
            model="mock-model",
            context=context,
        )

        self.assertEqual(result.text, "The pelvis was lowered.")
        self.assertEqual(result.provider_turns, 2)
        self.assertEqual(result.validation, {"valid": True, "issues": []})
        self.assertTrue(result.tool_executions[0].succeeded)
        self.assertEqual(session.state, AIEditSessionState.STAGED)
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.7)
        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        second_messages = provider.requests[1].messages
        self.assertEqual(second_messages[-1].role, MessageRole.TOOL)
        self.assertFalse(second_messages[-1].tool_results[0].is_error)

    async def test_tool_error_is_returned_to_provider_then_corrected(self):
        _committed, _store, session, context, provider, agent = _setup([
            _tool_response(
                "bad",
                "move_end_effector",
                {
                    "end_effector": "right_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, "higher", 0.1],
                },
            ),
            _tool_response(
                "fixed",
                "move_end_effector",
                {
                    "end_effector": "right_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, 0.0, 0.1],
                },
            ),
            ProviderResponse(text="Corrected and moved the hand."),
        ])

        result = await agent.run(
            "Move the right hand higher.",
            model="mock-model",
            context=context,
        )

        self.assertEqual(
            [execution.succeeded for execution in result.tool_executions],
            [False, True],
        )
        error_result = provider.requests[1].messages[-1].tool_results[0]
        self.assertTrue(error_result.is_error)
        self.assertIn("must be number", error_result.output["error"])
        self.assertTrue(session.has_changes)

    async def test_failed_tool_without_edit_does_not_stage_session(self):
        _committed, _store, session, context, _provider, agent = _setup([
            _tool_response(
                "bad",
                "set_joint_angle",
                {"joint": "waist_pitch", "time_seconds": 0.0, "angle_rad": "far"},
            ),
            ProviderResponse(text="I could not apply that Joint Angle."),
        ])

        result = await agent.run(
            "Set the waist Joint Angle.",
            model="mock",
            context=context,
        )

        self.assertFalse(result.tool_executions[0].succeeded)
        self.assertEqual(session.state, AIEditSessionState.READY)
        self.assertFalse(session.has_changes)

    async def test_agent_enforces_provider_turn_bound(self):
        _committed, _store, session, context, _provider, agent = _setup(
            [
                _tool_response("one", "ensure_keyframe", {"time_seconds": 1.0}),
            ],
            limits=AgentLimits(max_provider_turns=1, max_tool_calls=3),
        )

        with self.assertRaisesRegex(AgentLimitError, "provider-turn"):
            await agent.run("Add a middle Keyframe.", model="mock", context=context)

        self.assertEqual(session.state, AIEditSessionState.STAGED)
        self.assertEqual(session.working_document.qpos_timeline.times(), [0.0, 1.0, 2.0])

    async def test_progressing_workflow_can_exceed_old_eight_turn_limit(self):
        responses = [
            _tool_response(
                f"keyframe-{index}",
                "ensure_keyframe",
                {"time_seconds": index / 10.0},
            )
            for index in range(1, 10)
        ]
        responses.append(ProviderResponse(text="Nine Keyframes staged."))
        _committed, _store, session, context, _provider, agent = _setup(responses)

        result = await agent.run(
            "Add nine intermediate Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(result.provider_turns, 10)
        self.assertEqual(len(result.tool_executions), 9)
        self.assertTrue(session.has_changes)

    async def test_repeated_failed_call_stops_as_no_progress(self):
        repeated = _tool_response(
            "bad-1",
            "move_end_effector",
            {
                "end_effector": "right_hand",
                "time_seconds": 0.0,
                "delta_m": [0.0, "higher", 0.1],
            },
        )
        repeated_again = _tool_response(
            "bad-2",
            "move_end_effector",
            {
                "end_effector": "right_hand",
                "time_seconds": 0.0,
                "delta_m": [0.0, "higher", 0.1],
            },
        )
        _committed, _store, _session, context, provider, agent = _setup(
            [repeated, repeated_again]
        )

        with self.assertRaisesRegex(AgentLimitError, "without progress"):
            await agent.run("Move the hand higher.", model="mock", context=context)

        self.assertEqual(len(provider.requests), 2)

    async def test_prompt_requests_parallel_calls_and_automatic_validation(self):
        _committed, _store, _session, context, provider, agent = _setup([
            ProviderResponse(text="No edit needed."),
        ])

        await agent.run("Inspect the pose.", model="mock", context=context)

        system_prompt = provider.requests[0].messages[0].text
        self.assertIn("parallel tool calls", system_prompt)
        self.assertIn("validates staged motion automatically", system_prompt)

    async def test_provider_failure_and_timeout_leave_session_usable(self):
        _committed, _store, session, context, _provider, agent = _setup([
            MockStep(error=RuntimeError("offline")),
        ])
        with self.assertRaisesRegex(ProviderError, "offline"):
            await agent.run("Move the hand.", model="mock", context=context)
        self.assertEqual(session.state, AIEditSessionState.READY)

        slow_provider = MockProvider([
            MockStep(response=ProviderResponse(text="late"), delay_seconds=0.05)
        ])
        slow_agent = GhostGUIAgent(
            slow_provider,
            agent.tools,
            limits=AgentLimits(request_timeout_seconds=0.01),
        )
        with self.assertRaises(AgentTimeoutError):
            await slow_agent.run("Move the hand.", model="mock", context=context)
        self.assertEqual(session.state, AIEditSessionState.READY)

    async def test_clarification_response_does_not_stage_or_validate(self):
        _committed, _store, session, context, _provider, agent = _setup([
            ProviderResponse(text="Which arm should I move?"),
        ])

        result = await agent.run("Move that arm.", model="mock", context=context)

        self.assertEqual(result.validation, None)
        self.assertEqual(session.state, AIEditSessionState.READY)
        self.assertFalse(session.has_changes)

    async def test_ai_manual_refine_accept_preserves_manual_work(self):
        committed, _store, session, context, provider, agent = _setup([
            _tool_response(
                "pelvis-ai",
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 0.0,
                    "position_m": [0.0, 0.0, 0.8],
                    "mode": "absolute",
                },
            ),
            ProviderResponse(text="Initial edit staged."),
            _tool_response(
                "hand-refine",
                "move_end_effector",
                {
                    "end_effector": "right_hand",
                    "time_seconds": 0.0,
                    "delta_m": [0.0, 0.0, 0.1],
                },
            ),
            ProviderResponse(text="Refinement staged."),
        ])
        await agent.run("Lower the pelvis.", model="mock", context=context)
        pelvis = next(
            frame for frame in session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 0.0
        )
        pelvis_reference = context.working_metadata.reference_for_keyframe(pelvis)
        pelvis_index = session.working_document.trajectory.frames.index(pelvis)
        session.apply_manual(
            UpdateKeyframe(
                pelvis_index,
                TargetFrame(time=0.0, frame_name="pelvis", z=0.76),
            ),
            affected_entities=(pelvis_reference,),
        )

        await agent.run(
            "Now raise the right hand slightly.",
            model="mock",
            context=context,
        )
        refine_context = provider.requests[2].messages[-1].text
        self.assertIn('"author":"user"', refine_context)
        session.accept(EditorController(committed))

        committed_pelvis = next(
            frame for frame in committed.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 0.0
        )
        committed_hand = next(
            frame for frame in committed.trajectory.frames
            if frame.frame_name == "right_hand" and frame.time == 0.0
        )
        self.assertEqual(committed_pelvis.z, 0.76)
        self.assertEqual(committed_hand.z, 1.1)
        self.assertEqual(committed.revision, 1)
        self.assertEqual(session.state, AIEditSessionState.ACCEPTED)

    async def test_pre_cancelled_run_never_calls_provider(self):
        _committed, _store, session, context, provider, agent = _setup([
            ProviderResponse(text="unused"),
        ])
        with self.assertRaisesRegex(Exception, "cancelled"):
            await agent.run(
                "Move the hand.",
                model="mock",
                context=context,
                cancellation_token=_Token(cancelled=True),
            )
        self.assertEqual(provider.requests, ())
        self.assertEqual(session.state, AIEditSessionState.READY)


if __name__ == "__main__":
    unittest.main()

"""Tests for one-request text planning and deterministic local execution."""

from __future__ import annotations

import asyncio
import json
import unittest

from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_plan import (
    MAX_PLANNED_OPERATIONS,
    MotionEditPlan,
    MotionPlanError,
    PlannedOperation,
    motion_edit_plan_response_schema,
    parse_motion_edit_plan,
)
from application.ai.motion_services import MotionValidationReport
from application.ai.plan_executor import PlanExecutionError, PlanExecutor
from application.ai.providers import MockProvider, RequestCountingProvider
from application.ai.schemas import ProviderResponse
from application.ai.semantic_tools import (
    SemanticToolContext,
    build_semantic_tool_registry,
)
from application.ai.text_planner import TextMotionWorkflow
from application.ai.tool_registry import ToolCategory, ToolSpec
from application.editor_commands import UpdateKeyframe
from core.trajectory import TargetFrame
from gui.ai_assistant_controller import AIAssistantController
from tests.test_ai_semantic_tools import FakeMotionService, _document


def _setup(responses=(), *, motion=None):
    committed = _document()
    store = InMemoryMotionMetadataStore()
    metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
    session = AIEditSession(committed, metadata_store=store)
    context = SemanticToolContext(session, metadata, motion_name="planning test")
    tools = build_semantic_tool_registry(motion or FakeMotionService())
    delegate = MockProvider(responses)
    provider = RequestCountingProvider(delegate)
    return committed, session, context, tools, delegate, provider


def _payload(operations=(), *, summary="Provider-authored plan summary"):
    return {
        "summary": summary,
        "needs_clarification": False,
        "clarification_question": "",
        "operations": [
            {
                "tool": operation["tool"],
                "arguments": json.dumps(
                    operation["arguments"],
                    separators=(",", ":"),
                ),
            }
            for operation in operations
        ],
    }


def _ensure_operation(time_seconds: float):
    return {
        "tool": "ensure_keyframe",
        "arguments": {"time_seconds": time_seconds},
    }


def _logical_operation(
    logical_frame="pelvis",
    *,
    time_seconds=0.0,
    height=0.7,
):
    return {
        "tool": "set_logical_frame_target",
        "arguments": {
            "logical_frame": logical_frame,
            "time_seconds": time_seconds,
            "position_m": [0.0, 0.0, height],
            "mode": "absolute",
        },
    }


def _plan(*operations):
    return MotionEditPlan(
        summary="Plan",
        needs_clarification=False,
        clarification_question=None,
        operations=tuple(
            PlannedOperation(value["tool"], value["arguments"])
            for value in operations
        ),
    )


class MotionPlanContractTests(unittest.TestCase):
    def test_schema_is_derived_only_from_mutating_allowlisted_tools(self):
        _committed, _session, _context, tools, _delegate, _provider = _setup()

        schema = motion_edit_plan_response_schema(tools)
        operation = schema["properties"]["operations"]["items"]
        names = set(operation["properties"]["tool"]["enum"])

        self.assertEqual(names, {
            "ensure_keyframe",
            "move_end_effector",
            "protect_keyframe",
            "retime_segment",
            "set_joint_angle",
            "set_joint_group_angles",
            "set_logical_frame_target",
        })
        self.assertNotIn("inspect_motion", names)
        self.assertNotIn("validate_motion", names)
        self.assertNotIn("run_code", names)
        self.assertEqual(
            schema["properties"]["operations"]["maxItems"],
            MAX_PLANNED_OPERATIONS,
        )
        self.assertIs(operation["additionalProperties"], False)
        arguments = operation["properties"]["arguments"]
        self.assertEqual(arguments["type"], "string")
        self.assertNotIn("properties", arguments)
        self.assertEqual(
            schema["properties"]["clarification_question"]["type"],
            "string",
        )

    def test_parser_normalizes_plan_and_enforces_clarification_contract(self):
        plan = parse_motion_edit_plan(json.dumps(_payload([
            _logical_operation(),
        ])))

        self.assertEqual(plan.operations[0].tool, "set_logical_frame_target")
        self.assertEqual(plan.operations[0].arguments["logical_frame"], "pelvis")

        invalid = (
            "not json",
            json.dumps({"summary": "missing fields"}),
            json.dumps({
                "summary": "Ambiguous",
                "needs_clarification": True,
                "clarification_question": None,
                "operations": [],
            }),
            json.dumps({
                "summary": "Ambiguous",
                "needs_clarification": True,
                "clarification_question": "Which hand?",
                "operations": [_logical_operation()],
            }),
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(MotionPlanError):
                parse_motion_edit_plan(text)

        oversized = _payload(summary="x" * 4_001)
        with self.assertRaisesRegex(MotionPlanError, "size limit"):
            parse_motion_edit_plan(json.dumps(oversized))


class PlanExecutorTests(unittest.TestCase):
    def test_rolls_back_only_failing_operation_and_keeps_prior_success(self):
        _committed, session, context, tools, _delegate, _provider = _setup()

        def mutate_then_fail(_arguments, semantic_context):
            semantic_context.session.working_document.timeline_duration = 99.0
            raise RuntimeError("deliberate partial failure")

        tools.register(ToolSpec(
            name="test_partial_failure",
            description="Test operation that must be rolled back.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=mutate_then_fail,
            category=ToolCategory.EDIT,
            mutates_working_copy=True,
        ))
        plan = _plan(
            _ensure_operation(0.5),
            {"tool": "test_partial_failure", "arguments": {}},
        )

        result = PlanExecutor(tools).execute(plan, context=context)

        self.assertEqual(
            [operation.succeeded for operation in result.operations],
            [True, False],
        )
        self.assertEqual(session.working_document.timeline_duration, 4.0)
        self.assertIn(0.5, session.working_document.qpos_timeline.times())
        self.assertEqual(len(session.edits), 1)
        self.assertTrue(result.validation_passed)

    def test_unknown_nonediting_and_invalid_operations_never_execute(self):
        _committed, session, context, tools, _delegate, _provider = _setup()
        plan = _plan(
            {"tool": "run_code", "arguments": {"code": "unsafe"}},
            {"tool": "inspect_motion", "arguments": {}},
            {"tool": "ensure_keyframe", "arguments": {"time_seconds": "later"}},
        )

        result = PlanExecutor(tools).execute(plan, context=context)

        self.assertEqual(len(result.failed_operations), 3)
        self.assertFalse(session.has_changes)
        self.assertEqual(session.state, AIEditSessionState.READY)
        self.assertEqual(session.working_document.qpos_timeline.times(), [0.0, 2.0])

    def test_user_authored_work_wins_over_later_ai_plan(self):
        _committed, session, context, tools, _delegate, _provider = _setup()
        PlanExecutor(tools).execute(
            _plan(_logical_operation(height=0.8)),
            context=context,
        )
        pelvis = next(
            frame
            for frame in session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 0.0
        )
        reference = context.working_metadata.reference_for_keyframe(pelvis)
        index = session.working_document.trajectory.index_of_frame(pelvis)
        session.apply_manual(
            UpdateKeyframe(
                index,
                TargetFrame(time=0.0, frame_name="pelvis", z=0.76),
            ),
            affected_entities=(reference,),
        )

        result = PlanExecutor(tools).execute(
            _plan(_logical_operation(height=0.6)),
            context=context,
        )

        self.assertEqual(len(result.failed_operations), 1)
        self.assertEqual(
            session.working_document.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.76,
        )
        self.assertEqual(session.edits[-1].author.value, "user")

    def test_checkpoint_cannot_be_restored_into_another_session(self):
        _committed, session, _context, _tools, _delegate, _provider = _setup()
        _other_committed, other, _context, _tools, _delegate, _provider = _setup()

        with self.assertRaises(TypeError):
            other.restore_checkpoint(session.checkpoint())


class TextMotionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_four_and_twelve_operation_plans_each_use_one_request(self):
        for count in (1, 4, 12):
            with self.subTest(operation_count=count):
                operations = [
                    _ensure_operation(round(0.23 * index, 6))
                    for index in range(1, count + 1)
                ]
                response = ProviderResponse(text=json.dumps(_payload(operations)))
                (
                    _committed,
                    session,
                    context,
                    tools,
                    delegate,
                    provider,
                ) = _setup([response])

                result = await TextMotionWorkflow(provider, tools).run(
                    f"Create {count} Keyframes.",
                    model="mock",
                    context=context,
                )

                self.assertEqual(provider.counter.counts.total, 1)
                self.assertEqual(result.provider_requests, 1)
                self.assertEqual(len(result.execution.operations), count)
                self.assertEqual(len(delegate.requests), 1)
                self.assertEqual(delegate.requests[0].tools, ())
                self.assertIsNotNone(delegate.requests[0].response_schema)
                self.assertNotIn("Provider-authored plan summary", result.text)
                self.assertTrue(session.has_changes)
                delegate.assert_exhausted()

    async def test_failed_local_operation_does_not_trigger_implicit_repair(self):
        response = ProviderResponse(text=json.dumps(_payload([
            {"tool": "run_code", "arguments": {"code": "unsafe"}},
        ])))
        _committed, session, context, tools, delegate, provider = _setup([response])

        result = await TextMotionWorkflow(provider, tools).run(
            "Run arbitrary code.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 1)
        self.assertEqual(len(result.execution.failed_operations), 1)
        self.assertIn("local execution failed", result.text)
        self.assertFalse(session.has_changes)
        delegate.assert_exhausted()

    async def test_clarification_uses_one_request_and_makes_no_change(self):
        response = ProviderResponse(text=json.dumps({
            "summary": "The requested limb is ambiguous.",
            "needs_clarification": True,
            "clarification_question": "Which hand should move?",
            "operations": [],
        }))
        _committed, session, context, tools, _delegate, provider = _setup([response])

        result = await TextMotionWorkflow(provider, tools).run(
            "Move the hand.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 1)
        self.assertEqual(result.text, "Which hand should move?")
        self.assertFalse(session.has_changes)
        self.assertEqual(session.state, AIEditSessionState.READY)

    async def test_refine_prompt_uses_manually_modified_working_copy(self):
        first_response = ProviderResponse(text=json.dumps(_payload([
            _logical_operation(height=0.8),
        ])))
        _committed, session, context, tools, _delegate, provider = _setup([
            first_response
        ])
        await TextMotionWorkflow(provider, tools).run(
            "Lower the pelvis.",
            model="mock",
            context=context,
        )
        pelvis = next(
            frame
            for frame in session.working_document.trajectory.frames
            if frame.frame_name == "pelvis" and frame.time == 0.0
        )
        reference = context.working_metadata.reference_for_keyframe(pelvis)
        index = session.working_document.trajectory.index_of_frame(pelvis)
        session.apply_manual(
            UpdateKeyframe(
                index,
                TargetFrame(time=0.0, frame_name="pelvis", z=0.76),
            ),
            affected_entities=(reference,),
        )

        second_delegate = MockProvider([ProviderResponse(text=json.dumps(_payload([
            _logical_operation("right_hand", height=1.1),
        ])))])
        second_provider = RequestCountingProvider(second_delegate)
        await TextMotionWorkflow(second_provider, tools).run(
            "Keep the pelvis and move the hand.",
            model="mock",
            context=context,
        )

        prompt = second_delegate.requests[0].messages[-1].text
        self.assertIn('"author":"user"', prompt)
        self.assertEqual(
            session.working_document.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.76,
        )

    async def test_invalid_motion_fails_deterministic_validation(self):
        class _InvalidMotion(FakeMotionService):
            def validate_motion(self, document):
                return MotionValidationReport(False, ("unstable",))

        response = ProviderResponse(text=json.dumps(_payload([
            _logical_operation(),
        ])))
        _committed, _session, context, tools, _delegate, provider = _setup(
            [response],
            motion=_InvalidMotion(),
        )

        with self.assertRaisesRegex(PlanExecutionError, "unstable"):
            await TextMotionWorkflow(provider, tools).run(
                "Lower the pelvis.",
                model="mock",
                context=context,
            )
        self.assertEqual(provider.counter.counts.total, 1)

    async def test_controller_default_text_path_uses_new_workflow(self):
        response = ProviderResponse(text=json.dumps(_payload([
            _logical_operation(),
        ])))
        _committed, _session, context, tools, _delegate, provider = _setup([response])
        controller = AIAssistantController.__new__(AIAssistantController)
        controller.model = "mock"
        controller._provider = lambda: provider

        result = await controller._run_text_motion(
            "Lower the pelvis.",
            tools,
            context,
            None,
        )

        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(provider.counter.counts.total, 1)


if __name__ == "__main__":
    unittest.main()

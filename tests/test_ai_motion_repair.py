"""Tests for the single optional text-motion repair request."""

from __future__ import annotations

import json
import unittest

from application.ai.motion_plan import (
    MAX_PLANNED_OPERATIONS,
    MotionPlanError,
    motion_repair_response_schema,
    parse_motion_repair_plan,
)
from application.ai.plan_executor import PlanExecutor
from application.ai.providers import MockStep
from application.ai.schemas import ProviderResponse
from application.ai.text_planner import TextMotionWorkflow
from application.editor_commands import UpdateKeyframe
from core.trajectory import TargetFrame
from tests.test_ai_motion_planning import (
    _ensure_operation,
    _logical_operation,
    _payload,
    _plan,
    _repair_payload,
    _setup,
)


def _failed_initial_with_success():
    return ProviderResponse(text=json.dumps(_payload([
        _ensure_operation(0.5),
        {
            "tool": "ensure_keyframe",
            "arguments": {"time_seconds": "later"},
        },
    ])))


class MotionRepairContractTests(unittest.TestCase):
    def test_schema_allows_only_replacement_operations(self):
        _committed, _session, _context, tools, _delegate, _provider = _setup()

        schema = motion_repair_response_schema(tools)
        self.assertEqual(set(schema["properties"]), {"operations"})
        operations = schema["properties"]["operations"]
        self.assertEqual(operations["minItems"], 1)
        self.assertEqual(operations["maxItems"], MAX_PLANNED_OPERATIONS)
        names = operations["items"]["properties"]["tool"]["enum"]
        self.assertIn("set_logical_frame_target", names)
        self.assertNotIn("inspect_motion", names)
        self.assertNotIn("validate_motion", names)

    def test_parser_requires_one_bounded_replacement_operation(self):
        parsed = parse_motion_repair_plan(json.dumps(_repair_payload([
            _ensure_operation(1.0),
        ])))
        self.assertEqual(len(parsed.operations), 1)
        self.assertEqual(parsed.operations[0].arguments, {"time_seconds": 1.0})

        invalid = (
            "not json",
            json.dumps({"operations": []}),
            json.dumps({"operations": [], "summary": "not allowed"}),
            json.dumps({
                "operations": [{"tool": "ensure_keyframe", "arguments": "{"}],
            }),
            json.dumps(_repair_payload([
                _ensure_operation(float(index))
                for index in range(MAX_PLANNED_OPERATIONS + 1)
            ])),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(MotionPlanError):
                parse_motion_repair_plan(value)


class MotionRepairWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_repair_context_is_compact_and_success_stays_within_two_requests(self):
        responses = [
            _failed_initial_with_success(),
            ProviderResponse(text=json.dumps(_repair_payload([
                _ensure_operation(1.0),
            ]))),
        ]
        _committed, session, context, tools, delegate, provider = _setup(responses)

        result = await TextMotionWorkflow(provider, tools).run(
            "Add Keyframes at 0.5 and 1.0 seconds.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 2)
        self.assertEqual(result.provider_requests, 2)
        self.assertTrue(result.repair_attempted)
        self.assertFalse(result.unresolved_operations)
        self.assertEqual(
            session.working_document.qpos_timeline.times(),
            [0.0, 0.5, 1.0, 2.0],
        )
        repair_request = delegate.requests[1]
        self.assertEqual(repair_request.tools, ())
        self.assertEqual(
            set(repair_request.response_schema["properties"]),
            {"operations"},
        )
        repair_context = json.loads(repair_request.messages[-1].text)
        self.assertEqual(set(repair_context), {
            "failed_operations",
            "important_user_constraints",
            "original_intent",
            "successful_operations_already_applied",
            "updated_motion_context",
        })
        self.assertEqual(repair_context["original_intent"], (
            "Add Keyframes at 0.5 and 1.0 seconds."
        ))
        self.assertEqual(
            repair_context["failed_operations"][0]["tool"],
            "ensure_keyframe",
        )
        successful = repair_context["successful_operations_already_applied"]
        self.assertEqual(len(successful), 1)
        self.assertEqual(successful[0]["arguments"], {"time_seconds": 0.5})
        motion = repair_context["updated_motion_context"]["motion"]
        self.assertIn(0.5, motion["keyframe_times"]["values"])
        self.assertNotIn("trajectory", repair_context)
        self.assertNotIn("conversation", repair_context)
        delegate.assert_exhausted()

    async def test_failed_repair_stops_and_preserves_partial_candidate(self):
        responses = [
            _failed_initial_with_success(),
            ProviderResponse(text=json.dumps(_repair_payload([{
                "tool": "ensure_keyframe",
                "arguments": {"time_seconds": "still later"},
            }]))),
        ]
        _committed, session, context, tools, delegate, provider = _setup(responses)

        result = await TextMotionWorkflow(provider, tools).run(
            "Add two Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 2)
        self.assertEqual(len(result.unresolved_operations), 1)
        self.assertTrue(any(
            "single allowed attempt" in line
            for line in result.proposal_lines
        ))
        self.assertTrue(any(
            line.startswith("Unresolved Ensure Keyframe:")
            for line in result.proposal_lines
        ))
        self.assertIn(0.5, session.working_document.qpos_timeline.times())
        self.assertNotIn(1.0, session.working_document.qpos_timeline.times())
        delegate.assert_exhausted()

    async def test_provider_repair_failure_returns_partial_result_without_third_call(self):
        responses = [
            _failed_initial_with_success(),
            MockStep(error=RuntimeError("offline during repair")),
        ]
        _committed, session, context, tools, delegate, provider = _setup(responses)

        result = await TextMotionWorkflow(provider, tools).run(
            "Add two Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 2)
        self.assertIsNotNone(result.repair_error)
        self.assertEqual(len(result.unresolved_operations), 1)
        self.assertIn("could not produce replacements", result.text)
        self.assertIn(0.5, session.working_document.qpos_timeline.times())
        delegate.assert_exhausted()

    async def test_repair_prompt_carries_user_constraints_and_preserves_manual_edit(self):
        responses = [
            ProviderResponse(text=json.dumps(_payload([
                _ensure_operation(0.5),
                _logical_operation(height=0.6),
            ]))),
            ProviderResponse(text=json.dumps(_repair_payload([
                _ensure_operation(1.0),
            ]))),
        ]
        _committed, session, context, tools, delegate, provider = _setup(responses)
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

        result = await TextMotionWorkflow(provider, tools).run(
            "Keep my pelvis edit and add two Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(result.provider_requests, 2)
        repair_context = json.loads(delegate.requests[1].messages[-1].text)
        constraints = repair_context["important_user_constraints"]
        self.assertTrue(any(
            frame["logical_frame"] == "pelvis" and frame["author"] == "user"
            for frame in constraints["keyframes"]
        ))
        self.assertTrue(constraints["recent_user_edits"])
        self.assertEqual(
            session.working_document.trajectory.targets_at_time(0.0)["pelvis"].z,
            0.76,
        )
        delegate.assert_exhausted()

    async def test_malformed_repair_response_stops_after_second_request(self):
        responses = [
            _failed_initial_with_success(),
            ProviderResponse(text=json.dumps({"operations": []})),
        ]
        _committed, session, context, tools, delegate, provider = _setup(responses)

        result = await TextMotionWorkflow(provider, tools).run(
            "Add two Keyframes.",
            model="mock",
            context=context,
        )

        self.assertEqual(provider.counter.counts.total, 2)
        self.assertIsNotNone(result.repair_error)
        self.assertEqual(len(result.unresolved_operations), 1)
        self.assertIn(0.5, session.working_document.qpos_timeline.times())
        delegate.assert_exhausted()


if __name__ == "__main__":
    unittest.main()

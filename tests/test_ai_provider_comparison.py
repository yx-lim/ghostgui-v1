"""Deterministic tests for provider-neutral behavior comparison."""

from __future__ import annotations

import unittest

from application.ai import (
    AIEditSession,
    AgentLimitError,
    AgentRunResult,
    ProviderComparisonCase,
    ProviderRunStatus,
    ToolCategory,
    ToolExecutionRecord,
    ToolRegistry,
    ToolSpec,
    compare_provider_agents,
)
from application.ai.motion_state import ReplaceMotionState, capture_motion_state
from application.ai.schemas import (
    MessageRole,
    ProviderMessage,
    ToolCall,
    Usage,
)
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


def _document(*, height=0.9):
    document = ProjectDocument("g1")
    document.trajectory.add_frame(
        TargetFrame(time=1.0, frame_name="pelvis", z=height)
    )
    return document


def _stage(session, *, height=0.8):
    candidate = _document(height=height)
    session.apply_ai(ReplaceMotionState(capture_motion_state(candidate)))


def _registry():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="set_logical_frame_target",
        description="Set one logical frame target through IK.",
        input_schema={
            "type": "object",
            "properties": {
                "logical_frame": {"type": "string"},
                "time_seconds": {"type": "number"},
                "position_m": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "mode": {"type": "string", "enum": ["absolute", "delta"]},
            },
            "required": [
                "logical_frame", "time_seconds", "position_m", "mode"
            ],
            "additionalProperties": False,
        },
        handler=lambda arguments, context: {},
        category=ToolCategory.EDIT,
        mutates_working_copy=True,
    ))
    return registry


def _result(*, text, height=0.8, turns=2, tokens=(10, 4)):
    arguments = {
        "logical_frame": "pelvis",
        "time_seconds": 1.0,
        "position_m": [0.0, 0.0, height],
        "mode": "absolute",
    }
    call = ToolCall("provider-native-id", "set_logical_frame_target", arguments)
    return AgentRunResult(
        text=text,
        provider_turns=turns,
        tool_executions=(ToolExecutionRecord(
            "provider-native-id", "set_logical_frame_target", True
        ),),
        validation={"valid": True, "issues": []},
        usage=Usage(*tokens),
        transcript=(
            ProviderMessage(MessageRole.USER, text="lower the pelvis"),
            ProviderMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
            ProviderMessage(MessageRole.TOOL, text="tool result"),
            ProviderMessage(MessageRole.ASSISTANT, text=text),
        ),
    )


class ProviderComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_prose_and_usage_do_not_change_semantic_equivalence(self):
        gemini_base = _document()
        claude_base = _document()
        gemini_session = AIEditSession(gemini_base)
        claude_session = AIEditSession(claude_base)

        async def gemini_run():
            _stage(gemini_session)
            return _result(text="Pelvis lowered.", tokens=(12, 3))

        async def claude_run():
            _stage(claude_session)
            return _result(
                text="I lowered the pelvis at the requested Keyframe.",
                turns=3,
                tokens=(18, 9),
            )

        comparison = await compare_provider_agents("lower pelvis", (
            ProviderComparisonCase(
                "gemini", "gemini-model", gemini_base, gemini_session,
                _registry(), gemini_run
            ),
            ProviderComparisonCase(
                "anthropic", "claude-model", claude_base, claude_session,
                _registry(), claude_run
            ),
        ))

        self.assertTrue(comparison.semantically_equivalent)
        self.assertEqual(
            comparison.operational_differences,
            ("provider_turns", "token_usage"),
        )
        report = comparison.to_dict()
        self.assertNotIn("Pelvis lowered", str(report))
        self.assertEqual(report["runs"][0]["semantic_tools"][0]["name"],
                         "set_logical_frame_target")
        self.assertTrue(
            report["runs"][0]["semantic_tools"][0]["arguments_valid"]
        )

    async def test_different_arguments_and_motion_are_semantic_differences(self):
        first_base = _document()
        second_base = _document()
        first_session = AIEditSession(first_base)
        second_session = AIEditSession(second_base)

        async def first_run():
            _stage(first_session, height=0.8)
            return _result(text="done", height=0.8)

        async def second_run():
            _stage(second_session, height=0.7)
            return _result(text="done", height=0.7)

        comparison = await compare_provider_agents("lower pelvis", (
            ProviderComparisonCase(
                "gemini", "g", first_base, first_session, _registry(), first_run
            ),
            ProviderComparisonCase(
                "anthropic", "c", second_base, second_session, _registry(), second_run
            ),
        ))

        self.assertFalse(comparison.semantically_equivalent)
        self.assertIn("semantic_tool_sequence", comparison.semantic_differences)
        self.assertIn("resulting_motion", comparison.semantic_differences)

    async def test_expected_agent_failure_is_normalized_without_hiding_exceptions(self):
        first_base = _document()
        second_base = _document()
        first_session = AIEditSession(first_base)
        second_session = AIEditSession(second_base)

        async def limited_run():
            raise AgentLimitError("provider-specific detail")

        async def completed_run():
            return AgentRunResult(
                text="no edit needed",
                provider_turns=1,
                tool_executions=(),
                validation=None,
                usage=Usage(),
                transcript=(ProviderMessage(MessageRole.ASSISTANT, text="done"),),
            )

        comparison = await compare_provider_agents("inspect only", (
            ProviderComparisonCase(
                "gemini", "g", first_base, first_session, _registry(), limited_run
            ),
            ProviderComparisonCase(
                "anthropic", "c", second_base, second_session, _registry(), completed_run
            ),
        ))

        self.assertEqual(comparison.runs[0].status, ProviderRunStatus.LIMITED)
        self.assertEqual(comparison.runs[1].status, ProviderRunStatus.COMPLETED)
        self.assertIn("status", comparison.semantic_differences)
        self.assertNotIn("provider-specific detail", str(comparison.to_dict()))

    async def test_rejects_non_identical_baseline_motion(self):
        first_base = _document(height=0.9)
        second_base = _document(height=1.0)
        run_count = 0

        async def run():
            nonlocal run_count
            run_count += 1
            return AgentRunResult(
                text="done",
                provider_turns=1,
                tool_executions=(),
                validation=None,
                usage=Usage(),
                transcript=(ProviderMessage(MessageRole.ASSISTANT, text="done"),),
            )

        with self.assertRaisesRegex(ValueError, "identical baseline"):
            await compare_provider_agents("same input", (
                ProviderComparisonCase(
                    "gemini", "g", first_base, AIEditSession(first_base),
                    _registry(), run
                ),
                ProviderComparisonCase(
                    "anthropic", "c", second_base, AIEditSession(second_base),
                    _registry(), run
                ),
            ))
        self.assertEqual(run_count, 0)

    async def test_rejects_reused_or_pre_staged_sessions_before_provider_work(self):
        first_base = _document()
        second_base = _document()
        shared_session = AIEditSession(first_base)
        run_count = 0

        async def run():
            nonlocal run_count
            run_count += 1
            raise AssertionError("preflight must run before provider work")

        with self.assertRaisesRegex(ValueError, "separate edit sessions"):
            await compare_provider_agents("same input", (
                ProviderComparisonCase(
                    "gemini", "g", first_base, shared_session, _registry(), run
                ),
                ProviderComparisonCase(
                    "anthropic", "c", second_base, shared_session, _registry(), run
                ),
            ))
        self.assertEqual(run_count, 0)

        staged = AIEditSession(first_base)
        _stage(staged)
        with self.assertRaisesRegex(ValueError, "fresh edit sessions"):
            await compare_provider_agents("same input", (
                ProviderComparisonCase(
                    "gemini", "g", first_base, staged, _registry(), run
                ),
                ProviderComparisonCase(
                    "anthropic", "c", second_base, AIEditSession(second_base),
                    _registry(), run
                ),
            ))
        self.assertEqual(run_count, 0)


if __name__ == "__main__":
    unittest.main()

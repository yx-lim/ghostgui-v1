"""Tests for opt-in, secret-safe Motion Assistant diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

from application.ai.diagnostics import MotionAssistantDiagnostics
from application.ai.edit_session import AIEditSession
from application.ai.errors import ProviderResponseError
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.providers import MockProvider, MockStep
from application.ai.schemas import MessageRole, ProviderMessage, ProviderRequest, ProviderResponse, Usage
from application.ai.trajectory_edit_spec import TrajectoryEditMode, TrajectoryEditSpec, TrajectoryOperation, TrajectoryOperationType
from application.ai.trajectory_workflow import CompactMotionWorkflow
from application.project_document import ProjectDocument


def _spec():
    return TrajectoryEditSpec(
        TrajectoryEditMode.EDIT,
        "Raise robot.",
        (TrajectoryOperation(
            TrajectoryOperationType.ROOT_OFFSET,
            {"start_time": 0.0, "end_time": 1.0, "translation_m": [0, 0, 0.05]},
        ),),
    )


class MotionAssistantDiagnosticsTests(unittest.TestCase):
    def test_disabled_diagnostics_write_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MotionAssistantDiagnostics(enabled=False, directory=directory)
            self.assertIsNone(recorder.write())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_enabled_trace_records_bounded_metadata_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MotionAssistantDiagnostics(enabled=True, directory=directory)
            request = ProviderRequest(
                model="claude-test",
                messages=(ProviderMessage(
                    MessageRole.USER,
                    text="Raise it; accidental key sk-ant-secretvalue",
                ),),
            )
            response = ProviderResponse(text='{"ok":true}', usage=Usage(10, 4))
            recorder.record_planning(
                provider_name="anthropic",
                request=request,
                response=response,
                parsed_spec=_spec(),
                latency_seconds=0.25,
            )
            recorder.record_execution({
                "validation": {"valid": True},
                "authorization": "Bearer private-token",
            })
            path = recorder.write()
            payload_text = path.read_text(encoding="utf-8")
            payload = json.loads(payload_text)

        self.assertNotIn("sk-ant-secretvalue", payload_text)
        self.assertNotIn("private-token", payload_text)
        self.assertEqual(payload["planning"]["provider"], "anthropic")
        self.assertEqual(payload["planning"]["token_usage"], {
            "input_tokens": 10,
            "output_tokens": 4,
        })
        self.assertEqual(payload["execution"]["authorization"], "[REDACTED]")

    def test_structural_repair_retains_both_bounded_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MotionAssistantDiagnostics(enabled=True, directory=directory)
            requests = (
                ProviderRequest(
                    model="claude-test",
                    messages=(ProviderMessage(MessageRole.USER, text="Raise it"),),
                ),
                ProviderRequest(
                    model="claude-test",
                    messages=(ProviderMessage(MessageRole.USER, text="Repair it"),),
                ),
            )
            responses = (
                ProviderResponse(text="malformed", usage=Usage(10, 2)),
                ProviderResponse(text='{"valid":true}', usage=Usage(8, 4)),
            )
            recorder.record_planning_attempts(
                provider_name="anthropic",
                requests=requests,
                responses=responses,
                parser_errors=("invalid JSON", None),
                parsed_spec=_spec(),
                latency_seconds=0.5,
            )
            payload = json.loads(recorder.write().read_text(encoding="utf-8"))

        attempts = payload["planning_attempts"]
        self.assertEqual([item["attempt"] for item in attempts], [1, 2])
        self.assertEqual(attempts[0]["normalized_response"]["text"], "malformed")
        self.assertEqual(attempts[0]["parser_error"], "invalid JSON")
        self.assertIsNone(attempts[0]["parsed_spec"])
        self.assertEqual(attempts[1]["parsed_spec"]["summary"], "Raise robot.")
        self.assertIsNone(attempts[1]["parser_error"])

    def test_failure_records_safe_provider_stop_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MotionAssistantDiagnostics(enabled=True, directory=directory)
            error = ProviderResponseError(
                "Anthropic declined the motion request",
                diagnostic_details={
                    "stop_reason": "refusal",
                    "content_block_types": ("refusal",),
                    "authorization": "Bearer private-token",
                },
            )
            recorder.record_failure(
                provider_name="anthropic",
                error=error,
                latency_seconds=0.2,
            )
            payload_text = recorder.write().read_text(encoding="utf-8")
            payload = json.loads(payload_text)

        self.assertNotIn("private-token", payload_text)
        self.assertEqual(payload["failure"]["details"]["stop_reason"], "refusal")
        self.assertEqual(payload["failure"]["details"]["authorization"], "[REDACTED]")


class WorkflowFailureDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_failure_is_written_before_planning_returns(self):
        committed = ProjectDocument("g1")
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        session = AIEditSession(committed, metadata_store=store)
        provider = MockProvider([MockStep(error=ProviderResponseError(
            "Anthropic declined the motion request",
            diagnostic_details={
                "stop_reason": "refusal",
                "content_block_types": ("refusal",),
                "usage": {"input_tokens": 50, "output_tokens": 1},
            },
        ))], provider_name="anthropic")

        with tempfile.TemporaryDirectory() as directory:
            diagnostics = MotionAssistantDiagnostics(
                enabled=True,
                directory=directory,
            )
            with self.assertRaisesRegex(ProviderResponseError, "declined"):
                await CompactMotionWorkflow(
                    provider,
                    object(),
                    metadata,
                    diagnostics=diagnostics,
                ).run(
                    "Create a five-second burpee.",
                    model="mock",
                    context={},
                    session=session,
                )
            path = next(Path(directory).glob("motion-assistant-*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["failure"]["provider"], "anthropic")
        self.assertEqual(payload["failure"]["details"]["stop_reason"], "refusal")
        self.assertFalse(session.has_changes)

    async def test_execution_failure_records_the_successful_plan(self):
        committed = ProjectDocument("g1")
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        session = AIEditSession(committed, metadata_store=store)
        response = ProviderResponse(text=json.dumps({
            "mode": "edit",
            "summary": "Raise the robot.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "translation_m": [0.0, 0.0, 0.05],
                }),
            }],
        }))

        with tempfile.TemporaryDirectory() as directory:
            diagnostics = MotionAssistantDiagnostics(
                enabled=True,
                directory=directory,
            )
            with self.assertRaises(AttributeError):
                await CompactMotionWorkflow(
                    MockProvider([response]),
                    object(),
                    metadata,
                    diagnostics=diagnostics,
                ).run(
                    "Raise the robot.",
                    model="mock",
                    context={},
                    session=session,
                )
            path = next(Path(directory).glob("motion-assistant-*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["planning"]["parsed_spec"]["summary"], "Raise the robot.")
        self.assertEqual(payload["failure"]["error_type"], "AttributeError")
        self.assertFalse(session.has_changes)


if __name__ == "__main__":
    unittest.main()

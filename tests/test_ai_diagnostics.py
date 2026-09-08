"""Tests for opt-in, secret-safe Motion Assistant diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

from application.ai.diagnostics import MotionAssistantDiagnostics
from application.ai.schemas import MessageRole, ProviderMessage, ProviderRequest, ProviderResponse, Usage
from application.ai.trajectory_edit_spec import TrajectoryEditMode, TrajectoryEditSpec, TrajectoryOperation, TrajectoryOperationType


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


if __name__ == "__main__":
    unittest.main()

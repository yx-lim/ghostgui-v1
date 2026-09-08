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


if __name__ == "__main__":
    unittest.main()

"""Offline tests for the opt-in live AI provider smoke harness."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

from application.ai.schemas import ProviderResponse, StopReason, ToolCall
from scripts import smoke_ai_provider


class _FakeProvider:
    provider_name = "fake"

    def __init__(self):
        self.requests = []

    async def generate(self, request, cancellation_token=None):
        self.requests.append(request)
        if request.response_schema is not None:
            return ProviderResponse(text='{"status":"ok"}')
        if request.tools:
            return ProviderResponse(
                tool_calls=(ToolCall("call-1", "move_test_target", {"value": 1}),),
                stop_reason=StopReason.TOOL_CALLS,
            )
        if request.messages[0].motion_frames:
            return ProviderResponse(text="RED")
        return ProviderResponse(text="OK")


class _FailingProvider(_FakeProvider):
    async def generate(self, request, cancellation_token=None):
        raise RuntimeError("provider internals must not be printed")


class ProviderSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_four_independent_normalized_contract_requests(self):
        provider = _FakeProvider()
        output = io.StringIO()

        with redirect_stdout(output):
            passed = await smoke_ai_provider.run_smoke(provider, "test-model")

        self.assertTrue(passed)
        self.assertEqual(len(provider.requests), 4)
        self.assertIn("Text               PASS", output.getvalue())
        self.assertIn("Structured output  PASS", output.getvalue())
        self.assertIn("Tool request       PASS", output.getvalue())
        self.assertIn("Vision             PASS", output.getvalue())

        tool_request = provider.requests[2]
        self.assertEqual(tool_request.tools[0].name, "move_test_target")
        self.assertFalse(hasattr(tool_request.tools[0], "callable"))
        frame = provider.requests[3].messages[0].motion_frames[0]
        self.assertEqual(frame.time_seconds, 0.0)
        self.assertEqual(frame.mime_type, "image/png")
        self.assertTrue(frame.data.startswith(b"\x89PNG\r\n\x1a\n"))

    async def test_unexpected_provider_details_are_not_printed(self):
        output = io.StringIO()

        with redirect_stdout(output):
            passed = await smoke_ai_provider.run_smoke(
                _FailingProvider(),
                "test-model",
            )

        self.assertFalse(passed)
        self.assertEqual(output.getvalue().count("FAIL - RuntimeError"), 4)
        self.assertNotIn("provider internals", output.getvalue())


class ProviderSmokeUtilityTests(unittest.TestCase):
    def test_generated_vision_fixture_is_small_valid_png(self):
        image = smoke_ai_provider._solid_red_png()

        self.assertLess(len(image), 1024)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(image.endswith(b"IEND\xaeB`\x82"))

    def test_default_cli_models_match_current_ghostgui_models(self):
        self.assertEqual(
            smoke_ai_provider._DEFAULT_MODELS,
            {
                "gemini": "gemini-3.7-flash",
                "anthropic": "claude-sonnet-5",
            },
        )

    def test_live_workflow_is_manual_only(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "live-ai-provider-smoke.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertNotIn("\n  pull_request:", workflow)
        self.assertNotIn("\n  schedule:", workflow)


if __name__ == "__main__":
    unittest.main()

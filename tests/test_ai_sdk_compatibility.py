"""Pinned SDK baseline and provider-neutral response compatibility tests."""

from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from application.ai.provider_registry import DEFAULT_PROVIDER_REGISTRY
from application.ai.providers.anthropic import _parse_anthropic_response
from application.ai.providers.gemini import _parse_gemini_response
from application.ai.schemas import StopReason


EXPECTED_AI_DEPENDENCIES = {
    "anthropic==1.4.0",
    "google-genai==2.21.0",
    "keyring==25.7.0",
}


class AISDKCompatibilityTests(unittest.TestCase):
    def test_optional_dependencies_match_the_live_smoke_baseline(self):
        project = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        section = re.search(
            r"\[project\.optional-dependencies\]\s*ai\s*=\s*\[(.*?)\]",
            project,
            re.DOTALL,
        )
        self.assertIsNotNone(section)
        requirements = set(re.findall(r'"([^"]+)"', section.group(1)))

        self.assertEqual(requirements, EXPECTED_AI_DEPENDENCIES)
        registered_sdks = {
            registration.sdk_distribution
            for registration in DEFAULT_PROVIDER_REGISTRY.registrations
        }
        self.assertEqual(registered_sdks, {"anthropic", "google-genai"})

    def test_pinned_provider_parsers_share_the_normalized_tool_result(self):
        tool_call = SimpleNamespace(
            id="call-1",
            name="validate_motion",
            args={"scope": "candidate"},
        )
        gemini_raw = SimpleNamespace(
            response_id="gemini-response",
            candidates=(SimpleNamespace(
                content=SimpleNamespace(parts=(
                    SimpleNamespace(text="Checking.", function_call=None),
                    SimpleNamespace(text=None, function_call=tool_call),
                )),
                finish_reason="STOP",
            ),),
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=5,
            ),
        )
        anthropic_raw = SimpleNamespace(
            content=(
                SimpleNamespace(type="text", text="Checking."),
                SimpleNamespace(
                    type="tool_use",
                    id="call-1",
                    name="validate_motion",
                    input={"scope": "candidate"},
                ),
            ),
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=12, output_tokens=5),
        )

        gemini = _parse_gemini_response(gemini_raw)
        anthropic = _parse_anthropic_response(anthropic_raw)

        self.assertEqual(gemini, anthropic)
        self.assertEqual(gemini.stop_reason, StopReason.TOOL_CALLS)
        self.assertEqual(gemini.tool_calls[0].name, "validate_motion")


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the provider connection probe."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch


try:
    from gui.ai_assistant_controller import AIAssistantController
except ImportError:
    AIAssistantController = None


@unittest.skipUnless(
    AIAssistantController is not None,
    "Motion Assistant dependencies unavailable",
)
class AIConnectionProbeTests(unittest.TestCase):
    def test_probe_does_not_starve_thinking_models_of_output_tokens(self):
        calls = []

        class _Provider:
            def __init__(self, *, api_key):
                self.api_key = api_key

            async def generate(self, request):
                calls.append(request)
                return SimpleNamespace(text="OK")

            async def aclose(self):
                return None

        controller = AIAssistantController.__new__(AIAssistantController)
        controller._session_api_key = None
        with patch("gui.ai_assistant_controller.GeminiProvider", _Provider):
            result = asyncio.run(
                controller._run_connection_test(
                    "gemini",
                    "gemini-3.6-flash",
                    "test-key",
                )
            )

        self.assertEqual(result, "OK")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].max_output_tokens)


if __name__ == "__main__":
    unittest.main()

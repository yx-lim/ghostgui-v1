"""Regression tests for the provider connection probe."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from application.ai.connection_cache import ConnectionTestCache


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

    def test_success_is_cached_by_provider_model_and_credential_identity(self):
        class _Dialog:
            def __init__(self):
                self.results = []

            def set_test_result(self, succeeded, message):
                self.results.append((succeeded, message))

        class _Jobs:
            def __init__(self):
                self.submissions = 0

            def submit_cancellable(
                self,
                _name,
                work,
                succeeded,
                failed,
                _cancelled,
            ):
                self.submissions += 1
                try:
                    result = work(SimpleNamespace(cancellation_requested=False))
                except Exception as error:
                    failed(error)
                else:
                    succeeded(result)
                return object()

        controller = AIAssistantController.__new__(AIAssistantController)
        controller._settings_dialog = _Dialog()
        controller._session_api_keys = {}
        controller._connection_test_cache = ConnectionTestCache()
        controller.background_jobs = _Jobs()
        run_test = AsyncMock(return_value="OK")

        with patch.object(controller, "_run_connection_test", run_test):
            controller._test_connection("gemini", "model-a", "key-a")
            controller._test_connection("gemini", "model-a", "key-a")
            self.assertEqual(controller.background_jobs.submissions, 1)
            self.assertIn("Cached", controller._settings_dialog.results[-1][1])

            controller._test_connection("gemini", "model-b", "key-a")
            controller._test_connection("gemini", "model-b", "key-b")
            controller._test_connection("anthropic", "model-b", "key-b")

        self.assertEqual(controller.background_jobs.submissions, 4)
        self.assertEqual(run_test.await_count, 4)

    def test_failed_connection_test_is_not_cached(self):
        class _Dialog:
            def set_test_result(self, _succeeded, _message):
                return None

        class _Jobs:
            def __init__(self):
                self.submissions = 0

            def submit_cancellable(
                self,
                _name,
                work,
                _succeeded,
                failed,
                _cancelled,
            ):
                self.submissions += 1
                try:
                    work(SimpleNamespace(cancellation_requested=False))
                except Exception as error:
                    failed(error)
                return object()

        controller = AIAssistantController.__new__(AIAssistantController)
        controller._settings_dialog = _Dialog()
        controller._session_api_keys = {}
        controller._connection_test_cache = ConnectionTestCache()
        controller.background_jobs = _Jobs()
        run_test = AsyncMock(side_effect=RuntimeError("offline"))

        with patch.object(controller, "_run_connection_test", run_test):
            controller._test_connection("gemini", "model-a", "key-a")
            controller._test_connection("gemini", "model-a", "key-a")

        self.assertEqual(controller.background_jobs.submissions, 2)
        self.assertEqual(run_test.await_count, 2)

    def test_credential_identity_uses_explicit_session_then_default_source(self):
        controller = AIAssistantController.__new__(AIAssistantController)
        controller._session_api_keys = {"gemini": "session-key"}

        self.assertEqual(
            controller._effective_connection_credential("gemini", "explicit-key"),
            ("explicit-key", "settings-dialog"),
        )
        self.assertEqual(
            controller._effective_connection_credential("gemini", ""),
            ("session-key", "session-memory"),
        )

        controller._session_api_keys.clear()
        source = SimpleNamespace(get_secret=lambda _provider: "default-key")
        with patch(
            "gui.ai_assistant_controller.default_credential_source",
            return_value=source,
        ):
            self.assertEqual(
                controller._effective_connection_credential("gemini", ""),
                ("default-key", "keyring-or-environment"),
            )


if __name__ == "__main__":
    unittest.main()

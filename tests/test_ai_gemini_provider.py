"""Tests for Gemini conversion without network access or a real API key."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from application.ai.credentials import EnvironmentCredentialSource
from application.ai.errors import (
    ProviderAuthenticationError,
    ProviderCancelledError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from application.ai.providers import GeminiProvider, LLMProvider
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def _response(*parts, finish_reason="STOP", input_tokens=9, output_tokens=4):
    return SimpleNamespace(
        response_id="response-7",
        candidates=(
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=finish_reason,
            ),
        ),
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
        ),
    )


class _FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if isinstance(self.outcome, asyncio.Event):
            await self.outcome.wait()
        return self.outcome


class _FakeClient:
    def __init__(self, outcome):
        self.models = _FakeModels(outcome)
        self.aio = SimpleNamespace(models=self.models)


class _Token:
    cancellation_requested = False


class _StatusError(RuntimeError):
    def __init__(self, code, message="request contained secret-key"):
        super().__init__(message)
        self.status_code = code


class ServerError(RuntimeError):
    """Stand-in for SDK releases that omit a numeric status attribute."""


def _request(**changes):
    values = {
        "model": "gemini-test-model",
        "messages": (ProviderMessage(MessageRole.USER, text="Lower the pelvis"),),
    }
    values.update(changes)
    return ProviderRequest(**values)


class GeminiProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_converts_text_tools_and_response_to_common_contract(self):
        function_call = SimpleNamespace(
            id="call-4",
            name="set_logical_frame_target",
            args={"frame": "pelvis", "delta_z": -0.05},
        )
        client = _FakeClient(
            _response(
                SimpleNamespace(text="Working.", function_call=None),
                SimpleNamespace(text=None, function_call=function_call),
            )
        )
        provider = GeminiProvider(client=client, cancellation_poll_seconds=0.001)
        tool = ToolDefinition(
            "set_logical_frame_target",
            "Move a logical frame through IK.",
            {
                "type": "object",
                "properties": {"frame": {"type": "string"}},
                "required": ["frame"],
                "additionalProperties": False,
            },
        )

        result = await provider.generate(
            _request(
                messages=(
                    ProviderMessage(MessageRole.SYSTEM, text="Use semantic tools."),
                    ProviderMessage(MessageRole.USER, text="Lower the pelvis"),
                ),
                tools=(tool,),
                max_output_tokens=700,
            )
        )

        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(result.stop_reason, StopReason.TOOL_CALLS)
        self.assertEqual(result.text, "Working.")
        self.assertEqual(result.tool_calls[0].identifier, "call-4")
        self.assertEqual(result.usage.input_tokens, 9)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-test-model")
        self.assertEqual(call["config"]["system_instruction"], "Use semantic tools.")
        self.assertEqual(call["config"]["max_output_tokens"], 700)
        self.assertEqual(
            call["config"]["tools"][0]["function_declarations"][0]["name"],
            tool.name,
        )
        self.assertEqual(
            call["config"]["tools"][0]["function_declarations"][0][
                "parameters_json_schema"
            ],
            tool.input_schema,
        )
        self.assertEqual(
            call["config"]["automatic_function_calling"], {"disable": True}
        )

    async def test_converts_tool_history_and_structured_output(self):
        client = _FakeClient(_response(SimpleNamespace(text='{"valid":true}', function_call=None)))
        provider = GeminiProvider(client=client, cancellation_poll_seconds=0.001)
        assistant_call = ToolCall("call-1", "validate_motion", {})
        tool_result = ToolResult(
            "call-1", "validate_motion", {"valid": True}, is_error=False
        )
        schema = {
            "type": "object",
            "properties": {"valid": {"type": "boolean"}},
            "required": ["valid"],
        }

        result = await provider.generate(
            _request(
                messages=(
                    ProviderMessage(MessageRole.USER, text="Check it"),
                    ProviderMessage(MessageRole.ASSISTANT, tool_calls=(assistant_call,)),
                    ProviderMessage(MessageRole.TOOL, tool_results=(tool_result,)),
                ),
                response_schema=schema,
            )
        )

        self.assertEqual(result.text, '{"valid":true}')
        call = client.models.calls[0]
        self.assertEqual(call["config"]["response_mime_type"], "application/json")
        self.assertEqual(call["config"]["response_json_schema"], schema)
        function_call = call["contents"][1]["parts"][0]["function_call"]
        function_response = call["contents"][2]["parts"][0]["function_response"]
        self.assertEqual(function_call["id"], "call-1")
        self.assertEqual(function_response["id"], "call-1")
        self.assertEqual(function_response["response"], {"valid": True})

    async def test_preserves_gemini_3_thought_signature_across_tool_turn(self):
        function_call = SimpleNamespace(
            id="call-signed",
            name="validate_motion",
            args={},
        )
        responses = [
            _response(
                SimpleNamespace(
                    text=None,
                    function_call=function_call,
                    thought_signature=b"opaque-signature",
                )
            ),
            _response(SimpleNamespace(text="Done", function_call=None)),
        ]

        class _SequencedModels:
            def __init__(self):
                self.calls = []

            async def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return responses.pop(0)

        models = _SequencedModels()
        provider = GeminiProvider(
            client=SimpleNamespace(aio=SimpleNamespace(models=models)),
            cancellation_poll_seconds=0.001,
        )
        first = await provider.generate(_request())
        call = first.tool_calls[0]

        await provider.generate(
            _request(
                messages=(
                    ProviderMessage(MessageRole.USER, text="Check it"),
                    ProviderMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
                    ProviderMessage(
                        MessageRole.TOOL,
                        tool_results=(
                            ToolResult(call.identifier, call.name, {"valid": True}),
                        ),
                    ),
                )
            )
        )

        replayed_part = models.calls[1]["contents"][1]["parts"][0]
        self.assertEqual(
            replayed_part["thought_signature"],
            b"opaque-signature",
        )

    async def test_image_input_always_precedes_bytes_with_exact_time_metadata(self):
        frames = tuple(
            MotionFrameImage(
                data=variant.value.encode("ascii"),
                mime_type="image/png",
                time_seconds=2.8,
                variant=variant,
                comparison_id="frame-3",
                label="frame_3",
            )
            for variant in (ImageVariant.ORIGINAL, ImageVariant.CANDIDATE)
        )
        client = _FakeClient(_response(SimpleNamespace(text="At about 2.8 s...", function_call=None)))
        provider = GeminiProvider(client=client, cancellation_poll_seconds=0.001)

        await provider.generate(
            _request(messages=(ProviderMessage(MessageRole.USER, motion_frames=frames),))
        )

        parts = client.models.calls[0]["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "frame_3 (original) = t=2.800000 s")
        self.assertEqual(parts[1]["inline_data"]["data"], b"original")
        self.assertEqual(parts[2]["text"], "frame_3 (candidate) = t=2.800000 s")
        self.assertEqual(parts[3]["inline_data"]["data"], b"candidate")

    async def test_normalizes_auth_rate_limit_and_offline_errors_without_details(self):
        cases = (
            (400, ProviderError),
            (401, ProviderAuthenticationError),
            (404, ProviderConfigurationError),
            (429, ProviderRateLimitError),
            (503, ProviderError),
        )
        for code, expected in cases:
            with self.subTest(code=code):
                provider = GeminiProvider(
                    client=_FakeClient(_StatusError(code)),
                    cancellation_poll_seconds=0.001,
                    max_attempts=1,
                )
                with self.assertRaises(expected) as raised:
                    await provider.generate(_request())
                self.assertNotIn("secret-key", str(raised.exception))
                if code == 400:
                    self.assertIn("HTTP 400 INVALID_ARGUMENT", str(raised.exception))

    async def test_retries_transient_server_errors_then_returns_response(self):
        class _RecoveringModels:
            def __init__(self):
                self.calls = []

            async def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) < 3:
                    raise _StatusError(503)
                return _response(
                    SimpleNamespace(text="OK", function_call=None)
                )

        models = _RecoveringModels()
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        provider = GeminiProvider(
            client=client,
            cancellation_poll_seconds=0.001,
            retry_base_seconds=0.001,
        )

        result = await provider.generate(_request())

        self.assertEqual(result.text, "OK")
        self.assertEqual(len(models.calls), 3)

    async def test_reports_retryable_server_failure_as_actionable_error(self):
        cases = (_StatusError(500), ServerError("request contained secret-key"))
        for error in cases:
            with self.subTest(error=error):
                client = _FakeClient(error)
                provider = GeminiProvider(
                    client=client,
                    cancellation_poll_seconds=0.001,
                    max_attempts=2,
                    retry_base_seconds=0.001,
                )

                with self.assertRaisesRegex(
                    ProviderError,
                    r"temporarily unavailable.*after 2 attempts; try again",
                ) as raised:
                    await provider.generate(_request())

                self.assertEqual(len(client.models.calls), 2)
                self.assertNotIn("secret-key", str(raised.exception))

    async def test_can_cancel_during_retry_backoff(self):
        client = _FakeClient(_StatusError(503))
        provider = GeminiProvider(
            client=client,
            cancellation_poll_seconds=0.001,
            retry_base_seconds=1.0,
        )
        token = _Token()
        task = asyncio.create_task(provider.generate(_request(), token))
        while not client.models.calls:
            await asyncio.sleep(0)
        token.cancellation_requested = True

        with self.assertRaises(ProviderCancelledError):
            await task

        self.assertEqual(len(client.models.calls), 1)

    async def test_cancels_an_active_sdk_request(self):
        gate = asyncio.Event()
        provider = GeminiProvider(
            client=_FakeClient(gate), cancellation_poll_seconds=0.001
        )
        token = _Token()
        task = asyncio.create_task(provider.generate(_request(), token))
        await asyncio.sleep(0.005)
        token.cancellation_requested = True

        with self.assertRaises(ProviderCancelledError):
            await task

    async def test_rejects_empty_and_malformed_responses(self):
        cases = (
            SimpleNamespace(candidates=()),
            _response(SimpleNamespace(text=None, function_call=None)),
            _response(
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(id="x", name="tool", args="bad"),
                )
            ),
        )
        for response in cases:
            with self.subTest(response=response):
                provider = GeminiProvider(
                    client=_FakeClient(response), cancellation_poll_seconds=0.001
                )
                with self.assertRaises(ProviderResponseError):
                    await provider.generate(_request())

    def test_requires_configuration_only_when_constructing_real_client(self):
        class _EmptySource:
            def get_secret(self, provider_name):
                return None

        with self.assertRaisesRegex(ProviderConfigurationError, "not configured"):
            GeminiProvider(credential_source=_EmptySource())

    def test_environment_key_precedence_matches_official_sdk(self):
        source = EnvironmentCredentialSource(
            {"GEMINI_API_KEY": "gemini", "GOOGLE_API_KEY": "google"}
        )
        self.assertEqual(source.get_secret("gemini"), "google")
        self.assertIsNone(source.get_secret("anthropic"))

    def test_declares_provider_and_model_capabilities(self):
        provider = GeminiProvider(client=_FakeClient(None))
        self.assertEqual(provider.provider_name, "gemini")
        self.assertTrue(provider.capabilities.supports_tools)
        self.assertTrue(provider.capabilities.supports_vision)
        self.assertTrue(provider.capabilities.supports_structured_output)
        self.assertEqual(provider.capabilities.max_images_per_request, 16)


if __name__ == "__main__":
    unittest.main()

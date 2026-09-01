"""Tests for the Anthropic Claude provider adapter without network access."""

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
from application.ai.providers.anthropic import (
    DEFAULT_ANTHROPIC_CAPABILITIES,
    DEFAULT_CLAUDE_MODEL,
    AnthropicProvider,
)
from application.ai.providers.base import LLMProvider
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


class _FakeMessages:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def create(self, **arguments):
        self.calls.append(arguments)
        if isinstance(self.result, BaseException):
            raise self.result
        if isinstance(self.result, asyncio.Event):
            await self.result.wait()
        return self.result


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)
        self.closed = False

    async def close(self):
        self.closed = True


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__("secret provider detail")
        self.status_code = status_code


class _Token:
    def __init__(self, cancelled=False):
        self.cancellation_requested = cancelled


def _response(*blocks, stop_reason="end_turn", input_tokens=12, output_tokens=4):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _request(**changes):
    values = {
        "model": DEFAULT_CLAUDE_MODEL,
        "messages": (ProviderMessage(MessageRole.USER, text="Move the hand"),),
    }
    values.update(changes)
    return ProviderRequest(**values)


class AnthropicProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_declares_common_claude_capabilities(self):
        provider = AnthropicProvider(client=_FakeClient(_response()))

        self.assertEqual(provider.provider_name, "anthropic")
        self.assertIsInstance(provider, LLMProvider)
        self.assertEqual(provider.capabilities, DEFAULT_ANTHROPIC_CAPABILITIES)
        self.assertTrue(provider.capabilities.supports_tools)
        self.assertTrue(provider.capabilities.supports_vision)
        self.assertTrue(provider.capabilities.supports_structured_output)

    async def test_converts_system_images_tools_and_structured_output(self):
        client = _FakeClient(_response(
            SimpleNamespace(type="text", text='{"ok":true}'),
        ))
        provider = AnthropicProvider(client=client, cancellation_poll_seconds=0.001)
        frame = MotionFrameImage(
            data=b"png-bytes",
            mime_type="image/png",
            time_seconds=2.8,
            variant=ImageVariant.CANDIDATE,
            comparison_id="frame_3",
            label="frame_3",
        )
        tool = ToolDefinition(
            "inspect_motion",
            "Inspect motion.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["score"],
            "additionalProperties": False,
        }

        result = await provider.generate(_request(
            messages=(
                ProviderMessage(MessageRole.SYSTEM, text="Stay bounded."),
                ProviderMessage(
                    MessageRole.USER,
                    text="Inspect this frame.",
                    motion_frames=(frame,),
                ),
            ),
            tools=(tool,),
            response_schema=schema,
            max_output_tokens=800,
        ))

        self.assertEqual(result.text, '{"ok":true}')
        call = client.messages.calls[0]
        self.assertEqual(call["system"], "Stay bounded.")
        self.assertEqual(call["max_tokens"], 800)
        self.assertEqual(call["tools"][0]["input_schema"], tool.input_schema)
        content = call["messages"][0]["content"]
        self.assertEqual(content[1]["text"], "frame_3 (candidate) = t=2.800000 s")
        self.assertEqual(content[2]["source"]["data"], "cG5nLWJ5dGVz")
        output_schema = call["output_config"]["format"]["schema"]
        self.assertNotIn("minimum", output_schema["properties"]["score"])
        self.assertEqual(schema["properties"]["score"]["minimum"], 0)

    async def test_converts_tool_history_and_parallel_calls(self):
        raw = _response(
            SimpleNamespace(
                type="tool_use",
                id="call-2",
                name="validate_motion",
                input={},
            ),
            SimpleNamespace(type="text", text="Checking."),
            stop_reason="tool_use",
            input_tokens=20,
            output_tokens=7,
        )
        client = _FakeClient(raw)
        provider = AnthropicProvider(client=client, cancellation_poll_seconds=0.001)
        prior_call = ToolCall("call-1", "inspect_motion", {})
        prior_result = ToolResult("call-1", "inspect_motion", {"duration": 4.0})

        result = await provider.generate(_request(messages=(
            ProviderMessage(MessageRole.USER, text="Inspect"),
            ProviderMessage(MessageRole.ASSISTANT, tool_calls=(prior_call,)),
            ProviderMessage(MessageRole.TOOL, tool_results=(prior_result,)),
        )))

        self.assertEqual(result.stop_reason, StopReason.TOOL_CALLS)
        self.assertEqual(result.tool_calls[0].identifier, "call-2")
        self.assertEqual(result.usage.input_tokens, 20)
        request_messages = client.messages.calls[0]["messages"]
        self.assertEqual(request_messages[1]["content"][0]["type"], "tool_use")
        tool_result = request_messages[2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call-1")
        self.assertEqual(tool_result["content"], '{"duration":4.0}')

    async def test_normalizes_errors_without_leaking_provider_details(self):
        cases = (
            (401, ProviderAuthenticationError),
            (429, ProviderRateLimitError),
            (404, ProviderConfigurationError),
            (503, ProviderError),
        )
        for status, expected in cases:
            with self.subTest(status=status):
                provider = AnthropicProvider(
                    client=_FakeClient(_StatusError(status)),
                    cancellation_poll_seconds=0.001,
                )
                with self.assertRaises(expected) as raised:
                    await provider.generate(_request())
                self.assertNotIn("secret provider detail", str(raised.exception))

    async def test_rejects_empty_and_malformed_responses(self):
        cases = (
            _response(),
            _response(SimpleNamespace(
                type="tool_use", id="", name="inspect_motion", input={}
            )),
        )
        for raw in cases:
            with self.subTest(raw=raw):
                provider = AnthropicProvider(
                    client=_FakeClient(raw),
                    cancellation_poll_seconds=0.001,
                )
                with self.assertRaises(ProviderResponseError):
                    await provider.generate(_request())

    async def test_cancellation_stops_active_sdk_request(self):
        gate = asyncio.Event()
        provider = AnthropicProvider(
            client=_FakeClient(gate), cancellation_poll_seconds=0.001
        )
        token = _Token()
        task = asyncio.create_task(provider.generate(_request(), token))
        await asyncio.sleep(0.005)
        token.cancellation_requested = True

        with self.assertRaises(ProviderCancelledError):
            await task

    async def test_injected_client_is_not_closed(self):
        client = _FakeClient(_response(SimpleNamespace(type="text", text="OK")))
        provider = AnthropicProvider(client=client)
        await provider.aclose()
        self.assertFalse(client.closed)


class AnthropicCredentialTests(unittest.TestCase):
    def test_environment_source_reads_only_anthropic_official_key(self):
        source = EnvironmentCredentialSource({
            "ANTHROPIC_API_KEY": "anthropic-key",
            "GOOGLE_API_KEY": "google-key",
        })
        self.assertEqual(source.get_secret("anthropic"), "anthropic-key")
        self.assertEqual(source.get_secret("gemini"), "google-key")
        self.assertIsNone(source.get_secret("unknown"))


if __name__ == "__main__":
    unittest.main()

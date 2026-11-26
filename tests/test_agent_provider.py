"""M1: lossless streaming agent provider, pricing, replay and secret hygiene (network-free)."""
import asyncio
import json
import logging
import unittest

from application.ai.agent_wire import (
    AgentStopReason, AgentTurnRequest, AgentUsage, Effort, UsageIteration, canonical_json,
    tool_result_block,
)
from application.ai.errors import (
    ProviderAuthenticationError, ProviderBadRequestError, ProviderCancelledError,
    ProviderContextLimitError, ProviderMalformedStreamError, ProviderModelUnavailableError,
    ProviderOfflineError, ProviderOverloadedError, ProviderPermissionError, ProviderRateLimitError,
    ProviderResponseError,
)
from application.ai.pricing import cost_usd
from application.ai.provider_registry import DEFAULT_PROVIDER_REGISTRY
from application.ai.providers.anthropic_agent import (
    AnthropicAgentProvider, FALLBACK_BETA, build_stream_arguments, map_error,
)
from application.ai.providers.mock import ScriptedAgentProvider, scripted_turn
from application.ai.providers.replay import AgentReplayProvider, record_agent_turn
from application.ai.secrets import SecretScrubFilter, SecretStr, child_environment, contains_secret

TOOLS = ({"name": "b_tool", "description": "B.", "input_schema": {"type": "object", "properties": {},
          "required": [], "additionalProperties": False}},
         {"name": "a_tool", "description": "A.", "input_schema": {"type": "object", "properties": {},
          "required": [], "additionalProperties": False}})
THINKING = {"type": "thinking", "thinking": "plan", "signature": "sig-ABC=="}
FALLBACK = {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"}}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "a_tool", "input": {}}


def request(**overrides):
    values = dict(model="claude-opus-5-5", system="FROZEN", tools=TOOLS,
                  messages=({"role": "user", "content": [{"type": "text", "text": "hi"}]},))
    values.update(overrides)
    return AgentTurnRequest(**values)


class Obj:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeMessage:
    def __init__(self, data):
        self._data = data
        self._request_id = "req_123"

    def to_dict(self):
        return json.loads(json.dumps(self._data))


class FakeStream:
    def __init__(self, events, message, fail=None, hang=False):
        self.events, self.message, self.fail, self.hang = events, message, fail, hang
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event
        if self.fail is not None:
            raise self.fail
        if self.hang:
            await asyncio.sleep(30)

    async def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, stream=None, models=None):
        self.calls = []
        self._stream = stream
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return outer._stream

        class Beta:
            messages = Messages()

        class Models:
            async def retrieve(self, model):
                result = (models or {}).get(model)
                if isinstance(result, Exception):
                    raise result
                return result or {"id": model, "max_input_tokens": 1_000_000, "display_name": model}

        self.beta = Beta()
        self.models = Models()


class Token:
    def __init__(self):
        self.cancellation_requested = False


class StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__("body that must not leak: secret prompt")
        self.status_code = status
        self.response = Obj(headers=headers or {})


class RequestShapeTests(unittest.TestCase):
    def test_cache_thinking_effort_fallbacks_and_eager_tools(self):
        arguments = build_stream_arguments(request())
        self.assertEqual(arguments["system"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(arguments["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(arguments["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(arguments["output_config"], {"effort": "medium"})
        self.assertEqual(arguments["betas"], [FALLBACK_BETA])
        self.assertEqual(arguments["fallbacks"], "default")
        self.assertTrue(all(tool["eager_input_streaming"] for tool in arguments["tools"]))
        for absent in ("temperature", "tool_choice", "top_p"):
            self.assertNotIn(absent, arguments)

    def test_no_fallbacks_on_haiku_or_when_disabled(self):
        self.assertNotIn("fallbacks", build_stream_arguments(request(model="claude-haiku-5-5")))
        self.assertNotIn("betas", build_stream_arguments(request(fallbacks=False)))
        self.assertEqual(build_stream_arguments(request(effort=Effort.HIGH))["output_config"]["effort"], "high")

    def test_prefix_is_byte_stable_across_turns(self):
        first = build_stream_arguments(request())
        longer = request(messages=(
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [THINKING, TOOL_USE]},
            {"role": "user", "content": [tool_result_block("toolu_1", {"ok": True})]},
        ))
        second = build_stream_arguments(longer)
        for key in ("tools", "system", "thinking", "output_config", "model"):
            self.assertEqual(canonical_json(first[key]), canonical_json(second[key]))
        self.assertEqual(canonical_json(second["messages"][0]), canonical_json(first["messages"][0]))

    def test_request_validation(self):
        with self.assertRaises(ValueError):
            request(messages=({"role": "assistant", "content": "x"},))
        with self.assertRaises(ValueError):
            request(cache_ttl="2h")


class StreamTests(unittest.TestCase):
    def run_turn(self, stream, token=None, events=None):
        client = FakeClient(stream)
        provider = AnthropicAgentProvider(client=client, cancellation_poll_seconds=0.01)
        sink = [] if events is None else events
        result = asyncio.run(provider.stream_turn(request(), sink.append, token))
        return client, result

    def test_blocks_round_trip_verbatim_with_usage_tiers(self):
        message = FakeMessage({
            "model": "claude-opus-5-5", "stop_reason": "tool_use",
            "content": [THINKING, FALLBACK, {"type": "text", "text": "ok"}, TOOL_USE],
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 900,
                      "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 70},
                      "iterations": [{"type": "message", "model": "claude-opus-5-5", "input_tokens": 4,
                                      "output_tokens": 1},
                                     {"type": "fallback_message", "model": "claude-opus-5",
                                      "input_tokens": 6, "output_tokens": 4}]},
        })
        events = [Obj(type="thinking", thinking="pl"), Obj(type="text", text="o"),
                  Obj(type="content_block_start", index=3, content_block=Obj(type="tool_use", name="a_tool"))]
        sink = []
        client, result = self.run_turn(FakeStream(events, message), events=sink)
        self.assertEqual(canonical_json(list(result.content)),
                         canonical_json([THINKING, FALLBACK, {"type": "text", "text": "ok"}, TOOL_USE]))
        self.assertIs(result.stop_reason, AgentStopReason.TOOL_USE)
        self.assertEqual(result.usage.cache_read, 900)
        self.assertEqual(result.usage.cache_write_1h, 70)
        self.assertEqual(len(result.usage.iterations), 2)
        self.assertEqual(result.request_id, "req_123")
        self.assertEqual([e.kind for e in sink], ["turn_start", "thinking", "text", "tool_use_start"])
        self.assertEqual(len(client.calls), 1)

    def test_refusal_and_pause_are_values(self):
        message = FakeMessage({"model": "claude-opus-5-5", "stop_reason": "refusal", "content": [],
                               "stop_details": {"type": "refusal", "category": "cyber"}, "usage": {}})
        _, result = self.run_turn(FakeStream([], message))
        self.assertIs(result.stop_reason, AgentStopReason.REFUSAL)
        self.assertEqual(result.refusal_category, "cyber")
        message = FakeMessage({"model": "m", "stop_reason": "pause_turn", "content": [], "usage": {}})
        self.assertIs(self.run_turn(FakeStream([], message))[1].stop_reason, AgentStopReason.PAUSE_TURN)

    def test_malformed_eager_json_is_reissuable(self):
        with self.assertRaises(ProviderMalformedStreamError):
            self.run_turn(FakeStream([], None, fail=json.JSONDecodeError("bad", "{", 0)))

    def test_cancel_closes_stream(self):
        stream = FakeStream([], None, hang=True)
        token = Token()
        client = FakeClient(stream)
        provider = AnthropicAgentProvider(client=client, cancellation_poll_seconds=0.01)

        async def scenario():
            task = asyncio.ensure_future(provider.stream_turn(request(), None, token))
            await asyncio.sleep(0.05)
            token.cancellation_requested = True
            return await task

        with self.assertRaises(ProviderCancelledError):
            asyncio.run(scenario())
        self.assertTrue(stream.closed)

    def test_error_matrix(self):
        cases = {401: ProviderAuthenticationError, 403: ProviderPermissionError,
                 404: ProviderModelUnavailableError, 429: ProviderRateLimitError,
                 413: ProviderContextLimitError, 529: ProviderOverloadedError,
                 500: ProviderOverloadedError, 400: ProviderBadRequestError}
        for status, expected in cases.items():
            mapped = map_error(StatusError(status, {"retry-after": "7"}))
            self.assertIsInstance(mapped, expected, status)
            self.assertNotIn("secret prompt", str(mapped))
        self.assertEqual(map_error(StatusError(429, {"retry-after": "7"})).retry_after, 7.0)

        class APIConnectionError(Exception):
            pass

        self.assertIsInstance(map_error(APIConnectionError()), ProviderOfflineError)

    def test_typed_stream_error_is_mapped(self):
        with self.assertRaises(ProviderAuthenticationError):
            self.run_turn(FakeStream([], None, fail=StatusError(401)))

    def test_validate_key_uses_models_api(self):
        client = FakeClient(models={"claude-haiku-5-5": StatusError(404)})
        provider = AnthropicAgentProvider(client=client)
        results = asyncio.run(provider.validate_key())
        self.assertTrue(results["claude-opus-5-5"]["ok"])
        self.assertEqual(results["claude-opus-5-5"]["max_input_tokens"], 1_000_000)
        self.assertFalse(results["claude-haiku-5-5"]["ok"])
        self.assertEqual(client.calls, [])


class PricingTests(unittest.TestCase):
    def test_tiers_and_fallback_iterations(self):
        usage = AgentUsage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read=1_000_000,
                           cache_write_1h=1_000_000, served_model="claude-opus-5-5")
        self.assertAlmostEqual(cost_usd(usage, "claude-opus-5-5"), 4 + 20 + 0.20 + 8)
        iterated = AgentUsage(iterations=(
            UsageIteration("message", "claude-opus-5-5", input_tokens=1_000_000),
            UsageIteration("fallback_message", "claude-opus-5", input_tokens=1_000_000)))
        self.assertAlmostEqual(cost_usd(iterated, "claude-opus-5-5"), 4 + 5)
        self.assertIsNone(cost_usd(AgentUsage(input_tokens=1), "unknown-model"))
        self.assertAlmostEqual(cost_usd(AgentUsage(input_tokens=200_000), "claude-haiku-5-5"), 0.1)


class ScriptedAndReplayTests(unittest.TestCase):
    def test_scripted_records_requests(self):
        provider = ScriptedAgentProvider([scripted_turn(TOOL_USE), scripted_turn({"type": "text", "text": "done"})])
        events = []
        first = asyncio.run(provider.stream_turn(request(), events.append))
        self.assertEqual(first.stop_reason, AgentStopReason.TOOL_USE)
        self.assertEqual(asyncio.run(provider.stream_turn(request())).text, "done")
        self.assertEqual(len(provider.requests), 2)
        self.assertIn("tool_use_start", [event.kind for event in events])

    def test_replay_fails_closed(self):
        recorded = record_agent_turn(request(), scripted_turn({"type": "text", "text": "sk-ant-abcdefghijk"}))
        self.assertNotIn("sk-ant-abcdefghijk", json.dumps(recorded))
        provider = AgentReplayProvider([recorded])
        self.assertIn("[redacted]", asyncio.run(provider.stream_turn(request())).text)
        with self.assertRaises(ProviderResponseError):
            asyncio.run(provider.stream_turn(request(effort=Effort.HIGH)))
        changed_tools = request(tools=TOOLS[:1])
        with self.assertRaises(ProviderResponseError):
            asyncio.run(provider.stream_turn(changed_tools))


class SecretTests(unittest.TestCase):
    def test_secret_never_rendered(self):
        secret = SecretStr("sk-ant-test-CANARY1234567890")
        for rendered in (repr(secret), str(secret), f"{secret}", json.dumps({"k": str(secret)})):
            self.assertNotIn("CANARY", rendered)
        self.assertEqual(secret.reveal(), "sk-ant-test-CANARY1234567890")
        self.assertFalse(secret.is_admin_key)
        self.assertTrue(SecretStr("sk-ant-admin01-xyz").is_admin_key)

    def test_log_filter_and_child_environment(self):
        record = logging.LogRecord("anthropic", logging.INFO, __file__, 1, "key=%s",
                                   ("sk-ant-test-CANARY1234567890",), None)
        SecretScrubFilter().filter(record)
        self.assertFalse(contains_secret(record.getMessage()))
        self.assertNotIn("ANTHROPIC_API_KEY", child_environment({"ANTHROPIC_API_KEY": "x", "PATH": "/bin"}))

    def test_registry_agent_metadata(self):
        anthropic = DEFAULT_PROVIDER_REGISTRY.get("anthropic")
        self.assertTrue(anthropic.supports_agent_loop)
        self.assertEqual(anthropic.agent_models[0], "claude-opus-5-5")
        gemini = DEFAULT_PROVIDER_REGISTRY.get("gemini")
        self.assertFalse(gemini.supports_agent_loop)
        with self.assertRaises(ValueError):
            gemini.create_agent()


if __name__ == "__main__":
    unittest.main()

"""Tests for explicit sanitized provider record/replay development support."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from application.ai.providers import (
    JsonRecordingStore,
    MockProvider,
    ProviderRecordingError,
    RecordedProvider,
    ReplayProvider,
    provider_request_fingerprint,
)
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ToolDefinition,
    Usage,
)


def _request(*, model="model-a", prompt="confidential prompt api-key-123"):
    frame = MotionFrameImage(
        data=b"private-image-bytes",
        mime_type="image/png",
        time_seconds=1.25,
        variant=ImageVariant.CANDIDATE,
        comparison_id="frame_1",
        label="frame_1",
    )
    tool = ToolDefinition(
        "safe_tool",
        "A deterministic semantic test tool.",
        {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    return ProviderRequest(
        model=model,
        messages=(ProviderMessage(
            MessageRole.USER,
            text=prompt,
            motion_frames=(frame,),
        ),),
        tools=(tool,),
        response_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
        max_output_tokens=200,
    )


class ProviderFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_covers_provider_request_contract(self):
        request = _request()
        first = provider_request_fingerprint("gemini", request)
        self.assertEqual(first, provider_request_fingerprint("Gemini", request))
        self.assertEqual(len(first), 64)

        changes = (
            ("anthropic", request),
            ("gemini", _request(model="model-b")),
            ("gemini", _request(prompt="different context")),
            ("gemini", replace(
                request,
                tools=(ToolDefinition(
                    "safe_tool",
                    "A changed schema version.",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),),
            )),
        )
        for provider, changed in changes:
            with self.subTest(provider=provider, model=changed.model):
                self.assertNotEqual(
                    first,
                    provider_request_fingerprint(provider, changed),
                )


class ProviderRecordReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_only_sanitized_response_and_replays_exact_request(self):
        original = ProviderResponse(
            text='{"status":"user-secret"}',
            usage=Usage(input_tokens=20, output_tokens=4),
        )
        sanitized = ProviderResponse(
            text='{"status":"ok"}',
            usage=Usage(input_tokens=20, output_tokens=4),
        )
        delegate = MockProvider([original], provider_name="gemini")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider-recording.json"
            store = JsonRecordingStore(path)
            recorder = RecordedProvider(
                delegate,
                store,
                sanitizer=lambda _response: sanitized,
                development_mode=True,
            )
            request = _request()

            live_result = await recorder.generate(request)
            replay = ReplayProvider("gemini", delegate.capabilities, store)
            replay_result = await replay.generate(request)

            self.assertIs(live_result, original)
            self.assertEqual(replay_result, sanitized)
            self.assertEqual(len(delegate.requests), 1)
            contents = path.read_text(encoding="utf-8")
            self.assertNotIn("confidential prompt", contents)
            self.assertNotIn("api-key-123", contents)
            self.assertNotIn("private-image-bytes", contents)
            self.assertNotIn("user-secret", contents)
            self.assertIn('\\"status\\":\\"ok\\"', contents)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            with self.assertRaisesRegex(ProviderRecordingError, "no sanitized"):
                await replay.generate(_request(prompt="changed context"))

    async def test_recording_requires_explicit_mode_and_sanitizer_contract(self):
        delegate = MockProvider([ProviderResponse(text="OK")])
        with tempfile.TemporaryDirectory() as directory:
            store = JsonRecordingStore(Path(directory) / "recording.json")
            with self.assertRaisesRegex(ProviderRecordingError, "development_mode"):
                RecordedProvider(
                    delegate,
                    store,
                    sanitizer=lambda response: response,
                )

            recorder = RecordedProvider(
                delegate,
                store,
                sanitizer=lambda _response: "unsafe",
                development_mode=True,
            )
            with self.assertRaisesRegex(ProviderRecordingError, "sanitizer"):
                await recorder.generate(_request())
            self.assertFalse(store.path.exists())

    async def test_corrupt_recording_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.json"
            path.write_text(json.dumps({"prompt": "must not be accepted"}), encoding="utf-8")
            store = JsonRecordingStore(path)
            replay = ReplayProvider(
                "gemini",
                MockProvider([]).capabilities,
                store,
            )

            with self.assertRaisesRegex(ProviderRecordingError, "format is invalid"):
                await replay.generate(_request())


if __name__ == "__main__":
    unittest.main()

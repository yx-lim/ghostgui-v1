#!/usr/bin/env python3
"""Opt-in live contract smoke tests for GhostGUI AI providers.

This script intentionally performs four provider requests. It is not part of the
normal test suite and must never log credentials or raw provider responses.
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
import json
from pathlib import Path
import struct
import sys
from typing import Awaitable, Callable
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.ai.errors import ProviderError
from application.ai.providers.anthropic import (
    DEFAULT_CLAUDE_MODEL,
    AnthropicProvider,
)
from application.ai.providers.base import LLMProvider
from application.ai.providers.gemini import GeminiProvider
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    MotionFrameImage,
    ProviderMessage,
    ProviderRequest,
    ToolCall,
    ToolDefinition,
)


DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
_DEFAULT_MODELS = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "anthropic": DEFAULT_CLAUDE_MODEL,
}
_SDK_DISTRIBUTIONS = {
    "gemini": "google-genai",
    "anthropic": "anthropic",
}


class SmokeCheckError(RuntimeError):
    """A safe, locally generated contract failure suitable for display."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run four opt-in live requests against a GhostGUI AI provider. "
            "This consumes provider quota."
        )
    )
    parser.add_argument("provider", choices=tuple(_DEFAULT_MODELS))
    parser.add_argument(
        "--model",
        help="provider model identifier (defaults to GhostGUI's current model)",
    )
    return parser


def _sdk_version(provider_name: str) -> str:
    distribution = _SDK_DISTRIBUTIONS[provider_name]
    try:
        return f"{distribution} {metadata.version(distribution)}"
    except metadata.PackageNotFoundError:
        return f"{distribution} not installed"


def _provider(provider_name: str) -> LLMProvider:
    if provider_name == "gemini":
        # A smoke check should report the first live result, not silently consume
        # more quota through adapter retries.
        return GeminiProvider(max_attempts=1)
    return AnthropicProvider()


def _request(model: str, prompt: str, **changes) -> ProviderRequest:
    values = {
        "model": model,
        "messages": (ProviderMessage(MessageRole.USER, text=prompt),),
        "max_output_tokens": 128,
    }
    values.update(changes)
    return ProviderRequest(**values)


async def _check_text(provider: LLMProvider, model: str) -> None:
    response = await provider.generate(_request(model, "Reply with exactly OK."))
    if response.text.strip() != "OK":
        raise SmokeCheckError("normalized response was not exactly OK")


async def _check_structured_output(provider: LLMProvider, model: str) -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    response = await provider.generate(
        _request(
            model,
            'Return the requested object with status set to "ok".',
            response_schema=schema,
        )
    )
    try:
        parsed = json.loads(response.text)
    except (TypeError, json.JSONDecodeError) as error:
        raise SmokeCheckError("normalized response was not valid JSON") from error
    if parsed != {"status": "ok"}:
        raise SmokeCheckError("normalized JSON did not match the requested schema")


async def _check_tool_request(provider: LLMProvider, model: str) -> None:
    tool = ToolDefinition(
        name="move_test_target",
        description="Move a harmless test target to a numeric value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    response = await provider.generate(
        _request(
            model,
            "Request move_test_target once with value 1. Do not answer in text.",
            tools=(tool,),
        )
    )
    matching = [call for call in response.tool_calls if call.name == tool.name]
    if len(matching) != 1 or not isinstance(matching[0], ToolCall):
        raise SmokeCheckError("provider did not return one normalized ToolCall")
    if matching[0].arguments.get("value") != 1:
        raise SmokeCheckError("normalized ToolCall did not contain value 1")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload))
    )


def _solid_red_png(width: int = 16, height: int = 16) -> bytes:
    """Generate a dependency-free RGB PNG for the live vision contract check."""

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\xff\x00\x00" * width)
    pixels = zlib.compress(scanline * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


async def _check_vision(provider: LLMProvider, model: str) -> None:
    frame = MotionFrameImage(
        data=_solid_red_png(),
        mime_type="image/png",
        time_seconds=0.0,
        variant=ImageVariant.ORIGINAL,
        comparison_id="provider_smoke_frame",
        label="provider_smoke_frame",
    )
    response = await provider.generate(
        ProviderRequest(
            model=model,
            messages=(
                ProviderMessage(
                    MessageRole.USER,
                    text="What is the image's solid color? Reply with exactly RED.",
                    motion_frames=(frame,),
                ),
            ),
            max_output_tokens=128,
        )
    )
    if response.text.strip().upper() != "RED":
        raise SmokeCheckError("vision response was not exactly RED")


_CHECKS: tuple[
    tuple[str, Callable[[LLMProvider, str], Awaitable[None]]], ...
] = (
    ("Text", _check_text),
    ("Structured output", _check_structured_output),
    ("Tool request", _check_tool_request),
    ("Vision", _check_vision),
)


def _safe_failure(error: BaseException) -> str:
    if isinstance(error, (ProviderError, SmokeCheckError)):
        return str(error)
    return type(error).__name__


async def run_smoke(provider: LLMProvider, model: str) -> bool:
    """Run all checks, reporting failures without leaking raw provider details."""

    passed = True
    for label, check in _CHECKS:
        try:
            await check(provider, model)
        except Exception as error:
            passed = False
            print(f"{label:<18} FAIL - {_safe_failure(error)}")
        else:
            print(f"{label:<18} PASS")
    return passed


async def _run(provider_name: str, model: str) -> bool:
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"SDK version: {_sdk_version(provider_name)}")
    print()
    try:
        provider = _provider(provider_name)
    except Exception as error:
        reason = _safe_failure(error)
        for label, _check in _CHECKS:
            print(f"{label:<18} FAIL - {reason}")
        return False

    try:
        return await run_smoke(provider, model)
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = (args.model or "").strip() or _DEFAULT_MODELS[args.provider]
    return 0 if asyncio.run(_run(args.provider, model)) else 1


if __name__ == "__main__":
    sys.exit(main())

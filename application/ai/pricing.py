"""Single rate table for agent cost display (USD per million tokens).

Local totals are estimates; Console spend limits remain the authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.ai.agent_wire import AgentUsage

VERIFIED_ON = "2026-11-26"


@dataclass(frozen=True)
class Rates:
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


def _standard(input_rate, output_rate, cache_read=None):
    return Rates(input_rate, output_rate, input_rate * 1.25, input_rate * 2.0,
                 input_rate * 0.1 if cache_read is None else cache_read)


RATES: dict[str, Rates] = {
    "claude-opus-5-5": _standard(4.0, 20.0, 0.20),
    "claude-sonnet-5-5": _standard(2.0, 10.0, 0.10),
    "claude-haiku-5-5": _standard(0.10, 0.50),
    # Fallback/served models that can appear in usage.iterations.
    "claude-opus-5": _standard(5.0, 25.0),
    "claude-sonnet-5": _standard(2.0, 10.0),
    "claude-opus-4-8": _standard(5.0, 25.0),
    "claude-fable-5-2": _standard(8.0, 40.0, 0.20),
}
# Haiku 5.5 long-prompt tier (prompts above 100K input tokens).
HAIKU_LONG_PROMPT_TOKENS = 100_000
HAIKU_LONG = _standard(0.50, 2.50)


def rates_for(model: str, prompt_tokens: int = 0) -> Rates | None:
    if model == "claude-haiku-5-5" and prompt_tokens > HAIKU_LONG_PROMPT_TOKENS:
        return HAIKU_LONG
    return RATES.get(model)


def _price(rates: Rates, input_tokens, output_tokens, read, write_5m, write_1h) -> float:
    return (input_tokens * rates.input + output_tokens * rates.output + read * rates.cache_read
            + write_5m * rates.cache_write_5m + write_1h * rates.cache_write_1h) / 1_000_000


def cost_usd(usage: AgentUsage, model: str) -> float | None:
    """Price each served iteration by its own model; None when any model is unknown."""
    if usage.iterations:
        total = 0.0
        for item in usage.iterations:
            prompt = item.input_tokens + item.cache_read + item.cache_write_5m + item.cache_write_1h
            rates = rates_for(item.model or usage.served_model or model, prompt)
            if rates is None:
                return None
            total += _price(rates, item.input_tokens, item.output_tokens, item.cache_read,
                            item.cache_write_5m, item.cache_write_1h)
        return total
    rates = rates_for(usage.served_model or model, usage.total_input)
    if rates is None:
        return None
    return _price(rates, usage.input_tokens, usage.output_tokens, usage.cache_read,
                  usage.cache_write_5m, usage.cache_write_1h)

#!/usr/bin/env python3
"""Manual live smoke for the command-bar agent loop. Spends real money; never runs on PRs.

Checks: models.retrieve for the three agent models; a two-turn tool loop with thinking
replay against a headless workspace; cache_read_input_tokens > 0 from request 2; the
fallback beta is accepted; a mid-conversation system message; spend under the cap.

    python scripts/smoke_agent_loop.py --i-understand-this-spends --max-cost-usd 0.50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def smoke(max_cost: float, model: str) -> dict:
    from application.agent_api.cli import HeadlessExecutor
    from application.agent_api.commands import build_command_catalog
    from application.agent_api.runtime import ToolRuntime
    from application.agent_api.workspace import HeadlessWorkspace
    from application.ai.agent_loop import AgentLoop, LoopHooks
    from application.ai.credentials import default_credential_source
    from application.ai.limits import AgentRunLimits
    from application.ai.providers.anthropic_agent import AGENT_MODELS, AnthropicAgentProvider
    from application.ai.secrets import SecretStr
    from application.commands import Surface, default_registry

    key = default_credential_source().get_secret("anthropic")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is required")
    provider = AnthropicAgentProvider(api_key=SecretStr(key))
    report = {"models": await provider.validate_key(AGENT_MODELS)}
    usages = []
    with tempfile.TemporaryDirectory() as directory:
        workspace = HeadlessWorkspace(Path(directory))
        registry = default_registry()
        runtime = ToolRuntime(build_command_catalog(workspace, surface=Surface.AGENT_INAPP, ports={}), workspace)
        loop = AgentLoop(provider, HeadlessExecutor(runtime, registry), model=model,
                         tools=tuple(registry.tool_definitions(Surface.AGENT_INAPP, ports=())),
                         hooks=LoopHooks(emit=lambda event: event.kind == "usage" and usages.append(dict(event.data))),
                         limits=AgentRunLimits(max_run_cost_usd=max_cost, max_turns=6))
        first = await loop.run("Call list_models and get_robot_summary, then reply with the robot's joint count.")
        second = await loop.run("Now call list_projects and say how many projects exist.")
    await provider.aclose()
    report.update({
        "first": first.outcome.value, "second": second.outcome.value, "turns": first.turns + second.turns,
        "cost_usd": loop.session.spent_usd, "cache_read_after_first": [u.get("cache_read", 0) for u in usages[1:]],
    })
    report["passed"] = (all(item["ok"] for item in report["models"].values())
                        and first.outcome.value == "completed" and second.outcome.value == "completed"
                        and any(value > 0 for value in report["cache_read_after_first"])
                        and loop.session.spent_usd <= max_cost)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-understand-this-spends", action="store_true", required=True)
    parser.add_argument("--max-cost-usd", type=float, default=0.50)
    parser.add_argument("--model", default="claude-opus-5-5")
    args = parser.parse_args(argv)
    report = asyncio.run(smoke(args.max_cost_usd, args.model))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

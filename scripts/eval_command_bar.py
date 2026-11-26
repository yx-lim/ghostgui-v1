#!/usr/bin/env python3
"""Run the command-bar app-control scenarios and report pass rate and cost.

Replay mode (default, free) uses the recorded fixtures. Live mode spends real money and
reports separately:  python scripts/eval_command_bar.py --live --i-understand-this-spends
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--i-understand-this-spends", action="store_true")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--report", type=Path, help="Write the JSON report here")
    args = parser.parse_args(argv)
    if args.live and not args.i_understand_this_spends:
        parser.error("--live needs --i-understand-this-spends")

    import agent_scenarios as scenarios
    from application.ai.providers.replay import AgentReplayProvider

    recorded = json.loads((ROOT / "tests/fixtures/agent_replays/scenarios.json").read_text())["scenarios"]
    provider = None
    if args.live:
        from application.ai.credentials import default_credential_source
        from application.ai.providers.anthropic_agent import AnthropicAgentProvider
        from application.ai.secrets import SecretStr
        key = default_credential_source().get_secret("anthropic")
        if not key:
            parser.error("ANTHROPIC_API_KEY is required for --live")
        provider = AnthropicAgentProvider(api_key=SecretStr(key))
        scenarios.MODEL = args.model

    rows, spent = [], 0.0
    for scenario in scenarios.SCENARIOS:
        if args.live and scenario.name == "refusal_handled":
            continue  # cannot be induced deliberately
        if spent >= args.max_cost_usd:
            rows.append({"scenario": scenario.name, "skipped": "spend cap reached"})
            continue
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as directory:
            source = provider if args.live else AgentReplayProvider(recorded[scenario.name])
            result, workspace = scenarios.run(scenario, source, Path(directory))
            ok = scenarios.passed(scenario, result, workspace)
        spent += result.cost_usd or 0.0
        rows.append({"scenario": scenario.name, "passed": ok, "outcome": result.outcome.value,
                     "turns": result.turns, "actions": result.actions,
                     "cost_usd": round(result.cost_usd or 0.0, 4), "seconds": round(time.monotonic() - started, 2)})
    ran = [row for row in rows if "passed" in row]
    report = {"mode": "live" if args.live else "replay", "model": args.model if args.live else "recorded",
              "pass_rate": round(sum(row["passed"] for row in ran) / max(1, len(ran)), 3),
              "cost_usd": round(spent, 4), "scenarios": rows}
    encoded = json.dumps(report, indent=1, sort_keys=True)
    if args.report:
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

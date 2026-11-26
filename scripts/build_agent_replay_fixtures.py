#!/usr/bin/env python3
"""Dev-only: rebuild synthetic, sanitised agent replay fixtures from tests/agent_scenarios.py.

Run deliberately after a registry or prompt change (the replay test fails closed until then):
    python scripts/build_agent_replay_fixtures.py --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
FIXTURE = ROOT / "tests" / "fixtures" / "agent_replays" / "scenarios.json"


class Recording:
    def __init__(self, inner):
        self.inner, self.turns = inner, []

    async def stream_turn(self, request, on_event=None, cancellation_token=None):
        from application.ai.providers.replay import record_agent_turn
        response = await self.inner.stream_turn(request, on_event, cancellation_token)
        self.turns.append(record_agent_turn(request, response))
        return response


def build() -> dict:
    import agent_scenarios as scenarios

    output = {"note": "Synthetic scripted recordings; regenerate with scripts/build_agent_replay_fixtures.py",
              "scenarios": {}}
    for scenario in scenarios.SCENARIOS:
        recorder = Recording(scenarios.scripted_provider(scenario))
        with tempfile.TemporaryDirectory() as directory:
            result, workspace = scenarios.run(scenario, recorder, Path(directory))
            if not scenarios.passed(scenario, result, workspace):
                raise SystemExit(f"scenario {scenario.name} does not pass with its script: {result}")
        output["scenarios"][scenario.name] = recorder.turns
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Write the fixture file")
    args = parser.parse_args(argv)
    data = build()
    encoded = json.dumps(data, indent=1, sort_keys=True) + "\n"
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(encoded, encoding="utf-8")
        print(f"wrote {FIXTURE.relative_to(ROOT)} ({len(data['scenarios'])} scenarios)")
    else:
        print(encoded[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

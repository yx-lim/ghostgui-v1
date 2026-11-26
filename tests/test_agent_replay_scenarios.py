"""M5: app-control scenarios replayed from sanitised fixtures (fail closed on any request drift)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_scenarios  # noqa: E402
from application.ai.providers.replay import AgentReplayProvider  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "agent_replays" / "scenarios.json"


class ReplayScenarioTests(unittest.TestCase):
    def test_every_scenario_passes_on_replay(self):
        recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]
        self.assertEqual(sorted(recorded), sorted(s.name for s in agent_scenarios.SCENARIOS),
                         "rebuild fixtures with scripts/build_agent_replay_fixtures.py --write")
        for scenario in agent_scenarios.SCENARIOS:
            with self.subTest(scenario.name), tempfile.TemporaryDirectory() as directory:
                provider = AgentReplayProvider(recorded[scenario.name])
                result, workspace = agent_scenarios.run(scenario, provider, Path(directory))
                self.assertTrue(agent_scenarios.passed(scenario, result, workspace),
                                f"{scenario.name}: {result.outcome} {result.message} "
                                "(if the registry or prompt changed, rebuild the replay fixtures)")


if __name__ == "__main__":
    unittest.main()

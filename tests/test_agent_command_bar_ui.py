"""M2: offscreen command bar runs with a scripted provider (no network)."""
import asyncio
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QPushButton

from application.ai.agent_events import RunOutcome
from application.ai.errors import ProviderCancelledError
from application.ai.providers.mock import ScriptedAgentProvider, scripted_turn
from application.ai.secrets import SecretStr
from gui.agent_command_bar import palette_matches
from gui.main_window import RobotGuiMainWindow

CANARY = "sk-ant-test-CANARY" + "q" * 24


def use(call_id, name, arguments=None):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments or {}}


def text(value):
    return {"type": "text", "text": value}


class HangingProvider:
    """Streams until cancelled, like a long model turn."""
    provider_name = "hang"

    def __init__(self):
        self.started = False

    async def stream_turn(self, request, on_event=None, cancellation_token=None):
        self.started = True
        while not cancellation_token.cancellation_requested:
            await asyncio.sleep(0.01)
        raise ProviderCancelledError("cancelled")

    async def aclose(self):
        return None


class CommandBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        environment = patch.dict(os.environ, {
            "GHOSTGUI_CONFIG_DIR": str(self.root / "config"), "GHOSTGUI_CACHE_DIR": str(self.root / "cache"),
            "GHOSTGUI_PROJECTS_DIR": str(self.root / "projects"), "GHOSTGUI_USER_DATA_DIR": str(self.root / "data"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.window = RobotGuiMainWindow()
        self.bar = self.window.agent_command_bar
        self.controller = self.bar.controller
        self.results = []
        self.controller.run_finished.connect(self.results.append)
        self.addCleanup(self._close)

    def _close(self):
        self.bar.shutdown()
        self.window.agent_bridge_controller.shutdown()
        self.window.set_project_dirty(False)
        self.window.close()
        self.app.processEvents()

    def wait(self, condition, timeout=25):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(condition(), "timed out")

    def run_prompt(self, prompt, provider):
        self.controller.set_provider(provider)
        self.bar.prompt.setText(prompt)
        self.bar.submit()
        self.wait(lambda: self.results)
        return self.results.pop()

    def test_ui_commands_change_the_app(self):
        provider = ScriptedAgentProvider([
            scripted_turn(use("t1", "go_to_time", {"time_s": 1.0}), use("t2", "set_view", {"left_sidebar": False}),
                          use("t3", "get_app_state", {"sections": ["view"]})),
            scripted_turn(text("Moved to 1 s and hid the sidebar.")),
        ])
        result = self.run_prompt("go to 1 s and hide the left sidebar", provider)
        self.assertIs(result.outcome, RunOutcome.COMPLETED, result.message)
        self.assertEqual(result.actions, 3)
        self.assertAlmostEqual(self.window.viewer_3d.get_current_time(), 1.0, places=3)
        self.assertTrue(self.window._left_sidebar_collapsed)
        self.assertGreaterEqual(len(self.bar.transcript.chips), 3)
        self.assertTrue(all(chip.state == "ok" for chip in self.bar.transcript.chips.values()))
        digest = provider.requests[0].messages[0]["content"][-1]["text"]
        self.assertIn("<ghostgui_state>", digest)
        self.assertEqual(len(self.controller.action_log), 3)

    def test_motion_preview_accept_makes_one_undo_entry(self):
        def author(_request):
            # Runs after new_project: anchor on the project's own home pose (read-only access).
            home = [float(value) for value in self.window.document.qpos_timeline.sample_state(0.0)]
            middle = list(home)
            middle[-1] += 0.05
            return scripted_turn(use("t1", "author_keyframes", {"duration_s": 0.4, "keyframes": [
                {"time_seconds": 0.0, "qpos": home}, {"time_seconds": 0.2, "qpos": middle},
                {"time_seconds": 0.4, "qpos": home}]}))

        provider = ScriptedAgentProvider([
            scripted_turn(use("t0", "new_project", {"name": "motion", "duration_s": 0.4})),
            author,
            scripted_turn(text("Preview ready.")),
        ])
        before = len(self.window.undo_stack)
        result = self.run_prompt("make a small motion", provider)
        self.assertIs(result.outcome, RunOutcome.COMPLETED, result.message)
        self.assertTrue(self.controller.bridge.has_candidate, str(provider.requests[2].messages[-1])[:600])
        self.assertTrue(result.candidate_touched)
        future = self.controller.accept_candidate()
        self.wait(future.done)
        self.assertTrue(future.result()["ok"], future.result())
        self.wait(lambda: not self.controller.bridge.has_candidate)
        self.assertEqual(len(self.window.undo_stack), before + 1)

    def test_stop_mid_stream(self):
        provider = HangingProvider()
        self.controller.set_provider(provider)
        self.bar.prompt.setText("do something long")
        self.bar.submit()
        self.wait(lambda: provider.started)
        self.assertTrue(self.bar.pill.isVisibleTo(self.window) or self.controller.running)
        self.bar._escape()
        self.wait(lambda: self.results)
        self.assertIs(self.results.pop().outcome, RunOutcome.STOPPED)
        self.assertFalse(self.controller.running)
        self.assertEqual(self.bar.send.text(), "Send")

    def test_confirmation_card_deny(self):
        self.window.undo_stack.append(type("Entry", (), {"description": "Edit", "snapshot": None})())
        provider = ScriptedAgentProvider([scripted_turn(use("t1", "history_undo")), scripted_turn(text("ok"))])
        self.controller.set_provider(provider)
        self.bar.prompt.setText("undo that")
        self.bar.submit()
        self.wait(lambda: self.window.findChild(QPushButton, "agentApprovalDeny") is not None)
        self.window.findChild(QPushButton, "agentApprovalDeny").click()
        self.wait(lambda: self.results)
        self.window.undo_stack.pop()
        self.assertIn("user_declined", str(provider.requests[1].messages[-1]))

    def test_offline_palette(self):
        self.controller.provider_factory = lambda: None
        self.controller.set_provider(None)
        self.assertEqual(palette_matches("/recent actions")[0].tool, "get_recent_actions")
        self.bar.prompt.setText("/recent actions")
        self.bar.submit()
        self.wait(lambda: any(chip.state == "ok" for chip in self.bar.transcript.chips.values()))

    def test_key_in_prompt_is_blocked_and_canary_never_written(self):
        self.bar.session_key = SecretStr(CANARY)
        self.assertIsNotNone(self.bar._default_provider())
        self.bar.prompt.setText(f"my key is {CANARY}")
        self.bar.submit()
        self.assertEqual(self.controller.provider, None)  # nothing was started
        provider = ScriptedAgentProvider([scripted_turn(use("t1", "go_to_time", {"time_s": 0.5})),
                                          scripted_turn(text("ok"))])
        self.run_prompt("go to half a second", provider)
        self.window.save_ui_settings()
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(CANARY.encode(), path.read_bytes(), path)
        self.assertNotIn(CANARY, repr(self.bar.session_key))
        self.assertNotIn(CANARY, str(self.controller.action_log.tail()))


if __name__ == "__main__":
    unittest.main()

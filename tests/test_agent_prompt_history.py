"""Command-bar prompt history (Qt-free)."""
from pathlib import Path
import tempfile
import unittest

from gui.agent_prompt_history import MAX_ENTRIES, PromptHistory


class PromptHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "agent" / "prompt_history.json"

    def test_recall_keeps_draft_and_skips_repeats(self):
        history = PromptHistory(self.path)
        for prompt in ("a", "b", "b", "  "):
            history.add(prompt)
        self.assertEqual(history.entries, ["a", "b"])
        self.assertEqual(history.previous("typing"), "b")
        self.assertEqual(history.previous("ignored"), "a")
        self.assertIsNone(history.previous("x"))
        self.assertEqual(history.next(), "b")
        self.assertEqual(history.next(), "typing")
        self.assertIsNone(history.next())

    def test_persists_bounded_and_reloads(self):
        history = PromptHistory(self.path)
        for index in range(MAX_ENTRIES + 5):
            history.add(f"prompt {index}")
        reloaded = PromptHistory(self.path)
        self.assertEqual(len(reloaded.entries), MAX_ENTRIES)
        self.assertEqual(reloaded.entries[-1], f"prompt {MAX_ENTRIES + 4}")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_secrets_never_stored(self):
        history = PromptHistory(self.path)
        history.add("use sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX please")
        self.assertEqual(history.entries, [])
        self.assertFalse(self.path.exists())

    def test_disabled_does_not_write_and_corrupt_file_is_ignored(self):
        PromptHistory(self.path, enabled=False).add("hello")
        self.assertFalse(self.path.exists())
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json")
        self.assertEqual(PromptHistory(self.path).entries, [])


if __name__ == "__main__":
    unittest.main()

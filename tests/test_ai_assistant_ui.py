"""Focused Motion Assistant widget contracts; skipped without PySide6."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QLineEdit
except ImportError:
    QApplication = None


@unittest.skipUnless(QApplication is not None, "PySide6 unavailable")
class AIAssistantPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.panels.ai_assistant_panel import AIAssistantPanel

        self.panel = AIAssistantPanel()
        self.addCleanup(self.panel.close)

    def test_running_and_staged_states_expose_only_valid_actions(self):
        from gui.panels.ai_assistant_panel import AIAssistantPanelState

        self.assertEqual(self.panel.state, AIAssistantPanelState.READY)
        self.assertFalse(self.panel.accept_button.isEnabled())
        self.panel.begin_request()
        self.assertEqual(self.panel.state, AIAssistantPanelState.RUNNING)
        self.assertFalse(self.panel.cancel_button.isHidden())
        self.assertFalse(self.panel.prompt_input.isEnabled())

        self.panel.show_proposal("Done", ("Modified 2 Keyframes",))
        self.assertEqual(self.panel.state, AIAssistantPanelState.STAGED)
        self.assertTrue(self.panel.accept_button.isEnabled())
        self.assertEqual(self.panel.proposal_list.item(0).text(), "Modified 2 Keyframes")

    def test_apply_and_refine_emit_trimmed_instructions(self):
        submitted = []
        refined = []
        self.panel.submit_requested.connect(submitted.append)
        self.panel.refine_requested.connect(refined.append)
        self.panel.prompt_input.setPlainText("  lower the pelvis  ")
        self.panel.submit_button.click()
        self.assertEqual(submitted, ["lower the pelvis"])

        self.panel.show_proposal("Done", ("Moved pelvis",))
        self.panel.prompt_input.setPlainText("  make it subtler  ")
        self.panel.submit_button.click()
        self.assertEqual(refined, ["make it subtler"])

    def test_error_does_not_enable_accept_without_a_staged_session(self):
        from gui.panels.ai_assistant_panel import AIAssistantPanelState

        self.panel.show_error("No API key")
        self.assertEqual(self.panel.state, AIAssistantPanelState.ERROR)
        self.assertFalse(self.panel.accept_button.isEnabled())


@unittest.skipUnless(QApplication is not None, "PySide6 unavailable")
class AISettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_key_is_password_masked_and_returned_only_as_ephemeral_value(self):
        from application.ai.providers.gemini import DEFAULT_GEMINI_CAPABILITIES
        from gui.ai_settings_dialog import AISettingsDialog

        dialog = AISettingsDialog(capabilities=DEFAULT_GEMINI_CAPABILITIES)
        self.addCleanup(dialog.close)
        dialog.api_key_input.setText("secret")
        values = dialog.values()
        self.assertEqual(dialog.api_key_input.echoMode(), QLineEdit.EchoMode.Password)
        self.assertEqual(values.api_key, "secret")
        self.assertEqual(values.provider, "gemini")
        self.assertTrue(values.store_securely)


if __name__ == "__main__":
    unittest.main()

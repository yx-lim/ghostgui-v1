"""Focused Motion Assistant widget contracts; skipped without PySide6."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
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

        previews = []
        self.panel.preview_requested.connect(lambda: previews.append(True))
        self.assertEqual(self.panel.state, AIAssistantPanelState.READY)
        self.assertFalse(self.panel.accept_button.isEnabled())
        self.assertEqual(self.panel.preview_button.text(), "Preview candidate")
        self.panel.begin_request()
        self.assertEqual(self.panel.state, AIAssistantPanelState.RUNNING)
        self.assertFalse(self.panel.cancel_button.isHidden())
        self.assertFalse(self.panel.prompt_input.isEnabled())

        self.panel.show_proposal(
            "Done",
            ("Modified 2 Keyframes",),
            accept_permitted=True,
        )
        self.assertEqual(self.panel.state, AIAssistantPanelState.STAGED)
        self.assertTrue(self.panel.accept_button.isEnabled())
        self.assertTrue(self.panel.visual_refine_button.isEnabled())
        self.assertTrue(self.panel.visual_verify_button.isEnabled())
        self.assertEqual(self.panel.proposal_list.item(0).text(), "Modified 2 Keyframes")
        self.panel.set_accept_permitted(False)
        self.assertFalse(self.panel.accept_button.isEnabled())
        self.panel.preview_button.click()
        self.assertEqual(previews, [True])

    def test_worker_progress_updates_only_an_active_request(self):
        from application.ai.progress import AIProgressEvent, AIProgressStage

        self.panel.begin_request()
        self.panel.progress_received.emit(AIProgressEvent(
            AIProgressStage.LOCAL_OPERATION,
            operation_index=2,
            operation_count=5,
        ))
        self.app.processEvents()
        self.assertEqual(
            self.panel.response_label.text(),
            "Applying motion changes",
        )
        activity_text = tuple(
            entry.text
            for entry in self.panel.transcript.entries
            if entry.kind.value == "activity"
        )
        self.assertEqual(
            activity_text,
            ("Reading the current motion", "Applying motion changes"),
        )

        self.panel.show_proposal("Done", ("Modified motion",))
        self.panel.progress_received.emit(
            AIProgressEvent(AIProgressStage.VALIDATION)
        )
        self.app.processEvents()
        self.assertEqual(self.panel.response_label.text(), "Done")

    def test_apply_and_refine_emit_trimmed_instructions(self):
        submitted = []
        refined = []
        self.panel.submit_requested.connect(submitted.append)
        self.panel.refine_requested.connect(refined.append)
        self.panel.prompt_input.setPlainText("  lower the pelvis  ")
        self.panel.submit_button.click()
        self.assertEqual(submitted, ["lower the pelvis"])

        self.panel.show_proposal(
            "Done",
            ("Moved pelvis",),
            accept_permitted=True,
        )
        self.panel.prompt_input.setPlainText("  make it subtler  ")
        self.panel.submit_button.click()
        self.assertEqual(refined, ["make it subtler"])

    def test_enter_sends_and_shift_enter_inserts_a_newline(self):
        submitted = []
        self.panel.submit_requested.connect(submitted.append)
        self.panel.prompt_input.setPlainText("walk forward")

        QTest.keyClick(self.panel.prompt_input, Qt.Key.Key_Return)

        self.assertEqual(submitted, ["walk forward"])
        self.assertEqual(self.panel.transcript.entries[-1].text, "walk forward")
        self.assertEqual(self.panel.transcript.entries[-1].kind.value, "user")

        self.panel.prompt_input.clear()
        QTest.keyClicks(self.panel.prompt_input, "line one")
        QTest.keyClick(
            self.panel.prompt_input,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ShiftModifier,
        )
        QTest.keyClicks(self.panel.prompt_input, "line two")
        self.assertEqual(
            self.panel.prompt_input.toPlainText(),
            "line one\nline two",
        )

    def test_staged_send_refines_without_a_visible_refine_button(self):
        refined = []
        self.panel.refine_requested.connect(refined.append)
        self.panel.show_proposal("Done", ("Moved pelvis",))
        self.assertTrue(self.panel.refine_button.isHidden())

        self.panel.prompt_input.setPlainText("keep both feet planted")
        self.panel.submit_button.click()

        self.assertEqual(refined, ["keep both feet planted"])
        self.assertEqual(
            self.panel.transcript.entries[-1].text,
            "keep both feet planted",
        )

    def test_result_card_owns_accept_and_discard_actions(self):
        accepted = []
        discarded = []
        self.panel.accept_requested.connect(lambda: accepted.append(True))
        self.panel.reject_requested.connect(lambda: discarded.append(True))

        self.panel.show_proposal(
            "I created the motion.",
            ("generate motion", "Warning: minor contact concession"),
            accept_permitted=True,
            result_summary="5.0 s · 8 Keyframes · 1 warning",
        )
        card = self.panel._active_result_card
        self.assertIsNotNone(card)
        self.assertEqual(card.summary_label.text(), "5.0 s · 8 Keyframes · 1 warning")
        card.accept_button.click()
        card.discard_button.click()
        self.assertEqual(accepted, [True])
        self.assertEqual(discarded, [True])

    def test_completed_conversation_is_retained_until_next_user_turn(self):
        self.panel.add_user_message("make a squat")
        self.panel.show_proposal("Done", ("generate motion",))
        self.panel.complete_session("Motion accepted.", accepted=True)
        self.assertGreater(len(self.panel.transcript.entries), 2)

        self.panel.add_user_message("make a wave")

        self.assertEqual(len(self.panel.transcript.entries), 1)
        self.assertEqual(self.panel.transcript.entries[0].text, "make a wave")

    def test_error_summary_keeps_technical_details_expandable(self):
        self.panel.show_error(
            "motion planning request timed out; provider deadline 60 s; request id 7"
        )
        widget = self.panel.message_widgets[-1]
        self.assertIn("timed out", widget.body_label.text())
        self.assertTrue(widget.details_button.isVisible() or not self.panel.isVisible())
        self.assertIn("request id 7", widget.details_label.text())

    def test_error_does_not_enable_accept_without_a_staged_session(self):
        from gui.panels.ai_assistant_panel import AIAssistantPanelState

        self.panel.show_error("No API key")
        self.assertEqual(self.panel.state, AIAssistantPanelState.ERROR)
        self.assertFalse(self.panel.accept_button.isEnabled())

    def test_network_error_and_cancellation_preserve_previous_messages(self):
        self.panel.add_user_message("create a short walk")
        before = tuple(self.panel.transcript.entries)

        self.panel.show_error("network unavailable; connection refused")
        self.assertEqual(self.panel.transcript.entries[:len(before)], before)
        self.assertIn("network unavailable", self.panel.transcript.entries[-1].text)

        self.panel.begin_request()
        self.panel.show_cancelled()
        texts = tuple(entry.text for entry in self.panel.transcript.entries)
        self.assertIn("Request cancelled. The committed motion is unchanged.", texts)
        self.assertTrue(self.panel.prompt_input.isEnabled())

    def test_multiple_refinement_turns_remain_visible(self):
        self.panel.add_user_message("make a knee push-up")
        self.panel.show_proposal("Created it.", ("generate motion",))
        self.panel.add_user_message("move the knees forward")
        self.panel.begin_request(refinement=True)
        self.panel.show_proposal("Moved the knees.", ("patch motion",))
        self.panel.add_user_message("reduce the arm swing")
        self.panel.begin_request(refinement=True)
        self.panel.show_proposal("Reduced it.", ("patch motion",))

        visible_text = tuple(entry.text for entry in self.panel.transcript.entries)
        for expected in (
            "make a knee push-up",
            "move the knees forward",
            "reduce the arm swing",
            "Created it.",
            "Moved the knees.",
            "Reduced it.",
        ):
            self.assertIn(expected, visible_text)

    def test_critique_uses_default_prompt_and_does_not_enable_accept(self):
        critiques = []
        self.panel.critique_requested.connect(critiques.append)

        self.panel.critique_button.click()
        self.assertEqual(critiques, ["What is visually wrong with this motion?"])

        self.panel.show_critique(
            "Two visible issues.",
            ("Around 2.10 s: right foot: It appears to slide.",),
        )
        self.assertFalse(self.panel.accept_button.isEnabled())
        self.assertEqual(self.panel.proposal_heading.text(), "Visual observations")

    def test_visual_refine_is_available_only_for_a_staged_working_copy(self):
        refinements = []
        self.panel.visual_refine_requested.connect(refinements.append)
        self.assertFalse(self.panel.visual_refine_button.isEnabled())

        self.panel.show_proposal("Done", ("Moved pelvis",))
        self.panel.prompt_input.setPlainText("  keep the feet planted  ")
        self.panel.visual_refine_button.click()

        self.assertEqual(refinements, ["keep the feet planted"])

    def test_visual_verification_is_an_explicit_staged_action(self):
        verifications = []
        self.panel.visual_verify_requested.connect(verifications.append)
        self.assertFalse(self.panel.visual_verify_button.isEnabled())

        self.panel.show_proposal(
            "Done",
            ("Moved pelvis",),
            accept_permitted=True,
        )
        self.panel.prompt_input.setPlainText("  check the original goal  ")
        self.panel.visual_verify_button.click()

        self.assertEqual(verifications, ["check the original goal"])
        self.panel.show_verification(
            "The candidate is improved.",
            ("Around 1.80 s: torso is more upright",),
        )
        self.assertTrue(self.panel.accept_button.isEnabled())
        self.assertEqual(self.panel.proposal_heading.text(), "Visual verification")


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

    def test_switching_to_anthropic_updates_model_and_capabilities(self):
        from application.ai.providers.anthropic import DEFAULT_ANTHROPIC_CAPABILITIES
        from application.ai.providers.gemini import DEFAULT_GEMINI_CAPABILITIES
        from gui.ai_settings_dialog import AISettingsDialog

        dialog = AISettingsDialog(
            capabilities=DEFAULT_GEMINI_CAPABILITIES,
            provider_capabilities={
                "gemini": DEFAULT_GEMINI_CAPABILITIES,
                "anthropic": DEFAULT_ANTHROPIC_CAPABILITIES,
            },
        )
        self.addCleanup(dialog.close)
        dialog.provider_box.setCurrentIndex(
            dialog.provider_box.findData("anthropic")
        )

        self.assertEqual(dialog.values().provider, "anthropic")
        self.assertEqual(dialog.values().model, "claude-sonnet-5")
        self.assertIn("Vision", dialog.capabilities_label.text())

    def test_settings_enumerates_a_new_provider_from_the_registry(self):
        from application.ai.provider_registry import (
            ProviderRegistration,
            ProviderRegistry,
        )
        from application.ai.schemas import ProviderCapabilities
        from gui.ai_settings_dialog import AISettingsDialog

        registration = ProviderRegistration(
            name="future",
            display_name="Future AI",
            factory=lambda *, api_key=None: object(),
            models=("future-default", "future-fast"),
            capabilities=ProviderCapabilities(
                supports_tools=True,
                supports_vision=False,
            ),
            credential_identifier="future-key",
            environment_variables=("FUTURE_API_KEY",),
            sdk_distribution="future-sdk",
        )
        registry = ProviderRegistry((registration,), default_name="future")

        dialog = AISettingsDialog(provider_registry=registry, provider="future")
        self.addCleanup(dialog.close)

        self.assertEqual(dialog.provider_box.count(), 1)
        self.assertEqual(dialog.provider_box.currentText(), "Future AI")
        self.assertEqual(dialog.values().model, "future-default")
        self.assertIn("Tool Calling", dialog.capabilities_label.text())


if __name__ == "__main__":
    unittest.main()

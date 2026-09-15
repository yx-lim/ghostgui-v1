"""Controller contracts specific to the transcript-first Motion Assistant."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from gui.ai_assistant_controller import AIAssistantController
except ImportError:
    AIAssistantController = None

from application.ai.edit_session import AIEditSessionState


@unittest.skipUnless(AIAssistantController is not None, "PySide6 unavailable")
class AIAssistantControllerChatTests(unittest.TestCase):
    def test_successful_candidate_is_presented_then_previewed_automatically(self):
        controller = AIAssistantController.__new__(AIAssistantController)
        controller.active_handle = object()
        controller.panel = Mock()
        controller.session = SimpleNamespace(
            state=AIEditSessionState.STAGED,
            can_accept=True,
        )
        controller._pending_refinement = ""
        controller._trajectory_conversation = None
        controller.preview = Mock()
        result = SimpleNamespace(
            text="I created a short walk.",
            proposal_lines=("generate motion", "Warning: minor foot drift"),
            execution=SimpleNamespace(operations=(
                SimpleNamespace(output={
                    "duration_seconds": 4.0,
                    "sparse_keyframes": 7,
                }),
            )),
        )

        controller._request_succeeded(result)

        controller.panel.show_proposal.assert_called_once_with(
            "I created a short walk.",
            result.proposal_lines,
            accept_permitted=True,
            result_summary="4.0 s · 7 Keyframes · 1 warning",
        )
        controller.preview.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

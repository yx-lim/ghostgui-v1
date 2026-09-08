"""Tests for compact multi-turn motion intent."""

import unittest

from application.ai.trajectory_conversation import TrajectoryConversation


class TrajectoryConversationTests(unittest.TestCase):
    def test_context_keeps_goal_and_only_recent_refinements(self):
        conversation = TrajectoryConversation("Create a worm.")
        for instruction in (
            "Keep the legs.",
            "Plant both hands.",
            "Make the arms straighter.",
            "Make that section faster.",
        ):
            conversation.record_refinement(instruction)

        self.assertEqual(conversation.refinements, (
            "Plant both hands.",
            "Make the arms straighter.",
            "Make that section faster.",
        ))
        self.assertEqual(conversation.to_context(), {
            "original_goal": "Create a worm.",
            "recent_refinements": [
                "Plant both hands.",
                "Make the arms straighter.",
                "Make that section faster.",
            ],
            "source": "current_staged_candidate",
        })

    def test_empty_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TrajectoryConversation(" ")


if __name__ == "__main__":
    unittest.main()

"""Visible Motion Assistant transcript state contracts."""

from __future__ import annotations

import unittest

from application.ai.chat_transcript import ChatEntryKind, MotionChatTranscript


class MotionChatTranscriptTests(unittest.TestCase):
    def test_turn_entries_are_ordered_and_details_are_normalized(self):
        transcript = MotionChatTranscript()
        user = transcript.start_turn("  create a squat  ")
        activity = transcript.append(ChatEntryKind.ACTIVITY, "Planning")
        result = transcript.append(
            ChatEntryKind.RESULT,
            "Candidate ready",
            details=("  5.0 s  ", "", "1 warning"),
        )

        self.assertEqual(user.turn_id, 1)
        self.assertEqual(activity.turn_id, 1)
        self.assertEqual(result.details, ("5.0 s", "1 warning"))
        self.assertEqual(
            tuple(entry.kind for entry in transcript.entries),
            (
                ChatEntryKind.USER,
                ChatEntryKind.ACTIVITY,
                ChatEntryKind.RESULT,
            ),
        )

    def test_clear_starts_a_fresh_visible_conversation(self):
        transcript = MotionChatTranscript()
        transcript.start_turn("first")
        transcript.append(ChatEntryKind.ASSISTANT, "done")

        transcript.clear()
        next_user = transcript.start_turn("second")

        self.assertEqual(transcript.entries, (next_user,))
        self.assertEqual(next_user.turn_id, 1)

    def test_visible_transcript_is_not_provider_conversation_context(self):
        from application.ai.trajectory_conversation import TrajectoryConversation

        transcript = MotionChatTranscript()
        transcript.start_turn("make a squat")
        transcript.append(ChatEntryKind.ACTIVITY, "Validating the candidate")
        provider_context = TrajectoryConversation("make a squat")

        self.assertNotIn("Validating", provider_context.to_context())


if __name__ == "__main__":
    unittest.main()

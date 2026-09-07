"""Contracts for provider-neutral AI workflow progress."""

from __future__ import annotations

import unittest

from application.ai.progress import AIProgressEvent, AIProgressStage


class AIProgressEventTests(unittest.TestCase):
    def test_messages_cover_current_and_future_streaming_stages(self):
        cases = (
            (AIProgressEvent(AIProgressStage.PLANNING_STARTED), "Planning motion…"),
            (
                AIProgressEvent(
                    AIProgressStage.STRUCTURED_PLAN_COMPLETED,
                    operation_count=2,
                ),
                "Plan ready: 2 operations.",
            ),
            (
                AIProgressEvent(
                    AIProgressStage.LOCAL_OPERATION,
                    operation_index=1,
                    operation_count=2,
                ),
                "Executing operation 1/2…",
            ),
            (AIProgressEvent(AIProgressStage.VALIDATION), "Validating candidate…"),
            (AIProgressEvent(AIProgressStage.DONE), "Motion candidate ready."),
            (
                AIProgressEvent(AIProgressStage.TEXT_DELTA, text_delta="Planning"),
                "Planning",
            ),
        )

        for event, expected in cases:
            with self.subTest(stage=event.stage):
                self.assertEqual(event.message, expected)

    def test_operation_progress_requires_a_valid_one_based_index(self):
        for index, count in ((None, 2), (0, 2), (3, 2), (1, 0)):
            with self.subTest(index=index, count=count), self.assertRaises(ValueError):
                AIProgressEvent(
                    AIProgressStage.LOCAL_OPERATION,
                    operation_index=index,
                    operation_count=count,
                )

    def test_stage_specific_payloads_cannot_leak_into_other_events(self):
        with self.assertRaises(ValueError):
            AIProgressEvent(AIProgressStage.DONE, operation_count=1)
        with self.assertRaises(ValueError):
            AIProgressEvent(AIProgressStage.DONE, text_delta="unexpected")
        with self.assertRaises(ValueError):
            AIProgressEvent(AIProgressStage.TEXT_DELTA)


if __name__ == "__main__":
    unittest.main()

"""Integration tests for one-shot compact planning and local execution."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from application.ai.edit_session import AIEditSession
from application.ai.diagnostics import MotionAssistantDiagnostics
from application.ai.metadata import InMemoryMotionMetadataStore, MotionMetadataService, TimestampMotionIdentityResolver
from application.ai.motion_services import MotionValidationReport
from application.ai.motion_state import ReplaceMotionState, capture_motion_state
from application.ai.providers import MockProvider, RequestCountingProvider
from application.ai.schemas import ImageVariant, ProviderCapabilities, ProviderResponse
from application.ai.trajectory_workflow import CompactMotionWorkflow
from application.project_document import ProjectDocument
from application.editor_controller import EditorController
from application.history import HistoryStack
from gui.ai_assistant_controller import _capture_automatic_visual_context
from tests.test_ai_trajectory_operations import Timeline, _HoldMotion


class CompactMotionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_repair_persists_both_diagnostic_attempts(self):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        valid = {
            "mode": "edit",
            "summary": "Raise the robot.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "translation_m": [0, 0, 0.05],
                }),
            }],
        }
        provider = RequestCountingProvider(MockProvider([
            ProviderResponse(text="malformed"),
            ProviderResponse(text=json.dumps(valid)),
        ]))
        motion = _HoldMotion()
        motion.adapter.free_joints_by_body = {
            0: type("Joint", (), {"qpos_address": 0})()
        }
        motion.validate_motion = lambda _document: MotionValidationReport(True)

        with tempfile.TemporaryDirectory() as directory:
            diagnostics = MotionAssistantDiagnostics(
                enabled=True,
                directory=directory,
            )
            result = await CompactMotionWorkflow(
                provider,
                motion,
                metadata,
                diagnostics=diagnostics,
            ).run(
                "Move the entire robot 5 cm higher.",
                model="mock",
                context={"motion": {"working_copy": True}},
                session=session,
            )
            path = next(Path(directory).glob("motion-assistant-*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(result.provider_requests, 2)
        self.assertEqual(provider.counter.counts.total, 2)
        self.assertEqual(len(payload["planning_attempts"]), 2)
        self.assertIsNotNone(payload["planning_attempts"][0]["parser_error"])
        self.assertIsNone(payload["planning_attempts"][1]["parser_error"])
        self.assertEqual(
            payload["planning_attempts"][1]["parsed_spec"]["summary"],
            "Raise the robot.",
        )

    async def test_renderer_initialization_failure_falls_back_to_numerical_context(self):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        provider = RequestCountingProvider(MockProvider([ProviderResponse(text=json.dumps({
            "mode": "edit",
            "summary": "Raise the robot.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "translation_m": [0, 0, 0.05],
                }),
            }],
        }))]))
        capabilities = ProviderCapabilities(
            supports_tools=False,
            supports_vision=True,
            supports_structured_output=True,
            max_images_per_request=8,
        )
        with patch(
            "gui.ai_assistant_controller.RobotViewerFrameRenderer",
            side_effect=RuntimeError("renderer unavailable"),
        ):
            visual = _capture_automatic_visual_context(
                committed,
                object(),
                capabilities,
                selected_interval=None,
                current_time=0.0,
                variant=ImageVariant.ORIGINAL,
            )
        motion = _HoldMotion()
        motion.adapter.free_joints_by_body = {0: type("Joint", (), {"qpos_address": 0})()}
        motion.validate_motion = lambda _document: MotionValidationReport(True)

        result = await CompactMotionWorkflow(provider, motion, metadata).run(
            "Move the entire robot 5 cm higher.",
            model="mock",
            context={"motion": {"working_copy": True}},
            session=session,
            motion_frames=visual.frames,
            context_warnings=(visual.unavailable_reason,),
        )

        self.assertEqual(provider.counter.counts.total, 1)
        self.assertTrue(session.can_accept)
        self.assertIn(
            "Warning: visual context unavailable (RuntimeError)",
            result.proposal_lines,
        )

    async def test_mock_plan_executes_locally_and_validates_staged_candidate(self):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        provider = RequestCountingProvider(MockProvider([ProviderResponse(text=json.dumps({
            "mode": "edit",
            "summary": "Raise the robot.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "translation_m": [0, 0, 0.05],
                }),
            }],
        }))]))
        motion = _HoldMotion()
        motion.adapter.free_joints_by_body = {0: type("Joint", (), {"qpos_address": 0})()}
        motion.validate_motion = lambda _document: MotionValidationReport(True)

        result = await CompactMotionWorkflow(provider, motion, metadata).run(
            "Move the entire robot 5 cm higher.",
            model="mock",
            context={"motion": {"working_copy": True}},
            session=session,
        )

        self.assertEqual(provider.counter.counts.total, 1)
        self.assertEqual(result.provider_requests, 1)
        self.assertEqual(result.proposal_lines, ("root offset",))
        self.assertAlmostEqual(session.working_document.qpos_timeline.get_state(0.0)[2], 0.85)
        self.assertAlmostEqual(committed.qpos_timeline.get_state(0.0)[2], 0.8)
        self.assertTrue(session.can_accept)

    async def test_accept_is_one_history_entry_and_undo_restores_exact_motion(self):
        committed = ProjectDocument("g1", timeline_duration=1.0, qpos_timeline=Timeline())
        before = capture_motion_state(committed)
        store = InMemoryMotionMetadataStore()
        metadata = MotionMetadataService(store, TimestampMotionIdentityResolver())
        metadata.seed_document_as_user_owned(committed)
        session = AIEditSession(committed, metadata_store=store)
        response = ProviderResponse(text=json.dumps({
            "mode": "edit",
            "summary": "Raise the robot.",
            "operations": [{
                "type": "root_offset",
                "arguments": json.dumps({
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "translation_m": [0, 0, 0.05],
                }),
            }],
        }))
        motion = _HoldMotion()
        motion.adapter.free_joints_by_body = {0: type("Joint", (), {"qpos_address": 0})()}
        motion.validate_motion = lambda _document: MotionValidationReport(True)
        await CompactMotionWorkflow(MockProvider([response]), motion, metadata).run(
            "Move the entire robot 5 cm higher.",
            model="mock",
            context={},
            session=session,
        )
        history = HistoryStack(10)

        session.accept(
            EditorController(committed),
            commit_checkpoint=lambda: history.record(
                "Accept AI motion edit",
                before=before,
                after=capture_motion_state(committed),
            ),
        )

        self.assertEqual(len(history.undo_entries), 1)
        self.assertAlmostEqual(committed.qpos_timeline.get_state(0.0)[2], 0.85)
        transition = history.undo(capture_motion_state(committed))
        ReplaceMotionState(transition.target).execute(committed)
        restored = capture_motion_state(committed)
        self.assertEqual(restored.trajectory, before.trajectory)
        self.assertEqual(restored.timeline_duration, before.timeline_duration)
        for (_, actual), (_, expected) in zip(restored.qpos_states, before.qpos_states):
            self.assertTrue((actual == expected).all())


if __name__ == "__main__":
    unittest.main()

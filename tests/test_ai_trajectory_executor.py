"""Tests for deterministic compact-spec execution and revision validation."""

from __future__ import annotations

import unittest

from application.ai.edit_session import AIEditSession, AIEditSessionError
from application.ai.motion_services import MotionValidationReport
from application.ai.schemas import EditAuthor, MotionEntityRef
from application.ai.trajectory_edit_spec import (
    TrajectoryEditMode,
    TrajectoryEditSpec,
    TrajectoryOperation,
    TrajectoryOperationType,
)
from application.ai.trajectory_executor import (
    TrajectoryExecutionContext,
    TrajectoryExecutionError,
    TrajectorySpecExecutor,
)
from application.editor_commands import UpdateKeyframe
from application.editor_controller import EditorController
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


def _session():
    document = ProjectDocument("g1")
    document.trajectory.add_frame(TargetFrame(frame_name="pelvis", z=0.8))
    session = AIEditSession(document)
    reference = MotionEntityRef("pelvis")
    session.metadata.record(reference, EditAuthor.AI)
    return document, session, reference


def _spec():
    return TrajectoryEditSpec(
        TrajectoryEditMode.EDIT,
        "Raise pelvis.",
        (TrajectoryOperation(
            TrajectoryOperationType.ROOT_OFFSET,
            {"start_time": 0.0, "end_time": 1.0, "translation_m": [0, 0, 0.05]},
        ),),
    )


class TrajectorySpecExecutorTests(unittest.TestCase):
    def test_successful_execution_validates_exact_current_revision(self):
        committed, session, reference = _session()

        def edit(_operation, context):
            context.session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.85)),
                affected_entities=(reference,),
            )
            return {"changed": True}

        result = TrajectorySpecExecutor(
            {TrajectoryOperationType.ROOT_OFFSET: edit},
            lambda _document: MotionValidationReport(True),
        ).execute(_spec(), context=TrajectoryExecutionContext(session, object()))

        self.assertTrue(result.validation.valid)
        self.assertTrue(session.can_accept)
        session.accept(EditorController(committed))
        self.assertEqual(committed.trajectory.frames[0].z, 0.85)

    def test_authoring_warnings_do_not_block_validated_candidate(self):
        _committed, session, reference = _session()

        def edit(_operation, context):
            context.session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.85)),
                affected_entities=(reference,),
            )
            return {"changed": True}

        result = TrajectorySpecExecutor(
            {TrajectoryOperationType.ROOT_OFFSET: edit},
            lambda _document: MotionValidationReport(
                True,
                (),
                ("qpos Keyframe has a blocking collision",),
            ),
        ).execute(_spec(), context=TrajectoryExecutionContext(session, object()))

        self.assertEqual(
            result.validation.warnings,
            ("qpos Keyframe has a blocking collision",),
        )
        self.assertTrue(session.can_accept)

    def test_execution_failure_restores_checkpoint(self):
        committed, session, reference = _session()

        def fail(_operation, context):
            context.session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.85)),
                affected_entities=(reference,),
            )
            raise RuntimeError("local solver failed")

        with self.assertRaisesRegex(RuntimeError, "local solver failed"):
            TrajectorySpecExecutor(
                {TrajectoryOperationType.ROOT_OFFSET: fail},
                lambda _document: MotionValidationReport(True),
            ).execute(_spec(), context=TrajectoryExecutionContext(session, object()))

        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.8)
        self.assertFalse(session.has_changes)
        self.assertEqual(committed.trajectory.frames[0].z, 0.8)

    def test_failed_validation_keeps_candidate_visible_but_unacceptable(self):
        committed, session, reference = _session()

        def edit(_operation, context):
            context.session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.85)),
                affected_entities=(reference,),
            )
            return {"changed": True}

        with self.assertRaisesRegex(TrajectoryExecutionError, "unstable"):
            TrajectorySpecExecutor(
                {TrajectoryOperationType.ROOT_OFFSET: edit},
                lambda _document: MotionValidationReport(False, ("unstable",)),
            ).execute(_spec(), context=TrajectoryExecutionContext(session, object()))

        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.85)
        self.assertFalse(session.can_accept)
        with self.assertRaises(AIEditSessionError):
            session.accept(EditorController(committed))


if __name__ == "__main__":
    unittest.main()

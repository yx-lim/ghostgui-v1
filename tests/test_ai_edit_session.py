"""Tests for detached, multi-author AI motion editing sessions."""

from __future__ import annotations

import unittest

from application.ai.edit_session import (
    AIEditSession,
    AIEditSessionError,
    AIEditSessionState,
)
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_state import ReplaceMotionState, capture_motion_state
from application.ai.schemas import EditAuthor, MotionEntityRef
from application.editor_commands import UpdateKeyframe
from application.editor_controller import EditorController
from application.editor_events import DocumentChanged, EditorEventBus
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class FakeTimeline:
    def __init__(self, states=()):
        self.states = {}
        for time, value in states:
            self.set_state(time, value)

    def set_state(self, time, value):
        self.states[round(float(time), 6)] = list(value)

    def get_state(self, time):
        value = self.states.get(round(float(time), 6))
        return None if value is None else value.copy()

    def times(self):
        return sorted(self.states)


def _document():
    document = ProjectDocument(
        "g1",
        qpos_timeline=FakeTimeline(((0.0, [0.0, 1.0]),)),
    )
    document.trajectory.add_frame(
        TargetFrame(time=0.0, frame_name="pelvis", z=0.9)
    )
    return document


class AIEditSessionTests(unittest.TestCase):
    def test_working_copy_is_detached_from_committed_motion(self):
        committed = _document()
        session = AIEditSession(committed)
        reference = MotionEntityRef("keyframe-1")

        session.apply_ai(
            UpdateKeyframe(
                0,
                TargetFrame(time=0.0, frame_name="pelvis", z=0.7),
            ),
            affected_entities=(reference,),
        )
        session.working_document.qpos_timeline.set_state(0.0, [4.0, 5.0])

        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        self.assertEqual(committed.qpos_timeline.get_state(0.0), [0.0, 1.0])
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.7)
        self.assertEqual(session.metadata.get(reference).author, EditAuthor.AI)

    def test_manual_edit_is_recorded_and_blocks_later_ai_edit(self):
        session = AIEditSession(_document())
        reference = MotionEntityRef("pelvis-keyframe")
        session.apply_ai(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.8)),
            affected_entities=(reference,),
        )
        session.apply_manual(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.75)),
            affected_entities=(reference,),
        )

        self.assertEqual(session.metadata.get(reference).author, EditAuthor.USER)
        self.assertEqual(
            [record.author for record in session.edits],
            [EditAuthor.AI, EditAuthor.USER],
        )
        with self.assertRaisesRegex(AIEditSessionError, "user-authored"):
            session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.6)),
                affected_entities=(reference,),
            )
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.75)

    def test_refine_request_keeps_manually_modified_working_copy(self):
        session = AIEditSession(_document())
        session.apply_ai(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.8))
        )
        session.apply_manual(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.77))
        )

        session.begin_provider_request()
        self.assertTrue(session.provider_request_active)
        with self.assertRaises(AIEditSessionError):
            session.apply_manual(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.7))
            )
        session.finish_provider_request(result_staged=True)

        self.assertEqual(session.state, AIEditSessionState.STAGED)
        self.assertEqual(session.working_document.trajectory.frames[0].z, 0.77)

    def test_accept_is_one_atomic_committed_document_change(self):
        committed = _document()
        events = EditorEventBus()
        changed = []
        events.subscribe(DocumentChanged, changed.append)
        committed_controller = EditorController(committed, events)
        metadata = InMemoryMotionMetadataStore()
        session = AIEditSession(committed, metadata_store=metadata)
        reference = MotionEntityRef("pelvis-keyframe")
        session.apply_ai(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.7)),
            affected_entities=(reference,),
        )
        session.working_document.qpos_timeline.set_state(0.0, [2.0, 3.0])

        result = session.accept(committed_controller)

        self.assertTrue(result.changed)
        self.assertEqual(result.operation, "replace_motion_state")
        self.assertEqual(committed.revision, 1)
        self.assertEqual(len(changed), 1)
        self.assertEqual(committed.trajectory.frames[0].z, 0.7)
        self.assertEqual(committed.qpos_timeline.get_state(0.0), [2.0, 3.0])
        self.assertEqual(metadata.get(reference).author, EditAuthor.AI)
        self.assertEqual(session.state, AIEditSessionState.ACCEPTED)

    def test_reject_discards_working_copy_and_metadata(self):
        committed = _document()
        metadata = InMemoryMotionMetadataStore()
        reference = MotionEntityRef("pelvis-keyframe")
        session = AIEditSession(committed, metadata_store=metadata)
        session.apply_ai(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.7)),
            affected_entities=(reference,),
        )

        session.reject()

        self.assertEqual(committed.trajectory.frames[0].z, 0.9)
        self.assertIsNone(metadata.get(reference))
        self.assertEqual(session.state, AIEditSessionState.REJECTED)

    def test_accept_rejects_stale_committed_document(self):
        committed = _document()
        committed_controller = EditorController(committed)
        session = AIEditSession(committed)
        session.apply_ai(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.7))
        )
        committed_controller.execute(
            UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=1.0))
        )

        with self.assertRaisesRegex(AIEditSessionError, "changed after"):
            session.accept(committed_controller)
        self.assertEqual(committed.trajectory.frames[0].z, 1.0)

    def test_metadata_identity_strategy_is_behind_service(self):
        frame = TargetFrame(time=1.2345678, frame_name="torso")
        service = MotionMetadataService(
            InMemoryMotionMetadataStore(),
            TimestampMotionIdentityResolver(),
        )

        reference = service.reference_for_keyframe(frame)

        self.assertTrue(reference.identifier.startswith("legacy-keyframe-v1:"))
        self.assertFalse(hasattr(reference, "time_seconds"))
        self.assertFalse(hasattr(reference, "frame_name"))

    def test_protected_content_cannot_be_overridden(self):
        session = AIEditSession(_document())
        reference = MotionEntityRef("protected-foot")
        session.protect(reference)
        with self.assertRaises(AIEditSessionError):
            session.apply_ai(
                UpdateKeyframe(0, TargetFrame(frame_name="pelvis", z=0.7)),
                affected_entities=(reference,),
                allow_user_override=True,
            )

    def test_replace_motion_state_stays_unchanged_on_qpos_failure(self):
        document = _document()
        replacement_document = _document()
        replacement_document.trajectory.frames[0].z = 0.5
        replacement_document.qpos_timeline.set_state(0.0, [9.0, 9.0])
        state = capture_motion_state(replacement_document)
        def fail(time, value):
            raise RuntimeError("invalid qpos")

        document.qpos_timeline.set_state = fail
        with self.assertRaisesRegex(RuntimeError, "invalid qpos"):
            ReplaceMotionState(state).execute(document)

        self.assertEqual(document.trajectory.frames[0].z, 0.9)
        self.assertEqual(document.qpos_timeline.get_state(0.0), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

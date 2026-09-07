"""Conservative motion provenance, protection, and persistence contracts."""

from __future__ import annotations

import unittest

from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionEditMetadata,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.schemas import EditAuthor, MotionEntityRef
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


class FakeTimeline:
    def __init__(self, times=()):
        self._times = tuple(float(value) for value in times)

    def times(self):
        return self._times


def _document():
    document = ProjectDocument(
        "g1",
        qpos_timeline=FakeTimeline((0.0, 1.0)),
    )
    document.trajectory.add_frame(
        TargetFrame(time=0.0, frame_name="pelvis", z=0.9)
    )
    document.trajectory.add_frame(
        TargetFrame(time=1.0, frame_name="pelvis", z=0.8)
    )
    return document


class MotionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.document = _document()
        self.store = InMemoryMotionMetadataStore()
        self.service = MotionMetadataService(
            self.store,
            TimestampMotionIdentityResolver(),
        )

    def test_seed_classifies_existing_logical_and_qpos_keyframes_as_user(self):
        seeded = self.service.seed_document_as_user_owned(self.document)

        self.assertEqual(seeded, 4)
        self.assertEqual(
            {metadata.author for metadata in self.store.snapshot().values()},
            {EditAuthor.USER},
        )
        self.assertEqual(self.service.seed_document_as_user_owned(self.document), 0)

    def test_seed_does_not_replace_known_ai_provenance(self):
        frame = self.document.trajectory.frames[0]
        reference = self.service.reference_for_keyframe(frame)
        self.store.record(reference, EditAuthor.AI)

        self.assertEqual(self.service.seed_document_as_user_owned(self.document), 3)
        self.assertEqual(self.store.get(reference).author, EditAuthor.AI)

    def test_project_payload_round_trips_author_protection_and_sequence(self):
        self.service.seed_document_as_user_owned(self.document)
        frame = self.document.trajectory.frames[0]
        reference = self.service.reference_for_keyframe(frame)
        self.store.record(reference, EditAuthor.AI)
        self.store.set_protected(reference, True)

        payload = self.service.to_project_dict(self.document)
        restored_store = InMemoryMotionMetadataStore()
        restored = MotionMetadataService(
            restored_store,
            TimestampMotionIdentityResolver(),
        )
        restored.restore_project_dict(payload, self.document)

        self.assertEqual(restored_store.snapshot(), self.store.snapshot())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["entities"]), 4)

    def test_legacy_workspace_without_payload_seeds_user_ownership(self):
        self.service.restore_project_dict(None, self.document)

        self.assertEqual(len(self.store.snapshot()), 4)
        self.assertTrue(
            all(
                metadata.author is EditAuthor.USER
                for metadata in self.store.snapshot().values()
            )
        )

    def test_restore_seeds_entities_missing_from_older_metadata_payload(self):
        frame = self.document.trajectory.frames[0]
        reference = self.service.reference_for_keyframe(frame)
        payload = {
            "schema_version": 1,
            "entities": [
                {
                    "id": reference.identifier,
                    "author": "ai",
                    "protected": False,
                    "sequence": 7,
                }
            ],
        }

        self.service.restore_project_dict(payload, self.document)

        self.assertEqual(self.store.get(reference).author, EditAuthor.AI)
        self.assertEqual(len(self.store.snapshot()), 4)
        self.assertEqual(
            sum(
                metadata.author is EditAuthor.USER
                for metadata in self.store.snapshot().values()
            ),
            3,
        )

    def test_invalid_persisted_metadata_fails_without_replacing_store(self):
        existing = MotionEntityRef("stable-keyframe-id")
        self.store.replace({existing: MotionEditMetadata(EditAuthor.USER)})

        with self.assertRaisesRegex(ValueError, "author"):
            self.service.restore_project_dict(
                {
                    "schema_version": 1,
                    "entities": [
                        {
                            "id": "other-id",
                            "author": "unknown",
                            "protected": False,
                            "sequence": 0,
                        }
                    ],
                },
                self.document,
            )

        self.assertEqual(
            self.store.snapshot(),
            {existing: MotionEditMetadata(EditAuthor.USER)},
        )

    def test_human_mutation_claims_current_entities_and_preserves_protection(self):
        self.service.seed_document_as_user_owned(self.document)
        frame = self.document.trajectory.frames[0]
        reference = self.service.reference_for_keyframe(frame)
        self.store.record(reference, EditAuthor.AI)
        self.store.set_protected(reference, True)
        stale = MotionEntityRef("deleted-keyframe")
        self.store.record(stale, EditAuthor.AI)

        changed = self.service.claim_document_as_user_owned(self.document)

        self.assertGreaterEqual(changed, 1)
        self.assertEqual(self.store.get(reference).author, EditAuthor.USER)
        self.assertTrue(self.store.get(reference).protected)
        self.assertIsNone(self.store.get(stale))


if __name__ == "__main__":
    unittest.main()

"""Identity-agnostic provenance and protection for motion entities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Mapping, Protocol

from application.ai.schemas import EditAuthor, MotionEntityRef
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


MOTION_METADATA_SCHEMA_VERSION = 1
MAX_PERSISTED_MOTION_METADATA_ENTITIES = 100_000


@dataclass(frozen=True)
class MotionEditMetadata:
    author: EditAuthor
    protected: bool = False
    sequence: int = 0


class MotionIdentityResolver(Protocol):
    """Resolve model objects without exposing their identity strategy to AI."""

    def reference_for_keyframe(self, frame: TargetFrame) -> MotionEntityRef:
        ...

    def reference_for_qpos_keyframe(self, time_seconds: float) -> MotionEntityRef:
        ...


class TimestampMotionIdentityResolver:
    """MVP resolver isolated for later replacement by stable Keyframe IDs."""

    def reference_for_keyframe(self, frame: TargetFrame) -> MotionEntityRef:
        digest = self._digest(frame.frame_name, frame.time)
        return MotionEntityRef(f"legacy-keyframe-v1:{digest}")

    def reference_for_qpos_keyframe(self, time_seconds: float) -> MotionEntityRef:
        digest = self._digest("__qpos__", time_seconds)
        return MotionEntityRef(f"legacy-qpos-keyframe-v1:{digest}")

    @staticmethod
    def _digest(kind: str, time_seconds: float) -> str:
        legacy_key = f"{kind}\0{round(float(time_seconds), 6):.6f}"
        return sha256(legacy_key.encode("utf-8")).hexdigest()


class MotionMetadataStore(Protocol):
    def fork(self) -> "MotionMetadataStore":
        ...

    def snapshot(self) -> Mapping[MotionEntityRef, MotionEditMetadata]:
        ...

    def replace(self, values: Mapping[MotionEntityRef, MotionEditMetadata]) -> None:
        ...

    def get(self, reference: MotionEntityRef) -> MotionEditMetadata | None:
        ...

    def record(self, reference: MotionEntityRef, author: EditAuthor) -> None:
        ...

    def set_protected(self, reference: MotionEntityRef, protected: bool) -> bool:
        ...

    def permits_ai_edit(
        self,
        reference: MotionEntityRef,
        *,
        allow_user_override: bool = False,
    ) -> bool:
        ...


class InMemoryMotionMetadataStore:
    """Detached MVP store; persistence can keep the same public contract."""

    def __init__(
        self,
        values: Mapping[MotionEntityRef, MotionEditMetadata] | None = None,
    ) -> None:
        self._values = dict(values or {})
        self._sequence = max(
            (metadata.sequence for metadata in self._values.values()),
            default=0,
        )

    def fork(self) -> "InMemoryMotionMetadataStore":
        return InMemoryMotionMetadataStore(self._values)

    def snapshot(self) -> Mapping[MotionEntityRef, MotionEditMetadata]:
        return dict(self._values)

    def replace(self, values: Mapping[MotionEntityRef, MotionEditMetadata]) -> None:
        self._values = dict(values)
        self._sequence = max(
            (metadata.sequence for metadata in self._values.values()),
            default=0,
        )

    def get(self, reference: MotionEntityRef) -> MotionEditMetadata | None:
        return self._values.get(reference)

    def record(self, reference: MotionEntityRef, author: EditAuthor) -> None:
        self._sequence += 1
        previous = self._values.get(reference)
        self._values[reference] = MotionEditMetadata(
            author=author,
            protected=False if previous is None else previous.protected,
            sequence=self._sequence,
        )

    def set_protected(self, reference: MotionEntityRef, protected: bool) -> bool:
        previous = self._values.get(reference)
        if previous is None:
            previous = MotionEditMetadata(EditAuthor.USER)
        protected = bool(protected)
        if previous.protected is protected:
            return False
        self._sequence += 1
        self._values[reference] = replace(
            previous,
            protected=protected,
            sequence=self._sequence,
        )
        return True

    def permits_ai_edit(
        self,
        reference: MotionEntityRef,
        *,
        allow_user_override: bool = False,
    ) -> bool:
        metadata = self.get(reference)
        if metadata is None:
            return False
        if metadata.protected:
            return False
        return metadata.author is not EditAuthor.USER or allow_user_override


class MotionMetadataService:
    """Single lookup boundary used by motion services and future migrations."""

    def __init__(
        self,
        store: MotionMetadataStore,
        resolver: MotionIdentityResolver,
    ) -> None:
        self.store = store
        self.resolver = resolver

    def reference_for_keyframe(self, frame: TargetFrame) -> MotionEntityRef:
        return self.resolver.reference_for_keyframe(frame)

    def metadata_for_keyframe(self, frame: TargetFrame) -> MotionEditMetadata | None:
        return self.store.get(self.reference_for_keyframe(frame))

    def reference_for_qpos_keyframe(self, time_seconds: float) -> MotionEntityRef:
        return self.resolver.reference_for_qpos_keyframe(time_seconds)

    def seed_document_as_user_owned(self, document: ProjectDocument) -> int:
        """Conservatively classify every untracked committed motion entity."""

        seeded = 0
        for reference in self._document_references(document):
            if self.store.get(reference) is None:
                self.store.record(reference, EditAuthor.USER)
                seeded += 1
        return seeded

    def claim_document_as_user_owned(self, document: ProjectDocument) -> int:
        """Record a committed human mutation and drop identities no longer present."""

        references = self._document_references(document)
        previous = self.store.snapshot()
        values: dict[MotionEntityRef, MotionEditMetadata] = {}
        changed = 0
        sequence = max(
            (metadata.sequence for metadata in previous.values()),
            default=0,
        )
        for reference in references:
            metadata = previous.get(reference)
            if metadata is None or metadata.author is not EditAuthor.USER:
                sequence += 1
                changed += 1
                metadata = MotionEditMetadata(
                    author=EditAuthor.USER,
                    protected=False if metadata is None else metadata.protected,
                    sequence=sequence,
                )
            values[reference] = metadata
        self.store.replace(values)
        return changed

    def to_project_dict(self, document: ProjectDocument) -> dict[str, Any]:
        """Serialize opaque identities without exposing their implementation."""

        self.seed_document_as_user_owned(document)
        return {
            "schema_version": MOTION_METADATA_SCHEMA_VERSION,
            "entities": [
                {
                    "id": reference.identifier,
                    "author": metadata.author.value,
                    "protected": metadata.protected,
                    "sequence": metadata.sequence,
                }
                for reference, metadata in sorted(
                    self.store.snapshot().items(),
                    key=lambda item: item[0].identifier,
                )
            ],
        }

    def restore_project_dict(
        self,
        payload: Mapping[str, Any] | None,
        document: ProjectDocument,
    ) -> None:
        """Restore persisted metadata, then seed legacy or newly unknown content."""

        if payload is None:
            self.store.replace({})
            self.seed_document_as_user_owned(document)
            return
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "entities",
        }:
            raise ValueError("motion metadata must be a versioned object")
        if payload["schema_version"] != MOTION_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported motion metadata schema version")
        entities = payload["entities"]
        if not isinstance(entities, list):
            raise ValueError("motion metadata entities must be a list")
        if len(entities) > MAX_PERSISTED_MOTION_METADATA_ENTITIES:
            raise ValueError("motion metadata contains too many entities")

        values: dict[MotionEntityRef, MotionEditMetadata] = {}
        for entity in entities:
            if not isinstance(entity, Mapping) or set(entity) != {
                "id",
                "author",
                "protected",
                "sequence",
            }:
                raise ValueError("motion metadata entity is invalid")
            identifier = entity["id"]
            protected = entity["protected"]
            sequence = entity["sequence"]
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or len(identifier) > 512
            ):
                raise ValueError("motion metadata entity id is invalid")
            if not isinstance(protected, bool):
                raise ValueError("motion metadata protection must be boolean")
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
            ):
                raise ValueError("motion metadata sequence must be non-negative")
            try:
                author = EditAuthor(entity["author"])
            except (TypeError, ValueError) as error:
                raise ValueError("motion metadata author is invalid") from error
            reference = MotionEntityRef(identifier)
            if reference in values:
                raise ValueError("motion metadata contains a duplicate entity")
            values[reference] = MotionEditMetadata(author, protected, sequence)

        self.store.replace(values)
        self.seed_document_as_user_owned(document)

    def remap_keyframe(self, before: TargetFrame, after: TargetFrame) -> None:
        self._remap(
            self.reference_for_keyframe(before),
            self.reference_for_keyframe(after),
        )

    def remap_qpos_keyframe(self, before_time: float, after_time: float) -> None:
        self._remap(
            self.reference_for_qpos_keyframe(before_time),
            self.reference_for_qpos_keyframe(after_time),
        )

    def _remap(self, before: MotionEntityRef, after: MotionEntityRef) -> None:
        if before == after:
            return
        values = dict(self.store.snapshot())
        metadata = values.pop(before, None)
        if metadata is None:
            return
        existing = values.get(after)
        if existing is None or metadata.sequence >= existing.sequence:
            values[after] = metadata
        self.store.replace(values)

    def _document_references(
        self,
        document: ProjectDocument,
    ) -> tuple[MotionEntityRef, ...]:
        references = [
            self.reference_for_keyframe(frame)
            for frame in document.trajectory.frames
        ]
        timeline = document.qpos_timeline
        if timeline is not None:
            references.extend(
                self.reference_for_qpos_keyframe(time)
                for time in timeline.times()
            )
        return tuple(dict.fromkeys(references))

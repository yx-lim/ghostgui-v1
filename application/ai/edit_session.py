"""Detached, multi-turn AI motion editing sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionEditMetadata,
    MotionMetadataStore,
)
from application.ai.motion_state import (
    MotionStateSnapshot,
    ReplaceMotionState,
    capture_motion_state,
    detached_document,
)
from application.ai.schemas import EditAuthor, MotionEntityRef
from application.editor_commands import CommandResult, EditorCommand
from application.editor_controller import EditorController
from application.project_document import ProjectDocument


class AIEditSessionError(RuntimeError):
    """Base class for invalid AI editing session operations."""


class AIEditSessionState(str, Enum):
    READY = "ready"
    REQUESTING = "requesting"
    STAGED = "staged"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SessionEditRecord:
    author: EditAuthor
    operation: str
    affected_entities: tuple[MotionEntityRef, ...]
    working_revision: int


@dataclass(frozen=True)
class AIEditSessionCheckpoint:
    """Detached operation boundary used for local execution rollback."""

    motion_state: MotionStateSnapshot
    metadata: dict[MotionEntityRef, MotionEditMetadata]
    edits: tuple[SessionEditRecord, ...]
    state: AIEditSessionState
    working_revision: int
    validated_revision: int | None
    owner_token: object


class AIEditSession:
    """Own a detached working copy across AI, manual, and refine turns."""

    def __init__(
        self,
        committed_document: ProjectDocument,
        *,
        metadata_store: MotionMetadataStore | None = None,
    ) -> None:
        self._committed_document = committed_document
        self._base_revision = committed_document.revision
        self._committed_metadata = metadata_store or InMemoryMotionMetadataStore()
        self.metadata = self._committed_metadata.fork()
        self.working_document = detached_document(committed_document)
        self.controller = EditorController(self.working_document)
        self._state = AIEditSessionState.READY
        self._state_before_request = AIEditSessionState.READY
        self._edits: list[SessionEditRecord] = []
        self._validated_revision: int | None = None
        self._checkpoint_token = object()

    @property
    def state(self) -> AIEditSessionState:
        return self._state

    @property
    def edits(self) -> tuple[SessionEditRecord, ...]:
        return tuple(self._edits)

    @property
    def has_changes(self) -> bool:
        return bool(self._edits)

    @property
    def provider_request_active(self) -> bool:
        return self._state is AIEditSessionState.REQUESTING

    @property
    def working_revision(self) -> int:
        return int(self.working_document.revision)

    @property
    def validated_revision(self) -> int | None:
        return self._validated_revision

    @property
    def current_revision_validated(self) -> bool:
        return self._validated_revision == self.working_revision

    @property
    def committed_revision_current(self) -> bool:
        return self._committed_document.revision == self._base_revision

    @property
    def can_accept(self) -> bool:
        return (
            self._state is AIEditSessionState.STAGED
            and self.has_changes
            and self.current_revision_validated
            and self.committed_revision_current
        )

    def checkpoint(self) -> AIEditSessionCheckpoint:
        """Capture the working copy without exposing its identity strategy."""

        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        return AIEditSessionCheckpoint(
            motion_state=capture_motion_state(self.working_document),
            metadata=dict(self.metadata.snapshot()),
            edits=tuple(self._edits),
            state=self._state,
            working_revision=self.working_revision,
            validated_revision=self._validated_revision,
            owner_token=self._checkpoint_token,
        )

    def restore_checkpoint(self, checkpoint: AIEditSessionCheckpoint) -> None:
        """Roll back one local operation while retaining earlier session work."""

        if (
            not isinstance(checkpoint, AIEditSessionCheckpoint)
            or checkpoint.owner_token is not self._checkpoint_token
        ):
            raise TypeError("checkpoint must belong to an AI edit session")
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        self.controller.execute(ReplaceMotionState(checkpoint.motion_state))
        self.metadata.replace(checkpoint.metadata)
        self._edits = list(checkpoint.edits)
        self._state = checkpoint.state
        self._validated_revision = (
            self.working_revision
            if checkpoint.validated_revision == checkpoint.working_revision
            else None
        )

    def invalidate_validation(self) -> None:
        """Prevent acceptance until the current working revision validates."""

        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        self._validated_revision = None

    def mark_current_revision_validated(self) -> None:
        """Record successful validation of exactly the current working copy."""

        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        self._validated_revision = self.working_revision

    def begin_provider_request(self) -> None:
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        self._state_before_request = self._state
        self._state = AIEditSessionState.REQUESTING

    def finish_provider_request(self, *, result_staged: bool) -> None:
        self._require_state(AIEditSessionState.REQUESTING)
        self._state = (
            AIEditSessionState.STAGED
            if result_staged
            else self._state_before_request
        )

    def apply_ai(
        self,
        command: EditorCommand,
        *,
        affected_entities: tuple[MotionEntityRef, ...] = (),
        created_entities: tuple[MotionEntityRef, ...] = (),
        allow_user_override: bool = False,
    ) -> CommandResult:
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        created = frozenset(created_entities)
        if not created.issubset(affected_entities):
            raise ValueError("created AI entities must also be affected entities")
        blocked = tuple(
            reference
            for reference in affected_entities
            if (
                self.metadata.get(reference) is not None
                and not self.metadata.permits_ai_edit(
                    reference,
                    allow_user_override=allow_user_override,
                )
            ) or (
                self.metadata.get(reference) is None
                and reference not in created
            )
        )
        if blocked:
            raise AIEditSessionError(
                "AI edit targets user-authored or protected motion content"
            )
        return self._apply(command, EditAuthor.AI, affected_entities)

    def apply_manual(
        self,
        command: EditorCommand,
        *,
        affected_entities: tuple[MotionEntityRef, ...] = (),
    ) -> CommandResult:
        self._require_state(AIEditSessionState.STAGED)
        return self._apply(command, EditAuthor.USER, affected_entities)

    def protect(
        self,
        reference: MotionEntityRef,
        protected: bool = True,
        *,
        author: EditAuthor = EditAuthor.USER,
    ) -> bool:
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        if author is EditAuthor.AI and not protected:
            raise AIEditSessionError("AI edits cannot remove Keyframe protection")
        if not self.metadata.set_protected(reference, protected):
            return False
        self._edits.append(SessionEditRecord(
            author=author,
            operation="protect_keyframe" if protected else "unprotect_keyframe",
            affected_entities=(reference,),
            working_revision=self.working_document.revision,
        ))
        self._state = AIEditSessionState.STAGED
        return True

    def accept(
        self,
        committed_controller: EditorController,
        *,
        commit_checkpoint: Callable[[], None] | None = None,
    ) -> CommandResult:
        """Commit motion, provenance, and an optional history checkpoint.

        The callback is part of the logical transaction.  Presentation work must
        happen after this method returns.
        """
        self._require_state(AIEditSessionState.STAGED)
        if committed_controller.document is not self._committed_document:
            raise AIEditSessionError("session belongs to a different committed document")
        if self._committed_document.revision != self._base_revision:
            raise AIEditSessionError(
                "committed motion changed after the AI edit session started"
            )
        if not self.has_changes:
            raise AIEditSessionError("cannot accept an AI session with no motion changes")
        if not self.current_revision_validated:
            raise AIEditSessionError(
                "cannot accept an AI working copy that has not passed validation"
            )
        before_motion = capture_motion_state(self._committed_document)
        before_metadata = dict(self._committed_metadata.snapshot())
        before_revision = self._committed_document.revision
        before_dirty = self._committed_document.dirty
        try:
            # Provenance is an in-memory snapshot and can be compensated before
            # publishing the committed document change.
            self._committed_metadata.replace(self.metadata.snapshot())
            result = committed_controller.execute(
                ReplaceMotionState(
                    capture_motion_state(self.working_document),
                    force_change=True,
                )
            )
            if commit_checkpoint is not None:
                commit_checkpoint()
        except Exception:
            # Restore directly so compensation does not create a second editor
            # revision or history operation.
            ReplaceMotionState(before_motion, force_change=True).execute(
                self._committed_document
            )
            self._committed_document.revision = before_revision
            self._committed_document.dirty = before_dirty
            self._committed_metadata.replace(before_metadata)
            raise
        self._state = AIEditSessionState.ACCEPTED
        return result

    def reject(self) -> None:
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        self._state = AIEditSessionState.REJECTED

    def _apply(
        self,
        command: EditorCommand,
        author: EditAuthor,
        affected_entities: tuple[MotionEntityRef, ...],
    ) -> CommandResult:
        result = self.controller.execute(command)
        if result.changed:
            self._validated_revision = None
            for reference in affected_entities:
                self.metadata.record(reference, author)
            self._edits.append(
                SessionEditRecord(
                    author=author,
                    operation=result.operation,
                    affected_entities=tuple(affected_entities),
                    working_revision=self.working_document.revision,
                )
            )
            self._state = AIEditSessionState.STAGED
        return result

    def _require_state(self, *allowed: AIEditSessionState) -> None:
        if self._state not in allowed:
            expected = ", ".join(state.value for state in allowed)
            raise AIEditSessionError(
                f"operation requires session state {expected}; current state is "
                f"{self._state.value}"
            )

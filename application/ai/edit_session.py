"""Detached, multi-turn AI motion editing sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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

    def checkpoint(self) -> AIEditSessionCheckpoint:
        """Capture the working copy without exposing its identity strategy."""

        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        return AIEditSessionCheckpoint(
            motion_state=capture_motion_state(self.working_document),
            metadata=dict(self.metadata.snapshot()),
            edits=tuple(self._edits),
            state=self._state,
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
        allow_user_override: bool = False,
    ) -> CommandResult:
        self._require_state(AIEditSessionState.READY, AIEditSessionState.STAGED)
        blocked = tuple(
            reference
            for reference in affected_entities
            if not self.metadata.permits_ai_edit(
                reference,
                allow_user_override=allow_user_override,
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

    def accept(self, committed_controller: EditorController) -> CommandResult:
        self._require_state(AIEditSessionState.STAGED)
        if committed_controller.document is not self._committed_document:
            raise AIEditSessionError("session belongs to a different committed document")
        if self._committed_document.revision != self._base_revision:
            raise AIEditSessionError(
                "committed motion changed after the AI edit session started"
            )
        if not self.has_changes:
            raise AIEditSessionError("cannot accept an AI session with no motion changes")
        result = committed_controller.execute(
            ReplaceMotionState(
                capture_motion_state(self.working_document),
                force_change=True,
            )
        )
        self._committed_metadata.replace(self.metadata.snapshot())
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

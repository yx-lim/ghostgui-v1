"""Serialized, bounded and transactional dispatch of untrusted command inputs."""

from dataclasses import dataclass, replace
import json
from threading import RLock

from application.ai.motion_state import capture_motion_state
from application.ai.schemas import EditAuthor

from .events import AgentEvent


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    payload: dict
    is_error: bool = False
    images: tuple = ()


class ToolRuntime:
    def __init__(self, catalog, workspace, *, max_calls=256, max_request_bytes=262144,
                 max_result_bytes=65536, cancellation_token=None):
        self.catalog, self.workspace = catalog, workspace
        self.max_calls, self.calls = max_calls, 0
        self.max_request_bytes, self.max_result_bytes = max_request_bytes, max_result_bytes
        self.cancellation_token = cancellation_token
        self._lock = RLock()

    def _check_cancel(self):
        if self.cancellation_token is not None and self.cancellation_token.cancellation_requested:
            raise RuntimeError("operation cancelled")

    def call(self, name, arguments):
        with self._lock:
            checkpoint = None
            previous_generated = self.workspace.generated
            previous_session = self.workspace.candidate()
            validation_identity = getattr(self.workspace, "_validation_identity", None)
            previous_checkpoints = self.workspace.checkpoints.entries.copy()
            previous_identity = None
            mutation = False
            try:
                if self.calls >= self.max_calls:
                    raise ValueError("tool call budget exceeded")
                self.calls += 1
                self._check_cancel()
                if not isinstance(name, str) or len(name) > 64:
                    raise ValueError("tool name must be a string of at most 64 characters")
                encoded = json.dumps(arguments, allow_nan=False).encode()
                if len(encoded) > self.max_request_bytes:
                    raise ValueError("tool request exceeds the byte limit")
                self.catalog.registry.validate_arguments(name, arguments)
                meta = self.catalog.meta[name]
                mutation = meta.mutates_candidate
                if mutation or name in ("generate_trajectory", "validate_motion", "undo"):
                    previous_identity = self.workspace.identity()
                if mutation or name == "generate_trajectory":
                    session = self.workspace.candidate(create=True)
                    if arguments.get("expected_revision", session.working_revision) != session.working_revision:
                        raise ValueError("stale_revision: candidate revision changed")
                    checkpoint = session.checkpoint()
                    before_values = self.workspace.entity_values(session.working_document)
                    self.workspace.emit(AgentEvent("tool_started", name, session.working_revision))
                elif name in ("validate_motion", "undo") and previous_session is not None:
                    checkpoint = previous_session.checkpoint()
                result = self.catalog.registry.execute(name, arguments)
                if checkpoint is not None:
                    self._check_cancel()
                if mutation:
                    session = self.workspace.candidate()
                    after_values = self.workspace.entity_values(session.working_document)
                    affected = before_values.keys() | after_values.keys()
                    for reference in affected:
                        if before_values.get(reference) == after_values.get(reference):
                            continue
                        metadata = checkpoint.metadata.get(reference)
                        if reference in before_values and (
                            metadata is None or metadata.protected or
                            (metadata.author is EditAuthor.USER and not self.workspace.allow_user_override)
                        ):
                            raise ValueError("mutation affects protected or user-owned motion")
                    if len(session.working_document.qpos_timeline.times()) > self.workspace.max_samples:
                        raise ValueError("candidate exceeds the qpos sample limit")
                    # Malformed/kinematically inconsistent changes never become a new base.
                    structural = self.workspace.motion.validate_motion(session.working_document)
                    if not structural.valid:
                        raise ValueError("; ".join(structural.issues[:8]))
                    session.invalidate_validation()
                    self.workspace.generated = None
                    self.workspace._validation_identity = None
                payload = {"ok": True, **result, "revision": self.workspace.document().revision}
                if len(json.dumps(payload, allow_nan=False).encode()) > self.max_result_bytes:
                    raise ValueError("tool result exceeds the byte limit; request a smaller result")
                if mutation:
                    payload["checkpoint_id"] = self.workspace.checkpoints.add(checkpoint, previous_generated)
                    self.workspace.publish_candidate(capture_motion_state(session.working_document))
                self.workspace.emit(AgentEvent("tool_finished", name, payload["revision"]))
                return ToolOutcome(True, payload)
            except Exception as error:
                if checkpoint is not None:
                    session = self.workspace.candidate()
                    session.restore_checkpoint(checkpoint)
                    if previous_session is None:
                        self.workspace._session = None
                    self.workspace.generated = previous_generated
                    if previous_generated is not None and previous_generated.source == previous_identity:
                        self.workspace.generated = replace(previous_generated, source=self.workspace.identity())
                    self.workspace._validation_identity = (
                        self.workspace.identity() if validation_identity == previous_identity else None
                    )
                    self.workspace.checkpoints.entries = previous_checkpoints
                elif previous_session is None:
                    session = self.workspace.candidate()
                    if session is not None and not session.has_changes:
                        self.workspace._session = None
                self.workspace.emit(AgentEvent(
                    "tool_failed", name[:64] if isinstance(name, str) else "unknown",
                    self.workspace.document().revision, str(error)[:2000],
                ))
                return ToolOutcome(False, {
                    "ok": False, "code": "stale_revision" if "stale_revision" in str(error) else "command_failed",
                    "message": str(error)[:2000], "revision": self.workspace.document().revision,
                }, is_error=True)

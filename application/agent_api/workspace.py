"""Headless project and detached-candidate workflows shared by future clients."""

from collections import deque
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Protocol

from application.ai.edit_session import AIEditSession
from application.ai.metadata import (
    InMemoryMotionMetadataStore, MotionMetadataService, TimestampMotionIdentityResolver,
)
from application.ai.motion_services import GhostGUIMotionService
from application.ai.motion_state import ReplaceMotionState, capture_motion_state, detached_document
from application.ai.schemas import EditAuthor
from application.ai.trajectory_edit_spec import TrajectoryOperation, TrajectoryOperationType
from application.ai.trajectory_executor import TrajectoryExecutionContext
from application.ai.trajectory_operations import build_trajectory_operation_handlers
from application.backend_interface import BackendInterface, FallbackPolicy
from application.csv_io import TrajectoryExport, write_qpos_csv, write_trajectory_csv
from application.editor_controller import EditorController
from application.history import HistoryStack
from application.keyframe_service import plan_keyframe_commit
from application.project_document import ProjectDocument
from application.project_manager import GhostGUIProject, safe_project_path
from application.trajectory_export_formats import export_dsms_trajectory, export_mjlab_trajectory
from application.trajectory_export_service import (
    GeneratedMotion, build_trajectory_export, motion_source_identity,
)
from application.trajectory_generation import generate_trajectory_status
from application.workspace_state import default_workspace_payload, merge_workspace_payload
from core.models import MuJoCoRobotAdapter, ROBOT_MODELS, RobotStateTimeline

from .checkpoints import CheckpointRing
from .sandbox import OutputSandbox, safe_leaf
from .validation import validate_candidate


class Workspace(Protocol):
    """Thin front ends depend on local workflows, not Qt widgets."""
    committed: ProjectDocument
    adapter: MuJoCoRobotAdapter
    generated: GeneratedMotion | None
    models: dict
    approval: str
    allow_user_override: bool
    max_samples: int

    def candidate(self, *, create=False): ...
    def publish_candidate(self, snapshot): ...
    def emit(self, event): ...
    def document(self, source="candidate"): ...
    def identity(self, document=None): ...
    def status(self): ...
    def load_model(self, model_key): ...
    def new_project(self, name, model_key=None, duration_s=5.0): ...
    def open_project(self, name): ...
    def save_project(self): ...
    def commit_keyframe(self, time_s, phase=""): ...
    def operation(self, operation_type, arguments): ...
    def generate(self, dt=0.02, smoothing=0.0): ...
    def validate(self, dt=0.02, cancellation_token=None): ...
    def accept_candidate(self, *, user_approved=False): ...
    def discard_candidate(self): ...
    def export_trajectory(self, format, file_name, dt=0.02, source="auto", motion_speed=None): ...


class HeadlessWorkspace:
    def __init__(self, projects_dir, *, output_dir=None, model="g1", models=None,
                 approval="review", allow_user_override=False, max_samples=10001):
        if approval not in ("review", "auto_accept", "auto_accept_and_export"):
            raise ValueError("unknown approval policy")
        self.projects_dir = Path(projects_dir).expanduser().resolve()
        self.output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        self.models = dict(ROBOT_MODELS if models is None else models)
        self.approval = approval
        self.allow_user_override = bool(allow_user_override)
        self.max_samples = int(max_samples)
        self.events = deque(maxlen=256)
        self.checkpoints = CheckpointRing()
        self.history = HistoryStack()
        self.project = None
        self.workspace_payload = {}
        self.generated = None
        self._session = None
        self._install_model(model)

    def _install_model(self, key):
        if key not in self.models:
            raise ValueError(f"unknown model: {key}")
        adapter = MuJoCoRobotAdapter(self.models[key])
        document = ProjectDocument(key, qpos_timeline=RobotStateTimeline(adapter))
        self.adapter, self.committed = adapter, document
        self.controller = EditorController(document)
        self.motion = GhostGUIMotionService(adapter)
        self.metadata_store = InMemoryMotionMetadataStore()
        self.metadata = MotionMetadataService(self.metadata_store, TimestampMotionIdentityResolver())
        self.metadata.seed_document_as_user_owned(document)
        self._session = None
        self.generated = None
        self.checkpoints.clear()
        self.history.clear()

    def _transition_allowed(self):
        if self._session is not None and self._session.has_changes:
            raise ValueError("Accept or Discard the unresolved candidate first")
        if self.committed.dirty:
            raise ValueError("save the current committed project before changing projects/models")

    def load_model(self, model_key):
        self._transition_allowed()
        self._install_model(model_key)
        self.project, self.workspace_payload = None, {}
        return self.status()

    def new_project(self, name, model_key=None, duration_s=5.0):
        self._transition_allowed()
        if not 0 < float(duration_s) <= 120:
            raise ValueError("motion duration must be in (0, 120] seconds")
        path = safe_project_path(self.projects_dir, safe_leaf(name) + ".ghostgui")
        if path.exists():
            raise ValueError("project already exists; open it instead")
        key = model_key or self.committed.model_key
        replacement = HeadlessWorkspace(
            self.projects_dir, output_dir=self.output_dir, model=key, models=self.models,
            approval=self.approval, allow_user_override=self.allow_user_override,
            max_samples=self.max_samples,
        )
        replacement.committed.timeline_duration = float(duration_s)
        # A new empty project's initial home pose is app-created, not user work.
        for reference in replacement.metadata_store.snapshot():
            replacement.metadata_store.record(reference, EditAuthor.AI)
        replacement.project = GhostGUIProject.create(path, name, key, replacement.adapter.model_name)
        replacement.workspace_payload = default_workspace_payload(replacement.committed)
        replacement.save_project()
        self.__dict__.update(replacement.__dict__)
        return self.status()

    def open_project(self, name):
        self._transition_allowed()
        path = safe_project_path(self.projects_dir, safe_leaf(name) + ".ghostgui")
        project = GhostGUIProject.open(path)
        # Prepare the entire replacement first; malformed files leave this workspace intact.
        replacement = HeadlessWorkspace(
            self.projects_dir, output_dir=self.output_dir, model=project.model_key,
            models=self.models, approval=self.approval,
            allow_user_override=self.allow_user_override, max_samples=self.max_samples,
        )
        replacement.committed.trajectory.load_project_dict(project.read_trajectory_dict())
        project.load_qpos_timeline(replacement.committed.qpos_timeline)
        payload = project.read_workspace()
        replacement.committed.set_timeline_duration(payload.get("timeline_duration", 5.0))
        replacement.committed.set_current_time(payload.get("current_time", 0.0))
        replacement.committed.active_index = int(payload.get("active_index", -1))
        replacement.project = project
        replacement.workspace_payload = payload
        replacement.metadata.seed_document_as_user_owned(replacement.committed)
        metadata = payload.get("ai_motion_metadata")
        if metadata is not None:
            replacement.metadata.restore_project_dict(metadata, replacement.committed)
        self.__dict__.update(replacement.__dict__)
        return self.status()

    def save_project(self):
        if self._session is not None and self._session.has_changes:
            raise ValueError("accept or discard the candidate before saving committed motion")
        if self.project is None:
            raise ValueError("create or open a project before saving")
        payload = merge_workspace_payload(self.workspace_payload, {
            "current_time": self.committed.current_time,
            "timeline_duration": self.committed.timeline_duration,
            "active_index": self.committed.active_index,
            "ai_motion_metadata": self.metadata.to_project_dict(self.committed),
        })
        self.project.save_bundle(self.committed.trajectory, self.committed.qpos_timeline, payload)
        self.workspace_payload = payload
        self.controller.mark_saved()
        return {"saved": True, "project": self.project.project_name}

    def candidate(self, *, create=False):
        if self._session is None and create:
            self.metadata.seed_document_as_user_owned(self.committed)
            self._base_identity = self.identity(self.committed)
            self._session = AIEditSession(self.committed, metadata_store=self.metadata_store)
        return self._session

    def document(self, source="candidate"):
        if source == "candidate" and self._session is not None:
            return self._session.working_document
        return self.committed

    def identity(self, document=None):
        document = self.document() if document is None else document
        return motion_source_identity(self.adapter, document.qpos_timeline, document=document)

    def status(self):
        document = self.document()
        return {"model_key": document.model_key, "revision": document.revision,
                "candidate": self._session is not None, "duration_s": document.timeline_duration,
                "project": None if self.project is None else self.project.project_name,
                "generated": self.generated is not None}

    def emit(self, event):
        self.events.append(event)

    def publish_candidate(self, snapshot):
        # Headless execution has no observer. GUI adapters will consume detached snapshots.
        return None

    def operation(self, operation_type, arguments):
        session = self.candidate(create=True)
        operation = TrajectoryOperation(TrajectoryOperationType(operation_type), arguments)
        handlers = build_trajectory_operation_handlers(self.motion, self.metadata)
        return dict(handlers[operation.operation_type](
            operation, TrajectoryExecutionContext(session, self),
        ))

    def apply_document(self, candidate, operation):
        session = self.candidate(create=True)
        before = self.entity_values(session.working_document)
        after = self.entity_values(candidate)
        affected = tuple(ref for ref in before.keys() | after.keys() if before.get(ref) != after.get(ref))
        created = tuple(ref for ref in affected if ref not in before)
        result = session.apply_ai(
            ReplaceMotionState(capture_motion_state(candidate), operation=operation),
            affected_entities=affected, created_entities=created,
            allow_user_override=self.allow_user_override,
        )
        return {"changed": result.changed}

    def entity_values(self, document):
        values = {self.metadata.reference_for_keyframe(frame): frame.to_dict()
                  for frame in document.trajectory.frames}
        values.update({self.metadata.reference_for_qpos_keyframe(time):
                       tuple(document.qpos_timeline.get_state(time))
                       for time in document.qpos_timeline.times()})
        return values

    def commit_keyframe(self, time_s, phase=""):
        document = self.candidate(create=True).working_document
        candidate = detached_document(document)
        plan_keyframe_commit(candidate, self.adapter, time_s, phase=phase).execute(candidate)
        return self.apply_document(candidate, "commit_keyframe")

    def generate(self, dt=0.02, smoothing=0.0):
        document = self.candidate(create=True).working_document
        if document.timeline_duration / dt + 1 > self.max_samples:
            raise ValueError("generation sample budget exceeded")
        source = self.identity(document)
        candidate = detached_document(document)
        backend = BackendInterface(mj_model=self.adapter.mj_model, adapter=self.adapter,
                                   fallback_policy=FallbackPolicy.ERROR)
        with tempfile.TemporaryDirectory(prefix="ghostgui-agent-generate-") as directory:
            result = generate_trajectory_status(
                candidate.trajectory, backend, smoothing=smoothing, export_dt=dt,
                csv_path=Path(directory) / "generated.csv", state_timeline=candidate.qpos_timeline,
            )
        if self.identity(document) != source:
            raise ValueError("motion changed during generation")
        qposes = tuple(state.qpos.copy() for state in result.result_states)
        for qpos in qposes:
            qpos.setflags(write=False)
        self.generated = GeneratedMotion(TrajectoryExport(
            self.adapter.mj_model.nq, tuple(state.time for state in result.result_states),
            qposes, "generated trajectory", False,
        ), source)
        document_source = self.identity(document)
        self.candidate().invalidate_validation()
        self._validation_identity = None
        return {"samples": len(qposes), "sample_dt": dt, "source_revision": document_source.revision}

    def validate(self, dt=0.02, cancellation_token=None):
        document = self.document()
        source = self.identity(document)
        if self.generated is not None and self.generated.source != source:
            raise ValueError("generated motion is stale; Generate again")
        report = validate_candidate(
            self.adapter, self.motion, document, generated=self.generated, dt=dt,
            max_samples=self.max_samples, cancellation_token=cancellation_token,
        )
        if self.identity(document) != source:
            raise ValueError("motion changed during validation")
        session = self.candidate()
        if session is not None:
            session.invalidate_validation()
            if report["valid"]:
                session.mark_current_revision_validated()
        self._validation_identity = source if report["valid"] else None
        return report

    def accept_candidate(self, *, user_approved=False):
        session = self.candidate()
        if session is None:
            raise ValueError("no candidate to accept")
        if self.identity(self.committed) != self._base_identity:
            raise ValueError("committed motion changed; candidate is stale")
        if getattr(self, "_validation_identity", None) != self.identity():
            raise ValueError("candidate requires current sampled validation")
        # Protection may have changed in the committed metadata since the session began.
        for edit in session.edits:
            for reference in edit.affected_entities:
                metadata = self.metadata_store.get(reference)
                if metadata is not None and (
                    metadata.protected or
                    (metadata.author is EditAuthor.USER and not self.allow_user_override)
                ):
                    raise ValueError("candidate affects protected or user-owned content")
        if self.approval == "review" and not user_approved:
            return {"status": "pending_user_review"}
        before = (capture_motion_state(self.committed), self.metadata_store.snapshot())
        session.accept(self.controller, commit_checkpoint=lambda: self.history.record(
            "Accept agent candidate", before=before,
            after=(capture_motion_state(self.committed), self.metadata_store.snapshot()),
        ))
        self._session = None
        self.checkpoints.clear()
        if self.generated is not None:
            self.generated = replace(self.generated, source=self.identity(self.committed))
        return {"status": "accepted", "revision": self.committed.revision}

    def discard_candidate(self):
        if self._session is not None:
            self._session.reject()
        self._session = None
        self.generated = None
        self.checkpoints.clear()
        self._validation_identity = None
        return {"status": "discarded"}

    def export_trajectory(self, format, file_name, dt=0.02, source="auto", motion_speed=None):
        if self._session is not None:
            raise ValueError("accept the candidate before exporting committed motion")
        root = self.output_dir or (self.project.root_dir / "exports" if self.project else None)
        if root is None:
            raise ValueError("create a project or configure an output directory")
        sandbox = OutputSandbox(root)
        path = sandbox.resolve_output(file_name, suffix="" if format == "dsms" else ".csv")
        if path.exists():
            raise ValueError("output already exists; choose a new name")
        if format == "qpos_csv":
            pose = self.committed.qpos_timeline.sample_state(self.committed.current_time)
            return {"paths": [str(write_qpos_csv(path, pose))], "source": "committed pose", "samples": 1}
        export = build_trajectory_export(
            self.adapter, self.committed.qpos_timeline, self.generated, source=source,
            current_identity=self.identity(self.committed), sample_dt=dt,
        ).export
        if format == "mujoco_csv":
            paths = (write_trajectory_csv(path, export),)
        elif format == "mjlab_csv":
            paths = export_mjlab_trajectory(path, export, self.adapter, single_sample_dt=dt).paths
        elif format == "dsms":
            roots = tuple(self.adapter.free_joints_by_body.values())
            paths = export_dsms_trajectory(
                path, export, dof=len(self.adapter.actuated_joints),
                base_qpos_address=None if not roots else roots[0].qpos_address,
                motion_speed=(self.workspace_payload.get("display", {}).get("dsms_motion_speed", 1.0)
                              if motion_speed is None else motion_speed),
            ).paths
        else:
            raise ValueError("unsupported export format")
        return {"paths": [str(path) for path in paths], "source": export.source_name,
                "samples": len(export.times)}

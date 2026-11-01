"""Immutable cross-thread motion publications. No mutable document crosses."""
from dataclasses import dataclass, replace
import json

from application.ai.motion_state import MotionStateSnapshot, ReplaceMotionState, capture_motion_state
from application.project_document import ProjectDocument
from core.models import RobotStateTimeline


@dataclass(frozen=True)
class FrozenMotion:
    payload: str
    document_id: str
    revision: int
    dirty: bool

    @classmethod
    def capture(cls, document):
        state = capture_motion_state(document)
        return cls(json.dumps({
            'model_key': state.model_key, 'trajectory': state.trajectory,
            'active_index': state.active_index, 'current_time': state.current_time,
            'timeline_duration': state.timeline_duration,
            'qpos_states': [(time, qpos.tolist()) for time, qpos in state.qpos_states],
        }, allow_nan=False), document.document_id, document.revision, document.dirty)

    def with_qpos(self, states):
        """A derived preview still carries no writable arrays across threads."""
        values = json.loads(self.payload)
        values['qpos_states'] = [(float(time), [float(v) for v in qpos]) for time, qpos in states]
        return replace(self, payload=json.dumps(values, allow_nan=False))

    def document(self, adapter):
        values = json.loads(self.payload)
        state = MotionStateSnapshot(**values)
        document = ProjectDocument(values['model_key'], qpos_timeline=RobotStateTimeline(adapter))
        ReplaceMotionState(state).execute(document)
        document.document_id = self.document_id
        document.revision, document.dirty = self.revision, self.dirty
        return document

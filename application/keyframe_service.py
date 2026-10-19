"""Atomic logical-target and qpos Keyframe capture, independent of Qt."""

from application.ai.motion_state import ReplaceMotionState, capture_motion_state, detached_document
from application.editor_commands import UpsertKeyframes
from application.timeslice_service import capture_timeslice_from_committed_pose
from application.ai.qpos_trajectory import validate_qpos_anchors


def plan_keyframe_commit(document, adapter, time, *, qpos=None, phase="", selected_frame=None):
    """Build one command; all FK and validation happen on detached values."""
    time = float(time)
    if not 0 <= time <= document.timeline_duration:
        raise ValueError("Keyframe time is outside the motion duration")
    if document.qpos_timeline is None:
        raise ValueError("Commit Keyframe requires a qpos timeline")
    if qpos is None:
        qpos = document.qpos_timeline.sample_state(time)
    qpos = validate_qpos_anchors(adapter, [{"time_seconds": time, "qpos": qpos}])[0].qpos
    state = adapter.create_state()
    state.set_qpos(qpos)
    frames = capture_timeslice_from_committed_pose(
        state, time=time, phase=phase, frame_names=adapter.trajectory_frames,
        frame_bindings=adapter.logical_frame_bindings,
    )
    candidate = detached_document(document)
    if frames:
        UpsertKeyframes(frames, selected_frame_name=selected_frame).execute(candidate)
        if selected_frame is not None:
            candidate.active_index = next(
                (index for index, frame in enumerate(candidate.trajectory.frames)
                 if frame.frame_name == selected_frame and abs(frame.time - time) <= 1e-6),
                candidate.active_index,
            )
        candidate.qpos_timeline.set_state(time, qpos)
    return ReplaceMotionState(capture_motion_state(candidate), operation="commit_keyframe")


def commit_keyframe(controller, adapter, time, **options):
    return controller.execute(plan_keyframe_commit(controller.document, adapter, time, **options))

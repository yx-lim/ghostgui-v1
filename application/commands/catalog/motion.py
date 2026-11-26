"""Candidate-side timeline, clipboard, pose, project-list and import commands (Qt-free).

Every edit detaches the candidate, runs the same application planner the GUI uses,
and applies it through ``workspace.apply_document`` so provenance, protection and
structural validation gates apply. Committed motion changes only through Accept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from application.ai.motion_state import detached_document
from application.timeline_editing import (
    ApplyTimelineEditPlan, plan_insert_time, plan_move_time_range, plan_shift_entire_motion,
)
from core.trajectory import Trajectory

from ..limits import Name, TimeSeconds
from ..sanitize import untrusted
from ..schema import Doc, Range
from ..spec import Binding, Effect, Surface

MAX_TIME = 120.0
EDIT_SURFACES = frozenset({Surface.AGENT_INAPP, Surface.MCP_ATTACHED, Surface.PALETTE})
READ_SURFACES = frozenset({Surface.AGENT_INAPP, Surface.MCP_ATTACHED, Surface.PALETTE})
Duration = Annotated[float, Range(0.01, MAX_TIME), Doc("Seconds.")]
Offset = Annotated[float, Range(-MAX_TIME, MAX_TIME), Doc("Seconds; negative moves earlier.")]


def _candidate(ctx):
    return detached_document(ctx.workspace.candidate(create=True).working_document)


def _apply_plan(ctx, plan, operation):
    candidate = _candidate(ctx)
    ApplyTimelineEditPlan(plan).execute(candidate)
    return ctx.workspace.apply_document(candidate, operation)


def insert_time(ctx, at_s: TimeSeconds, duration_s: Duration):
    """Insert a held pause of duration_s at at_s on the Orange preview, shifting later motion right.
    Call when the user asks to make room, add a pause or hold a pose longer."""
    return _apply_plan(ctx, plan_insert_time(_candidate(ctx), at_s, duration_s, maximum_time=MAX_TIME),
                       "insert_time")


def shift_motion(ctx, offset_s: Offset):
    """Shift the entire motion earlier or later in time on the Orange preview."""
    return _apply_plan(ctx, plan_shift_entire_motion(_candidate(ctx), offset_s, maximum_time=MAX_TIME),
                       "shift_motion")


def move_time_range(ctx, start_s: TimeSeconds, end_s: TimeSeconds, destination_s: TimeSeconds):
    """Move the Keyframes in [start_s, end_s] so the range starts at destination_s (Orange preview)."""
    return _apply_plan(ctx, plan_move_time_range(_candidate(ctx), start_s, end_s, destination_s,
                                                 maximum_time=MAX_TIME), "move_time_range")


def set_motion_duration(ctx, duration_s: Duration):
    """Set the motion's timeline duration on the Orange preview. It cannot cut off existing Keyframes."""
    candidate = _candidate(ctx)
    times = [frame.time for frame in candidate.trajectory.frames] + list(candidate.qpos_timeline.times())
    if times and max(times) > duration_s + 1e-9:
        raise ValueError(f"a Keyframe exists at {max(times):.3f} s; delete or move it first")
    candidate.timeline_duration = float(duration_s)
    return ctx.workspace.apply_document(candidate, "set_motion_duration")


def copy_motion_range(ctx, start_s: TimeSeconds, end_s: TimeSeconds,
                      source: Literal["candidate", "committed"] = "candidate"):
    """Copy the motion in [start_s, end_s] to the agent clipboard for paste_motion."""
    from application.motion_clipboard import capture_motion_clip

    clip = capture_motion_clip(ctx.workspace.document(source), start_s, end_s)
    ctx.workspace.agent_clipboard = clip
    return {"copied": True, "start_s": start_s, "end_s": end_s, "model_key": untrusted(clip.model_key)}


def paste_motion(ctx, at_s: TimeSeconds, reverse: bool = False):
    """Paste the copied motion starting at at_s on the Orange preview, optionally time-reversed."""
    from application.motion_clipboard import plan_paste_motion

    clip = getattr(ctx.workspace, "agent_clipboard", None)
    candidate = _candidate(ctx)
    if clip is None or getattr(clip, "model_key", None) != candidate.model_key:
        raise ValueError("copy a motion range for the active robot first (copy_motion_range)")
    plan = plan_paste_motion(candidate, clip, at_s, reverse=reverse, maximum_time=MAX_TIME)
    return _apply_plan(ctx, plan, "paste_motion")


def clear_motion(ctx):
    """Remove every Keyframe from the Orange preview, keeping only the pose at 0 s."""
    candidate = _candidate(ctx)
    start = candidate.qpos_timeline.sample_state(0.0)
    for time in list(candidate.qpos_timeline.times()):
        candidate.qpos_timeline.delete_state(time)
    candidate.qpos_timeline.set_state(0.0, start)
    candidate.trajectory = Trajectory()
    candidate.active_index = -1
    return ctx.workspace.apply_document(candidate, "clear_motion")


def reset_pose(ctx, time_s: TimeSeconds):
    """Reset the pose at time_s to the robot's home pose on the Orange preview."""
    home = getattr(ctx.workspace.adapter, "home_qpos", None)
    if home is None:
        raise ValueError("this robot has no home pose")
    candidate = _candidate(ctx)
    if time_s > candidate.timeline_duration + 1e-9:
        raise ValueError("time is past the motion duration")
    time = float(time_s)
    candidate.qpos_timeline.set_state(time, [float(value) for value in home])
    if any(abs(frame.time - time) <= 1e-6 for frame in candidate.trajectory.frames):
        # Keep the logical Keyframe consistent with the new pose (FK), as the GUI does.
        from application.keyframe_service import plan_keyframe_commit
        plan_keyframe_commit(candidate, ctx.workspace.adapter, time).execute(candidate)
    return ctx.workspace.apply_document(candidate, "reset_pose")


def list_projects(ctx):
    """List projects in the GhostGUI projects folder (names only)."""
    root = Path(ctx.workspace.projects_dir)
    names = sorted(path.stem for path in root.glob("*.ghostgui"))[:100] if root.is_dir() else []
    return {"projects": [untrusted(name) for name in names]}


def _imports_dir(workspace) -> Path:
    project = getattr(workspace, "project", None)
    return (Path(project.root_dir) if project is not None else Path(workspace.projects_dir)) / "imports"


def import_file(ctx, kind: Literal["qpos", "trajectory"], file_name: Name,
                time_s: TimeSeconds | None = None):
    """Import a CSV from the project's imports folder onto the Orange preview. kind=qpos sets one pose at
    time_s; kind=trajectory replaces the preview with up to 64 evenly spaced Keyframes from the file.
    Only plain file names inside the imports folder are accepted."""
    from application.agent_api.sandbox import safe_leaf
    from application.csv_io import read_qpos_csv, read_trajectory_csv

    workspace = ctx.workspace
    folder = _imports_dir(workspace).resolve()
    leaf = safe_leaf(Path(file_name).stem) + ".csv"
    path = (folder / leaf).resolve()
    if path.parent != folder or not path.is_file():
        available = sorted(item.name for item in folder.glob("*.csv"))[:20] if folder.is_dir() else []
        raise ValueError(f"no {leaf} in the imports folder; available: {[untrusted(n) for n in available]}")
    nq = int(workspace.adapter.mj_model.nq)
    if kind == "qpos":
        loaded = read_qpos_csv(path, nq)
        candidate = _candidate(ctx)
        time = 0.0 if time_s is None else float(time_s)
        candidate.qpos_timeline.set_state(time, [float(value) for value in loaded.qpos])
        return {**workspace.apply_document(candidate, "import_qpos"), "file": leaf}
    loaded = read_trajectory_csv(path, nq)
    times, qposes = list(loaded.times), [list(map(float, row)) for row in loaded.qposes]
    if len(times) < 2:
        raise ValueError("the trajectory needs at least two samples")
    step = max(1, (len(times) - 1) // 63)
    picks = list(range(0, len(times), step))
    if picks[-1] != len(times) - 1:
        picks.append(len(times) - 1)
    start = times[0]
    keyframes = [{"time_seconds": round(times[i] - start, 6), "qpos": qposes[i]} for i in picks[:64]]
    duration = keyframes[-1]["time_seconds"]
    if duration <= 0 or duration > MAX_TIME:
        raise ValueError("trajectory duration must be within (0, 120] seconds")
    result = workspace.operation("qpos_keyframes", {
        "mode": "replace", "duration_seconds": duration, "start_time": 0.0, "end_time": duration,
        "keyframes": keyframes})
    return {"operation": result, "file": leaf, "keyframes": len(keyframes), "duration_s": duration}


def register(registry):
    edit = dict(effect=Effect.CANDIDATE, surfaces=EDIT_SURFACES, mutates_candidate=True)
    command = registry.command
    command("timeline.insert_time", title="Insert Time", tool="insert_time",
            gui=(Binding("insertTimelineTimeAction"),), **edit)(insert_time)
    command("timeline.shift", title="Shift Entire Motion", tool="shift_motion",
            gui=(Binding("shiftEntireMotionAction"),), **edit)(shift_motion)
    command("timeline.move_range", title="Move Time Range", tool="move_time_range",
            gui=(Binding("moveTimelineRangeAction"),), **edit)(move_time_range)
    command("timeline.duration", title="Set Motion Duration", tool="set_motion_duration", **edit)(set_motion_duration)
    command("clipboard.copy", title="Copy Motion Range", tool="copy_motion_range", effect=Effect.UI,
            surfaces=EDIT_SURFACES, gui=(Binding("copyMotionRangeAction"),))(copy_motion_range)
    command("clipboard.paste", title="Paste Motion", tool="paste_motion",
            gui=(Binding("pasteMotionAction"), Binding("pasteMotionReversedAction", {"reverse": True})),
            **edit)(paste_motion)
    command("motion.clear", title="Clear Motion", tool="clear_motion",
            gui=(Binding("clearToolbarButtonAction"), Binding("clearToolbarButton")), **edit)(clear_motion)
    command("pose.reset", title="Reset Pose", tool="reset_pose",
            gui=(Binding("resetToolbarButtonAction"), Binding("resetToolbarButton")), **edit)(reset_pose)
    command("projects.list", title="List Projects", tool="list_projects", effect=Effect.READ,
            surfaces=READ_SURFACES)(list_projects)
    command("imports.file", title="Import File", tool="import_file",
            gui=(Binding("importQposAction", {"kind": "qpos"}),
                 Binding("importTrajectoryAction", {"kind": "trajectory"})), **edit)(import_file)
    return registry

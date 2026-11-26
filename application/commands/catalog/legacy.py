"""The 24 shipped agent tools, migrated into the registry with byte-identical names and schemas.

``tests/fixtures/legacy_agent_tools.json`` pins the MCP projection; change it only deliberately.
"""

from __future__ import annotations

from dataclasses import replace

from application.ai.motion_state import detached_document
from application.ai.trajectory_edit_spec import parse_trajectory_edit_spec
from application.timeline_editing import ApplyTimelineEditPlan, plan_scale_time_range
from core.trajectory import Trajectory

from ..registry import CommandRegistry
from ..spec import Binding, CommandSpec, Effect, LEGACY_SURFACES, Surface


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def number(minimum=0, maximum=120):
    return {"type": "number", "minimum": minimum, "maximum": maximum}


def text(maximum=80):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def array(items, minimum=0, maximum=128):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def choice(*values):
    return {"type": "string", "enum": list(values)}


REVISION = {"type": "integer", "minimum": 0}
# get_status duplicates get_app_state in-app; keep it MCP-only.
MCP_ONLY = frozenset({Surface.MCP_ATTACHED, Surface.MCP_HEADLESS})


def _list_models(ctx):
    return {"models": [{"key": key, "name": info.display_name} for key, info in ctx.workspace.models.items()]}


def _robot_summary(ctx):
    w = ctx.workspace
    return {
        "model_key": w.committed.model_key, "nq": w.adapter.mj_model.nq,
        "joints": list(w.motion.joint_names), "logical_frames": list(w.motion.logical_frames),
        "end_effectors": list(w.motion.end_effectors),
    }


def _list_keyframes(ctx, source="candidate", limit=64):
    document = ctx.workspace.document(source)
    frames = document.trajectory.frames
    return {"frames": [frame.to_dict() for frame in frames[:limit]], "total": len(frames),
            "qpos_times": document.qpos_timeline.times()[:limit],
            "qpos_count": len(document.qpos_timeline.times())}


def _get_pose(ctx, time_s, source="candidate"):
    return {"qpos": ctx.workspace.document(source).qpos_timeline.sample_state(time_s).tolist()}


def _author(ctx, keyframes, duration_s=None):
    w = ctx.workspace
    duration = w.document().timeline_duration if duration_s is None else duration_s
    return w.operation("qpos_keyframes", {
        "mode": "replace", "duration_seconds": duration, "start_time": 0.0,
        "end_time": duration, "keyframes": keyframes,
    })


def _joints(ctx, time_s, values):
    names = [value["joint"] for value in values]
    if len(names) != len(set(names)):
        raise ValueError("Joint Angles must have unique names")
    return {"operations": [ctx.workspace.operation("set_joint_target", {
        "time_seconds": time_s, "joint": value["joint"], "angle_rad": value["angle_rad"],
    }) for value in values]}


def _solve(ctx, time_s, logical_frame, position_m, orientation_rpy_rad=None, locked_end_effectors=()):
    return ctx.workspace.operation("set_logical_frame_target", {
        "logical_frame": logical_frame, "start_time": time_s, "end_time": time_s, "mode": "absolute",
        "position_m": position_m, "orientation_rpy_rad": orientation_rpy_rad,
        "locked_end_effectors": list(locked_end_effectors),
    })


def _repeat(ctx, start_s, end_s, additional_copies, ping_pong=False):
    return ctx.workspace.operation("repeat_motion", {
        "start_time": start_s, "end_time": end_s,
        "additional_copies": additional_copies, "ping_pong": ping_pong,
    })


def _delete(ctx, times):
    w = ctx.workspace
    candidate = detached_document(w.candidate(create=True).working_document)
    selected = set(times)
    frames = [frame for frame in candidate.trajectory.frames if frame.time not in selected]
    candidate.trajectory = Trajectory()
    for frame in frames:
        candidate.trajectory.add_frame(frame)
    for time in selected:
        candidate.qpos_timeline.delete_state(time)
    candidate.active_index = -1
    return w.apply_document(candidate, "delete_keyframes")


def _scale(ctx, start_s, end_s, scale):
    w = ctx.workspace
    candidate = detached_document(w.candidate(create=True).working_document)
    plan = plan_scale_time_range(candidate, start_s, end_s, 1.0 / scale, maximum_time=120)
    ApplyTimelineEditPlan(plan).execute(candidate)
    return w.apply_document(candidate, "scale_time_range")


def _apply_spec(ctx, spec_json):
    spec = parse_trajectory_edit_spec(spec_json)
    return {"operations": [ctx.workspace.operation(op.operation_type, op.arguments) for op in spec.operations]}


def _undo(ctx, checkpoint_id=None):
    w = ctx.workspace
    session = w.candidate()
    if session is None:
        raise ValueError("no candidate to restore")
    checkpoint, generated = w.checkpoints.get(checkpoint_id)
    session.restore_checkpoint(checkpoint)
    w.checkpoints.discard_from(checkpoint_id)
    w.generated = generated
    if generated is not None:
        w.generated = replace(generated, source=w.identity())
    w._validation_identity = None
    session.invalidate_validation()
    return {"restored": True}


def _render(ctx, **arguments):
    from application.agent_rendering import MuJoCoViewRenderer, render_workspace
    if ctx.renderer is None:
        ctx.renderer = MuJoCoViewRenderer()
    return render_workspace(ctx.workspace, arguments, ctx.renderer)


def _spec(id, tool, title, description, params, handler, effect, *, mutate=False, app=False,
          surfaces=LEGACY_SURFACES, **options):
    if mutate:
        params = {**params, "properties": {**params["properties"], "expected_revision": REVISION}}
    return CommandSpec(id=id, tool=tool, title=title, description=description, handler=handler,
                       params=params, effect=effect, surfaces=surfaces, mutates_candidate=mutate,
                       app_level=app, **options)


def legacy_specs() -> tuple[CommandSpec, ...]:
    """In MCP registration order (pinned by the golden snapshot)."""
    w = lambda name: (lambda ctx, **a: getattr(ctx.workspace, name)(**a))  # noqa: E731
    return (
        _spec("models.list", "list_models", "List Models", "List available robot models.",
              obj(), _list_models, Effect.READ),
        _spec("workspace.status", "get_status", "Workspace Status", "Inspect workspace and candidate revision.",
              obj(), lambda ctx: ctx.workspace.status(), Effect.READ, surfaces=MCP_ONLY,
              exposure_reason="get_app_state replaces it in-app"),
        _spec("models.load", "load_model", "Load Robot Model",
              "Load a registered model when no unsaved motion is unresolved.",
              obj({"model_key": text()}, ("model_key",)), w("load_model"), Effect.TRANSITION, app=True),
        _spec("project.new", "new_project", "New Project",
              "Create a new project; never overwrite an existing project.",
              obj({"name": text(), "model_key": text(), "duration_s": number(0.01)}, ("name",)),
              w("new_project"), Effect.TRANSITION, app=True, gui=(Binding("newProjectAction"),)),
        _spec("project.open", "open_project", "Open Project", "Open an existing project by local project name.",
              obj({"name": text()}, ("name",)), w("open_project"), Effect.TRANSITION, app=True,
              gui=(Binding("openProjectAction"),)),
        _spec("project.save", "save_project", "Save Project",
              "Save accepted committed motion, preserving workspace fields.",
              obj(), lambda ctx: ctx.workspace.save_project(), Effect.PERSIST, app=True,
              gui=(Binding("saveProjectAction"),)),
        _spec("robot.summary", "get_robot_summary", "Robot Summary",
              "Inspect the compiled robot and named End Effectors/Joint Angles.",
              obj(), _robot_summary, Effect.READ),
        _spec("keyframes.list", "list_keyframes", "List Keyframes",
              "List committed or candidate Keyframes with a bounded result.",
              obj({"source": choice("candidate", "committed"),
                   "limit": {"type": "integer", "minimum": 1, "maximum": 128}}), _list_keyframes, Effect.READ),
        _spec("pose.get", "get_pose", "Get Pose", "Sample a complete qpos without creating a Keyframe.",
              obj({"time_s": number(), "source": choice("candidate", "committed")}, ("time_s",)),
              _get_pose, Effect.READ),
        _spec("keyframes.author", "author_keyframes", "Author Keyframes",
              "Author sparse complete qpos anchors on the detached Orange preview.",
              obj({"duration_s": number(0.01), "keyframes": array(obj({
                  "time_seconds": number(), "qpos": array(number(-1e6, 1e6), 1, 512),
              }, ("time_seconds", "qpos")), 2, 16)}, ("keyframes",)), _author, Effect.CANDIDATE, mutate=True),
        _spec("keyframes.commit", "commit_keyframe", "Commit Keyframe",
              "Commit Keyframe on the candidate using FK of its sampled pose.",
              obj({"time_s": number(), "phase": text()}, ("time_s",)),
              w("commit_keyframe"), Effect.CANDIDATE, mutate=True),
        _spec("pose.set_joint_angles", "set_joint_angles", "Set Joint Angles",
              "Set named Joint Angles in radians on the candidate.",
              obj({"time_s": number(), "values": array(obj({"joint": text(),
                   "angle_rad": number(-100, 100)}, ("joint", "angle_rad")), 1, 128)}, ("time_s", "values")),
              _joints, Effect.CANDIDATE, mutate=True),
        _spec("pose.solve_ik", "solve_ik", "Solve IK",
              "Solve a logical frame/End Effector pose in meters and radians.",
              obj({"time_s": number(), "logical_frame": text(),
                   "position_m": array(number(-100, 100), 3, 3),
                   "orientation_rpy_rad": array(number(-100, 100), 3, 3),
                   "locked_end_effectors": array(text(), 0, 16)}, ("time_s", "logical_frame", "position_m")),
              _solve, Effect.CANDIDATE, mutate=True),
        _spec("timeline.repeat", "repeat_motion", "Repeat Motion",
              "Append exact copies of a candidate motion interval.",
              obj({"start_s": number(), "end_s": number(),
                   "additional_copies": {"type": "integer", "minimum": 1, "maximum": 32},
                   "ping_pong": {"type": "boolean"}}, ("start_s", "end_s", "additional_copies")),
              _repeat, Effect.CANDIDATE, mutate=True),
        _spec("keyframes.delete", "delete_keyframes", "Delete Keyframes",
              "Delete logical and qpos Keyframes atomically on the candidate.",
              obj({"times": array(number(), 1, 128)}, ("times",)), _delete, Effect.CANDIDATE, mutate=True),
        _spec("timeline.scale", "edit_timeline", "Scale Time Range",
              "Scale a candidate time interval through the shared planner.",
              obj({"start_s": number(), "end_s": number(), "scale": number(0.01, 100)},
                  ("start_s", "end_s", "scale")), _scale, Effect.CANDIDATE, mutate=True),
        _spec("motion.apply_edit_spec", "apply_edit_spec", "Apply Edit Spec",
              "Apply a validated compact motion spec locally; no nested LLM call.",
              obj({"spec_json": text(24000)}, ("spec_json",)), _apply_spec, Effect.CANDIDATE, mutate=True),
        _spec("motion.generate", "generate_trajectory", "Generate Trajectory",
              "Generate a detached candidate path using existing exact backend contracts.",
              obj({"dt": number(0.001, 1), "smoothing": number(0, 1)}),
              lambda ctx, **a: ctx.workspace.generate(**a), Effect.CANDIDATE, app=True),
        _spec("motion.validate", "validate_motion", "Validate Motion",
              "Check structure, limits and collisions over the sampled candidate path.",
              obj({"dt": number(0.001, 1)}), lambda ctx, **a: ctx.workspace.validate(**a), Effect.READ),
        _spec("candidate.undo", "undo", "Undo Candidate Step",
              "Restore an owned candidate checkpoint; committed history is untouched.",
              obj({"checkpoint_id": text()}), _undo, Effect.CANDIDATE, app=True),
        _spec("candidate.accept", "accept_candidate", "Accept",
              "Accept validated motion under the caller-configured review policy.",
              obj(), lambda ctx: ctx.workspace.accept_candidate(), Effect.COMMIT, app=True),
        _spec("candidate.discard", "discard_candidate", "Discard",
              "Discard the Orange preview without changing committed motion.",
              obj(), lambda ctx: ctx.workspace.discard_candidate(), Effect.CANDIDATE, app=True),
        _spec("export.trajectory", "export_trajectory", "Export Trajectory",
              "Export accepted motion through unchanged legacy format writers.",
              obj({"format": choice("mujoco_csv", "qpos_csv", "mjlab_csv", "dsms"),
                   "file_name": text(), "dt": number(0.001, 1), "motion_speed": number(0.1, 4),
                   "source": choice("auto", "generated", "editable")}, ("format", "file_name")),
              w("export_trajectory"), Effect.PERSIST, app=True),
        _spec("views.render", "render_views", "Render Views",
              "Render timestamped front/side views of current motion; reports unavailable GL explicitly.",
              obj({"times_s": array(number(), 1, 4), "views": array(choice("front", "side"), 1, 2),
                   "width": {"type": "integer", "minimum": 160, "maximum": 640},
                   "height": {"type": "integer", "minimum": 120, "maximum": 480},
                   "source": choice("candidate", "committed")}),
              _render, Effect.READ, returns_images=True),
    )


def register_legacy(registry: CommandRegistry) -> None:
    for spec in legacy_specs():
        registry.register(spec)

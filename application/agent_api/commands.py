"""Closed-schema deterministic command catalog; no model or transport calls."""

from dataclasses import dataclass

from application.ai.motion_state import detached_document
from application.ai.tool_registry import ToolCategory, ToolRegistry, ToolSpec
from application.ai.trajectory_edit_spec import parse_trajectory_edit_spec
from application.timeline_editing import ApplyTimelineEditPlan, plan_scale_time_range
from core.trajectory import Trajectory


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


@dataclass(frozen=True)
class ToolMeta:
    read_only: bool = False
    mutates_candidate: bool = False
    app_level: bool = False
    returns_images: bool = False


class CommandCatalog:
    def __init__(self):
        self.registry = ToolRegistry()
        self.meta = {}


def build_command_catalog(workspace):
    catalog = CommandCatalog()
    w = workspace

    def register(name, description, properties, required, handler, *, mutate=False, app=False):
        properties = dict(properties)
        if mutate:
            properties["expected_revision"] = {"type": "integer", "minimum": 0}
        catalog.registry.register(ToolSpec(
            name, description, obj(properties, required), lambda args, context: handler({
                key: value for key, value in args.items() if key != "expected_revision"
            }), ToolCategory.EDIT if mutate or app else ToolCategory.INSPECT,
            mutates_working_copy=mutate,
        ))
        catalog.meta[name] = ToolMeta(not mutate and not app, mutate, app)

    register("list_models", "List available robot models.", {}, (), lambda a: {
        "models": [{"key": key, "name": info.display_name} for key, info in w.models.items()]
    })
    register("get_status", "Inspect workspace and candidate revision.", {}, (), lambda a: w.status())
    register("load_model", "Load a registered model when no unsaved motion is unresolved.",
             {"model_key": text()}, ("model_key",), lambda a: w.load_model(**a), app=True)
    register("new_project", "Create a new project; never overwrite an existing project.",
             {"name": text(), "model_key": text(), "duration_s": number(0.01)},
             ("name",), lambda a: w.new_project(**a), app=True)
    register("open_project", "Open an existing project by local project name.",
             {"name": text()}, ("name",), lambda a: w.open_project(**a), app=True)
    register("save_project", "Save accepted committed motion, preserving workspace fields.",
             {}, (), lambda a: w.save_project(), app=True)
    register("get_robot_summary", "Inspect the compiled robot and named End Effectors/Joint Angles.",
             {}, (), lambda a: {
                 "model_key": w.committed.model_key, "nq": w.adapter.mj_model.nq,
                 "joints": list(w.motion.joint_names), "logical_frames": list(w.motion.logical_frames),
                 "end_effectors": list(w.motion.end_effectors),
             })

    def list_keyframes(args):
        document = w.document(args.get("source", "candidate"))
        frames = document.trajectory.frames
        limit = args.get("limit", 64)
        return {"frames": [frame.to_dict() for frame in frames[:limit]], "total": len(frames),
                "qpos_times": document.qpos_timeline.times()[:limit],
                "qpos_count": len(document.qpos_timeline.times())}

    register("list_keyframes", "List committed or candidate Keyframes with a bounded result.",
             {"source": choice("candidate", "committed"),
              "limit": {"type": "integer", "minimum": 1, "maximum": 128}}, (), list_keyframes)
    register("get_pose", "Sample a complete qpos without creating a Keyframe.",
             {"time_s": number(), "source": choice("candidate", "committed")},
             ("time_s",), lambda a: {
                 "qpos": w.document(a.get("source", "candidate")).qpos_timeline.sample_state(a["time_s"]).tolist()
             })

    def author(args):
        duration = args.get("duration_s", w.document().timeline_duration)
        return w.operation("qpos_keyframes", {
            "mode": "replace", "duration_seconds": duration, "start_time": 0.0,
            "end_time": duration, "keyframes": args["keyframes"],
        })

    register("author_keyframes", "Author sparse complete qpos anchors on the detached Orange preview.",
             {"duration_s": number(0.01), "keyframes": array(obj({
                 "time_seconds": number(), "qpos": array(number(-1e6, 1e6), 1, 512),
             }, ("time_seconds", "qpos")), 2, 16)}, ("keyframes",), author, mutate=True)
    register("commit_keyframe", "Commit Keyframe on the candidate using FK of its sampled pose.",
             {"time_s": number(), "phase": text()}, ("time_s",),
             lambda a: w.commit_keyframe(**a), mutate=True)

    def joints(args):
        names = [value["joint"] for value in args["values"]]
        if len(names) != len(set(names)):
            raise ValueError("Joint Angles must have unique names")
        outputs = [w.operation("set_joint_target", {
            "time_seconds": args["time_s"], "joint": value["joint"],
            "angle_rad": value["angle_rad"],
        }) for value in args["values"]]
        return {"operations": outputs}

    register("set_joint_angles", "Set named Joint Angles in radians on the candidate.",
             {"time_s": number(), "values": array(obj({"joint": text(),
                "angle_rad": number(-100, 100)}, ("joint", "angle_rad")), 1, 128)},
             ("time_s", "values"), joints, mutate=True)

    def solve(args):
        return w.operation("set_logical_frame_target", {
            "logical_frame": args["logical_frame"], "start_time": args["time_s"],
            "end_time": args["time_s"], "mode": "absolute",
            "position_m": args["position_m"], "orientation_rpy_rad": args.get("orientation_rpy_rad"),
            "locked_end_effectors": args.get("locked_end_effectors", []),
        })

    register("solve_ik", "Solve a logical frame/End Effector pose in meters and radians.",
             {"time_s": number(), "logical_frame": text(),
              "position_m": array(number(-100, 100), 3, 3),
              "orientation_rpy_rad": array(number(-100, 100), 3, 3),
              "locked_end_effectors": array(text(), 0, 16)},
             ("time_s", "logical_frame", "position_m"), solve, mutate=True)
    register("repeat_motion", "Append exact copies of a candidate motion interval.",
             {"start_s": number(), "end_s": number(),
              "additional_copies": {"type": "integer", "minimum": 1, "maximum": 32},
              "ping_pong": {"type": "boolean"}}, ("start_s", "end_s", "additional_copies"),
             lambda a: w.operation("repeat_motion", {
                 "start_time": a["start_s"], "end_time": a["end_s"],
                 "additional_copies": a["additional_copies"], "ping_pong": a.get("ping_pong", False),
             }), mutate=True)

    def delete(args):
        candidate = detached_document(w.candidate(create=True).working_document)
        times = set(args["times"])
        frames = [frame for frame in candidate.trajectory.frames if frame.time not in times]
        candidate.trajectory = Trajectory()
        for frame in frames:
            candidate.trajectory.add_frame(frame)
        for time in times:
            candidate.qpos_timeline.delete_state(time)
        candidate.active_index = -1
        return w.apply_document(candidate, "delete_keyframes")

    register("delete_keyframes", "Delete logical and qpos Keyframes atomically on the candidate.",
             {"times": array(number(), 1, 128)}, ("times",), delete, mutate=True)

    def scale(args):
        candidate = detached_document(w.candidate(create=True).working_document)
        plan = plan_scale_time_range(candidate, args["start_s"], args["end_s"], 1.0 / args["scale"], maximum_time=120)
        ApplyTimelineEditPlan(plan).execute(candidate)
        return w.apply_document(candidate, "scale_time_range")

    register("edit_timeline", "Scale a candidate time interval through the shared planner.",
             {"start_s": number(), "end_s": number(), "scale": number(0.01, 100)},
             ("start_s", "end_s", "scale"), scale, mutate=True)

    def apply_spec(args):
        spec = parse_trajectory_edit_spec(args["spec_json"])
        return {"operations": [w.operation(op.operation_type, op.arguments) for op in spec.operations]}

    register("apply_edit_spec", "Apply a validated compact motion spec locally; no nested LLM call.",
             {"spec_json": text(24000)}, ("spec_json",), apply_spec, mutate=True)
    register("generate_trajectory", "Generate a detached candidate path using existing exact backend contracts.",
             {"dt": number(0.001, 1), "smoothing": number(0, 1)}, (), lambda a: w.generate(**a))
    catalog.meta["generate_trajectory"] = ToolMeta(app_level=True)
    register("validate_motion", "Check structure, limits and collisions over the sampled candidate path.",
             {"dt": number(0.001, 1)}, (), lambda a: w.validate(**a))

    def undo(args):
        session = w.candidate()
        if session is None:
            raise ValueError("no candidate to restore")
        checkpoint, generated = w.checkpoints.get(args.get("checkpoint_id"))
        session.restore_checkpoint(checkpoint)
        w.checkpoints.discard_from(args.get("checkpoint_id"))
        w.generated = generated
        if generated is not None:
            from dataclasses import replace
            w.generated = replace(generated, source=w.identity())
        w._validation_identity = None
        session.invalidate_validation()
        return {"restored": True}

    register("undo", "Restore an owned candidate checkpoint; committed history is untouched.",
             {"checkpoint_id": text()}, (), undo, app=True)
    register("accept_candidate", "Accept validated motion under the caller-configured review policy.",
             {}, (), lambda a: w.accept_candidate(), app=True)
    register("discard_candidate", "Discard the Orange preview without changing committed motion.",
             {}, (), lambda a: w.discard_candidate(), app=True)
    register("export_trajectory", "Export accepted motion through unchanged legacy format writers.",
             {"format": choice("mujoco_csv", "qpos_csv", "mjlab_csv", "dsms"),
              "file_name": text(), "dt": number(0.001, 1),
              "motion_speed": number(0.1, 4),
              "source": choice("auto", "generated", "editable")},
             ("format", "file_name"), lambda a: w.export_trajectory(**a), app=True)
    return catalog

"""Plain-Python helper for generating sampled robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from application.paths import (
    mujoco_playback_cache_path,
    prepare_csv_save_path,
)
from core.ik import (
    adaptive_trajectory_collision_reports,
    format_collision_diagnostics,
    pose_target_errors,
    project_qpos_above_flat_ground,
)
from core.trajectory import SampledTrajectory


@dataclass(frozen=True)
class TrajectoryGenerationResult:
    csv_path: str
    result_states: list
    status_text: str
    safety_warning_report: object | None = None
    ground_correction_count: int = 0


class TrajectorySafetyError(ValueError):
    """Raised when generated motion cannot be safely published or exported."""

    def __init__(
        self,
        message,
        *,
        report=None,
        candidate_states=(),
        candidate_qposes=(),
        candidate_times=(),
    ):
        super().__init__(message)
        self.report = report
        self.candidate_states = tuple(candidate_states)
        self.candidate_qposes = tuple(
            np.asarray(qpos, dtype=float).copy() for qpos in candidate_qposes
        )
        self.candidate_times = tuple(float(time) for time in candidate_times)


def _discard_backend_solution(backend_interface):
    """Best-effort removal of a solved path that failed safety validation."""
    clear = getattr(backend_interface, "clear_last_solution", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            # Cleanup must not replace the actionable validation error.
            pass
        return

    solution = getattr(backend_interface, "last_solution", None)
    if hasattr(solution, "clear"):
        solution.clear()


def _safety_robot_model(backend_interface, state_timeline):
    robot_model = getattr(state_timeline, "robot_model", None)
    if robot_model is None:
        robot_model = getattr(backend_interface, "adapter", None)
    if robot_model is None:
        raise TrajectorySafetyError(
            "Generated motion cannot be safety-validated because no active "
            "robot model is available. The CSV was not published."
        )
    return robot_model


def _configuration_qpos(configuration, robot_model, sample_index):
    """Convert exact or compatibility backend output to canonical model qpos."""
    expected_width = int(robot_model.mj_model.nq)
    exact_qpos = getattr(configuration, "qpos", None)
    if exact_qpos is not None:
        qpos = np.asarray(exact_qpos, dtype=float).copy()
    else:
        qpos = np.asarray(robot_model.home_qpos, dtype=float).copy()
        free_joints = tuple(
            getattr(robot_model, "free_joints_by_body", {}).values()
        )
        if free_joints:
            if len(free_joints) != 1:
                raise TrajectorySafetyError(
                    "Generated motion cannot be safety-validated for a model "
                    "with more than one floating root. The CSV was not published."
                )
            address = int(free_joints[0].qpos_address)
            try:
                qpos[address:address + 7] = (
                    configuration.base_x,
                    configuration.base_y,
                    configuration.base_z,
                    configuration.base_qw,
                    configuration.base_qx,
                    configuration.base_qy,
                    configuration.base_qz,
                )
            except AttributeError as exc:
                raise TrajectorySafetyError(
                    "Generated backend state "
                    f"{sample_index} has neither canonical qpos nor a complete "
                    "floating-root pose. The CSV was not published."
                ) from exc

        joint_names = getattr(configuration, "joint_names", ()) or ()
        joint_positions = getattr(configuration, "joint_positions", ()) or ()
        for name, value in zip(joint_names, joint_positions):
            joint = getattr(robot_model, "joints", {}).get(name)
            if joint is not None:
                qpos[int(joint.qpos_address)] = float(value)

    if qpos.shape != (expected_width,):
        raise TrajectorySafetyError(
            f"Generated backend state {sample_index} has qpos shape "
            f"{qpos.shape}; expected ({expected_width},). The CSV was not "
            "published."
        )
    if not np.all(np.isfinite(qpos)):
        raise TrajectorySafetyError(
            f"Generated backend state {sample_index} contains non-finite "
            "Joint Angles. The CSV was not published."
        )
    for joint in getattr(robot_model, "joints", {}).values():
        if joint.limits is None:
            continue
        value = float(qpos[int(joint.qpos_address)])
        lo, hi = map(float, joint.limits)
        if value < lo - 1e-9 or value > hi + 1e-9:
            raise TrajectorySafetyError(
                f"Generated backend state {sample_index} has Joint Angle "
                f"{joint.name}={value:.6g} outside [{lo:.6g}, {hi:.6g}]. "
                "The CSV was not published."
            )
    return qpos


def _generated_motion_arrays(result_states, robot_model):
    qposes = [
        _configuration_qpos(configuration, robot_model, sample_index)
        for sample_index, configuration in enumerate(result_states)
    ]
    times = [
        float(getattr(configuration, "time", sample_index))
        for sample_index, configuration in enumerate(result_states)
    ]
    if not np.all(np.isfinite(times)):
        raise TrajectorySafetyError(
            "Generated motion contains a non-finite timestamp. The CSV was "
            "not published."
        )
    return qposes, times


def _synchronize_configuration_qpos(configuration, robot_model, qpos):
    """Write a projected canonical qpos back to every backend representation."""
    qpos = np.asarray(qpos, dtype=float).copy()
    if getattr(configuration, "qpos", None) is not None:
        configuration.qpos = qpos

    free_joints = tuple(
        getattr(robot_model, "free_joints_by_body", {}).values()
    )
    if len(free_joints) == 1:
        address = int(free_joints[0].qpos_address)
        root = qpos[address:address + 7]
        for attribute, value in zip(
            (
                "base_x", "base_y", "base_z",
                "base_qw", "base_qx", "base_qy", "base_qz",
            ),
            root,
        ):
            if hasattr(configuration, attribute):
                setattr(configuration, attribute, float(value))

    names = getattr(configuration, "joint_names", ()) or ()
    positions = getattr(configuration, "joint_positions", None)
    if positions is not None:
        corrected = list(positions)
        for index, name in enumerate(names):
            joint = getattr(robot_model, "joints", {}).get(name)
            if joint is not None and index < len(corrected):
                corrected[index] = float(qpos[int(joint.qpos_address)])
        configuration.joint_positions = corrected


def _format_safety_report(report):
    time = getattr(report, "time", None)
    if time is not None:
        location = f"at {float(time):.3f} s"
    else:
        location = f"at generated sample {int(report.sample_index)}"

    fraction = getattr(report, "segment_fraction", None)
    segment_index = getattr(report, "segment_index", None)
    if fraction is not None and segment_index is not None and 0.0 < fraction < 1.0:
        location += (
            f" (between samples {int(segment_index)} and "
            f"{int(segment_index) + 1})"
        )

    diagnostics = format_collision_diagnostics(report.collisions, limit=3)
    return f"{location}: {diagnostics}"


def _validate_generated_motion(
    result_states,
    backend_interface,
    state_timeline,
    sampled_tracks,
):
    robot_model = _safety_robot_model(backend_interface, state_timeline)
    qposes, times = _generated_motion_arrays(result_states, robot_model)
    correction_count = 0
    max_correction = 0.0
    if callable(getattr(robot_model, "create_state", None)):
        projected_qposes = []
        for index, (configuration, qpos) in enumerate(
            zip(result_states, qposes)
        ):
            projection = project_qpos_above_flat_ground(robot_model, qpos)
            if not projection.success:
                raise TrajectorySafetyError(
                    "Generated motion was blocked before CSV publication at "
                    f"sample {index}: {projection.reason}.",
                    candidate_states=tuple(result_states),
                    candidate_qposes=tuple(qposes),
                    candidate_times=tuple(times),
                )
            projected_qposes.append(projection.qpos)
            if projection.changed:
                sample = (
                    sampled_tracks[index]
                    if index < len(sampled_tracks) else {}
                )
                if sample.get("qpos_anchor") is not None:
                    raise TrajectorySafetyError(
                        "Generated motion requires a ground correction at an "
                        f"exact Keyframe anchor ({times[index]:.3f} s). "
                        "Correct and recommit that Keyframe instead; the CSV "
                        "was not published."
                    )
                targets = sample.get("targets", {})
                frame_bindings = getattr(
                    robot_model, "logical_frame_bindings", {}
                )
                if targets and frame_bindings:
                    projected_state = robot_model.create_state()
                    projected_state.set_qpos(projection.qpos)
                    position_error, orientation_error, _solved, _ignored = (
                        pose_target_errors(
                            projected_state,
                            targets,
                            frame_bindings,
                            include_orientation=True,
                        )
                    )
                    if position_error > 0.01 or orientation_error > 0.06:
                        raise TrajectorySafetyError(
                            "Generated ground correction would violate required "
                            f"End Effector targets at {times[index]:.3f} s "
                            f"(position error {position_error:.4f} m, "
                            f"orientation error {orientation_error:.4f} rad). "
                            "Correct the surrounding Keyframes; the CSV was "
                            "not published."
                        )
                correction_count += 1
                max_correction = max(
                    max_correction, float(projection.applied_offset)
                )
                _synchronize_configuration_qpos(
                    configuration, robot_model, projection.qpos
                )
        qposes = projected_qposes
    warning_report, blocking_report = adaptive_trajectory_collision_reports(
        robot_model,
        qposes,
        times=times,
    )
    if blocking_report is not None:
        raise TrajectorySafetyError(
            "Generated motion was blocked before CSV publication due to a "
            f"penetrating collision {_format_safety_report(blocking_report)}. "
            "Adjust the surrounding Keyframes or try a safe reroute.",
            report=blocking_report,
            candidate_states=tuple(result_states),
            candidate_qposes=tuple(qposes),
            candidate_times=tuple(times),
        )
    return warning_report, correction_count, max_correction


def _attach_qpos_references(sampled_tracks, trajectory, state_timeline):
    """Attach model-width references only for complete committed timeslices."""
    if state_timeline is None or not sampled_tracks:
        return 0
    robot_model = getattr(state_timeline, "robot_model", None)
    expected_frames = set(getattr(robot_model, "trajectory_frames", ()))
    if not expected_frames:
        return 0

    frames_by_time = {}
    for frame in trajectory.frames:
        key = state_timeline.time_key(frame.time)
        frames_by_time.setdefault(key, set()).add(frame.frame_name)
    anchor_states = {
        time: state_timeline.get_state(time)
        for time, frame_names in frames_by_time.items()
        if expected_frames.issubset(frame_names)
        and state_timeline.get_state(time) is not None
    }
    if not anchor_states:
        return 0

    sample_times = [float(sample["time"]) for sample in sampled_tracks]
    for anchor_time in anchor_states:
        if not any(
            abs(sample_time - anchor_time) <= 1e-7
            for sample_time in sample_times
        ):
            raise ValueError(
                f"Committed Keyframe time {anchor_time:.6g}s is not aligned "
                "with the selected export interval. Choose an interval that "
                "lands exactly on every committed Keyframe."
            )

    for sample in sampled_tracks:
        time = float(sample["time"])
        sample["qpos_reference"] = state_timeline.sample_state(time)
        matching_anchor = next(
            (
                qpos for anchor_time, qpos in anchor_states.items()
                if abs(anchor_time - time) <= 1e-7
            ),
            None,
        )
        if matching_anchor is not None:
            sample["qpos_anchor"] = matching_anchor
    return len(anchor_states)


def generate_trajectory_status(
    trajectory,
    backend_interface,
    *,
    smoothing,
    export_dt=0.01,
    csv_path=None,
    state_timeline=None,
):
    if csv_path is None:
        csv_path = mujoco_playback_cache_path()
    sampled_tracks = trajectory.sample_tracks_uniform_dt(
        dt=export_dt,
        smoothing=smoothing,
    )
    anchor_count = _attach_qpos_references(
        sampled_tracks, trajectory, state_timeline
    )
    sampled_trajectory = SampledTrajectory(samples=sampled_tracks)

    result_states = backend_interface.solve_trajectory(sampled_trajectory)
    try:
        (
            safety_warning_report,
            ground_correction_count,
            max_ground_correction,
        ) = _validate_generated_motion(
            result_states,
            backend_interface,
            state_timeline,
            sampled_tracks,
        )
    except Exception:
        # A solved but unvalidated path must not remain available to playback or
        # a later cache regeneration attempt.
        _discard_backend_solution(backend_interface)
        raise

    # Path normalization may create directories, and export publishes the
    # backend solution. Both deliberately happen only after the safety gate.
    csv_path = prepare_csv_save_path(csv_path)
    backend_interface.export_last_solution_csv(csv_path)

    lines = []
    lines.append("Generated uniformly sampled per-frame target tracks.")
    lines.append(f"Backend: {backend_interface.last_backend_name()}")
    lines.append(f"Export dt: {export_dt:.4f} s")
    lines.append(f"Corner smoothing: {smoothing * 100.0:.0f}%")
    lines.append(f"Number of GUI keyframes: {len(trajectory.frames)}")
    lines.append(f"Number of sampled time steps: {len(sampled_tracks)}")
    lines.append(f"Number of backend states: {len(result_states)}")
    if ground_correction_count:
        lines.append(
            f"Ground barrier: raised {ground_correction_count} generated "
            f"samples (maximum {max_ground_correction * 1000.0:.1f} mm) "
            "before adaptive validation."
        )
    if safety_warning_report is not None:
        lines.append(
            "Motion safety warning: advisory contact "
            f"{_format_safety_report(safety_warning_report)}."
        )
    else:
        lines.append(
            "Motion safety: all generated samples and adaptive between-sample "
            "checks passed."
        )
    if anchor_count:
        lines.append(
            f"Exact committed qpos anchors: {anchor_count}; posture preserved "
            "as a secondary null-space objective between Keyframes."
        )
    else:
        lines.append(
            "Exact committed qpos anchors: 0; used target-only compatibility mode."
        )
    if result_states:
        max_ik_error = max(state.ik_error for state in result_states)
        lines.append(f"Max IK position error: {max_ik_error:.4f} m")
        max_orientation_error = max(
            state.orientation_error for state in result_states
        )
        lines.append(
            "Max IK orientation error: "
            f"{max_orientation_error:.4f} rad"
        )
    lines.append(f"Exported CSV to: {csv_path}")
    lines.append("")
    lines.append("First few sampled time groups:")
    lines.append("")

    for sample in sampled_tracks[:10]:
        frame_names = ", ".join(sorted(sample["targets"].keys()))
        lines.append(f"t={sample['time']:.3f}s | targets={frame_names}")

    return TrajectoryGenerationResult(
        csv_path=str(csv_path),
        result_states=result_states,
        status_text="\n".join(lines),
        safety_warning_report=safety_warning_report,
        ground_correction_count=ground_correction_count,
    )

"""Plain-Python helper for generating sampled robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from application.paths import (
    mujoco_playback_cache_path,
    prepare_csv_save_path,
)
from core.trajectory import SampledTrajectory


@dataclass(frozen=True)
class TrajectoryGenerationResult:
    csv_path: str
    result_states: list
    status_text: str


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
    csv_path = prepare_csv_save_path(csv_path)
    sampled_tracks = trajectory.sample_tracks_uniform_dt(
        dt=export_dt,
        smoothing=smoothing,
    )
    anchor_count = _attach_qpos_references(
        sampled_tracks, trajectory, state_timeline
    )
    sampled_trajectory = SampledTrajectory(samples=sampled_tracks)

    result_states = backend_interface.solve_trajectory(sampled_trajectory)
    backend_interface.export_last_solution_csv(csv_path)

    lines = []
    lines.append("Generated uniformly sampled per-frame target tracks.")
    lines.append(f"Backend: {backend_interface.last_backend_name()}")
    lines.append(f"Export dt: {export_dt:.4f} s")
    lines.append(f"Corner smoothing: {smoothing * 100.0:.0f}%")
    lines.append(f"Number of GUI keyframes: {len(trajectory.frames)}")
    lines.append(f"Number of sampled time steps: {len(sampled_tracks)}")
    lines.append(f"Number of backend states: {len(result_states)}")
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
    )

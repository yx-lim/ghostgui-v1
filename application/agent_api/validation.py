"""Bounded structural and sampled kinematic checks for agent acceptance."""

import math

import numpy as np

from application.csv_io import TrajectoryExport
from application.trajectory_export_formats import resample_trajectory_export
from core.ik import trajectory_collision_reports


def validate_candidate(adapter, motion_service, document, *, generated=None, dt=0.02,
                       max_samples=10001, cancellation_token=None):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("validation interval must be positive and finite")
    structural = motion_service.validate_motion(document)
    issues, warnings = list(structural.issues), list(structural.warnings)
    timeline = document.qpos_timeline
    if not timeline.times():
        issues.append("motion has no committed qpos anchors")
    times = tuple(timeline.times()) if generated is None else generated.export.times
    qposes = tuple(timeline.qpos_trajectory()) if generated is None else generated.export.qposes
    if not times or issues:
        return dict(valid=False, issues=issues, warnings=warnings, samples=0, sample_dt=dt)
    if (times[-1] - times[0]) / dt + len(times) + 1 > max_samples:
        raise ValueError("sample budget exceeded; choose a larger validation interval")
    export = TrajectoryExport(adapter.mj_model.nq, times, qposes, "candidate", False)
    sampled = resample_trajectory_export(export, adapter, dt)
    # Include exact anchors/endpoints even if they are off the regular grid.
    samples = dict(zip(sampled.times, sampled.qposes))
    samples.update(zip(times, qposes))
    ordered = sorted(samples)
    for time in ordered:
        if cancellation_token is not None and cancellation_token.cancellation_requested:
            raise RuntimeError("operation cancelled")
        qpos = samples[time]
        for name, joint in adapter.joints.items():
            limits = adapter.get_joint_limits(name)
            value = float(qpos[joint.qpos_address])
            if limits is not None and not limits[0] - 1e-8 <= value <= limits[1] + 1e-8:
                issues.append(f"Joint Angle {name} exceeds limits at {time:.6g} s")
        for joint in adapter.free_joints_by_body.values():
            quaternion = qpos[joint.qpos_address + 3:joint.qpos_address + 7]
            if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
                issues.append(f"root quaternion is not normalized at {time:.6g} s")
    # Check the actual sampled path, not just logical target Keyframes.
    warning, blocking = trajectory_collision_reports(adapter, [samples[t] for t in ordered])
    if warning is not None:
        warnings.append(f"collision warning at {ordered[warning.sample_index]:.6g} s")
    if blocking is not None:
        issues.append(f"blocking collision at {ordered[blocking.sample_index]:.6g} s")
    return dict(
        valid=not issues, issues=issues, warnings=warnings,
        samples=len(ordered), sample_dt=dt,
        source="generated" if generated is not None else "editable",
        unavailable_metrics=["joint_velocity_limits", "dynamics", "continuous_collision_freedom"],
    )

"""Shared export-source selection, content freshness, and collision checks."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from application.csv_io import TrajectoryExport
from application.trajectory_export_formats import resample_trajectory_export
from core.ik import trajectory_collision_reports, format_collision_pairs, format_collision_diagnostics


@dataclass(frozen=True)
class MotionSourceIdentity:
    document_id: str
    revision: int
    model_identity: int
    content_digest: str
    settings: tuple


def motion_source_identity(adapter, timeline, *, document=None, settings=()):
    """Include legacy direct edits as well as typed document revisions."""
    digest = hashlib.sha256()
    if document is not None:
        digest.update(json.dumps(document.trajectory.to_project_dict(), sort_keys=True,
                                 allow_nan=False).encode())
        digest.update(str(document.timeline_duration).encode())
    for time in timeline.times():
        digest.update(np.asarray([time], dtype="<f8").tobytes())
        digest.update(np.asarray(timeline.get_state(time), dtype="<f8").tobytes())
    return MotionSourceIdentity(
        "" if document is None else document.document_id,
        0 if document is None else document.revision,
        id(adapter.mj_model), digest.hexdigest(), tuple(settings),
    )


@dataclass(frozen=True)
class GeneratedMotion:
    export: TrajectoryExport
    source: MotionSourceIdentity


@dataclass(frozen=True)
class ExportSnapshot:
    export: TrajectoryExport
    warning_report: object = None
    blocking_report: object = None


def build_trajectory_export(adapter, timeline, generated=None, *, source="auto",
                            current_identity=None, sample_dt=None, preview_active=False,
                            checker=None, collision_reports=None):
    """Select a source explicitly; sampling never inserts qpos Keyframes."""
    if timeline is None:
        raise ValueError("no robot timeline is available")
    if source not in ("auto", "generated", "editable"):
        raise ValueError("export source must be auto, generated, or editable")
    use_generated = source == "generated" or (source == "auto" and generated is not None)
    if use_generated:
        if generated is None:
            raise ValueError("No generated motion; Generate first or choose editable motion")
        if current_identity is None or generated.source != current_identity:
            raise ValueError("Generated motion is stale; Generate again or explicitly choose editable motion")
        values = generated.export
        export = TrajectoryExport(
            adapter.mj_model.nq, tuple(values.times),
            tuple(np.asarray(qpos).copy() for qpos in values.qposes),
            "generated trajectory", bool(preview_active),
        )
    else:
        export = TrajectoryExport(
            adapter.mj_model.nq, tuple(timeline.times()),
            tuple(timeline.qpos_trajectory()), "editable timeline", bool(preview_active),
        )
    if not export.times:
        raise ValueError("trajectory contains no committed samples")
    if sample_dt is not None:
        export = resample_trajectory_export(export, adapter, sample_dt)
    if use_generated and sample_dt is None and collision_reports is not None:
        warning, blocking = collision_reports
    else:
        warning, blocking = trajectory_collision_reports(adapter, export.qposes, checker=checker)
    if blocking is not None:
        raise ValueError(
            f"{export.source_name} has a blocking collision at sample {blocking.sample_index}: "
            f"{format_collision_pairs(blocking.collisions)}; contact geometry: "
            f"{format_collision_diagnostics(blocking.collisions)}"
        )
    return ExportSnapshot(export, warning, blocking)

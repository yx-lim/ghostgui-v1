"""Bounded frame selection and provider-neutral motion image capture."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from application.ai.schemas import ImageVariant, MotionFrameImage
from application.project_document import ProjectDocument


class FrameCaptureError(RuntimeError):
    """Raised when representative motion frames cannot be sampled or rendered."""


@dataclass(frozen=True)
class FrameSamplingRequest:
    """Hints for a bounded 4--8 frame observation window."""

    selected_interval: tuple[float, float] | None = None
    suspected_times: tuple[float, ...] = ()
    minimum_frames: int = 4
    maximum_frames: int = 8

    def __post_init__(self) -> None:
        if not 4 <= self.minimum_frames <= self.maximum_frames <= 8:
            raise ValueError("frame sampling must stay within the 4--8 frame bound")
        if self.selected_interval is not None:
            start, end = map(float, self.selected_interval)
            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError("selected interval must be finite")
            if start < 0.0 or end <= start:
                raise ValueError("selected interval must have non-negative increasing times")
            object.__setattr__(self, "selected_interval", (start, end))
        suspected = tuple(float(value) for value in self.suspected_times)
        if any(not math.isfinite(value) or value < 0.0 for value in suspected):
            raise ValueError("suspected motion times must be finite and non-negative")
        object.__setattr__(self, "suspected_times", suspected)


@dataclass(frozen=True)
class FrameSamplingPlan:
    """Exact timestamps shared by original and candidate frame capture."""

    times_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 4 <= len(self.times_seconds) <= 8:
            raise ValueError("a frame sampling plan must contain 4--8 timestamps")
        times = tuple(float(value) for value in self.times_seconds)
        if any(not math.isfinite(value) or value < 0.0 for value in times):
            raise ValueError("frame sampling timestamps must be finite and non-negative")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("frame sampling timestamps must be unique and increasing")
        object.__setattr__(self, "times_seconds", times)


@dataclass(frozen=True)
class EncodedFrame:
    data: bytes
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("rendered frame data must not be empty")
        if self.mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported rendered frame MIME type")


class MotionFrameRenderer(Protocol):
    """GUI-owned renderer boundary used by the Qt-free application workflow."""

    def render_frame(
        self,
        qpos: Any,
        *,
        time_seconds: float,
        variant: ImageVariant,
    ) -> EncodedFrame:
        ...


class FrameSampler:
    """Select representative Keyframe, interval, and suspected-area times."""

    def plan(
        self,
        document: ProjectDocument,
        request: FrameSamplingRequest | None = None,
    ) -> FrameSamplingPlan:
        request = request or FrameSamplingRequest()
        duration = float(document.timeline_duration)
        if request.selected_interval is None:
            start, end = 0.0, duration
        else:
            start, end = request.selected_interval
            if end > duration + 1e-9:
                raise FrameCaptureError("selected interval exceeds the motion duration")

        uniform = _uniform_times(start, end, request.minimum_frames)
        keyframe_times = _document_keyframe_times(document, start, end)
        suspected = tuple(
            value for value in request.suspected_times if start <= value <= end
        )
        candidates = _unique_times((start, end, *suspected, *keyframe_times, *uniform))
        selected = _bounded_representative_times(
            candidates,
            protected=_unique_times((start, end, *suspected)),
            limit=request.maximum_frames,
        )
        return FrameSamplingPlan(tuple(sorted(selected)))


def capture_comparison_frames(
    original: ProjectDocument,
    candidate: ProjectDocument,
    plan: FrameSamplingPlan,
    renderer: MotionFrameRenderer,
) -> tuple[MotionFrameImage, ...]:
    """Render paired original/candidate evidence at exactly the same times."""

    if original.model_key != candidate.model_key:
        raise FrameCaptureError("comparison documents must use the same robot model")
    frames: list[MotionFrameImage] = []
    for index, time_seconds in enumerate(plan.times_seconds, start=1):
        comparison_id = f"frame_{index}"
        label = comparison_id
        for document, variant in (
            (original, ImageVariant.ORIGINAL),
            (candidate, ImageVariant.CANDIDATE),
        ):
            qpos = _sample_qpos(document, time_seconds)
            rendered = renderer.render_frame(
                qpos,
                time_seconds=time_seconds,
                variant=variant,
            )
            frames.append(
                MotionFrameImage(
                    data=rendered.data,
                    mime_type=rendered.mime_type,
                    time_seconds=time_seconds,
                    variant=variant,
                    comparison_id=comparison_id,
                    label=label,
                )
            )
    return tuple(frames)


def _sample_qpos(document: ProjectDocument, time_seconds: float) -> Any:
    timeline = document.qpos_timeline
    if timeline is None or not hasattr(timeline, "sample_state"):
        raise FrameCaptureError("motion document has no sampleable qpos timeline")
    value = timeline.sample_state(float(time_seconds))
    if value is None:
        raise FrameCaptureError(f"motion could not be sampled at t={time_seconds:.3f} s")
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else value


def _document_keyframe_times(
    document: ProjectDocument,
    start: float,
    end: float,
) -> tuple[float, ...]:
    values = [
        float(frame.time)
        for frame in document.trajectory.frames
        if start <= float(frame.time) <= end
    ]
    timeline = document.qpos_timeline
    if timeline is not None and hasattr(timeline, "times"):
        values.extend(
            float(value)
            for value in timeline.times()
            if start <= float(value) <= end
        )
    return _unique_times(values)


def _uniform_times(start: float, end: float, count: int) -> tuple[float, ...]:
    step = (end - start) / (count - 1)
    return tuple(start + index * step for index in range(count))


def _unique_times(values) -> tuple[float, ...]:
    by_key: dict[float, float] = {}
    for value in values:
        number = float(value)
        by_key.setdefault(round(number, 9), number)
    return tuple(sorted(by_key.values()))


def _bounded_representative_times(
    candidates: tuple[float, ...],
    *,
    protected: tuple[float, ...],
    limit: int,
) -> tuple[float, ...]:
    """Thin deterministically while retaining hints and temporal coverage."""

    if len(candidates) <= limit:
        return candidates
    if len(protected) > limit:
        selected = [protected[0], protected[-1]]
        protected_remaining = list(protected[1:-1])
        while protected_remaining and len(selected) < limit:
            value = max(
                protected_remaining,
                key=lambda item: (min(abs(item - kept) for kept in selected), -item),
            )
            selected.append(value)
            protected_remaining.remove(value)
    else:
        selected = list(protected)
    remaining = [value for value in candidates if value not in selected]
    while remaining and len(selected) < limit:
        value = max(
            remaining,
            key=lambda item: (min(abs(item - kept) for kept in selected), -item),
        )
        selected.append(value)
        remaining.remove(value)
    return tuple(sorted(selected))

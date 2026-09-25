"""Read-only adaptation from an AI working copy to Orange preview pose data."""

from __future__ import annotations

from copy import copy
from typing import Any

from application.ai.edit_session import (
    AIEditSession,
    AIEditSessionError,
    AIEditSessionState,
)


def sample_working_preview_qpos(
    session: AIEditSession,
    time_seconds: float,
) -> Any:
    """Sample one detached working-copy pose for presentation only.

    This never mutates or replaces the committed document. The presentation
    layer may call it for each scrub or playback time, but each call returns
    one pose rather than exposing or generating a raw qpos trajectory.
    """

    if session.state is not AIEditSessionState.STAGED:
        raise AIEditSessionError("Orange preview requires a staged AI session")
    return sample_document_preview_qpos(
        session.working_document,
        time_seconds,
        missing_message="AI working copy has no sampleable qpos timeline",
    )


def sample_document_preview_qpos(
    document,
    time_seconds: float,
    *,
    missing_message: str = "Motion has no sampleable qpos timeline",
) -> Any:
    """Sample a detached document used as an immutable preview reference."""

    timeline = document.qpos_timeline
    if timeline is None or not hasattr(timeline, "sample_state"):
        raise AIEditSessionError(missing_message)
    value = timeline.sample_state(float(time_seconds))
    if value is None:
        raise AIEditSessionError("Motion could not sample an Orange preview")
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else copy(value)

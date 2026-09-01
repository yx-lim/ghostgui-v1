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

    This never mutates or replaces the committed document. It intentionally
    returns one pose, not a generated raw qpos trajectory.
    """

    if session.state is not AIEditSessionState.STAGED:
        raise AIEditSessionError("Orange preview requires a staged AI session")
    timeline = session.working_document.qpos_timeline
    if timeline is None or not hasattr(timeline, "sample_state"):
        raise AIEditSessionError("AI working copy has no sampleable qpos timeline")
    value = timeline.sample_state(float(time_seconds))
    if value is None:
        raise AIEditSessionError("AI working copy could not sample an Orange preview")
    copier = getattr(value, "copy", None)
    return copier() if callable(copier) else copy(value)

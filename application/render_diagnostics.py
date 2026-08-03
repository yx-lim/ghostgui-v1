"""Opt-in native-memory diagnostics for the OpenGL viewer.

This module deliberately has no Qt imports so the launcher and viewer can
record the same machine-readable events without changing application-layer
ownership.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


RENDER_DIAGNOSTICS_ENV = "GHOSTGUI_RENDER_DIAGNOSTICS"
DIAGNOSTIC_PREFIX = "GHOSTGUI_RENDER_DIAGNOSTIC "
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def render_diagnostics_enabled() -> bool:
    """Return whether the current process should emit render diagnostics."""
    return os.environ.get(RENDER_DIAGNOSTICS_ENV, "").strip().lower() in (
        _TRUE_VALUES
    )


def current_rss_bytes() -> int | None:
    """Return resident memory, including native Qt/OpenGL allocations."""
    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(
                encoding="ascii"
            ).split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            return None

    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            return int(result.stdout.strip()) * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    return None


def maximum_rss_bytes() -> int | None:
    """Return the process high-water RSS using the platform's native units."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the other supported POSIX platforms
    # report KiB through getrusage(2).
    return value if sys.platform == "darwin" else value * 1024


def memory_snapshot() -> dict[str, int | None]:
    return {
        "rss_bytes": current_rss_bytes(),
        "maximum_rss_bytes": maximum_rss_bytes(),
    }


def emit_render_diagnostic(event: str, **fields):
    """Write one JSON diagnostic event to stderr and return its payload."""
    if not render_diagnostics_enabled():
        return None
    payload = {
        "event": str(event),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "timestamp": time.time(),
        "monotonic_seconds": time.monotonic(),
        **memory_snapshot(),
        **fields,
    }
    try:
        print(
            DIAGNOSTIC_PREFIX + json.dumps(payload, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
    except (OSError, TypeError, ValueError):
        # Diagnostics must never make the editor fail to launch or render.
        return None
    return payload

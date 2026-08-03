#!/usr/bin/env python3
"""Compare GhostGUI's compatibility and Qt-default OpenGL memory behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PREFIX = "GHOSTGUI_RENDER_DIAGNOSTIC "
MODES = ("compatibility", "default")


def diagnostic_command(model: str, mode: str, seconds: float) -> list[str]:
    return [
        sys.executable,
        "-m",
        "application.launcher",
        "--model",
        model,
        "--opengl-mode",
        mode,
        "--render-diagnostics",
        "--diagnostic-seconds",
        str(float(seconds)),
    ]


def read_diagnostic_events(path: Path) -> list[dict]:
    events = []
    try:
        lines = Path(path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return events
    for line in lines:
        marker = line.find(DIAGNOSTIC_PREFIX)
        if marker < 0:
            continue
        try:
            event = json.loads(line[marker + len(DIAGNOSTIC_PREFIX):])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_events(events: list[dict]) -> dict:
    rss_values = [
        int(event["rss_bytes"])
        for event in events
        if event.get("rss_bytes") is not None
    ]
    maximum_values = [
        int(event["maximum_rss_bytes"])
        for event in events
        if event.get("maximum_rss_bytes") is not None
    ]
    contexts = [
        event for event in events
        if event.get("event") == "opengl_context_created"
    ]
    starts = [
        event for event in events
        if event.get("event") == "geometry_compile_started"
    ]
    finishes = [
        event for event in events
        if event.get("event") == "geometry_compile_finished"
    ]
    context = contexts[-1] if contexts else {}
    geometry = finishes[-1] if finishes else {}
    start = starts[-1] if starts else {}
    return {
        "event_count": len(events),
        "peak_observed_rss_bytes": max(rss_values, default=None),
        "maximum_rss_bytes": max(maximum_values, default=None),
        "context_count": max(
            (int(item.get("context_count", 0)) for item in events),
            default=0,
        ),
        "gl_renderer": context.get("gl_renderer"),
        "gl_version": context.get("gl_version"),
        "opengl_profile": context.get("opengl_profile"),
        "device_pixel_ratio": context.get("device_pixel_ratio"),
        "physical_width": context.get("physical_width"),
        "physical_height": context.get("physical_height"),
        "mesh_face_count": start.get("mesh_face_count"),
        "geometry_finished": bool(finishes),
        "compile_rss_delta_bytes": geometry.get("compile_rss_delta_bytes"),
    }


def _mib(value) -> str:
    return "unavailable" if value is None else f"{int(value) / 1048576:.1f} MiB"


def _print_summary(mode: str, summary: dict, stderr_path: Path) -> None:
    framebuffer = (
        f"{summary['physical_width']}x{summary['physical_height']}"
        if summary.get("physical_width") and summary.get("physical_height")
        else "unavailable"
    )
    print(f"\n{mode}:")
    print(f"  peak observed RSS: {_mib(summary['peak_observed_rss_bytes'])}")
    print(f"  process high-water RSS: {_mib(summary['maximum_rss_bytes'])}")
    print(f"  display-list compile delta: {_mib(summary['compile_rss_delta_bytes'])}")
    print(f"  geometry compile finished: {summary['geometry_finished']}")
    print(f"  contexts created: {summary['context_count']}")
    print(
        "  renderer: "
        f"{summary.get('gl_renderer') or 'unavailable'} / "
        f"{summary.get('gl_version') or 'unavailable'}"
    )
    print(
        f"  profile: {summary.get('opengl_profile') or 'unavailable'}; "
        f"DPR: {summary.get('device_pixel_ratio') or 'unavailable'}; "
        f"framebuffer: {framebuffer}"
    )
    print(f"  detailed log: {stderr_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch two timed GhostGUI runs and compare native OpenGL memory."
        )
    )
    parser.add_argument("--model", default="g1")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-non-macos",
        action="store_true",
        help="run the harness on another platform for development",
    )
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be greater than zero")
    if platform.system() != "Darwin" and not args.allow_non_macos:
        parser.error(
            "this comparison targets macOS; use --allow-non-macos only for "
            "harness development"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="ghostgui-render-diagnostics-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostic output: {output_dir}")
    print(
        "Each GhostGUI window will close automatically after "
        f"{args.seconds:g} seconds."
    )

    results = {}
    exit_code = 0
    for mode in MODES:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = mode_dir / "stdout.log"
        stderr_path = mode_dir / "stderr.log"
        environment = os.environ.copy()
        environment.update(
            {
                "GHOSTGUI_OPENGL_MODE": mode,
                "GHOSTGUI_RENDER_DIAGNOSTICS": "1",
                "GHOSTGUI_CONFIG_DIR": str(mode_dir / "config"),
                "GHOSTGUI_USER_DATA_DIR": str(mode_dir / "user-data"),
                "GHOSTGUI_PROJECTS_DIR": str(mode_dir / "projects"),
                "GHOSTGUI_CACHE_DIR": str(mode_dir / "cache"),
                "QT_LOGGING_RULES": "qt.qpa.gl=true",
            }
        )
        print(f"\nLaunching {mode} mode...")
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed = subprocess.run(
                diagnostic_command(args.model, mode, args.seconds),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
                text=True,
            )
        events = read_diagnostic_events(stderr_path)
        summary = summarize_events(events)
        results[mode] = summary
        _print_summary(mode, summary, stderr_path)
        if completed.returncode:
            exit_code = completed.returncode
            print(f"  process exit code: {completed.returncode}")

    compatibility_peak = results["compatibility"][
        "peak_observed_rss_bytes"
    ]
    default_peak = results["default"]["peak_observed_rss_bytes"]
    if compatibility_peak is not None and default_peak is not None:
        difference = compatibility_peak - default_peak
        print(
            "\nCompatibility minus default peak RSS: "
            f"{difference / 1048576:+.1f} MiB"
        )
    if not all(result["geometry_finished"] for result in results.values()):
        print(
            "\nAt least one geometry build did not finish. Rerun with a larger "
            "--seconds value before drawing a conclusion."
        )
    print("\nAttach both stderr.log files when reporting the result.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

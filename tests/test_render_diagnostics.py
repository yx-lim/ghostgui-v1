"""Contracts for opt-in renderer memory diagnostics and A/B summaries."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from application import render_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = PROJECT_ROOT / "scripts" / "diagnose_macos_rendering.py"
    spec = importlib.util.spec_from_file_location("render_diagnostic_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class RenderDiagnosticEventTests(unittest.TestCase):
    def test_events_are_disabled_by_default(self):
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "stderr", stderr),
        ):
            result = render_diagnostics.emit_render_diagnostic("sample")

        self.assertIsNone(result)
        self.assertEqual(stderr.getvalue(), "")

    def test_event_is_machine_readable_and_contains_native_memory(self):
        stderr = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {render_diagnostics.RENDER_DIAGNOSTICS_ENV: "1"},
                clear=True,
            ),
            patch.object(sys, "stderr", stderr),
            patch.object(
                render_diagnostics,
                "memory_snapshot",
                return_value={
                    "rss_bytes": 123,
                    "maximum_rss_bytes": 456,
                },
            ),
        ):
            result = render_diagnostics.emit_render_diagnostic(
                "geometry_compile_finished", context_count=2
            )

        line = stderr.getvalue().strip()
        self.assertTrue(line.startswith(render_diagnostics.DIAGNOSTIC_PREFIX))
        payload = json.loads(
            line[len(render_diagnostics.DIAGNOSTIC_PREFIX):]
        )
        self.assertEqual(payload["event"], "geometry_compile_finished")
        self.assertEqual(payload["rss_bytes"], 123)
        self.assertEqual(payload["maximum_rss_bytes"], 456)
        self.assertEqual(payload["context_count"], 2)
        self.assertEqual(result, payload)


class RenderDiagnosticRunnerTests(unittest.TestCase):
    def test_command_selects_mode_and_timed_diagnostics(self):
        command = runner.diagnostic_command("z1", "default", 7.5)

        self.assertEqual(command[:3], [sys.executable, "-m", "application.launcher"])
        self.assertIn("z1", command)
        self.assertIn("default", command)
        self.assertIn("--render-diagnostics", command)
        self.assertEqual(command[-1], "7.5")

    def test_reader_ignores_qt_noise_and_summary_tracks_compile_delta(self):
        events = [
            {
                "event": "opengl_context_created",
                "rss_bytes": 100,
                "maximum_rss_bytes": 110,
                "context_count": 1,
                "gl_renderer": "Example GPU",
                "physical_width": 1600,
                "physical_height": 1000,
            },
            {
                "event": "geometry_compile_started",
                "rss_bytes": 120,
                "maximum_rss_bytes": 120,
                "context_count": 1,
                "mesh_face_count": 50,
            },
            {
                "event": "geometry_compile_finished",
                "rss_bytes": 300,
                "maximum_rss_bytes": 310,
                "context_count": 1,
                "compile_rss_delta_bytes": 180,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_path = Path(temporary_dir) / "stderr.log"
            lines = ["qt.qpa.gl: context details"] + [
                runner.DIAGNOSTIC_PREFIX + json.dumps(event)
                for event in events
            ]
            log_path.write_text("\n".join(lines), encoding="utf-8")

            parsed = runner.read_diagnostic_events(log_path)
            summary = runner.summarize_events(parsed)

        self.assertEqual(parsed, events)
        self.assertEqual(summary["peak_observed_rss_bytes"], 300)
        self.assertEqual(summary["maximum_rss_bytes"], 310)
        self.assertEqual(summary["compile_rss_delta_bytes"], 180)
        self.assertTrue(summary["geometry_finished"])
        self.assertEqual(summary["gl_renderer"], "Example GPU")

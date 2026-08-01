#!/usr/bin/env python3
"""Smoke-test an installed wheel without importing from the source checkout."""

from __future__ import annotations

import argparse
from importlib import metadata, util
import os
from pathlib import Path
import sys
import tempfile


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--gui", action="store_true")
    return parser


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    args = _parser().parse_args(argv)
    os.environ.pop("GHOSTGUI_RESOURCE_DIR", None)

    distribution = metadata.distribution("ghostgui")
    environment_prefix = Path(sys.prefix).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    _require(
        distribution_root.is_relative_to(environment_prefix),
        f"ghostgui distribution is outside the smoke environment: {distribution_root}",
    )
    entry_points = {
        (entry.group, entry.name): entry.value
        for entry in distribution.entry_points
        if entry.group in {"console_scripts", "gui_scripts"}
    }
    _require(
        entry_points.get(("console_scripts", "ghostgui"))
        == "application.launcher:main",
        "installed console entry point is missing or incorrect",
    )
    _require(
        entry_points.get(("gui_scripts", "ghostgui-gui"))
        == "application.launcher:main",
        "installed GUI entry point is missing or incorrect",
    )
    _require(
        util.find_spec("application.mujoco_viewer_process") is not None,
        "installed MuJoCo viewer process module is missing",
    )
    for package_name in ("application", "core", "gui"):
        spec = util.find_spec(package_name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        _require(
            origin is not None and origin.is_relative_to(environment_prefix),
            f"{package_name} imported outside the smoke environment: {origin}",
        )

    from core.models import MuJoCoRobotAdapter, ROBOT_MODELS
    from core.resources import (
        bundled_resource_root,
        installed_resource_root,
        is_source_checkout,
        resource_path,
    )

    root = bundled_resource_root().resolve()
    _require(not is_source_checkout(), "smoke test imported a source checkout")
    _require(
        root == installed_resource_root().resolve(),
        f"resources did not resolve through the install scheme: {root}",
    )
    required = (
        "docs/user_guide.md",
        "gui/assets/app/ghostlogo.svg",
        "gui/assets/theme/play-dark.svg",
        "gui/assets/theme/play-light.svg",
    )
    for relative in required:
        _require(resource_path(relative, required=True).is_file(), relative)
    for key, info in ROBOT_MODELS.items():
        _require(info.model_path.is_file(), f"missing {key} model source")
        for package_root in info.package_map.values():
            _require(
                Path(package_root).is_dir(),
                f"missing {key} package asset directory: {package_root}",
            )

    adapter = None
    if args.load_model or args.gui:
        adapter = MuJoCoRobotAdapter("g1")
        _require(adapter.mj_model.nq > 0, "installed G1 model did not compile")
        _require(adapter.trajectory_frames, "installed G1 has no logical frames")

    if args.gui:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        with tempfile.TemporaryDirectory(prefix="ghostgui-wheel-smoke-") as directory:
            runtime = Path(directory)
            os.environ["GHOSTGUI_CONFIG_DIR"] = str(runtime / "config")
            os.environ["GHOSTGUI_CACHE_DIR"] = str(runtime / "cache")
            os.environ.pop("GHOSTGUI_PROJECTS_DIR", None)
            os.environ["GHOSTGUI_USER_DATA_DIR"] = str(runtime / "data")
            from application.project_manager import ghostgui_projects_dir
            from PySide6.QtWidgets import QApplication
            from gui.main_window import RobotGuiMainWindow

            _require(
                ghostgui_projects_dir() == runtime / "data" / "projects",
                "installed projects do not default to writable user data",
            )
            app = QApplication.instance() or QApplication([])
            window = RobotGuiMainWindow("g1")
            _require(window.robot_model_3d is not None, "GUI model is unavailable")
            _require(
                window.visualization_manager.initialized,
                "visualization runtime was not initialized",
            )
            window.current_project = None
            _require(window.close(), "GUI did not accept clean shutdown")
            app.processEvents()

    print(
        f"Installed package smoke test passed: ghostgui "
        f"{distribution.version} ({root})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line launcher for GhostGUI."""

from __future__ import annotations

import argparse
import os
import sys

from application.render_diagnostics import (
    RENDER_DIAGNOSTICS_ENV,
    emit_render_diagnostic,
)
from core.models import ROBOT_MODELS
from core.resources import resource_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ghostgui",
        description="Launch the GhostGUI MuJoCo trajectory editor.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(ROBOT_MODELS),
        default="g1",
        help="robot model to load at startup",
    )
    configured_mode = os.environ.get(
        "GHOSTGUI_OPENGL_MODE", "compatibility"
    ).strip().lower()
    if configured_mode not in ("compatibility", "default"):
        configured_mode = "compatibility"
    parser.add_argument(
        "--opengl-mode",
        choices=("compatibility", "default"),
        default=configured_mode,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--render-diagnostics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--diagnostic-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=argparse.SUPPRESS,
    )
    args, qt_args = parser.parse_known_args()
    if args.diagnostic_seconds < 0.0:
        parser.error("--diagnostic-seconds must be zero or greater")

    os.environ["GHOSTGUI_OPENGL_MODE"] = args.opengl_mode
    if args.render_diagnostics or args.diagnostic_seconds > 0.0:
        os.environ[RENDER_DIAGNOSTICS_ENV] = "1"
    emit_render_diagnostic(
        "launcher_start",
        model=args.model,
        opengl_mode=args.opengl_mode,
    )

    from PySide6.QtCore import QCoreApplication, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from gui.viewers.opengl_compat import configure_default_surface_format

    QCoreApplication.setApplicationName("GhostGUI")
    QCoreApplication.setOrganizationName("GhostGUI")
    QCoreApplication.setOrganizationDomain("github.com/yx-lim")
    configure_default_surface_format()
    app = QApplication([sys.argv[0], *qt_args])
    emit_render_diagnostic("qapplication_created")

    from gui.main_window import RobotGuiMainWindow
    from gui.theme import apply_application_theme

    apply_application_theme(app)
    app.setApplicationDisplayName("GhostGUI")
    app.setDesktopFileName("ghostgui")
    app.setWindowIcon(
        QIcon(str(resource_path("gui/assets/app/ghostlogo.svg", required=True)))
    )

    window = RobotGuiMainWindow(model_key=args.model)
    emit_render_diagnostic("window_constructed")
    window.show()
    QTimer.singleShot(
        0,
        lambda: emit_render_diagnostic(
            "window_shown",
            logical_width=window.width(),
            logical_height=window.height(),
            device_pixel_ratio=float(window.devicePixelRatioF()),
        ),
    )
    if args.diagnostic_seconds > 0.0:
        QTimer.singleShot(
            max(1, int(round(args.diagnostic_seconds * 1000.0))),
            window.close,
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Command-line launcher for GhostGUI."""

from __future__ import annotations

import argparse
import sys

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
    args, qt_args = parser.parse_known_args()

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from gui.viewers.opengl_compat import configure_default_surface_format

    QCoreApplication.setApplicationName("GhostGUI")
    QCoreApplication.setOrganizationName("GhostGUI")
    QCoreApplication.setOrganizationDomain("github.com/yx-lim")
    configure_default_surface_format()
    app = QApplication([sys.argv[0], *qt_args])

    from gui.main_window import RobotGuiMainWindow
    from gui.theme import apply_application_theme

    apply_application_theme(app)
    app.setApplicationDisplayName("GhostGUI")
    app.setDesktopFileName("ghostgui")
    app.setWindowIcon(
        QIcon(str(resource_path("gui/assets/app/ghostlogo.svg", required=True)))
    )

    window = RobotGuiMainWindow(model_key=args.model)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

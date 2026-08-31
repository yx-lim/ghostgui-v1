"""Command-line launcher for GhostGUI."""

from __future__ import annotations

import argparse
import sys

from core.models import ROBOT_MODELS


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

    from PySide6.QtWidgets import QApplication

    from gui.viewers.opengl_compat import configure_default_surface_format

    # The canvas uses the fixed-function API. On macOS the format must be
    # requested before QApplication creates any shared OpenGL contexts.
    configure_default_surface_format()
    app = QApplication([sys.argv[0], *qt_args])

    from gui.main_window import RobotGuiMainWindow
    from gui.theme import apply_application_theme

    apply_application_theme(app)

    window = RobotGuiMainWindow(model_key=args.model)
    window.resize(1200, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

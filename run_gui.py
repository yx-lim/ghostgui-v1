"""
python3 run_gui.py
"""

import sys
import argparse
from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", choices=("g1", "go2"), default="g1")
    args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0], *qt_args])

    window = RobotGuiMainWindow(model_key=args.model)
    window.resize(1200, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

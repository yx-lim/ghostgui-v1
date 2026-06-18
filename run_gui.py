"""
python3 run_gui.py
"""

import sys
from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow


def main():
    app = QApplication(sys.argv)

    window = RobotGuiMainWindow()
    window.resize(1200, 700)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
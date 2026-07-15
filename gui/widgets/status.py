"""Status label widgets."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel


class StatusValueLabel(QLabel):
    text_changed = Signal(str)

    def setText(self, text):
        previous = self.text()
        super().setText(text)
        if text != previous:
            self.text_changed.emit(text)

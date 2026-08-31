"""Compact keyboard and mouse shortcut reference dialog."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
)


SHORTCUT_ROWS = (
    ("T", "Move/translate gizmo"),
    ("R", "Rotate gizmo"),
    ("E / Esc", "Cancel the active drag"),
    ("Shift + drag", "Fine movement"),
    ("Ctrl + drag", "Snap movement"),
    ("Ctrl+Z", "Undo"),
    ("Ctrl+Shift+Z", "Redo"),
)


class KeyboardShortcutsDialog(QDialog):
    """Show the essential editing shortcuts without the full help center."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setObjectName("keyboardShortcutsDialog")
        self.setModal(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(12)

        shortcuts = QGridLayout()
        shortcuts.setHorizontalSpacing(24)
        shortcuts.setVerticalSpacing(8)
        for row, (keys, description) in enumerate(SHORTCUT_ROWS):
            key_label = QLabel(keys)
            key_label.setObjectName(f"shortcutKey{row}")
            key_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            action_label = QLabel(description)
            action_label.setObjectName(f"shortcutAction{row}")
            shortcuts.addWidget(key_label, row, 0)
            shortcuts.addWidget(action_label, row, 1)
        shortcuts.setColumnStretch(1, 1)
        root.addLayout(shortcuts)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        root.addWidget(buttons)
        self.setFixedSize(self.sizeHint())

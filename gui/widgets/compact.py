"""Compact sizing helpers for dense sidebar controls."""

from PySide6.QtWidgets import QComboBox, QSizePolicy


def compact_combo(combo, minimum_chars=10, minimum_width=96):
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(minimum_chars)
    # QMacStyle may give an Ignored control no horizontal space when it sits
    # inside a narrow form and a late-populated QStackedWidget.  Preserve a
    # useful floor while still allowing the combo to grow with the sidebar.
    combo.setMinimumWidth(max(0, int(minimum_width)))
    combo.setSizePolicy(
        QSizePolicy.Policy.MinimumExpanding,
        QSizePolicy.Policy.Fixed,
    )


def compact_spinbox(spinbox, width=78):
    spinbox.setMinimumWidth(0)
    spinbox.setMaximumWidth(width)
    spinbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

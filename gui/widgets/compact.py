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
    # Let Qt's font- and style-aware size hint account for digits, suffixes,
    # and native stepper buttons.  A hard maximum clips values with larger
    # system fonts and on styles whose steppers are wider than Fusion's.
    spinbox.setMinimumWidth(max(0, int(width)))
    spinbox.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
    spinbox.setSizePolicy(
        QSizePolicy.Policy.MinimumExpanding,
        QSizePolicy.Policy.Fixed,
    )

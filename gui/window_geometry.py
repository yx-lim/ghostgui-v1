"""Small, testable helpers for fitting GUI windows to a desktop work area."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication


def bounded_window_size(desired, available, *, margin=24):
    """Return a positive desired size bounded by an available screen size."""
    desired = QSize(desired)
    available = QSize(available)
    margin = max(0, int(margin))
    maximum_width = max(1, available.width() - 2 * margin)
    maximum_height = max(1, available.height() - 2 * margin)
    return QSize(
        max(1, min(desired.width(), maximum_width)),
        max(1, min(desired.height(), maximum_height)),
    )


def resize_to_available_screen(widget, width, height, *, margin=24):
    """Apply a preferred size without exceeding the owning screen work area."""
    parent = widget.parentWidget()
    screen = parent.screen() if parent is not None else widget.screen()
    if screen is None:
        screen = QApplication.primaryScreen()
    desired = QSize(max(1, int(width)), max(1, int(height)))
    if screen is not None:
        desired = bounded_window_size(
            desired,
            screen.availableGeometry().size(),
            margin=margin,
        )
    widget.resize(desired)
    return desired

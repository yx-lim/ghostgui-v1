"""Bind registry commands to existing widgets and QActions.

Tag mode (M0) only labels controls with their command id; existing signal wiring
is untouched. Route mode (M3) lets bound QActions dispatch through the registry.
"""

from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractButton, QMenu, QTabBar

COMMAND_PROPERTY = "ghostguiCommand"


def collect_controls(root: QObject) -> dict[str, QObject]:
    """Named, user-triggerable controls: QActions (not menu titles) and buttons."""
    controls: dict[str, QObject] = {}
    for item in root.findChildren(QAction):
        if item.menu() is not None or item.isSeparator():
            continue
        name = item.objectName()
        if name and not name.startswith("qt_"):
            controls.setdefault(name, item)
    for item in root.findChildren(QAbstractButton):
        name = item.objectName()
        if name and not name.startswith("qt_") and not isinstance(item.parent(), (QMenu, QTabBar)):
            controls.setdefault(name, item)
    return controls


class CommandBinder:
    def __init__(self, registry, root: QObject):
        self.registry = registry
        self.root = root
        self.tagged: dict[str, str] = {}

    def tag(self) -> dict[str, str]:
        """Label every bound control with its command id; returns objectName → command id."""
        controls = collect_controls(self.root)
        for name, (spec, _args) in self.registry.bindings().items():
            control = controls.get(name)
            if control is not None:
                control.setProperty(COMMAND_PROPERTY, spec.id)
                self.tagged[name] = spec.id
        return dict(self.tagged)

"""Small persistent sidebar wrapper for a QSplitter-based application shell."""

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QWidget,
)


class CollapsibleSidebar(QWidget):
    """Hide/show existing content without deleting or rebuilding its widgets."""

    collapsed_changed = Signal(bool)

    def __init__(
        self,
        title,
        content,
        *,
        side="left",
        minimum_expanded_width=260,
        maximum_expanded_width=620,
    ):
        super().__init__()
        self.title = title
        self.content = content
        self.side = side
        self.minimum_expanded_width = int(minimum_expanded_width)
        self.maximum_expanded_width = int(maximum_expanded_width)
        self.collapsed_width = 52
        hinted_width = content.sizeHint().width() + 24
        self.expanded_width = max(
            self.minimum_expanded_width,
            min(self.maximum_expanded_width, hinted_width),
        )
        self._collapsed = False

        self.toggle_button = QToolButton()
        self.toggle_button.setToolTip(f"Collapse {title} sidebar")
        self.toggle_button.clicked.connect(self.toggle_collapsed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if side == "right":
            layout.addWidget(self.toggle_button)
            layout.addWidget(content, 1)
        else:
            layout.addWidget(content, 1)
            layout.addWidget(self.toggle_button)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._update_button()

    @property
    def is_collapsed(self):
        return self._collapsed

    def remember_width(self, width):
        if self._collapsed:
            return
        self.expanded_width = max(
            self.minimum_expanded_width,
            min(self.maximum_expanded_width, int(width)),
        )

    def toggle_collapsed(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed):
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return

        splitter = self.parentWidget()
        old_width = self.width()
        if not collapsed:
            self._collapsed = False
            self.setMinimumWidth(0)
            self.setMaximumWidth(self.maximum_expanded_width)
            self.content.show()
        else:
            self.remember_width(old_width)
            self._collapsed = True
            self.content.hide()
            self.setMinimumWidth(self.collapsed_width)
            self.setMaximumWidth(self.collapsed_width)

        self._update_button()
        self.collapsed_changed.emit(self._collapsed)

        # QSplitter resizing preserves the live center widget and its OpenGL
        # context. A queued size update lets the new min/max constraints settle.
        if isinstance(splitter, QSplitter):
            target_width = (
                self.collapsed_width if collapsed else self.expanded_width
            )
            QTimer.singleShot(
                0,
                lambda: self._restore_splitter_width(splitter, target_width),
            )

    def _restore_splitter_width(self, splitter, target_width):
        index = splitter.indexOf(self)
        if index < 0 or splitter.count() < 2:
            return
        sizes = splitter.sizes()
        if splitter.count() == 2:
            receiver_index = 1 - index
        else:
            # Main-window sidebars both donate space to the center viewport.
            receiver_index = 1 if index != 1 else max(
                (candidate for candidate in range(len(sizes)) if candidate != index),
                key=lambda candidate: sizes[candidate],
            )
        difference = int(target_width) - sizes[index]
        sizes[index] = int(target_width)
        sizes[receiver_index] = max(100, sizes[receiver_index] - difference)
        splitter.setSizes(sizes)

    def _update_button(self):
        if self._collapsed:
            arrow = "▶" if self.side == "left" else "◀"
            self.toggle_button.setText(f"{self.title}\n{arrow}")
            self.toggle_button.setFixedWidth(self.collapsed_width - 4)
            self.toggle_button.setToolTip(f"Expand {self.title} sidebar")
        else:
            arrow = "◀" if self.side == "left" else "▶"
            self.toggle_button.setText(arrow)
            self.toggle_button.setFixedWidth(24)
            self.toggle_button.setToolTip(f"Collapse {self.title} sidebar")
        self.toggle_button.setArrowType(Qt.ArrowType.NoArrow)

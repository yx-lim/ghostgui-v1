"""Reusable transcript, composer, and staged-motion widgets."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ChatComposer(QPlainTextEdit):
    """Multiline composer where Enter sends and Shift+Enter adds a line."""

    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
            return
        super().keyPressEvent(event)


class ChatMessageWidget(QFrame):
    def __init__(
        self,
        role: str,
        text: str,
        *,
        details=(),
        heading_text: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("aiChatMessage")
        self.setProperty("chatRole", role)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(4)

        if role in {"user", "assistant", "error", "system"}:
            default_heading = {
                "user": "You",
                "assistant": "Assistant",
                "error": "Motion Assistant",
                "system": "Motion Assistant",
            }[role]
            heading = QLabel(heading_text or default_heading)
            heading.setObjectName("aiChatMessageRole")
            heading_font = heading.font()
            heading_font.setBold(True)
            heading.setFont(heading_font)
            layout.addWidget(heading)

        self.body_label = QLabel(str(text))
        self.body_label.setObjectName("aiChatMessageBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.body_label)

        detail_values = tuple(str(value) for value in details if str(value).strip())
        self.details_label = QLabel("\n".join(detail_values))
        self.details_label.setObjectName("aiChatTechnicalDetails")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.hide()
        self.details_button = QToolButton()
        self.details_button.setObjectName("aiChatDetailsButton")
        self.details_button.setText("Details")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.details_label.setVisible)
        if detail_values:
            layout.addWidget(self.details_button, alignment=Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(self.details_label)
        else:
            self.details_button.hide()


class MotionActivityWidget(QFrame):
    """Compact, single-card progress list for one assistant request."""

    def __init__(self, first_step: str, parent=None):
        super().__init__(parent)
        self.setObjectName("aiMotionActivity")
        self._rows = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(1)
        self.add_step(first_step)

    @property
    def step_labels(self):
        return tuple(label for _button, label in self._rows)

    def add_step(self, text: str) -> None:
        if self._rows:
            self._set_row_state(len(self._rows) - 1, "done")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        status = QToolButton(self)
        status.setObjectName("aiActivityStatus")
        status.setFixedSize(16, 16)
        status.setEnabled(False)
        status.setText("…")
        status.setToolTip("In progress")
        label = QLabel(str(text), self)
        label.setObjectName("aiActivityText")
        label.setWordWrap(True)
        font = label.font()
        if font.pointSizeF() > 1.0:
            font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
            label.setFont(font)
        row.addWidget(status, alignment=Qt.AlignmentFlag.AlignTop)
        row.addWidget(label, stretch=1)
        self._layout.addLayout(row)
        self._rows.append((status, label))

    def finish(self, *, success: bool = True) -> None:
        if self._rows:
            self._set_row_state(
                len(self._rows) - 1,
                "done" if success else "failed",
            )

    def _set_row_state(self, index: int, state: str) -> None:
        button, _label = self._rows[index]
        if state == "done":
            button.setText("✓")
            button.setToolTip("Done")
        elif state == "failed":
            button.setText("!")
            button.setToolTip("Stopped")
        else:
            button.setText("…")
            button.setToolTip("In progress")
        button.setProperty("activityState", state)
        button.style().unpolish(button)
        button.style().polish(button)


class MotionResultCard(QFrame):
    accept_requested = Signal()
    discard_requested = Signal()
    refine_requested = Signal()

    def __init__(
        self,
        summary: str,
        *,
        details=(),
        accept_permitted: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("aiMotionResultCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(5)

        self.title_label = QLabel("Motion updated")
        self.title_label.setObjectName("aiMotionResultTitle")
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)
        self.summary_label = QLabel(summary)
        self.summary_label.setObjectName("aiMotionResultSummary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.validation_label = QLabel(
            "Accept unavailable: this candidate did not pass validation."
        )
        self.validation_label.setObjectName("aiMotionResultValidation")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        detail_values = tuple(str(value) for value in details if str(value).strip())
        self.details_button = QToolButton()
        self.details_button.setObjectName("aiMotionResultDetailsButton")
        self.details_button.setText("Warnings and details")
        self.details_button.setCheckable(True)
        self.details_label = QLabel("\n".join(f"• {value}" for value in detail_values))
        self.details_label.setObjectName("aiMotionResultDetails")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details_label.hide()
        if detail_values:
            self.details_button.toggled.connect(self.details_label.setVisible)
            layout.addWidget(self.details_button, alignment=Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(self.details_label)
        else:
            self.details_button.hide()

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 2, 0, 0)
        self.refine_button = QPushButton("Refine preview")
        self.refine_button.setObjectName("aiResultRefineButton")
        self.accept_button = QPushButton("Accept")
        self.accept_button.setObjectName("aiResultAcceptButton")
        self.discard_button = QPushButton("Discard")
        self.discard_button.setObjectName("aiResultDiscardButton")
        self.refine_button.clicked.connect(self.refine_requested.emit)
        self.accept_button.clicked.connect(self.accept_requested.emit)
        self.discard_button.clicked.connect(self.discard_requested.emit)
        actions.addWidget(self.refine_button)
        actions.addWidget(self.accept_button)
        actions.addWidget(self.discard_button)
        layout.addLayout(actions)
        self.set_actions_enabled(accept_permitted, True)

    def set_actions_enabled(
        self,
        accept: bool,
        discard: bool = True,
        refine: bool | None = None,
    ) -> None:
        if refine is None:
            refine = discard
        self.refine_button.setEnabled(bool(refine))
        self.accept_button.setEnabled(bool(accept))
        self.discard_button.setEnabled(bool(discard))
        self.validation_label.setVisible(bool(discard) and not bool(accept))

    def set_resolution(self, text: str) -> None:
        self.title_label.setText(text)
        self.set_actions_enabled(False, False, False)


class ChatTranscriptView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aiChatTranscript")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.body = QWidget()
        self.body.setObjectName("aiChatTranscriptBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(2, 4, 2, 4)
        self.body_layout.setSpacing(7)
        self.body_layout.addStretch(1)
        self.setWidget(self.body)
        self.verticalScrollBar().rangeChanged.connect(
            lambda _minimum, _maximum: self.scroll_to_bottom()
        )

    def append_widget(self, widget: QWidget) -> None:
        self.body_layout.insertWidget(self.body_layout.count() - 1, widget)
        QTimer.singleShot(0, lambda: self._reveal_widget(widget))
        QTimer.singleShot(1, self.scroll_to_bottom)

    def clear_entries(self) -> None:
        while self.body_layout.count() > 1:
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def scroll_to_bottom(self) -> None:
        try:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        except RuntimeError:
            # A queued scroll may outlive a panel closed during shutdown/tests.
            return

    def _reveal_widget(self, widget: QWidget) -> None:
        try:
            self.ensureWidgetVisible(widget, 0, 0)
            self.scroll_to_bottom()
        except RuntimeError:
            return

"""Anchored, non-modal transcript popover for the Claude command bar.

Model text is rendered as plain text, so model output cannot inject HTML or links.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

STATUS_ICONS = {"preparing": "…", "running": "⟳", "ok": "✓", "error": "✗", "deferred": "⏸", "declined": "⊘"}


def _label(text="", *, name=None, wrap=True, plain=True):
    label = QLabel(text)
    label.setWordWrap(wrap)
    if plain:
        label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if name:
        label.setObjectName(name)
    return label


class ToolCallChip(QFrame):
    def __init__(self, title, summary, parent=None):
        super().__init__(parent)
        self.setObjectName("agentToolChip")
        self.title, self.summary = title, summary
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        self.status = _label(STATUS_ICONS["running"], wrap=False)
        self.text = _label()
        self.duration = _label(wrap=False)
        layout.addWidget(self.status)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.duration)
        self.set_state("running")

    def set_state(self, state, *, detail="", duration_ms=0):
        self.state = state
        self.status.setText(STATUS_ICONS.get(state, "?"))
        words = {"ok": "done", "error": "failed", "running": "running", "preparing": "preparing",
                 "deferred": "planned", "declined": "declined"}.get(state, state)
        text = f"{self.title} · {self.summary}" if self.summary else self.title
        if detail:
            text += f" — {detail}"
        self.text.setText(text)
        self.setAccessibleName(f"{self.title} {words}")
        self.duration.setText(f"{duration_ms} ms" if duration_ms else "")


class Card(QFrame):
    """Base for approval, question, plan and result cards."""

    def __init__(self, title, body, parent=None, *, name="agentCard"):
        super().__init__(parent)
        self.setObjectName(name)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(8, 6, 8, 6)
        header = _label(title)
        header.setStyleSheet("font-weight: 600;")
        self.layout_.addWidget(header)
        if body:
            self.layout_.addWidget(_label(body))
        self.buttons = QHBoxLayout()
        self.layout_.addLayout(self.buttons)

    def add_button(self, text, callback, *, name=None, default=False):
        button = QPushButton(text)
        if name:
            button.setObjectName(name)
        button.setAutoDefault(default)
        button.clicked.connect(callback)
        self.buttons.addWidget(button)
        if default:
            QTimer.singleShot(0, button.setFocus)
        return button

    def resolve(self, text):
        for index in reversed(range(self.buttons.count())):
            widget = self.buttons.itemAt(index).widget()
            if widget is not None:
                widget.setEnabled(False)
        self.layout_.addWidget(_label(text))


class AgentTranscript(QFrame):
    """Child overlay of the main window, anchored top-right under the command bar."""

    closed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("agentTranscript")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        header = QHBoxLayout()
        header.addWidget(_label("Claude", wrap=False))
        header.addStretch(1)
        self.status = _label("", name="agentTranscriptStatus", wrap=False)
        header.addWidget(self.status)
        self.close_button = QPushButton("Hide")
        self.close_button.setObjectName("agentTranscriptHideButton")
        self.close_button.clicked.connect(self.hide_popover)
        header.addWidget(self.close_button)
        outer.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.rows = QVBoxLayout(self.body)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.addStretch(1)
        self.scroll.setWidget(self.body)
        outer.addWidget(self.scroll, 1)
        self.chips: dict[str, ToolCallChip] = {}
        self._stream_label = None
        self._stream_text = ""
        self._flush = QTimer(self)
        self._flush.setInterval(50)
        self._flush.setSingleShot(True)
        self._flush.timeout.connect(self._flush_stream)
        self.hide()

    # ---------------------------------------------------------------- layout
    def show_popover(self):
        self.reposition()
        self.show()
        self.raise_()

    def hide_popover(self):
        self.hide()
        self.closed.emit()

    def reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        width = min(520, max(320, parent.width() // 3))
        height = max(240, int(parent.height() * 0.6))
        top = 0
        bar = getattr(parent, "agent_command_bar", None)
        if bar is not None and bar.isVisible():
            top = bar.mapTo(parent, bar.rect().bottomLeft()).y() + 2
        self.setGeometry(parent.width() - width - 12, top, width, min(height, parent.height() - top - 12))

    def _add(self, widget):
        self.rows.insertWidget(self.rows.count() - 1, widget)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        return widget

    # --------------------------------------------------------------- content
    def add_user(self, text):
        self._end_stream()
        self._add(_label(f"You  {text}", name="agentUserMessage"))

    def add_notice(self, text):
        self._end_stream()
        self._add(_label(text, name="agentNotice"))

    def append_stream(self, kind, text):
        if kind == "thinking":
            return  # summarized thinking stays out of the main transcript in v1
        if self._stream_label is None:
            self._stream_label = self._add(_label("", name="agentAssistantMessage"))
            self._stream_text = "Claude  "
        self._stream_text += text
        if not self._flush.isActive():
            self._flush.start()

    def _flush_stream(self):
        if self._stream_label is not None:
            self._stream_label.setText(self._stream_text)

    def _end_stream(self):
        self._flush_stream()
        self._stream_label = None

    def chip(self, call_id, title, summary, state, **detail):
        self._end_stream()
        key = call_id or f"{title}:{summary}:{len(self.chips)}"
        chip = self.chips.get(key)
        if chip is None:
            chip = self._add(ToolCallChip(title, summary))
            self.chips[key] = chip
        chip.set_state(state, **detail)
        return chip

    def add_card(self, card):
        self._end_stream()
        self.show_popover()
        return self._add(card)

    def set_status(self, text):
        self.status.setText(text)

    def add_question_card(self, card_info, answer):
        card = Card(card_info.title, card_info.text, name="agentQuestionCard")
        for option in card_info.options:
            card.add_button(option, lambda checked=False, value=option: (answer(value), card.resolve(f"You: {value}")))
        field = QLineEdit()
        field.setObjectName("agentQuestionAnswer")
        field.setPlaceholderText("Type an answer and press Enter")
        field.returnPressed.connect(lambda: field.text().strip() and (
            answer(field.text().strip()), card.resolve(f"You: {field.text().strip()}")))
        card.layout_.addWidget(field)
        card.add_button("Skip", lambda: (answer(None), card.resolve("Skipped")))
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return self.add_card(card)

    def add_confirmation_card(self, card_info, answer):
        body = card_info.text + (f"\nYour request: “{card_info.prompt}”" if card_info.prompt else "")
        card = Card(f"Claude wants to: {card_info.title}", body, name="agentApprovalCard")
        card.add_button("Deny", lambda: (answer("deny"), card.resolve("Denied")),
                        name="agentApprovalDeny", default=True)
        card.add_button("Allow once", lambda: (answer("once"), card.resolve("Allowed once")),
                        name="agentApprovalOnce")
        card.add_button("Allow for this run", lambda: (answer("run"), card.resolve("Allowed for this run")),
                        name="agentApprovalRun")
        return self.add_card(card)

"""The "Ask Claude" command bar: a second top toolbar row, offline palette and run UI."""

from __future__ import annotations

from difflib import SequenceMatcher
from threading import Event

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QToolBar, QVBoxLayout,
)

from application.ai.agent_events import RunOutcome
from application.ai.secrets import SecretStr, contains_secret
from application.commands import Surface, default_registry

from .agent_run_controller import AgentRunController
from .agent_transcript import Card

MODELS = (("Opus 5.5", "claude-opus-5-5"), ("Sonnet 5.5", "claude-sonnet-5-5"), ("Haiku 5.5", "claude-haiku-5-5"))
OUTCOME_TEXT = {
    RunOutcome.COMPLETED: "Done", RunOutcome.STOPPED: "Stopped by you", RunOutcome.REFUSED: "Claude declined",
    RunOutcome.CONTEXT_FULL: "This conversation is full; start a new conversation",
    RunOutcome.TURN_LIMIT: "Stopped at the run limit", RunOutcome.BUDGET: "Stopped at the spend cap",
    RunOutcome.TOOL_ERRORS: "Stopped after repeated tool errors", RunOutcome.ERROR: "Claude could not finish",
    RunOutcome.PLAN_READY: "Plan ready for your approval",
}


def palette_matches(text, registry=None, limit=5):
    """Fuzzy match over palette command titles, tools and aliases (offline mode)."""
    registry = registry or default_registry()
    query = text.strip().lstrip("/").lower()
    scored = []
    for spec in registry.projection(Surface.PALETTE):
        names = [spec.title.lower(), spec.tool.replace("_", " "), *[alias.lower() for alias in spec.aliases]]
        score = max(SequenceMatcher(None, query, name).ratio() + (0.5 if query and query in name else 0)
                    for name in names)
        scored.append((score, spec))
    scored.sort(key=lambda item: -item[0])
    return [spec for score, spec in scored[:limit] if score > 0.45]


class AnthropicKeyDialog(QDialog):
    """User-initiated key entry: session-only or system keychain. The key is never displayed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Claude Settings")
        self.setObjectName("agentKeyDialog")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.key = QLineEdit()
        self.key.setObjectName("agentKeyField")
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("sk-ant-…")
        form.addRow("Anthropic API key", self.key)
        self.store = QCheckBox("Store in the system keychain")
        self.store.setObjectName("agentKeyStore")
        self.store.setChecked(True)
        form.addRow("", self.store)
        layout.addLayout(form)
        layout.addWidget(QLabel("Your key is used only for requests you start. GhostGUI never writes it to "
                                "projects, logs or exports."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def secret(self):
        text = self.key.text().strip()
        return SecretStr(text) if text else None


class AgentCommandBar(QToolBar):
    def __init__(self, window, *, provider_factory=None):
        super().__init__("Claude", window)
        self.setObjectName("agentCommandBar")
        self.setMovable(False)
        self.window_ = window
        self.session_key: SecretStr | None = None
        self.controller = AgentRunController(window, provider_factory=provider_factory or self._default_provider)
        self.transcript = window.agent_transcript
        self.prompt = QLineEdit()
        self.prompt.setObjectName("agentPromptField")
        self.prompt.setPlaceholderText("Ask Claude to do anything in GhostGUI…   Ctrl+K")
        self.prompt.setClearButtonEnabled(True)
        self.prompt.returnPressed.connect(self.submit)
        self.prompt.setAccessibleName("Ask Claude")
        self.addWidget(QLabel(" ✦ "))
        self.addWidget(self.prompt)
        self.model = QComboBox()
        self.model.setObjectName("agentModelCombo")
        for label, model in MODELS:
            self.model.addItem(label, model)
        self.model.setToolTip("Model for this conversation; changing it starts a new conversation")
        self.model.currentIndexChanged.connect(self._model_changed)
        self.addWidget(self.model)
        self.approval = QComboBox()
        self.approval.setObjectName("agentApprovalCombo")
        self.approval.addItem("Review", "review")
        self.approval.addItem("Apply automatically", "auto")
        self.approval.setToolTip("Review: you Accept the Orange preview. Apply automatically lasts this session only.")
        self.approval.currentIndexChanged.connect(self._policy_changed)
        self.addWidget(self.approval)
        self.plan = QCheckBox("Plan first")
        self.plan.setObjectName("agentPlanFirst")
        self.plan.toggled.connect(self._policy_changed)
        self.addWidget(self.plan)
        self.send = QPushButton("Send")
        self.send.setObjectName("agentSendButton")
        self.send.clicked.connect(self._send_or_stop)
        self.addWidget(self.send)
        self.cost = QLabel("$0.00")
        self.cost.setObjectName("agentCostLabel")
        self.cost.setToolTip("Estimated session spend; set Console spend limits too")
        self.addWidget(self.cost)
        self.history_button = QPushButton("Transcript")
        self.history_button.setObjectName("agentTranscriptButton")
        self.history_button.clicked.connect(self._toggle_transcript)
        self.addWidget(self.history_button)
        self.new_button = QPushButton("New")
        self.new_button.setObjectName("agentNewConversationButton")
        self.new_button.setToolTip("Start a new conversation")
        self.new_button.clicked.connect(self.new_conversation)
        self.addWidget(self.new_button)
        self.key_button = QPushButton("Key…")
        self.key_button.setObjectName("agentKeyButton")
        self.key_button.clicked.connect(self.edit_key)
        self.addWidget(self.key_button)
        self.pill = QLabel("Claude is controlling GhostGUI · Stop (Esc)", window.viewer_3d_stack)
        self.pill.setObjectName("agentControlPill")
        self.pill.setStyleSheet("background: palette(highlight); color: palette(highlighted-text);"
                                " border-radius: 10px; padding: 3px 10px;")
        self.pill.hide()
        self._history: list[str] = []
        self._history_index = 0
        self._last_prompt = ""
        self.controller.event.connect(self._on_event)
        self.controller.card.connect(self._on_card)
        self.controller.run_finished.connect(self._on_finished)
        self.controller.busy_changed.connect(self._on_busy)
        window.agent_bridge_controller.external_activity.connect(self._on_external)
        self.prompt.installEventFilter(self)
        focus = QShortcut(QKeySequence("Ctrl+K"), window)
        focus.setObjectName("agentFocusShortcut")
        focus.activated.connect(self.focus_prompt)
        # Esc is window-wide only while Claude runs, so it never steals the canvas's
        # "cancel the active drag" key otherwise.
        stop = QShortcut(QKeySequence(Qt.Key.Key_Escape), window)
        stop.setObjectName("agentStopShortcut")
        stop.setContext(Qt.ShortcutContext.WindowShortcut)
        stop.setEnabled(False)
        stop.activated.connect(self._escape)
        self.stop_shortcut = stop
        self._shortcuts = (focus, stop)

    # ------------------------------------------------------------- provider
    def _credential(self):
        if self.session_key is not None:
            return self.session_key
        try:
            from application.ai.credentials import default_credential_source
            value = default_credential_source().get_secret("anthropic")
        except Exception:
            value = None
        return SecretStr(value) if value else None

    def _default_provider(self):
        key = self._credential()
        if key is None:
            return None
        try:
            from application.ai.provider_registry import DEFAULT_PROVIDER_REGISTRY
            return DEFAULT_PROVIDER_REGISTRY.get("anthropic").create_agent(api_key=key)
        except Exception as error:
            self.transcript.add_notice(str(error)[:300])
            return None

    def edit_key(self):
        dialog = AnthropicKeyDialog(self.window_)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        secret = dialog.secret()
        if secret is None:
            return
        if secret.is_admin_key:
            self.transcript.add_notice("That looks like an Admin API key; use a regular API key instead.")
            self.transcript.show_popover()
            return
        if dialog.store.isChecked():
            try:
                from application.ai.credentials import SystemKeyringCredentialStore
                SystemKeyringCredentialStore().set_secret("anthropic", secret.reveal())
                self.session_key = None
            except Exception:
                self.session_key = secret
                self.transcript.add_notice("The system keychain is unavailable; the key is kept for this session only.")
        else:
            self.session_key = secret
        self.controller.set_provider(None)

    # ------------------------------------------------------------------ input
    def focus_prompt(self):
        self.prompt.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.prompt.selectAll()

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if (watched is self.prompt and event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape):
            self._escape()
            return True
        if watched is self.prompt and event.type() == QEvent.Type.FocusIn:
            try:
                self.controller.prewarm_if_idle()
            except Exception:
                pass
        return super().eventFilter(watched, event)

    def _on_external(self, tool, ok, message):
        """External MCP activity is shown and logged, so the user and Claude can see it."""
        self.controller.action_log.record(origin="agent_mcp", command=tool, arguments={},
                                          status="ok" if ok else "error")
        self.transcript.add_notice(f"External agent · {tool} {'✓' if ok else '✗ ' + message}")

    def _escape(self):
        if self.controller.running:
            self.controller.stop()
        elif self.transcript.isVisible():
            self.transcript.hide_popover()

    def _send_or_stop(self):
        if self.controller.running:
            self.controller.stop()
        else:
            self.submit()

    def submit(self):
        text = self.prompt.text().strip()
        if not text:
            return
        if contains_secret(text):
            self.transcript.add_notice("Your message looks like it contains an API key. Use Key… to store it securely.")
            self.transcript.show_popover()
            return
        self._history.append(text)
        self._history_index = len(self._history)
        self.prompt.clear()
        self.transcript.show_popover()
        self.transcript.add_user(text)
        if text.startswith("/") or self.controller.ensure_provider() is None:
            self.run_local(text)
            return
        self._last_prompt = text
        self.controller.start(text)

    def run_local(self, text):
        """Offline palette: run the best zero-argument match through the same registry."""
        matches = palette_matches(text)
        runnable = [spec for spec in matches if not spec.params.get("required")]
        if not runnable:
            hint = ", ".join(spec.title for spec in matches[:3])
            self.transcript.add_notice("Claude is unavailable — GhostGUI commands still work. "
                                       + (f"Did you mean: {hint}?" if hint else "No matching command."))
            return
        spec = runnable[0]
        chip = self.transcript.chip("", spec.title, "", "running")
        future = self.controller.bridge.submit_inapp(spec.tool, {}, Event())

        def done():
            if not future.done():
                QTimer.singleShot(50, done)
                return
            result = future.result()
            ok = bool(result.get("ok"))
            chip.set_state("ok" if ok else "error",
                           detail="" if ok else str(result.get("payload", {}).get("message", ""))[:200])

        QTimer.singleShot(50, done)

    # -------------------------------------------------------------- settings
    def _model_changed(self):
        try:
            self.controller.set_model(self.model.currentData())
            self.transcript.add_notice(f"New conversation with {self.model.currentText()}.")
        except RuntimeError as error:
            self.transcript.add_notice(str(error))

    def _policy_changed(self):
        self.controller.policy.approval = self.approval.currentData()
        self.controller.policy.plan_mode = self.plan.isChecked()

    def new_conversation(self):
        try:
            self.controller.new_conversation()
            self.transcript.add_notice("Started a new conversation.")
        except RuntimeError as error:
            self.transcript.add_notice(str(error))

    def _toggle_transcript(self):
        if self.transcript.isVisible():
            self.transcript.hide_popover()
        else:
            self.transcript.show_popover()

    # ----------------------------------------------------------------- events
    def _on_busy(self, busy):
        self.send.setText("Stop" if busy else "Send")
        self.stop_shortcut.setEnabled(bool(busy))
        self.model.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.pill.setVisible(busy)
        if busy:
            self.pill.adjustSize()
            self.pill.move(max(8, (self.pill.parentWidget().width() - self.pill.width()) // 2), 8)
            self.pill.raise_()
        self.transcript.set_status("Working…" if busy else "")

    def _on_event(self, event):
        kind = event.kind
        if kind in ("text", "thinking"):
            self.transcript.append_stream(kind, event.text)
        elif kind == "tool_preparing":
            self.transcript.set_status(f"Working · {event.title}")
        elif kind == "tool_started":
            self.transcript.chip(event.call_id, event.title, event.summary, "running")
        elif kind == "tool_finished":
            detail = "" if event.ok else str(event.data.get("message") or event.summary)[:200]
            state = "ok" if event.ok else ("declined" if event.summary == "Declined by you" else "error")
            self.transcript.chip(event.call_id, event.title, event.summary, state, detail=detail,
                                 duration_ms=event.duration_ms)
        elif kind == "tool_deferred":
            self.transcript.chip(event.call_id, event.title, event.summary, "deferred")
        elif kind == "notice":
            self.transcript.add_notice(event.text)
        elif kind == "usage":
            self.cost.setText(f"${event.data.get('session_usd', 0.0):.2f} · {event.data.get('cache_read', 0) // 1000}k↺")

    def _on_card(self, card):
        answer = lambda value, identifier=card.identifier: self.controller.answer(identifier, value)  # noqa: E731
        if card.kind == "confirmation":
            self.transcript.add_confirmation_card(card, answer)
        else:
            self.transcript.add_question_card(card, answer)

    def _on_finished(self, result):
        summary = OUTCOME_TEXT.get(result.outcome, result.outcome.value)
        if result.message and result.outcome is not RunOutcome.COMPLETED:
            summary += f" — {result.message}"
        cost = "" if result.cost_usd is None else f" · ${result.cost_usd:.2f}"
        self.transcript.add_notice(f"Run · {result.turns} turns · {result.actions} actions{cost} · {summary}")
        if result.outcome is RunOutcome.PLAN_READY and result.deferred:
            self._plan_card(result.deferred)
        if self.controller.bridge.has_candidate:
            self._review_card()

    def _plan_card(self, deferred):
        steps = "\n".join(f"{index + 1}. {call.preview}" for index, call in enumerate(deferred))
        card = Card("Plan", steps, name="agentPlanCard")
        card.add_button("Approve & run", lambda: (self.controller.approve_plan(deferred), card.resolve("Approved")),
                        name="agentPlanApprove")
        card.add_button("Discard plan", lambda: card.resolve("Discarded"), name="agentPlanDiscard")
        self.transcript.add_card(card)

    def _review_card(self):
        card = Card("Orange preview ready", "Review the candidate motion in the viewport.", name="agentReviewCard")
        card.add_button("Accept", lambda: (self.controller.accept_candidate(), card.resolve("Accepted")),
                        name="agentReviewAccept")
        card.add_button("Discard", lambda: (self.controller.discard_candidate(), card.resolve("Discarded")),
                        name="agentReviewDiscard")
        self.transcript.add_card(card)

    def shutdown(self):
        self.controller.shutdown()

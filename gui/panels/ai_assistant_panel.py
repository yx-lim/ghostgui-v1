"""Transcript-first Motion Assistant panel."""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from application.ai.chat_transcript import ChatEntryKind, MotionChatTranscript
from application.ai.progress import AIProgressEvent, AIProgressStage
from gui.widgets.ai_chat import (
    ChatComposer,
    ChatMessageWidget,
    ChatTranscriptView,
    MotionResultCard,
)


class AIAssistantPanelState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    STAGED = "staged"
    ERROR = "error"


class AIAssistantPanel(QWidget):
    """Render a conversation while the controller owns motion/session state."""

    submit_requested = Signal(str)
    critique_requested = Signal(str)
    visual_refine_requested = Signal(str)
    visual_verify_requested = Signal(str)
    refine_requested = Signal(str)
    preview_requested = Signal()
    accept_requested = Signal()
    reject_requested = Signal()
    cancel_requested = Signal()
    settings_requested = Signal()
    progress_received = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aiAssistantPanel")
        self.setMinimumWidth(0)
        self._state = AIAssistantPanelState.READY
        self._accept_permitted = False
        self._conversation_complete = False
        self._active_result_card: MotionResultCard | None = None
        self._assistant_display_name = "Assistant"
        self._message_widgets: list[QWidget] = []
        self.transcript = MotionChatTranscript()
        self.progress_received.connect(
            self.show_progress,
            Qt.ConnectionType.QueuedConnection,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setContentsMargins(2, 0, 0, 0)
        title = QLabel("Motion Assistant")
        title.setObjectName("aiChatTitle")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        header.addWidget(title)
        self.provider_label = QLabel("Gemini")
        self.provider_label.setObjectName("aiProviderLabel")
        header.addWidget(self.provider_label, stretch=1)
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("aiSettingsButton")
        self.settings_button.setText("Settings")
        self.settings_button.setToolTip("Configure provider, model, and API key")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        header.addWidget(self.settings_button)
        layout.addLayout(header)

        self.transcript_view = ChatTranscriptView()
        self.transcript_view.setMinimumHeight(260)
        layout.addWidget(self.transcript_view, stretch=1)

        self.context_label = QLabel("Current motion")
        self.context_label.setObjectName("aiContextIndicators")
        self.context_label.setWordWrap(True)
        layout.addWidget(self.context_label)

        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(0, 0, 0, 0)
        composer_row.setSpacing(5)
        self.prompt_input = ChatComposer()
        self.prompt_input.setObjectName("aiPromptInput")
        self.prompt_input.setPlaceholderText(
            "Ask Claude to create or modify this motion…"
        )
        self.prompt_input.setMaximumHeight(96)
        self.prompt_input.send_requested.connect(self._emit_submit)
        composer_row.addWidget(self.prompt_input, stretch=1)
        self.submit_button = QPushButton("↑")
        self.submit_button.setObjectName("aiSubmitButton")
        self.submit_button.setToolTip("Send message (Enter)")
        self.submit_button.setFixedWidth(36)
        self.submit_button.clicked.connect(self._emit_submit)
        composer_row.addWidget(self.submit_button)
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setObjectName("aiCancelButton")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        composer_row.addWidget(self.cancel_button)
        layout.addLayout(composer_row)

        # Compatibility affordances remain wired for older integrations and
        # developer tests, but the normal user sees the transcript result card.
        self.response_label = QLabel(self)
        self.response_label.setObjectName("aiResponseLabel")
        self.response_label.hide()
        self.proposal_heading = QLabel("Proposed changes", self)
        self.proposal_heading.setObjectName("aiProposalHeading")
        self.proposal_heading.hide()
        self.proposal_list = QListWidget(self)
        self.proposal_list.setObjectName("aiProposalList")
        self.proposal_list.hide()
        self.preview_button = QPushButton("Preview candidate", self)
        self.preview_button.setObjectName("aiPreviewButton")
        self.preview_button.clicked.connect(self.preview_requested.emit)
        self.preview_button.hide()
        self.accept_button = QPushButton("Accept", self)
        self.accept_button.setObjectName("aiAcceptButton")
        self.accept_button.clicked.connect(self.accept_requested.emit)
        self.accept_button.hide()
        self.reject_button = QPushButton("Discard", self)
        self.reject_button.setObjectName("aiRejectButton")
        self.reject_button.clicked.connect(self.reject_requested.emit)
        self.reject_button.hide()
        self.refine_button = QPushButton("Refine", self)
        self.refine_button.setObjectName("aiRefineButton")
        self.refine_button.clicked.connect(self._emit_refine)
        self.refine_button.hide()
        self.visual_refine_button = QPushButton("Visual refine", self)
        self.visual_refine_button.clicked.connect(self._emit_visual_refine)
        self.visual_refine_button.hide()
        self.visual_verify_button = QPushButton("Verify visually", self)
        self.visual_verify_button.clicked.connect(self._emit_visual_verify)
        self.visual_verify_button.hide()
        self.critique_button = QPushButton("Critique", self)
        self.critique_button.clicked.connect(self._emit_critique)
        self.critique_button.hide()

        self._append_entry(
            ChatEntryKind.SYSTEM,
            "Tell me what you want to create or change in the current motion.",
        )
        self.set_state(AIAssistantPanelState.READY)

    @property
    def state(self) -> AIAssistantPanelState:
        return self._state

    @property
    def message_widgets(self) -> tuple[QWidget, ...]:
        return tuple(self._message_widgets)

    def set_provider(self, provider_name: str, model: str = "") -> None:
        provider = provider_name.strip().title() or "Not configured"
        model = model.strip()
        self._assistant_display_name = {
            "anthropic": "Claude",
            "gemini": "Gemini",
        }.get(provider_name.strip().lower(), provider)
        self.provider_label.setText(model or provider)
        self.provider_label.setToolTip(
            provider + (f" · {model}" if model else "")
        )
        self.prompt_input.setPlaceholderText(
            f"Ask {self._assistant_display_name} to create or modify this motion…"
        )

    def set_context_indicators(self, values) -> None:
        indicators = tuple(
            str(value).strip() for value in values if str(value).strip()
        )
        self.context_label.setText("  ·  ".join(indicators) or "Current motion")

    def set_state(self, state: AIAssistantPanelState) -> None:
        self._state = AIAssistantPanelState(state)
        running = self._state is AIAssistantPanelState.RUNNING
        staged = self._state is AIAssistantPanelState.STAGED
        self.settings_button.setEnabled(not running)
        self.prompt_input.setEnabled(not running)
        self.submit_button.setVisible(not running)
        self.submit_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.preview_button.setEnabled(staged)
        self.accept_button.setEnabled(staged and self._accept_permitted)
        self.reject_button.setEnabled(staged)
        self.refine_button.setEnabled(staged)
        self.visual_refine_button.setEnabled(staged)
        self.visual_verify_button.setEnabled(staged)
        if self._active_result_card is not None and not running:
            self._active_result_card.set_actions_enabled(
                staged and self._accept_permitted,
                staged,
            )

    def set_accept_permitted(self, permitted: bool) -> None:
        self._accept_permitted = bool(permitted)
        self.set_state(self._state)

    def add_user_message(self, instruction: str) -> None:
        if self._conversation_complete:
            self.clear_transcript()
        entry = self.transcript.start_turn(instruction)
        self._append_widget(ChatMessageWidget("user", entry.text))

    def begin_request(
        self,
        *,
        refinement: bool = False,
        critique: bool = False,
        visual_refinement: bool = False,
        visual_verification: bool = False,
    ) -> None:
        self.set_state(AIAssistantPanelState.RUNNING)
        if self._active_result_card is not None:
            self._active_result_card.set_actions_enabled(False, False)
        if visual_verification:
            message = "Comparing the original and staged motion"
        elif visual_refinement:
            message = "Inspecting the staged motion"
        elif critique:
            message = "Inspecting representative motion views"
        elif refinement:
            message = "Reading the staged motion"
        else:
            message = "Reading the current motion"
        self.response_label.setText(message)
        self._append_entry(ChatEntryKind.ACTIVITY, message)

    def show_progress(self, event: object) -> None:
        if self._state is not AIAssistantPanelState.RUNNING:
            return
        if not isinstance(event, AIProgressEvent):
            return
        messages = {
            AIProgressStage.PLANNING_STARTED: "Inspecting representative states",
            AIProgressStage.STRUCTURED_PLAN_COMPLETED: "Preparing the motion update",
            AIProgressStage.LOCAL_OPERATION: "Applying motion changes",
            AIProgressStage.VALIDATION: "Validating the candidate",
            AIProgressStage.DONE: "✓ Candidate ready",
        }
        message = messages.get(event.stage, event.message)
        if (
            self.transcript.entries
            and self.transcript.entries[-1].kind is ChatEntryKind.ACTIVITY
            and self.transcript.entries[-1].text == message
        ):
            return
        self.response_label.setText(message)
        self._append_entry(ChatEntryKind.ACTIVITY, message)

    def show_proposal(
        self,
        response: str,
        changes: tuple[str, ...],
        *,
        accept_permitted: bool = False,
        result_summary: str | None = None,
    ) -> None:
        if self._active_result_card is not None:
            self._active_result_card.set_resolution("Superseded motion")
        response = response.strip() or "I updated the motion for review."
        self.response_label.setText(response)
        self._append_entry(ChatEntryKind.ASSISTANT, response)

        detail_values = tuple(changes) or ("Motion working copy updated",)
        warning_count = sum(
            value.lower().startswith("warning:") for value in detail_values
        )
        summary = result_summary or (
            "Candidate ready"
            + (f" · {warning_count} warnings" if warning_count else "")
        )
        entry = self.transcript.append(
            ChatEntryKind.RESULT,
            summary,
            details=detail_values,
        )
        card = MotionResultCard(
            entry.text,
            details=entry.details,
            accept_permitted=accept_permitted,
        )
        card.accept_requested.connect(self.accept_requested.emit)
        card.discard_requested.connect(self.reject_requested.emit)
        self._active_result_card = card
        self._append_widget(card)

        self.proposal_heading.setText("Proposed changes")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(detail_values))
        self.prompt_input.clear()
        self._accept_permitted = bool(accept_permitted)
        self.set_state(AIAssistantPanelState.STAGED)

    def show_critique(
        self,
        summary: str,
        observations: tuple[str, ...],
        *,
        session_staged: bool = False,
    ) -> None:
        summary = summary.strip() or "I finished inspecting the motion."
        self.response_label.setText(summary)
        self._append_entry(ChatEntryKind.ASSISTANT, summary, details=observations)
        self.proposal_heading.setText("Visual observations")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(observations) or ["No visible issue reported"])
        self.prompt_input.clear()
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.READY
        )

    def show_verification(self, summary: str, observations: tuple[str, ...]) -> None:
        summary = summary.strip() or "I finished verifying the staged motion."
        self.response_label.setText(summary)
        self._append_entry(ChatEntryKind.ASSISTANT, summary, details=observations)
        self.proposal_heading.setText("Visual verification")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(observations) or ["No visible issue reported"])
        self.set_state(AIAssistantPanelState.STAGED)

    def show_error(self, message: str, *, session_staged: bool = False) -> None:
        summary, details = _compact_error(message)
        self.response_label.setText(f"AI error: {message}")
        self._append_entry(ChatEntryKind.ERROR, summary, details=details)
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.ERROR
        )

    def show_cancelled(self, *, session_staged: bool = False) -> None:
        message = "Request cancelled. The committed motion is unchanged."
        self.response_label.setText(message)
        self._append_entry(ChatEntryKind.SYSTEM, message)
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.READY
        )

    def complete_session(self, message: str, *, accepted: bool) -> None:
        if self._active_result_card is not None:
            self._active_result_card.set_resolution(
                "Motion accepted" if accepted else "Motion discarded"
            )
        self._append_entry(ChatEntryKind.SYSTEM, message)
        self.response_label.setText(message)
        self._conversation_complete = True
        self._active_result_card = None
        self._accept_permitted = False
        self.prompt_input.clear()
        self.set_state(AIAssistantPanelState.READY)

    def reset_session(self, message: str) -> None:
        self.clear_transcript()
        self._append_entry(ChatEntryKind.SYSTEM, message)
        self.response_label.setText(message)
        self.prompt_input.clear()
        self.set_state(AIAssistantPanelState.READY)

    def clear_transcript(self) -> None:
        self.transcript.clear()
        self.transcript_view.clear_entries()
        self._message_widgets.clear()
        self._active_result_card = None
        self._conversation_complete = False
        self._accept_permitted = False

    def _append_entry(self, kind, text, *, details=()) -> None:
        entry = self.transcript.append(kind, text, details=details)
        self._append_widget(
            ChatMessageWidget(
                entry.kind.value,
                entry.text,
                details=entry.details,
                heading_text=(
                    self._assistant_display_name
                    if entry.kind is ChatEntryKind.ASSISTANT
                    else None
                ),
            )
        )

    def _append_widget(self, widget: QWidget) -> None:
        self._message_widgets.append(widget)
        self.transcript_view.append_widget(widget)

    def _instruction(self) -> str:
        return self.prompt_input.toPlainText().strip()

    def _emit_submit(self) -> None:
        instruction = self._instruction()
        if not instruction:
            self._append_entry(
                ChatEntryKind.SYSTEM,
                f"Describe the motion you want {self._assistant_display_name} "
                "to create or change.",
            )
            self.prompt_input.setFocus()
            return
        self.add_user_message(instruction)
        if self._state is AIAssistantPanelState.STAGED:
            self.refine_requested.emit(instruction)
        else:
            self.submit_requested.emit(instruction)

    def _emit_refine(self) -> None:
        instruction = self._instruction()
        if instruction:
            self.add_user_message(instruction)
            self.refine_requested.emit(instruction)

    def _emit_critique(self) -> None:
        self.critique_requested.emit(
            self._instruction() or "What is visually wrong with this motion?"
        )

    def _emit_visual_refine(self) -> None:
        self.visual_refine_requested.emit(self._instruction())

    def _emit_visual_verify(self) -> None:
        self.visual_verify_requested.emit(self._instruction())


def _compact_error(message: str) -> tuple[str, tuple[str, ...]]:
    text = " ".join(str(message).split())
    if not text:
        return "I couldn't complete that request.", ()
    clauses = tuple(value.strip() for value in text.split(";") if value.strip())
    first = clauses[0]
    if len(first) > 180:
        first = first[:177].rstrip() + "…"
    summary = f"I couldn't produce a usable motion: {first}"
    details = clauses if len(clauses) > 1 or len(text) > len(first) else ()
    return summary, details

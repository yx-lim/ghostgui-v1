"""Compact Motion Assistant panel with explicit request/session states."""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class AIAssistantPanelState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    STAGED = "staged"
    ERROR = "error"


class AIAssistantPanel(QWidget):
    """Render AI workflow state and emit intent without owning motion state."""

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aiAssistantPanel")
        self.setMinimumWidth(0)
        self._state = AIAssistantPanelState.READY

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        provider_row = QHBoxLayout()
        provider_row.setContentsMargins(0, 0, 0, 0)
        self.provider_label = QLabel("Provider: Gemini")
        self.provider_label.setObjectName("aiProviderLabel")
        provider_row.addWidget(self.provider_label, stretch=1)
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("aiSettingsButton")
        self.settings_button.setText("Settings")
        self.settings_button.setToolTip("Configure AI provider, model, and API key")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        provider_row.addWidget(self.settings_button)
        layout.addLayout(provider_row)

        self.response_label = QLabel(
            "Describe a focused edit to the current motion or selection."
        )
        self.response_label.setObjectName("aiResponseLabel")
        self.response_label.setWordWrap(True)
        self.response_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.response_label)

        self.proposal_heading = QLabel("Proposed changes")
        self.proposal_heading.setObjectName("aiProposalHeading")
        self.proposal_heading.hide()
        layout.addWidget(self.proposal_heading)

        self.proposal_list = QListWidget()
        self.proposal_list.setObjectName("aiProposalList")
        self.proposal_list.setMaximumHeight(104)
        self.proposal_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.proposal_list.hide()
        layout.addWidget(self.proposal_list)

        self.preview_button = QPushButton("Preview")
        self.preview_button.setObjectName("aiPreviewButton")
        self.preview_button.clicked.connect(self.preview_requested.emit)
        layout.addWidget(self.preview_button)

        decision_row = QHBoxLayout()
        decision_row.setContentsMargins(0, 0, 0, 0)
        decision_row.setSpacing(4)
        self.accept_button = QPushButton("Accept")
        self.accept_button.setObjectName("aiAcceptButton")
        self.reject_button = QPushButton("Reject")
        self.reject_button.setObjectName("aiRejectButton")
        self.refine_button = QPushButton("Refine")
        self.refine_button.setObjectName("aiRefineButton")
        self.visual_refine_button = QPushButton("Visual refine")
        self.visual_refine_button.setObjectName("aiVisualRefineButton")
        self.visual_refine_button.setToolTip(
            "Inspect staged frames and apply one semantic motion plan"
        )
        self.visual_verify_button = QPushButton("Verify visually")
        self.visual_verify_button.setObjectName("aiVisualVerifyButton")
        self.visual_verify_button.setToolTip(
            "Compare original and staged frames without changing the motion"
        )
        self.accept_button.clicked.connect(self.accept_requested.emit)
        self.reject_button.clicked.connect(self.reject_requested.emit)
        self.refine_button.clicked.connect(self._emit_refine)
        self.visual_refine_button.clicked.connect(self._emit_visual_refine)
        self.visual_verify_button.clicked.connect(self._emit_visual_verify)
        for button in (self.accept_button, self.reject_button, self.refine_button):
            decision_row.addWidget(button)
        layout.addLayout(decision_row)
        visual_row = QHBoxLayout()
        visual_row.setContentsMargins(0, 0, 0, 0)
        visual_row.setSpacing(4)
        visual_row.addWidget(self.visual_refine_button)
        visual_row.addWidget(self.visual_verify_button)
        layout.addLayout(visual_row)

        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setObjectName("aiPromptInput")
        self.prompt_input.setPlaceholderText("Describe a motion or edit…")
        self.prompt_input.setMaximumHeight(84)
        layout.addWidget(self.prompt_input)

        prompt_actions = QHBoxLayout()
        prompt_actions.setContentsMargins(0, 0, 0, 0)
        self.submit_button = QPushButton("Apply")
        self.submit_button.setObjectName("aiSubmitButton")
        self.submit_button.clicked.connect(self._emit_submit)
        self.cancel_button = QPushButton("Cancel request")
        self.cancel_button.setObjectName("aiCancelButton")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.critique_button = QPushButton("Critique")
        self.critique_button.setObjectName("aiCritiqueButton")
        self.critique_button.setToolTip(
            "Inspect timestamped rendered frames without changing the motion"
        )
        self.critique_button.clicked.connect(self._emit_critique)
        prompt_actions.addWidget(self.submit_button, stretch=1)
        prompt_actions.addWidget(self.critique_button)
        prompt_actions.addWidget(self.cancel_button)
        layout.addLayout(prompt_actions)

        self.set_state(AIAssistantPanelState.READY)

    @property
    def state(self) -> AIAssistantPanelState:
        return self._state

    def set_provider(self, provider_name: str, model: str = "") -> None:
        label = provider_name.strip().title() or "Not configured"
        self.provider_label.setText(
            f"Provider: {label}" + (f" · {model.strip()}" if model.strip() else "")
        )
        self.provider_label.setToolTip(model.strip())

    def set_state(self, state: AIAssistantPanelState) -> None:
        self._state = AIAssistantPanelState(state)
        running = self._state is AIAssistantPanelState.RUNNING
        staged = self._state is AIAssistantPanelState.STAGED
        self.settings_button.setEnabled(not running)
        self.prompt_input.setEnabled(not running)
        self.submit_button.setVisible(not running)
        self.submit_button.setEnabled(not running)
        self.submit_button.setText("Refine" if staged else "Apply")
        self.critique_button.setVisible(not running)
        self.critique_button.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.preview_button.setEnabled(staged)
        self.accept_button.setEnabled(staged)
        self.reject_button.setEnabled(staged)
        self.refine_button.setEnabled(staged)
        self.visual_refine_button.setEnabled(staged)
        self.visual_verify_button.setEnabled(staged)

    def begin_request(
        self,
        *,
        refinement: bool = False,
        critique: bool = False,
        visual_refinement: bool = False,
        visual_verification: bool = False,
    ) -> None:
        self.set_state(AIAssistantPanelState.RUNNING)
        if visual_verification:
            self.response_label.setText("Comparing original and staged motion…")
        elif visual_refinement:
            self.response_label.setText("Inspecting and refining staged motion…")
        elif critique:
            self.response_label.setText("Inspecting timestamped rendered frames…")
        elif refinement:
            self.response_label.setText("Refining the staged working copy…")
        else:
            self.response_label.setText("Creating a detached AI working copy…")

    def show_proposal(self, response: str, changes: tuple[str, ...]) -> None:
        self.response_label.setText(response.strip() or "AI edit staged for review.")
        self.proposal_heading.setText("Proposed changes")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(changes) or ["Motion working copy updated"])
        self.proposal_heading.show()
        self.proposal_list.show()
        self.prompt_input.clear()
        self.set_state(AIAssistantPanelState.STAGED)

    def show_critique(
        self,
        summary: str,
        observations: tuple[str, ...],
        *,
        session_staged: bool = False,
    ) -> None:
        self.response_label.setText(summary.strip() or "Visual critique complete.")
        self.proposal_heading.setText("Visual observations")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(observations) or ["No visible issue reported"])
        self.proposal_heading.show()
        self.proposal_list.show()
        self.prompt_input.clear()
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.READY
        )

    def show_verification(
        self,
        summary: str,
        observations: tuple[str, ...],
    ) -> None:
        self.response_label.setText(summary.strip() or "Visual verification complete.")
        self.proposal_heading.setText("Visual verification")
        self.proposal_list.clear()
        self.proposal_list.addItems(list(observations) or ["No visible issue reported"])
        self.proposal_heading.show()
        self.proposal_list.show()
        self.prompt_input.clear()
        self.set_state(AIAssistantPanelState.STAGED)

    def show_error(self, message: str, *, session_staged: bool = False) -> None:
        self.response_label.setText(f"AI error: {message}")
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.ERROR
        )

    def show_cancelled(self, *, session_staged: bool = False) -> None:
        self.response_label.setText("AI request cancelled; committed motion is unchanged.")
        self.set_state(
            AIAssistantPanelState.STAGED
            if session_staged
            else AIAssistantPanelState.READY
        )

    def reset_session(self, message: str) -> None:
        self.response_label.setText(message)
        self.proposal_list.clear()
        self.proposal_heading.hide()
        self.proposal_list.hide()
        self.prompt_input.clear()
        self.set_state(AIAssistantPanelState.READY)

    def _instruction(self) -> str:
        return self.prompt_input.toPlainText().strip()

    def _emit_submit(self) -> None:
        instruction = self._instruction()
        if not instruction:
            self.response_label.setText("Describe the motion edit before applying it.")
            self.prompt_input.setFocus()
            return
        if self._state is AIAssistantPanelState.STAGED:
            self.refine_requested.emit(instruction)
        else:
            self.submit_requested.emit(instruction)

    def _emit_refine(self) -> None:
        instruction = self._instruction()
        if not instruction:
            self.response_label.setText(
                "Describe how the staged result should be refined."
            )
            self.prompt_input.setFocus()
            return
        self.refine_requested.emit(instruction)

    def _emit_critique(self) -> None:
        instruction = self._instruction() or "What is visually wrong with this motion?"
        self.critique_requested.emit(instruction)

    def _emit_visual_refine(self) -> None:
        self.visual_refine_requested.emit(self._instruction())

    def _emit_visual_verify(self) -> None:
        self.visual_verify_requested.emit(self._instruction())

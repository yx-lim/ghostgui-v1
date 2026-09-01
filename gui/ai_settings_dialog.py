"""Provider/model and secure BYOK settings for the Motion Assistant."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from application.ai.schemas import ProviderCapabilities


PROVIDER_MODELS = {
    "gemini": ("gemini-3.7-flash", "gemini-3.6-flash"),
    "anthropic": (
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ),
}


@dataclass(frozen=True)
class AISettingsValues:
    provider: str
    model: str
    api_key: str
    store_securely: bool


class AISettingsDialog(QDialog):
    """Collect non-secret provider settings and an optional in-memory key."""

    test_connection_requested = Signal(str, str, str)
    clear_stored_key_requested = Signal(str)

    def __init__(
        self,
        *,
        provider="gemini",
        model="gemini-3.7-flash",
        capabilities: ProviderCapabilities,
        secure_key_available=False,
        provider_capabilities=None,
        secure_key_availability=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("aiSettingsDialog")
        self.setWindowTitle("AI Assistant Settings")
        self.setModal(True)
        self.setMinimumWidth(430)
        self._provider_capabilities = dict(provider_capabilities or {})
        self._provider_capabilities.setdefault(provider, capabilities)
        self._secure_key_availability = dict(secure_key_availability or {})
        self._secure_key_availability.setdefault(
            provider,
            bool(secure_key_available),
        )

        layout = QVBoxLayout(self)
        description = QLabel(
            "GhostGUI uses your provider key only for AI requests. Keys are never "
            "saved in projects or plain preference files."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.provider_box = QComboBox()
        self.provider_box.setObjectName("aiProviderBox")
        self.provider_box.addItem("Gemini", "gemini")
        self.provider_box.addItem("Anthropic", "anthropic")
        provider_index = self.provider_box.findData(provider)
        self.provider_box.setCurrentIndex(max(0, provider_index))
        form.addRow("Provider", self.provider_box)

        self.model_box = QComboBox()
        self.model_box.setObjectName("aiModelBox")
        self.model_box.setEditable(True)
        self.model_box.addItems(PROVIDER_MODELS.get(provider, ()))
        self.model_box.setCurrentText(model)
        form.addRow("Model", self.model_box)

        self.api_key_input = QLineEdit()
        self.api_key_input.setObjectName("aiApiKeyInput")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText(
            "Stored securely" if secure_key_available else "Enter API key"
        )
        form.addRow("API key", self.api_key_input)
        layout.addLayout(form)

        self.store_securely_checkbox = QCheckBox("Store securely in the system keyring")
        self.store_securely_checkbox.setObjectName("aiStoreSecurelyCheckbox")
        self.store_securely_checkbox.setChecked(True)
        layout.addWidget(self.store_securely_checkbox)

        self.clear_key_button = QPushButton("Remove stored API key")
        self.clear_key_button.setObjectName("aiClearStoredKeyButton")
        self.clear_key_button.setEnabled(bool(secure_key_available))
        self.clear_key_button.clicked.connect(
            lambda: self.clear_stored_key_requested.emit(
                str(self.provider_box.currentData())
            )
        )
        layout.addWidget(self.clear_key_button)

        self.capabilities_label = QLabel()
        self.capabilities_label.setObjectName("aiCapabilitiesLabel")
        layout.addWidget(self.capabilities_label)

        self.test_status_label = QLabel("Not tested")
        self.test_status_label.setObjectName("aiConnectionStatusLabel")
        self.test_status_label.setWordWrap(True)
        layout.addWidget(self.test_status_label)

        self.test_button = QPushButton("Test Connection")
        self.test_button.setObjectName("aiTestConnectionButton")
        self.test_button.clicked.connect(self._request_test)
        layout.addWidget(self.test_button)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.provider_box.currentIndexChanged.connect(self._provider_changed)
        self._update_provider_details(reset_model=False)

    def values(self) -> AISettingsValues:
        return AISettingsValues(
            provider=str(self.provider_box.currentData()),
            model=self.model_box.currentText().strip(),
            api_key=self.api_key_input.text(),
            store_securely=self.store_securely_checkbox.isChecked(),
        )

    def set_test_running(self) -> None:
        self.test_button.setEnabled(False)
        self.test_status_label.setText("Testing connection…")

    def set_test_result(self, succeeded: bool, message: str) -> None:
        self.test_button.setEnabled(True)
        self.test_status_label.setText(
            ("Connection succeeded: " if succeeded else "Connection failed: ")
            + message
        )

    def mark_stored_key_removed(self, removed: bool) -> None:
        provider = str(self.provider_box.currentData())
        self._secure_key_availability[provider] = False
        self.clear_key_button.setEnabled(False)
        self.api_key_input.clear()
        self.api_key_input.setPlaceholderText("Enter API key")
        self.test_status_label.setText(
            "Stored API key removed."
            if removed
            else "No stored API key was found."
        )

    def _request_test(self) -> None:
        values = self.values()
        if not values.model:
            self.test_status_label.setText("Choose or enter a model first.")
            self.model_box.setFocus()
            return
        self.set_test_running()
        self.test_connection_requested.emit(
            values.provider,
            values.model,
            values.api_key,
        )

    def _provider_changed(self, _index=None) -> None:
        self._update_provider_details(reset_model=True)

    def _update_provider_details(self, *, reset_model: bool) -> None:
        provider = str(self.provider_box.currentData())
        if reset_model:
            self.model_box.clear()
            self.model_box.addItems(PROVIDER_MODELS.get(provider, ()))
            self.api_key_input.clear()
            self.test_status_label.setText("Not tested")
        secure = bool(self._secure_key_availability.get(provider, False))
        self.api_key_input.setPlaceholderText(
            "Stored securely" if secure else "Enter API key"
        )
        self.clear_key_button.setEnabled(secure)
        capabilities = self._provider_capabilities.get(provider)
        if capabilities is None:
            self.capabilities_label.setText("Capabilities unavailable")
            return
        self.capabilities_label.setText(
            "Capabilities\n"
            f"{'✓' if capabilities.supports_tools else '—'} Tool Calling\n"
            f"{'✓' if capabilities.supports_vision else '—'} Vision\n"
            f"{'✓' if capabilities.supports_structured_output else '—'} "
            "Structured Output"
        )

    def _validate_and_accept(self) -> None:
        if not self.model_box.currentText().strip():
            self.test_status_label.setText("Model must not be empty.")
            self.model_box.setFocus()
            return
        self.accept()

"""Qt orchestration for the Motion Assistant panel and application AI services."""

from __future__ import annotations

import asyncio
from collections import Counter

from application.ai import (
    AIEditSession,
    AIEditSessionState,
    AgentRunResult,
    EditorSelectionContext,
    GhostGUIAgent,
    GhostGUIMotionService,
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    SemanticToolContext,
    TimestampMotionIdentityResolver,
    build_semantic_tool_registry,
    sample_working_preview_qpos,
)
from application.ai.credentials import (
    CredentialStorageError,
    SystemKeyringCredentialStore,
)
from application.ai.errors import ProviderCancelledError
from application.ai.providers.gemini import (
    DEFAULT_GEMINI_CAPABILITIES,
    GeminiProvider,
)
from application.ai.schemas import MessageRole, ProviderMessage, ProviderRequest
from gui.ai_settings_dialog import AISettingsDialog
from gui.panels.ai_assistant_panel import AIAssistantPanelState


DEFAULT_AI_PROVIDER = "gemini"
DEFAULT_AI_MODEL = "gemini-3.7-flash"


class AIAssistantController:
    """Bind panel intents to a detached session without moving state into Qt."""

    def __init__(self, host, panel, settings, background_jobs):
        self.host = host
        self.panel = panel
        self.settings = settings
        self.background_jobs = background_jobs
        self.credential_store = SystemKeyringCredentialStore()
        self.metadata_store = InMemoryMotionMetadataStore()
        self.identity_resolver = TimestampMotionIdentityResolver()
        self.session = None
        self.active_handle = None
        self._session_api_key = None
        self._settings_dialog = None

        self.provider_name = str(
            settings.value("ai/provider", DEFAULT_AI_PROVIDER)
        )
        self.model = str(settings.value("ai/model", DEFAULT_AI_MODEL))
        self.panel.set_provider(self.provider_name, self.model)
        self.panel.submit_requested.connect(self.start_edit)
        self.panel.refine_requested.connect(self.refine)
        self.panel.preview_requested.connect(self.preview)
        self.panel.accept_requested.connect(self.accept)
        self.panel.reject_requested.connect(self.reject)
        self.panel.cancel_requested.connect(self.cancel_request)
        self.panel.settings_requested.connect(self.open_settings)

    @property
    def session_staged(self) -> bool:
        return (
            self.session is not None
            and self.session.state is AIEditSessionState.STAGED
        )

    def start_edit(self, instruction: str) -> None:
        if self.active_handle is not None:
            return
        if self.session is None or self.session.state in {
            AIEditSessionState.ACCEPTED,
            AIEditSessionState.REJECTED,
        }:
            self.host._refresh_history_baseline()
            self.session = AIEditSession(
                self.host.document,
                metadata_store=self.metadata_store,
            )
        self._start_request(instruction, refinement=False)

    def refine(self, instruction: str) -> None:
        if not self.session_staged or self.active_handle is not None:
            return
        self._start_request(instruction, refinement=True)

    def _start_request(self, instruction: str, *, refinement: bool) -> None:
        if self.host.robot_model_3d is None:
            self.panel.show_error("No robot model is available for motion editing.")
            return
        try:
            motion = GhostGUIMotionService(self.host.robot_model_3d)
            tools = build_semantic_tool_registry(motion)
            metadata = MotionMetadataService(
                self.metadata_store,
                self.identity_resolver,
            )
            context = SemanticToolContext(
                session=self.session,
                metadata=metadata,
                selection=EditorSelectionContext(
                    logical_frame=self.host.controls.frame_box.currentText(),
                ),
                motion_name=(
                    None
                    if self.host.current_project is None
                    else self.host.current_project.project_name
                ),
            )
        except Exception as error:
            self.panel.show_error(str(error), session_staged=self.session_staged)
            return

        self.panel.begin_request(refinement=refinement)
        self.host.set_ai_motion_controls_enabled(False)

        def work(token):
            return asyncio.run(
                self._run_agent(instruction, tools, context, token)
            )

        self.active_handle = self.background_jobs.submit_cancellable(
            "AI motion edit",
            work,
            self._request_succeeded,
            self._request_failed,
            self._request_cancelled,
        )
        if self.active_handle is None:
            self.panel.show_error(
                "The background worker is shutting down.",
                session_staged=self.session_staged,
            )
            if not self.session_staged:
                self.host.set_ai_motion_controls_enabled(True)

    async def _run_agent(self, instruction, tools, context, token):
        provider = self._provider()
        try:
            return await GhostGUIAgent(provider, tools).run(
                instruction,
                model=self.model,
                context=context,
                cancellation_token=token,
            )
        finally:
            await provider.aclose()

    def _provider(self, *, api_key=None):
        if self.provider_name != "gemini":
            raise ValueError(f"Unsupported AI provider: {self.provider_name}")
        return GeminiProvider(api_key=api_key or self._session_api_key)

    def _request_succeeded(self, result: AgentRunResult) -> None:
        self.active_handle = None
        changes = self._proposal_lines(result)
        if self.session_staged:
            self.panel.show_proposal(result.text, changes)
            self.preview()
        else:
            self.panel.reset_session(
                result.text.strip() or "The assistant made no motion changes."
            )
            self.session = None
            self.host.set_ai_motion_controls_enabled(True)

    def _request_failed(self, error: Exception) -> None:
        self.active_handle = None
        if isinstance(error, ProviderCancelledError):
            self._request_cancelled()
            return
        self.panel.show_error(str(error), session_staged=self.session_staged)
        if not self.session_staged:
            self.host.set_ai_motion_controls_enabled(True)

    def _request_cancelled(self) -> None:
        self.active_handle = None
        self.panel.show_cancelled(session_staged=self.session_staged)
        if not self.session_staged:
            self.host.set_ai_motion_controls_enabled(True)

    def cancel_request(self) -> None:
        if self.active_handle is not None:
            self.active_handle.cancel()

    def preview(self) -> None:
        if not self.session_staged:
            return
        viewer = self.host.viewer_3d
        if viewer.preview_state is None:
            self.panel.show_error(
                "The staged motion has no Orange preview state.",
                session_staged=True,
            )
            return
        try:
            qpos = sample_working_preview_qpos(
                self.session,
                viewer.get_current_time(),
            )
            viewer.preview_state.set_qpos(qpos)
            viewer.preview_active = True
            viewer._use_editor_canvas_states()
            viewer.canvas.set_preview_visible(True)
            viewer._update_preview_collisions()
            viewer.canvas.update()
            viewer.status_label.setText(
                "Orange preview shows the staged AI working copy; committed "
                "motion is unchanged."
            )
        except Exception as error:
            self.panel.show_error(str(error), session_staged=True)

    def accept(self) -> None:
        if not self.session_staged or self.active_handle is not None:
            return
        try:
            self.session.accept(self.host.editor_controller)
            self.metadata_store.replace(self.session.metadata.snapshot())
            self.host.viewer_3d.cancel_preview()
            self.host.viewer_3d.clear_robot_trajectory()
            self.host.viewer_3d_mujoco.clear_trajectory()
            self.host.backend_interface.clear_last_solution()
            self.host.set_editor_timeline_duration(
                self.host.document.timeline_duration
            )
            self.host.viewer_3d.set_current_time(
                self.host.document.current_time
            )
            self.host.refresh_display()
            self.host.record_history_action("Accept AI motion edit")
        except Exception as error:
            self.panel.show_error(str(error), session_staged=True)
            return
        self.panel.reset_session("AI motion edit accepted as one history entry.")
        self.session = None
        self.host.set_ai_motion_controls_enabled(True)

    def reject(self) -> None:
        if self.session is None or self.active_handle is not None:
            return
        try:
            self.session.reject()
        except Exception as error:
            self.panel.show_error(str(error), session_staged=self.session_staged)
            return
        self.host.viewer_3d.cancel_preview()
        self.panel.reset_session("AI working copy rejected; committed motion is unchanged.")
        self.session = None
        self.host.set_ai_motion_controls_enabled(True)

    def open_settings(self) -> None:
        secure_key_available = bool(
            self.credential_store.get_secret(self.provider_name)
        )
        dialog = AISettingsDialog(
            provider=self.provider_name,
            model=self.model,
            capabilities=DEFAULT_GEMINI_CAPABILITIES,
            secure_key_available=secure_key_available,
            parent=self.host,
        )
        self._settings_dialog = dialog
        dialog.test_connection_requested.connect(self._test_connection)
        dialog.clear_stored_key_requested.connect(self._clear_stored_key)
        accepted = bool(dialog.exec())
        if accepted:
            values = dialog.values()
            try:
                if values.api_key and values.store_securely:
                    self.credential_store.set_secret(
                        values.provider,
                        values.api_key,
                    )
                    self._session_api_key = None
                elif values.api_key:
                    self._session_api_key = values.api_key
            except CredentialStorageError as error:
                self.panel.show_error(str(error), session_staged=self.session_staged)
            self.provider_name = values.provider
            self.model = values.model
            self.settings.setValue("ai/provider", self.provider_name)
            self.settings.setValue("ai/model", self.model)
            self.settings.sync()
            self.panel.set_provider(self.provider_name, self.model)
        self._settings_dialog = None

    def _test_connection(self, provider_name: str, model: str, api_key: str) -> None:
        dialog = self._settings_dialog
        if dialog is None:
            return

        def work(token):
            if token.cancellation_requested:
                raise ProviderCancelledError("Connection test cancelled")
            return asyncio.run(
                self._run_connection_test(provider_name, model, api_key)
            )

        handle = self.background_jobs.submit_cancellable(
            "test AI connection",
            work,
            lambda response: self._connection_test_finished(True, response),
            lambda error: self._connection_test_finished(False, str(error)),
            lambda: self._connection_test_finished(False, "cancelled"),
        )
        if handle is None:
            dialog.set_test_result(False, "The background worker is shutting down.")

    async def _run_connection_test(self, provider_name, model, api_key):
        if provider_name != "gemini":
            raise ValueError(f"Unsupported AI provider: {provider_name}")
        provider = GeminiProvider(api_key=api_key or self._session_api_key)
        try:
            response = await provider.generate(
                ProviderRequest(
                    model=model,
                    messages=(
                        ProviderMessage(
                            MessageRole.USER,
                            text="Reply with the single word OK.",
                        ),
                    ),
                    max_output_tokens=8,
                )
            )
            return response.text.strip() or "Provider returned an empty acknowledgement"
        finally:
            await provider.aclose()

    def _connection_test_finished(self, succeeded: bool, message: str) -> None:
        if self._settings_dialog is not None:
            self._settings_dialog.set_test_result(succeeded, message)

    def _clear_stored_key(self, provider_name: str) -> None:
        if self._settings_dialog is None:
            return
        try:
            removed = self.credential_store.delete_secret(provider_name)
        except CredentialStorageError as error:
            self._settings_dialog.set_test_result(False, str(error))
            return
        self._settings_dialog.mark_stored_key_removed(removed)

    def shutdown(self) -> None:
        self.cancel_request()

    @staticmethod
    def _proposal_lines(result: AgentRunResult) -> tuple[str, ...]:
        successful = Counter(
            record.name.replace("_", " ")
            for record in result.tool_executions
            if record.succeeded and record.name != "inspect_motion"
        )
        lines = tuple(
            f"{name.title()}" + (f" ×{count}" if count > 1 else "")
            for name, count in successful.items()
        )
        if result.validation is not None:
            lines += ("Validated staged motion",)
        return lines or ("No semantic motion change was reported",)

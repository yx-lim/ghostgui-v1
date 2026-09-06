"""Qt orchestration for the Motion Assistant panel and application AI services."""

from __future__ import annotations

import asyncio

from application.ai import (
    AIEditSession,
    AIEditSessionState,
    ContextBuilder,
    EditorSelectionContext,
    FrameSampler,
    FrameSamplingRequest,
    GhostGUIMotionService,
    InMemoryMotionMetadataStore,
    MotionMetadataService,
    RobotCapabilityContext,
    SemanticToolContext,
    TimestampMotionIdentityResolver,
    TextMotionRunResult,
    TextMotionWorkflow,
    VisualCritic,
    VisualCritiqueResult,
    VisualRefinementAction,
    VisualRefinementLimits,
    VisualRefinementProgress,
    VisualRefinementStep,
    VisualRefinementStepResult,
    build_semantic_tool_registry,
    capture_comparison_frames,
    capture_motion_frames,
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
from application.ai.providers.anthropic import (
    DEFAULT_ANTHROPIC_CAPABILITIES,
    DEFAULT_CLAUDE_MODEL,
    AnthropicProvider,
)
from application.ai.schemas import (
    ImageVariant,
    MessageRole,
    ProviderMessage,
    ProviderRequest,
)
from gui.ai_frame_capture import RobotViewerFrameRenderer
from gui.ai_settings_dialog import AISettingsDialog


DEFAULT_AI_PROVIDER = "gemini"
DEFAULT_AI_MODEL = "gemini-3.7-flash"
DEFAULT_PROVIDER_MODELS = {
    "gemini": DEFAULT_AI_MODEL,
    "anthropic": DEFAULT_CLAUDE_MODEL,
}


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
        self._session_api_keys = {}
        self._settings_dialog = None
        self._session_goal = ""
        self._visual_refinement_progress = None
        self._visual_refinement_plan = None
        self._visual_refinement_goal = ""

        self.provider_name = str(
            settings.value("ai/provider", DEFAULT_AI_PROVIDER)
        )
        self.model = str(settings.value(
            "ai/model",
            DEFAULT_PROVIDER_MODELS.get(self.provider_name, DEFAULT_AI_MODEL),
        ))
        self.panel.set_provider(self.provider_name, self.model)
        self.panel.submit_requested.connect(self.start_edit)
        self.panel.critique_requested.connect(self.start_critique)
        self.panel.visual_refine_requested.connect(self.start_visual_refinement)
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
        if not self.session_staged:
            self._session_goal = instruction.strip()
        self._start_request(instruction, refinement=False)

    def refine(self, instruction: str) -> None:
        if not self.session_staged or self.active_handle is not None:
            return
        self._session_goal = self._combined_goal(instruction)
        self._start_request(instruction, refinement=True)

    def start_visual_refinement(self, instruction: str = "") -> None:
        if not self.session_staged or self.active_handle is not None:
            return
        if self.host.robot_model_3d is None:
            self.panel.show_error(
                "No robot model is available for visual refinement.",
                session_staged=True,
            )
            return
        self._visual_refinement_goal = self._combined_goal(instruction)
        self._visual_refinement_progress = VisualRefinementProgress(
            VisualRefinementLimits(max_edit_iterations=2)
        )
        try:
            self._visual_refinement_plan = FrameSampler().plan(
                self.session.working_document,
                FrameSamplingRequest(
                    suspected_times=(float(self.host.viewer_3d.display_time),),
                ),
            )
        except Exception as error:
            self._clear_visual_refinement()
            self.panel.show_error(str(error), session_staged=True)
            return
        self.panel.begin_request(visual_refinement=True)
        self.host.set_ai_motion_controls_enabled(False)
        self._submit_visual_refinement_step(allow_edit=True)

    def _submit_visual_refinement_step(self, *, allow_edit: bool) -> None:
        try:
            renderer = RobotViewerFrameRenderer(self.host.viewer_3d)
            frames = capture_comparison_frames(
                self.host.document,
                self.session.working_document,
                self._visual_refinement_plan,
                renderer,
            )
            motion_context = self._critique_context(self.session.working_document)
            motion = GhostGUIMotionService(self.host.robot_model_3d)
            tools = build_semantic_tool_registry(motion)
            metadata = MotionMetadataService(
                self.metadata_store,
                self.identity_resolver,
            )
            semantic_context = SemanticToolContext(
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
            self._clear_visual_refinement()
            self.panel.show_error(str(error), session_staged=True)
            return

        def work(token):
            return asyncio.run(self._run_visual_refinement_step(
                self._visual_refinement_goal,
                motion_context,
                frames,
                tools,
                semantic_context,
                allow_edit,
                token,
            ))

        self.active_handle = self.background_jobs.submit_cancellable(
            "AI visual refinement",
            work,
            self._visual_refinement_step_succeeded,
            self._request_failed,
            self._request_cancelled,
        )
        if self.active_handle is None:
            self._clear_visual_refinement()
            self.panel.show_error(
                "The background worker is shutting down.",
                session_staged=True,
            )

    async def _run_visual_refinement_step(
        self,
        goal,
        motion_context,
        frames,
        tools,
        semantic_context,
        allow_edit,
        token,
    ):
        provider = self._provider()
        try:
            return await VisualRefinementStep(
                provider,
                tools,
                limits=self._visual_refinement_progress.limits,
            ).run(
                goal,
                model=self.model,
                motion_context=motion_context,
                comparison_frames=frames,
                semantic_context=semantic_context,
                allow_edit=allow_edit,
                cancellation_token=token,
            )
        finally:
            await provider.aclose()

    def _visual_refinement_step_succeeded(
        self,
        result: VisualRefinementStepResult,
    ) -> None:
        self.active_handle = None
        try:
            action = self._visual_refinement_progress.after_step(result)
            self._update_visual_refinement_plan(result)
        except Exception as error:
            self._request_failed(error)
            return
        if action is VisualRefinementAction.REFINE:
            self._submit_visual_refinement_step(allow_edit=True)
            return
        if action is VisualRefinementAction.ASSESS_ONLY:
            self._submit_visual_refinement_step(allow_edit=False)
            return
        self._finish_visual_refinement(result)

    def _update_visual_refinement_plan(self, result) -> None:
        suspected = tuple(
            observation.time_seconds
            for observation in result.comparison_result.comparison.observations
            if observation.time_seconds is not None
        )
        if not suspected:
            return
        self._visual_refinement_plan = FrameSampler().plan(
            self.session.working_document,
            FrameSamplingRequest(suspected_times=suspected),
        )

    def _finish_visual_refinement(self, result) -> None:
        comparison = result.comparison_result.comparison
        edit_count = self._visual_refinement_progress.completed_edit_iterations
        lines = tuple(comparison.reasons)
        lines += tuple(
            self._format_observation(value)
            for value in comparison.observations
        )
        if edit_count:
            lines += (f"Visual semantic refinement iterations: {edit_count}",)
        if comparison.should_refine and (
            edit_count >= self._visual_refinement_progress.limits.max_edit_iterations
        ):
            lines += ("Automatic refinement limit reached; review remaining issues",)
        self.panel.show_proposal(
            comparison.summary,
            lines or ("No additional visual refinement was needed",),
        )
        self._session_goal = self._visual_refinement_goal
        self._clear_visual_refinement()
        self.preview()

    def _clear_visual_refinement(self) -> None:
        self._visual_refinement_progress = None
        self._visual_refinement_plan = None
        self._visual_refinement_goal = ""

    def _combined_goal(self, instruction: str) -> str:
        addition = instruction.strip()
        if self._session_goal and addition:
            return f"{self._session_goal}\nAdditional user direction: {addition}"
        return self._session_goal or addition or "Improve the remaining visual motion issues."

    def start_critique(self, instruction: str) -> None:
        """Inspect committed or staged motion without opening an edit session."""

        if self.active_handle is not None:
            return
        if self.host.robot_model_3d is None:
            self.panel.show_error("No robot model is available for visual critique.")
            return
        document = (
            self.session.working_document
            if self.session_staged
            else self.host.document
        )
        viewer = self.host.viewer_3d
        try:
            plan = FrameSampler().plan(
                document,
                FrameSamplingRequest(
                    suspected_times=(float(viewer.display_time),),
                ),
            )
            frames = capture_motion_frames(
                document,
                plan,
                RobotViewerFrameRenderer(viewer),
                variant=(
                    ImageVariant.CANDIDATE
                    if self.session_staged
                    else ImageVariant.ORIGINAL
                ),
            )
            context = self._critique_context(document)
        except Exception as error:
            self.panel.show_error(str(error), session_staged=self.session_staged)
            return

        self.panel.begin_request(critique=True)
        self.host.set_ai_motion_controls_enabled(False)

        def work(token):
            return asyncio.run(
                self._run_critique(instruction, context, frames, token)
            )

        self.active_handle = self.background_jobs.submit_cancellable(
            "AI visual critique",
            work,
            self._critique_succeeded,
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

    def _critique_context(self, document):
        motion = GhostGUIMotionService(self.host.robot_model_3d)
        capabilities = RobotCapabilityContext(
            logical_frames=tuple(motion.logical_frames),
            end_effectors=tuple(motion.end_effectors),
            joints=tuple(motion.joint_names),
            joint_groups=tuple(
                (name, tuple(values))
                for name, values in motion.joint_groups.items()
            ),
        )
        selection = EditorSelectionContext(
            logical_frame=self.host.controls.frame_box.currentText(),
        )
        metadata = MotionMetadataService(
            self.metadata_store,
            self.identity_resolver,
        )
        motion_name = (
            None
            if self.host.current_project is None
            else self.host.current_project.project_name
        )
        builder = ContextBuilder()
        if self.session_staged:
            return builder.build_for_session(
                self.session,
                selection=selection,
                robot_capabilities=capabilities,
                metadata=metadata,
                motion_name=motion_name,
            ).to_dict()
        return builder.build(
            document,
            selection=selection,
            robot_capabilities=capabilities,
            metadata=metadata,
            motion_name=motion_name,
        ).to_dict()

    async def _run_critique(self, instruction, context, frames, token):
        provider = self._provider()
        try:
            return await VisualCritic(provider).run(
                instruction,
                model=self.model,
                motion_context=context,
                motion_frames=frames,
                cancellation_token=token,
            )
        finally:
            await provider.aclose()

    def _critique_succeeded(self, result: VisualCritiqueResult) -> None:
        self.active_handle = None
        observations = tuple(
            self._format_observation(observation)
            for observation in result.critique.observations
        )
        self.panel.show_critique(
            result.critique.summary,
            observations,
            session_staged=self.session_staged,
        )
        if not self.session_staged:
            self.host.set_ai_motion_controls_enabled(True)

    @staticmethod
    def _format_observation(observation) -> str:
        prefix = (
            ""
            if observation.time_seconds is None
            else f"Around {observation.time_seconds:.2f} s: "
        )
        body = "" if observation.body_part is None else f"{observation.body_part}: "
        return f"{prefix}{body}{observation.issue}"

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
                self._run_text_motion(instruction, tools, context, token)
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

    async def _run_text_motion(self, instruction, tools, context, token):
        provider = self._provider()
        try:
            return await TextMotionWorkflow(provider, tools).run(
                instruction,
                model=self.model,
                context=context,
                cancellation_token=token,
            )
        finally:
            await provider.aclose()

    def _provider(self, *, api_key=None):
        key = api_key or self._session_api_keys.get(self.provider_name)
        if self.provider_name == "gemini":
            return GeminiProvider(api_key=key)
        if self.provider_name == "anthropic":
            return AnthropicProvider(api_key=key)
        raise ValueError(f"Unsupported AI provider: {self.provider_name}")

    def _request_succeeded(self, result: TextMotionRunResult) -> None:
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
            self._session_goal = ""
            self.host.set_ai_motion_controls_enabled(True)

    def _request_failed(self, error: Exception) -> None:
        self.active_handle = None
        if isinstance(error, ProviderCancelledError):
            self._request_cancelled()
            return
        self.panel.show_error(str(error), session_staged=self.session_staged)
        self._clear_visual_refinement()
        if not self.session_staged:
            self.host.set_ai_motion_controls_enabled(True)

    def _request_cancelled(self) -> None:
        self.active_handle = None
        self.panel.show_cancelled(session_staged=self.session_staged)
        self._clear_visual_refinement()
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
        self._session_goal = ""
        self._clear_visual_refinement()
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
        self._session_goal = ""
        self._clear_visual_refinement()
        self.host.set_ai_motion_controls_enabled(True)

    def open_settings(self) -> None:
        secure_key_availability = {
            provider: bool(self.credential_store.get_secret(provider))
            for provider in DEFAULT_PROVIDER_MODELS
        }
        dialog = AISettingsDialog(
            provider=self.provider_name,
            model=self.model,
            capabilities=self._provider_capabilities(self.provider_name),
            secure_key_available=secure_key_availability[self.provider_name],
            provider_capabilities={
                "gemini": DEFAULT_GEMINI_CAPABILITIES,
                "anthropic": DEFAULT_ANTHROPIC_CAPABILITIES,
            },
            secure_key_availability=secure_key_availability,
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
                    self._session_api_keys.pop(values.provider, None)
                elif values.api_key:
                    self._session_api_keys[values.provider] = values.api_key
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
        key = api_key or self._session_api_keys.get(provider_name)
        if provider_name == "gemini":
            provider = GeminiProvider(api_key=key)
        elif provider_name == "anthropic":
            provider = AnthropicProvider(api_key=key)
        else:
            raise ValueError(f"Unsupported AI provider: {provider_name}")
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

    @staticmethod
    def _provider_capabilities(provider_name):
        if provider_name == "gemini":
            return DEFAULT_GEMINI_CAPABILITIES
        if provider_name == "anthropic":
            return DEFAULT_ANTHROPIC_CAPABILITIES
        raise ValueError(f"Unsupported AI provider: {provider_name}")

    def shutdown(self) -> None:
        self.cancel_request()

    @staticmethod
    def _proposal_lines(result: TextMotionRunResult) -> tuple[str, ...]:
        return result.proposal_lines

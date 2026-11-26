"""Opt-in GUI attachment. Tools own worker copies; Qt owns presentation/Accept."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
import base64
from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event, Lock
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
from application.agent_api.sandbox import safe_leaf
from application.agent_api.workspace import HeadlessWorkspace
from application.agent_bridge.server import BridgeServer
from application.agent_bridge.snapshots import FrozenMotion
from application.ai.edit_session import AIEditSession
from application.ai.metadata import InMemoryMotionMetadataStore, MotionMetadataService, TimestampMotionIdentityResolver
from application.ai.motion_services import GhostGUIMotionService
from application.ai.motion_state import ReplaceMotionState, capture_motion_state
from application.editor_controller import EditorController
from application.project_manager import GhostGUIProject, ghostgui_projects_dir, safe_project_path
from application.trajectory_export_service import motion_source_identity


class CancelToken:
    def __init__(self, event):
        self.event = event

    @property
    def cancellation_requested(self):
        return self.event.is_set()


class QtInvoker(QObject):
    requested = Signal(object)

    def __init__(self, parent=None, timeout=15):
        super().__init__(parent)
        self.timeout = timeout
        self.closed = False
        self._pending = set()
        self._lock = Lock()
        self.requested.connect(self._execute, Qt.ConnectionType.QueuedConnection)

    def call(self, callback, cancellation):
        future = Future()
        deadline = time.monotonic() + self.timeout
        with self._lock:
            if self.closed or cancellation.is_set():
                raise RuntimeError('GUI invocation cancelled')
            self._pending.add(future)
        self.requested.emit((callback, cancellation, deadline, future))
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            cancellation.set()
            future.cancel()
            raise RuntimeError('GUI invocation timed out; late callback ignored') from None
        finally:
            with self._lock:
                self._pending.discard(future)

    @Slot(object)
    def _execute(self, request):
        callback, cancellation, deadline, future = request
        if self.closed or cancellation.is_set() or time.monotonic() >= deadline:
            future.cancel()
            return
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = callback()
            if cancellation.is_set() or self.closed:
                raise RuntimeError('GUI invocation cancelled')
            future.set_result(result)
        except Exception as error:
            future.set_exception(error)

    def shutdown(self):
        with self._lock:
            self.closed = True
            for future in tuple(self._pending):
                future.cancel()


@dataclass(frozen=True)
class LiveWorkspace:
    motion: FrozenMotion
    adapter: object  # Compiled model is immutable; each consumer creates private MjData.
    models: tuple
    metadata: tuple
    project_path: str | None
    workspace_json: str
    project_dirty: bool


class GuiWorkspace(HeadlessWorkspace):
    """Same catalog on one worker; all live reads/transitions use Qt invocations."""
    def __init__(self, bridge, live):
        self.bridge = bridge
        self.call_cancel = Event()
        super().__init__(bridge.projects_dir, model=live.motion.document(live.adapter).model_key,
                         models=dict(live.models), approval=bridge.approval,
                         allow_user_override=bridge.allow_user_override)
        self._install_live(live)

    def gui(self, callback):
        return self.bridge.invoker.call(callback, self.call_cancel)

    def _install_live(self, live):
        old_identity = self.identity(self.committed) if hasattr(self, 'committed') else None
        self.committed = live.motion.document(live.adapter)
        self.adapter = live.adapter
        self.controller = EditorController(self.committed)
        self.motion = GhostGUIMotionService(self.adapter)
        self.models = dict(live.models)
        self.metadata_store = InMemoryMotionMetadataStore(dict(live.metadata))
        self.metadata = MotionMetadataService(self.metadata_store, TimestampMotionIdentityResolver())
        self.project = None if live.project_path is None else GhostGUIProject.open(live.project_path)
        self.workspace_payload = json.loads(live.workspace_json)
        if old_identity != self.identity(self.committed):
            self.generated = None
        self.live_identity = self.identity(self.committed)

    def sync_live(self):
        live = self.gui(self.bridge.capture_live)
        current = live.motion.document(live.adapter)
        identity = motion_source_identity(live.adapter, current.qpos_timeline, document=current)
        if self.candidate() is not None:
            if identity != self.live_identity:
                raise ValueError('committed GUI motion changed; Discard the stale candidate')
        else:
            self._install_live(live)
        return live

    def call_checkpoint(self):
        session = self.candidate()
        return (session, None if session is None else session.checkpoint(), self.generated,
                getattr(self, '_validation_identity', None), self.identity(), self.checkpoints.entries.copy())

    def restore_call(self, checkpoint):
        session, state, generated, validation, source, entries = checkpoint
        self._session = session
        if session is not None:
            session.restore_checkpoint(state)
        self.generated = generated
        if generated is not None and generated.source == source:
            self.generated = replace(generated, source=self.identity())
        self._validation_identity = self.identity() if validation == source else None
        self.checkpoints.entries = entries

    def status(self):
        return {**super().status(), 'workspace': 'gui', 'attach': 'required',
                'approval': self.approval, 'projects_dir': str(self.projects_dir)}

    def publish_candidate(self, snapshot):
        # The runtime may create a mutable local snapshot; never hand it to Qt.
        pass

    def _transition_allowed(self):
        if self.candidate() is not None:
            raise ValueError('Accept or Discard the unresolved candidate first')
        live = self.sync_live()
        if live.project_dirty or self.committed.dirty:
            raise ValueError('Save unsaved GUI edits before changing projects/models')

    def _load_gui_model(self, key):
        if key not in self.models:
            raise ValueError('unknown model')
        if key != self.committed.model_key:
            from core.models import MuJoCoRobotAdapter
            adapter = MuJoCoRobotAdapter(self.models[key])
            self.gui(lambda: self.bridge.load_model(key, adapter))
        return self.gui(self.bridge.capture_live)

    def load_model(self, model_key):
        self._transition_allowed()
        self._install_live(self._load_gui_model(model_key))
        return self.status()

    def new_project(self, name, model_key=None, duration_s=5.0):
        self._transition_allowed()
        path = safe_project_path(self.projects_dir, safe_leaf(name) + '.ghostgui')
        if path.exists():
            raise ValueError('project already exists; open it instead')
        if model_key is not None and model_key != self.committed.model_key:
            self._load_gui_model(model_key)
        self.gui(lambda: self.bridge.new_project(path, name, duration_s))
        self._install_live(self.gui(self.bridge.capture_live))
        return self.status()

    def open_project(self, name):
        self._transition_allowed()
        path = safe_project_path(self.projects_dir, safe_leaf(name) + '.ghostgui')
        # Fully preflight a detached load before the GUI can change documents.
        check = HeadlessWorkspace(self.projects_dir, models=self.models, model=self.committed.model_key)
        check.open_project(name)
        if check.project.autosave_exists():
            raise ValueError('Resolve the project autosave in the GUI before external opening')
        self._load_gui_model(check.committed.model_key)
        self.gui(lambda: self.bridge.open_project(path))
        self._install_live(self.gui(self.bridge.capture_live))
        return self.status()

    def save_project(self):
        if self.candidate() is not None:
            raise ValueError('Accept or Discard the candidate before saving')
        self.sync_live()
        result = self.gui(self.bridge.save_project)
        self._install_live(self.gui(self.bridge.capture_live))
        return result

    def accept_candidate(self, *, user_approved=False):
        self.sync_live()
        session = self._preflight_accept()
        if self.approval == 'review' and not user_approved:
            return {'status': 'pending_user_review'}
        frozen = FrozenMotion.capture(session.working_document)
        generated = self.generated
        metadata = tuple(session.metadata.snapshot().items())
        # Recompute effects on the GUI thread against its actual metadata/protection.
        self.gui(lambda: self.bridge.accept_motion(frozen, metadata, self.live_identity,
                                                  self.allow_user_override))
        self._session = None
        self.checkpoints.clear()
        self._install_live(self.gui(self.bridge.capture_live))
        if generated is not None:
            self.generated = replace(generated, source=self.identity(self.committed))
        return {'status': 'accepted', 'revision': self.committed.revision}


class GuiViewRenderer:
    """Capture immutable worker poses on the owning Qt/OpenGL thread."""
    def __init__(self, workspace):
        self.workspace = workspace

    def render(self, model, samples, views, width, height):
        from application.agent_rendering import RenderingUnavailable
        from application.ai.frame_capture import FrameCaptureError
        frozen = tuple((float(t), tuple(float(v) for v in q)) for t, q in samples)
        try:
            return self.workspace.gui(lambda: self._capture(frozen, tuple(views), width, height))
        except FrameCaptureError as error:
            raise RenderingUnavailable(str(error)) from error

    def _capture(self, samples, views, width, height):
        from PySide6.QtGui import QImage, QPainter, QColor
        from application.agent_rendering import RenderedImage
        from gui.ai_frame_capture import RobotViewerFrameRenderer, _encode_png
        viewer = self.workspace.bridge.host.viewer_3d
        canvas = viewer.canvas
        if getattr(canvas, '_rendering_failure', None):
            from application.ai.frame_capture import FrameCaptureError
            raise FrameCaptureError(canvas._rendering_failure)
        if getattr(canvas, '_geometry_queue', ()):
            from application.ai.frame_capture import FrameCaptureError
            raise FrameCaptureError('3D geometry is still loading; retry rendering when it is ready')
        saved = (canvas.camera_yaw, canvas.camera_pitch, canvas.camera_distance, canvas.camera_center.copy())
        renderer = RobotViewerFrameRenderer(viewer, maximum_dimension=max(width, height))
        images = []
        try:
            for time_s, qpos in samples:
                renderer._capture_state.set_qpos(qpos)
                canvas.camera_center = renderer._capture_state.mj_data.subtree_com[0].copy()
                canvas.camera_distance = max(float(viewer.robot_model.mj_model.stat.extent) * 1.6, 1.0)
                canvas.camera_pitch = 10.0
                for view in views:
                    if self.workspace.call_cancel.is_set():
                        raise RuntimeError('capture cancelled')
                    canvas.camera_yaw = 90.0 if view == 'front' else 180.0
                    frame = renderer.render_frame(qpos, time_seconds=time_s, variant=view)
                    image = QImage.fromData(frame.data).scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                                              Qt.TransformationMode.SmoothTransformation)
                    painter = QPainter(image)
                    painter.setPen(QColor('white'))
                    painter.drawText(8, image.height() - 8, view)
                    painter.end()
                    images.append(RenderedImage(_encode_png(image)))
            return tuple(images)
        finally:
            canvas.camera_yaw, canvas.camera_pitch, canvas.camera_distance, canvas.camera_center = saved
            canvas.update()


class ExternalAgentPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('externalAgentPanel')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        self.allow = QCheckBox('Allow external agent control')
        self.allow.setChecked(False)
        self.allow.setToolTip('Opt in for this session. The local agent stages a separate Orange preview.')
        self.policy = QComboBox()
        self.policy.addItem('Review before accepting', 'review')
        self.policy.addItem('Allow unattended acceptance', 'auto_accept')
        self.policy.setMinimumWidth(0)
        self.policy.setMinimumContentsLength(12)
        self.policy.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.override = QCheckBox('Edit existing Keyframes')
        self.override.setToolTip('Protected Keyframes remain blocked. This applies only to the next connection.')
        self.banner = QLabel('External agent control is off')
        self.banner.setWordWrap(True)
        self.activity = QLabel('No external tool activity')
        self.activity.setWordWrap(True)
        self.banner.setTextFormat(Qt.TextFormat.PlainText)
        self.activity.setTextFormat(Qt.TextFormat.PlainText)
        self.disconnect = QPushButton('Stop / Disconnect')
        self.accept = QPushButton('Accept')
        self.discard = QPushButton('Discard')
        row = QHBoxLayout()
        self.disconnect.setEnabled(False)
        for button in (self.accept, self.discard):
            row.addWidget(button)
            button.setEnabled(False)
        for widget in (self.allow, self.policy, self.override, self.banner, self.activity):
            layout.addWidget(widget)
        layout.addWidget(self.disconnect)
        layout.addLayout(row)


class AgentBridgeController(QObject):
    connected_changed = Signal(bool)
    external_activity = Signal(str, bool, str)  # tool, ok, message — external MCP calls only

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.panel = ExternalAgentPanel()
        host.ai_assistant_panel.layout().insertWidget(1, self.panel)
        self.invoker = QtInvoker(self)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ghostgui-agent')
        self.server = None
        self.workspace = self.runtime = None
        self.inapp_runtime = None
        self.inapp_port = None
        self.action_log = None
        # The user reviews every in-app candidate before Accept, so command-bar edits may change
        # user-owned (not protected) motion; external agents keep the panel's policy.
        self.inapp_allow_user_override = True
        self.candidate_origin = None
        # Undo-history label for in-app (command bar) accepts, e.g. "Claude: add a pause".
        self.accept_label = None
        self._external_futures = set()
        self.closed = False
        self.internal_transition = False
        self._pending = []
        self._tokens = set()
        self._lock = Lock()
        self.has_candidate = False
        self._last_publication = None
        self.connected = False
        self._owned_controls = False
        self.projects_dir = ghostgui_projects_dir().expanduser().resolve()
        self.approval = 'review'
        self.allow_user_override = False
        self.panel.allow.toggled.connect(self.set_enabled)
        self.panel.disconnect.clicked.connect(self.disconnect)
        self.panel.accept.clicked.connect(lambda: self.submit_local('_user_accept'))
        self.panel.discard.clicked.connect(lambda: self.submit_local('discard_candidate'))
        self.connected_changed.connect(self._connection_changed, Qt.ConnectionType.QueuedConnection)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    @property
    def owns_document(self):
        return self.server is not None or self.has_candidate or bool(self._pending)

    def capture_live(self):
        host = self.host
        if self.closed or getattr(host, '_shutting_down', False):
            raise RuntimeError('GUI is shutting down')
        if host.robot_model_3d is None:
            raise ValueError('No robot is loaded')
        metadata = host.ai_assistant_controller._metadata_service()
        metadata.seed_document_as_user_owned(host.document)
        return LiveWorkspace(FrozenMotion.capture(host.document), host.robot_model_3d,
                             tuple(host.model_registry.items()), tuple(metadata.store.snapshot().items()),
                             None if host.current_project is None else str(host.current_project.root_dir),
                             json.dumps(host.capture_project_workspace(), allow_nan=False), host.project_dirty)

    def set_enabled(self, enabled):
        if not enabled:
            self.disconnect()
            return
        try:
            ai = self.host.ai_assistant_controller
            if self.closed or self.has_candidate or self._pending or ai.session is not None or ai.active_handle is not None:
                raise ValueError('Resolve the current candidate/request before enabling external control')
            if self.host.background_jobs.is_busy() or self.host.model_loaders:
                raise ValueError('Wait for the current GUI operation to finish')
            self.capture_live()
            self.workspace = self.runtime = None
            self.approval = self.panel.policy.currentData()
            self.allow_user_override = self.panel.override.isChecked()
            self.host._refresh_history_baseline()
            self.server = BridgeServer(self.submit, info={'approval': self.approval, 'projects_dir': str(self.projects_dir)},
                                       connection_changed=self.connected_changed.emit)
            path = self.server.start()
            self.panel.banner.setText('External control enabled; waiting for one client.\n' + str(path))
            self.panel.disconnect.setEnabled(True)
            self.panel.policy.setEnabled(False)
            self.panel.override.setEnabled(False)
            self._owned_controls = True
            self.host.set_ai_motion_controls_enabled(False)
        except Exception as error:
            self.panel.banner.setText(str(error))
            self.panel.allow.blockSignals(True)
            self.panel.allow.setChecked(False)
            self.panel.allow.blockSignals(False)
            if self.server is not None:
                self.server.stop()
                self.server = None

    @Slot(bool)
    def _connection_changed(self, connected):
        if self.closed:
            return
        self.connected = connected
        self.panel.banner.setText('External agent connected — GUI workspace' if connected else
                                  ('Client disconnected; review the Orange preview or connect again'
                                   if self.server is not None else 'External control disconnected'))

    def submit(self, name, arguments, cancellation):
        return self._submit(name, arguments, cancellation, trusted=False)

    def submit_inapp(self, name, arguments, cancellation):
        """Command bar entry: same worker, in-app registry projection with GUI ports."""
        return self._submit(name, arguments, cancellation, trusted=False, inapp=True)

    def _submit(self, name, arguments, cancellation, *, trusted, inapp=False):
        with self._lock:
            self._tokens.add(cancellation)
            future = self.executor.submit(self._work, name, arguments, cancellation, trusted, inapp)
            self._pending.append((future, name, cancellation))
            if not trusted and not inapp:
                self._external_futures.add(future)
        return future

    def submit_local(self, name):
        with self._lock:
            busy = any(not future.done() for future, _, _ in self._pending)
        if busy:
            future = Future()
            future.set_result({'ok': False, 'payload': {'message': 'Wait for the current tool or Stop / Disconnect first'}})
            self.panel.activity.setText('Wait for the current tool or Stop / Disconnect first')
            return future
        return self._submit(name, {}, Event(), trusted=True)

    def _external_runtime(self, workspace):
        from application.commands import Origin, Surface
        from .commands.gui_ports import GuiCommandPort
        self.inapp_port = self.inapp_port or GuiCommandPort(self.host)
        return ToolRuntime(build_command_catalog(
            workspace, renderer=GuiViewRenderer(workspace), surface=Surface.MCP_ATTACHED,
            ports={'gui': self.inapp_port}, origin=Origin.AGENT_MCP, action_log=self.action_log,
            gui=lambda callback: workspace.gui(callback)), workspace)

    def _inapp_runtime(self):
        if self.inapp_runtime is None:
            from application.commands import Origin, Surface
            from .commands.gui_ports import GuiCommandPort
            workspace = self.workspace
            self.inapp_port = self.inapp_port or GuiCommandPort(self.host)
            self.inapp_runtime = ToolRuntime(build_command_catalog(
                workspace, renderer=GuiViewRenderer(workspace), surface=Surface.AGENT_INAPP,
                ports={'gui': self.inapp_port}, origin=Origin.AGENT_INAPP, action_log=self.action_log,
                gui=lambda callback: workspace.gui(callback)), workspace)
        return self.inapp_runtime

    def _work(self, name, arguments, cancellation, trusted=False, inapp=False):
        checkpoint = None
        try:
            if cancellation.is_set():
                raise RuntimeError('operation cancelled')
            if self.workspace is None:
                live = self.invoker.call(self.capture_live, cancellation)
                self.workspace = GuiWorkspace(self, live)
                self.runtime = self._external_runtime(self.workspace)
                self.inapp_runtime = None
            workspace = self.workspace
            # Whichever surface creates the candidate owns it until Accept or Discard.
            if not trusted:
                surface = 'inapp' if inapp else 'external'
                if workspace.candidate() is None:
                    self.candidate_origin = surface
                elif self.candidate_origin not in (None, surface):
                    raise ValueError('document_owned_elsewhere: resolve the other Orange preview first')
            workspace.allow_user_override = (self.inapp_allow_user_override if self.candidate_origin == 'inapp'
                                             else self.allow_user_override)
            workspace.call_cancel = cancellation
            runtime = self._inapp_runtime() if inapp else self.runtime
            runtime.cancellation_token = CancelToken(cancellation)
            spec = runtime.catalog.specs.get(name) if isinstance(name, str) else None
            if spec is not None:
                from application.commands import Effect
                light = spec.effect in (Effect.READ, Effect.UI, Effect.DESTRUCTIVE) and 'gui' in spec.requires
                if inapp and self.server is not None and not light and spec.effect is not Effect.READ:
                    raise ValueError('document_owned_elsewhere: external control owns the document')
            else:
                light = False
            if name != 'discard_candidate' and not light:
                workspace.sync_live()
            if not inapp:
                self.invoker.call(lambda: self.panel.activity.setText('Running: ' + str(name)[:64]), cancellation)
            meta = runtime.catalog.meta.get(name) if isinstance(name, str) else None
            if (meta is not None and meta.mutates_candidate) or name in ('generate_trajectory', 'validate_motion', 'undo'):
                checkpoint = workspace.call_checkpoint()
            if name == '_user_accept' and trusted:
                report = workspace.validate(cancellation_token=CancelToken(cancellation))
                if not report['valid']:
                    raise ValueError('; '.join(report['issues']))
                result = workspace.accept_candidate(user_approved=True)
                outcome = {'ok': True, 'payload': {'ok': True, **result}, 'images': [], 'is_error': False}
            else:
                result = runtime.call(name, arguments)
                outcome = {'ok': result.ok, 'payload': result.payload, 'is_error': result.is_error,
                           'images': [{'data': base64.b64encode(image.data).decode('ascii'), 'mime_type': image.mime_type}
                                      for image in result.images]}
            if cancellation.is_set():
                raise RuntimeError('operation cancelled; late result ignored')
            session = workspace.candidate()
            frozen = None if session is None or not session.has_changes else FrozenMotion.capture(session.working_document)
            if (frozen is not None and workspace.generated is not None
                    and workspace.generated.source == workspace.identity()):
                export = workspace.generated.export
                frozen = frozen.with_qpos(zip(export.times, export.qposes))
            self.invoker.call(lambda: self.present(frozen), cancellation)
            return outcome
        except Exception as error:
            if checkpoint is not None and cancellation.is_set():
                self.workspace.restore_call(checkpoint)
                session = self.workspace.candidate()
                restored = None if session is None or not session.has_changes else FrozenMotion.capture(session.working_document)
                if restored is not None and self.workspace.generated is not None:
                    export = self.workspace.generated.export
                    restored = restored.with_qpos(zip(export.times, export.qposes))
                try:
                    # UI cleanup is independent of the revoked network credential.
                    # The invoker's closed gate still suppresses shutdown callbacks.
                    self.invoker.call(lambda: self.present(restored), Event())
                except Exception:
                    pass
            return {'ok': False, 'payload': {'ok': False, 'code': 'gui_bridge_failed', 'message': str(error)[:2000],
                                                 'revision': 0 if self.workspace is None else self.workspace.document().revision},
                    'is_error': True, 'images': []}

    def present(self, frozen):
        if frozen is not None and frozen == self._last_publication:
            return
        self._last_publication = frozen
        self.has_candidate = frozen is not None
        if frozen is not None:
            # This document belongs only to the GUI and will never be mutated.
            preview = frozen.document(self.host.robot_model_3d)
            self.host.viewer_3d.begin_candidate_preview(
                preview.qpos_timeline.sample_state, preview.timeline_duration)
            self.host.set_sidebar_timeline_duration(preview.timeline_duration)
        else:
            self.host.viewer_3d.end_candidate_preview()
            self.host.set_sidebar_timeline_duration(self.host.document.timeline_duration)
        self.panel.accept.setEnabled(self.has_candidate)
        self.panel.discard.setEnabled(self.has_candidate)

    @contextmanager
    def transition(self):
        self.internal_transition = True
        try:
            yield
        finally:
            self.internal_transition = False

    def load_model(self, key, adapter):
        with self.transition():
            if key in self.host.model_sessions:
                self.host.on_model_changed(key)
            else:
                self.host.on_model_loaded(key, adapter)
            self.host.set_ai_motion_controls_enabled(False)

    def new_project(self, path, name, duration):
        with self.transition():
            self.host.create_project_at(path, name, reset_workspace=True)
            self.host.set_editor_timeline_duration(duration)
            self.host.save_current_project(show_status=False)
            self.host._refresh_history_baseline()
            self.host.set_ai_motion_controls_enabled(False)

    def open_project(self, path):
        host = self.host
        before = host.capture_history_snapshot()
        workspace = host.capture_project_workspace()
        project, revision, dirty = host.current_project, host.document.revision, host.document.dirty
        project_dirty = host.project_dirty
        original_payload = host._project_workspace_payload
        with self.transition():
            try:
                if not host.restore_project(GhostGUIProject.open(path), report_error=False):
                    raise ValueError('GUI could not restore the project')
            except Exception:
                host.restore_history_snapshot(before)
                host.restore_project_workspace(workspace)
                host.current_project = project
                host._project_workspace_payload = original_payload
                host.document.revision, host.document.dirty = revision, dirty
                host.set_project_dirty(project_dirty)
                raise
            finally:
                host.set_ai_motion_controls_enabled(False)

    def save_project(self):
        if self.host.current_project is None:
            raise ValueError('Create or open a project before saving')
        if not self.host.save_current_project(show_status=False):
            raise ValueError('GUI save failed')
        return {'saved': True, 'project': self.host.current_project.project_name}

    def accept_motion(self, frozen, metadata, source, allow_user_override):
        host = self.host
        current = motion_source_identity(host.robot_model_3d, host.document.qpos_timeline, document=host.document)
        if current != source:
            raise ValueError('committed GUI motion changed; candidate is stale')
        ai = host.ai_assistant_controller
        if ai.session is not None or ai.active_handle is not None:
            raise ValueError('Another Motion Assistant session is active')
        candidate = frozen.document(host.robot_model_3d)
        store = ai.metadata_store
        service = ai._metadata_service()
        def values(document):
            result = {service.reference_for_keyframe(frame): frame.to_dict() for frame in document.trajectory.frames}
            result.update({service.reference_for_qpos_keyframe(t): tuple(document.qpos_timeline.get_state(t))
                           for t in document.qpos_timeline.times()})
            return result
        before, after = values(host.document), values(candidate)
        affected = tuple(ref for ref in before.keys() | after.keys() if before.get(ref) != after.get(ref))
        created = tuple(ref for ref in affected if ref not in before)
        session = AIEditSession(host.document, metadata_store=store)
        session.apply_ai(ReplaceMotionState(capture_motion_state(candidate)), affected_entities=affected,
                         created_entities=created, allow_user_override=allow_user_override)
        merged = dict(metadata)
        merged.update({ref: value for ref, value in store.snapshot().items() if ref not in affected})
        session.metadata.replace(merged)
        session.mark_current_revision_validated()
        ai.session = session
        # Reuse the existing atomic Accept/provenance/history/teardown path.
        ai.accept(self.accept_label if self.candidate_origin == 'inapp' else None)
        if ai.session is not None:
            ai.session = None
            raise ValueError('GUI acceptance failed; inspect the Motion Assistant error')
        self.present(None)
        host.set_ai_motion_controls_enabled(False)

    def _poll(self):
        with self._lock:
            done = [item for item in self._pending if item[0].done()]
            self._pending = [item for item in self._pending if not item[0].done()]
            for _, _, token in done:
                self._tokens.discard(token)
        for future, name, token in done:
            external = future in self._external_futures
            self._external_futures.discard(future)
            if not future.cancelled():
                result = future.result()
                if external:
                    self.external_activity.emit(str(name)[:64], bool(result['ok']),
                                                str(result['payload'].get('message', ''))[:200])
                self.panel.activity.setText(('Finished: ' if result['ok'] else 'Failed: ') + str(name)[:64] +
                                            ('' if result['ok'] else '\n' + result['payload'].get('message', '')[:400]))
        busy = bool(self._pending)
        self.panel.accept.setEnabled(self.has_candidate and not busy)
        self.panel.discard.setEnabled(self.has_candidate and not busy)
        if self._owned_controls and self.server is None and not self._pending and not self.has_candidate and not self.closed:
            self._owned_controls = False
            self.host.set_ai_motion_controls_enabled(True)
            self.panel.policy.setEnabled(True)
            self.panel.override.setEnabled(True)

    def disconnect(self):
        if self.server is not None:
            self.server.stop()
            self.server = None
        with self._lock:
            for token in self._tokens:
                token.set()
        self.panel.allow.blockSignals(True)
        self.panel.allow.setChecked(False)
        self.panel.allow.blockSignals(False)
        self.panel.disconnect.setEnabled(False)
        self.panel.banner.setText('External control disconnected; resolve any Orange preview')
        self.connected = False

    def shutdown(self):
        self.closed = True
        self.disconnect()
        self.timer.stop()
        self.invoker.shutdown()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.host.viewer_3d.end_candidate_preview()

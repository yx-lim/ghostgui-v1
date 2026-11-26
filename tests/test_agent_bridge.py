"""Authenticated attachment, Qt ownership, review, cancellation and isolation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
from threading import Event, get_ident
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from application.agent_bridge.client import BridgeClient
from application.agent_bridge.protocol import BridgeError, read_discovery, read_message, write_message
from application.agent_bridge.snapshots import FrozenMotion
from application.trajectory_export_service import motion_source_identity
from gui.agent_bridge import QtInvoker
from gui.main_window import RobotGuiMainWindow


class AgentBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict(os.environ, {
            'GHOSTGUI_CONFIG_DIR': str(self.root / 'config'),
            'GHOSTGUI_CACHE_DIR': str(self.root / 'cache'),
            'GHOSTGUI_PROJECTS_DIR': str(self.root / 'projects'),
            'GHOSTGUI_USER_DATA_DIR': str(self.root / 'data'),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.window = RobotGuiMainWindow()
        self.bridge = self.window.agent_bridge_controller
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.addCleanup(self.cleanup_window)
        self.client = None

    def cleanup_window(self):
        if self.client is not None:
            self.client.close()
        self.bridge.shutdown()
        self.window.set_project_dirty(False)
        self.window.close()
        self.app.processEvents()

    def wait(self, future, timeout=25):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)
        self.assertTrue(future.done(), 'asynchronous bridge operation timed out')
        self.app.processEvents()
        return future.result()

    def enable(self):
        self.bridge.panel.allow.setChecked(True)
        self.assertIsNotNone(self.bridge.server, self.bridge.panel.banner.text())
        self.path = self.bridge.server.path
        self.client = self.wait(self.pool.submit(BridgeClient, self.path))
        return self.client

    def call(self, name, arguments=None, ok=True):
        outcome = self.wait(self.pool.submit(self.client.call, name, arguments or {}))
        self.assertEqual(outcome['ok'], ok, outcome)
        return outcome['payload']

    def author(self):
        self.call('new_project', {'name': 'motion', 'duration_s': 0.4})
        home = self.call('get_pose', {'time_s': 0})['qpos']
        middle = list(home)
        middle[-1] += 0.05
        return self.call('author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.2, 'qpos': middle},
            {'time_seconds': 0.4, 'qpos': home}]})

    def test_opt_in_authentication_identity_and_competing_client(self):
        self.assertFalse(self.bridge.panel.allow.isChecked())
        self.assertIsNone(self.bridge.server)
        self.assertFalse((self.root / 'config' / 'agent-bridges').exists())
        self.enable()
        discovery = read_discovery(self.path)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
        with self.assertRaises(BridgeError):
            self.wait(self.pool.submit(BridgeClient, self.path))
        def bad_hello():
            with socket.create_connection(('127.0.0.1', discovery['port'])) as connection:
                with connection.makefile('rwb') as stream:
                    write_message(stream, {**discovery, 'token': 'x' * 40})
                    return read_message(stream)
        self.assertFalse(self.wait(self.pool.submit(bad_hello))['ok'])
        status = self.call('get_status')
        self.assertEqual(status['workspace'], 'gui')
        self.assertEqual(status['approval'], 'review')
        self.call('_user_accept', ok=False)
        self.call('accept_candidate', {'user_approved': True}, ok=False)
        path = self.path
        self.bridge.disconnect()
        self.assertFalse(path.exists())

    def test_candidate_isolation_review_accept_one_undo_and_redo(self):
        self.enable()
        self.author()
        window = self.window
        committed = FrozenMotion.capture(window.document)
        depth = len(window.undo_stack)
        self.assertTrue(window.viewer_3d.candidate_preview_active)
        sampler = window.viewer_3d._candidate_preview_sampler
        self.assertIsNot(sampler.__self__, self.bridge.workspace.document().qpos_timeline)
        candidate_before = sampler(0.2).copy()
        self.bridge.workspace.document().qpos_timeline.states[0.2][-1] += 0.01
        self.assertEqual(tuple(sampler(0.2)), tuple(candidate_before))
        self.bridge.workspace.document().qpos_timeline.states[0.2][-1] -= 0.01
        window.viewer_3d.preview_trajectory_time(0.1)
        self.assertEqual(FrozenMotion.capture(window.document), committed)
        self.assertEqual(len(window.undo_stack), depth)
        self.call('generate_trajectory', {'dt': 0.02})
        self.assertGreater(len(window.viewer_3d._candidate_preview_sampler.__self__.times()), 3)
        report = self.call('validate_motion')
        self.assertTrue(report['valid'], report)
        self.assertEqual(self.call('accept_candidate')['status'], 'pending_user_review')
        self.assertEqual(FrozenMotion.capture(window.document), committed)
        self.wait(self.bridge.submit_local('_user_accept'))
        self.assertEqual(len(window.undo_stack), depth + 1)
        self.assertFalse(window.viewer_3d.candidate_preview_active)
        self.assertFalse(self.bridge.has_candidate)
        self.call('save_project')
        export = self.call('export_trajectory', {'format': 'mujoco_csv', 'file_name': 'accepted'})
        self.assertTrue(Path(export['paths'][0]).exists())
        self.assertEqual(export['source'], 'generated trajectory')
        self.assertEqual(export['samples'], 21)
        self.bridge.disconnect()
        self.bridge._poll()
        accepted = FrozenMotion.capture(window.document)
        window.undo_last_action()
        self.assertEqual(FrozenMotion.capture(window.document).payload, committed.payload)
        window.redo_last_action()
        self.assertEqual(FrozenMotion.capture(window.document).payload, accepted.payload)

    def test_stale_acceptance_and_late_protection_are_blocked(self):
        self.enable()
        self.author()
        self.call('validate_motion')
        self.window.document.qpos_timeline.states[0.0][-1] += 0.005
        result = self.wait(self.bridge.submit_local('_user_accept'))
        self.assertFalse(result['ok'], result)
        self.assertIn('changed', result['payload']['message'])
        self.call('discard_candidate')
        self.assertFalse(self.window.viewer_3d.candidate_preview_active)

    def test_dirty_and_candidate_transitions_never_discard(self):
        self.enable()
        self.call('new_project', {'name': 'first'})
        self.window.document.mark_changed()
        self.window.set_project_dirty(True)
        before = FrozenMotion.capture(self.window.document)
        self.call('new_project', {'name': 'second'}, ok=False)
        self.assertEqual(FrozenMotion.capture(self.window.document), before)
        self.call('save_project')
        self.call('new_project', {'name': 'second', 'duration_s': 0.4})
        home = self.call('get_pose', {'time_s': 0})['qpos']
        self.call('author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.4, 'qpos': home}]})
        self.call('load_model', {'model_key': 'go2'}, ok=False)
        self.call('open_project', {'name': 'first'}, ok=False)
        self.assertFalse(self.window.confirm_project_transition('open another project'))
        self.assertEqual(self.window.model_key, 'g1')
        self.call('discard_candidate')
        self.call('open_project', {'name': 'first'})
        self.assertEqual(self.window.current_project.project_name, 'first')
        self.call('load_model', {'model_key': 'go2'})
        self.assertEqual(self.window.model_key, 'go2')
        self.assertFalse(self.window.controls.isEnabled())

    def test_unattended_policy_is_gui_only_and_protection_still_applies(self):
        self.bridge.panel.policy.setCurrentIndex(1)
        self.enable()
        self.author()
        self.call('validate_motion')
        service = self.window.ai_assistant_controller._metadata_service()
        reference = service.reference_for_qpos_keyframe(0.0)
        # Protection of an unchanged pose must survive acceptance as well.
        service.store.set_protected(reference, True)
        self.assertEqual(self.call('accept_candidate')['status'], 'accepted')
        self.assertTrue(service.store.get(reference).protected)
        self.assertEqual(len(self.window.undo_stack), 1)

    def test_disconnect_mid_mutation_rolls_back_and_drops_publication(self):
        self.enable()
        self.call('get_status')
        started, release = Event(), Event()
        original = self.bridge.workspace.operation
        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            started.set()
            release.wait(5)
            return result
        self.bridge.workspace.operation = delayed
        home = self.window.robot_model_3d.home_qpos.tolist()
        future = self.pool.submit(self.client.call, 'author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.4, 'qpos': home}]})
        deadline = time.monotonic() + 5
        while not started.is_set() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)
        self.assertTrue(started.is_set())
        before = FrozenMotion.capture(self.window.document)
        self.bridge.disconnect()
        release.set()
        with self.assertRaises(Exception):
            self.wait(future)
        for pending, _, _ in list(self.bridge._pending):
            self.wait(pending)
        self.bridge._poll()
        self.assertEqual(FrozenMotion.capture(self.window.document), before)
        self.assertFalse(self.window.viewer_3d.candidate_preview_active)
        self.assertIsNone(self.bridge.workspace.candidate())

    def test_disconnect_after_runtime_before_publication_rolls_back(self):
        self.enable()
        self.call('get_status')
        completed, release = Event(), Event()
        original = self.bridge.runtime.call
        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            completed.set()
            release.wait(5)
            return result
        self.bridge.runtime.call = delayed
        home = self.window.robot_model_3d.home_qpos.tolist()
        future = self.pool.submit(self.client.call, 'author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.4, 'qpos': home}]})
        deadline = time.monotonic() + 5
        while not completed.is_set() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)
        self.assertTrue(completed.is_set())
        self.bridge.disconnect()
        release.set()
        with self.assertRaises(Exception):
            self.wait(future)
        for pending, _, _ in list(self.bridge._pending):
            self.wait(pending)
        self.bridge._poll()
        self.assertIsNone(self.bridge.workspace.candidate())
        self.assertFalse(self.window.viewer_3d.candidate_preview_active)
        self.assertTrue(self.window.controls.isEnabled())

    def test_disconnect_during_publication_restores_the_visible_boundary(self):
        self.enable()
        self.call('get_status')
        original = self.bridge.present
        def stop_during_publication(frozen):
            original(frozen)
            if frozen is not None:
                self.bridge.disconnect()
        self.bridge.present = stop_during_publication
        home = self.window.robot_model_3d.home_qpos.tolist()
        future = self.pool.submit(self.client.call, 'author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.4, 'qpos': home}]})
        with self.assertRaises(Exception):
            self.wait(future)
        for pending, _, _ in list(self.bridge._pending):
            self.wait(pending)
        self.bridge._poll()
        self.assertIsNone(self.bridge.workspace.candidate())
        self.assertFalse(self.window.viewer_3d.candidate_preview_active)
        self.assertFalse(self.bridge.has_candidate)

    def test_invoker_timeout_shutdown_and_thread_ownership(self):
        main_thread = get_ident()
        self.assertEqual(self.wait(self.pool.submit(self.bridge.invoker.call, get_ident, Event())), main_thread)
        invoker = QtInvoker(timeout=0.03)
        called = []
        future = self.pool.submit(invoker.call, lambda: called.append(True), Event())
        time.sleep(0.06)
        with self.assertRaises(RuntimeError):
            future.result()
        self.app.processEvents()
        self.assertFalse(called)
        future = self.pool.submit(self.bridge.invoker.call, lambda: called.append(True), Event())
        time.sleep(0.02)
        self.bridge.shutdown()
        with self.assertRaises(Exception):
            future.result(timeout=1)
        self.app.processEvents()
        self.assertFalse(called)

    def test_late_protection_of_affected_content_blocks_acceptance(self):
        self.bridge.panel.override.setChecked(True)
        self.enable()
        home = self.call('get_pose', {'time_s': 0})['qpos']
        changed = list(home)
        changed[-1] += 0.05
        self.call('author_keyframes', {'duration_s': 0.4, 'keyframes': [
            {'time_seconds': 0, 'qpos': changed}, {'time_seconds': 0.4, 'qpos': home}]})
        self.call('validate_motion')
        service = self.window.ai_assistant_controller._metadata_service()
        reference = service.reference_for_qpos_keyframe(0.0)
        service.store.set_protected(reference, True)
        before = FrozenMotion.capture(self.window.document)
        result = self.wait(self.bridge.submit_local('_user_accept'))
        self.assertFalse(result['ok'], result)
        self.assertIn('protected', result['payload']['message'])
        self.assertEqual(FrozenMotion.capture(self.window.document), before)
        self.assertFalse(self.window.undo_stack)
        self.call('discard_candidate')

    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional agent extra is not installed')
    def test_real_mcp_client_attaches_and_cannot_override_gui_policy(self):
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import sys
        self.bridge.panel.allow.setChecked(True)
        self.assertIsNotNone(self.bridge.server, self.bridge.panel.banner.text())
        path = self.bridge.server.path
        async def scenario():
            with tempfile.TemporaryFile(mode='w+') as errors:
                async with stdio_client(StdioServerParameters(command=sys.executable, args=[
                    '-m', 'application.mcp_server', '--attach', 'required', '--discovery-file', str(path),
                    '--approval', 'auto_accept']), errlog=errors) as (read, write):
                    async with ClientSession(read, write) as session:
                        info = await session.initialize()
                        self.assertIn('gui', info.instructions)
                        listed = {tool.name for tool in (await session.list_tools()).tools}
                        legacy = {tool['name'] for tool in json.loads(
                            (Path(__file__).parent / 'fixtures' / 'legacy_agent_tools.json').read_text())['tools']}
                        # Attached sessions keep every shipped tool and add GUI app commands (M3).
                        self.assertLessEqual(legacy, listed)
                        self.assertIn('set_view', listed)
                        resource = await session.read_resource('ghostgui://workspace')
                        self.assertEqual(json.loads(resource.contents[0].text)['approval'], 'review')
                        async def call(name, arguments):
                            result = await session.call_tool(name, arguments)
                            self.assertFalse(result.isError, result)
                            return result.structuredContent
                        home = (await call('get_pose', {'time_s': 0}))['qpos']
                        await call('author_keyframes', {'duration_s': 0.4, 'keyframes': [
                            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.4, 'qpos': home}]})
                        validated = await call('validate_motion', {})
                        self.assertTrue(validated['valid'])
                        self.assertEqual((await call('accept_candidate', {}))['status'], 'pending_user_review')
                errors.seek(0)
                self.assertIn('workspace=gui', errors.read())
        before = FrozenMotion.capture(self.window.document)
        self.wait(self.pool.submit(asyncio.run, scenario()))
        self.assertEqual(FrozenMotion.capture(self.window.document), before)
        self.assertTrue(self.window.viewer_3d.candidate_preview_active)
        self.wait(self.bridge.submit_local('_user_accept'))
        self.assertEqual(len(self.window.undo_stack), 1)

    def test_save_and_autosave_during_preview_keep_committed_domain(self):
        self.enable()
        self.call('new_project', {'name': 'committed', 'duration_s': 2.0})
        home = self.call('get_pose', {'time_s': 0})['qpos']
        self.call('author_keyframes', {'duration_s': 1.0, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 1, 'qpos': home}]})
        self.window.viewer_3d.preview_trajectory_time(0.8)
        self.assertEqual(self.window.document.timeline_duration, 2.0)
        self.assertTrue(self.window.save_current_project(show_status=False))
        self.assertTrue(self.window.autosave_current_project(show_status=False))
        project = self.window.current_project
        for autosave in (False, True):
            payload = project.read_workspace(autosave=autosave)
            self.assertEqual(payload['timeline_duration'], 2.0)
            self.assertEqual(payload['current_time'], self.window.document.current_time)
            self.assertEqual(payload['control_frame']['time'], self.window.document.current_time)
        self.assertEqual(self.window.document.qpos_timeline.times(), [0.0])
        self.call('discard_candidate')

    def test_attached_open_save_preserves_opaque_project_workspace(self):
        self.enable()
        self.call('new_project', {'name': 'compatible'})
        project = self.window.current_project
        payload = project.read_workspace()
        payload['extension'] = {'opaque': ['keep', 1]}
        payload['camera']['future_setting'] = 42
        project.write_workspace(payload)
        self.call('open_project', {'name': 'compatible'})
        self.call('save_project')
        result = project.read_workspace()
        self.assertEqual(result['extension'], {'opaque': ['keep', 1]})
        self.assertEqual(result['camera']['future_setting'], 42)

    def test_read_only_calls_preserve_preview_sampler_and_loading_capture_is_unavailable(self):
        self.enable()
        self.author()
        viewer = self.window.viewer_3d
        sampler = viewer._candidate_preview_sampler
        self.call('get_status')
        self.call('get_pose', {'time_s': 0.2})
        self.assertIs(viewer._candidate_preview_sampler, sampler)
        with patch.object(viewer.canvas, '_geometry_queue', [1]):
            report = self.call('render_views', {'times_s': [0.2]})
        self.assertFalse(report['available'])
        self.assertEqual(report['code'], 'rendering_unavailable')
        self.assertIn('loading', report['message'])
        self.assertIs(viewer._candidate_preview_sampler, sampler)

    def test_discovery_permissions_and_request_size_fail_closed(self):
        self.enable()
        identity = read_discovery(self.path)
        if os.name == 'posix':
            self.path.chmod(0o644)
            try:
                with self.assertRaises(BridgeError):
                    read_discovery(self.path)
            finally:
                self.path.chmod(0o600)
        self.client.close()
        self.client = None
        self.bridge.disconnect()
        self.bridge._poll()
        # A new session has fresh credentials and rejects stale discovery.
        self.bridge.panel.allow.setChecked(True)
        self.assertNotEqual(self.bridge.server.session_id, identity['session_id'])
        self.client = self.wait(self.pool.submit(BridgeClient, self.bridge.server.path))
        oversized = {'extra': 'x' * 262144}
        with self.assertRaises(BridgeError):
            self.wait(self.pool.submit(self.client.call, 'get_status', oversized))
        self.assertIsNone(self.bridge.workspace)

    def test_gui_capture_cancellation_restores_camera(self):
        from gui.agent_bridge import GuiViewRenderer
        from application.ai.frame_capture import FrameCaptureError
        self.enable()
        self.call('get_status')
        canvas = self.window.viewer_3d.canvas
        before = (canvas.camera_yaw, canvas.camera_pitch, canvas.camera_distance, tuple(canvas.camera_center))
        def capture_failure(*args, **kwargs):
            self.bridge.invoker.shutdown()
            raise FrameCaptureError('shutdown mid-capture')
        renderer = GuiViewRenderer(self.bridge.workspace)
        sample = ((0, tuple(self.window.robot_model_3d.home_qpos)),)
        with patch('gui.ai_frame_capture.RobotViewerFrameRenderer.render_frame', side_effect=capture_failure):
            with self.assertRaises(FrameCaptureError):
                renderer._capture(sample, ('front',), 160, 120)
        self.assertEqual((canvas.camera_yaw, canvas.camera_pitch, canvas.camera_distance, tuple(canvas.camera_center)), before)

    def test_queued_capture_ignored_on_shutdown(self):
        called = []
        future = self.pool.submit(self.bridge.invoker.call, lambda: called.append('capture'), Event())
        time.sleep(0.02)
        self.bridge.shutdown()
        self.app.processEvents()
        self.assertFalse(called)
        with self.assertRaises(Exception):
            future.result(timeout=1)


if __name__ == '__main__':
    unittest.main()

"""M3: every in-app agent command runs without modal dialogs; GUI/agent parity; route mode."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from threading import Event
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox

from application.commands import Surface, default_registry
from gui.main_window import RobotGuiMainWindow
from gui.viewers.mujoco_player import Mujoco3DViewerPanel


def _forbidden(*args, **kwargs):
    raise AssertionError("DIALOG_OPENED")


DIALOG_PATCHES = (
    (QDialog, "exec"), (QMessageBox, "question"), (QMessageBox, "warning"), (QMessageBox, "information"),
    (QMessageBox, "critical"), (QInputDialog, "getDouble"), (QInputDialog, "getText"), (QInputDialog, "getInt"),
    (QInputDialog, "getItem"), (QFileDialog, "getOpenFileName"), (QFileDialog, "getSaveFileName"),
    (QFileDialog, "getExistingDirectory"),
)


def examples(window):
    frame = window.controls.frame_box.itemText(0)
    return [
        ("get_app_state", {}), ("capture_viewport", {}), ("get_recent_actions", {}), ("get_robot_summary", {}),
        ("list_models", {}), ("list_projects", {}), ("list_keyframes", {}), ("get_pose", {"time_s": 0}),
        ("validate_motion", {}), ("render_views", {"times_s": [0]}), ("go_to_time", {"time_s": 0.2}),
        ("go_to_time", {"keyframe": "first"}), ("playback", {"action": "play", "speed": 1.5}),
        ("playback", {"action": "pause"}), ("set_view", {"left_sidebar": False, "right_sidebar": True}),
        ("set_view", {"left_sidebar": True}), ("set_display", {"show_keyframes": False}),
        ("set_camera", {"preset": "front"}), ("set_gizmo", {"mode": "rotate"}), ("select_target", {"frame": frame}),
        ("preview_path", {}), ("set_export_settings", {"sample_interval_s": 0.02}),
        ("set_generation_settings", {"smoothing": 0.2}), ("set_ik_settings", {"damping": 0.05}),
        ("show_help", {"topic": "keyboard_shortcuts"}),
        ("insert_time", {"at_s": 0.2, "duration_s": 0.1}), ("shift_motion", {"offset_s": 0.1}),
        ("set_motion_duration", {"duration_s": 2.0}), ("move_time_range", {"start_s": 0.1, "end_s": 0.3, "destination_s": 0.9}),
        ("copy_motion_range", {"start_s": 0.0, "end_s": 0.3}), ("paste_motion", {"at_s": 1.2}),
        ("reset_pose", {"time_s": 0.0}), ("set_joint_angles", {"time_s": 0.0, "values": [{"joint": "x", "angle_rad": 0}]}),
        ("solve_ik", {"time_s": 0, "logical_frame": frame, "position_m": [0.2, 0.2, 0.9]}),
        ("commit_keyframe", {"time_s": 0.0}), ("repeat_motion", {"start_s": 0, "end_s": 0.4, "additional_copies": 1}),
        ("delete_keyframes", {"times": [0.4]}), ("edit_timeline", {"start_s": 0, "end_s": 0.4, "scale": 2}),
        ("apply_edit_spec", {"spec_json": "{}"}), ("author_keyframes", {"keyframes": [
            {"time_seconds": 0, "qpos": [0.0]}, {"time_seconds": 0.1, "qpos": [0.0]}]}),
        ("generate_trajectory", {}), ("undo", {}), ("import_file", {"kind": "qpos", "file_name": "none"}),
        ("clear_motion", {}), ("discard_candidate", {}), ("accept_candidate", {}),
        ("export_trajectory", {"format": "qpos_csv", "file_name": "x"}), ("save_project", {}),
        ("history_undo", {}), ("history_redo", {}), ("simulation_viewer", {"action": "open"}),
        ("simulation_viewer", {"action": "close"}), ("open_project", {"name": "motion"}),
        ("load_model", {"model_key": "g1"}), ("new_project", {"name": "second"}),
    ]


class GuiCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        environment = patch.dict(os.environ, {
            "GHOSTGUI_CONFIG_DIR": str(root / "config"), "GHOSTGUI_CACHE_DIR": str(root / "cache"),
            "GHOSTGUI_PROJECTS_DIR": str(root / "projects"), "GHOSTGUI_USER_DATA_DIR": str(root / "data"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.window = RobotGuiMainWindow()
        self.bridge = self.window.agent_bridge_controller
        self.addCleanup(self._close)

    def _close(self):
        self.window.agent_command_bar.shutdown()
        self.bridge.shutdown()
        self.window.set_project_dirty(False)
        self.window.close()
        self.app.processEvents()

    def wait(self, future, timeout=30):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.002)
        self.assertTrue(future.done(), "timed out")
        self.app.processEvents()
        return future.result()

    def call(self, name, arguments, ok=None):
        result = self.wait(self.bridge.submit_inapp(name, arguments, Event()))
        if ok is not None:
            self.assertEqual(bool(result["ok"]), ok, result)
        return result

    def accept(self):
        result = self.wait(self.bridge.submit_local("_user_accept"))
        self.assertTrue(result["ok"], result)

    def seed_motion(self):
        self.call("new_project", {"name": "motion", "duration_s": 0.6}, ok=True)
        home = [float(v) for v in self.window.document.qpos_timeline.sample_state(0.0)]
        middle = list(home)
        middle[-1] += 0.05
        self.call("author_keyframes", {"duration_s": 0.6, "keyframes": [
            {"time_seconds": t, "qpos": q} for t, q in ((0, home), (0.2, middle), (0.4, middle), (0.6, home))]},
            ok=True)
        self.accept()

    def test_every_inapp_command_is_dialog_free(self):
        self.seed_motion()
        listed = examples(self.window)
        tools = {tool["name"] for tool in default_registry().tool_definitions(Surface.AGENT_INAPP, ports=("gui",))}
        self.assertEqual(tools - {"ask_user"} - {name for name, _ in listed}, set(), "add an example")
        patches = [patch.object(owner, name, _forbidden) for owner, name in DIALOG_PATCHES]
        patches.append(patch.object(Mujoco3DViewerPanel, "open_viewer", lambda self: None))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        for name, arguments in listed:
            result = self.call(name, arguments)
            self.assertNotIn("DIALOG_OPENED", str(result), name)

    def test_gui_and_agent_insert_time_parity(self):
        self.seed_motion()

        def state():
            document = self.window.document
            return ([round(t, 6) for t in document.qpos_timeline.times()],
                    [round(float(v), 9) for t in document.qpos_timeline.times()
                     for v in document.qpos_timeline.get_state(t)],
                    sorted((round(f.time, 6), tuple(sorted(f.to_dict().items()))) for f in document.trajectory.frames))

        deadline = time.monotonic() + 5
        while self.bridge.owns_document and time.monotonic() < deadline:
            self.app.processEvents()  # the bridge releases finished calls on its poll timer
            time.sleep(0.01)
        self.assertFalse(self.bridge.owns_document)
        before = len(self.window.undo_stack)
        self.window.insert_timeline_time(0.2, 0.3)
        gui_state, gui_entries = state(), len(self.window.undo_stack)
        self.window.undo_last_action()
        self.assertEqual(len(self.window.undo_stack), before)
        self.call("insert_time", {"at_s": 0.2, "duration_s": 0.3}, ok=True)
        self.accept()
        self.assertEqual(state()[0], gui_state[0])
        self.assertEqual(state()[1], gui_state[1])
        self.assertEqual(len(self.window.undo_stack), gui_entries)

    def test_route_mode_logs_user_actions(self):
        log = self.window.agent_command_bar.controller.action_log
        self.window.findChild(type(self.window.left_sidebar_action), "leftSidebarAction").trigger()
        self.assertEqual(log.tail(1)[0]["command"], "set_view")
        self.assertEqual(log.tail(1)[0]["origin"], "user_gui")


if __name__ == "__main__":
    unittest.main()

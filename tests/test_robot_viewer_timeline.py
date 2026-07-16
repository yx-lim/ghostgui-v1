import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from core.ik import Collision
from application.backend_interface import PythonRobotConfiguration
from gui.main_window import INITIAL_RENDER_PROGRESS_DELAY_MS, RobotGuiMainWindow
from gui.viewers.transform_gizmo import GizmoInteractionState
from core.trajectory import quat_to_rpy, rpy_to_quat
from scripts.view_g1_mujoco import RAW_QPOS_KEY, load_trajectory_csv


class RobotViewerTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = RobotGuiMainWindow()
        self.viewer = self.window.viewer_3d

    def tearDown(self):
        self.window.close()

    def test_time_control_loads_distinct_editable_state(self):
        at_zero = self.viewer.get_current_keyframe()
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertIsNotNone(self.viewer.get_current_keyframe())

        name = self.viewer.robot_state.get_joint_names()[-1]
        old_value = self.viewer.robot_state.get_joint_value(name)
        self.viewer.robot_state.set_joint_value(name, old_value + 0.05)
        self.viewer.update_current_keyframe_from_robot_state()

        self.viewer.set_current_time(0.0)
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), at_zero)
        self.viewer.set_current_time(0.2)
        self.assertAlmostEqual(
            self.viewer.robot_state.get_joint_value(name), old_value + 0.05
        )

    def test_reset_changes_only_active_time(self):
        at_zero = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.2)
        changed = self.viewer.robot_state.get_qpos()
        changed[-1] += 0.1
        self.viewer.robot_state.set_qpos(changed)
        self.viewer.update_current_keyframe_from_robot_state()
        self.viewer.reset_robot_pose()
        np.testing.assert_allclose(
            self.viewer.get_current_keyframe(), self.viewer.robot_model.home_qpos
        )
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.0), at_zero
        )

    def test_reset_is_one_shot_and_does_not_replace_playback(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        first[-1] += 0.05
        second[-1] += 0.10
        self.viewer.set_robot_trajectory([first, second])
        playback_before = [qpos.copy() for qpos in self.viewer.robot_trajectory]

        self.viewer.set_current_time(0.2)
        self.viewer.play_timer.start()
        self.viewer.canvas.gizmo.state = GizmoInteractionState.DRAG_TRANSLATE_FREE
        self.viewer.reset_robot_pose()

        self.assertFalse(self.viewer.play_timer.isActive())
        self.assertEqual(
            self.viewer.canvas.gizmo.state, GizmoInteractionState.NONE
        )
        for actual, expected in zip(
            self.viewer.robot_trajectory, playback_before
        ):
            np.testing.assert_allclose(actual, expected)

        self.viewer._advance_frame()
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), second)
        self.viewer._advance_frame()
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), first)
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.2),
            self.viewer.robot_model.home_qpos,
        )

    def test_pelvis_drag_updates_preview_only_until_accept(self):
        self.viewer.select_target("body", "robot/pelvis", emit=False)
        self.viewer._set_target_to_selected_pose()
        start = self.viewer.last_valid_target_position.copy()
        quaternion = self.viewer.last_valid_target_quaternion.copy()
        self.viewer._on_transform_moved(
            start + np.array([0.01, 0.0, 0.0]), quaternion
        )
        root_address = next(
            iter(self.viewer.robot_model.free_joints_by_body.values())
        ).qpos_address
        self.assertAlmostEqual(self.viewer.committed_state.get_qpos()[root_address], start[0])
        self.assertAlmostEqual(
            self.viewer.get_current_keyframe()[root_address], start[0]
        )
        self.assertAlmostEqual(
            self.viewer.preview_state.get_qpos()[root_address], start[0] + 0.01
        )
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.preview_visible)
        self.viewer.accept_preview()
        self.assertAlmostEqual(
            self.viewer.committed_state.get_qpos()[root_address], start[0] + 0.01
        )
        self.assertAlmostEqual(
            self.viewer.get_current_keyframe()[root_address], start[0] + 0.01
        )
        self.assertFalse(self.viewer.preview_active)

    def test_target_pose_sliders_update_3d_preview_only_until_accept(self):
        self.viewer.select_target("body", "robot/pelvis", emit=False)
        self.viewer._set_target_to_selected_pose()
        self.window.controls.frame_box.setCurrentText("pelvis")
        start_position = self.viewer.last_valid_target_position.copy()
        start_quaternion = self.viewer.last_valid_target_quaternion.copy()
        root_address = next(
            iter(self.viewer.robot_model.free_joints_by_body.values())
        ).qpos_address
        before = self.viewer.committed_state.get_qpos()

        self.window.controls.set_position_values(
            x=float(start_position[0] + 0.01),
            y=float(start_position[1]),
            z=float(start_position[2]),
            roll=0.02,
            pitch=-0.01,
            yaw=0.03,
        )

        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)
        self.assertGreater(
            self.viewer.preview_state.get_qpos()[root_address],
            before[root_address],
        )
        self.assertFalse(
            np.allclose(
                self.viewer.preview_state.get_qpos()[root_address + 3:root_address + 7],
                start_quaternion,
            )
        )
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.preview_visible)

    def test_cancel_discards_preview_without_touching_committed(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[-1]
        old_value = self.viewer.preview_state.get_joint_value(name)
        self.viewer._joint_changed(name, old_value + 0.05)
        self.assertTrue(self.viewer.preview_active)
        self.assertFalse(np.allclose(self.viewer.preview_state.get_qpos(), before))
        self.viewer.cancel_preview()
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.preview_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)

    def test_plan_creates_ghost_path_without_committing(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        preview = self.viewer.preview_state.get_qpos()
        self.viewer.plan_preview()
        self.assertEqual(len(self.viewer.robot_trajectory), 40)
        np.testing.assert_allclose(self.viewer.robot_trajectory[0], before)
        np.testing.assert_allclose(self.viewer.robot_trajectory[-1], preview)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)
        self.assertTrue(self.viewer.preview_active)

    def test_plan_preview_rejects_midpoint_collision_without_publishing(self):
        class MidpointCollisionChecker:
            def __init__(self, address, lower, upper):
                self.address = address
                self.lower = lower
                self.upper = upper

            def get_collisions(self, state):
                value = state.get_qpos()[self.address]
                if self.lower <= value <= self.upper:
                    return [
                        Collision(
                            "geom_a", "geom_b", "body_a", "body_b",
                            -0.01, "self",
                        )
                    ]
                return []

        joint = next(
            item for item in self.viewer.robot_model.joints.values()
            if item.limits is not None
        )
        start_value = self.viewer.preview_state.get_joint_value(joint.name)
        lo, hi = joint.limits
        delta = 0.1 if start_value + 0.1 < hi else -0.1
        goal_value = start_value + delta
        lower = min(start_value, goal_value) + abs(delta) * 0.45
        upper = min(start_value, goal_value) + abs(delta) * 0.55

        self.viewer._joint_changed(joint.name, goal_value)
        self.viewer.collision_checker = MidpointCollisionChecker(
            joint.qpos_address, lower, upper
        )
        self.viewer.plan_preview()

        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIn("Cannot plan preview", self.viewer.status_label.text())
        self.assertIn("collision at path sample", self.viewer.status_label.text())

    def test_plan_preview_rejects_raw_joint_limit_violation_before_clamp(self):
        joint = next(
            item for item in self.viewer.robot_model.joints.values()
            if item.limits is not None
        )
        _, hi = joint.limits
        self.viewer.begin_preview()
        qpos = self.viewer.preview_state.get_qpos()
        qpos[joint.qpos_address] = hi + 0.5
        self.viewer.preview_state.mj_data.qpos[:] = qpos
        self.viewer.collision_checker = None

        self.viewer.plan_preview()

        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIn("Cannot plan preview", self.viewer.status_label.text())
        self.assertIn("outside limits", self.viewer.status_label.text())

    def test_accept_preview_is_timeline_local(self):
        at_zero = self.viewer.committed_state.get_qpos()
        self.viewer.set_current_time(0.2)
        name = self.viewer.preview_state.get_joint_names()[-1]
        value = self.viewer.preview_state.get_joint_value(name) + 0.05
        self.viewer._joint_changed(name, value)
        self.viewer.accept_preview()
        accepted = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.0)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), at_zero)
        self.viewer.set_current_time(0.2)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), accepted)

    def test_joint_slider_edits_preview_not_committed(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[0]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.03
        )
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        self.assertFalse(np.allclose(self.viewer.preview_state.get_qpos(), before))

    def test_timeline_change_discards_unaccepted_preview(self):
        name = self.viewer.preview_state.get_joint_names()[0]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.03
        )
        self.assertTrue(self.viewer.preview_active)
        self.viewer.set_current_time(0.2)
        self.assertFalse(self.viewer.preview_active)
        self.assertFalse(self.viewer.canvas.preview_visible)
        np.testing.assert_allclose(
            self.viewer.preview_state.get_qpos(),
            self.viewer.committed_state.get_qpos(),
        )

    def test_logical_pelvis_selection_maps_to_mujoco_pelvis(self):
        self.window.controls.frame_box.setCurrentText("pelvis")
        self.assertEqual(
            self.viewer._selected_target(), ("body", "robot/pelvis")
        )

    def test_completed_edits_upsert_logical_targets_by_time(self):
        self.window.on_target_pose_drag_finished(0.10, 0.20, 0.80)
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.window.on_target_pose_drag_finished(0.30, 0.20, 0.85)

        frame_name = self.window.controls.frame_box.currentText()
        track = self.window.trajectory.tracks[frame_name]
        self.assertEqual([frame.time for frame in track], [0.0, 0.2])
        self.assertAlmostEqual(track[0].x, 0.10)
        self.assertAlmostEqual(track[1].x, 0.30)

        self.window.controls.time_slider.set_value(0.0)
        self.window.controls.emit_time_changed(0.0)
        self.assertAlmostEqual(self.window.controls.x_slider.value(), 0.10)

    def test_viewer_timeslice_scrubber_loads_editable_time(self):
        self.viewer.timeslice_slider.setValue(20)
        self.viewer._emit_timeslice_slider_time()

        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.2)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.2)

    def test_timeslice_wheel_scroll_updates_active_time(self):
        self.assertEqual(self.viewer.get_current_time(), 0.0)

        event = QWheelEvent(
            QPointF(100, 15),
            QPointF(100, 15),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.viewer.timeslice_slider.wheelEvent(event)

        self.assertEqual(self.viewer.timeslice_slider.value(), 3)
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.03)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.03)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.03)

    def test_timeslice_bar_uses_single_compact_time_display(self):
        self.assertEqual(self.viewer.timeslice_label.text(), "Time")
        self.assertEqual(self.viewer.timeslice_time_input.suffix(), " s")
        self.assertFalse(hasattr(self.viewer, "timeslice_time_label"))
        self.assertEqual(self.viewer.accept_timeslice_button.text(), "Accept")
        self.assertEqual(self.viewer.delete_timeslice_button.text(), "Delete")

    def test_accept_slice_captures_all_logical_targets_from_committed_pose(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        preview_qpos = self.viewer.preview_state.get_qpos()

        self.viewer.accept_timeslice()

        self.assertFalse(self.viewer.preview_active)
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.2), preview_qpos
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.2})
        state = self.viewer.committed_state
        expected_names = []
        for frame_name, (kind, object_name) in self.viewer.frame_bindings.items():
            try:
                state.get_body_pose(object_name, kind)
            except KeyError:
                continue
            expected_names.append(frame_name)

        actual_names = {
            frame.frame_name
            for frame in self.window.trajectory.frames
            if abs(frame.time - 0.2) <= 1e-6
        }
        self.assertEqual(actual_names, set(expected_names))

        for frame_name in expected_names:
            with self.subTest(frame_name=frame_name):
                frame = next(
                    frame for frame in self.window.trajectory.tracks[frame_name]
                    if abs(frame.time - 0.2) <= 1e-6
                )
                kind, object_name = self.viewer.frame_bindings[frame_name]
                position, quaternion = state.get_body_pose(object_name, kind)
                roll, pitch, yaw = quat_to_rpy(quaternion)
                np.testing.assert_allclose(
                    [frame.x, frame.y, frame.z], position, atol=1e-9
                )
                np.testing.assert_allclose(
                    [frame.roll, frame.pitch, frame.yaw],
                    [roll, pitch, yaw],
                    atol=1e-9,
                )

    def test_defined_timeslice_marker_snaps_back_to_slice_time(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        self.viewer.set_current_time(0.0)

        snapped = self.viewer.timeslice_slider.snap_to_nearest_defined_time(0.21)

        self.assertTrue(snapped)
        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.2)

    def test_clicking_undefined_timeslice_groove_changes_time(self):
        self.viewer.timeslice_slider.resize(500, 30)
        self.viewer.timeslice_slider.set_defined_times([0.2])
        pixel = self.viewer.timeslice_slider._time_to_pixel(0.37)

        self.viewer.timeslice_slider.activate_time_at_pixel(pixel)

        self.assertEqual(self.viewer.get_current_time(), 0.37)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.37)

    def test_pressing_current_defined_slice_handle_allows_normal_drag(self):
        self.viewer.timeslice_slider.resize(500, 30)
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer.timeslice_slider.set_defined_times([0.2])
        activated = []
        self.viewer.timeslice_slider.marker_activated.connect(activated.append)
        handle_center = self.viewer.timeslice_slider._handle_rect().center()
        position = QPointF(handle_center)
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        self.viewer.timeslice_slider.mousePressEvent(event)

        self.assertEqual(activated, [])
        self.assertEqual(self.viewer.get_current_time(), 0.2)

    def test_delete_slice_removes_defined_marker_and_logical_targets(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.2})

        self.viewer.delete_timeslice()

        self.assertNotIn(0.2, self.viewer.timeslice_slider.defined_times)
        self.assertFalse(
            any(abs(frame.time - 0.2) <= 1e-6 for frame in self.window.trajectory.frames)
        )
        self.assertIsNone(self.viewer.state_timeline.get_state(0.2))

    def test_clear_trajectory_requires_confirmation(self):
        self.window.on_target_pose_drag_finished(0.10, 0.20, 0.80)
        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.No
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(len(self.window.trajectory.frames), 1)

    def test_clear_trajectory_removes_keyframes_markers_and_playback(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory([first, second])
        self.viewer.show_ghosts.setChecked(True)
        pose_before = self.viewer.committed_state.get_qpos()
        self.assertTrue(self.window.trajectory.frames)
        self.assertTrue(self.viewer.robot_trajectory)
        self.assertTrue(self.viewer.timeslice_slider.defined_times)

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.window.trajectory.frames, [])
        self.assertEqual(self.window.active_index, -1)
        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.robot_trajectory_times, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertEqual(self.viewer.timeslice_slider.defined_times, set())
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), pose_before)

    def test_clear_trajectory_drops_stale_editable_qpos_times(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        joint_name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            joint_name,
            self.viewer.preview_state.get_joint_value(joint_name) + 0.05,
        )
        self.viewer.accept_timeslice()
        stale_qpos = self.viewer.state_timeline.get_state(0.2)
        self.assertFalse(np.allclose(stale_qpos, self.viewer.robot_model.home_qpos))

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.viewer.state_timeline.times(), [0.2])

        self.viewer.reset_robot_pose()
        self.window.controls.time_slider.set_value(0.0)
        self.window.controls.emit_time_changed(0.0)

        np.testing.assert_allclose(
            self.viewer.committed_state.get_qpos(),
            self.viewer.robot_model.home_qpos,
        )
        self.assertFalse(np.allclose(self.viewer.committed_state.get_qpos(), stale_qpos))

    def test_viewer_quick_actions_mirror_common_controls(self):
        self.assertIs(self.viewer.quick_actions_panel.parent(), self.viewer.canvas_workspace)

        generated = []
        self.viewer.generate_requested.connect(lambda: generated.append(True))
        self.viewer.quick_generate_button.click()
        self.assertEqual(generated, [True])

        cleared = []
        self.viewer.clear_trajectory_requested.connect(lambda: cleared.append(True))
        self.viewer.quick_clear_button.click()
        self.assertEqual(cleared, [True])

        self.viewer.quick_show_ghosts.setChecked(True)
        self.assertTrue(self.viewer.show_ghosts.isChecked())
        self.viewer.show_ghosts.setChecked(False)
        self.assertFalse(self.viewer.quick_show_ghosts.isChecked())

        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory([first, second])
        self.viewer.quick_play_button.click()
        self.assertTrue(self.viewer.play_timer.isActive())
        self.assertEqual(self.viewer.quick_play_button.text(), "Pause")
        self.assertEqual(self.viewer.play_button.text(), "Pause")
        self.viewer.quick_play_button.click()
        self.assertFalse(self.viewer.play_timer.isActive())
        self.assertEqual(self.viewer.quick_play_button.text(), "Play")
        self.assertEqual(self.viewer.play_button.text(), "Play")

    def test_render_progress_overlay_waits_for_active_3d_geometry_progress(self):
        self.window.begin_render_progress(
            "Rendering test model",
            "Preparing geometry...",
            viewer=self.viewer,
        )

        overlay = self.window.render_progress_overlay
        self.assertFalse(overlay.isHidden())
        self.assertIs(overlay.parentWidget(), self.window.viewer_3d_stack)
        self.assertEqual(overlay.geometry(), self.window.viewer_3d_stack.rect())
        self.assertLessEqual(
            abs(overlay.card.geometry().center().x() - overlay.rect().center().x()),
            1,
        )
        self.assertLessEqual(
            abs(overlay.card.geometry().center().y() - overlay.rect().center().y()),
            1,
        )
        self.assertFalse(overlay._allow_close)

        self.window.on_viewer_geometry_progress(self.viewer, 2, 8)

        self.assertFalse(overlay.isHidden())
        self.assertEqual(overlay.progress_bar.value(), 25)
        self.assertIn(
            "2/8",
            overlay.detail_label.text(),
        )

        self.window.on_viewer_geometry_progress(self.viewer, 8, 8)

        self.assertTrue(overlay.isHidden())
        self.assertTrue(overlay._allow_close)

    def test_initial_render_progress_overlay_is_deferred_until_window_is_shown(self):
        self.assertTrue(self.window.render_progress_overlay.isHidden())
        self.assertIsNotNone(self.window.pending_initial_render_progress)

        self.window.show()
        for _ in range(3):
            self.app.processEvents()

        self.assertIsNotNone(self.window.pending_initial_render_progress)
        self.assertTrue(self.window.render_progress_overlay.isHidden())

        QTest.qWait(INITIAL_RENDER_PROGRESS_DELAY_MS + 100)
        self.app.processEvents()

        self.assertIsNone(self.window.pending_initial_render_progress)
        overlay = self.window.render_progress_overlay
        self.assertFalse(overlay.isHidden())
        self.assertIs(overlay.parentWidget(), self.window.viewer_3d_stack)
        self.assertEqual(overlay.geometry(), self.window.viewer_3d_stack.rect())

    def test_sidebars_are_fixed_shells_with_collapsible_sections(self):
        self.window.resize(1700, 800)
        self.window.viewer_tabs.setCurrentWidget(self.window.viewer_3d_stack)
        self.window.show()
        self.app.processEvents()
        viewer_identity = id(self.window.viewer_3d)
        model_identity = id(self.window.robot_model_3d.mj_model)
        qpos = self.viewer.robot_state.get_qpos()

        for sidebar in (self.window.left_sidebar, self.window.right_sidebar):
            with self.subTest(sidebar=sidebar):
                self.assertFalse(hasattr(sidebar, "set_collapsed"))
                self.assertFalse(hasattr(sidebar, "toggle_collapsed"))
                self.assertFalse(hasattr(sidebar, "collapsed_width"))
                self.assertFalse(sidebar.isHidden())

        self.assertFalse(hasattr(self.viewer, "controls_sidebar"))
        self.assertFalse(hasattr(self.viewer, "viewer_splitter"))
        self.assertIsNone(self.window.right_sidebar_content.current_context_widget())
        self.assertLessEqual(self.window.left_sidebar.maximumWidth(), 250)
        self.assertLessEqual(self.window.right_sidebar.maximumWidth(), 270)
        self.assertEqual(self.window.status_panel.maximumWidth(), 244)
        left_titles = [
            section.title for section in self.window.left_sidebar_content.sections
        ]
        right_titles = [
            section.title for section in self.window.right_sidebar_content.sections
        ]
        self.assertEqual(
            left_titles,
            ["Robot", "Target", "Pose", "Trajectory", "Advanced IK", "View"],
        )
        self.assertEqual(
            right_titles,
            ["Status"],
        )
        for section in (
            self.window.left_sidebar_content.sections
            + self.window.right_sidebar_content.sections
        ):
            self.assertFalse(section.content.isVisible())
        self.assertTrue(self.window.viewer_tabs.tabBar().isHidden())
        self.assertEqual(self.window.viewer_tabs.tabText(0), "3D Pose")
        self.assertIs(
            self.window.viewer_tabs.widget(0), self.window.viewer_3d_stack
        )
        view_buttons = [
            button.text()
            for button in self.window.left_sidebar_content.view_panel.findChildren(
                QPushButton
            )
        ]
        self.assertEqual(
            view_buttons,
            ["3D Pose", "2D Side View", "2D Skeleton", "Simulation"],
        )
        self.assertTrue(
            self.window.model_source_label.text().startswith("Model source:")
        )
        robot_labels = [
            label.text()
            for label in self.viewer.robot_context_widget().findChildren(QLabel)
        ]
        self.assertFalse(
            any(text.startswith("Model:") for text in robot_labels)
        )
        trajectory_widgets = set(
            self.window.controls.trajectory_panel.findChildren(QWidget)
        )
        for removed_widget in (
            self.window.controls.time_slider,
            self.window.controls.corner_smoothing_slider,
            self.window.controls.add_button,
            self.window.controls.update_button,
            self.window.controls.delete_button,
            self.window.controls.generate_button,
            self.viewer.generate_button,
            self.viewer.play_button,
        ):
            self.assertNotIn(removed_widget, trajectory_widgets)
        self.assertIs(
            self.window.controls.corner_smoothing_slider.parent(),
            self.viewer.timeslice_editor,
        )
        self.assertIs(
            self.viewer.quick_clear_button.parent(), self.viewer.quick_actions_panel
        )
        for section in self.window.left_sidebar_content.sections:
            section.set_expanded(True)
            self.app.processEvents()
            self.assertLessEqual(section.maximumWidth(), 250)
            self.assertLessEqual(section.sizeHint().width(), 250)
            self.assertLessEqual(section.content.sizeHint().width(), 250)
            section.set_expanded(False)
        self.assertIsNone(self.viewer.status_label.parent())
        self.assertIsNone(self.viewer.timeline_state_label.parent())
        self.assertIsNone(self.viewer.root_pose_label.parent())
        self.viewer.status_label.setText("Viewer status moved")
        self.viewer._update_timeline_label()
        self.viewer._update_root_pose_label()
        self.app.processEvents()
        self.assertEqual(
            self.window.viewer_status_label.text(),
            "Status: Viewer status moved",
        )
        self.assertTrue(self.window.viewer_time_label.text().startswith("Time:"))
        self.assertTrue(self.window.viewer_root_pose_label.text().startswith("Root:"))
        preview_section = next(
            section for section in self.window.left_sidebar_content.sections
            if section.title == "Advanced IK"
        )
        preview_section.set_expanded(True)
        ik_tabs = self.viewer.preview_ik_context_widget().findChild(QTabWidget)
        ik_tabs.setCurrentIndex(1)
        self.app.processEvents()
        visible_scroll_areas = [
            area for area in self.viewer.preview_ik_context_widget().findChildren(
                QScrollArea
            )
            if area.isVisible()
        ]
        self.assertTrue(visible_scroll_areas)
        for area in visible_scroll_areas:
            self.assertEqual(area.horizontalScrollBar().maximum(), 0)
            self.assertFalse(area.horizontalScrollBar().isVisible())
            self.assertLessEqual(area.widget().width(), area.viewport().width())
        preview_section.set_expanded(False)
        all_titles = left_titles + right_titles
        for removed_title in (
            "Project", "Editors", "Display", "Selection", "Properties",
            "3D Selection", "Export / Import", "Playback / Ghosts",
            "Joints / IK", "Transform", "Preview / IK",
        ):
            self.assertNotIn(removed_title, all_titles)
        self.assertLessEqual(self.window.controls.table.maximumHeight(), 180)
        self.assertIs(
            self.window.controls.robot_context_widget(),
            self.viewer.robot_context_widget(),
        )
        self.assertIs(
            self.window.controls.selection_context_widget(),
            self.viewer.selection_context_widget(),
        )
        self.assertIs(
            self.window.controls.trajectory_context_widget(),
            self.viewer.trajectory_context_widget(),
        )
        self.assertIs(
            self.window.controls.display_context_widget(),
            self.viewer.display_context_widget(),
        )
        self.assertIs(
            self.window.controls.preview_ik_context_widget(),
            self.viewer.preview_ik_context_widget(),
        )

        self.assertEqual(id(self.window.viewer_3d), viewer_identity)
        self.assertEqual(id(self.window.robot_model_3d.mj_model), model_identity)
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), qpos)

    def test_model_colors_toggle_defaults_on_without_mutating_materials(self):
        before = self.window.robot_model_3d.mj_model.mat_rgba.copy()
        self.window.viewer_tabs.setCurrentWidget(self.window.viewer_3d_stack)
        self.window.update_editor_context()
        self.assertTrue(self.viewer.model_colors_box.isChecked())
        self.assertTrue(self.viewer.canvas.use_model_colors)
        display_layout = self.viewer.display_context_widget().layout()
        display_widgets = [
            display_layout.itemAt(index).widget()
            for index in range(display_layout.count())
        ]
        self.assertEqual(display_widgets[:3], [
            self.viewer.model_colors_box,
            self.window.controls.show_lines_box,
            self.viewer.show_ghosts,
        ])
        self.viewer.model_colors_box.setChecked(False)
        self.assertFalse(self.viewer.canvas.use_model_colors)
        self.viewer.model_colors_box.setChecked(True)
        np.testing.assert_allclose(
            self.window.robot_model_3d.mj_model.mat_rgba, before
        )

    def test_controls_store_full_rpy_in_keyframe_and_table(self):
        self.window.controls.set_position_values(
            roll=0.12, pitch=-0.23, yaw=0.34,
            emit_pose_changed=False,
        )
        self.window.on_add_keyframe()
        frame = self.window.trajectory.frames[0]
        self.assertAlmostEqual(frame.roll, 0.12)
        self.assertAlmostEqual(frame.pitch, -0.23)
        self.assertAlmostEqual(frame.yaw, 0.34)
        headers = [
            self.window.controls.table.horizontalHeaderItem(index).text()
            for index in range(self.window.controls.table.columnCount())
        ]
        self.assertEqual(headers[-3:], ["roll", "pitch", "yaw"])

    def test_pose_controls_use_compact_static_labels(self):
        controls = self.window.controls
        expected = {
            controls.x_slider: "X [m]",
            controls.y_slider: "Y [m]",
            controls.z_slider: "Z [m]",
            controls.roll_slider: "Roll [rad]",
            controls.pitch_slider: "Pitch [rad]",
            controls.yaw_slider: "Yaw [rad]",
        }

        for slider, label in expected.items():
            with self.subTest(label=label):
                self.assertEqual(slider.label.text(), label)
                self.assertNotIn(":", slider.label.text())
                self.assertLessEqual(slider.label.maximumWidth(), 86)

    def test_corner_smoothing_slider_maps_percent_to_fraction(self):
        controls = self.window.controls

        self.assertEqual(controls.corner_smoothing_slider.label.text(), "Smoothing [%]")
        self.assertAlmostEqual(controls.corner_smoothing(), 0.0)

        controls.corner_smoothing_slider.set_value(50.0)

        self.assertAlmostEqual(controls.corner_smoothing(), 0.5)

    def test_refresh_display_passes_smoothing_to_trajectory_line_views(self):
        captured = {}

        def capture(name):
            def update_scene(*args, **kwargs):
                captured[name] = kwargs["trajectory_smoothing"]
            return update_scene

        self.window.viewer_2d.update_scene = capture("viewer_2d")
        self.window.viewer_3d.update_scene = capture("viewer_3d")
        self.window.viewer_2d_stickman.update_scene = capture("viewer_2d_stickman")
        self.window.controls.corner_smoothing_slider.set_value(75.0)

        self.window.refresh_display()

        self.assertEqual(captured, {
            "viewer_2d": 0.75,
            "viewer_3d": 0.75,
            "viewer_2d_stickman": 0.75,
        })

    def test_accepted_3d_rotation_is_upserted_into_keyframe(self):
        self.window.on_3d_target_frame_changed("left_hand")
        self.viewer._set_target_to_selected_pose()
        position = self.viewer.last_valid_target_position.copy()
        start_rpy = quat_to_rpy(self.viewer.last_valid_target_quaternion)
        target_rpy = (
            start_rpy[0] + 0.05,
            start_rpy[1] - 0.03,
            start_rpy[2] + 0.04,
        )
        self.viewer._on_transform_moved(position, rpy_to_quat(*target_rpy))
        self.assertTrue(self.viewer.preview_active)
        self.viewer.accept_preview()

        frame = self.window.trajectory.tracks["left_hand"][0]
        solved_rpy = quat_to_rpy(self.viewer.last_valid_target_quaternion)
        self.assertAlmostEqual(frame.roll, solved_rpy[0], delta=0.011)
        self.assertAlmostEqual(frame.pitch, solved_rpy[1], delta=0.011)
        self.assertAlmostEqual(frame.yaw, solved_rpy[2], delta=0.011)

    def test_load_edit_accept_and_save_headerless_qpos_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            expected = self.viewer.robot_model.home_qpos.copy()
            np.savetxt(source, expected[None, :], delimiter=",")
            self.viewer.load_qpos_csv(source)
            np.testing.assert_allclose(
                self.viewer.committed_state.get_qpos(), expected
            )
            np.testing.assert_allclose(self.viewer.get_current_keyframe(), expected)

            joint_name = self.viewer.preview_state.get_joint_names()[-1]
            self.viewer._joint_changed(
                joint_name,
                self.viewer.preview_state.get_joint_value(joint_name) + 0.02,
            )
            self.viewer.accept_preview()

            output = self.viewer.save_qpos_csv(Path(directory) / "updated")
            self.assertEqual(output.suffix, ".csv")
            saved = np.loadtxt(output, delimiter=",")
        np.testing.assert_allclose(saved, self.viewer.committed_state.get_qpos())
        self.assertFalse(np.allclose(saved, expected))

    def test_save_timeline_as_headerless_time_plus_qpos_csv(self):
        at_zero = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.2)
        changed = self.viewer.committed_state.get_qpos()
        changed[-1] += 0.05
        self.viewer.set_robot_state_for_current_time(changed)
        self.viewer.update_current_keyframe_from_robot_state()

        with tempfile.TemporaryDirectory() as directory:
            output = self.viewer.save_trajectory_csv(
                Path(directory) / "timed_trajectory"
            )
            saved = np.loadtxt(output, delimiter=",")
            _, rows = load_trajectory_csv(
                output, qpos_width=self.viewer.robot_model.mj_model.nq
            )

        self.assertEqual(output.suffix, ".csv")
        self.assertEqual(saved.shape, (2, self.viewer.robot_model.mj_model.nq + 1))
        np.testing.assert_allclose(saved[:, 0], [0.0, 0.2])
        np.testing.assert_allclose(rows[0][RAW_QPOS_KEY], at_zero)
        np.testing.assert_allclose(rows[1][RAW_QPOS_KEY], changed)

    def test_save_generated_trajectory_uses_backend_times(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05

        self.viewer.load_backend_states([
            PythonRobotConfiguration(time=0.0, qpos=first),
            PythonRobotConfiguration(time=0.1, qpos=second),
        ])

        with tempfile.TemporaryDirectory() as directory:
            output = self.viewer.save_trajectory_csv(
                Path(directory) / "generated_trajectory"
            )
            saved = np.loadtxt(output, delimiter=",")
            _, rows = load_trajectory_csv(
                output, qpos_width=self.viewer.robot_model.mj_model.nq
            )

        self.assertEqual(saved.shape, (2, self.viewer.robot_model.mj_model.nq + 1))
        np.testing.assert_allclose(saved[:, 0], [0.0, 0.1])
        np.testing.assert_allclose(rows[0][RAW_QPOS_KEY], first)
        np.testing.assert_allclose(rows[1][RAW_QPOS_KEY], second)

    def test_load_headerless_time_plus_qpos_trajectory_csv(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.07

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.csv"
            np.savetxt(
                source,
                np.vstack(
                    [
                        np.concatenate(([0.0], first)),
                        np.concatenate(([0.15], second)),
                    ]
                ),
                delimiter=",",
            )
            self.viewer.load_trajectory_csv(source)
            mujoco_panel_path = self.window.viewer_3d_mujoco.trajectory_csv_path
            mujoco_panel_times = list(self.window.viewer_3d_mujoco.trajectory_times)

        self.assertEqual(len(self.viewer.robot_trajectory), 2)
        np.testing.assert_allclose(self.viewer.robot_trajectory_times, [0.0, 0.15])
        np.testing.assert_allclose(self.viewer.robot_trajectory[0], first)
        np.testing.assert_allclose(self.viewer.robot_trajectory[1], second)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), first)
        np.testing.assert_allclose(self.viewer.state_timeline.times(), [0.0, 0.15])
        self.assertEqual(
            len(self.window.trajectory.frames),
            2 * len(self.window.editable_logical_frame_names()),
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.0, 0.15})
        self.assertEqual(mujoco_panel_path, source)
        np.testing.assert_allclose(mujoco_panel_times, [0.0, 0.15])

    def test_loaded_trajectory_csv_import_interval_downsamples_editable_keyframes(self):
        base = self.viewer.robot_model.home_qpos.copy()
        rows = []
        for index in range(11):
            qpos = base.copy()
            qpos[-1] += index * 0.01
            rows.append(np.concatenate(([index * 0.01], qpos)))

        self.viewer.trajectory_import_dt.setValue(0.05)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dense_trajectory.csv"
            np.savetxt(source, np.vstack(rows), delimiter=",")
            self.viewer.load_trajectory_csv(source)

        editable_times = sorted({frame.time for frame in self.window.trajectory.frames})
        self.assertEqual(len(self.viewer.robot_trajectory), 11)
        np.testing.assert_allclose(self.viewer.robot_trajectory_times, np.arange(0, 0.11, 0.01))
        np.testing.assert_allclose(editable_times, [0.0, 0.05, 0.10])
        self.assertEqual(
            len(self.window.trajectory.frames),
            3 * len(self.window.editable_logical_frame_names()),
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.0, 0.05, 0.10})

    def test_loaded_trajectory_csv_can_be_cleared_from_keyframe_controls(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.07

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.csv"
            np.savetxt(
                source,
                np.vstack(
                    [
                        np.concatenate(([0.0], first)),
                        np.concatenate(([0.15], second)),
                    ]
                ),
                delimiter=",",
            )
            self.viewer.load_trajectory_csv(source)

        self.assertTrue(self.window.trajectory.frames)
        self.assertTrue(self.viewer.robot_trajectory)

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.window.trajectory.frames, [])
        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.timeslice_slider.defined_times, set())

    def test_load_qpos_rejects_wrong_value_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 36 qpos values"):
                self.viewer.load_qpos_csv(source)


if __name__ == "__main__":
    unittest.main()

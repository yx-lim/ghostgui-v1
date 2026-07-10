import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.collision_checker import Collision
from gui.main_window import RobotGuiMainWindow
from gui.transform_gizmo import GizmoInteractionState
from gui.trajectory import quat_to_rpy, rpy_to_quat


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

    def test_sidebars_collapse_without_recreating_viewer_or_state(self):
        self.window.resize(1700, 800)
        self.window.viewer_tabs.setCurrentWidget(self.window.viewer_3d_stack)
        self.window.show()
        self.app.processEvents()
        viewer_identity = id(self.window.viewer_3d)
        model_identity = id(self.window.robot_model_3d.mj_model)
        qpos = self.viewer.robot_state.get_qpos()

        for sidebar in (self.window.left_sidebar, self.window.right_sidebar):
            with self.subTest(sidebar=sidebar.title):
                before_width = sidebar.width()
                sidebar.set_collapsed(True)
                self.app.processEvents()
                self.app.processEvents()
                self.assertTrue(sidebar.content.isHidden())
                self.assertEqual(sidebar.width(), sidebar.collapsed_width)

                sidebar.set_collapsed(False)
                self.app.processEvents()
                self.app.processEvents()
                self.assertFalse(sidebar.content.isHidden())
                self.assertAlmostEqual(sidebar.width(), before_width, delta=2)

        controls_sidebar = self.viewer.controls_sidebar
        controls_before = controls_sidebar.width()
        controls_sidebar.set_collapsed(True)
        self.app.processEvents()
        self.app.processEvents()
        self.assertTrue(controls_sidebar.content.isHidden())
        self.assertEqual(
            self.viewer.viewer_splitter.sizes()[1],
            controls_sidebar.collapsed_width,
        )
        controls_sidebar.set_collapsed(False)
        self.app.processEvents()
        self.app.processEvents()
        self.assertFalse(controls_sidebar.content.isHidden())
        self.assertAlmostEqual(
            controls_sidebar.width(), controls_before, delta=2
        )

        self.assertEqual(id(self.window.viewer_3d), viewer_identity)
        self.assertEqual(id(self.window.robot_model_3d.mj_model), model_identity)
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), qpos)

    def test_model_colors_toggle_defaults_on_without_mutating_materials(self):
        before = self.window.robot_model_3d.mj_model.mat_rgba.copy()
        self.assertTrue(self.viewer.model_colors_box.isChecked())
        self.assertTrue(self.viewer.canvas.use_model_colors)
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

    def test_load_qpos_rejects_wrong_value_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 36 qpos values"):
                self.viewer.load_qpos_csv(source)


if __name__ == "__main__":
    unittest.main()

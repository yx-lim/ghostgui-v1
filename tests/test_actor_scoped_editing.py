import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.scene import Transform
from core.trajectory import rpy_to_quat
from gui.main_window import RobotGuiMainWindow


class ActorScopedEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config_dir = tempfile.TemporaryDirectory()
        self.config_patch = patch.dict(
            os.environ,
            {
                "GHOSTGUI_CONFIG_DIR": str(
                    Path(self.config_dir.name) / "config"
                ),
                "GHOSTGUI_PROJECTS_DIR": str(
                    Path(self.config_dir.name) / "projects"
                ),
            },
        )
        self.config_patch.start()
        self.window = RobotGuiMainWindow()

    def tearDown(self):
        self.window.set_project_dirty(False)
        self.window.close()
        self.config_patch.stop()
        self.config_dir.cleanup()

    def select_actor(self, actor_id, frame_id=None):
        if frame_id is not None:
            self.window.scene.select_actor(actor_id, frame_id=frame_id)

        tree = self.window.scene_tree
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole) == actor_id:
                tree.setCurrentItem(item)
                self.window.on_scene_tree_selection_changed()
                QApplication.processEvents()
                return
        self.fail(f"Scene tree did not contain actor {actor_id}")

    def set_actor_edit_state(self, actor_id, frame_name, target_x, qpos_delta):
        self.select_actor(actor_id, frame_id=frame_name)
        self.window.controls.frame_box.setCurrentText(frame_name)
        self.window.on_target_pose_drag_finished(
            target_x, 0.20, 0.80, 0.0, 0.0, 0.0
        )

        viewer = self.window.viewer_3d
        qpos = viewer.get_current_keyframe().copy()
        qpos[-1] += qpos_delta
        viewer.set_robot_state_for_current_time(qpos)
        viewer.update_current_keyframe_from_robot_state()
        return qpos

    def expected_gizmo_pose(self, frame_name, transform):
        viewer = self.window.viewer_3d
        kind, name = viewer.frame_bindings[frame_name]
        local_position, local_quaternion = viewer.committed_state.get_body_pose(
            name, kind
        )
        world_quaternion = np.asarray(transform.quaternion, dtype=float)
        local_quaternion = np.asarray(local_quaternion, dtype=float)
        vector = np.asarray(local_position, dtype=float)
        vector_quaternion = np.array([0.0, *vector])
        inverse_world = world_quaternion * np.array([1.0, -1.0, -1.0, -1.0])

        def multiply(left, right):
            lw, lx, ly, lz = left
            rw, rx, ry, rz = right
            return np.array(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ]
            )

        rotated = multiply(
            multiply(world_quaternion, vector_quaternion), inverse_world
        )[1:]
        position = np.asarray(transform.position, dtype=float) + rotated
        quaternion = multiply(world_quaternion, local_quaternion)
        return position, quaternion / np.linalg.norm(quaternion)

    def assert_equivalent_quaternion(self, actual, expected):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        self.assertAlmostEqual(abs(float(np.dot(actual, expected))), 1.0, places=6)

    def test_object_selection_redirects_gizmo_editing_away_from_robot(self):
        robot_id = self.window.editor_robot_actor_id
        box = self.window.add_scene_object(
            name="Fixture",
            transform=Transform(position=(0.35, -0.10, 0.25)),
        )

        self.select_actor(robot_id, frame_id="left_hand")
        self.select_actor(box.id)

        canvas = self.window.viewer_3d.canvas
        self.assertEqual(self.window.scene.selection.actor_id, box.id)
        self.assertEqual(self.window.editor_robot_actor_id, robot_id)
        self.assertEqual(self.window.scene_edit_actor_id(), box.id)
        self.assertEqual(canvas.scene_edit_actor_id, box.id)
        np.testing.assert_allclose(canvas.gizmo.position, (0.35, -0.10, 0.25))
        self.assertFalse(self.window.controls.target_panel.isEnabled())
        self.assertFalse(self.window.controls.preview_ik_panel.isEnabled())
        self.assertFalse(self.window.viewer_3d.quick_actions_panel.isEnabled())
        self.assertFalse(self.window.viewer_3d.timeslice_editor.isEnabled())

    def test_selecting_robot_after_object_places_gizmo_on_selected_logical_frame(self):
        robot_id = self.window.editor_robot_actor_id
        box = self.window.add_scene_object(name="Fixture")

        self.select_actor(box.id)
        self.select_actor(robot_id, frame_id="left_hand")

        expected_position, expected_quaternion = self.expected_gizmo_pose(
            "left_hand", Transform.identity()
        )
        canvas = self.window.viewer_3d.canvas
        self.assertIsNone(canvas.scene_edit_actor_id)
        np.testing.assert_allclose(canvas.gizmo.position, expected_position)
        self.assert_equivalent_quaternion(canvas.gizmo.quaternion, expected_quaternion)

    def test_viewport_robot_selection_includes_actor_id(self):
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        body_name = self.window.viewer_3d.frame_bindings["pelvis"][1]

        self.window.viewer_3d.canvas.scene_robot_body_double_clicked.emit(
            second.id, body_name
        )
        QApplication.processEvents()

        self.assertEqual(self.window.scene.selection.actor_id, second.id)
        self.assertEqual(self.window.editor_robot_actor_id, second.id)
        self.assertEqual(self.window.scene.selection.frame_id, "pelvis")

    def test_viewport_frame_selection_preserves_same_robot_multi_limb_preview(self):
        viewer = self.window.viewer_3d
        actor_id = self.window.editor_robot_actor_id
        left = viewer.frame_bindings["left_hand"]
        right = viewer.frame_bindings["right_hand"]
        right_body = next(
            body_name
            for body_name in viewer.robot_model.body_names
            if viewer.robot_model.logical_frame_for_body(body_name) == "right_hand"
        )

        viewer.select_target(*left, emit=False)
        viewer._set_target_to_selected_pose()
        left_position = viewer.last_valid_target_position.copy()
        left_quaternion = viewer.last_valid_target_quaternion.copy()
        viewer._on_transform_moved(
            left_position + np.array([0.04, 0.03, 0.0]),
            left_quaternion,
        )
        preview_before_selection = viewer.preview_state.get_qpos()
        pinned_left = viewer.preview_state.get_body_pose(left[1], left[0])[0]

        viewer.canvas.scene_robot_body_double_clicked.emit(actor_id, right_body)
        QApplication.processEvents()

        self.assertEqual(self.window.scene.selection.actor_id, actor_id)
        self.assertEqual(self.window.scene.selection.frame_id, "right_hand")
        self.assertEqual(self.window.controls.frame_box.currentText(), "right_hand")
        self.assertTrue(viewer.preview_active)
        self.assertEqual(set(viewer.pinned_frame_targets), {"left_hand"})
        np.testing.assert_allclose(
            viewer.preview_state.get_qpos(),
            preview_before_selection,
        )

        right_position = viewer.last_valid_target_position.copy()
        right_quaternion = viewer.last_valid_target_quaternion.copy()
        viewer._on_transform_moved(
            right_position + np.array([0.04, -0.03, 0.0]),
            right_quaternion,
        )

        final_left = viewer.preview_state.get_body_pose(left[1], left[0])[0]
        self.assertLessEqual(
            np.linalg.norm(final_left - pinned_left),
            viewer.ik_position_tolerance.value(),
        )
        self.assertEqual(
            set(viewer.pinned_frame_targets),
            {"left_hand", "right_hand"},
        )

    def test_scene_tree_reselection_does_not_reload_active_robot_preview(self):
        viewer = self.window.viewer_3d
        binding = viewer.frame_bindings["left_hand"]
        viewer.select_target(*binding, emit=False)
        viewer._set_target_to_selected_pose()
        position = viewer.last_valid_target_position.copy()
        quaternion = viewer.last_valid_target_quaternion.copy()
        viewer._on_transform_moved(
            position + np.array([0.02, 0.0, 0.0]),
            quaternion,
        )
        preview_qpos = viewer.preview_state.get_qpos()

        self.window.on_scene_tree_selection_changed()
        QApplication.processEvents()

        self.assertTrue(viewer.preview_active)
        self.assertEqual(set(viewer.pinned_frame_targets), {"left_hand"})
        np.testing.assert_allclose(viewer.preview_state.get_qpos(), preview_qpos)

    def test_ray_picking_distinguishes_an_offset_second_robot(self):
        first_id = self.window.editor_robot_actor_id
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        self.select_actor(first_id)
        self.window.refresh_display(apply_stickman_frame=False)

        canvas = self.window.viewer_3d.canvas
        state = canvas.scene_robot_states[second.id]
        geom_id = next(
            geom_id
            for geom_id in canvas.render_geom_ids(state.mj_model)
            if int(state.mj_model.geom_bodyid[geom_id]) != 0
        )
        transform = self.window.scene.actors.require(second.id).world_transform
        rotation = canvas._quaternion_rotation_matrix(transform.quaternion)
        center = (
            rotation @ np.asarray(state.mj_data.geom_xpos[geom_id], dtype=float)
            + np.asarray(transform.position, dtype=float)
        )

        hit = canvas.pick_scene_robot_body_from_ray(center, (1.0, 0.0, 0.0))

        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], second.id)

    def test_hidden_active_robot_is_not_pickable(self):
        robot_id = self.window.editor_robot_actor_id
        canvas = self.window.viewer_3d.canvas
        state = self.window.viewer_3d.committed_state
        geom_id = next(
            geom_id
            for geom_id in canvas.render_geom_ids(state.mj_model)
            if int(state.mj_model.geom_bodyid[geom_id]) != 0
        )
        center = np.asarray(state.mj_data.geom_xpos[geom_id], dtype=float)
        self.window.set_scene_actor_visibility(robot_id, False)

        hit = canvas.pick_scene_robot_body_from_ray(center, (1.0, 0.0, 0.0))

        self.assertIsNone(hit)

    def test_same_model_robot_actors_keep_independent_edit_contexts(self):
        first_id = self.window.editor_robot_actor_id
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )

        first_qpos = self.set_actor_edit_state(
            first_id, "left_hand", 0.11, 0.03
        )
        second_qpos = self.set_actor_edit_state(
            second.id, "right_hand", 0.72, -0.04
        )

        self.select_actor(first_id)
        self.assertEqual(self.window.controls.frame_box.currentText(), "left_hand")
        self.assertAlmostEqual(
            self.window.trajectory.targets_at_time(0.0)["left_hand"].x,
            0.11,
        )
        np.testing.assert_allclose(
            self.window.viewer_3d.get_current_keyframe(), first_qpos
        )

        self.select_actor(second.id)
        self.assertEqual(self.window.controls.frame_box.currentText(), "right_hand")
        self.assertAlmostEqual(
            self.window.trajectory.targets_at_time(0.0)["right_hand"].x,
            0.72,
        )
        np.testing.assert_allclose(
            self.window.viewer_3d.get_current_keyframe(), second_qpos
        )

        first_trajectory = self.window.scene.tracks.robot_trajectory(first_id)
        second_trajectory = self.window.scene.tracks.robot_trajectory(second.id)
        first_targets = first_trajectory.targets_at_time(0.0)
        second_targets = second_trajectory.targets_at_time(0.0)
        self.assertAlmostEqual(first_targets["left_hand"].x, 0.11)
        self.assertAlmostEqual(second_targets["right_hand"].x, 0.72)
        self.assertNotIn("right_hand", first_targets)
        self.assertNotIn("left_hand", second_targets)
        self.assertFalse(np.allclose(first_qpos, second_qpos))

    def test_robot_world_transform_is_composed_into_frame_gizmo_pose(self):
        robot_id = self.window.editor_robot_actor_id
        transform = Transform(
            position=(0.9, -0.4, 0.25),
            quaternion=tuple(rpy_to_quat(0.0, 0.0, np.pi / 2.0)),
        )
        self.window.scene.actors.require(robot_id).world_transform = transform

        self.select_actor(robot_id, frame_id="left_hand")

        expected_position, expected_quaternion = self.expected_gizmo_pose(
            "left_hand", transform
        )
        canvas = self.window.viewer_3d.canvas
        np.testing.assert_allclose(canvas.gizmo.position, expected_position)
        self.assert_equivalent_quaternion(canvas.gizmo.quaternion, expected_quaternion)

    def test_frame_change_preserves_offset_additional_robot_pose(self):
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        transform = Transform(
            position=(0.85, -0.35, 0.20),
            quaternion=tuple(rpy_to_quat(0.0, 0.0, np.pi / 3.0)),
        )
        second.world_transform = transform
        self.select_actor(second.id, frame_id="left_hand")

        viewer = self.window.viewer_3d
        qpos = viewer.committed_state.get_qpos()
        qpos[0] += 0.30
        viewer.set_robot_state_for_current_time(qpos)
        viewer.update_current_keyframe_from_robot_state()

        self.window.controls.frame_box.setCurrentText("right_hand")
        QApplication.processEvents()
        self.window.refresh_display(apply_stickman_frame=False)

        np.testing.assert_allclose(viewer.committed_state.get_qpos(), qpos)
        np.testing.assert_allclose(viewer.state_timeline.get_state(0.0), qpos)
        self.assertEqual(second.world_transform, transform)
        kind, name = viewer.frame_bindings["right_hand"]
        position, quaternion = viewer.committed_state.get_body_pose(name, kind)
        np.testing.assert_allclose(
            [
                self.window.controls.x_slider.value(),
                self.window.controls.y_slider.value(),
                self.window.controls.z_slider.value(),
            ],
            position,
            atol=1e-3,
        )
        expected_position, expected_quaternion = self.expected_gizmo_pose(
            "right_hand", transform
        )
        np.testing.assert_allclose(viewer.canvas.gizmo.position, expected_position)
        self.assert_equivalent_quaternion(
            viewer.canvas.gizmo.quaternion,
            expected_quaternion,
        )

    def test_frame_change_preserves_live_preview_pose(self):
        robot_id = self.window.editor_robot_actor_id
        self.select_actor(robot_id, frame_id="left_hand")
        viewer = self.window.viewer_3d
        committed_qpos = viewer.committed_state.get_qpos()
        preview_qpos = committed_qpos.copy()
        preview_qpos[1] += 0.25
        viewer.preview_state.set_qpos(preview_qpos)
        viewer.preview_active = True
        viewer.canvas.set_preview_visible(True)

        self.window.controls.frame_box.setCurrentText("right_hand")
        QApplication.processEvents()

        self.assertTrue(viewer.preview_active)
        self.assertTrue(viewer.canvas.preview_visible)
        np.testing.assert_allclose(
            viewer.committed_state.get_qpos(),
            committed_qpos,
        )
        np.testing.assert_allclose(viewer.preview_state.get_qpos(), preview_qpos)
        kind, name = viewer.frame_bindings["right_hand"]
        position, _quaternion = viewer.preview_state.get_body_pose(name, kind)
        np.testing.assert_allclose(
            [
                self.window.controls.x_slider.value(),
                self.window.controls.y_slider.value(),
                self.window.controls.z_slider.value(),
            ],
            position,
            atol=1e-3,
        )

    def test_project_roundtrip_restores_each_actor_editor_state(self):
        first_id = self.window.editor_robot_actor_id
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        first_qpos = self.set_actor_edit_state(
            first_id, "left_hand", 0.18, 0.02
        )
        first_qpos_t1 = first_qpos.copy()
        first_qpos_t1[-1] += 0.05
        self.window.viewer_3d.state_timeline.set_state(1.0, first_qpos_t1)
        second_qpos = self.set_actor_edit_state(
            second.id, "right_hand", 0.64, -0.03
        )

        with tempfile.TemporaryDirectory() as directory:
            project = self.window.create_project_at(
                Path(directory) / "actors.ghostgui", "actors"
            )
            self.assertTrue(
                self.window.save_current_project(
                    show_status=False, capture_snapshot=False
                )
            )
            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ):
                self.assertTrue(self.window.open_project_path(project.root_dir))

        self.window.scene.timeline.current_time = 0.0
        inactive_first = self.window.scene_robot_render_states()[first_id]
        np.testing.assert_allclose(inactive_first.get_qpos(), first_qpos)
        self.window.scene.timeline.current_time = 1.0
        inactive_first = self.window.scene_robot_render_states()[first_id]
        np.testing.assert_allclose(inactive_first.get_qpos(), first_qpos_t1)
        self.window.scene.timeline.current_time = 0.0

        self.select_actor(first_id)
        self.assertEqual(self.window.controls.frame_box.currentText(), "left_hand")
        self.assertAlmostEqual(
            self.window.trajectory.targets_at_time(0.0)["left_hand"].x,
            0.18,
        )
        np.testing.assert_allclose(
            self.window.viewer_3d.get_current_keyframe(), first_qpos
        )

        self.select_actor(second.id)
        self.assertEqual(self.window.controls.frame_box.currentText(), "right_hand")
        self.assertAlmostEqual(
            self.window.trajectory.targets_at_time(0.0)["right_hand"].x,
            0.64,
        )
        np.testing.assert_allclose(
            self.window.viewer_3d.get_current_keyframe(), second_qpos
        )

        first_targets = self.window.scene.tracks.robot_trajectory(
            first_id
        ).targets_at_time(0.0)
        second_targets = self.window.scene.tracks.robot_trajectory(
            second.id
        ).targets_at_time(0.0)
        self.assertNotIn("right_hand", first_targets)
        self.assertNotIn("left_hand", second_targets)

    def test_render_sampling_does_not_create_qpos_keyframes(self):
        first_id = self.window.editor_robot_actor_id
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        self.select_actor(second.id)
        timeline = self.window.viewer_3d.state_timeline
        start = timeline.get_state(0.0)
        end = start.copy()
        end[-1] += 0.1
        timeline.set_state(1.0, end)
        self.select_actor(first_id)
        self.window.scene.timeline.current_time = 0.5

        before = timeline.times()
        self.window.scene_robot_render_states()

        self.assertEqual(timeline.times(), before)
        self.assertIsNone(timeline.get_state(0.5))

    def test_undo_redo_added_robot_rebinds_actor_session(self):
        first_id = self.window.editor_robot_actor_id
        second = self.window.add_scene_robot(
            self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        self.assertEqual(self.window.editor_robot_actor_id, second.id)

        self.window.undo_last_action()
        self.assertNotIn(second.id, self.window.scene.actors.actors)
        self.assertEqual(self.window.editor_robot_actor_id, first_id)
        self.assertEqual(self.window.current_robot_session().actor_id, first_id)

        self.window.redo_last_action()
        self.assertIn(second.id, self.window.scene.actors.actors)
        self.assertEqual(self.window.editor_robot_actor_id, second.id)
        self.assertEqual(self.window.current_robot_session().actor_id, second.id)


if __name__ == "__main__":
    unittest.main()

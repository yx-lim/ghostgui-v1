import os
import unittest
import numpy as np
import mujoco
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from OpenGL import GL
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow
from gui.robot_model_adapter import MuJoCoRobotAdapter
from gui.viewer_3d import RobotCanvas3D
from gui.collision_checker import CollisionAwareIKSolver, CollisionChecker
from gui.model_assets import resolve_mesh_path, validate_model_assets
from gui.robot_model_registry import ROBOT_MODELS
from gui.transform_gizmo import GizmoInteractionState


class RobotModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_g1_metadata_and_logical_frames(self):
        adapter = MuJoCoRobotAdapter("g1")
        self.assertEqual(adapter.model_type, "humanoid")
        self.assertEqual(len(adapter.actuated_joints), 29)
        self.assertEqual(
            adapter.resolve_logical_frame("left_hand"),
            ("site", "robot/left_palm"),
        )
        self.assertGreater(len(adapter.get_kinematic_edges()), 0)

    def test_go2_urdf_loads_with_quadruped_metadata(self):
        adapter = MuJoCoRobotAdapter("go2")
        self.assertEqual(adapter.model_type, "quadruped")
        self.assertEqual(len(adapter.actuated_joints), 12)
        self.assertEqual(adapter.root_body, "base")
        self.assertIn("FL_foot", adapter.trajectory_frames)
        self.assertEqual(
            adapter.resolve_logical_frame("RR_foot"), ("site", "RR_foot")
        )
        self.assertIn(("base", "FL_hip"), adapter.get_kinematic_edges())
        self.assertIn("converted 17 COLLADA visual references", adapter.load_warning)
        self.assertEqual(adapter.model_path.name, "go2_description.urdf")
        self.assertGreater(adapter.mj_model.nmesh, 0)

    def test_go2_window_uses_go2_controls_and_skeleton(self):
        window = RobotGuiMainWindow("go2")
        try:
            self.assertEqual(len(window.viewer_3d.joint_controls), 12)
            frames = [
                window.controls.frame_box.itemText(index)
                for index in range(window.controls.frame_box.count())
            ]
            self.assertEqual(
                frames,
                ["base", "trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            )
            window.viewer_2d_stickman.update_scene(
                window.trajectory, window.controls.current_frame()
            )
            self.assertGreater(len(window.viewer_2d_stickman.scene.items()), 10)
        finally:
            window.close()

    def test_generated_skeleton_marks_only_editable_model_objects(self):
        window = RobotGuiMainWindow("go2")
        try:
            viewer = window.viewer_2d_stickman
            viewer.update_scene(window.trajectory, window.controls.current_frame())
            handle_names = {
                item.data(2)
                for item in viewer.scene.items()
                if item.data(0) == "editable_handle"
            }
            label_texts = {
                item.toPlainText()
                for item in viewer.scene.items()
                if item.data(0) == "editable_label"
            }

            self.assertEqual(
                handle_names,
                {"base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"},
            )
            self.assertTrue(handle_names <= label_texts)
            self.assertNotIn("FL_hip", handle_names)
            self.assertNotIn("FL_hip", label_texts)
            self.assertEqual(
                viewer._last_editable_projected_points["FL_foot"][1],
                "FL_foot",
            )
        finally:
            window.close()

    def test_go2_real_mesh_visuals_are_rendered_instead_of_collision_primitives(self):
        adapter = MuJoCoRobotAdapter("go2")
        render_ids = RobotCanvas3D.render_geom_ids(adapter.mj_model)
        self.assertGreater(len(render_ids), 0)
        self.assertTrue(all(
            int(adapter.mj_model.geom_type[index])
            == int(mujoco.mjtGeom.mjGEOM_MESH)
            for index in render_ids
        ))
        self.assertTrue(all(
            int(adapter.mj_model.geom_group[index]) == 1
            for index in render_ids
        ))

    def test_registered_mesh_assets_resolve_without_current_working_directory(self):
        for key, info in ROBOT_MODELS.items():
            with self.subTest(model=key):
                results = validate_model_assets(info.model_path, info.package_map)
                self.assertGreater(len(results), 0)
                self.assertTrue(all(result.error is None for result in results))
        resolved = resolve_mesh_path(
            "package://go2_description/dae/base.dae",
            Path("/tmp"),
            ROBOT_MODELS["go2"].package_map,
        )
        self.assertIsNone(resolved.error)
        self.assertEqual(resolved.path.name, "base.dae")

    def test_missing_mesh_has_actionable_error(self):
        result = resolve_mesh_path("package://missing/dae/nope.dae", Path("/tmp"))
        self.assertIsNone(result.path)
        self.assertIn("unresolved mesh", result.error)

    def test_canvas_is_opaque_and_preview_alpha_is_clamped(self):
        canvas = RobotCanvas3D()
        try:
            self.assertEqual(canvas.format().alphaBufferSize(), 0)
            self.assertTrue(canvas.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent))
            self.assertFalse(canvas.testAttribute(
                Qt.WidgetAttribute.WA_TranslucentBackground
            ))
            canvas.set_preview_alpha(5.0)
            self.assertEqual(canvas.preview_alpha, 1.0)
            canvas.set_preview_alpha(0.0)
            self.assertEqual(canvas.preview_alpha, 0.1)
        finally:
            canvas.close()

    def test_gizmo_highlight_applies_only_while_hovered(self):
        canvas = RobotCanvas3D()
        try:
            base_color = (0.9, 0.1, 0.1)
            canvas.gizmo.state = GizmoInteractionState.HOVER_TRANSLATE_X
            self.assertEqual(
                canvas._gizmo_color("x", "TRANSLATE", base_color),
                (1.0, 0.9, 0.15),
            )
            canvas.gizmo.state = GizmoInteractionState.DRAG_TRANSLATE_X
            self.assertEqual(
                canvas._gizmo_color("x", "TRANSLATE", base_color),
                base_color,
            )
        finally:
            canvas.close()

    def test_transparent_pass_preserves_framebuffer_alpha_and_restores_state(self):
        with (
            patch.object(GL, "glEnable") as enable,
            patch.object(GL, "glDisable") as disable,
            patch.object(GL, "glBlendFuncSeparate") as blend,
            patch.object(GL, "glDepthMask") as depth_mask,
            patch.object(GL, "glColorMask") as color_mask,
        ):
            RobotCanvas3D._begin_transparent_pass()
            blend.assert_called_once_with(
                GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE
            )
            color_mask.assert_called_with(
                GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE
            )
            RobotCanvas3D._end_transparent_pass()
            color_mask.assert_called_with(
                GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE
            )
            depth_mask.assert_called_with(GL.GL_TRUE)
            disable.assert_called_with(GL.GL_BLEND)
            enable.assert_called_with(GL.GL_BLEND)

    def test_go2_runtime_has_lit_scene_and_distinct_model_colors(self):
        adapter = MuJoCoRobotAdapter("go2")
        model = adapter.mj_model
        self.assertGreaterEqual(float(model.vis.headlight.ambient[0]), 0.4)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"), 0
        )
        colors = np.unique(np.round(model.geom_rgba, 2), axis=0)
        self.assertGreaterEqual(len(colors), 5)

    def test_go2_foot_drag_converges_without_orientation_overconstraint(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        checker = CollisionChecker(adapter)
        solver = CollisionAwareIKSolver(
            adapter, checker, orientation_weight=0.0
        )
        kind, name = adapter.resolve_logical_frame("FL_foot")
        position, quaternion = state.get_body_pose(name, kind)
        result = solver.solve_drag(
            state.get_qpos(), position, quaternion,
            position + np.array([0.03, 0.0, 0.0]), quaternion,
            object_name=name, kind=kind,
        )
        self.assertTrue(result.success, result.status)
        self.assertEqual(result.accepted_fraction, 1.0)

    def test_go2_viewer_drag_is_preview_only_until_accept(self):
        window = RobotGuiMainWindow("go2")
        try:
            viewer = window.viewer_3d
            kind, name = viewer.robot_model.resolve_logical_frame("FL_foot")
            viewer.select_target(kind, name, emit=False)
            viewer._set_target_to_selected_pose()
            committed = viewer.committed_state.get_qpos()
            position = viewer.last_valid_target_position.copy()
            quaternion = viewer.last_valid_target_quaternion.copy()
            viewer._on_transform_moved(
                position + np.array([0.02, 0.0, 0.0]), quaternion
            )
            np.testing.assert_allclose(viewer.committed_state.get_qpos(), committed)
            self.assertFalse(np.allclose(viewer.preview_state.get_qpos(), committed))
            viewer.accept_preview()
            np.testing.assert_allclose(
                viewer.committed_state.get_qpos(), viewer.preview_state.get_qpos()
            )
        finally:
            window.close()

    def test_picked_bodies_map_to_logical_frames_for_both_models(self):
        g1 = MuJoCoRobotAdapter("g1")
        self.assertEqual(
            g1.logical_frame_for_body("robot/left_wrist_yaw_link"), "left_hand"
        )
        self.assertEqual(
            g1.logical_frame_for_body("robot/right_ankle_roll_link"), "right_foot"
        )
        go2 = MuJoCoRobotAdapter("go2")
        self.assertEqual(go2.logical_frame_for_body("FL_hip"), "FL_foot")
        self.assertEqual(go2.logical_frame_for_body("base"), "base")

    def test_ray_pick_returns_a_robot_body(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        canvas = RobotCanvas3D()
        canvas.set_robot_states(state, adapter.create_state())
        calf_id = mujoco.mj_name2id(
            adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf"
        )
        geom_id = min(
            (
                index for index in RobotCanvas3D.render_geom_ids(adapter.mj_model)
                if int(adapter.mj_model.geom_bodyid[index]) == calf_id
            ),
            key=lambda index: state.mj_data.geom_xpos[index][2],
        )
        center = state.mj_data.geom_xpos[geom_id]
        picked = canvas.pick_robot_body_from_ray(
            center + np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0])
        )
        self.assertIsNotNone(picked)
        self.assertEqual(adapter.logical_frame_for_body(picked), "FL_foot")

    def test_double_click_body_selection_updates_frame_editor(self):
        g1_window = RobotGuiMainWindow("g1")
        go2_window = RobotGuiMainWindow("go2")
        try:
            g1_window.viewer_3d._on_body_double_clicked(
                "robot/left_wrist_yaw_link"
            )
            self.assertEqual(g1_window.controls.frame_box.currentText(), "left_hand")
            self.assertEqual(
                g1_window.viewer_3d._selected_target(),
                ("site", "robot/left_palm"),
            )
            go2_window.viewer_3d._on_body_double_clicked("FR_thigh")
            self.assertEqual(go2_window.controls.frame_box.currentText(), "FR_foot")
            self.assertEqual(
                go2_window.viewer_3d._selected_target(), ("site", "FR_foot")
            )
        finally:
            g1_window.close()
            go2_window.close()

    def test_generated_skeleton_uses_ik_and_whole_body_follow(self):
        window = RobotGuiMainWindow("go2")
        try:
            viewer = window.viewer_2d_stickman
            viewer.update_scene(window.trajectory, None)
            base_before = viewer._last_projected_positions["base"]
            frame = window.controls.current_frame()
            frame.frame_name = "FL_foot"
            frame.x = 2.0
            frame.y = 0.142
            frame.z = 1.0
            viewer.update_scene(window.trajectory, frame)
            _, model_name = window.robot_model_3d.resolve_logical_frame("FL_foot")
            foot = viewer._last_projected_positions[model_name]
            base_after = viewer._last_projected_positions["base"]
            self.assertAlmostEqual(foot[0], frame.x, delta=0.02)
            self.assertAlmostEqual(foot[1], frame.z, delta=0.02)
            self.assertGreater(np.linalg.norm(np.subtract(base_after, base_before)), 0.2)
            for name, joint in window.robot_model_3d.joints.items():
                limits = joint.limits
                if limits is not None:
                    value = viewer.skeleton_state.get_joint_value(name)
                    self.assertGreaterEqual(value, limits[0] - 1e-9)
                    self.assertLessEqual(value, limits[1] + 1e-9)
        finally:
            window.close()

    def test_model_switch_preserves_selected_editor_tab(self):
        window = RobotGuiMainWindow("g1")
        try:
            for mode in ("skeleton", "3d"):
                selected = (
                    window.viewer_2d_skeleton_stack if mode == "skeleton"
                    else window.viewer_3d_stack
                )
                window.viewer_tabs.setCurrentWidget(selected)
                target_key = "go2" if window.model_key == "g1" else "g1"
                window.on_model_changed(target_key)
                loader = window.model_loaders.get(target_key)
                if loader is not None:
                    loader.wait()
                    self.app.processEvents()
                self.assertEqual(window.model_key, target_key)
                self.assertIs(window.viewer_tabs.currentWidget(), selected)
                expected_page = (
                    window.viewer_2d_stickman if mode == "skeleton"
                    else window.viewer_3d
                )
                self.assertIs(selected.currentWidget(), expected_page)
        finally:
            window.close()

    def test_model_session_is_reused_after_switching_back(self):
        window = RobotGuiMainWindow("g1")
        try:
            g1_viewer = window.viewer_3d
            window.on_model_changed("go2")
            loader = window.model_loaders.get("go2")
            if loader is not None:
                loader.wait()
                self.app.processEvents()
            go2_viewer = window.viewer_3d
            window.on_model_changed("g1")
            self.assertIs(window.viewer_3d, g1_viewer)
            window.on_model_changed("go2")
            self.assertIs(window.viewer_3d, go2_viewer)
        finally:
            window.close()

    def test_user_urdf_uses_versioned_persistent_cache(self):
        urdf = """<robot name='cache_test'>
          <link name='base'><inertial><mass value='1'/><inertia ixx='1' iyy='1' izz='1' ixy='0' ixz='0' iyz='0'/></inertial><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        </robot>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "robot.urdf"
            source.write_text(urdf, encoding="utf-8")
            old_cache = os.environ.get("GHOSTGUI_CACHE_DIR")
            os.environ["GHOSTGUI_CACHE_DIR"] = str(temp / "cache")
            try:
                first = MuJoCoRobotAdapter.load_model(source)
                second = MuJoCoRobotAdapter.load_model(source)
                self.assertEqual(first.runtime_model_path, second.runtime_model_path)
                self.assertTrue(first.runtime_model_path.exists())
                self.assertTrue(first.runtime_model_path.with_name("metadata.json").exists())
            finally:
                if old_cache is None:
                    os.environ.pop("GHOSTGUI_CACHE_DIR", None)
                else:
                    os.environ["GHOSTGUI_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    unittest.main()

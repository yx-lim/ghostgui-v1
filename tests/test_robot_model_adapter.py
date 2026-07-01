import os
import unittest
import numpy as np
import mujoco
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow
from gui.robot_model_adapter import MuJoCoRobotAdapter
from gui.viewer_3d import RobotCanvas3D
from gui.collision_checker import CollisionAwareIKSolver, CollisionChecker


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
        self.assertIsNone(adapter.load_warning)
        self.assertEqual(adapter.model_path.name, "go2.xml")

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

    def test_go2_collision_geometries_are_rendered_when_visual_group_is_absent(self):
        adapter = MuJoCoRobotAdapter("go2")
        self.assertFalse(any(
            int(adapter.mj_model.geom_group[index]) == 2
            for index in range(adapter.mj_model.ngeom)
        ))
        self.assertEqual(
            RobotCanvas3D.render_geom_ids(adapter.mj_model),
            set(range(adapter.mj_model.ngeom)),
        )

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

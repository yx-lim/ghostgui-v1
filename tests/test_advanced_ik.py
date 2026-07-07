import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.ik_tasks import FootLockTask, JointRegularizationTask, PostureTask, RootPoseTask
from gui.main_window import RobotGuiMainWindow
from gui.robot_model_adapter import MuJoCoRobotAdapter
from gui.backend_interface import MujocoIKBackend
from gui.trajectory import TargetFrame, quat_to_rpy, rpy_to_quat


class AdvancedIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_zero_influence_locks_joint_and_larger_weight_prefers_it(self):
        model = MuJoCoRobotAdapter("g1")
        deltas = {}
        for influence in (0.0, 3.0):
            state = model.create_state()
            position, _ = state.get_body_pose("robot/right_palm", "site")
            before = state.get_joint_value("right_elbow_joint")
            weights = {name: 1.0 for name in model.get_joint_names()}
            weights["right_elbow_joint"] = influence
            result = state.solve_ik(
                "robot/right_palm",
                position + np.array([0.01, 0.0, 0.0]),
                kind="site",
                joint_weights=weights,
            )
            self.assertTrue(result.success, result.message)
            deltas[influence] = abs(
                state.get_joint_value("right_elbow_joint") - before
            )
        self.assertEqual(deltas[0.0], 0.0)
        self.assertGreater(deltas[3.0], deltas[0.0])

    def test_joint_influence_controls_are_model_generated(self):
        g1 = RobotGuiMainWindow("g1")
        go2 = RobotGuiMainWindow("go2")
        try:
            self.assertEqual(len(g1.viewer_3d.ik_influence_controls), 29)
            self.assertEqual(len(go2.viewer_3d.ik_influence_controls), 12)
            self.assertTrue(all(
                value == 1.0 for value in g1.viewer_3d.ik_joint_weights.values()
            ))
            self.assertIn("Upper body only", [
                g1.viewer_3d.ik_preset_box.itemText(index)
                for index in range(g1.viewer_3d.ik_preset_box.count())
            ])
            self.assertIn("Quadruped legs only", [
                go2.viewer_3d.ik_preset_box.itemText(index)
                for index in range(go2.viewer_3d.ik_preset_box.count())
            ])
        finally:
            g1.close()
            go2.close()

    def test_selected_limb_and_feet_planted_presets(self):
        window = RobotGuiMainWindow("g1")
        try:
            viewer = window.viewer_3d
            viewer.select_target("site", "robot/right_palm", emit=False)
            viewer.ik_preset_box.setCurrentText("Selected limb only")
            viewer.apply_ik_preset()
            self.assertGreater(viewer.ik_joint_weights["right_elbow_joint"], 0.0)
            self.assertEqual(viewer.ik_joint_weights["left_elbow_joint"], 0.0)
            self.assertEqual(viewer.ik_joint_weights["right_knee_joint"], 0.0)

            viewer.ik_preset_box.setCurrentText("Feet planted")
            viewer.apply_ik_preset()
            foot_checkbox, foot_weight = viewer.ik_task_controls["foot_lock"]
            self.assertTrue(foot_checkbox.isChecked())
            self.assertGreaterEqual(foot_weight.value(), 1.0)
        finally:
            window.close()

    def test_secondary_tasks_have_priority_metadata(self):
        window = RobotGuiMainWindow("go2")
        try:
            viewer = window.viewer_3d
            viewer.begin_preview()
            for task_name in ("foot_lock", "posture", "regularization"):
                viewer.ik_task_controls[task_name][0].setChecked(True)
            tasks = viewer._secondary_ik_tasks()
            self.assertEqual(len([task for task in tasks if isinstance(task, FootLockTask)]), 4)
            self.assertTrue(any(isinstance(task, RootPoseTask) for task in tasks))
            self.assertTrue(any(isinstance(task, PostureTask) for task in tasks))
            self.assertTrue(any(
                isinstance(task, JointRegularizationTask) for task in tasks
            ))
            self.assertTrue(all(task.priority == 1 for task in tasks if isinstance(
                task, (FootLockTask, RootPoseTask)
            )))
        finally:
            window.close()

    def test_influence_values_persist_in_cached_model_session(self):
        window = RobotGuiMainWindow("g1")
        try:
            viewer = window.viewer_3d
            viewer.ik_influence_controls["right_elbow_joint"].set_value(2.5)
            viewer._ik_influence_changed("right_elbow_joint", 2.5)
            window.on_model_changed("go2")
            loader = window.model_loaders.get("go2")
            if loader is not None:
                loader.wait()
                self.app.processEvents()
            window.on_model_changed("g1")
            self.assertIs(window.viewer_3d, viewer)
            self.assertEqual(
                window.viewer_3d.ik_joint_weights["right_elbow_joint"], 2.5
            )
        finally:
            window.close()

    def test_default_secondary_tasks_do_not_clamp_normal_limb_drag(self):
        cases = (
            ("g1", "right_hand", np.array([0.1, 0.0, 0.0])),
            ("go2", "FL_foot", np.array([0.0, 0.0, 0.1])),
        )
        for model_key, logical_frame, offset in cases:
            with self.subTest(model=model_key):
                window = RobotGuiMainWindow(model_key)
                try:
                    viewer = window.viewer_3d
                    kind, name = viewer.robot_model.resolve_logical_frame(
                        logical_frame
                    )
                    viewer.select_target(kind, name, emit=False)
                    viewer._set_target_to_selected_pose()
                    start = viewer.last_valid_target_position.copy()
                    quaternion = viewer.last_valid_target_quaternion.copy()
                    committed = viewer.committed_state.get_qpos().copy()

                    viewer._on_transform_moved(
                        start + offset, quaternion
                    )

                    self.assertAlmostEqual(
                        np.linalg.norm(viewer.last_valid_target_position - start),
                        0.1,
                        places=6,
                    )
                    np.testing.assert_array_equal(
                        viewer.committed_state.get_qpos(), committed
                    )
                finally:
                    window.close()

    def test_batch_ik_solves_hand_position_and_orientation(self):
        adapter = MuJoCoRobotAdapter("g1")
        home = adapter.create_state()
        position, quaternion = home.get_body_pose("robot/left_palm", "site")
        roll, pitch, yaw = quat_to_rpy(quaternion)
        target_rpy = (roll + 0.10, pitch, yaw)
        target = TargetFrame(
            time=0.0,
            frame_name="left_hand",
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            roll=target_rpy[0],
            pitch=target_rpy[1],
            yaw=target_rpy[2],
        )
        backend = MujocoIKBackend(
            mj_model=adapter.mj_model, adapter=adapter
        )
        result = backend.solve_grouped_trajectory([{
            "time": 0.0,
            "targets": {"left_hand": target},
        }])[0]

        solved = adapter.create_state()
        solved.set_qpos(result.qpos)
        solved_position, solved_quaternion = solved.get_body_pose(
            "robot/left_palm", "site"
        )
        target_quaternion = rpy_to_quat(*target_rpy)
        orientation_error = 2.0 * np.arccos(np.clip(
            abs(float(np.dot(solved_quaternion, target_quaternion))),
            -1.0,
            1.0,
        ))
        self.assertTrue(result.success, result.status)
        self.assertLess(np.linalg.norm(solved_position - position), 0.005)
        self.assertLess(orientation_error, 0.03)


if __name__ == "__main__":
    unittest.main()

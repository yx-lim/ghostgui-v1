import unittest
from unittest.mock import patch

from backend.interface import G1AnalyticFallbackBackend, MujocoIKBackend
from core.ik.tasks import TCPPositionTask
from core.models.adapter import MuJoCoRobotAdapter
from core.models.model import IKResult


class BackendTests(unittest.TestCase):
    def test_python_fallback_is_explicitly_g1_only(self):
        backend = G1AnalyticFallbackBackend()
        self.assertIn("left_hip_pitch_joint", backend.joint_names)

    def test_mujoco_backend_delegates_to_shared_weighted_solver(self):
        adapter = MuJoCoRobotAdapter("g1")
        backend = MujocoIKBackend(mj_model=adapter.mj_model, adapter=adapter)
        kind, name = adapter.resolve_logical_frame("right_hand")
        position, _ = backend.state.get_body_pose(name, kind)

        with patch.object(
            backend.state,
            "solve_weighted_tasks",
            return_value=IKResult(True, 0.0, 1, "test"),
        ) as solve:
            class Target:
                x, y, z = position

            error, success, iterations = backend.solve_position_ik(
                {"right_hand": Target()}
            )

        self.assertTrue(success)
        self.assertEqual((error, iterations), (0.0, 1))
        tasks = solve.call_args.args[0]
        self.assertTrue(all(isinstance(task, TCPPositionTask) for task in tasks))


if __name__ == "__main__":
    unittest.main()

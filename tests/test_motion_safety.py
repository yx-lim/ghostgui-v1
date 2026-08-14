import unittest

import numpy as np

from core.ik import CollisionChecker, project_qpos_above_flat_ground
from core.models import MuJoCoRobotAdapter


class FlatGroundProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MuJoCoRobotAdapter("g1")
        cls.checker = CollisionChecker(cls.model)
        cls.free_joint = next(iter(cls.model.free_joints_by_body.values()))

    def test_safe_pose_is_returned_unchanged(self):
        qpos = self.model.home_qpos.copy()

        result = project_qpos_above_flat_ground(
            self.model, qpos, checker=self.checker
        )

        self.assertTrue(result.success, result.reason)
        self.assertFalse(result.changed)
        np.testing.assert_allclose(result.qpos, qpos)

    def test_penetrating_pose_is_lifted_without_mutating_input(self):
        qpos = self.model.home_qpos.copy()
        original = qpos.copy()
        qpos[self.free_joint.qpos_address + 2] -= 0.03

        result = project_qpos_above_flat_ground(
            self.model, qpos, checker=self.checker
        )

        self.assertTrue(result.success, result.reason)
        self.assertTrue(result.changed)
        self.assertGreater(result.applied_offset, 0.0)
        np.testing.assert_allclose(original, self.model.home_qpos)
        self.assertFalse(self.checker.get_blocking_collisions(
            self._state(result.qpos)
        ))

    def test_excessive_lift_is_rejected(self):
        qpos = self.model.home_qpos.copy()
        qpos[self.free_joint.qpos_address + 2] -= 0.30

        result = project_qpos_above_flat_ground(
            self.model, qpos, checker=self.checker, max_lift=0.05
        )

        self.assertFalse(result.success)
        self.assertIn("automatic-repair limit", result.reason)

    def _state(self, qpos):
        state = self.model.create_state()
        state.set_qpos(qpos)
        return state


if __name__ == "__main__":
    unittest.main()

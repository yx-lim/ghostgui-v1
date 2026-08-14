from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from application.motion_safety import propose_safe_motion_repair
from core.ik import (
    Collision,
    adaptive_trajectory_collision_reports as adaptive_validator,
)
from core.models import RobotModel3D


_TWO_JOINT_MODEL = """
<mujoco model="repair_test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link">
      <joint name="avoid_joint" type="hinge" axis="1 0 0"
             limited="true" range="-1 1"/>
      <joint name="path_joint" type="hinge" axis="0 0 1"
             limited="true" range="-1 1"/>
      <geom name="link_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_FREE_BOX_MODEL = """
<mujoco model="ground_sweep_test">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="floating_box" pos="0 0 0.06">
      <freejoint name="root"/>
      <geom name="box" type="box" size="0.30 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


class _WindowSelfCollisionChecker:
    def __init__(
        self,
        avoid_address,
        path_address,
        *,
        path_center=0.0,
        path_half_width=0.021,
        avoid_half_width=0.041,
    ):
        self.avoid_address = int(avoid_address)
        self.path_address = int(path_address)
        self.path_center = float(path_center)
        self.path_half_width = float(path_half_width)
        self.avoid_half_width = float(avoid_half_width)

    def get_collisions(self, state):
        qpos = state.get_qpos()
        if (
            abs(float(qpos[self.path_address]) - self.path_center)
            <= self.path_half_width
            and abs(float(qpos[self.avoid_address]))
            <= self.avoid_half_width
        ):
            return [
                Collision(
                    "moving_geom",
                    "torso_geom",
                    "arm",
                    "torso",
                    -0.01,
                    "self",
                )
            ]
        return []

    def get_blocking_collisions(self, state):
        return [
            collision for collision in self.get_collisions(state)
            if collision.blocking
        ]


class ApplicationMotionSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory()
        model_path = Path(cls._temporary_directory.name) / "repair_test.xml"
        model_path.write_text(_TWO_JOINT_MODEL, encoding="utf-8")
        cls.model = RobotModel3D(model_path)
        cls.avoid = cls.model.joints["avoid_joint"]
        cls.path = cls.model.joints["path_joint"]

    @classmethod
    def tearDownClass(cls):
        cls._temporary_directory.cleanup()

    def _crossing_motion(self):
        start = self.model.home_qpos.copy()
        end = self.model.home_qpos.copy()
        start[self.path.qpos_address] = -0.16
        end[self.path.qpos_address] = 0.16
        return start, end

    def test_safe_motion_returns_independent_unchanged_copies(self):
        start = self.model.home_qpos.copy()
        end = self.model.home_qpos.copy()
        end[self.path.qpos_address] = 0.04
        originals = (start.copy(), end.copy())

        result = propose_safe_motion_repair(
            self.model, (start, end), (1.0, 2.0)
        )

        self.assertTrue(result.success, result.status)
        self.assertFalse(result.changed)
        self.assertFalse(result.requires_review)
        self.assertIsNot(result.qposes[0], start)
        np.testing.assert_allclose(result.qposes, originals)
        np.testing.assert_allclose(start, originals[0])
        np.testing.assert_allclose(end, originals[1])

    def test_flat_ground_projection_is_per_sample_and_explicit(self):
        start = self.model.home_qpos.copy()
        end = self.model.home_qpos.copy()
        original_start = start.copy()
        calls = 0

        def project(_model, qpos, **_kwargs):
            nonlocal calls
            calls += 1
            projected = np.asarray(qpos, dtype=float).copy()
            changed = calls == 1
            if changed:
                projected[self.avoid.qpos_address] += 0.02
            return SimpleNamespace(
                success=True,
                qpos=projected,
                changed=changed,
                reason="test projection",
            )

        with patch(
            "application.motion_safety.project_qpos_above_flat_ground",
            side_effect=project,
        ), patch(
            "application.motion_safety."
            "adaptive_trajectory_collision_reports",
            return_value=(None, None),
        ):
            result = propose_safe_motion_repair(
                self.model, (start, end), (0.0, 1.0)
            )

        self.assertTrue(result.success, result.status)
        self.assertTrue(result.requires_review)
        self.assertEqual(calls, 2)
        self.assertEqual(result.ground_correction_count, 1)
        self.assertAlmostEqual(
            result.qposes[0][self.avoid.qpos_address], 0.02
        )
        np.testing.assert_allclose(start, original_start)
        self.assertIn("input motion was not modified or published", result.status)

    def test_interior_self_collision_returns_reviewable_local_detour(self):
        start, end = self._crossing_motion()
        original_start = start.copy()
        original_end = end.copy()
        checker = _WindowSelfCollisionChecker(
            self.avoid.qpos_address, self.path.qpos_address
        )

        with patch(
            "application.motion_safety."
            "adaptive_trajectory_collision_reports",
            wraps=adaptive_validator,
        ) as validate:
            result = propose_safe_motion_repair(
                self.model,
                (start, end),
                (2.0, 4.0),
                checker=checker,
                max_detour_waypoints=1,
                detour_step=0.1,
                max_detour_offset=0.2,
            )

        self.assertTrue(result.success, result.status)
        self.assertTrue(result.requires_review)
        self.assertEqual(result.detour_waypoint_count, 1)
        self.assertEqual(result.ground_correction_count, 0)
        self.assertEqual(result.times, (2.0, 3.0, 4.0))
        self.assertAlmostEqual(
            result.qposes[1][self.avoid.qpos_address], 0.1
        )
        self.assertAlmostEqual(result.qposes[1][self.path.qpos_address], 0.0)
        self.assertIsNone(result.blocking_report)
        self.assertIn("Review candidate", result.status)
        self.assertIn("avoid_joint +0.100 rad", result.status)
        self.assertIn("complete path pass", result.status)
        np.testing.assert_allclose(start, original_start)
        np.testing.assert_allclose(end, original_end)

        # Initial whole path, left replacement interval, right replacement
        # interval, then the final whole candidate.
        self.assertEqual(validate.call_count, 4)
        validated_times = [
            tuple(call.kwargs["times"]) for call in validate.call_args_list
        ]
        self.assertEqual(
            validated_times,
            [(2.0, 4.0), (2.0, 3.0), (3.0, 4.0), (2.0, 3.0, 4.0)],
        )

    def test_endpoint_self_collision_is_not_rerouted(self):
        start, end = self._crossing_motion()
        start[self.path.qpos_address] = 0.0
        checker = _WindowSelfCollisionChecker(
            self.avoid.qpos_address, self.path.qpos_address
        )

        result = propose_safe_motion_repair(
            self.model, (start, end), (0.0, 1.0), checker=checker
        )

        self.assertFalse(result.success)
        self.assertFalse(result.changed)
        self.assertIsNotNone(result.blocking_report)
        self.assertFalse(result.blocking_report.is_interior)
        self.assertEqual(result.detour_waypoint_count, 0)
        np.testing.assert_allclose(result.qposes, (start, end))
        self.assertIn("input motion was not modified", result.status)

    def test_failed_bounded_search_returns_original_motion(self):
        start, end = self._crossing_motion()
        # A very wide avoid window means none of the permitted detour offsets
        # can escape the collision region.
        checker = _WindowSelfCollisionChecker(
            self.avoid.qpos_address,
            self.path.qpos_address,
            path_half_width=0.08,
            avoid_half_width=2.0,
        )

        result = propose_safe_motion_repair(
            self.model,
            (start, end),
            (0.0, 1.0),
            checker=checker,
            max_detour_waypoints=1,
            detour_step=0.1,
            max_detour_offset=0.2,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.detour_waypoint_count, 0)
        self.assertIsNotNone(result.blocking_report)
        np.testing.assert_allclose(result.qposes, (start, end))
        self.assertIn("could not reroute", result.status)

    def test_between_frame_ground_sweep_gets_lifted_waypoint(self):
        model_path = (
            Path(self._temporary_directory.name) / "ground_sweep_test.xml"
        )
        model_path.write_text(_FREE_BOX_MODEL, encoding="utf-8")
        model = RobotModel3D(model_path)
        free_joint = next(iter(model.free_joints_by_body.values()))
        address = free_joint.qpos_address
        start = model.home_qpos.copy()
        end = start.copy()
        # Both endpoints clear the plane by roughly 10 mm. Linear root
        # translation plus quaternion SLERP makes the long box corner dip
        # through the plane between them.
        start[address + 2] = 0.06
        end[address + 2] = 0.31
        end[address + 3:address + 7] = (
            np.cos(np.pi / 4.0),
            0.0,
            np.sin(np.pi / 4.0),
            0.0,
        )
        warning, blocking = adaptive_validator(
            model, (start, end), times=(0.0, 1.0)
        )
        self.assertIsNotNone(blocking)
        self.assertTrue(blocking.is_interior)
        self.assertTrue(all(
            collision.kind == "environment"
            for collision in blocking.collisions
        ))

        result = propose_safe_motion_repair(
            model, (start, end), (0.0, 1.0)
        )

        self.assertTrue(result.success, result.status)
        self.assertEqual(result.detour_waypoint_count, 1)
        self.assertGreaterEqual(result.ground_correction_count, 1)
        self.assertEqual(len(result.qposes), 3)
        _warning, repaired_blocking = adaptive_validator(
            model, result.qposes, times=result.times
        )
        self.assertIsNone(repaired_blocking)


if __name__ == "__main__":
    unittest.main()

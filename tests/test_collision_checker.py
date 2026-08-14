import unittest

import numpy as np

from core.ik import (
    Collision,
    CollisionAwareIKSolver,
    CollisionChecker,
    CollisionPolicy,
    adaptive_trajectory_collision_reports,
    first_trajectory_collision,
)
from core.models import IKResult, MuJoCoRobotAdapter, RobotModel3D


class FakeCandidateState:
    def __init__(self, fail=False):
        self.qpos = np.zeros(1)
        self.fail = fail

    def set_qpos(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float).copy()

    def get_qpos(self):
        return self.qpos.copy()

    def solve_ik(self, name, position, quaternion, kind=None, **kwargs):
        if self.fail:
            return IKResult(False, 1.0, 1, "failed")
        self.qpos[0] = position[0]
        return IKResult(True, 0.0, 1, "converged")


class ThresholdCollisionChecker:
    def get_collisions(self, state):
        return [Collision("a", "b", "one", "two", -0.01, "self")] \
            if state.qpos[0] >= 0.6 else []


class JointWindowCollisionChecker:
    def __init__(self, address, advisory_window, blocking_window):
        self.address = int(address)
        self.advisory_window = advisory_window
        self.blocking_window = blocking_window

    def get_collisions(self, state):
        value = float(state.get_qpos()[self.address])
        if self.blocking_window[0] <= value <= self.blocking_window[1]:
            return [
                Collision(
                    "moving", "obstacle", "arm", "torso", -0.01,
                    "self", blocking=True,
                )
            ]
        if self.advisory_window[0] <= value <= self.advisory_window[1]:
            return [
                Collision(
                    "moving", "obstacle", "arm", "torso", -0.0005,
                    "self", blocking=False,
                )
            ]
        return []


class CollisionCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel3D()

    def test_actual_mujoco_self_collision_is_reported(self):
        state = self.model.create_state()
        qpos = np.array([
            0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0, -1.3199862104,
            1.2988490986, 0.7785858799, 2.6991265477, -0.0600187980,
            -0.1215624368, 2.4998459890, -1.2506345490, 0.9695769303,
            1.3251723420, -0.5697072985, 0.1008202506, 1.4170212095,
            -0.3215799545, -0.0416873467, -1.0052899810, -0.9326487123,
            -1.4590027090, 1.9766132834, 1.5155447738, -0.3800632141,
            0.7732255758, -2.8474808931, 1.2620451356, 0.1946265770,
            1.5297533315, -0.8837980993, -0.3999788646, -0.4906455578,
        ])
        state.set_qpos(qpos)
        collisions = CollisionChecker(self.model).get_collisions(state)
        self.assertTrue(collisions)
        self.assertTrue(any(item.kind == "self" for item in collisions))
        self.assertTrue(
            all(
                item.geom1_id is not None
                and item.geom2_id is not None
                and item.body1_id is not None
                and item.body2_id is not None
                for item in collisions
            )
        )

    def test_actual_ground_collision_is_reported(self):
        state = self.model.create_state()
        qpos = state.get_qpos()
        free_joint = next(iter(self.model.free_joints_by_body.values()))
        qpos[free_joint.qpos_address + 2] -= 0.2
        state.set_qpos(qpos)

        collisions = CollisionChecker(self.model).get_collisions(state)

        self.assertTrue(collisions)
        self.assertTrue(any(item.kind == "environment" for item in collisions))
        floor_contact = next(
            item for item in collisions if "floor" in (item.geom1, item.geom2)
        )
        self.assertIn("floor", floor_contact.pair_label)
        self.assertNotIn("world", floor_contact.pair_label)

    def test_z1_duplicate_contact_points_use_one_semantic_warning(self):
        model = MuJoCoRobotAdapter("z1")
        state = model.create_state()
        state.set_joint_values({name: 0.0 for name in state.get_joint_names()})

        collisions = CollisionChecker(model).get_collisions(state)

        self.assertEqual(len(collisions), 1)
        collision = collisions[0]
        self.assertEqual(collision.pair_label, "link02 ↔ link06")
        self.assertEqual(collision.geom1, "link02__contact_1")
        self.assertEqual(collision.geom2, "link06__contact_1")
        self.assertIn("mm penetration", collision.diagnostic_label)
        self.assertTrue(collision.blocking)

    def test_environment_penetration_uses_hard_half_mm_barrier(self):
        model = MuJoCoRobotAdapter("z1")
        checker = CollisionChecker(model)
        free_joint = next(iter(model.free_joints_by_body.values()))

        shallow = model.create_state()
        shallow_qpos = shallow.get_qpos()
        shallow_qpos[free_joint.qpos_address + 2] -= 0.004
        shallow.set_qpos(shallow_qpos)
        shallow_collisions = checker.get_collisions(shallow)

        deep = model.create_state()
        deep_qpos = deep.get_qpos()
        deep_qpos[free_joint.qpos_address + 2] -= 0.006
        deep.set_qpos(deep_qpos)
        deep_collisions = checker.get_collisions(deep)

        self.assertTrue(shallow_collisions)
        self.assertTrue(any(item.blocking for item in shallow_collisions))
        self.assertTrue(any(item.blocking for item in deep_collisions))
        report = first_trajectory_collision(
            model,
            (model.home_qpos, shallow_qpos, deep_qpos),
            checker=checker,
            blocking_only=True,
        )
        self.assertEqual(report.sample_index, 1)

    def test_adaptive_reports_find_contacts_between_safe_samples(self):
        joint = next(
            item for item in self.model.joints.values()
            if item.limits is not None
        )
        address = joint.qpos_address
        center = float(self.model.home_qpos[address])
        start = self.model.home_qpos.copy()
        end = self.model.home_qpos.copy()
        start[address] = center - 0.16
        end[address] = center + 0.16
        checker = JointWindowCollisionChecker(
            address,
            (center - 0.085, center - 0.075),
            (center - 0.005, center + 0.005),
        )

        # The legacy sampled-only scan deliberately sees two safe endpoints.
        self.assertIsNone(first_trajectory_collision(
            self.model, (start, end), checker=checker
        ))

        warning, blocking = adaptive_trajectory_collision_reports(
            self.model,
            (start, end),
            times=(2.0, 4.0),
            checker=checker,
            max_joint_step=0.02,
            max_body_step=1.0,
        )

        self.assertIsNotNone(warning)
        self.assertFalse(warning.blocking)
        self.assertEqual(warning.segment_index, 0)
        self.assertAlmostEqual(warning.segment_fraction, 0.25)
        self.assertAlmostEqual(warning.time, 2.5)
        self.assertEqual(warning.sample_index, 0)
        self.assertTrue(warning.is_interior)
        self.assertIn("segment 0 at 25.0%", warning.location_label)
        self.assertIsNotNone(blocking)
        self.assertTrue(blocking.blocking)
        self.assertAlmostEqual(blocking.segment_fraction, 0.5)
        self.assertAlmostEqual(blocking.time, 3.0)

    def test_adaptive_reports_keep_endpoint_sample_location(self):
        joint = next(
            item for item in self.model.joints.values()
            if item.limits is not None
        )
        address = joint.qpos_address
        start = self.model.home_qpos.copy()
        end = self.model.home_qpos.copy()
        end[address] += 0.1
        checker = JointWindowCollisionChecker(
            address,
            (end[address] - 0.001, end[address] + 0.001),
            (float("inf"), float("inf")),
        )

        warning, blocking = adaptive_trajectory_collision_reports(
            self.model, (start, end), times=(0.0, 0.5), checker=checker
        )

        self.assertIsNone(blocking)
        self.assertEqual(warning.sample_index, 1)
        self.assertEqual(warning.segment_index, 0)
        self.assertEqual(warning.segment_fraction, 1.0)
        self.assertEqual(warning.time, 0.5)
        self.assertFalse(warning.is_interior)
        self.assertEqual(warning.location_label, "sample 1 at 0.5 s")

    def test_audited_body_pair_exclusion_has_a_depth_limit(self):
        policy = CollisionPolicy(
            allowed_body_pair_tolerances=(
                (frozenset(("link_a", "link_b")), 0.002),
            ),
        )

        self.assertTrue(policy.allows(
            "geom_a", "geom_b", "link_a", "link_b", 1, 2, -0.001
        ))
        self.assertFalse(policy.allows(
            "geom_a", "geom_b", "link_a", "link_b", 1, 2, -0.003
        ))

    def test_world_owned_ground_keeps_its_source_name(self):
        model = MuJoCoRobotAdapter("z1")
        state = model.create_state()
        qpos = state.get_qpos()
        free_joint = next(iter(model.free_joints_by_body.values()))
        qpos[free_joint.qpos_address + 2] -= 0.1
        state.set_qpos(qpos)

        collisions = CollisionChecker(model).get_collisions(state)
        ground_contact = next(
            item for item in collisions if "ground" in (item.geom1, item.geom2)
        )

        self.assertIn("ground", ground_contact.pair_label)
        self.assertNotIn("world", ground_contact.pair_label)

    def _fake_solver(self, fail=False):
        solver = CollisionAwareIKSolver.__new__(CollisionAwareIKSolver)
        solver.candidate_state = FakeCandidateState(fail=fail)
        solver.collision_checker = ThresholdCollisionChecker()
        solver.collision_drag_substeps = 4
        solver.ik_tolerance = 0.001
        solver.orientation_weight = 0.25
        return solver

    def test_blocking_collision_clamps_preview_drag_at_last_safe_substep(self):
        result = self._fake_solver().solve_drag(
            np.zeros(1), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            object_name="target",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.accepted_fraction, 0.5)
        self.assertAlmostEqual(result.qpos[0], 0.5)
        self.assertIn("Safety barrier stopped", result.status)
        self.assertTrue(result.collisions)
        self.assertTrue(all(
            collision.blocking for collision in result.collisions
        ))

    def test_failed_ik_preserves_last_valid_qpos(self):
        start = np.array([0.25])
        result = self._fake_solver(fail=True).solve_drag(
            start, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            object_name="target",
        )
        self.assertFalse(result.success)
        np.testing.assert_allclose(result.qpos, start)
        self.assertEqual(result.accepted_fraction, 0.0)
        self.assertIn("IK reach limit", result.status)

    def test_failed_optional_tasks_retry_with_required_translation_only(self):
        class FakeWeightedState:
            def __init__(self):
                self.qpos = np.zeros(1)
                self.calls = []

            def set_qpos(self, qpos):
                self.qpos = np.asarray(qpos, dtype=float).copy()

            def get_qpos(self):
                return self.qpos.copy()

            def resolve_object(self, _name, _kind):
                return "site", 0

            def solve_weighted_tasks(self, tasks, **_kwargs):
                self.calls.append([task.required for task in tasks])
                position_task = next(
                    (task for task in tasks if hasattr(task, "target_position")),
                    None,
                )
                if position_task is not None:
                    self.qpos[0] = float(position_task.target_position[0])
                has_optional = any(not task.required for task in tasks)
                return IKResult(
                    not has_optional,
                    0.02 if has_optional else 0.0,
                    1,
                    "optional conflict" if has_optional else "required converged",
                )

            def get_body_pose(self, _name, _kind):
                return (
                    np.array([self.qpos[0], 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )

        class FakeRobotModel:
            @staticmethod
            def free_joint_for_body(_body_id):
                return None

        class NoCollisions:
            @staticmethod
            def get_collisions(_state):
                return []

        solver = CollisionAwareIKSolver.__new__(CollisionAwareIKSolver)
        solver.robot_model = FakeRobotModel()
        solver.candidate_state = FakeWeightedState()
        solver.collision_checker = NoCollisions()
        solver.collision_drag_substeps = 1
        solver.ik_tolerance = 0.001
        solver.orientation_weight = 0.25

        result = solver.solve_drag(
            np.zeros(1),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.05, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            object_name="tool",
            kind="site",
            tcp_position_required=True,
            tcp_orientation_required=False,
        )

        self.assertTrue(result.success, result.status)
        self.assertTrue(result.relaxed_constraints)
        self.assertEqual(
            solver.candidate_state.calls,
            [[True, False], [True]],
        )
        self.assertIn("optional constraints relaxed", result.status)

    def test_ground_barrier_projects_free_root_drag_above_ground(self):
        state = self.model.create_state()
        start_qpos = state.get_qpos()
        start_position, start_quaternion = state.get_body_pose("robot/pelvis", "body")
        solver = CollisionAwareIKSolver(
            self.model,
            CollisionChecker(self.model),
            collision_drag_substeps=8,
            orientation_weight=0.0,
        )

        result = solver.solve_drag(
            start_qpos,
            start_position,
            start_quaternion,
            start_position + np.array([0.0, 0.0, -0.2]),
            start_quaternion,
            object_name="robot/pelvis",
            kind="body",
            tcp_orientation_weight=0.0,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.accepted_fraction, 1.0)
        self.assertFalse(np.allclose(result.qpos, start_qpos))
        self.assertGreater(
            result.position[2],
            start_position[2] - 0.2,
        )
        self.assertIn("ground barrier raised", result.status)
        self.assertFalse(any(item.blocking for item in result.collisions))


if __name__ == "__main__":
    unittest.main()

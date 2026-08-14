from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from application.trajectory_generation import (
    TrajectorySafetyError,
    generate_trajectory_status,
)
from core.models import RobotModel3D


class _Trajectory:
    def __init__(self):
        self.frames = [SimpleNamespace(time=0.0)]

    def sample_tracks_uniform_dt(self, *, dt, smoothing):
        del dt, smoothing
        return [
            {"time": 0.0, "targets": {}},
            {"time": 1.0, "targets": {}},
        ]


class _RobotModel:
    def __init__(self):
        self.mj_model = SimpleNamespace(nq=2)
        self.home_qpos = np.zeros(2, dtype=float)
        self.free_joints_by_body = {}
        self.joints = {}


class _Backend:
    def __init__(self, robot_model):
        self.adapter = robot_model
        self.states = [
            SimpleNamespace(
                time=0.0,
                qpos=np.array([0.0, 0.1]),
                ik_error=0.001,
                orientation_error=0.002,
            ),
            SimpleNamespace(
                time=1.0,
                qpos=np.array([0.2, 0.3]),
                ik_error=0.003,
                orientation_error=0.004,
            ),
        ]
        self.last_solution = []
        self.export_calls = []
        self.clear_calls = 0

    def solve_trajectory(self, sampled_trajectory):
        self.sampled_trajectory = sampled_trajectory
        self.last_solution = list(self.states)
        return self.last_solution

    def export_last_solution_csv(self, csv_path):
        self.export_calls.append(Path(csv_path))
        Path(csv_path).write_text("published", encoding="utf-8")

    def last_backend_name(self):
        return "test backend"

    def clear_last_solution(self):
        self.clear_calls += 1
        self.last_solution.clear()


def _report(*, blocking, time=0.5):
    severity = "blocking" if blocking else "advisory"
    collision = SimpleNamespace(
        diagnostic_label=(
            "left_forearm ↔ torso (4.8 mm penetration, "
            f"{severity})"
        )
    )
    return SimpleNamespace(
        sample_index=0,
        segment_index=0,
        segment_fraction=0.5,
        time=time,
        collisions=(collision,),
    )


class TrajectoryGenerationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.robot_model = _RobotModel()
        self.backend = _Backend(self.robot_model)
        self.trajectory = _Trajectory()

    def test_blocking_interval_collision_clears_solution_before_export(self):
        blocking_report = _report(blocking=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.csv"
            output.write_text("existing safe export", encoding="utf-8")

            with patch(
                "application.trajectory_generation."
                "adaptive_trajectory_collision_reports",
                return_value=(blocking_report, blocking_report),
            ) as validate:
                with self.assertRaises(TrajectorySafetyError) as caught:
                    generate_trajectory_status(
                        self.trajectory,
                        self.backend,
                        smoothing=0.0,
                        export_dt=1.0,
                        csv_path=output,
                    )

            args, kwargs = validate.call_args
            self.assertIs(args[0], self.robot_model)
            np.testing.assert_allclose(args[1][0], self.backend.states[0].qpos)
            np.testing.assert_allclose(args[1][1], self.backend.states[1].qpos)
            self.assertEqual(kwargs["times"], [0.0, 1.0])
            self.assertIs(caught.exception.report, blocking_report)
            self.assertIn("between samples 0 and 1", str(caught.exception))
            self.assertIn("0.500 s", str(caught.exception))
            self.assertEqual(self.backend.export_calls, [])
            self.assertEqual(self.backend.clear_calls, 1)
            self.assertEqual(self.backend.last_solution, [])
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "existing safe export",
            )

    def test_advisory_contact_is_returned_and_included_in_status(self):
        warning_report = _report(blocking=False, time=0.25)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.csv"
            with patch(
                "application.trajectory_generation."
                "adaptive_trajectory_collision_reports",
                return_value=(warning_report, None),
            ):
                result = generate_trajectory_status(
                    self.trajectory,
                    self.backend,
                    smoothing=0.0,
                    export_dt=1.0,
                    csv_path=output,
                )

            self.assertIs(result.safety_warning_report, warning_report)
            self.assertIn("Motion safety warning", result.status_text)
            self.assertIn("0.250 s", result.status_text)
            self.assertIn("4.8 mm penetration, advisory", result.status_text)
            self.assertEqual(self.backend.export_calls, [output.resolve()])
            self.assertEqual(output.read_text(encoding="utf-8"), "published")
            self.assertEqual(self.backend.clear_calls, 0)

    def test_missing_model_fails_closed_and_discards_solution(self):
        self.backend.adapter = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.csv"
            with self.assertRaisesRegex(
                TrajectorySafetyError,
                "no active robot model",
            ):
                generate_trajectory_status(
                    self.trajectory,
                    self.backend,
                    smoothing=0.0,
                    export_dt=1.0,
                    csv_path=output,
                )

            self.assertFalse(output.exists())
            self.assertEqual(self.backend.export_calls, [])
            self.assertEqual(self.backend.clear_calls, 1)
            self.assertEqual(self.backend.last_solution, [])

    def test_generated_ground_penetration_is_projected_before_export(self):
        robot_model = RobotModel3D()
        backend = _Backend(robot_model)
        free_joint = next(iter(robot_model.free_joints_by_body.values()))
        z_address = free_joint.qpos_address + 2
        states = []
        for time in (0.0, 1.0):
            qpos = robot_model.home_qpos.copy()
            qpos[z_address] -= 0.02
            states.append(SimpleNamespace(
                time=time,
                qpos=qpos,
                ik_error=0.0,
                orientation_error=0.0,
            ))
        backend.states = states
        original_z = states[0].qpos[z_address]

        with tempfile.TemporaryDirectory() as directory:
            result = generate_trajectory_status(
                self.trajectory,
                backend,
                smoothing=0.0,
                export_dt=1.0,
                csv_path=Path(directory) / "ground-corrected.csv",
            )

        self.assertEqual(result.ground_correction_count, 2)
        self.assertGreater(states[0].qpos[z_address], original_z)
        self.assertGreater(states[1].qpos[z_address], original_z)
        self.assertIn("Ground barrier: raised 2", result.status_text)

    def test_ground_projection_never_rewrites_exact_keyframe_anchor(self):
        robot_model = RobotModel3D()
        backend = _Backend(robot_model)
        free_joint = next(iter(robot_model.free_joints_by_body.values()))
        z_address = free_joint.qpos_address + 2
        qpos = robot_model.home_qpos.copy()
        qpos[z_address] -= 0.02
        backend.states = [
            SimpleNamespace(
                time=time,
                qpos=qpos.copy(),
                ik_error=0.0,
                orientation_error=0.0,
            )
            for time in (0.0, 1.0)
        ]

        class AnchoredTrajectory(_Trajectory):
            def sample_tracks_uniform_dt(self, *, dt, smoothing):
                samples = super().sample_tracks_uniform_dt(
                    dt=dt, smoothing=smoothing
                )
                for sample in samples:
                    sample["qpos_anchor"] = robot_model.home_qpos.copy()
                return samples

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                TrajectorySafetyError, "exact Keyframe anchor"
            ):
                generate_trajectory_status(
                    AnchoredTrajectory(),
                    backend,
                    smoothing=0.0,
                    export_dt=1.0,
                    csv_path=Path(directory) / "blocked-anchor.csv",
                )

        self.assertEqual(backend.export_calls, [])
        self.assertEqual(backend.clear_calls, 1)


if __name__ == "__main__":
    unittest.main()

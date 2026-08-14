import tempfile
from pathlib import Path
import unittest

import numpy as np

from application.backend_interface import BackendInterface, FallbackPolicy
from application.timeslice_service import capture_timeslice_from_committed_pose
from application.trajectory_generation import generate_trajectory_status
from core.models import MuJoCoRobotAdapter
from core.models.model import RobotStateTimeline
from core.trajectory import Trajectory


class ModelAgnosticGenerationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MuJoCoRobotAdapter("g1")

    def _add_committed_timeslice(self, trajectory, timeline, time, qpos):
        state = self.adapter.create_state()
        state.set_qpos(qpos)
        timeline.set_state(time, state.get_qpos())
        frames = capture_timeslice_from_committed_pose(
            state,
            time=time,
            phase="test",
            frame_names=self.adapter.trajectory_frames,
            frame_bindings=self.adapter.logical_frame_bindings,
        )
        for frame in frames:
            trajectory.add_frame(frame)

    def test_generated_motion_preserves_all_seven_g1_arm_joints_at_anchors(self):
        trajectory = Trajectory()
        timeline = RobotStateTimeline(self.adapter)
        first = self.adapter.home_qpos.copy()
        second = self.adapter.home_qpos.copy()
        values = {
            "left_shoulder_pitch_joint": (0.21, 0.22),
            "left_shoulder_roll_joint": (0.19, 0.18),
            "left_shoulder_yaw_joint": (0.01, 0.02),
            "left_elbow_joint": (0.59, 0.58),
            "left_wrist_roll_joint": (0.01, 0.02),
            "left_wrist_pitch_joint": (-0.01, -0.02),
            "left_wrist_yaw_joint": (0.01, 0.02),
        }
        for name, (first_value, second_value) in values.items():
            address = self.adapter.joints[name].qpos_address
            first[address] = first_value
            second[address] = second_value

        self._add_committed_timeslice(trajectory, timeline, 0.0, first)
        self._add_committed_timeslice(trajectory, timeline, 0.1, second)
        backend = BackendInterface(
            mj_model=self.adapter.mj_model,
            adapter=self.adapter,
            fallback_policy=FallbackPolicy.ERROR,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.csv"
            result = generate_trajectory_status(
                trajectory,
                backend,
                smoothing=0.0,
                export_dt=0.05,
                csv_path=output,
                state_timeline=timeline,
            )
            saved = np.loadtxt(output, delimiter=",")

        np.testing.assert_allclose(result.result_states[0].qpos, first)
        np.testing.assert_allclose(result.result_states[-1].qpos, second)
        self.assertEqual(saved.shape[1], self.adapter.mj_model.nq + 1)
        self.assertIn("Exact committed qpos anchors: 2", result.status_text)
        self.assertIn("hierarchical pose/posture IK", result.result_states[1].status)

    def test_off_grid_committed_anchor_is_rejected_before_generation(self):
        trajectory = Trajectory()
        timeline = RobotStateTimeline(self.adapter)
        self._add_committed_timeslice(
            trajectory, timeline, 0.0, self.adapter.home_qpos
        )
        self._add_committed_timeslice(
            trajectory, timeline, 0.15, self.adapter.home_qpos
        )
        backend = BackendInterface(
            mj_model=self.adapter.mj_model,
            adapter=self.adapter,
            fallback_policy=FallbackPolicy.ERROR,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not aligned"):
                generate_trajectory_status(
                    trajectory,
                    backend,
                    smoothing=0.0,
                    export_dt=0.02,
                    csv_path=Path(directory) / "generated.csv",
                    state_timeline=timeline,
                )


if __name__ == "__main__":
    unittest.main()

import unittest

from gui.trajectory import TargetFrame, Trajectory


class TrajectorySmoothingTests(unittest.TestCase):
    def make_corner_trajectory(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            y=0.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=1.0,
            y=1.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=2.0,
            frame_name="pelvis",
            x=2.0,
            y=0.0,
            z=0.0,
        ))
        return trajectory

    def sample_pelvis_at(self, trajectory, time_value, smoothing):
        samples = trajectory.sample_tracks_uniform_dt(
            dt=0.5,
            smoothing=smoothing,
        )
        for sample in samples:
            if abs(sample["time"] - time_value) <= 1e-9:
                return sample["targets"]["pelvis"]
        raise AssertionError(f"missing sample at t={time_value}")

    def test_zero_smoothing_preserves_linear_position_interpolation(self):
        trajectory = self.make_corner_trajectory()

        midpoint = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.0)

        self.assertAlmostEqual(midpoint.x, 0.5)
        self.assertAlmostEqual(midpoint.y, 0.5)
        self.assertAlmostEqual(midpoint.z, 0.0)

    def test_full_smoothing_rounds_corner_without_moving_keyframes(self):
        trajectory = self.make_corner_trajectory()

        midpoint = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)
        keyframe = self.sample_pelvis_at(trajectory, 1.0, smoothing=1.0)

        self.assertAlmostEqual(midpoint.x, 0.5)
        self.assertGreater(midpoint.y, 0.5)
        self.assertAlmostEqual(keyframe.x, 1.0)
        self.assertAlmostEqual(keyframe.y, 1.0)

    def test_partial_smoothing_blends_linear_and_smooth_positions(self):
        trajectory = self.make_corner_trajectory()

        linear = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.0)
        smooth = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)
        halfway = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.5)

        self.assertAlmostEqual(halfway.x, (linear.x + smooth.x) * 0.5)
        self.assertAlmostEqual(halfway.y, (linear.y + smooth.y) * 0.5)
        self.assertAlmostEqual(halfway.z, (linear.z + smooth.z) * 0.5)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.view_g1_mujoco import RAW_QPOS_KEY, TrajectoryPlayer, load_trajectory_csv


class MujocoCsvPlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path("models/g1_29dof.xml").resolve()
        cls.model = mujoco.MjModel.from_xml_path(str(model_path))

    def test_loads_headerless_qpos_pose(self):
        source = Path("crawl_home_qpos_t0.5.csv").resolve()
        _, rows = load_trajectory_csv(source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0][RAW_QPOS_KEY]), self.model.nq)

        data = mujoco.MjData(self.model)
        player = TrajectoryPlayer(self.model, data)
        player.load_csv(source)
        np.testing.assert_allclose(
            data.qpos,
            np.loadtxt(source, delimiter=","),
        )

    def test_rejects_raw_pose_for_wrong_model_nq(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("1,2,3\n", encoding="utf-8")
            player = TrajectoryPlayer(self.model, mujoco.MjData(self.model))
            with self.assertRaisesRegex(ValueError, "expected 36 qpos values"):
                player.load_csv(source)


if __name__ == "__main__":
    unittest.main()

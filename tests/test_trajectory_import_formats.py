from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from application.csv_io import TrajectoryExport
from application.trajectory_export_formats import (
    G1_JOINT_ORDER,
    export_dsms_trajectory,
    export_mjlab_trajectory,
)
from application.trajectory_import_formats import (
    read_dsms_trajectory,
    read_mjlab_trajectory,
)


class FakeG1Adapter:
    def __init__(self):
        self.mj_model = SimpleNamespace(nq=36)
        self.actuated_joints = list(G1_JOINT_ORDER)
        self.joints = {
            name: SimpleNamespace(qpos_address=7 + index)
            for index, name in enumerate(G1_JOINT_ORDER)
        }
        self.free_joints_by_body = {1: SimpleNamespace(qpos_address=0)}


def sample_export(times=(0.0, 0.02)):
    qposes = []
    for sample_index, _time in enumerate(times):
        qpos = np.zeros(36, dtype=float)
        qpos[:3] = (1.0 + sample_index, 2.0, 3.0)
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        qpos[7:] = np.arange(29, dtype=float) + sample_index * 100.0
        qposes.append(qpos)
    return TrajectoryExport(
        expected_qpos_count=36,
        times=tuple(times),
        qposes=tuple(qposes),
        source_name="test trajectory",
        preview_active=False,
    )


class TrajectoryImportFormatTests(unittest.TestCase):
    def test_dsms_folder_roundtrip_preserves_times_and_qposes(self):
        source = sample_export()
        with tempfile.TemporaryDirectory() as directory:
            export_dsms_trajectory(
                directory,
                source,
                dof=29,
                base_qpos_address=0,
            )
            loaded = read_dsms_trajectory(
                directory,
                36,
                expected_dof=29,
            )

        self.assertEqual(loaded.source_format, "dsms")
        np.testing.assert_allclose(loaded.times, source.times)
        np.testing.assert_allclose(loaded.qposes, source.qposes)

    def test_dsms_rejects_mismatched_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savetxt(path / "time.csv", (0.0, 0.01), delimiter=",")
            np.savetxt(
                path / "qpos_29dof.csv",
                np.zeros((1, 36)),
                delimiter=",",
            )
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                read_dsms_trajectory(path, 36, expected_dof=29)

    def test_dsms_rejects_wrong_active_model_dof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savetxt(path / "time.csv", (0.0,), delimiter=",")
            np.savetxt(
                path / "qpos_23dof.csv",
                np.zeros((1, 36)),
                delimiter=",",
            )
            with self.assertRaisesRegex(ValueError, "qpos_29dof.csv"):
                read_dsms_trajectory(path, 36, expected_dof=29)

    def test_mjlab_roundtrip_restores_mujoco_order_and_supplied_interval(self):
        adapter = FakeG1Adapter()
        source = sample_export()
        with tempfile.TemporaryDirectory() as directory:
            result = export_mjlab_trajectory(
                Path(directory) / "motion.csv",
                source,
                adapter,
            )
            loaded = read_mjlab_trajectory(
                result.paths[0],
                adapter,
                sample_interval=0.01,
            )

        self.assertEqual(loaded.source_format, "mjlab")
        np.testing.assert_allclose(loaded.times, (0.0, 0.01))
        np.testing.assert_allclose(loaded.qposes, source.qposes)

    def test_mjlab_rejects_invalid_sample_interval(self):
        adapter = FakeG1Adapter()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "motion.csv"
            np.savetxt(source, np.zeros((1, 36)), delimiter=",")
            with self.assertRaisesRegex(ValueError, "sample interval"):
                read_mjlab_trajectory(source, adapter, 0.0)


if __name__ == "__main__":
    unittest.main()

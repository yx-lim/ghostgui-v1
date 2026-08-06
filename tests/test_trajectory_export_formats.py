from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from application.csv_io import TrajectoryExport
from application.trajectory_export_formats import (
    G1_JOINT_ORDER,
    export_dsms_trajectory,
    export_mjlab_trajectory,
    mjlab_compatibility_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeG1Adapter:
    def __init__(self):
        self.mj_model = SimpleNamespace(nq=36)
        self.actuated_joints = list(G1_JOINT_ORDER)
        self.joints = {
            name: SimpleNamespace(qpos_address=7 + index)
            for index, name in enumerate(G1_JOINT_ORDER)
        }
        self.free_joints_by_body = {
            1: SimpleNamespace(qpos_address=0),
        }


def sample_export(times=(0.0, 0.02)):
    qposes = []
    for sample_index, _time in enumerate(times):
        qpos = np.zeros(36, dtype=float)
        qpos[:3] = (1.0 + sample_index, 2.0, 3.0)
        qpos[3:7] = (2.0, 0.0, 0.0, 0.0)
        qpos[7:] = np.arange(29, dtype=float) + sample_index * 100.0
        qposes.append(qpos)
    return TrajectoryExport(
        expected_qpos_count=36,
        times=tuple(times),
        qposes=tuple(qposes),
        source_name="generated trajectory",
        preview_active=False,
    )


class TrajectoryExportFormatTests(unittest.TestCase):
    def test_dsms_writes_qpos_and_time_folder_files(self):
        with tempfile.TemporaryDirectory() as directory:
            result = export_dsms_trajectory(
                directory,
                sample_export(),
                dof=29,
                base_qpos_address=0,
            )

            self.assertEqual(
                [path.name for path in result.paths],
                ["qpos_29dof.csv", "time.csv"],
            )
            qposes = np.loadtxt(result.paths[0], delimiter=",", ndmin=2)
            times = np.loadtxt(result.paths[1], delimiter=",", ndmin=1)

        self.assertEqual(qposes.shape, (2, 36))
        np.testing.assert_allclose(times, (0.0, 0.02))
        np.testing.assert_allclose(qposes[:, 3], (1.0, 1.0))
        self.assertAlmostEqual(result.input_fps, 50.0)

    def test_dsms_rejects_nonuniform_editable_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Export interval"):
                export_dsms_trajectory(
                    directory,
                    sample_export((0.0, 0.02, 0.05)),
                    dof=29,
                    base_qpos_address=0,
                )

    def test_mjlab_writes_xyzw_and_named_g1_joint_order(self):
        adapter = FakeG1Adapter()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "motion.csv"
            result = export_mjlab_trajectory(
                output,
                sample_export(),
                adapter,
            )
            rows = np.loadtxt(result.paths[0], delimiter=",", ndmin=2)

        self.assertEqual(rows.shape, (2, 36))
        np.testing.assert_allclose(rows[0, :7], (1, 2, 3, 0, 0, 0, 1))
        np.testing.assert_allclose(rows[0, 7:], np.arange(29, dtype=float))
        self.assertAlmostEqual(result.input_fps, 50.0)

    def test_mjlab_rejects_incompatible_model_contract(self):
        adapter = FakeG1Adapter()
        adapter.mj_model.nq = 19

        self.assertIn("requires G1 29-DoF", mjlab_compatibility_error(adapter))

    def test_standalone_converter_scripts_remain_directly_runnable(self):
        with tempfile.TemporaryDirectory() as directory:
            working_dir = Path(directory)
            for script_name in (
                "convert_ghostgui_to_dsms.py",
                "ghostgui_to_mjlab.py",
            ):
                with self.subTest(script=script_name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts" / script_name),
                            "--help",
                        ],
                        cwd=working_dir,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("usage:", completed.stdout)

            export = sample_export()
            input_path = working_dir / "ghostgui.csv"
            np.savetxt(
                input_path,
                np.column_stack((export.times, np.asarray(export.qposes))),
                delimiter=",",
            )
            commands = (
                (
                    "convert_ghostgui_to_dsms.py",
                    [str(input_path), "dsms", "--dof", "29", "--nq", "36"],
                ),
                (
                    "ghostgui_to_mjlab.py",
                    [str(input_path), "mjlab.csv"],
                ),
            )
            for script_name, arguments in commands:
                with self.subTest(conversion=script_name):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts" / script_name),
                            *arguments,
                        ],
                        cwd=working_dir,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertTrue((working_dir / "dsms" / "qpos_29dof.csv").exists())
            self.assertTrue((working_dir / "dsms" / "time.csv").exists())
            self.assertTrue((working_dir / "mjlab.csv").exists())


if __name__ == "__main__":
    unittest.main()

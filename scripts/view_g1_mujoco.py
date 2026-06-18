"""
Standalone MuJoCo viewer for G1 29-DoF model.

Run directly:
    python3 scripts/view_g1_mujoco.py

Or launch from the PySide6 GUI.
"""

from pathlib import Path
import sys

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


def main():
    if not MODEL_PATH.exists():
        print(f"Could not find model: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Your XML has an init_state keyframe.
    # This loads the robot's intended initial pose.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    print("Model loaded.")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print("nbody:", model.nbody)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
"""
python3 view_g1.py
"""

from pathlib import Path
import mujoco
import mujoco.viewer


# ------------------------------------------------------------
# Path to your robot XML
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


# ------------------------------------------------------------
# Load MuJoCo model and data
# ------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# ------------------------------------------------------------
# If the XML has a keyframe, load the first keyframe
# Your XML has an init_state keyframe, so this should set
# the robot to the intended initial pose.
# ------------------------------------------------------------
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)


# ------------------------------------------------------------
# Print useful model information
# ------------------------------------------------------------
print("Loaded model:", MODEL_PATH)
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)


# ------------------------------------------------------------
# Launch interactive MuJoCo viewer
# ------------------------------------------------------------
mujoco.viewer.launch(model, data)

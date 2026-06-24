"""
Reference-frame bindings for the real MuJoCo robot model.

The GUI frame names are user-facing targets. This module maps them to concrete
MuJoCo bodies/sites so new targets can start from the real robot geometry
instead of the simplified 2D stickman skeleton.
"""

from pathlib import Path

try:
    import mujoco
except ImportError:
    mujoco = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


REFERENCE_FRAME_BINDINGS = {
    "pelvis": ("body", "robot/pelvis"),
    "torso": ("body", "robot/torso_link"),
    "left_foot": ("site", "robot/left_foot"),
    "right_foot": ("site", "robot/right_foot"),
    "left_hand": ("site", "robot/left_palm"),
    "right_hand": ("site", "robot/right_palm"),
}


class MujocoReferenceFrames:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.data = None
        self.error = None
        self.load()

    def load(self):
        if mujoco is None:
            self.error = "mujoco Python package is not installed."
            return

        if not self.model_path.exists():
            self.error = f"Robot model not found: {self.model_path}"
            return

        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)

            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            mujoco.mj_forward(self.model, self.data)
            self.error = None
        except Exception as exc:
            self.model = None
            self.data = None
            self.error = str(exc)

    def position_for_frame(self, frame_name):
        if self.model is None or self.data is None:
            return None

        binding = REFERENCE_FRAME_BINDINGS.get(frame_name)
        if binding is None:
            return None

        kind, mujoco_name = binding

        if kind == "site":
            object_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                mujoco_name,
            )
            if object_id < 0:
                return None
            pos = self.data.site_xpos[object_id]
        else:
            object_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                mujoco_name,
            )
            if object_id < 0:
                return None
            pos = self.data.xpos[object_id]

        return float(pos[0]), float(pos[1]), float(pos[2])

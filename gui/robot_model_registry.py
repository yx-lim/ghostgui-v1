"""Robot model registrations and logical-frame naming hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RobotModelInfo:
    key: str
    display_name: str
    model_type: str
    model_path: Path
    root_body_candidates: tuple[str, ...]
    root_joint_candidates: tuple[str, ...] = ()
    logical_frames: dict[str, tuple[str, ...]] = field(default_factory=dict)
    package_map: dict[str, Path] = field(default_factory=dict)
    ignored_body_tokens: tuple[str, ...] = (
        "camera", "imu", "radar", "rotor", "logo", "sensor", "contour",
        "constraint", "support",
    )
    home_joints: dict[str, float] = field(default_factory=dict)


ROBOT_MODELS = {
    "g1": RobotModelInfo(
        key="g1",
        display_name="Unitree G1",
        model_type="humanoid",
        model_path=PROJECT_ROOT / "models" / "g1_29dof.xml",
        root_body_candidates=("robot/pelvis", "pelvis", "torso", "base"),
        root_joint_candidates=("robot/root", "root", "floating_base"),
        logical_frames={
            "pelvis": ("robot/pelvis", "pelvis", "base", "base_link"),
            "torso": ("robot/torso_link", "torso_link", "torso", "trunk"),
            "left_hand": ("robot/left_palm", "left_palm", "left_hand"),
            "right_hand": ("robot/right_palm", "right_palm", "right_hand"),
            "left_foot": ("robot/left_foot", "left_foot", "left_ankle_roll_link"),
            "right_foot": ("robot/right_foot", "right_foot", "right_ankle_roll_link"),
        },
    ),
    "go2": RobotModelInfo(
        key="go2",
        display_name="Unitree Go2",
        model_type="quadruped",
        model_path=PROJECT_ROOT / "models" / "go2_description.urdf",
        root_body_candidates=("base", "trunk", "base_link", "world"),
        root_joint_candidates=("floating_base", "root", "freejoint"),
        logical_frames={
            "base": ("base", "trunk", "base_link", "world"),
            "trunk": ("base", "trunk", "base_link", "world"),
            "FL_foot": ("FL_foot", "FL_calf"),
            "FR_foot": ("FR_foot", "FR_calf"),
            "RL_foot": ("RL_foot", "RL_calf"),
            "RR_foot": ("RR_foot", "RR_calf"),
        },
        package_map={
            # The vendored ROS package stores its DAE files in a flattened
            # directory instead of go2_description/dae.
            "go2_description": PROJECT_ROOT / "models" / "go2_assets",
        },
        home_joints={
            **{f"{leg}_thigh_joint": 0.8 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.5 for leg in ("FL", "FR", "RL", "RR")},
        },
    ),
}


def get_model_info(model: str | RobotModelInfo = "g1") -> RobotModelInfo:
    if isinstance(model, RobotModelInfo):
        return model
    try:
        return ROBOT_MODELS[model]
    except KeyError as exc:
        raise KeyError(
            f"Unknown robot model {model!r}; choose one of {', '.join(ROBOT_MODELS)}"
        ) from exc

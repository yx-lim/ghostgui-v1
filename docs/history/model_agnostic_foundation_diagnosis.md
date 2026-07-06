# Model-agnostic foundation diagnosis (historical)

The application starts in `run_gui.py` and is assembled by
`gui.main_window.RobotGuiMainWindow`.  Before this refactor the same G1 MJCF
path was independently selected in `robot_model_3d.py`, `model_reference.py`,
`backend_interface.py`, `viewer_3d_mujoco.py`, and the external viewer script.
`RobotModel3D` already used MuJoCo metadata for scalar joints, but the other
subsystems did not share a robot description.

G1 assumptions were spread across the six logical frame bindings
(`pelvis`, torso, hands, and feet), a fixed 29-joint backend list, floating-base
qpos assumptions, G1 labels/paths in the CSV viewer, and the frame combo in the
left control panel.  `viewer_2d_stickman.py` additionally defined a complete
humanoid pose and limb topology in Python, independent of the loaded MuJoCo
model.

The live 3D editor considered configured G1 logical frames first, then exposed
every MuJoCo body and site as a possible target.  The legacy 2D view only knew
the six hardcoded frame names.  Asset lookup worked for G1 because absolute
paths happened to reach its MJCF, while no shared policy existed for another
model or URDF package assets.

Supporting Go2 and future robots therefore requires one registry-selected
adapter to own path resolution, model type, joints, logical frame candidate
matching, root/end-effector metadata, home state, and the MuJoCo kinematic tree.
Both viewers and controls must consume that adapter.  The 2D view can then draw
projected MuJoCo body positions and parent-child edges, with a generic graph
fallback, instead of selecting a robot-specific drawing class.

The supplied Go2 source is a URDF whose visual blocks contain repeated material
elements and `package://` DAE meshes. MuJoCo cannot compile that file directly.
Go2 is therefore prepared at build time as `models/go2/go2.xml`, preserving its
12 joints, collision model, logical foot sites, colors, and lit scene without
runtime conversion. User-imported URDFs use a versioned persistent conversion
cache instead.

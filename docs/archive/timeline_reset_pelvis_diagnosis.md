# 3D reset, timeline editing, and pelvis diagnosis

> Historical note: this diagnosis may not describe the current implementation.
> See [Preview And Keyframe Concepts](../concepts.md) for supported behavior.

## Architecture

- `gui/viewer_3d.py::RobotCanvas3D` handles gizmo mouse interaction and drawing.
- `gui/robot_viewer_3d.py::RobotViewer3D` owns the visible `RobotState3D`, selected
  target, joint controls, collision-aware IK, and trajectory/ghost controls.
- `gui/robot_model_3d.py::RobotState3D` owns MuJoCo qpos, FK, and Jacobian IK.
- `gui/controls.py::TrajectoryControlPanel` owns the current `Time [s]` value;
  `gui/trajectory.py::Trajectory` stores logical target-frame keyframes.
- The removed 2D viewers dragged a logical target position. The 3D viewer maps
  logical `pelvis` to MuJoCo body `robot/pelvis` through `FRAME_BINDINGS`.

## Root causes

1. There is no 3D reset action. MuJoCo home state is loaded only when a
   `RobotState3D` is constructed, and that logic is not exposed as one reusable
   reset method.
2. The time slider is not connected to `RobotViewer3D`. The viewer has one
   mutable qpos rather than qpos states keyed by time, so `0.2s` neither loads
   nor creates an editable robot-state keyframe. All accepted IK continues to
   mutate that one initial state.
3. Pelvis naming/mapping is already correct, but generic IK only includes
   controllable hinge/slide joints. The pelvis is attached to
   `robot/floating_base_joint` (a MuJoCo free joint), so internal joint columns
   cannot translate it and the solve cannot converge.

## Minimal fix

- Store the model home qpos and expose `RobotState3D.reset_to_default()`.
- Add a small `RobotStateTimeline` keyed by GUI time. Exact states load directly;
  missing times are created with MuJoCo manifold-aware qpos interpolation (or a
  nearest-state clone at the ends).
- Connect the time control to the 3D timeline and save accepted IK, joint-slider,
  and reset changes into the active time only.
- Detect a selected body driven by a free joint and set that free-joint position
  and quaternion in the temporary candidate state before collision checking.

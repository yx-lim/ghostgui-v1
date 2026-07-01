# Interactive gizmo, collision, and flicker diagnosis

## Current ownership

- `gui/viewer_3d.py::RobotCanvas3D` draws the target marker/gizmo, projects it
  to screen coordinates, handles mouse drag state, and schedules OpenGL paints.
- `gui/robot_viewer_3d.py::RobotViewer3D` owns the selected body/site, calls
  `RobotState3D.solve_ik`, synchronizes joint sliders, and owns the one playback
  timer.
- `gui/robot_model_3d.py::RobotState3D` owns MuJoCo qpos/FK/IK state.
- `TrajectoryGhostRenderer` caches sampled FK transforms; robot mesh display
  lists are compiled once by `RobotCanvas3D`.

## Root causes

1. The existing gizmo is three plain lines. Picking checks an 18-pixel circle
   around each endpoint, then falls back to a separate 32-pixel invisible center
   target drag. There are no hover states, arrow shapes, rotation rings, or
   orientation drag signal.
2. IK mutates the visible robot state directly. There is no candidate state,
   no `data.ncon` inspection, and a failed IK solve can leave partial qpos
   changes visible.
3. Geometry is not recreated per paint and there is only one playback timer.
   The actual overlapping draw paths are the problem: the legacy red yaw line
   lies directly on the red X gizmo line, and both MuJoCo visual geoms (group 2)
   and translucent collision geoms (group 3) are rendered together. Coincident
   lines/surfaces and a ghost at the exact current pose can shimmer or z-fight.
   Accepted 3D targets were also round-tripped through centimetre-resolution
   sliders during each mouse move, snapping precise gizmo motion, while each
   move synchronously refreshed every viewer, the table, and status panel.

## Minimal implementation

- Add a persistent world-axis `TransformGizmo` with explicit hover/drag states,
  screen-space line/ring picking, quaternion ring rotation, and no center drag.
- Add a persistent candidate `RobotState3D` plus `CollisionChecker`; solve each
  drag in substeps, accept only converged collision-free qpos, and retain the
  furthest valid intermediate pose.
- Render visual geom group 2 only, remove the duplicate legacy yaw line, skip a
  ghost coincident with the main pose, and continue using Qt's coalesced
  `update()` scheduling and the existing single playback timer. Synchronize XYZ
  controls at millimetre resolution so they do not quantize live dragging, and
  defer the full cross-view refresh until mouse release.

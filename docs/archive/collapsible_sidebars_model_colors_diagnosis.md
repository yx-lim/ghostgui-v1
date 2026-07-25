# Collapsible sidebars and model-color diagnosis

> Historical note: this diagnosis may not describe the current implementation.
> See the [User Guide](../user_guide.md) for supported behavior.

## Sidebars

`RobotGuiMainWindow` places `TrajectoryControlPanel`, the viewer tabs, and the
trajectory/backend status `QGroupBox` directly in a `QHBoxLayout`. They are plain
layout widgets—not docks, splitters, tabs, or collapsible containers—so their
visibility and widths have no independent UI state.

The fix uses horizontal `QSplitter` instances and persistent
`CollapsibleSidebar` wrappers for Frames, Status, and the 3D tab's own controls.
Collapse hides only sidebar content and leaves a
small labeled handle; expand shows the same widget and restores its remembered
width. Nothing is deleted or reconstructed, so the `QOpenGLWidget`, MuJoCo
model, pose, gizmo, timeline, and playback objects survive a toggle.

## Robot appearance

The live 3D tab is a custom fixed-function OpenGL renderer in
`RobotCanvas3D`, not MuJoCo's native renderer. It currently colors every mesh
from `model.geom_rgba`. In this compiled model all 35 visual geoms retain
MuJoCo's default geom color `(0.5, 0.5, 0.5, 1)`, while their real material is
selected by `model.geom_matid`:

- `robot/silver`: `(0.7, 0.7, 0.7, 1)`
- `robot/black`: `(0.2, 0.2, 0.2, 1)`

The fix resolves `mat_rgba[geom_matid]` when a material exists and falls back
to `geom_rgba` for unmaterialed geoms. Ghost alpha remains a draw-time
multiplier, so it never mutates main-robot material data. The XML's only texture
is the checkerboard ground plane; robot visual materials contain no textures.

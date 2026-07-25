# GhostGUI User Guide

GhostGUI helps you create robot motion by editing target frames, previewing IK
results on the live MuJoCo model, saving time slices, generating trajectories,
and exporting qpos data for validation or downstream tools.

## Launch

From the repository checkout:

```bash
python3 scripts/run_gui.py --model g1
```

If GhostGUI is installed as an editable package, you can also run:

```bash
ghostgui --model g1
```

Common bundled models include `g1`, `go2`, `h2`, and `z1`.

## Core Concepts

**Robot model** controls the available bodies, sites, joints, logical frames,
home pose, and MuJoCo geometry.

**Target robot frame** is the frame you are editing. For a humanoid this might
be `left_hand`, `right_hand`, `left_foot`, `right_foot`, `torso`, or `pelvis`.

**Preview state** is the orange robot. It is a temporary IK result from dragging
the gizmo or changing pose controls.

**Committed state** is the accepted robot pose at the active time. The timeline,
generation, and exports use committed states, not unsaved previews.

**Preview** checks or plans the path from the committed pose to the orange
preview. It does not save the pose.

**Slice** accepts the current preview if needed, records the committed pose at
the active time, and advances the timeline by the configured slice step.

**Generated trajectory** is a sampled sequence made from saved time slices and
IK. This is the usual export target.

## First Motion Workflow

1. Choose a robot from the **Robot** menu.
2. Select a **Target robot frame**, such as `left_hand`.
3. Move the target with the 3D transform gizmo or the pose sliders.
4. Inspect the orange preview robot.
5. Click **Preview** if you want to plan/check the path before storing it.
6. Click **Slice** to commit the preview and save it at the current time.
7. Move to another time, adjust the frame again, and click **Slice** again.
8. Click **Generate** or **Generate / Simulate**.
9. Use **File -> Export -> Trajectory** to save timed qpos rows.

The beginner workflow is:

```text
select frame -> move target -> orange preview -> Slice -> Generate -> Export
```

## GUI Sections

### Menu Bar

Create/open/save projects, choose the active robot model, import or export data,
switch views, and open help. Import supports models, qpos poses, and
trajectories. Export can save a single committed qpos pose or a timed
trajectory.

### Target

Choose the logical robot frame to edit, or use **Advanced target** to select
another body or site exposed by the model.

### Editing Mode

Choose **End Effector** to edit the target frame with X, Y, Z, Roll, Pitch, and
Yaw controls or the 3D transform gizmo. Choose **Joint Angles** to edit the
active robot's joints directly or move the gizmo through IK while watching the
joint values update. Both modes update the orange preview and stay synchronized
as you switch between them.

Both sidebars can be resized from 200 to 400 pixels by dragging their dividers.
Use the divider arrows or the corresponding **View → Left Sidebar** and
**View → Right Sidebar** actions to collapse and restore them. Their last
expanded widths and collapsed states are remembered between sessions.

### Planning

Manage the editable timeline. The active time determines where the next slice is
stored. The single Time slider drives both live scrubbing and playback; the
robot is sampled immediately while dragging, and the derived frame appears as a
readout beside the time. Releasing the slider commits the selected edit time
once. Smoothing, collision substeps, playback opacity, and preview opacity are
configured here. Playback speed changes the viewing rate without changing the
trajectory's stored timestamps. Saved slices become the source material for
generated trajectories.

### Workflow Toolbar

**Preview** plans/checks the current orange preview path.

**Slice** commits the current preview and stores it at the active time.

**Generate** creates a sampled robot trajectory from saved timeline states.

**Play/Pause** controls the active generated or editable timeline.

**Reset** returns the active time to the model home pose.

**Clear** clears the editable trajectory.

**Move/Rotate** select the active transform gizmo in either editing mode.

**Gizmo** shows or hides the transform gizmo. Its visibility preference remains
active when switching editing modes.

**Undo/Redo** navigate recorded editing history.

### Right Sidebar

**Status** shows the latest important event or problem in a compact summary.
Expand **Details** to inspect the latest operation's frame, IK result, solver
metrics, and other diagnostics.

**IK / Constraints** contains solver settings, task weights, collision controls,
and preview controls.

## Keyboard / Mouse Shortcuts

### Editing

**T** switches the 3D transform gizmo to translate mode.

**R** switches the 3D transform gizmo to rotate mode.

### Inline Value Sliders

Drag a filled slider for continuous live adjustment.

Click its left or right half to decrease or increase by one logical step.

Use the arrow keys for one step, Page Up/Down for ten steps, and Home/End for
the minimum or maximum.

Press Enter or F2, or double-click the displayed value, to type directly.
Enter commits the typed value and Esc cancels it.

**E** or **Esc** cancels the current transform drag.

**Shift + drag** gives finer gizmo movement.

**Ctrl + drag** snaps gizmo movement.

### History

**Ctrl+Z** undoes the last recorded action.

**Ctrl+Shift+Z** redoes the last undone action.

### 3D Camera

**Left drag** orbits the camera unless a gizmo handle is active.

**Right drag** pans the camera.

**Middle drag** zooms the camera.

**Mouse wheel** zooms the camera.

## Import And Export

Use **File -> Import** for model, qpos, or trajectory input.

Use **File -> Export** for:

- **Qpos**: the current committed pose as one headerless qpos row.
- **Trajectory**: timed qpos rows from the generated trajectory or editable
  timeline.

GhostGUI deliberately does not export an unaccepted orange preview. Click
**Slice** first when the current preview should become part of the output.

## Troubleshooting

### The orange robot changed, but export did not

The orange robot is still only a preview. Click **Slice** to commit and record
the pose before exporting.

### Preview fails or only moves partway

IK, joint limits, singularity checks, or collision checks may be blocking the
motion. Check **Status** and **IK / Constraints**.

### Generate does not match the intended motion

Check the saved slice times and the active target frame. Use playback ghosts to
inspect the generated path before exporting.

### A model has different target frames

Target frames come from the active model. After switching models, re-check the
**Target robot frame** selector and the selected object details.

## Known Limitations

The first help system is static. It explains the workflow but does not yet
highlight widgets or verify each action. Future guided tutorials can build on
the same concepts with an overlay and action-aware completion checks.

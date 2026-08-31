# GhostGUI User Guide

GhostGUI creates robot motion by editing target frames or joint angles,
previewing the result on a live MuJoCo model, committing keyframes, and
generating a trajectory for playback or export.

## Launch

From an installed checkout:

```bash
ghostgui --model g1
```

The bundled model keys are `g1`, `go2`, `h2`, and `z1`. See
[Installation](install.md) for complete setup and launch instructions.

## Core Concepts

**Robot model** determines the available bodies, sites, joints, logical target
frames, home pose, and 3D geometry.

**Target robot frame** is the body, site, or logical frame being edited, such as
`left_hand`, `right_foot`, `base`, or `tool`.

**Orange preview** is the temporary robot pose produced by the transform gizmo,
pose controls, or joint controls. It is not saved automatically.

**Committed state** is the accepted robot pose at the active time. Keyframes,
trajectory generation, and exports use committed states.

**Preview Path** validates the transition from the committed state to the orange
preview. It displays a ghost path, marks collision samples red, and does not
save anything.

**Commit Keyframe** records the current pose at the active time and advances by
the configured keyframe interval.

For the complete state model, see [Preview And Keyframe Concepts](concepts.md).

## First Motion Workflow

1. Choose a robot from the **Robot** menu.
2. Select a **Target robot frame**, such as `left_hand`, or double-click the
   robot to choose the target to edit.
3. Move the target with the 3D transform gizmo or the pose controls.
4. Inspect the orange preview and the **Status** panel.
5. Optionally select **Preview Path** to validate the transition.
6. Select **Commit Keyframe** to save the pose at the active time.
7. Move to another time and commit another keyframe.
8. Select **Generate**.
9. Use playback to inspect the motion.
10. Use **File → Export** to save a pose or trajectory.

The main workflow is:

```text
select target → edit → orange preview → Commit Keyframe → Generate → Export
```

## Interface

The interface is arranged around the 3D viewport:

```text
┌──────────────────────────── 1. Menu bar ─────────────────────────────┐
├──────────────────────── 2. Workflow toolbar ─────────────────────────┤
│ 3. Left sidebar       │                         │ 5. Right sidebar   │
│ Target                │     4. 3D viewport      │ Status             │
│ End Effector /        │                         │ IK / Constraints   │
│ Joint Angles          │                         │                    │
├───────────────────────┴─────────────────────────┴────────────────────┤
│                  6. Planning and timeline controls                   │
└──────────────────────────────────────────────────────────────────────┘
```

| Region | Purpose |
| --- | --- |
| 1. Menu bar | Projects, robots, import/export, timeline tools, views, and help |
| 2. Workflow toolbar | Preview Path, Commit Keyframe, Generate, playback, and history |
| 3. Left sidebar | Target selection and End Effector or Joint Angles editing |
| 4. 3D viewport | Committed robot, Orange preview, gizmo, and path ghosts |
| 5. Right sidebar | Status diagnostics and IK constraints |
| 6. Planning controls | Time, intervals, smoothing, speed, and display options |

### 1. Menu Bar

Use the menu bar to create, open, and save projects; select the active robot;
import or export data; retime Keyframes; switch views; and open help.

## Timeline Editing

**Insert Time at Current Time** opens a new interval at the active time. All
Keyframes at or after that time move later, and GhostGUI inserts matching hold
anchors at both ends of the new interval. Use this to move a `1–2 s` motion to
`2–3 s` before creating a new `1–2 s` motion.

**Shift Entire Motion** moves every logical target and committed robot-pose
Keyframe by one offset. Positive offsets prepend room for new motion. A negative
offset is rejected if any Keyframe would move before `0 s`.

**Move Time Range** moves every Keyframe in an inclusive source range to a
non-overlapping destination. GhostGUI checks the complete operation first and
rejects it if the destination already has a conflicting logical target or robot
pose. Adjacent ranges are allowed, so `1–2 s` can move to `2–3 s`.

**Scale Time Range** changes actual motion speed by scaling Keyframe timestamps
around the range start. Select **Scale entire motion** for the complete timeline,
or clear it and enter a range. `2.00×` halves the duration and `0.50×` doubles
it. Unlike visual Playback speed or DSMS motion speed, this changes the
authoritative timeline and therefore affects Generate and every export format.

**Copy Motion Range…** stores the committed Keyframes in an inclusive source
range as a transient, model-specific Motion Clip. The clip contains both the
logical target frames and committed robot poses, including Joint Angles. If a
range boundary lies between Keyframes, GhostGUI samples the committed robot pose
and uses forward kinematics to add logical target anchors, making the clip
self-contained. Exact target Keyframes at a boundary are preserved. The clip
does not contain the Orange preview or generated trajectory samples. Copying
does not change the project, mark it dirty, or create an Undo/Redo action.

**Paste Motion at Current Time** preserves the copied ordering and maps each
source time with `current time + (source time - source start)`. To append a
forward copy directly to the source, the first and last states must form a
closed seam. An identical shared seam is coalesced into one Keyframe; a
different target or committed robot pose at the shared boundary rejects the
complete paste.

**Paste Motion Reversed at Current Time** maps each source time with `current
time + (source end - source time)`. This reverses time order only: it does not
mirror positions, negate Joint Angles, or otherwise transform a pose. Appending
a reversed copy to an `A → B` motion produces a continuous `A → B → A`
motion because the shared `B` seam is identical.

**Repeat Motion…** applies a selected number of additional copies in one edit.
**Forward** keeps the original time order for every added copy and therefore
requires closed seams. **Ping-pong** alternates reversed and forward copies so
an `A → B` source continues as `B → A → B`. The count is the number of
copies added after the original, not the total number of cycles.

The dialogs snap values to the **Export interval** by default; clear the
checkbox when exact off-grid timing is intentional. Each successful retiming,
Paste, or Repeat operation updates logical targets and committed robot poses
together, clears stale generated motion, expands the timeline when needed, and
is one Undo/Redo action. Select **Generate** again before playback or export.

These tools do not layer or blend simultaneous whole-body motions. A Paste or
Repeat destination may touch existing motion only at a boundary. Any interior
overlap with a copied target track or committed robot-pose range rejects the
complete edit, even when no Keyframe shares the exact time. At an adjacent
boundary, an equivalent seam coalesces and a different seam is rejected. Slower
range scaling is also rejected when it would expand into a later Keyframe; move
or insert time first to create enough room.

## Interface Control Reference

### 2. Workflow Toolbar

**Preview Path** validates the transition to the orange preview without saving
it.

**Commit Keyframe** records the current pose at the active time.

**Generate** samples the saved keyframes into a robot trajectory.

**Play/Pause** controls the active generated or editable timeline. **Reset**
restores the model home pose at the active time, **Clear** clears the editable
trajectory, and **Undo/Redo** navigate recorded editing history.

### 3. Left Sidebar

#### Target

Choose a registered logical frame or use **Advanced target** to select another
body or site exposed by the active MuJoCo model.

#### Editing Mode

Use **End Effector** to edit the target frame with X, Y, Z, Roll, Pitch, and Yaw
controls or the 3D transform gizmo.

Use **Joint Angles** to edit joints directly. The controls and 3D view remain
synchronized when switching modes, and both modes update the orange preview.

### 4. 3D Viewport

The model-colored robot shows the committed state. Edits appear as the Orange
preview until **Commit Keyframe** is selected. Use the transform gizmo to move
or rotate the selected target; Preview Path and playback ghosts show sampled
motion without adding Keyframes.

Camera controls orbit, pan, and zoom the view. **Move/Rotate** select the gizmo
mode, and **Gizmo** shows or hides it.

### 5. Right Sidebar And Layout

The right sidebar contains a compact **Status** summary and the
**IK / Constraints** controls. Expand **Details** to inspect solver and
operation diagnostics.

Drag the dividers to resize the sidebars. Use the divider arrows or
**View → Left Sidebar** and **View → Right Sidebar** to collapse or restore them.
GhostGUI remembers the expanded widths and collapsed states.

### 6. Planning And Timeline Controls

The active time determines where the next keyframe is stored. The time slider
supports live scrubbing and playback. Releasing the slider selects an editable
time.

Planning controls include the keyframe interval, timeline duration, Export
interval, DSMS motion speed, playback speed, smoothing, collision substeps, and
preview/playback opacity.

**Export interval** sets the uniform time step used by **Generate** and the
resulting trajectory export. Enter a value from `0.01 s` to `10.00 s`; the
default `0.01 s` interval produces a 100 Hz trajectory.

This is separate from the **Keyframe interval**. The Keyframe interval only
controls how far the editor advances after **Commit Keyframe**; it does not set
the generated trajectory's time step.

**DSMS motion speed** changes the actual timestamps written to DSMS `time.csv`
without changing qpos samples. For example, `0.50×` doubles the DSMS reference
duration. It does not change Keyframes, visual **Playback speed**, MuJoCo CSV,
or mjlab export.


## Keyboard And Mouse

Open **Help → Keyboard Shortcuts…** for a concise reference to the essential
editing shortcuts.

### Transform Editing

- **T** switches the transform gizmo to translation.
- **R** switches the transform gizmo to rotation.
- **E** or **Esc** cancels the current transform drag.
- **Shift + drag** enables finer movement.
- **Ctrl + drag** snaps movement.

### Inline Values

- Drag a filled value control for continuous adjustment.
- Click either half to move by one logical step.
- Use arrow keys for one step and Page Up/Down for ten steps.
- Use Home/End for the minimum or maximum.
- Press Enter or F2, or double-click the value, to type directly.
- Press Enter to commit typed input or Esc to cancel it.

### History

- **Ctrl+Z** undoes the last recorded action.
- **Ctrl+Shift+Z** redoes the last undone action.

### Camera

- **Left drag** orbits unless a gizmo handle is active.
- **Right drag** pans.
- **Middle drag** or the mouse wheel zooms.

## Import And Export

Use **File → Import** for MuJoCo XML/URDF models, qpos poses, or trajectories.

Use **File → Export** for:

- **Qpos**: the committed pose as one headerless qpos row.
- **Trajectory → MuJoCo**: time plus MuJoCo qpos values in one CSV.
- **Trajectory → DSMS**: a folder containing `qpos_<dof>dof.csv` and
  `time.csv`.
- **Trajectory → mjlab**: a headerless G1 29-DoF CSV with `x, y, z`, an
  `x, y, z, w` quaternion, and named G1 joints in mjlab order.

DSMS and mjlab require uniformly sampled timestamps. Their GUI exports sample
the current generated trajectory, or the editable qpos timeline, at the selected
**Export interval**. The mjlab selection is available only for a model matching
the Unitree G1 29-DoF joint contract. It creates the mjlab input CSV but does not
launch mjlab's external NPZ converter.

After sampling, DSMS export divides elapsed timestamps by **DSMS motion speed**.
The sample count and qpos path stay unchanged, so `0.50×` also halves the DSMS
reference frequency. Downstream DSMS configuration should derive total duration
from `time.csv` rather than overriding it with a shorter fixed duration.

An uncommitted orange preview is not exported. Select **Commit Keyframe** first
when the pose should become part of the saved motion.

See [Data Formats](data_formats.md) and [Adding Models](adding_models.md) for the
file contracts and import requirements.

## Common Problems

If the orange robot changed but an export did not, the pose is still only a
preview. Commit a keyframe and export again.

If an edit stops or fails, check **Status** and **IK / Constraints** for joint
limits, singularities, or collisions.

If generation does not match the intended motion, verify the keyframe times and
active target frame, then inspect playback before exporting.

See [Troubleshooting](troubleshooting.md) for installation, rendering, model,
and workflow diagnostics.

## Known Limitations

- Linux/Ubuntu is the primary tested platform.
- The transform gizmo is world-aligned; local-frame controls are not available.
- Imported models depend on resolvable mesh files and may require mesh-folder
  selection.
- IK priority numbers are descriptive metadata; the current solver uses one
  weighted task stack rather than strict null-space priority projection.

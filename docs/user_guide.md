# GhostGUI User Guide

GhostGUI creates robot motion by editing target frames or **Joint Angles**,
previewing the result on a live MuJoCo model, committing Keyframes, and
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
pose controls, or **Joint Angles** controls. It is not saved automatically and
may remain visible when GhostGUI needs to explain why a proposed edit is unsafe.

**Committed state** is the accepted robot pose at the active time. Keyframes,
trajectory generation, and exports use committed states.

**Preview Path** displays the adaptive safety check between the committed state
and the Orange preview. It marks unsafe intervals with red ghosts and does not
save anything.

**Commit Keyframe** records the current pose at the active time and advances by
the configured Keyframe interval only when the pose and its neighboring motion
are free of blocking penetration.

Blocking penetration is never promoted into a committed Keyframe, generated
motion, playback-ready motion, or export. Intended support contact and shallow
advisory contact remain distinguishable from blocking penetration.

For the complete state model, see [Preview And Keyframe Concepts](concepts.md).

## First Motion Workflow

1. Choose a robot from the **Robot** menu.
2. Select a **Target robot frame**, such as `left_hand`.
3. Move the target with the 3D transform gizmo or the pose controls.
4. Inspect the Orange preview and the **Status** panel.
5. Optionally select **Preview Path** to inspect the automatic transition check.
6. Select **Commit Keyframe** to validate and save the pose at the active time.
7. Move to another time and commit another Keyframe.
8. Select **Generate**. GhostGUI promotes the result only after motion safety
   validation succeeds.
9. Use playback to inspect the motion.
10. Use **File → Export** to save a pose or trajectory.

The main workflow is:

```text
select target → edit → Orange preview → Commit Keyframe → Generate → Export
```

## Interface

### Menu Bar

Use the menu bar to create, open, and save projects; select the active robot;
import or export data; retime Keyframes; switch views; and open help.

### Timeline Menu

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

The dialogs snap values to the **Export interval** by default; clear the
checkbox when exact off-grid timing is intentional. Each successful retiming
operation updates logical targets and qpos together, clears stale generated
motion, expands the timeline when needed, and is one Undo/Redo action. Select
**Generate** again before playback or export.

These tools do not layer or blend simultaneous whole-body motions. Source and
destination ranges cannot overlap, and conflicting destination Keyframes are
left unchanged. Slower range scaling is also rejected when it would expand into
a later Keyframe; move or insert time first to create enough room.

### Target

Choose a registered logical frame or use **Advanced target** to select another
body or site exposed by the active MuJoCo model.

### Editing Mode

Use **End Effector** to edit the target frame with X, Y, Z, Roll, Pitch, and Yaw
controls or the 3D transform gizmo.

Use **Joint Angles** to edit joints directly. The controls and 3D view remain
synchronized when switching modes, and both modes update the Orange preview.

### Planning

The active time determines where the next keyframe is stored. The time slider
supports live scrubbing and playback. Releasing the slider selects an editable
time.

Planning controls include the Keyframe interval, timeline duration, Export
interval, DSMS motion speed, playback speed, smoothing, collision substeps, and
preview/playback opacity.

### Workflow Toolbar

**Preview Path** visualizes the adaptive transition check to the Orange preview
without saving it. Commit and Generate run their required safety checks even if
Preview Path was not selected manually.

**Commit Keyframe** records the current pose at the active time only after the
pose and affected between-Keyframe intervals pass the hard safety checks.

**Generate** samples the saved Keyframes into a candidate robot trajectory,
checks samples and the motion between them, and promotes only a safe result.

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

**Play/Pause** controls the active generated or editable timeline.

**Reset** restores the model home pose at the active time.

**Clear** clears the editable trajectory.

**Move/Rotate** select the transform-gizmo mode.

**Gizmo** shows or hides the transform gizmo.

**Undo/Redo** navigate recorded editing history.

### Motion Safety

GhostGUI separates inspection from promotion. Requested-contact highlights,
Preview Path, and quarantined-motion ghosts expose rejected motion for
diagnosis. The Orange preview clamps at its last safe IK substep, and blocking
penetration cannot enter committed or exportable motion.

Motion validation is adaptive. It checks the interpolated path between adjacent
states and refines intervals according to Joint Angle and collision-geometry
movement. This check is independent of the **Export interval**, so two
collision-free Keyframes do not make an unchecked gap safe.

This is a resolution-bounded adaptive check, not an analytic proof of
continuous collision freedom. Very small features or inaccurate collision
geometry can still require tighter validator thresholds or an external motion
planner.

Automatic ground projection is intentionally narrow. GhostGUI may raise a
single movable floating root over a flat ground plane and reports the applied
lift. During generation, the correction is rejected if it would violate a
required End Effector target or exact Keyframe anchor. It is not silently
written into existing Keyframes. Ground projection is not used for fixed-base
models, uneven terrain, other environment geometry, or body-to-body collision.

A quarantined between-state ground sweep may receive a lifted waypoint as a
review candidate. GhostGUI revalidates both replacement intervals and the full
path before **Accept Safe Motion** is available.

For body-to-body or other non-repairable collisions, edit or add a Keyframe, or
explicitly choose **Try Safe Reroute** when it is offered. Rerouting is a
bounded local Joint Angle search, never forced or silent, and can fail. It
preserves endpoint qpos values but is not a global or dynamic planner.

### Sidebars

Drag the dividers to resize the sidebars. Use the divider arrows or
**View → Left Sidebar** and **View → Right Sidebar** to collapse or restore them.
GhostGUI remembers the expanded widths and collapsed states.

The right sidebar contains a compact **Status** summary and the
**IK / Constraints** controls. Expand **Details** to inspect solver and
operation diagnostics.

## Keyboard And Mouse

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

An uncommitted Orange preview is not exported. Select **Commit Keyframe** first
when the pose should become part of the saved motion.

Imported motion receives the same state and between-state validation. Motion
with blocking penetration is quarantined for inspection: it is not treated as
the active safe generated result and cannot be exported until it is repaired or
rerouted and passes validation.

Export repeats the safety gate for the path being written. Advisory contact is
reported, while blocking penetration stops the export.

See [Data Formats](data_formats.md) and [Adding Models](adding_models.md) for the
file contracts and import requirements.

## Common Problems

If the orange robot changed but an export did not, the pose is still only an
Orange preview. Use **Commit Keyframe** and export again.

If an edit stops or fails, check **Status** and **IK / Constraints** for joint
limits, singularities, or collisions.

If generation does not match the intended motion, verify the Keyframe times and
active target frame, then inspect playback before exporting.

If generation or import reports a blocked interval, jump to the reported time
and inspect **Preview Path**. Ground-only failures may offer a reviewed repair.
For body-to-body collision, adjust or add a Keyframe or explicitly try a safe
reroute.

See [Troubleshooting](troubleshooting.md) for installation, rendering, model,
and workflow diagnostics.

## Known Limitations

- Linux/Ubuntu is the primary tested platform.
- The transform gizmo is world-aligned; local-frame controls are not available.
- Imported models depend on resolvable mesh files and may require mesh-folder
  selection.
- IK priority numbers are descriptive metadata; the current solver uses one
  weighted task stack rather than strict null-space priority projection.
- Motion safety is kinematic and depends on the model's collision geometry. It
  does not prove balance, actuator feasibility, controller tracking, or safety
  on a physical robot.

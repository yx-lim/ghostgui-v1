"""Static help content for GhostGUI's in-app guide."""

from dataclasses import dataclass
from pathlib import Path

from core.resources import resource_path

USER_GUIDE_PATH = resource_path("docs/user_guide.md")


@dataclass(frozen=True)
class HelpSection:
    title: str
    body: str


HELP_SECTIONS = [
    HelpSection(
        "First Motion",
        """
# First Motion Walkthrough

Use this workflow to create one editable robot motion.

1. Choose the robot model in the app toolbar.
2. Select a **Target robot frame**, such as `left_hand`.
3. Move the target with the 3D gizmo or the position/orientation sliders.
4. The orange robot is the temporary preview. It is not saved yet.
5. Click **Preview Path** when you want to validate the path from the committed pose to the orange preview. Collision samples remain visible as red ghosts.
6. Click **Commit Keyframe** to record the preview at the active time and advance the timeline.
7. Add another keyframe at a later time.
8. Click **Generate** or **Generate / Simulate**.
9. Export the trajectory or qpos data from the app toolbar.

The main idea is:

```text
move target -> orange preview -> Commit Keyframe -> Generate -> Export
```
""".strip(),
    ),
    HelpSection(
        "Controls Map",
        """
# Controls Map

## Menu Bar

Create/open/save projects, choose the robot model, import or export data, retime Keyframes, switch views, and open help.

Use **Timeline → Insert Time at Current Time** to open a held interval and shift later Keyframes. **Shift Entire Motion** applies one offset to the whole motion. **Move Time Range** relocates an inclusive range when its non-overlapping destination is free. **Scale Time Range** changes actual timeline speed: `2×` halves duration and `0.5×` doubles it.

Use **Copy Motion Range…** to store committed Keyframes and robot poses, including Joint Angles, in a transient, model-specific Motion Clip. At an in-between boundary, GhostGUI samples the committed pose and uses forward kinematics to materialize logical target anchors; exact boundary Keyframes are preserved. It does not copy the Orange preview or generated samples, and copying does not change history or dirty the project. **Paste Motion at Current Time** preserves relative time; a forward append requires a closed seam. **Paste Motion Reversed at Current Time** reverses time order only, producing `A → B → A` without mirroring poses or negating Joint Angles. **Repeat Motion…** adds the requested number of copies in **Forward** or alternating **Ping-pong** order. A destination may touch existing motion only at a boundary: equivalent seams coalesce, different seams and any interior overlap reject the complete edit. A successful Paste or Repeat keeps targets and committed robot poses synchronized, clears generated motion, and is one Undo/Redo action.

## Target

Choose the logical robot frame to edit, or use **Advanced target** to select
another body or site exposed by the model.

## Editing Mode

Use **End Effector** to edit the target frame with X/Y/Z/Roll/Pitch/Yaw controls or the 3D transform gizmo. Use **Joint Angles** to edit joints directly or move the gizmo through IK while watching the joint values update. Both modes update the same orange preview, and the controls stay synchronized when you switch modes.

## Sidebars

Drag either divider to resize its sidebar between 200 and 400 pixels. Use the divider arrows or the **View → Left Sidebar** and **View → Right Sidebar** actions to collapse or restore them. GhostGUI remembers both expanded widths.

## Planning

Use the single Time slider to scrub the robot live or follow time-based playback. The frame readout is derived from the trajectory, and releasing the slider commits the selected edit time once. Configure playback speed without changing trajectory timestamps. DSMS motion speed separately changes actual DSMS export timestamps while preserving qpos. Configure smoothing, collision substeps, playback opacity, and preview opacity; capture committed robot states; generate trajectories; and manage the editable timeline here.

## Workflow Toolbar

- **Preview Path** validates the path from the committed pose to the orange preview and marks collision samples with red ghosts.
- **Commit Keyframe** records the preview at the active time.
- **Generate** samples the saved keyframes into a robot trajectory.
- **Play/Pause** controls the current generated or editable timeline.
- **Reset** returns the active time to the model home pose.
- **Clear** clears the editable trajectory.
- **Move/Rotate** select the active transform gizmo in either editing mode.
- **Gizmo** shows or hides the transform gizmo.
- **Undo/Redo** navigate recorded editing history.

## Right Sidebar

- **Status** shows the latest important event or problem. Expand **Details** for the latest operation's frame, IK result, and solver diagnostics.
- **IK / Constraints** exposes IK weights, solver settings, collision checks, and preview controls.
""".strip(),
    ),
    HelpSection(
        "Core Concepts",
        """
# Core Concepts

## Target Frame

The frame you are editing, such as a hand, foot, torso, pelvis, body, or site exposed by the active model.

## Preview State

The orange robot. It is a temporary IK result from your current drag or slider edit.

## Committed State

The accepted robot pose at the active time. Generated trajectories and exports use committed timeline states, not unsaved orange previews.

## Preview Path Button

Validates the motion between the committed pose and the orange preview. It does not save the pose by itself.

## Commit Keyframe Button

Records the preview at the active time and advances by the configured keyframe interval.

## Generated Trajectory

A sampled sequence built from saved keyframes and IK. This is the data you usually export for MuJoCo validation or downstream tools.

Export interval sets the generated time step from 0.01 s to 10.00 s. This is separate from the Keyframe interval, which only advances the editing time after Commit Keyframe. DSMS motion speed divides elapsed `time.csv` timestamps after sampling; 0.50× doubles actual DSMS duration without changing qpos or other export formats.
""".strip(),
    ),
    HelpSection(
        "Keyboard / Mouse Shortcuts",
        """
# Keyboard / Mouse Shortcuts

## Editing

- **T** switches the 3D transform gizmo to translate mode.
- **R** switches the 3D transform gizmo to rotate mode.
- **E** or **Esc** cancels the current transform drag.
- **Shift + drag** gives finer gizmo movement.
- **Ctrl + drag** snaps gizmo movement.

## Inline Value Sliders

- **Drag** for continuous live adjustment.
- **Click the left/right half** to decrease/increase by one logical step.
- **Arrow keys** decrease or increase; **Page Up/Down** moves ten steps.
- **Home/End** selects the minimum or maximum.
- **Enter**, **F2**, or double-click the displayed number to type a value.
- **Enter** commits typed input; **Esc** cancels it.

## History

- **Ctrl+Z** undoes the last recorded action.
- **Ctrl+Shift+Z** redoes the last undone action.

## 3D Camera

- **Left drag** orbits the camera unless a gizmo handle is active.
- **Right drag** pans the camera.
- **Middle drag** zooms the camera.
- **Mouse wheel** zooms the camera.
""".strip(),
    ),
    HelpSection(
        "Export Format",
        """
# Export Format

Use **File > Export** to choose what to save.

- **Qpos** saves the current committed pose as one headerless qpos row.
- **Trajectory > MuJoCo** saves timed qpos rows in one CSV.
- **Trajectory > DSMS** saves qpos and time CSVs in one reference folder.
- **Trajectory > mjlab** saves a headerless Unitree G1 29-DoF input CSV. It does not launch mjlab's external NPZ converter.

DSMS and mjlab need uniform timestamps. Their GUI exports sample the current generated or editable trajectory at the selected Export interval. DSMS motion speed then scales DSMS timestamps only; visual Playback speed remains independent.

Uncommitted orange previews are intentionally not exported. Use **Commit Keyframe** first when you want the current preview to become part of the saved motion.
""".strip(),
    ),
    HelpSection(
        "Troubleshooting",
        """
# Troubleshooting

## The orange robot moves but export did not change

The orange robot is only a preview. Click **Commit Keyframe** to store it at the active time before exporting.

## Preview fails or moves only partway

An **IK reach limit** means the required handle task could not be reached; collision warnings are reported separately. Translation may relax optional orientation and lock tasks once while keeping the required position. **Commit Keyframe** blocks meaningful penetration but allows shallow advisory contact with a warning. Check the **Status** panel and the **IK / Constraints** section.

## Generate gives an unexpected path

Confirm that the saved keyframes are at the intended times, then inspect **Playback** ghosts and the timeline controls.

## A model has different frames than expected

Changing robot models changes the available bodies, sites, joints, and logical target frames. Re-check the **Target robot frame** selector after switching models.
""".strip(),
    ),
]


def load_user_guide():
    try:
        return USER_GUIDE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return (
            "# User Guide\n\n"
            f"The written guide could not be loaded from `{USER_GUIDE_PATH}`.\n\n"
            f"Error: {exc}"
        )

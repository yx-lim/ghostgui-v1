"""Static help content for GhostGUI's in-app guide."""

from dataclasses import dataclass
from pathlib import Path


USER_GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "user_guide.md"


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

1. Choose the robot model in **Setup**.
2. Select a **Target robot frame**, such as `left_hand`.
3. Move the target with the 3D gizmo or the position/orientation sliders.
4. The orange robot is the temporary preview. It is not saved yet.
5. Click **Preview** when you want to plan/check the path from the committed pose to the orange preview.
6. Click **Slice** to accept the current preview, record it at the active time, and advance the timeline.
7. Add another slice at a later time.
8. Click **Generate** or **Generate / Simulate**.
9. Export the trajectory or qpos data from **Setup**.

The main idea is:

```text
move target -> orange preview -> Slice -> committed time slice -> Generate -> Export
```
""".strip(),
    ),
    HelpSection(
        "Controls Map",
        """
# Controls Map

## Setup

Choose the robot model, import qpos or trajectory data, and export qpos rows or timed trajectories.

## Target / Pose

Choose the active target frame and edit its target position/orientation. These controls drive the same preview path as the 3D transform gizmo.

## Time Slices

Set the active time, capture committed robot states, generate trajectories, and manage the editable timeline.

## 3D Quick Toolbar

- **Preview** plans/checks the path from the committed pose to the orange preview.
- **Slice** accepts the current preview and stores it at the active time.
- **Generate** samples the saved slices into a robot trajectory.
- **Play** plays the current generated or editable timeline.
- **Reset** returns the active time to the model home pose.
- **Clear** clears the editable trajectory.
- **Playback** toggles trajectory ghost/playback visibility.

## Right Sidebar

- **Selected Object** shows the picked body/site/frame details.
- **IK / Constraints** exposes IK weights, solver settings, collision checks, and preview controls.
- **Status** shows the current state, selected frame, IK result, root pose, and detailed messages.
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

## Preview Button

Plans/checks the motion between the committed pose and the orange preview. It does not save the pose by itself.

## Slice Button

Accepts the preview if needed, records the committed pose at the active time, and advances by the configured slice step.

## Generated Trajectory

A sampled sequence built from saved slices/keyframes and IK. This is the data you usually export for MuJoCo validation or downstream tools.
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

Use **Setup -> Export** to choose what to save.

- **Qpos** saves the current committed pose as one headerless qpos row.
- **Trajectory** saves timed qpos rows from the generated trajectory, or from the editable timeline when no generated trajectory is active.

Unaccepted orange previews are intentionally not exported. Use **Slice** first when you want the current preview to become part of the saved motion.
""".strip(),
    ),
    HelpSection(
        "Troubleshooting",
        """
# Troubleshooting

## The orange robot moves but export did not change

The orange robot is only a preview. Click **Slice** to commit and store it at the active time before exporting.

## Preview fails or moves only partway

IK, joint limits, singularity checks, or collision checks may be blocking an unsafe pose. Check the **Status** panel and the **IK / Constraints** section.

## Generate gives an unexpected path

Confirm that the saved time slices are at the intended times, then inspect **Playback** ghosts and the keyframe/timeline controls.

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

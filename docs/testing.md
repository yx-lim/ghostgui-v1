# Testing

Run automated tests before submitting changes and perform focused GUI checks
for behavior that depends on rendering or interaction.

## Automated Suite

From an installed checkout:

```bash
python3 -m unittest discover -s tests -v
```

The suite covers model adapters, IK and collision behavior, timeline semantics,
CSV playback, transform-gizmo math, GUI controls, themes, status handling, and
trajectory smoothing.

Run documentation validation separately:

```bash
python3 scripts/check_docs.py
```

## Focused Tests

Use a test module while iterating:

```bash
python3 -m unittest tests.test_robot_viewer_timeline -v
python3 -m unittest tests.test_advanced_ik -v
python3 -m unittest tests.test_model_resources -v
```

Choose the test that owns the changed contract rather than relying only on a
manual launch.

## Manual GUI Checks

### Shared Workflow

- Start GhostGUI and confirm the selected model renders.
- Edit an end effector and confirm only the orange preview moves.
- Use **Preview Path** and confirm no keyframe is created.
- Use **Commit Keyframe** and confirm the timeline advances by the configured
  keyframe interval.
- Edit another time and confirm the earlier keyframe remains unchanged.
- Generate, play, pause, scrub, and export the trajectory.
- Confirm preview opacity does not make the application window transparent.
- Confirm reset affects the active time only.

### Model Checks

- **G1:** confirm humanoid hand, foot, torso, and pelvis targets and detailed
  geometry.
- **Go2:** confirm twelve scalar leg joints, four foot targets, the ground, and
  COLLADA-derived visual geometry.
- **H2:** confirm the humanoid targets and bundled STL visuals.
- **Z1:** confirm base, wrist, and tool targets and six arm joints.
- Switch away from a model and back; confirm its editor session is preserved.

### Import And Export

- Load a qpos file with the correct active-model width.
- Reject a qpos or trajectory with the wrong width or non-finite values.
- Export a committed pose and confirm an uncommitted orange preview is absent.
- Import a model with relative meshes.
- Exercise the mesh-folder retry path for an unresolved model asset.

### Path Independence

Launch the compatibility script from a different working directory:

```bash
cd /tmp
python3 /path/to/ghostgui/scripts/run_gui.py --model go2
```

Registry and model paths should remain anchored to the checkout.

## Documentation Checks

Documentation validation should reject:

- broken relative links;
- missing linked files or anchors;
- more than one level-one heading per public page;
- legacy user-facing workflow terminology;
- undocumented public pages;
- trailing whitespace.

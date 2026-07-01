![ghostgui](ghostgui.png)
# GhostGUI

A lightweight graphical interface for editing reference-frame and robot-state
trajectories against the G1 MuJoCo model.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install PySide6 PyOpenGL numpy mujoco
python3 run_gui.py
```

The live **3D View** tab loads `models/g1_29dof.xml`, exposes all controllable
joint sliders, supports free TCP translation from the central sphere, X/Y/Z
arrow translation, and X/Y/Z ring rotation with MuJoCo Jacobian IK, and can
generate/play a simple qpos trajectory with
transparent ghosts. Drag IK is solved in a temporary state and accepted only
when MuJoCo reports no self/environment collision; the collision-substep control
sets how finely motion is clamped at a contact boundary.

The **Reset 3D Pose** button is a one-shot action: it pauses playback, cancels an
active gizmo drag, and restores model home qpos at the currently selected GUI
time only. It does not replace or repeatedly modify the playback list. Changing
`Time [s]` loads or creates an independent 3D
qpos keyframe, so edits at `0.2s` do not modify `0s`. Selecting `pelvis` drives
the model's floating root joint and is still checked for collisions.
The existing 2D editors and separate **3D MuJoCo** CSV player remain available.

Right-drag rotates the live 3D camera and the mouse wheel zooms. Gizmo axes are
world-aligned; a local-frame mode is a future extension.

The **Frames**, **Status**, and in-tab **3D** control sidebars have compact
chevron handles and remember their expanded widths. `Ctrl+[` toggles Frames and
`Ctrl+]` toggles Status. **Use model colors** is enabled by default: the live
renderer resolves MuJoCo material assignments, giving this model its original
black/silver appearance. Its robot materials do not reference texture assets;
textured visual materials in other models currently fall back to material RGBA.

## Project Structure

```
ghostgui/
├── README.md
├── backend/
├── gui/
├── models/
├── scripts/
├── tests/
└── run_gui.py
```

Run the model/state sanity checks with:

```bash
python3 -m unittest discover -s tests -v
```

# Troubleshooting

Start with the section that matches the failure. Include the complete status or
terminal message when reporting a problem.

## Installation And Launch

### The `ghostgui` command is not found

Activate the environment created by the installer.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `.venv` does not exist, rerun the platform installer from the repository
root.

### Installation succeeds but dependencies are broken

From the activated environment, run:

```bash
python -m pip check
```

If it reports missing or conflicting packages, rerun the platform installer.
This is especially useful when the installer reused an existing `.venv`.

### Qt fails to initialize on Linux

Confirm that the packages in `scripts/install_linux.sh` were installed.
Messages about Qt platform plugins commonly indicate missing XCB libraries such
as `libxcb-cursor0`, `libxcb-xinerama0`, `libxcb-xinput0`, or
`libxkbcommon-x11-0`.

### MuJoCo cannot create an OpenGL context

GhostGUI requests desktop OpenGL 2.1 compatibility rendering with a 24-bit
depth buffer. Use the rendering error in **Status** to compare that request
with the context supplied by Qt. Confirm that the machine has working GPU
drivers or a software OpenGL stack. Remote desktops, containers, WSL, and
headless sessions may need separate display and OpenGL configuration.

### The passive viewer fails on macOS

MuJoCo passive-viewer scripts must run through `mjpython` on macOS. Recreate
`.venv` with a Python matching the machine's native architecture if `python`,
the environment, and `mjpython` do not agree.

## Model Loading

### A bundled model cannot be found

Install GhostGUI in editable mode from the cloned repository. The registered
model sources and bundled meshes are loaded from the checkout.

### A mesh cannot be resolved

For imported models, select the directory containing the referenced mesh files
when GhostGUI offers to retry. Preserve filename case on case-sensitive file
systems.

If the model uses ROS `package://` references, select either the package root or
the directory containing the referenced `meshes`, `dae`, or `assets` files.

### A cached model appears stale

The runtime cache is content-addressed, so source, mesh, MuJoCo, and cache-format
changes normally create a new entry. To force regeneration, close GhostGUI and
remove only the affected prepared-model entry under:

```text
~/.cache/ghostgui/models/
```

If `GHOSTGUI_CACHE_DIR` is set, inspect its `models` directory instead. Do not
remove imported model sources or project data; they are not disposable caches.

## Projects

### A project cannot open after an interrupted save

Do not delete `.ghostgui-transactions` or edit individual project files. Close
all GhostGUI processes using the project, preserve a copy of the complete
`.ghostgui` folder, and follow [Interrupted Save Recovery](operations.md#interrupted-save-recovery).

## Editing And IK

### The orange preview moves only partway

The target may be unreachable, near a singularity, outside a joint limit, or
constrained by optional IK tasks. Try a smaller edit, use the all-joints preset,
and disable optional posture, foot-lock, and regularization tasks.

### A keyframe cannot be committed

GhostGUI rejects non-finite qpos values and colliding final poses. Inspect the
**Status** details for the collision pair or numerical error, then adjust the
orange preview.

### Preview Path fails

The final pose may be valid while an intermediate sample violates a joint limit
or collides. Add a closer intermediate keyframe or change the motion around the
obstacle.

### The wrong body moves

Re-check **Target robot frame** after switching models. Logical frames are
model-specific. Use **Advanced target** to inspect the underlying MuJoCo body or
site.

## Timeline And Export

### The orange robot changed but the export did not

The orange robot is temporary. Select **Commit Keyframe** before exporting.

### Generation does not match the intended motion

Check the active frame and keyframe times, then inspect playback. Delete or
update an unintended keyframe and regenerate.

### A CSV is rejected

Confirm the file matches the active model. A qpos file requires exactly
`model.nq` headerless values. A headerless trajectory requires time followed by
that many qpos values on every row, with nondecreasing times.

See [Data Formats](data_formats.md) for the full contracts.

### A DSMS or mjlab import is rejected

For DSMS, select the folder containing `time.csv` and exactly one matching
`qpos_<dof>dof.csv`. mjlab import supports only G1 29-DoF and requires the
source sample interval. See [Import Behavior](data_formats.md#import-behavior)
for the complete requirements.

## Reporting A Problem

Include:

- operating system and architecture;
- Python and MuJoCo versions;
- selected robot key;
- the command used to launch GhostGUI;
- the complete terminal and **Status** messages;
- whether the issue occurs with a bundled model;
- minimal model or CSV input when the problem is data-dependent.

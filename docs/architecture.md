# Architecture

GhostGUI is a Python/PySide6 desktop application with MuJoCo-backed model,
kinematics, collision, trajectory, and playback services.

## Data Flow

```text
PySide6 controls and 3D gizmo
              ↓
       separate preview state
              ↓
   IK, limits, and collision checks
              ↓
        committed qpos state
              ↓
 logical target and qpos keyframes
              ↓
      trajectory generation
              ↓
 live playback / MuJoCo simulation / CSV
```

## Source Responsibilities

| Path | Responsibility |
| --- | --- |
| `application/` | Use cases, background jobs, projects, import/export, generation |
| `core/models/` | Registry, model preparation, MuJoCo metadata, mutable state |
| `core/ik/` | IK tasks and collision-aware drag solving |
| `core/trajectory/` | Keyframes, sampling, interpolation, smoothing |
| `gui/` | Main window, sidebars, widgets, help, tutorial, 3D interaction |
| `backend/` | Optional native robot simulation implementation |
| `models/` | Bundled and imported model sources and assets |
| `scripts/` | Installation, launch, and standalone viewer entry points |
| `tests/` | Unit, contract, GUI, model, and trajectory tests |

## Application Assembly

`application.launcher:main` parses the model key, creates the Qt application,
and opens `RobotGuiMainWindow`.

The main window maintains one session per loaded model. A session groups the
MuJoCo adapter, backend, logical-frame reference provider, 3D viewer, editable
trajectory, and active selection. Model loading runs outside the GUI thread.

## Model Layer

`MuJoCoRobotAdapter` is the model-agnostic facade. It exposes compiled MuJoCo
joints, limits, free roots, logical frames, home qpos, kinematic relationships,
and state factories.

URDF preparation is content-addressed and produces a cached runtime XML.
MuJoCo XML sources load directly. See [Models](models.md) for the asset and cache
contract.

## Editor State

`RobotViewer3D` owns distinct committed, preview, and playback states:

- committed state is authoritative at the active time;
- preview state isolates temporary edits;
- playback state samples motion without changing the edit time.

`RobotStateTimeline` stores qpos keyframes. The application trajectory stores
logical target-frame keyframes used by the backend. A commit updates both
representations for the same time.

## Trajectory Generation

The trajectory layer interpolates target positions and orientations. Quaternion
orientation interpolation uses SLERP. The generation service samples tracks at
a uniform interval and sends them to the active backend.

The Python MuJoCo backend solves target frames with Jacobian IK and returns named
robot configurations. The application can also play raw qpos trajectories
directly.

## Files And Projects

Project workspaces live below `projects/` and persist application state in
`ghostgui_project.json`. CSV import/export is isolated in
`application/csv_io.py`; project and cache paths are centralized in
`application/paths.py`.

The UI should not write files directly when an application service already owns
the relevant workflow.

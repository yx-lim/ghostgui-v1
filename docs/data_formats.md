# Data Formats

GhostGUI uses model-dependent MuJoCo `qpos` arrays for pose and trajectory
interchange. The active model determines the required array width and ordering.

## Qpos vs. Trajectory

A `qpos` is one complete MuJoCo robot configuration. It contains Joint Angles
and, for a floating-base robot, base translation and orientation. A qpos pose is
one robot state without time; a trajectory is a sequence of qpos states that
describes motion.

| Format | Contents | Timing | Compatibility | Typical use |
| --- | --- | --- | --- | --- |
| Qpos Pose | One qpos row | None | Active model | Save or load one pose |
| MuJoCo trajectory | Time plus qpos per row | In each row | Active model | GhostGUI and MuJoCo interchange |
| DSMS | Qpos rows and `time.csv` | Separate file | Model-dependent | DSMS reference motion |
| mjlab | Floating base and 29 named joints | Not included | G1 29-DoF only | mjlab input |
| Named backend CSV | Named semantic columns | In each row | Active backend | Simulation and inspection tools |

## Qpos Pose CSV

A qpos pose file contains one headerless numeric row:

```text
qpos_0,qpos_1,...,qpos_n
```

The row must:

- contain exactly `model.nq` values for the active model;
- contain only finite numbers;
- use the MuJoCo qpos ordering compiled for that model.

For a floating free joint, MuJoCo stores three translation values followed by a
quaternion in `w, x, y, z` order. Remaining entries follow the compiled model's
joint qpos addresses. Do not assume that two different robots share a qpos
width or column order.

**File → Export → Qpos** writes the current committed pose. An orange preview is
not included.

## MuJoCo Trajectory CSV

The headerless MuJoCo trajectory format contains one row per sample:

```text
time,qpos_0,qpos_1,...,qpos_n
```

Requirements:

- time is measured in seconds;
- time cannot be negative;
- every row contains `1 + model.nq` finite values;
- times are nondecreasing;
- every qpos uses the active model's compiled ordering.

**File → Export → Trajectory → MuJoCo** writes this format from the generated
trajectory, or from the editable qpos timeline when no generated trajectory is
active.

## DSMS Reference Folder

**File → Export → Trajectory → DSMS** writes two headerless files into the
selected folder:

```text
qpos_<dof>dof.csv  # one MuJoCo qpos row per sample
time.csv           # one timestamp per sample
```

The joint DoF in the filename is derived from the active model. A floating-base
quaternion is normalized when the model has one free joint. The GUI samples the
current generated trajectory, or the editable qpos timeline, uniformly at the
selected **Export interval**, then divides elapsed timestamps by **DSMS motion
speed** before writing the DSMS files. qpos rows and sample count do not change.
The first timestamp is preserved; at `0.50×`, duration and sample interval
double while reference frequency halves.

The terminal converter remains available for existing GhostGUI CSV files:

```bash
python3 scripts/convert_ghostgui_to_dsms.py input.csv output_folder --speed 0.5
```

The converter defaults to `--speed 1.0` and reports source/output duration,
sample interval, and reference frequency. DSMS should derive its run duration
from `time.csv`; a fixed downstream duration must be updated to match.

## mjlab Input CSV

**File → Export → Trajectory → mjlab** writes the G1 29-DoF input layout:

```text
base_x,base_y,base_z,base_qx,base_qy,base_qz,base_qw,joint_0,...,joint_28
```

The file is numeric and headerless. Joint values are selected by the required
G1 joint names rather than by assuming the active model's raw column order.
The GUI samples the trajectory uniformly at the selected **Export interval** and
reports the resulting input frequency. It does not run mjlab's external 50 Hz
NPZ converter.

The standalone converter, including its optional external-converter flags,
remains available:

```bash
python3 scripts/ghostgui_to_mjlab.py input.csv output.csv
```

## Named Backend CSV

Trajectory generation also produces a named solver CSV for the MuJoCo
simulation view. Its header begins with:

```text
time,base_x,base_y,base_z,base_qw,base_qx,base_qy,base_qz,...
```

The remaining columns are the active backend's joint names. This format is
useful when consumers need semantic columns rather than raw qpos addresses.
Floating-base orientation uses `w, x, y, z` quaternion order.

## Import Behavior

Qpos import reads the first non-empty row and rejects a header or incorrect
width.

**File → Import → Trajectory** accepts these formats:

- **MuJoCo** reads one headerless time-plus-qpos CSV and requires nondecreasing
  times.
- **DSMS** asks for the reference folder, not its files separately. The folder
  must contain `time.csv` and exactly one `qpos_<dof>dof.csv` matching the active
  model. The files must have equal sample counts, and each qpos row must have
  the active model's `nq` width. Existing DSMS timestamps, including motion-speed
  scaling, are preserved.
- **mjlab** reads the headerless Unitree G1 29-DoF layout and converts its `x, y,
  z, w` quaternion and named joint order back to MuJoCo qpos ordering. Because
  mjlab CSV has no timestamps, the import asks for its source sample interval;
  `0.01 s` represents 100 Hz. mjlab import is unavailable for incompatible
  active models.

The editable Keyframe interval prompt is separate from the mjlab source sample
interval: it controls how densely imported samples become editable Keyframes.
The standalone MuJoCo viewer also accepts named backend CSV output.

## Project Folder

A GhostGUI project is a folder ending in `.ghostgui`. Its
`ghostgui_project.json` metadata currently uses schema version 2 and points to
the target trajectory, qpos timeline, workspace, snapshot, and session log by
relative path.

All metadata paths must stay inside the project folder. Absolute paths,
parent-directory traversal, duplicate destinations, and symlink escapes are
rejected when the project opens.

Schema version 1 projects are migrated in memory when opened and written as
version 2 on the next save. Projects from a newer unsupported schema are
rejected without modification.

A manual save stages the target-frame JSON, qpos NPZ, workspace JSON, and
project metadata as one recoverable transaction. If replacement fails, the
previous coherent set is restored. If the process stops during replacement,
the transaction journal is recovered before project metadata is read.
Autosaves use the same transaction mechanism and publish their manifest last.

## Examples

Repository examples are grouped by purpose:

```text
csv/qpos/        single-pose qpos files
csv/trajectory/  trajectory files
```

Treat each example as model-specific. Confirm its intended robot before loading
it.

## Precision

Qpos exports use scientific notation with 18 digits after the decimal point.
Trajectory time values are written with six digits after the decimal point.
Consumers should parse numeric values rather than relying on their textual
formatting.

# Motion Math

This page summarizes how GhostGUI turns committed Keyframes into robot motion.
For controls and tuning advice, see [Advanced IK](advanced_ik.md).

## Motion Pipeline

```text
committed Keyframes
        ↓
interpolated Cartesian targets and qpos posture reference
        ↓
MuJoCo forward kinematics and Jacobians
        ↓
damped-least-squares IK
        ↓
joint limits and collision validation
        ↓
sampled qpos trajectory
```

Positions are meters, angles are radians, and quaternions use MuJoCo
`w, x, y, z` order.

## Keyframe Sampling

For time `t` between Keyframes at `t0` and `t1`, GhostGUI uses the normalized
segment coordinate

```text
a = clamp((t - t0) / (t1 - t0), 0, 1).
```

Cartesian position is linear by default:

```text
p(t) = (1 - a) p0 + a p1.
```

Corner smoothing blends that line toward a cubic Hermite curve. With segment
duration `h`, endpoint tangents `m0` and `m1`, and the standard Hermite basis,

```text
pH(a) = (2a³ - 3a² + 1) p0
      + (a³ - 2a² + a) h m0
      + (-2a³ + 3a²) p1
      + (a³ - a²) h m1.
```

The smoothing control `s` produces `p = pLinear + s(pH - pLinear)`. End
tangents use one-sided slopes; interior tangents use the slope between the
neighboring Keyframes.

Orientation uses spherical linear interpolation (SLERP). After choosing the
shortest quaternion arc, for angle `theta = acos(q0 · q1)`:

```text
q(a) = sin((1 - a) theta) / sin(theta) q0
     + sin(a theta) / sin(theta) q1.
```

Nearly identical orientations use normalized linear interpolation to avoid
numerical instability. Committed qpos states are interpolated with MuJoCo's
position-manifold operations, which correctly handle free-joint quaternions.

Implementation: `core/trajectory/model.py`, `core/math3d.py`, and
`RobotStateTimeline` in `core/models/model.py`.

## Task Errors And Jacobians

At each sample, MuJoCo forward kinematics gives the current End Effector pose.
The position error is

```text
ep = ptarget - pcurrent.
```

MuJoCo supplies the translational Jacobian `Jp` and rotational Jacobian `Jr`,
which locally relate a joint update to End Effector motion. Orientation error
uses the cross products of corresponding current and target rotation-matrix
axes:

```text
er = 1/2 sum_i (Rcurrent[:, i] × Rtarget[:, i]).
```

Task errors and Jacobian rows are scaled by `sqrt(weight)`. Posture tasks use

```text
eposture = qreference - qcurrent
```

over controllable scalar joints and normalize their weight by the joint count.

Implementation: `core/ik/tasks.py`.

## Damped-Least-Squares IK

The interactive solver stacks all enabled task errors into `e` and their
Jacobians into `J`. Let `D` be the diagonal matrix of per-joint influence
values and `lambda` the damping value. Each iteration computes

```text
delta_q = D J^T (J D J^T + lambda² I)^-1 e.
```

The linear system is solved directly rather than forming an explicit inverse.
Damping keeps the update bounded near singular configurations. Each component
is clipped to the configured maximum joint step, multiplied by the step size,
applied to qpos, and clamped to the joint limits. Forward kinematics is then
recomputed and the process repeats until required tasks meet their tolerances
or the iteration limit is reached.

Singular values of the influence-adjusted Jacobian are also used for the
singularity diagnostics shown in **Status**.

Implementation: `RobotState3D.solve_weighted_tasks` in `core/models/model.py`.

## Primary And Secondary Motion

Generated trajectories with complete committed qpos anchors use hierarchical
IK. Cartesian End Effector tasks are primary. The interpolated qpos posture is
secondary, so it can preserve choices such as elbow configuration without
intentionally reducing primary Cartesian accuracy.

For the primary influence-adjusted Jacobian `A`, GhostGUI obtains a damped
pseudoinverse

```text
A+ = A^T (A A^T + lambda² I)^-1
```

and an SVD null-space projector `N`. The primary update is `A+ eprimary`; the
secondary posture update is solved through its projected Jacobian and then
restricted by `N`. The combined update is step-limited and integrated using
the same iteration rules as interactive IK.

Implementation: `RobotState3D.solve_hierarchical_tasks` in
`core/models/model.py` and `application/backend_interface.py`.

## Validation And Output

IK convergence alone does not make a state acceptable. GhostGUI also checks
finite qpos values, joint limits, and MuJoCo contacts. Collision policy
distinguishes advisory support contact from blocking penetration. Preview Path
shows colliding intermediate samples; Commit Keyframe and export reject
blocking states according to their documented policies.

Trajectory generation samples at the selected **Export interval**. Exact qpos
anchors must lie on that uniform time grid. The result is a time-ordered qpos
trajectory used for playback and export; playback speed does not modify its
timestamps.

Implementation: `core/ik/collision.py`, `application/trajectory_generation.py`,
and `application/trajectory_export_formats.py`.


# Advanced IK

GhostGUI uses MuJoCo position and rotation Jacobians with damped-least-squares
inverse kinematics. The **IK / Constraints** sidebar controls the weighted
solver used for the orange preview.

Interactive preview and generated-trajectory solving share the same task and
coordinate contracts. Generated motion with complete committed keyframes uses
a hierarchical solver: Cartesian targets are primary and posture is optimized
only in their joint-space null space. Positions are meters, angles are radians,
and quaternions use MuJoCo `w, x, y, z` order throughout.

## Solver Controls

| Control | Purpose | Default |
| --- | --- | --- |
| Damping | Stabilizes the least-squares solve | `0.04` |
| Max iterations | Bounds work per solve | `80` |
| Step size | Scales each update | `0.7` |
| Max joint step | Limits one update | `0.08` |
| Position tolerance | Required TCP position error | `0.005` |
| Orientation tolerance | Required TCP angular error | `0.03` |

Increasing damping can improve stability near singular configurations but may
slow convergence. Large step sizes can converge faster but are more likely to
overshoot or hit limits.

## Task Weights

Available tasks include:

- selected TCP position;
- selected TCP orientation;
- posture preservation relative to the committed pose;
- planted-foot position locks;
- root/base orientation preservation;
- joint regularization toward the model home pose.

TCP position is enabled by default. Secondary posture and regularization tasks
are opt-in so an ordinary target drag can use the model's available kinematic
range.

A weight of zero disables a task's influence. In the interactive solver, larger
values give the task more influence within the shared weighted solve. Generated
posture references use a separate secondary hierarchy and cannot intentionally
trade primary Cartesian accuracy for posture accuracy.

## Joint Influence

Each controllable joint has an influence value:

- `0` excludes the joint from limb IK;
- `1` gives normal influence;
- values above `1` prefer movement through that joint.

Model-aware presets include all joints, selected limb, planted feet, and
body-region choices appropriate to humanoid or quadruped models. The selected
limb is derived from the compiled MuJoCo root-to-target body chain. Joints in a
prefix shared with another End Effector, such as a humanoid waist shared by
both arms, are excluded. This does not depend on names such as `shoulder`,
`elbow`, `wrist`, or `waist`.

After **Selected limb only** is applied, it remains live: changing the target
robot frame from the sidebar, advanced-target selector, or 3D body selection
recomputes the enabled branch immediately. Editing an individual joint weight
switches the preset to **Custom**, so later target changes preserve the manual
weights.

Floating roots are handled separately from ordinary limb-joint weighting.

## Priority Metadata

Tasks record descriptive priority values:

1. locks and root constraints;
2. selected TCP position and orientation;
3. posture and regularization.

The interactive orange-preview solver sorts these tasks but combines them into
one weighted stack, so optional interactive tasks may still trade off according
to their weights. Generated motion with qpos anchors instead projects the
posture task into the null space of the Cartesian task stack.

Interactive requirements follow the active gizmo handle. Translation requires
the selected TCP position while keeping enabled orientation as a best-effort
task. Rotation requires both the requested orientation and the held TCP
position. If optional orientation, posture, foot-lock, or root constraints stop
an otherwise reachable edit, the drag retries once using only required tasks
and reports that the optional constraints were relaxed. Go2 TCP orientation
remains enabled by default.

## Collision Behavior

During a drag, GhostGUI solves incremental substeps and displays collision
warnings on the orange preview. Collision substeps control how finely the drag
motion is sampled.

Later checks distinguish advisory contact from meaningful penetration:

- **Preview Path** still rejects non-finite states and raw joint-limit
  violations. It renders colliding samples as red ghosts instead of hiding the
  path.
- **Commit Keyframe** permits advisory shallow contact with a warning, but
  rejects blocking penetration and non-finite qpos values.
- trajectory generation reports its first collision, and trajectory export
  rejects blocking penetration while permitting advisory contact with a
  warning.

Model-aware contact policy distinguishes shallow intended foot support from
penetrating ground collision. Support contacts within the numerical tolerance
are allowed. Other contacts become blocking only after the registered model's
penetration threshold. Maintained models may also declare narrowly audited
body-frame pairs with an explicit maximum allowed penetration.

## Backend Selection

The normal trajectory backend is the MuJoCo hierarchical pose/posture solver.
An approximate Unitree G1 analytic backend exists for degraded environments,
but it has no exact-qpos or whole-body-IK guarantee. It is never selected for a
generic or non-G1 model. G1 use requires the `allow_approximate` fallback
policy, emits a runtime warning, and is identified as approximate in the status
output. Unexpected solver exceptions are never converted into fallback results.

Inspect the **Status** details for the selected frame, error, singularity
metrics, and collision names. `IK reach limit` identifies solver reach or
required-constraint failure separately from collision warnings.

## Tuning Order

When an edit does not converge:

1. Confirm the selected target and model.
2. Check whether the target is physically reachable.
3. Use the all-joints preset.
4. Disable optional foot, posture, and regularization tasks.
5. Reduce the target displacement.
6. Increase damping near a singularity.
7. Increase iterations only after checking the constraints above.

See [Troubleshooting](troubleshooting.md#editing-and-ik) for common failure
symptoms.

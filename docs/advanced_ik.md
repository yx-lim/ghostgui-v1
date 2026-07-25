# Advanced IK

GhostGUI uses MuJoCo position and rotation Jacobians with weighted,
damped-least-squares inverse kinematics. The **IK / Constraints** sidebar
controls the solver used for the orange preview.

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

A weight of zero disables a task's influence. Larger values give the task more
influence within the shared weighted solve, but do not create a hard hierarchy.

## Joint Influence

Each controllable joint has an influence value:

- `0` excludes the joint from limb IK;
- `1` gives normal influence;
- values above `1` prefer movement through that joint.

Model-aware presets include all joints, selected limb, planted feet, and
body-region choices appropriate to humanoid or quadruped models.

Floating roots are handled separately from ordinary limb-joint weighting.

## Priority Metadata

Tasks record descriptive priority values:

1. locks and root constraints;
2. selected TCP position and orientation;
3. posture and regularization.

The current solver sorts the tasks but combines them into one weighted stack.
It does not implement strict null-space projection, so lower-priority tasks can
still trade off with higher-priority tasks according to their weights.

## Collision Behavior

During a drag, GhostGUI solves incremental substeps and displays collision
warnings on the orange preview. Collision substeps control how finely the drag
motion is sampled.

Two later checks protect saved motion:

- **Preview Path** rejects intermediate path samples that violate raw joint
  limits or collide.
- **Commit Keyframe** rejects a final orange preview that contains a collision
  or non-finite qpos values.

Inspect the **Status** details for the selected frame, error, singularity
metrics, and collision names when a solve fails.

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

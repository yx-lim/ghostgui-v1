# Advanced IK

GhostGUI uses MuJoCo position and rotation Jacobians with damped-least-squares
inverse kinematics. The **IK / Constraints** sidebar controls the weighted
solver used for the Orange preview.

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

GhostGUI treats collision inspection and motion promotion as separate steps.
The requested-contact highlight, Preview Path, and quarantined-motion ghosts
keep a rejected candidate diagnosable. The Orange preview itself clamps at its
last safe IK substep. Blocking penetration cannot be promoted into a committed
Keyframe, generated motion, playback-ready motion, or export.

Collision results have three practical levels:

- intended support contact is valid contact at the ground boundary;
- shallow advisory contact or a configured near-contact margin is reported but
  may be committed and exported;
- blocking penetration is a hard failure.

The model-aware policy identifies intended support bodies and narrowly audited
body pairs. A small numerical tolerance prevents contact noise from making a
resting foot flicker between valid and invalid states. This tolerance is not
permission for a foot or another body to pass through the ground.

### Between-Keyframe Validation

Checking only the Keyframes or generated rows is insufficient: interpolation
can collide even when both endpoints are safe. GhostGUI therefore validates the
manifold-interpolated qpos path between adjacent states. The validator
adaptively subdivides intervals according to Joint Angle and collision-geometry
movement, then reports the earliest unsafe time, interval, body pair, and
penetration depth.

This safety resolution is independent of the **Export interval** and the
**Keyframe interval**. Export interval controls generated output timestamps;
Keyframe interval controls how far the editor advances after **Commit
Keyframe**. Neither is a collision-sampling guarantee.

The current method is resolution-bounded adaptive discrete validation, not an
analytic continuous-collision proof. By default it subdivides until each
accepted subinterval moves no generalized coordinate more than `0.08` and no
collision geometry more than `0.02 m`, with a recursion-depth limit of 12.
Smaller thresholds increase coverage and cost; unusually small or inaccurate
collision geometry can still defeat any finite sampling method.

**Preview Path** visualizes the same path contract with red collision ghosts.
It remains a read-only diagnostic. **Commit Keyframe** automatically checks the
candidate pose and each affected neighboring interval, even when Preview Path
was not opened. **Generate** validates the solved trajectory before it becomes
the active generated result, and export repeats the gate for the path being
written.

### Repair And Rerouting

Automatic projection is limited to ground-only penetration that can be
corrected by raising one movable floating root over a flat ground plane. Live
edits and generated samples apply this hard projection immediately and report
the lift. Generated projection is rejected if it would violate a required End
Effector target or exact Keyframe anchor. Committed Keyframes are not rewritten
behind the user's back.

For an unsafe imported or between-state path, **Try Safe Reroute** can propose a
review candidate. A flat-ground sweep may receive a lifted waypoint. An
interior self-collision may receive a bounded scalar-joint detour. Every
replacement interval and the complete candidate are adaptively revalidated
before **Accept Safe Motion** becomes available.

Ground repair does not apply to a fixed-base model, uneven terrain, other
environment geometry, or body-to-body collision. A whole-robot lift cannot
resolve self-collision and may invalidate world-space targets.

Self-collision and other non-repairable failures remain blocked. The local
rerouter preserves endpoint qpos values and bounded joint limits; it is not a
global or dynamically optimal planner. It is never forced, its result is
previewed before acceptance, and it can report that no safe route was found.

### Imported Motion

Imported motion uses the same state and adaptive interval validation. A file
with blocking penetration is quarantined as an inspection candidate rather than
replacing the active safe generated result. It cannot be promoted or exported
until an edit, accepted ground repair, or explicit reroute produces a path that
passes validation.

These checks validate the modeled kinematic path. They do not prove dynamic
balance, actuator or torque feasibility, controller tracking, collision-model
accuracy, or safety on physical hardware.

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

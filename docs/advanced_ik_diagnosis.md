# Advanced IK controls diagnosis

Live drag IK is solved in `RobotState3D.solve_ik()` using MuJoCo body/site
Jacobians (`mj_jacBody` and `mj_jacSite`) and damped least squares. It supports
position plus optional orientation, clamps every scalar hinge/slide joint to
its MuJoCo limits, and is wrapped by `CollisionAwareIKSolver`, which solves in
a temporary candidate state and only copies collision-free results into the
orange preview. The committed timeline state is not modified until Accept.

Before this change, all controllable scalar joints were columns in the same
unweighted Jacobian solve. A joint could only be excluded by changing solver
code; there was no session-owned influence map or task abstraction. The joint
angle sliders edit preview qpos, but do not control how IK distributes motion.
Root/free-joint dragging is handled directly by MuJoCo free qpos; limb IK uses
the actuated hinge/slide set, so the floating root remains hard-locked unless it
is the selected target.

Joint influence belongs in the DLS inverse, not in the task error. The weighted
solve uses `dq = W J^T (J W J^T + lambda^2 I)^-1 e`, with a zero diagonal entry
locking that joint. Task weights separately scale position, orientation,
posture, planted-foot, upright-root, and regularization objectives. Tasks carry
priority metadata now, but this version intentionally solves one weighted stack;
strict null-space projection is a documented follow-up rather than being
misrepresented as hierarchy.

Joint-space posture and regularization tasks are normalized by their number of
controlled joints. Without that normalization, their many stacked Jacobian rows
overpower the three/six TCP rows and make the default stabilizers clamp ordinary
limb drags after only a few centimetres, especially across models with different
joint counts.

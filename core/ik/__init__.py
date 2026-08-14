"""IK and collision logic."""

from .collision import (
    Collision,
    CollisionAwareIKSolver,
    CollisionChecker,
    CollisionPolicy,
    DragSolveResult,
    TrajectoryCollisionReport,
    adaptive_trajectory_collision_reports,
    first_trajectory_collision,
    trajectory_collision_reports,
    format_collision_diagnostics,
    format_collision_pairs,
)
from .tasks import (
    BodyPoseTask,
    FootLockTask,
    IKTask,
    JointRegularizationTask,
    PostureTask,
    RootPoseTask,
    TCPOrientationTask,
    TCPPositionTask,
    TaskLinearization,
)
from .solver import (
    IKSolverSettings,
    PoseIKResult,
    pose_target_errors,
    solve_pose_targets,
)
from .motion_safety import (
    DEFAULT_GROUND_PENETRATION_TOLERANCE,
    DEFAULT_MAX_AUTOMATIC_GROUND_LIFT,
    GroundProjectionResult,
    project_qpos_above_flat_ground,
)

__all__ = [
    "BodyPoseTask",
    "Collision",
    "CollisionAwareIKSolver",
    "CollisionChecker",
    "CollisionPolicy",
    "DragSolveResult",
    "TrajectoryCollisionReport",
    "adaptive_trajectory_collision_reports",
    "first_trajectory_collision",
    "trajectory_collision_reports",
    "format_collision_diagnostics",
    "format_collision_pairs",
    "FootLockTask",
    "GroundProjectionResult",
    "IKTask",
    "IKSolverSettings",
    "JointRegularizationTask",
    "PostureTask",
    "PoseIKResult",
    "pose_target_errors",
    "RootPoseTask",
    "TCPOrientationTask",
    "TCPPositionTask",
    "TaskLinearization",
    "solve_pose_targets",
    "DEFAULT_GROUND_PENETRATION_TOLERANCE",
    "DEFAULT_MAX_AUTOMATIC_GROUND_LIFT",
    "project_qpos_above_flat_ground",
]

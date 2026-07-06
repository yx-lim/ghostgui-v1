# Option C implementation plan (historical)

GhostGUI already has a compatible 2D editor, a lightweight OpenGL 3D target
view, MuJoCo position IK, and a separate MuJoCo trajectory player. Option C is
therefore an incremental extension of the existing `3D View` tab:

1. Add `RobotModel3D` and `RobotState3D` as the shared MuJoCo model/FK state
   API. The main window loads one `MjModel` and shares it with reference-frame
   lookup, the trajectory backend, and the live 3D editor.
2. Extend the existing OpenGL canvas with cached MuJoCo geometry display lists,
   body transforms, a three-axis translation gizmo, and cached trajectory ghost
   transforms. Geometry is built once; interaction updates only state/transforms.
3. Wrap the canvas in `RobotViewer3D`, adding joint controls, end-effector/IK
   status, and trajectory playback/ghost controls without changing the 2D tabs.
4. Feed generated backend states into the live viewer and preserve the separate
   MuJoCo process tab as the high-fidelity CSV playback/debug path.
5. Add headless sanity tests for model discovery, joint updates/FK, body poses,
   demo trajectories, and ghost-cache replacement.

Initial scope keeps the existing X/Z target drag behavior compatible and adds
axis-constrained X/Y/Z translation in 3D. Rotation handles are deferred; the
state/IK API accepts a target orientation so they can be added without another
architecture change.

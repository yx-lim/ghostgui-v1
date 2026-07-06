# Committed/preview and 3D picking diagnosis (historical)

`RobotViewer3D` currently owns one `RobotState3D` named `robot_state`. That one
state is used as the IK seed, overwritten after every accepted drag substep,
drawn as the main robot by `RobotCanvas3D`, exposed through the joint sliders,
and written into `RobotStateTimeline` by
`update_current_keyframe_from_robot_state()`. The drag-finished callback also
updates the GUI trajectory immediately. There is no distinct preview qpos.

The visible robot reads `robot_state.mj_data.geom_xpos/geom_xmat`. Trajectory
ghosts are separate cached FK transform arrays, but they are playback/history
samples rather than an editable goal robot. Timeline selection calls
`set_robot_state_for_current_time()`, which loads the selected qpos directly
into the same visible/editable state.

The transform gizmo already converts mouse coordinates to camera rays and
picks its own sphere, arrows, and rings. Robot geometry is drawn from MuJoCo
geom transforms, but no mouse path currently ray-tests those geoms or maps a
geom body to an adapter logical frame.

MoveIt-style behavior therefore needs two persistent `MjData` states sharing
the immutable model: a timeline-backed committed state and a disposable
preview state. Drag IK/collision checks and preview-mode joint sliders must use
only preview qpos. The renderer can reuse the same compiled geometry lists to
draw committed transforms normally and preview transforms with an orange
alpha override. Only Accept writes preview qpos to the timeline. Plan creates
ghost samples without committing; Cancel resynchronizes preview from committed.
Double-click selection can reuse the existing camera ray, intersect MuJoCo
geom bounding spheres, map the closest geom to its body, then ask the model
adapter for the nearest logical editable frame.

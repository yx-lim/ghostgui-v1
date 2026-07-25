# One-shot reset and TCP sphere diagnosis

> Historical note: this diagnosis may not describe the current implementation.
> See [Preview And Keyframe Concepts](../concepts.md) for supported behavior.

## Reset/playback

- `RobotViewer3D.reset_robot_pose()` is called directly by the
  **Reset 3D Pose** button. It immediately applies `RobotModel3D.home_qpos`;
  there is no reset mode, pending flag, or callback stored in the timer.
- `RobotViewer3D._advance_playback()` is the playback-timer callback. It
  advances elapsed trajectory time and samples the surrounding qpos states;
  the Time slider and derived frame readout follow that sampled time.
- The apparent repeated reset is caused by `_refresh_timeline_trajectory()`:
  after an edit/reset it replaces `robot_trajectory` with the editor timeline.
  The reset home-qpos keyframe is then revisited every time that short list
  loops. The reset callback itself is not repeating.

The fix keeps the editable qpos timeline/ghost cache separate from the explicit
playback list. Reset pauses playback, cancels any active gizmo drag, applies home
qpos once to the current edit time, updates FK/sliders/gizmo/keyframe/ghosts,
and returns without leaving persistent reset state.

## TCP gizmo

- `TransformGizmo` owns arrow/ring picking and drag state; `RobotCanvas3D`
  renders it and forwards candidate transforms to `RobotViewer3D`.
- The existing origin point is decorative. There is no free-translation handle
  and no TCP drag-plane state.

The fix adds `HOVER_TRANSLATE_FREE` / `DRAG_TRANSLATE_FREE`, renders a small
central sphere, and intersects mouse rays with a camera-facing plane through the
starting TCP. It changes position only, preserves quaternion orientation, and
uses the existing collision-aware candidate IK acceptance path.

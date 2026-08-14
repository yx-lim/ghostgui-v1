# Preview And Keyframe Concepts

GhostGUI deliberately separates temporary edits from saved trajectory state.
Understanding that separation prevents accidental exports and makes timeline
behavior predictable.

## State Flow

```text
model home pose
      ↓
committed state at the active time
      ↓ edit target or Joint Angles
Orange preview
      ├─ Preview Path → adaptive validation ghosts
      ├─ cancel/time change → discard
      └─ Commit Keyframe → state and neighboring-path safety gate
                                ├─ blocked → keep diagnostic preview
                                └─ safe → committed Keyframe
                                               ↓
                                      trajectory candidate
                                               ↓
                                      adaptive safety gate
                                ┌──────────────┼──────────────┐
                              safe      repairable ground    blocked
                                ↓               ↓                ↓
                       playback/export   review + recheck   quarantine/edit/
                                                            explicit reroute
```

## Committed State

The model-colored robot represents the pose accepted at the active timeline
time. It is the reference state for edits and the source used by Keyframes,
generation, and pose export.

Moving to a time without an existing Keyframe creates an editable state derived
from the timeline. Existing Keyframes at other times remain independent.

## Orange Preview

Dragging the transform gizmo, changing a pose value, or changing a value in
**Joint Angles** starts an Orange preview. The preview is solved in a separate
MuJoCo state, so the committed robot remains fixed while the proposed result
changes.

Requested blocking contacts remain highlighted for diagnosis while the Orange
preview clamps at the last safe IK substep. Intended support contact is valid,
shallow advisory contact is reported, and blocking penetration cannot be
promoted. Changing the selected time discards an uncommitted preview.

Preview opacity affects only the Orange preview. It does not change the Qt
window or scene background opacity.

## Preview Path

**Preview Path** visualizes the transition between the committed pose and the
Orange preview. It checks finite qpos values, joint limits, and collisions by
adaptively refining the manifold-interpolated path between states. Unsafe
intervals remain visible as red ghosts.

GhostGUI displays the temporary path ghosts even when collision warnings are
present. The command does not change a Keyframe, committed state, or export.
Commit and Generate run their safety gates whether or not Preview Path was
selected manually.

## Commit Keyframe

**Commit Keyframe** performs the save operation for the active time:

1. It rejects a non-finite preview, a joint-limit violation, or blocking
   penetration.
2. It adaptively validates the affected paths to the neighboring Keyframes.
3. It copies the preview into the committed MuJoCo state.
4. It records the committed qpos state at the active time.
5. It captures the registered logical target frames for trajectory generation.
6. It advances by the configured Keyframe interval when possible.

If there is no active preview, the current committed pose is recorded directly.
The same safety checks still apply. Committing at an existing time updates that
Keyframe rather than adding a second state at the same time.

## Reset And Delete

**Reset** is a one-time action at the active time. It pauses playback, cancels
an active drag, and restores the selected model's home qpos.

**Delete Keyframe** removes both the logical target data and qpos state stored at
the active time when present. It does not delete neighboring Keyframes.

## Timeline Retiming

The **Timeline** menu changes Keyframe timestamps without re-solving the robot.
**Insert Time** creates a held interval, **Shift Entire Motion** translates all
Keyframes by one offset, **Move Time Range** relocates a non-overlapping
inclusive range, and **Scale Time Range** changes actual motion speed around a
fixed range start.

Retiming always treats the logical End Effector targets and committed qpos
states as one edit. GhostGUI validates the complete result before replacing
either source, so a range conflict or negative timestamp leaves both unchanged.
Generated motion is derived data and is cleared after a successful retime; run
**Generate** again. The complete replacement is one Undo/Redo action.

Insert Time samples both sources at the insertion boundary and writes the same
pose at the beginning and end of the opened interval. This makes the inserted
time a hold instead of stretching the preceding interpolation. Move Time Range
is a literal Keyframe move: it does not layer, blend, or independently combine
limbs, and a conflicting destination is rejected.

Scale Time Range applies `new time = range start + (old time - range start) /
speed`. Thus `2×` halves a range's duration while `0.5×` doubles it. Slower
scaling is rejected if the expanded range would overlap a later Keyframe.
Optional Export-interval snapping applies to every resulting Keyframe and is
also preflighted for timestamp collapse. Because this changes authoritative
Keyframe times, it affects Generate and all export formats.

## Generation And Playback

Generation samples the logical target tracks created by the Keyframes and the
complete qpos states stored by **Commit Keyframe**. A committed qpos is exact at
its Keyframe time. Between Keyframes, manifold-aware qpos interpolation provides
a posture reference while Cartesian End Effector targets remain primary IK
constraints. This preserves redundant elbow and wrist choices without allowing
posture to intentionally reduce End Effector accuracy.

Legacy or target-only trajectories without complete committed timeslices retain
target-only IK behavior. If a logical target conflicts with an exact qpos
anchor, generation stops and asks for the Keyframe to be recommitted.

The solver output is a candidate until GhostGUI validates every state and the
interpolated motion between adjacent states. Two safe Keyframes or generated
rows do not prove that the motion between them is safe. A candidate with
blocking penetration is not installed as the active generated motion.

**Export interval** controls this uniform sampling time step from `0.01 s` to
`10.00 s`. The default `0.01 s` interval is equivalent to 100 Hz. This is
independent of the **Keyframe interval**, which only advances the editing time
after **Commit Keyframe**.

Adaptive collision validation is also independent of the Export interval. It
refines a path according to motion and clearance instead of assuming that the
generated output rows are sufficient collision samples.

Every exact qpos anchor must lie on the uniform export-time grid. If a Keyframe
time is not divisible from the trajectory start by the selected **Export
interval**, generation stops with an alignment message instead of inserting a
short nonuniform sample.

Playback speed changes wall-clock viewing speed, not the trajectory timestamps.
DSMS motion speed is different: it divides elapsed timestamps during DSMS
export, producing an actual slower or faster downstream reference while leaving
the qpos path unchanged. It does not affect MuJoCo or mjlab exports.
Playback ghosts and Preview Path ghosts are temporary visualizations and are
not exported as additional states.

## Ground Repair And Rerouting

GhostGUI applies a hard projection only for ground penetration that can be
removed by raising one movable floating root over a flat ground plane. Live
edits and generated samples report the applied lift. Generated projection is
rejected if it would violate a required End Effector target or exact Keyframe
anchor.

For a quarantined path, a between-state ground sweep can receive a lifted
waypoint as a reversible candidate. The complete candidate is adaptively
revalidated before acceptance.

This repair does not apply to fixed-base models, uneven terrain, other
environment obstacles, or body-to-body collision. Those failures remain blocked
until the motion is edited. **Try Safe Reroute**, when offered, performs a
bounded local Joint Angle search with fixed endpoint qpos values; it is never
forced, never silent, and may fail. It is not a global or dynamic planner.

## Import And Export Rule

Exports use committed or generated states. An Orange preview is intentionally
excluded. If a proposed pose should appear in an export, use **Commit Keyframe**
first.

Imported motion is checked before it can replace the active safe generated
result. Blocking penetration places the imported path in quarantine for
inspection. It cannot be committed or exported until it is repaired or
rerouted and passes state and between-state validation.

Export repeats the path-safety gate. Intended support contact remains valid,
advisory contact is reported, and blocking penetration stops the write.

Motion safety describes the modeled kinematic path. It does not guarantee
balance, actuator feasibility, controller tracking, or safety on physical
hardware.

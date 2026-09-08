# Motion Assistant Security And Data Boundaries

The Motion Assistant is an optional authoring aid, not a robot controller or a
safety system. Provider output is untrusted. GhostGUI converts that output into
strict compact motion operations, executes deterministic application services on a
detached working copy, validates the result, and waits for human review before
Accept can change the committed motion.

## Data Sent To A Provider

A text edit can send:

- the user's instruction;
- a compact numerical summary containing the robot model, named Joint Angles,
  root pose, logical-frame and End Effector FK, pelvis/torso state, selection,
  current time, representative samples, and protection context;
- up to eight automatically captured timestamped renders when vision is
  supported;
- the strict compact operation contracts; and
- after a structural parsing failure, at most one compact repair payload with
  the original instruction, exact error, and expected contract.

The compact context intentionally excludes raw qpos values, project file paths,
terminal logs, credentials, and unrestricted application state. A project or
motion name can be present in the semantic context, so avoid sensitive names
when using an external provider.

Selection and camera details are captured from existing GUI owners when the
request begins; the Motion Assistant does not maintain a second document,
timeline, or camera selection model. A time interval is disclosed only when at
least two distinct Keyframe times are selected.

Every image has explicit time metadata and a visible timestamp overlay. Refine
renders the staged candidate, not the committed original. Before/after pairs,
when used by a developer-only legacy path, use identical timestamps. Treat
anything visible in those renders as data disclosed to the selected provider.
GhostGUI does not upload video.

Provider handling, retention, and regional processing of submitted content are
governed by the selected provider and account. Review those terms before using
Gemini or Claude with confidential motion.

## Credentials

GhostGUI supports Bring Your Own Key. A session-only key remains in process
memory. A securely stored key is written through the operating-system keyring.
Official provider environment variables are also supported. Keys are never
written to a GhostGUI project or plain UI preferences, included in prompts, or
placed in comparison reports.

The session connection-test cache stores only provider/model strings and a
SHA-256 fingerprint of the effective credential and its configuration source.
It never retains the plaintext key, is not persisted, and disappears when the
application exits.

Standard editing, playback, import, and export do not require an AI package,
network connection, or provider credential.

## Enforced Boundaries

- The default TrajectoryEditSpec vocabulary is an explicit allowlist with closed
  argument schemas; the legacy ToolRegistry remains developer-only.
- There is no shell, filesystem, arbitrary-code, raw-qpos-trajectory, RL,
  hardware, or DSMS tool.
- Request time, instruction/context size, response size, output tokens,
  operation count, sparse Keyframes, numerical samples, and images are bounded.
- Normal edits use one planning request. Only structural parsing failure may use
  one repair request, after which autonomous execution stops.
- A rendered frame is limited to 8 MiB, and provider capability limits still
  constrain the total image count.
- Normal Apply/Refine receives no executable tool declarations. It accepts a
  compact structured plan and executes deterministic local handlers only.
- Gemini makes one outbound attempt by default. Explicit transient-server retry
  settings never make quota-exhaustion or HTTP 429 responses retryable.
- Baseline user-authored content may change only within the explicit operation
  scope. Protected content and in-session human corrections take priority over
  later AI edits. Missing provenance is treated as user-owned.
- Joint Angle tools stage qpos plus FK-derived affected logical Keyframes in one
  atomic replacement. Provenance checks cover both representations before
  mutation, preventing a partial qpos-only edit.
- Local motion validation checks finite values, time and model contracts, qpos
  shape, Joint Angle limits, logical TargetFrame
  names, and same-time TargetFrame/qpos forward-kinematics consistency. Its
  machine-readable result explicitly identifies this scope as structural and
  kinematic and reports that dynamic feasibility was not assessed. Collision
  observations are surfaced separately as authoring warnings.
- The provider-facing protection tool can only add protection. Removing a
  protection requires a human-owned path outside autonomous tool execution.
- Accept uses one atomic `ReplaceMotionState` command and rejects a session if
  the committed document changed after the working copy was created.
- Whole-motion candidate preview is local and read-only. Scrub and playback
  ticks sample the detached working qpos timeline into the existing Orange
  preview state while a separate viewer state samples the committed reference;
  they emit no editable-time or per-Keyframe commit operation.

Provider comparison performs all fairness checks before making a provider
request. Every candidate must start with an identical committed motion and a
fresh, separate detached session. Reports compare validated semantic calls and
motion digests, not provider prose, raw motion values, or provider-native IDs.

Development record/replay is not exposed by the production UI. `RecordedProvider`
requires explicit development mode and a caller-supplied response sanitizer.
Its JSON store persists only the deterministic request fingerprint and the
sanitized normalized response; request prompts, context text, rendered image
bytes, and credentials are not written. Use synthetic inputs and keep recording
files outside the repository. `ReplayProvider` fails closed when the provider,
model, normalized prompt/context, image digest, or semantic schema differs.

Motion provenance and protection use opaque entity identifiers behind the
metadata service. The versioned project-workspace payload persists author,
protection, and ordering metadata without exposing timestamp identity to the AI
layer. Legacy workspaces with no payload seed all existing motion as user-owned.

## Failure And Cancellation

Authentication errors, rate limits, unavailable models, malformed responses,
timeouts, oversized payloads, failed tools, and cancellation stay inside the
Motion Assistant workflow. They do not commit the working copy or disable the
rest of GhostGUI. Motion-mutating controls remain disabled during an active
request so the response cannot race a committed edit; Accept also performs a
revision check as a final stale-state guard.

Do not rely on visual critique or structural/kinematic motion validation as
proof of real-world stability, continuous-path collision safety, actuator
feasibility, contact stability, or hardware safety. Dynamic validation remains
future DSMS/v3.2 work. Review and test accepted motion through the appropriate
robotics workflow before deployment.

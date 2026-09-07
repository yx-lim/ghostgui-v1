# Motion Assistant Security And Data Boundaries

The Motion Assistant is an optional authoring aid, not a robot controller or a
safety system. Provider output is untrusted. GhostGUI converts that output into
strict semantic tool requests, executes deterministic application services on a
detached working copy, validates the result, and waits for human review before
Accept can change the committed motion.

## Data Sent To A Provider

A text edit can send:

- the user's instruction;
- a compact semantic summary containing the robot model, logical-frame, End
  Effector, Joint Angle, selection, current timeline time, selected Keyframe
  interval, active view/3D camera, Keyframe-time, and protection context;
- the strict semantic tool names and argument schemas; and
- after a local operation failure, at most one compact repair payload containing
  the failed operations and reasons, successful operations already applied,
  updated semantic context, and important user constraints.

The compact context intentionally excludes raw qpos values, project file paths,
terminal logs, credentials, and unrestricted application state. A project or
motion name can be present in the semantic context, so avoid sensitive names
when using an external provider.

Selection and camera details are captured from existing GUI owners when the
request begins; the Motion Assistant does not maintain a second document,
timeline, or camera selection model. A time interval is disclosed only when at
least two distinct Keyframe times are selected.

Critique and Visual refine additionally send 4--8 rendered images from the
current 3D camera. **Verify visually** sends 4--8 original/candidate image pairs.
Every image has explicit time metadata, and each verification pair uses an
identical timestamp. Treat anything visible in those renders as data disclosed
to the selected provider. GhostGUI does not upload video.

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

- The ToolRegistry is an explicit allowlist with closed argument schemas.
- There is no shell, filesystem, arbitrary-code, raw-qpos-trajectory, RL,
  hardware, or DSMS tool.
- Agent provider turns, tool calls, request time, instruction size, response
  size, tool-result size, and requested output tokens are locally bounded.
- Normal text edits use one planning request. A local operation failure may use
  one replacement-operation request, after which autonomous execution stops.
- A rendered frame is limited to 8 MiB, and provider capability limits still
  constrain the total image count.
- Critique, Visual refine, and visual verification receive no executable tool
  declarations. Visual refine accepts semantic operation data through a strict
  structured schema and executes it only through the local ToolRegistry.
- Visual refine makes one multimodal planning request and does not automatically
  repeat. **Verify visually** is an optional separate read-only request.
- Gemini makes one outbound attempt by default. Explicit transient-server retry
  settings never make quota-exhaustion or HTTP 429 responses retryable.
- User-authored and protected Keyframes take priority over later AI edits.
  Missing provenance is treated as user-owned, never as implicit AI ownership.
- Joint Angle tools stage qpos plus FK-derived affected logical Keyframes in one
  atomic replacement. Provenance checks cover both representations before
  mutation, preventing a partial qpos-only edit.
- Local motion validation checks finite values, time and model contracts, qpos
  shape, Joint Angle limits, configured blocking collisions, logical TargetFrame
  names, and same-time TargetFrame/qpos forward-kinematics consistency. Its
  machine-readable result explicitly identifies this scope as structural and
  kinematic and reports that dynamic feasibility was not assessed.
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

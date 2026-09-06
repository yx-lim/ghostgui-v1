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
  Effector, Joint Angle, selection, Keyframe-time, and protection context;
- the strict semantic tool names and argument schemas; and
- after a local operation failure, at most one compact repair payload containing
  the failed operations and reasons, successful operations already applied,
  updated semantic context, and important user constraints.

The compact context intentionally excludes raw qpos values, project file paths,
terminal logs, credentials, and unrestricted application state. A project or
motion name can be present in the semantic context, so avoid sensitive names
when using an external provider.

Critique and Visual refine additionally send 4--8 rendered images from the
current 3D camera. Every image has explicit time metadata, and before/after
images use identical timestamps. Treat anything visible in those renders as
data disclosed to the selected provider. GhostGUI does not upload video.

Provider handling, retention, and regional processing of submitted content are
governed by the selected provider and account. Review those terms before using
Gemini or Claude with confidential motion.

## Credentials

GhostGUI supports Bring Your Own Key. A session-only key remains in process
memory. A securely stored key is written through the operating-system keyring.
Official provider environment variables are also supported. Keys are never
written to a GhostGUI project or plain UI preferences, included in prompts, or
placed in comparison reports.

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
- Critique and visual comparison requests receive no edit tools.
- Automatic visual refinement permits at most two edit iterations by default
  and always ends with a read-only assessment.
- User-authored and protected Keyframes take priority over later AI edits.
- Accept uses one atomic `ReplaceMotionState` command and rejects a session if
  the committed document changed after the working copy was created.

Provider comparison performs all fairness checks before making a provider
request. Every candidate must start with an identical committed motion and a
fresh, separate detached session. Reports compare validated semantic calls and
motion digests, not provider prose, raw motion values, or provider-native IDs.

## Failure And Cancellation

Authentication errors, rate limits, unavailable models, malformed responses,
timeouts, oversized payloads, failed tools, and cancellation stay inside the
Motion Assistant workflow. They do not commit the working copy or disable the
rest of GhostGUI. Motion-mutating controls remain disabled during an active
request so the response cannot race a committed edit; Accept also performs a
revision check as a final stale-state guard.

Do not rely on visual critique or deterministic motion validation as proof of
real-world stability, collision safety, actuator feasibility, or hardware
safety. Review and test accepted motion through the appropriate robotics
workflow before deployment.

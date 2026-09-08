# Motion Assistant

The right-sidebar **Motion Assistant** creates and edits motion from ordinary
language. AI support is optional; standard GhostGUI editing works without an AI
package or API key.

## Configure A Provider

Open **Settings**, choose Gemini or Anthropic, and select a compatible model.
Keys can be stored in the operating-system credential store, kept only for the
current process, or supplied through `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or
`ANTHROPIC_API_KEY`. GhostGUI never saves a key in a project or plain UI
preferences. **Test Connection** makes a small read-only provider request and
caches a successful result for that provider, model, and key identity during
the current process.

The pinned optional AI baseline is:

- `google-genai==2.21.0` with `gemini-3.7-flash`;
- `anthropic==1.4.0` with `claude-sonnet-5`;
- `keyring==25.7.0` for secure credential storage.

## Create Or Edit Motion

1. Optionally select a Keyframe, End Effector, Joint Angle, or time range.
2. Describe the result naturally, such as “Move the entire robot 5 cm higher,”
   “keep both hands planted,” or “create a 5-second burpee.”
3. Choose **Apply**.
4. Review the proposed changes and **Orange preview**. Use **Preview candidate**
   to scrub or play the whole staged result.
5. Choose **Accept**, **Reject**, or enter another direction and choose
   **Refine**.

GhostGUI automatically supplies a bounded description of the active robot and
motion. It includes named Joint Angles, root pose, End Effector forward
kinematics, pelvis and torso state, current time, the selected interval, and
8–20 representative numerical samples. For vision-capable models, normal
**Apply** and **Refine** also capture up to eight rendered views automatically.
Each image carries an explicit motion time in both its metadata and a visible
timestamp overlay. The user does not need to take screenshots or calculate
coordinates.

If frame capture is unavailable, numerical context is still used and the
candidate shows a warning. A text-only provider can use the same workflow
without images.

## How Motion Is Produced

The provider returns a compact motion specification, not CSV or dense qpos.
GhostGUI performs all numerical work locally using its existing robot model,
timeline interpolation, FK, IK, Joint Angle editing, retiming, and constraint
machinery. Supported intent includes:

- exact root offsets over a time range;
- holding a sampled whole-body, Joint Angle, or joint-group pose;
- retiming an interval;
- explicit named Joint Angle and joint-group targets;
- End Effector targets and locks through IK;
- pelvis, torso, and other logical-frame targets through IK; and
- sparse semantic Keyframes for new motion, interpolated locally to a normal
  dense qpos trajectory.

The default workflow makes one provider planning request. If the returned
structure is malformed, GhostGUI may make exactly one repair request containing
the original instruction, parser error, and expected compact contract. An IK or
execution failure does not start an autonomous repair loop. No path accepts raw
dense qpos generation, arbitrary code, shell commands, DSMS, RL, hardware, or
robot-control operations.

The older semantic ToolRegistry/PlanExecutor workflow remains available in the
codebase for compatibility and developer use, but it is not the normal Motion
Assistant path.

## Working Copy And Review

AI work happens in a detached document-level working copy. Orange is only its
presentation. **Accept** atomically replaces the committed motion and creates
one history entry, so Undo restores the exact previous motion. **Reject** drops
the complete candidate. **Refine** samples and renders the current staged
candidate rather than restarting from committed motion, and retains only the
original goal plus a bounded number of recent refinements.

Motion-mutating controls are currently disabled while the UI owns an unresolved
AI session. The session and motion-service boundaries already support manual
edits on the working copy: those edits are recorded as human-authored, invalidate
validation, and take priority over later AI refinement. This direct-manipulation
UI can therefore be enabled later without redesigning the AI layer.

Existing motion is conservatively seeded as human-authored. Metadata and
protection persist with projects, and unknown content fails closed. Baseline
human motion may change only when the requested operation explicitly scopes it;
protected content and human corrections made during the current session remain
immutable to later AI operations. All identity lookup stays behind the metadata
service so timestamp-based MVP identity can be migrated to stable Keyframe IDs.

## Validation And Warnings

After local execution, GhostGUI validates the exact current candidate revision.
Malformed trajectories, invalid time data or qpos width, NaN/Inf, out-of-range
Joint Angles, inconsistent FK targets, impossible execution, and protection
violations prevent **Accept**. Collision observations are authoring warnings and
remain visible for review; they do not claim dynamics, balance, actuator,
contact-stability, or hardware feasibility. Any later candidate mutation
invalidates validation until that exact revision passes again.

## Failures, Limits, And Diagnostics

Missing credentials, authentication errors, provider rate limits, timeouts,
network failures, malformed responses, and cancellation leave committed motion
unchanged. Requests are bounded by instruction/context size, 8 images at a
512-pixel maximum dimension, 4,096 default output tokens, 16 operations, 16
sparse Keyframes, and a 90-second default timeout. Gemini uses one outbound SDK
attempt by default and does not automatically retry quota errors.

Developer diagnostics are disabled by default. Setting `GHOSTGUI_AI_DEBUG=1`
records bounded, secret-scrubbed JSON in `.ghostgui-ai-debug/`, which is
gitignored. It records prompts, selected timestamps and frame hashes, normalized
responses, parsed specifications, execution and validation results, usage, and
latency—never API keys, authorization headers, keyring contents, or image bytes.

See [Motion Assistant Security And Data Boundaries](ai_security.md) for provider
disclosure and safety boundaries. Normal automated tests use MockProvider and
consume no provider credits; live checks are explicit and require local SDKs
and credentials.

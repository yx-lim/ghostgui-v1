# Motion Assistant

The right-sidebar **Motion Assistant** applies focused language edits to the
current motion. AI support is optional: standard GhostGUI editing continues to
work without an AI package or API key.

## Configure Gemini Or Claude

Open **Settings** in the Motion Assistant and choose the provider and model.
Choose **Gemini** or **Anthropic**. The model field is editable so a newer
compatible Gemini or Claude model can be selected without changing GhostGUI.

Enter a Bring Your Own Key only when AI support is wanted. With **Store
securely** enabled, GhostGUI writes the key to the operating-system credential
store. Without it, the entered key remains in memory for the current process.
GhostGUI never writes the key to a project or its plain UI preferences. The
official `GOOGLE_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`
environment variables remain supported. Session-only keys are kept separately
per provider so switching providers cannot reuse a key with the wrong service.
**Test Connection** performs a small provider request without changing the
motion. A successful result is cached for the current provider, model, and API
key identity during this GhostGUI process. Repeating the identical test uses
the cached result; changing any of those settings invalidates it. GhostGUI does
not test the connection automatically at startup.

## Edit And Review

1. Select the relevant logical frame and active time.
2. Describe one focused motion edit and choose **Apply**.
3. Continue inspecting the camera and timeline while the request runs.
4. Review the proposed changes and Orange preview.
5. Choose **Accept**, **Reject**, or enter another instruction and choose
   **Refine**.

AI changes are made on a detached working copy. The Orange preview presents
that copy but is not its source of truth. **Accept** atomically replaces the
committed motion and creates one history entry. **Reject** discards the complete
working copy. **Refine** continues from the staged copy rather than restarting
from committed motion.

Each normal **Apply** or **Refine** action asks the provider once for a complete,
structured semantic plan. GhostGUI then validates and executes every operation
locally through its strict tool registry, validates the resulting working copy,
and builds the proposal summary from recorded local results. Plans containing
one operation or many operations therefore use the same single provider
request; GhostGUI does not send a second request merely to obtain a “done”
message.

If a planned operation fails local validation or execution, GhostGUI may make
one additional repair request. That request contains the original intent,
compact failure information, successful operations already applied, the
updated semantic motion context, and important user constraints. It requests
replacement operations only. If those replacements also fail, GhostGUI stops
without a third request and presents the partial working copy plus unresolved
failure information for human review.

Motion-mutating direct controls are temporarily disabled while the current UI
owns an unresolved AI session. This prevents them from editing the committed
document by accident. The underlying session already distinguishes user and AI
authorship so direct manipulation can later target the same working copy.

## Visual Critique

Choose **Critique** to inspect the motion without editing it. The Motion
Assistant captures 4--8 representative frames using the current 3D camera,
adds a visible timestamp to every frame, and asks the configured vision-capable
provider for structured observations. Leave the prompt empty for a general
critique, or enter a focused question first.

Observations use approximate motion times such as “Around 2.10 s” rather than
image indexes. Critique-only mode sends no semantic edit tools, does not open an
AI edit session, and does not change committed motion. If an Orange preview is
already staged, **Critique** inspects that detached working copy while leaving
it available for Accept, Reject, or Refine.

## Visual Refinement

After an edit is staged, choose **Visual refine** to inspect 4--8 timestamped
frames from the staged candidate. One multimodal provider request returns both
structured observations and a complete semantic `MotionEditPlan`. GhostGUI
then executes that plan locally through the same strict `PlanExecutor` used by
ordinary AI edits and updates the Orange preview. It never sends the visual
observations through a second language-model turn, asks the provider for raw
trajectory samples, or executes provider-generated code.

Visual refinement does not automatically repeat. Any text entered before
choosing **Visual refine** is treated as additional user direction, and
user-authored or protected Keyframes retain priority.

Choose **Verify visually** when a staged candidate should be compared with the
committed motion. Verification sends original and candidate frames captured at
identical timestamps in one separate, read-only request. It reports which view
better satisfies the goal and any remaining timestamped observations, but
cannot edit the working copy. Thus Visual refine normally uses one provider
request; Visual refine followed by explicit verification uses two.

## Failures And Cancellation

Use **Cancel request** to stop a running provider call. Missing credentials,
authentication failure, rate limits, timeouts, network failure, and malformed
responses are shown inside the assistant. These failures leave committed
motion unchanged and do not stop standard GhostGUI workflows.

Gemini uses one outbound SDK attempt by default so a transient failure cannot
silently consume extra free-tier requests. Developers may explicitly configure
a small retry count for transient server failures. Quota exhaustion and HTTP
429 responses are never retried automatically.

The assistant can call only GhostGUI's registered semantic motion tools. It
cannot execute arbitrary code or generate an unrestricted raw qpos trajectory.
See [Motion Assistant Security And Data Boundaries](ai_security.md) for the
exact provider disclosure, credential, payload-limit, and safety contracts.

## Compare Providers

Provider evaluation uses the same committed motion, selection, instruction,
semantic ToolRegistry, and a fresh detached session for each model. Compare
validated tool arguments, edit authorship, deterministic validation, and the
resulting motion rather than wording in the provider response. Token use and
provider turns are operational measurements, not semantic quality scores.

Normal tests use MockProvider and do not consume provider credits. Live
Gemini-versus-Claude checks are explicit manual smoke tests requiring both SDKs
and locally configured credentials.

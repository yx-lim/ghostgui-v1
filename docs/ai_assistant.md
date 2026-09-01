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
motion.

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

After an edit is staged, choose **Visual refine** to compare the committed
motion with the staged candidate at identical timestamps. GhostGUI separates
each cycle into observation, a small semantic plan, and execution through the
same strict motion tools used by ordinary AI edits. It never asks the provider
for raw trajectory samples or executes provider-generated code.

Automatic refinement is capped at two edit iterations. GhostGUI then performs
one final read-only comparison and returns control for Orange preview review,
Accept, Reject, or manual Refine. If the final comparison still finds an issue,
the panel reports that the automatic limit was reached instead of continuing
indefinitely. Any text entered before choosing **Visual refine** is treated as
additional user direction and user-authored or protected Keyframes retain
priority.

## Failures And Cancellation

Use **Cancel request** to stop a running provider call. Missing credentials,
authentication failure, rate limits, timeouts, network failure, and malformed
responses are shown inside the assistant. These failures leave committed
motion unchanged and do not stop standard GhostGUI workflows.

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

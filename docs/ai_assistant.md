# Motion Assistant

The right-sidebar **Motion Assistant** applies focused language edits to the
current motion. AI support is optional: standard GhostGUI editing continues to
work without an AI package or API key.

## Configure Gemini

Open **Settings** in the Motion Assistant and choose the provider and model.
The model field is editable so a newer compatible Gemini model can be selected
without changing GhostGUI.

Enter a Bring Your Own Key only when AI support is wanted. With **Store
securely** enabled, GhostGUI writes the key to the operating-system credential
store. Without it, the entered key remains in memory for the current process.
GhostGUI never writes the key to a project or its plain UI preferences. The
official `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables remain
supported. **Test Connection** performs a small provider request without
changing the motion.

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

## Failures And Cancellation

Use **Cancel request** to stop a running provider call. Missing credentials,
authentication failure, rate limits, timeouts, network failure, and malformed
responses are shown inside the assistant. These failures leave committed
motion unchanged and do not stop standard GhostGUI workflows.

The assistant can call only GhostGUI's registered semantic motion tools. It
cannot execute arbitrary code or generate an unrestricted raw qpos trajectory.

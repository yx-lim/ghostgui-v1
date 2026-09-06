# Testing

Run automated tests before submitting changes and perform focused GUI checks
for behavior that depends on rendering or interaction.

## Automated Suite

From a checkout, use the isolated runner so local application settings,
projects, caches, and XDG state cannot affect results:

```bash
python3 scripts/run_test_suite.py
```

The suite covers model adapters, IK and collision behavior, timeline semantics,
CSV playback, transaction recovery, cancellation and shutdown, visualization
lifecycles, transform-gizmo math, GUI controls, themes, status handling,
packaging contracts, and trajectory smoothing.

Run documentation validation separately:

```bash
python3 scripts/check_docs.py
python3 scripts/check_architecture.py
```

## Focused Tests

Use a test module while iterating:

```bash
python3 -m unittest tests.test_robot_viewer_timeline -v
python3 -m unittest tests.test_advanced_ik -v
python3 -m unittest tests.test_project_persistence.ResourceLocationTests -v
```

Choose the test that owns the changed contract rather than relying only on a
manual launch.

### Motion Assistant

The v3.0 AI integration suite is deterministic and needs no provider account,
API key, or internet connection:

```bash
python3 -m unittest tests.test_ai_v3_integration -v
```

It drives real context building, strict ToolRegistry validation, semantic
tools, the detached AIEditSession, working-copy Orange preview sampling, and
atomic Accept/Reject through scripted MockProvider responses. Scenarios cover
contextual retiming, End Effector IK, protected user constraints, editable
multi-Keyframe plans, manual-edit-then-Refine, cancellation, offline failure,
and rejection of arbitrary-code tool requests.

The v3.1 frame-selection boundary can be checked without Qt, OpenGL, or a
provider account:

```bash
python3 -m unittest tests.test_ai_frame_capture -v
```

It verifies the 4--8 frame bound, selected-interval and suspected-time hints,
and exact timestamp correspondence for original/candidate image pairs. The
GUI capture adapter adds the same timestamp as an image overlay and must run
on the GUI thread because it reads a `QOpenGLWidget` framebuffer.

Critique-only parsing, provider capability checks, cancellation, timeouts, and
the no-tool request boundary are covered by:

```bash
python3 -m unittest tests.test_ai_visual_critique -v
```

The bounded visual Observe/Plan/Edit contract is covered by:

```bash
python3 -m unittest tests.test_ai_visual_refinement -v
```

These tests verify complete timestamp-matched image pairs, comparison-only
provider turns, semantic ToolRegistry execution, unchanged committed motion,
the two-edit default, and the mandatory final read-only assessment.

Gemini and Claude live contract smoke tests are opt-in and perform four provider
requests: text, structured output, a non-executed tool request, and a tiny
generated-image vision check. They use GhostGUI's normal keychain/environment
credential lookup, print provider/model/SDK compatibility details, and never
print keys or raw provider responses:

```bash
python3 scripts/smoke_ai_provider.py gemini
python3 scripts/smoke_ai_provider.py anthropic
```

Pass `--model MODEL_ID` to check a different model. These commands consume live
provider quota. The **Live AI Provider Smoke** GitHub Actions workflow is also
available through `workflow_dispatch`; it is never triggered by a push, pull
request, or schedule. Normal CI does not require either provider key or consume
provider credits. Anthropic request and response conversion is covered without
network access by:

```bash
python3 -m unittest tests.test_ai_anthropic_provider -v
```

Provider comparison is likewise deterministic in CI:

```bash
python3 -m unittest tests.test_ai_provider_comparison -v
```

Prepare each comparison case from the same committed motion, selection, and
instruction, but give it a separate detached `AIEditSession`. The comparison
runner executes cases sequentially and reports normalized completion status,
semantic tool names and argument digests, tool success, edit authorship,
validation, and the resulting motion digest. Provider-native call IDs and prose
are deliberately excluded; token use and provider-turn counts are reported as
operational differences, not semantic quality. Live Gemini-versus-Claude runs
remain an explicit manual smoke test because they require both local SDKs and
credentials and can consume provider credits.

### Provider-request baseline

`RequestCountingProvider` measures calls at GhostGUI's normalized provider
boundary. The Phase 3 pre-refactor baseline is enforced by
`tests.test_ai_request_counting`:

| User action | Normalized provider requests |
| --- | ---: |
| One semantic edit | 2 |
| Four-operation generation (one parallel tool batch) | 2 |
| Failed semantic tool followed by recovery | 3 |
| Critique-only | 1 |
| One visual refinement iteration plus final assessment | 4 |
| Two visual refinement iterations plus final assessment | 7 |
| Test Connection | 1 |

These are measured workflow calls to `LLMProvider.generate`, not estimates.
Provider-SDK transport retries below that boundary are deliberately excluded.
Each successful semantic edit iteration currently needs a tool-producing turn
and a second completion-summary turn. Each visual iteration adds one structured
comparison request, and the bounded workflow always ends with one read-only
assessment request.

The AI regression suite also checks local instruction, response, tool-result,
output-token, and rendered-frame budgets. Comparison preflight failures must be
detected before any provider callback is invoked.

## Package Smoke Test

Build and inspect a wheel, then install it into a clean environment. Run the
smoke script from outside the checkout so source files cannot mask missing
package data:

```bash
python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir wheelhouse
python3 scripts/check_wheel.py --require-resources wheelhouse/*.whl
python3 -m venv --system-site-packages /tmp/ghostgui-wheel-smoke
/tmp/ghostgui-wheel-smoke/bin/python -m pip install --no-deps wheelhouse/*.whl
cd /tmp
/tmp/ghostgui-wheel-smoke/bin/python \
  /path/to/ghostgui/scripts/smoke_installed_package.py --load-model --gui
```

The smoke gate verifies the console entry point, all registered model sources,
theme and documentation resources, G1 compilation, visualization startup, and
clean GUI shutdown.

## Headless And Visual Gates

The regular suite uses Qt's offscreen platform. CI additionally starts an Xvfb
display with Mesa software OpenGL and runs:

```bash
GHOSTGUI_VISUAL_TESTS=1 QT_QPA_PLATFORM=xcb \
  python3 -m unittest tests.test_visual_smoke -v
```

That test requires a valid OpenGL context, captures the composed window, checks
that it is nonblank and opaque, and compares stable toolbar, tab, and sidebar
structure. It skips outside the explicitly configured visual environment.

## Manual GUI Checks

### Shared Workflow

- Start GhostGUI and confirm the selected model renders.
- Edit an end effector and confirm only the orange preview moves.
- Use **Preview Path** and confirm no keyframe is created.
- Use **Commit Keyframe** and confirm the timeline advances by the configured
  keyframe interval.
- Edit another time and confirm the earlier keyframe remains unchanged.
- Generate, play, pause, scrub, and export the trajectory.
- Confirm preview opacity does not make the application window transparent.
- Confirm reset affects the active time only.

### Motion Clips

- Commit an `A → B` motion, use **Copy Motion Range…**, and confirm copying
  does not change project history, dirty state, or generated motion.
- Leave an uncommitted Orange preview visible while copying and confirm the
  Motion Clip contains only committed Keyframes and robot poses, including
  Joint Angles, not the preview or generated samples.
- At the `B` time, use **Paste Motion Reversed at Current Time** and confirm the
  result is `A → B → A`. Confirm the shared `B` seam is one Keyframe and
  no pose values or Joint Angles were mirrored or negated.
- Close the source seam, use **Paste Motion at Current Time**, and confirm the
  forward copy preserves relative timing. Confirm an open seam or another
  conflicting destination rejects the complete paste without changing either
  targets or committed robot poses.
- Choose a destination whose interval crosses existing motion without sharing
  an exact Keyframe time and confirm the interior overlap rejects the complete
  paste without changing editor, history, playback, or generated state.
- Copy a range whose boundaries fall between Keyframes and confirm its
  materialized logical targets match forward kinematics of the sampled
  committed robot poses.
- Use **Repeat Motion…** with an additional-copy count in both **Forward** and
  **Ping-pong** modes. Confirm Forward repeats the original order and Ping-pong
  alternates direction.
- Confirm each successful Paste or Repeat clears generated motion, expands the
  timeline when needed, and is restored by one Undo/Redo action.
- Switch robot models and confirm the model-specific Motion Clip cannot be
  pasted into the other model.

### Motion Assistant

- Launch without an API key and confirm all standard editing remains usable.
- Open Motion Assistant settings, verify the API key is password-masked, and
  test both session-only and system-keyring storage.
- Apply a focused edit and confirm the committed motion stays unchanged while
  the Orange preview shows the staged working copy.
- Scrub the viewer timeline and move the camera while the provider request is
  running; confirm document-mutating controls remain unavailable.
- Use **Refine** twice and confirm each turn builds on the same staged result.
- Use **Accept** and confirm the entire session is one Undo/Redo entry. Repeat
  with **Reject** and confirm committed motion does not change.
- Cancel a request, remove network access, use an invalid key, and exercise a
  rate-limited response; confirm the error stays inside Motion Assistant.
- Run the same focused request with MockProvider and opt-in Gemini and Claude
  smoke configurations. Compare semantic tool results rather than provider-native
  response objects.

### Model Checks

- **G1:** confirm humanoid hand, foot, torso, and pelvis targets and detailed
  geometry.
- **Go2:** confirm twelve scalar leg joints, four foot targets, the ground, and
  COLLADA-derived visual geometry.
- **H2:** confirm the humanoid targets and bundled STL visuals.
- **Z1:** confirm base, wrist, and tool targets and six arm joints.
- **Generic manipulator:** confirm a synthetic nine-joint model exposes all
  Joint Angles while its selected arm chain excludes gripper-only joints.
- Switch away from a model and back; confirm its editor session is preserved.

### Import And Export

- Load a qpos file with the correct active-model width.
- Reject a qpos or trajectory with the wrong width or non-finite values.
- Import a DSMS reference folder and confirm `time.csv` and
  `qpos_<dof>dof.csv` are paired automatically, timestamps are preserved, and
  mismatched sample counts or active-model DoF are rejected.
- Import a G1 29-DoF mjlab CSV at `0.01 s` and another selected source sample
  interval; confirm `x, y, z, w` is converted to MuJoCo `w, x, y, z`, named
  joints return to compiled qpos addresses, and incompatible models are blocked.
- Export a committed pose and confirm an uncommitted orange preview is absent.
- Export MuJoCo, DSMS, and mjlab trajectories; confirm DSMS creates two files,
  both specialized formats apply Export interval to the current trajectory, and
  mjlab is limited to the G1 29-DoF contract.
- Export DSMS at `0.50×` and `2.00×`; confirm elapsed timestamps and reported
  duration/frequency scale correctly while qpos rows and sample count remain
  unchanged. Reject zero, negative, and non-finite speeds.
- Confirm `convert_ghostgui_to_dsms.py` and `ghostgui_to_mjlab.py` remain
  directly runnable from the terminal, including DSMS `--speed`.
- Import a model with relative meshes.
- Import a schema-2 model profile and confirm fixed/floating base policy,
  logical frames, End Effectors, joint groups, and passive-joint influence.
- Exercise the mesh-folder retry path for an unresolved model asset.
- Commit two redundant seven-joint G1 arm postures, generate motion, and confirm
  every generated anchor reproduces the complete committed qpos exactly.
- Confirm an off-grid committed Keyframe is rejected with an Export interval
  alignment message and a generic model cannot use the G1 analytic fallback.

### Path Independence

Launch the compatibility script from a different working directory:

```bash
cd /tmp
python3 /path/to/ghostgui/scripts/run_gui.py --model go2
```

Registry and model paths should remain anchored to the checkout.

## Documentation Checks

Documentation validation should reject:

- broken relative links;
- missing linked files or anchors;
- more than one level-one heading per public page;
- legacy user-facing workflow terminology;
- undocumented public pages;
- trailing whitespace.

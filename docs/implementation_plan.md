# GhostGUI Implementation Plan

Status: proposed execution guide; the features below are not implemented by
this document. Update the guide as milestones are accepted.

This combines the [UI/UX review](../review/01-uiux-redesign.md),
[software critique](../review/02-software-critique.md), and
[Claude integration review](../review/03-claude-agent-integration.md).
The reviews inform the backlog; current code and regression tests determine
the implementation. Follow [Architecture](architecture.md),
[Data Formats](data_formats.md), and the repository instructions throughout.

## 1. Planning Basis And Scope Decisions

### Capacity, Priorities, And Budget

- One implementer using Codex through ChatGPT Plus, with users available for
  acceptance testing. The implementer owns integration and release decisions;
  Codex performs implementation and automated checks; users assess the result.
- Priority: autonomous Claude control, then motion correctness and reliability,
  then the UI redesign. Correctness prerequisites still precede autonomy.
- Interpret the requested two agent experiences as external Claude Code through
  MCP and an autonomous assistant inside GhostGUI. Retain the headless CLI in
  the complete scope because it reuses the same services.
- Assume 16–21 focused hours for the weekend, including verification. A few
  prompts reduce handoffs; they do not remove debugging or acceptance work.
- Preserve every currently supported model, project workflow, and export.
  Use G1 for the first new agent scenario, with existing G1/Go2/H2/Z1 and
  imported-model regression coverage. Linux remains the primary tested
  platform; test locally on macOS and retain the experimental Windows path.
- Ship to a small internal team first. Public distribution follows a separate
  packaging, documentation, and platform gate.

**Full scope is retained, but completing all three reviews in one weekend is
not a credible release commitment.** The weekend commitment is a tested command
layer, MCP access, and the minimum correctness guards. Live GUI attachment and
the in-app loop are stretch milestones. The complete backlog below adds an
estimated 140–264 focused hours, including unfinished stretch milestones;
roughly 9–17 further weekends at 16 hours each. These are planning estimates,
not measured delivery rates. Re-estimate after the first end-to-end workflow.

ChatGPT Plus provides Codex usage subject to plan limits; subscription usage
and API billing are separate. Prompt count alone does not predict allowance
consumption. See [official OpenAI pricing guidance](https://learn.chatgpt.com/docs/pricing).
Do not assume the subscription funds Claude or application API calls.
Claude Code requires its own eligible authentication/billing setup; see
[Anthropic authentication](https://code.claude.com/docs/en/authentication).
The existing in-app adapter requires Anthropic API credentials; see
[Motion Assistant setup](ai_assistant.md).

For this budget, develop and test the in-app loop using Mock/Replay providers.
Leave paid execution disabled until a funded credential and explicit spending
limit are configured. Test external Claude live only if an existing usable
Claude account is available. MCP transport can be accepted without paid LLM
calls; that does not establish live Claude success. Do not purchase credits,
copy subscription tokens into the app, or add paid model judges by default.

### Reconcile Conflicting Recommendations

| Conflict | Implementation decision |
| --- | --- |
| Delete legacy AI pipelines versus evolve their agent machinery | Reuse `AIEditSession`, strict tool validation, deterministic operations, Mock/Replay, and useful loop mechanics. Retain the current compact editor until the new path passes parity checks. Remove redundant planners and hidden routes after their callers are replaced. Do not delete reusable dependencies first. |
| Remove explicit commits versus retain the redesigned Commit Keyframe workflow | Keep explicit Commit Keyframe and Orange preview as the default. Cover auto-key later as an opt-in authoring mode, after document/history correctness. Never change existing project behavior implicitly. |
| Automatic generation versus a visible Generate action | First add freshness tracking and cancellable Generate. Later add opt-in affected-segment background solving while retaining Generate as an explicit refresh and recovery action. |
| Remove historical hidden qpos Keyframes versus preserve old motions | Stop navigation creating new qpos states. Preserve all existing states. Offer a separate backed-up, opt-in cleanup analysis later; equality with interpolation alone cannot prove a state was accidental. |
| Quaternion storage and contact tracks versus full project compatibility | First use quaternion values internally behind existing serialization. Keep legacy save representation and defaults. Persist new features through explicitly versioned, optional extensions or a new format with a legacy save path, only after compatibility tests. |
| New contact labels/metadata versus unchanged export layouts | Keep legacy numeric files, ordering, quaternion conventions, timestamps, and filenames unchanged. Add opt-in sidecars or separately named formats. Never append columns to an existing CSV contract. |
| Make all application code Qt-free versus a weekend-sized change | Make new services, agent modules, and CLI imports Qt-free immediately. Move existing Qt adapters to `gui/` incrementally, preserving the launch composition root. Avoid a package-wide rewrite before MCP works. |
| Expose hidden visual buttons versus consolidate AI | Surface supported critique/verification through the shared agent tools and result UI. Remove redundant hidden implementations only after equivalent behavior is covered. |
| Copy review model IDs, prices, and beta SDK parameters versus working integration | Verify current official provider/MCP documentation and installed signatures during implementation. Record the tested versions. Use supported configured models; do not assume the review's presets, prices, system-message placement, or beta flags work. |

Compatibility means old projects remain readable, saveable through the legacy
path, and produce equivalent output under the legacy settings. New features
must be opt-in. If older GhostGUI versions cannot read a new feature, preserve
the original project and offer a separate extended copy plus legacy export;
do not silently upgrade the original. Do not remove a documented backend or
format merely because it is inconvenient to expose to the agent.

## 2. Phase-By-Phase Roadmap And Deliverables

### Weekend Schedule

Times include implementation and focused tests. The final verification window
is protected; reduce feature scope when earlier work overruns.

| Phase | Timebox | Suggested window | Deliverable | Dependency |
| --- | --- | --- | --- | --- |
| W0 Preparation | 1.5 h | Saturday start | Baseline, compatibility samples, test environment, agreed gates | None |
| W1 Shared services and correctness | 4–5 h | Saturday morning/afternoon | Qt-free command workflow on a detached candidate | W0 |
| W2 External agent access | 2–3 h | Saturday afternoon/evening | Installed stdio MCP server and deterministic end-to-end test | W1 |
| W3 Live GUI attachment, stretch | 3–4 h | Sunday morning | Authenticated local bridge, Orange preview, Accept/Discard | W2 and GUI lifecycle tests |
| W4 In-app autonomy, stretch | 3–4 h | Sunday midday | Bounded opt-in loop, activity, Stop, Mock/Replay acceptance | W1; W3 snapshot/session adapter |
| W5 Test and internal release | 2–3 h | Sunday final block | Compatibility checks, user acceptance, release candidate | Every included feature's gate |

If W2 has not passed by Saturday evening, spend Sunday on W1/W2 and W5.
Start neither bridge nor loop work merely to populate files or check boxes.
If W3 or W4 cannot pass its tests before the final window, keep that feature
disabled and continue it in F6. The weekend target is 10–13 hours for W0–W2
and W5, with about 6–8 additional hours for the two stretch milestones.

### W0 — Preparation And Compatibility Baseline

- Work on an isolated feature branch and preserve unrelated changes. Do not
  implement three reviews in separate branches that all rewrite the window.
- Read current workflow code and compare it with the reviews. Inventory live
  AI callers, exports, registry data, project schema, and interpreter/SDK
  versions. The current checkout differs from the reviews' environment notes.
- Activate the existing environment or use the supported platform installer.
  Run the existing automated suite, documentation check, and architecture check.
  Record pre-existing failures separately; never weaken tests to obtain green.
- Select representative old projects: sparse targets plus qpos, dense qpos,
  a floating-base model, an imported model, and each legacy export format.
  Use existing examples or temporary copies, without editing original assets,
  example CSVs, or fixtures. Add focused synthetic regression tests as needed.
- Capture expected sample arrays and exact serialized output in the same
  runtime for unchanged paths. Include DSMS speed behavior, G1 mjlab joint
  order/quaternion layout, qpos width, save/reopen, and Undo/Redo.
- Define feature flags with existing behavior as default: bridge off, agent
  mode off, legacy interpolation/export defaults, review approval policy.

**Exit:** reproducible baseline and an automated compatibility comparison for
the services about to be extracted. No paid provider calls are required.

### W1 — Shared Services And Minimum Trustworthy Behavior

- Extract `application/keyframe_service.py`,
  `application/trajectory_export_service.py`, and workspace payload helpers
  from Qt callbacks. Reuse existing project, generation, CSV, and motion
  services. Make GUI callbacks delegate to them with parity tests.
- Build `application/agent_api/` with a `Workspace` port,
  `HeadlessWorkspace`, command catalog, runtime, typed events, bounded
  checkpoints, and an output sandbox. Reuse `AIEditSession` and `ToolRegistry`.
- Route every candidate mutation through local schema validation, provenance,
  checkpoints, revision checks, and atomic rollback. A model response never
  executes Python, shell commands, or arbitrary file operations.
- Implement the initial useful catalog: model/status inspection, project
  create/open/save, pose/Keyframe inspection, sparse authoring or the existing
  edit-spec operation, Joint Angle/IK operations, repeat, validation,
  generation, checkpoint restoration, accept/discard, and existing exports.
  Register only implemented tools. Complete the remaining catalog in F6/F8.
- Preserve the distinction between caller-authorized project destinations and
  model-generated filenames. Reject traversal, absolute output filenames, and
  symlink escapes. Never overwrite an existing project as an incidental
  consequence of `new_project`; require the explicit supported save workflow.
- Make timeline navigation read-only. Only explicit editing commands create
  Keyframes. Do not delete old qpos-only states when loading a project.
- Track derived-motion identity using document ID, model/session identity,
  content revision, generation settings, and qpos revision. Invalidate on every
  edit path, including Joint Angles, import, delete, Undo/Redo, and AI Accept.
  A late generation result must not become the current motion after an edit.
- Block stale generated playback/export with an explicit solve-or-editable
  source choice. Preserve numeric export contracts and make the selected
  source visible. Orange preview never leaks into committed exports.
- Add Qt-free structural checks and sampled collision/limit checks over the
  selected candidate path. Report sampling resolution and unsupported metrics.
  Hard-gate blocking defects for the new agent approval path; preserve existing
  manual export behavior unless the user selects stricter validation.
- Strengthen architecture checks for the new headless modules and their
  transitive import path. Keep model-specific data in the registry/model layer.

**Exit:** a headless deterministic test creates a project, authors a simple
motion, validates it, repeats it, generates samples, accepts under an explicit
test policy, saves/reopens, and exports through unchanged writers. Invalid,
protected, stale, or unsupported requests leave committed motion untouched.

### W2 — MCP Access For Claude Code

- Add the optional `agent` dependency with an exact validated MCP SDK version
  and a `ghostgui-mcp` console entry point. Keep base installation usable
  without MCP. Verify the SDK against official documentation and the runtime.
- Expose the same catalog through stdio MCP: JSON tool schemas, bounded text
  results, structured errors, terminology/robot resources, and a motion
  authoring prompt. Keep protocol stdout free of diagnostic prints.
- Implement `render_views` using MuJoCo offscreen rendering and a deterministic
  stub in unit tests. Add front/side views or a contact sheet, timestamp labels,
  image limits, and an explicit unavailable-rendering response.
- Serialize mutations. Read operations must observe stable snapshots; do not
  run shared mutable MuJoCo state concurrently just because a tool is read-only.
- Default to `--attach never` until W3 passes. Afterward support
  `never|required|auto` and report the selected workspace prominently.
  `required` must fail when no GUI is attached; `auto` must announce fallback.
- Test listing/calling tools through an MCP client, image delivery, invalid
  schemas, unsupported exports, path containment, and clean shutdown.
- Document installation and project-scoped Claude Code configuration only
  after checking the actual command help. Do not commit personal paths or keys.

**Exit:** a real MCP client drives the deterministic workflow and gets a valid
image on a supported renderer. A user with funded Claude access can then run a
simple live prompt. Record transport, rendering, and live-LLM acceptance as
three distinct results.

### W3 — Live GUI Attachment

- Add Qt-free bridge protocol/client modules and `gui/agent_bridge.py` with
  `GuiWorkspace` and a queued main-thread invoker.
- Bind only to loopback on a dynamic port, with an unguessable per-process
  token, restricted discovery-file permissions, version/session identity,
  one active client, bounded message sizes, and cleanup on exit.
- Add **Allow external agent control**, default off, plus a connected banner,
  visible tool activity, Stop/Disconnect, and Accept/Discard controls.
- Keep working-copy mutation on its owning worker. Publish immutable detached
  snapshots; Qt owns viewer, committed document, history, camera capture, and
  project transitions. Never send a mutable working document to a sampler.
- Give invocation futures timeouts/cancellation and ignore late callbacks.
  Serialize a single active run per document. Refuse model/project transitions
  while an unresolved candidate exists. Resolve dirty-project transitions
  explicitly through the existing workflow; never silently discard edits.
- Make `review` the default: tools stage the candidate; Accept validates the
  latest revision and records one Undo transition. Unattended acceptance is a
  separate explicit run policy and cannot bypass protection or stale checks.
- Test wrong token, second client, disconnect mid-tool, shutdown mid-capture,
  stale acceptance, and candidate isolation. Verify a real GUI manually.

**Exit:** external tool calls update the visible Orange preview, while the
committed document stays unchanged until Accept. One Undo reverses acceptance.

### W4 — In-App Autonomous Loop

- Add an opt-in Agent mode alongside the current compact Edit mode. Invoke
  the same catalog and `GuiWorkspace` in-process; do not require an MCP socket
  or a second set of motion handlers inside the application.
- Evolve useful legacy loop mechanics into one bounded provider-neutral loop.
  Preserve complete provider-required content blocks and tool-call IDs, combine
  each turn's tool results correctly, and support image results.
- Check actual Anthropic SDK/model support before enabling streaming, thinking,
  caching, effort controls, or pause handling. Unsupported optional features
  must not prevent basic tool use. Replay required opaque blocks unchanged;
  display only provider-supported summaries in the UI.
- Start with small configurable turn/tool/image/time budgets and repeated-error
  detection. Check cancellation before calls, between tools, and within long
  local operations. Restore the best validated checkpoint on failure/budget
  exhaustion; if none exists, restore the initial state and report failure.
- Show activity, selected scope, run status, Stop, and a result card with
  validation/metrics, warnings, and Accept/Discard. Make New request versus
  Refine preview explicit; initially retain current replacement defaults.
- Default paid execution to disabled for the stated budget. Mock/Replay runs
  demonstrate planning → tool use → observations → correction → validation.
  Enabling a funded provider requires a configured run ceiling and token caps;
  estimate the next request before dispatch and report actual usage afterward.
- Add `ghostgui agent` through headless dispatch before Qt imports. Keep
  existing `ghostgui --model ...` behavior. GUI and CLI share the same loop.
- Preserve Gemini's current Edit path. Enable Agent mode for a provider only
  after its transcript/tool semantics are supported and tested.

**Exit:** scripted GUI and headless runs complete, cancel safely, reject stale
acceptance, and stay within configured local limits. Live Anthropic acceptance
is pending until separately funded; a mock pass is never a live pass.

### W5 — Verification, Small UI Fixes, And Internal Release

- Run the complete existing suite plus new command, MCP, bridge, loop, and
  compatibility tests for enabled features. Run docs/architecture checks and
  the existing wheel/installed-package and real-render smoke where applicable.
- Reserve 30–45 minutes with a user: generate a simple motion, inspect the
  Orange preview, accept or discard, Undo/Redo, save/reopen, export, and repeat
  through an old project. Try an incompatible request and cancellation.
- Fix disabled primary-button styling, create the status bar at startup, show
  candidate scope and freshness, and hide Commit Keyframe/gizmo actions during
  whole-motion candidate review. Correct stale terminology and degree units.
- If time permits, remove repeated assistant headings, provide readable
  provider text/right-panel width, expose Export interval near its workflow,
  and improve Keyframe markers. Defer cosmetic changes before reducing tests.
- Update relevant user guide, help/tutorial copy, AI/security docs, architecture,
  changelog, and documentation index together with delivered behavior.
- Prepare an internal release candidate and concise known limitations. Keep
  original projects backed up and the previous runnable version available.
  No automatic public publishing or credential configuration is part of this
  implementation guide.

**Exit:** all enabled features pass their gates, at least one real user signs
off, and unfinished features stay disabled. A live-provider gate can remain
explicitly pending while an internal offline/transport preview is distributed;
do not label it a completed autonomous Claude release.

### Full-Scope Follow-On Schedule

Estimates below include focused tests and documentation, not waiting for user
feedback or access to additional operating systems. F6 includes unfinished
W3/W4 work; do not count that work twice.

| Phase | Estimate | Depends on | Deliverables and acceptance |
| --- | --- | --- | --- |
| F6 Complete autonomy | 0–20 h | W1/W2; W3/W4 where unfinished | Finish GUI bridge, in-app loop, headless agent CLI, remaining deterministic catalog, checkpoints, resources, progress, image tools, and vision self-critique. Wire the prompt-first start experience. Pass all offline scenarios; live Claude acceptance depends on funded access. |
| F7 Motion reliability and performance | 16–28 h | W1; snapshot/revision contract | Cancellable background Generate; explicit source/freshness; true neighboring-Keyframe Preview Path; sampled quality metrics and plots; opt-in smooth interpolation; time-resolution consistency; performance caches; separate AI and file jobs. Pass legacy output and cancellation regressions. |
| F8 Document state and consolidation | 20–36 h | W1; acceptance/history regressions | Move qpos ownership into the document; document-level Undo with shared immutable arrays; stable Keyframe identity across retiming; typed protocols/events; quaternion internal values; incremental window/viewer extraction; AI/backend cleanup; Qt adapter migration. Save/reopen and legacy output stay equivalent. |
| F9 Robotics quality and new motion features | 32–56 h | F7/F8; registry metadata | Contacts, true-priority/constraint IK, mirroring, make-cyclic/seam tools, CoM/support metrics and overlays, inverse-dynamics report, optional PD rollout, per-model limits/camera/primitive/export profiles. Check synthetic numerical cases and model-specific limitations. |
| F10 Research workflow extensions | 28–48 h | F7–F9 data/serialization contracts | Dense retargeted qpos import and reduction; batch generate/validate/export CLI; graph editor and tangent/easing tools; opt-in additive motion layers; clip/dataset index; optional native NPZ export. Existing formats and editing defaults remain supported. |
| F11 Full UI/UX redesign | 32–56 h | W3/W4 or F6; F8; F9/F10 schemas for new controls | Tokens/themes, shell/command registry, docks/HUD/state reducer, dopesheet/range selection, inspector/numeric fields, unified export/settings, Prompt Mode/result cards, shortcuts/palette, onboarding and accessibility. Validate manual and agent workflows at multiple sizes/themes. |
| F12 Public release preparation | 12–20 h | All included features accepted | Reproducible dependencies, installer extras, resource packaging, logging/diagnostics, lint/type/coverage gates, platform matrix, wheel smoke, migration docs, support/rollback instructions, internal soak. Publish only after a separate release decision. |

### Follow-On Action Checklists

**F6 — Finish the agent experience**

- Complete the enumerated catalog by capability. Review 03 lists 29 tools;
  omitting `simulate_rollout` and `plan_motion_edit` leaves 27, despite a prompt
  saying 28. Derive catalog assertions from the actual registered names.
- Keep `plan_motion_edit` optional because it introduces a nested provider call.
  Implement it only when its cost and cancellation count toward the same run;
  otherwise return an explicit unsupported capability.
- Add semantic critique criteria over front/side contact sheets and metrics,
  with a bounded correction loop. Preserve export incompatibility reporting.
- Cover squat, wave, repeated push-up, kinematic jump/cartwheel, Go2 motion,
  and invalid-format scenarios. Do not promise arbitrary physically feasible
  acrobatics because a rendered candidate looks plausible.
- Add model-neutral presets based on tested capabilities, not unverified model
  IDs from the review. Keep optional Agent SDK wrapping documented as an
  integration alternative; a second embedded harness has no initial owner.

**F7 — Trust the generated and exported path**

- Run Generate from a detached snapshot with progress, cancellation, and a
  stale-result guard. Preserve editable playback while generation runs.
- Fix Preview Path to evaluate previous Keyframe → proposed pose → next
  Keyframe at actual times, without mutating the document. Cover endpoints and
  missing-neighbor cases.
- Add velocity, acceleration, jerk, limit margin, interpolated collision,
  penetration, end-pose, root-continuity/rotation, and loop-seam metrics.
  Report units, dt, thresholds, and unavailable model metadata explicitly.
- Add smooth manifold-aware interpolation as an opt-in mode. Keep linear
  legacy defaults and exact anchors. Define derivative continuity, boundary
  behavior, quaternion sign handling, and long-rotation anchor requirements.
- Cache sorted timeline keys, rendered polylines, and derived samples by
  revision; incrementally refresh Keyframe rows; coalesce drag IK to display
  cadence. Move export preprocessing off the GUI thread and split long AI
  work from file/model jobs with detached state and lifecycle cancellation.
- Add opt-in affected-segment automatic solving only after freshness and
  invalidation tests pass. Keep manual Generate and an explicit export-source
  choice. Never silently snap old Keyframes onto a new sampling grid.

**F8 — Simplify ownership without changing files**

- Give the document typed ownership of targets, qpos anchors, duration, and
  stable motion IDs. Keep display time, camera, selection, and docks out of
  document history. Do not add history transitions on scrub or camera changes.
- Replace full deep copies with immutable/shared motion arrays or reversible
  document commands. Restore through public APIs and test Undo/Redo across AI,
  repeat, retime, import, generation invalidation, and project transitions.
- Keep Euler-based legacy serialization while adding normalized quaternion
  operations internally. Test equivalent rotations near singularities and
  round-trip behavior before proposing any persisted schema extension.
- Extract focused project, model-session, history, import, timeline, Keyframe,
  and scene presenters/controllers. Keep the main window as composition root
  and compatibility facade while migrating callers. A line-count target alone
  is not an acceptance criterion.
- Consolidate duplicate loaders, preview resets, cancellation/timeout helpers,
  and export paths. Simplify visualization lifecycle indirection only where
  tests show it adds no isolation or extension value.
- Remove unreachable planners/hidden widgets after replacement coverage and
  import tracing. Move model recipes/aliases/export ordering into the registry.
  Remove native/analytic backends only after confirming no supported workflow
  relies on them; otherwise deprecate and retain compatibility.
- Move Qt file-dialog/background adapters to `gui/` and leave Qt-free service
  contracts in `application/`. Keep any import compatibility shims narrow and
  temporary. Strengthen architecture checks as ownership becomes accurate.

**F9 — Add robotics analysis and editing capabilities**

- Populate mirror pairs/signs, support frames, velocity/actuator limits, camera
  presets, primitive aliases, and export profiles in the model layer. For
  imported models, accept profile data and report unknown capabilities rather
  than using G1 defaults. Extend profile discovery instead of requiring core
  edits for every imported robot; arbitrary Python plugin loading is unnecessary.
- Add contact labels and contact-aware generation through a tested hard
  constraint/priority solver. Benchmark a QP option before changing defaults;
  retain the existing solver for legacy projects.
- Add mirror pose/range and make-cyclic tools that transform logical targets,
  qpos, root offsets, contacts, and ownership consistently. Test mirroring
  twice, loop seams, and exact repetition invariants.
- Add stance-based foot-slide/contact metrics and quasi-static support margin.
  Show the CoM projection and support polygon; report undefined support during
  flight. Treat ZMP as a separate dynamics estimate with explicit assumptions,
  not as another name for CoM projection.
- Add inverse-dynamics torque/root-residual diagnostics using documented
  actuator transmission and contact assumptions; do not compare generalized
  torques directly with raw control ranges as if they were interchangeable.
- Add an optional PD rollout with model-specific gains and fall/divergence
  criteria. Keep it a heuristic analysis tool, separately labeled from
  kinematic validity or hardware suitability.
- Persist new contacts/solver settings only through an explicit compatible
  extension strategy. Add contact labels/metrics as sidecars or new formats.

**F10 — Complete the researcher workflow**

- Import retargeted robot qpos sequences from validated numeric NPZ/CSV first,
  with explicit robot/joint mapping, coordinates, units, and frame rate. Add
  GMR adapters after inspecting real samples. Never unpickle an arbitrary
  external file in the normal application process; isolate a trusted conversion
  path if a legacy pickle adapter is required.
- Reduce dense samples into editable Keyframes with bounded manifold, root,
  and End Effector error while preserving contact transitions and endpoints.
- Add batch generate/validate/export over the same workspace services. Keep
  proposed CLI command names provisional until implemented and help-tested.
- Build a graph editor only after tangent/interpolation contracts exist.
  Expose velocity/acceleration plots, tangents, easing, and precise retiming.
  Add optional additive layers with defined composition and flattening into
  existing exports; the comparison-table gap is included without changing the
  default single-motion document.
- Add a clip index with names/tags, model, duration, validation revision, and
  export status. Keep it outside legacy project payloads unless versioned.
- Add separately named NPZ output for joint positions/velocities and body
  poses only after checking the downstream consumer's schema. Retain G1 mjlab
  CSV and its converter exactly; do not infer native compatibility from a name.

**F11 — Deliver the complete UI proposal**

- Create semantic color, spacing, type, radius, elevation, and motion tokens;
  generate QSS/painter colors and retain the existing theme API as a shim.
  Cover enabled/disabled/focused states and reserve clear robot-state hues.
  Consolidate SVG icons with license attribution and accessible labels.
- Assemble a command registry, native menus where appropriate, project/model
  bar, tool rail, dock layout, permanent status, typed status presenter, HUD,
  3D/Simulation switch, and one editor-view-state reducer. Persist appearance
  and dock layout in QSettings, never in legacy project files.
- Build the dopesheet with summary/per-target rows, readable markers, ruler,
  zoom/pan, transport, snap, selection/ranges, drag retiming, and derived/candidate
  lanes. Preview drag conflicts and apply one validated Undo transaction on
  release; keep typed dialogs for precision.
- Bind one Qt-free selection model to timeline, inspector, assistant scope,
  copy/paste/repeat, and bulk actions. Preserve keyboard focus in text/numeric
  inputs; do not let global shortcuts intercept typing.
- Add selection-driven inspector sections, editable numeric fields, grouped
  and filtered Joint Angles, zero-centered joint sliders, target picking,
  Display/IK sections, unified settings, export-source/rate preview, and Repeat
  last export. Wheel edits require focus. Keep a local/world gizmo control
  unavailable until local-frame math is implemented and tested.
- Consolidate Prompt Mode and the agent start screen into one implementation.
  Add explicit New request/Refine preview controls, scope chips, comparison,
  warning jumps, Retry, Copy details, metrics, and a review banner. Initially
  preserve the replacement default; changing it requires matching help/tests.
- Finish camera framing, gizmo hover/delta readouts and neutral center color,
  contextual menus, Undo actions/history affordances, command palette,
  shortcuts overlay, empty/loading states, and the revised seven-step tutorial
  with an assistant entry. Explain uncommitted-pose loss before navigation;
  keep/discard/commit choices must be explicit and undoable where applicable.
- Validate light/dark themes, 1280×720 and 1400×860 windows, scaling, contrast,
  focus, keyboard access, accessible names, and minimum hit targets. Color must
  not be the only carrier of state. Extend UI terminology checks to all new
  modules and update help/tutorial/public docs with every label change.

**F12 — Prepare public distribution**

- Add secret-scrubbed rotating diagnostic logs, reproducible tested dependency
  constraints, installer choices for AI/agent extras, and actionable missing
  dependency messages. Keep the core application useful without AI extras.
- Migrate resource loading toward package data/`importlib.resources` while
  preserving model-relative mesh resolution, writable data separation, and
  installed read-only resource support. Do not rewrite robot assets or cache
  versions incidentally.
- Add Ruff/basic type checking incrementally for changed code, coverage of
  critical contracts, generated-case interpolation/retiming tests, and faster
  isolated service tests. Share expensive GUI setup only with full state reset
  between cases; speed must not create order-dependent tests.
- Preserve existing Python 3.10/3.13 CI coverage and installed-wheel checks;
  add MCP/AI-extra jobs so new features are actually exercised. Test real GL
  on Linux and the local macOS machine. Keep Windows experimental until tested.
- Distill stale diagnosis/history passages into current task-oriented docs.
  Publish migration, support, and rollback instructions. Require one internal
  workweek of use without unresolved correctness/data-loss defects before the
  public release decision; elapsed soak time is additional to effort estimates.

## 3. Dependencies And Critical Path

```text
W0 baseline
  -> W1 shared services + correctness + compatibility
     -> W2 MCP -> W3 GUI attach -> W4 in-app integration -> W5 internal gate
     -> W4 loop core and headless CLI (can be implemented before GUI attach)

unfinished W3/W4 -> F6 complete autonomy
W1 -> F7 reliability -> F8 document/history -> F9 robotics -> F10 research tools
F6 + F8 -> F11 shell/assistant; F9/F10 -> their corresponding UI controls
F6 through F11 -> F12 public gate
```

This is an execution order for one implementer; it does not assume parallel
developers or concurrent Codex sessions. Some tasks can be reordered once their
contracts exist, but simultaneous rewrites of state, history, UI, and agents
would make regressions harder to isolate.

| Must exist first | Enables | Why |
| --- | --- | --- |
| Current-code baseline and legacy comparisons | Every extraction | Prevents old review citations becoming implementation assumptions |
| Shared services and detached session | MCP, GUI attach, in-app loop, batch CLI | One mutation/export contract across interfaces |
| Read-only navigation and revision/freshness contract | Safe preview, async Generate, acceptance/export | Prevents hidden edits, stale output, and late results |
| Bounded command/runtime schemas | MCP and loop | Prevents provider output bypassing local validation |
| Immutable snapshot and Qt invocation lifecycle | GUI observer/bridge | Prevents data races and use-after-close callbacks |
| Stable document identity/history | Contacts, layers, complex retiming | Prevents provenance loss and split target/qpos history |
| Registry metadata and validated metric definitions | Mirror, contact IK, dynamics, visual critique | Avoids G1 assumptions and misleading numerical reports |
| Optional serialization/export design and migration tests | Persistent contacts, tangents, layers, metadata | Preserves existing projects and consumers |
| Funded Claude credentials and spending limit | Live external/in-app acceptance | Plus-funded coding and application inference are separate |
| Accepted internal workflows and packaging checks | Public release | Avoids releasing a prototype as a supported product |

## 4. Risk Management

| Risk or bottleneck | Trigger/impact | Mitigation and fallback |
| --- | --- | --- |
| Weekend and subscription allowance are exhausted | Broad agent/threading work consumes the final test window or Codex allowance | Six scoped prompts, local tests, resumable commits/checkpoints, and a fixed final verification block. Deliver W0–W2 plus W5 first; leave bridge/loop disabled if unfinished. Use the plan's phase boundary to resume later. |
| Claude cannot run within the stated budget | No usable Claude credentials or API funding; mock results mistaken for live success | Build MCP independently, test transport with an SDK client, and test the loop with Mock/Replay. Explicitly mark live acceptance pending. No paid defaults or hidden billing; enable funded execution only through deliberate configuration. |
| Legacy projects or exports change | Navigation cleanup, new interpolation, schema migration, quaternion conversion, backend removal | Preserve originals and all qpos states; use legacy default settings and numerical/golden comparisons. Gate new math/formats behind options; add sidecars or extended copies. Reject release on any unexplained compatibility difference. |
| GUI/worker/GL lifecycle fails | Frozen UI, concurrent document access, stale acceptance, platform renderer failures | Immutable snapshots, serialized mutation, GUI-thread marshalling, cancellation/timeouts, document identity checks, wrong-token/disconnect/close tests. Keep headless MCP usable; report rendering unavailable rather than fabricating an image. |
| Autonomous output looks convincing but is wrong | Keyframe-only validation misses intermediate defects, missing limits masquerade as passes, apparent balance mistaken for physics | Validate the exact selected/exported path with declared dt, numerical synthetic cases, model-specific thresholds, and user review. Report unavailable metrics/concessions. Keep kinematic, quasi-static, and dynamic evidence distinct; require a separate public quality gate. |

## 5. Success Metrics And Acceptance Criteria

Targets are proposed release criteria. Measure them on the same machine,
runtime, model, and scenario established in W0. They are not claims about the
current application.

| Area | KPI / criterion | Required by |
| --- | --- | --- |
| Backward compatibility | 100% of selected old projects reopen and save through the legacy path; all existing model/export contract tests pass | Every release |
| Export parity | Exact bytes for unchanged deterministic writers in the pinned runtime; same arrays/timestamps/order within existing tolerances elsewhere; DSMS speed and mjlab mapping unchanged | W1/W5 and every math/format change |
| Navigation integrity | At least 100 varied scrubs leave Keyframes, qpos anchors, document revision, dirty state, and document history unchanged | W1 |
| Freshness | Every tested mutation invalidates derived output; late generation cannot replace newer document/model state | W1/F7 |
| Transactional editing | Discard/failure leaves committed motion unchanged; acceptance is one Undo entry; stale/protected/invalid acceptance always rejects | W1/W3/W4 |
| Agent command contracts | Every registered tool accepts valid examples, rejects invalid schemas, returns bounded structured results, and obeys path/revision rules | W2/F6 |
| External access | Real MCP client completes create → author → validate → generate → accept/save/export and retrieves a real render where GL is available | W2 |
| In-app loop | Scripted multi-turn scenario observes tool results, corrects a defect, validates, and resolves through review; budget/error/cancel paths tested | W4/F6 |
| Live autonomy | A user completes three simple scenarios in external Claude and in-app Claude, without manual Keyframe editing, with all declared hard gates passing | F6; pending funded provider access |
| Scenario coverage | Seven review scenarios have deterministic offline contracts; negative Go2→mjlab writes no output. Difficult motion scenarios report concessions rather than false feasibility | F6/F9 |
| Stop/responsiveness | Stop acknowledges within 1 s; local worker cancellation reaches a checkpoint within 2 s in the reference scenario; GUI timer gaps during Generate stay below 200 ms at p95 | W3/W4/F7; log device-specific results |
| Motion quality | Check all sampled states at the declared dt; zero blocking collision/limit violations for accepted agent output; limits/contact/velocity tests use model-specific thresholds or report unavailable | W1/F7/F9 |
| Example numerical targets | Squat duration 3 ± 0.05 s, end-pose error < 1 cm / 0.02 rad; stance foot slide < 1 cm as a goal; cyclic seam translation < 1 cm; no unsupported dynamics claim | F9 evaluation; calibrate per model |
| History/performance | No motion-array copy on navigation; repeated unchanged playback reuses cached samples; 100 history entries share unchanged arrays | F7/F8 |
| UI usability | Users finish prompt → inspect → accept/discard → export without unexplained disabled actions; no clipped primary controls at target sizes; keyboard/manual workflow still works | W5/F11 |
| Spend | Zero new paid API calls/purchases under the current budget; local runs stop at configured limits; external-client model spending is explicitly outside MCP's enforcement | W0 onward |
| Documentation/package | Docs, architecture, suite, extra-dependency jobs, wheel resource inspection, and installed smoke pass for included features | W5/F12 |
| Internal acceptance | At least one available user approves the weekend workflow; all core workflows receive internal-team sign-off before public release | W5/F12 |

For live autonomy, count every attempted scenario and report successes/attempts,
retries, tool calls, duration, tokens, and cost when available. Mock/Replay tests
establish mechanics and regressions, not creative model success rates.
Sampled checks do not establish continuous collision freedom between samples.
MCP can cap tool/image work but cannot cap the external client's model bill.

## 6. Six Sequential Codex Prompts

Use one prompt per checkpoint, in order. A new session can resume from this
guide, the current diff, and the phase's tests. The repository instruction to
preserve unrelated changes applies to every prompt. Checkpoint a passing phase
in version control when ready; do not automatically publish or merge.

### Prompt 1 — Establish The Baseline

```text
Read AGENTS.md, docs/architecture.md, docs/data_formats.md, and
docs/implementation_plan.md. Implement W0 only. Inspect current code rather
than assuming the reviews' line references or SDK claims are current.
Run the baseline using the existing environment and supported test runner.
Add focused compatibility tests for the services W1 will extract, using
temporary copies/synthetic inputs without modifying existing assets or fixtures.
Preserve every existing project/export/model behavior and unrelated change.
Report baseline failures, actual dependency versions, and the W1 entry gate.
Do not start broad refactoring, install new paid services, or make provider calls.
```

### Prompt 2 — Build The Shared Workflow

```text
Implement W1 in docs/implementation_plan.md. Extract Keyframe/export/workspace
services and delegate existing GUI callers to them. Build the Qt-free workspace,
catalog/runtime, candidate checkpoint/revision/provenance gates, and output sandbox.
Implement the initial end-to-end tool subset; expose no placeholder tools.
Fix navigation-created qpos states and stale derived motion, retaining all old
states and legacy serialization/export defaults. Add regression/parity tests,
including invalid/protected/stale requests and the exact chosen export source.
Verify headless imports do not transitively import Qt or GUI modules.
Run focused tests plus the docs/architecture checks, then the full phase gate.
Report delivered tool names and deferred F6/F9 items. Stop at the W1 boundary.
```

### Prompt 3 — Expose MCP

```text
Implement W2 in docs/implementation_plan.md using the passing W1 catalog.
Verify the current official MCP SDK API and pin the tested optional dependency.
Add the ghostgui-mcp stdio entry point, resources/prompt, structured tool errors,
bounded image output, MuJoCo renderer and deterministic test renderer.
Default to headless attachment until W3 exists. Do not duplicate tool handlers.
Test through a real MCP client, including errors, shutdown and image delivery;
run real rendering where supported and report unavailable GL accurately.
Document only implemented/help-tested commands. No funded model is required
for transport tests. Report live Claude acceptance separately if unavailable.
Run the phase gate and stop before GUI bridge implementation.
```

### Prompt 4 — Attach The GUI

```text
Implement W3 in docs/implementation_plan.md, provided W2 is passing and the
final verification window remains protected. Add the opt-in authenticated
loopback bridge and GuiWorkspace adapter, immutable candidate publications,
queued Qt invocations with cancellation/timeouts, visible activity and disconnect.
Use the same catalog and existing atomic Accept/history path. Keep review as
default; explicit unattended policy cannot bypass validation or protection.
Test wrong token, competing clients, stale acceptance, disconnect, shutdown,
candidate isolation, dirty-project transitions and one-entry Undo. Keep manual
editing and every supported model/export/project flow intact. Run a real GUI
inspection as well as offscreen tests. If the gate cannot pass, keep the feature
disabled and document the remaining F6 work. Do not start the redesign.
```

### Prompt 5 — Add The In-App Loop

```text
Implement W4 in docs/implementation_plan.md only after shared workspace gates
pass. Add the opt-in bounded provider-neutral loop and headless CLI dispatch;
reuse the catalog in-process and the existing compact Edit mode as a fallback.
Verify the installed Anthropic SDK and current official provider contracts before
using model IDs, raw block replay, tool results, streaming, caching or beta flags.
Use Mock/Replay for a multi-turn inspect/correct/validate scenario, cancellation,
repeated failures, best-checkpoint restoration, budgets and provider capabilities.
Wire activity, scope, Stop and result review without a broad shell rewrite.
Paid execution remains disabled under the stated budget; do not reuse Claude
subscription tokens as API credentials. Report offline and live acceptance
separately. Preserve Gemini Edit behavior. Run the phase gate; stop at W4.
```

### Prompt 6 — Verify And Prepare The Internal Release

```text
Implement W5 in docs/implementation_plan.md. Start no new architectural work.
Run the full suite, docs/architecture checks, extra-dependency tests and applicable
wheel/real-render smoke. Recheck old project load/save/export and model contracts.
Apply only the listed small UI fixes while preserving test time. Update help,
tutorial, public docs and changelog for delivered behavior. Prepare the manual
user acceptance steps and an internal release candidate with rollback guidance.
Keep incomplete bridge/agent features disabled. Distinguish implemented code,
passing mocks, real transport/rendering and live provider acceptance. Report
remaining F6–F12 scope with realistic estimates. Do not publish publicly.
```

For later sessions, use the same pattern: implement one F-phase or one coherent
subphase, cite its dependency/exit gate, preserve compatibility, and report
evidence. Do not paste all three review implementation prompts into one run.

## 7. Recommendation Coverage

This is a scope map, not a claim of completion. Optional items remain optional
but have an explicit destination and decision boundary.

### Software Critique: Fifteen Prioritized Recommendations

| Review 02 item | Scheduled work |
| --- | --- |
| 1 Navigation-created qpos states | W1 fix; F8 optional backed-up analysis, never automatic deletion |
| 2 Background Generate | Freshness in W1; progress/cancel/worker isolation in F7 |
| 3 Smooth interpolation and quality report | F7, opt-in legacy-compatible math |
| 4 Physics feasibility | F9 inverse dynamics and optional rollout |
| 5 Contact tracks | F9 with constraint and serialization gates |
| 6 AI pruning | F6 replacement path; F8 dependency-aware consolidation |
| 7 G1 leakage/dead backends | F8/F9 registry ownership; removal only after compatibility checks |
| 8 Document history | F8 |
| 9 Single motion document/quaternions | F8 internal ownership/math; optional persistence requires compatibility gate |
| 10 Mirror | F9 |
| 11 Headless CLI | W4/F6 agent CLI; F10 batch commands; F8 Qt adapter migration |
| 12 Real Preview Path | F7 |
| 13 Retargeted import | F10 |
| 14 Hot-path performance | F7 |
| 15 Tooling/logging/packaging | Baseline W0; F12 public gate |

Review 02's additional feature list is covered too: loop/cycle tools and
CoM/support/ZMP analysis in F9; graph editor, optional layers, dataset indexing,
and new NPZ export in F10; true-priority IK in F9. Automatic solving is in F7;
auto-key remains a later opt-in mode after F8, with explicit Commit Keyframe
preserved. Dynamic import profiles, typed interfaces, deduplicated helpers,
test isolation/coverage, and documentation cleanup are assigned to F8–F12.

### UI/UX And Agent Reviews

| Review workstream | Scheduled work |
| --- | --- |
| 01 phase 0: tokens/theme/components/icons | Disabled-state fix W5; full design system F11 |
| 01 phase 1: shell/controllers/HUD/freshness | Freshness W1; extraction F8; full shell F11 |
| 01 phase 2: dopesheet/transport/selection | F11 after shared selection/edit contracts |
| 01 phase 3: inspector/numerics/export/settings | F11; underlying export source W1 |
| 01 phase 4: assistant/Prompt Mode | W3/W4/F6 functional flow; F11 consolidated interface |
| 01 phase 5: polish/onboarding/accessibility | F11, including shortcut focus, colors, camera and tutorial |
| 01 quick wins 1–5 | W1 freshness; W5 disabled styles, candidate actions, terminology and permanent status |
| 01 quick wins 6–9 | W5 if time; remaining export placement, markers/zoom, headings and panel width in F11 |
| 03 phase 1: command layer | W1; expanded catalog/metrics F6/F9 |
| 03 phase 2: MCP | W2 |
| 03 phase 3: GUI attach | W3 or F6 if deferred |
| 03 phase 4: in-app loop/CLI | W4 or F6 if deferred |
| 03 phase 5: vision/metrics | Render subset W2; bounded critique F6; full metrics/rollout F7/F9 |
| 03 phase 6: prompt-first UX | Functional entry F6; shared Prompt Mode/empty state F11 |
| 03 client setup/evaluation/security | W2–W5 docs/contracts; F6 complete scenarios; F12 release validation |
| 03 optional Agent SDK wrapper/nested planner | F6 explicit optional integration, with no duplicate default harness or unbudgeted nested calls |

## 8. Verification And Release Procedure

Use the supported installer from [Installation](install.md) if the environment
is not ready. Activate `.venv` as documented there. Existing supported checks:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_docs.py
python3 scripts/check_architecture.py
```

For local GUI tests, the existing isolated runner creates temporary application
and XDG directories and defaults to offscreen Qt:

```bash
python3 scripts/run_test_suite.py
```

Use the isolated runner for the full gate when protecting personal projects
and settings. Offscreen Qt alone does not prove OpenGL rendering works. Retain
the existing real-render CI smoke and perform local GUI inspection. Baseline
check results from a review are historical; rerun them for each implementation.

Do not document proposed `ghostgui-mcp` or `ghostgui agent` invocations as
available commands until their entry points exist and help/smoke tests pass.
Install/test the optional SDK extras in dedicated CI jobs so missing optional
dependencies do not make all new tests skip silently.

The user acceptance script is: open a copied old project; inspect/export it;
create a simple candidate; inspect its full Orange preview; cancel/discard one
attempt; accept another; Undo/Redo; save/reopen; export each applicable format;
try unsupported export and stale acceptance; close during a running request.
Compare source identity, sample rate, output arrays, warnings and project state.

An internal release requires compatibility and transactional gates, enabled
feature tests, user sign-off, and rollback instructions. Public release also
requires completed included backlog, funded live-provider acceptance for any
advertised live autonomy, platform/package verification, and the internal soak.

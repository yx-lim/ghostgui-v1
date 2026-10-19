# 03 — Making GhostGUI Fully Drivable By Claude

Scope: design for an autonomous Claude agent that takes one text prompt (for
example *"design a 6-second G1 cartwheel-like motion that ends standing, loop it
twice, export mjlab CSV"*) and does everything: loads the model, creates or
opens the project, authors Keyframes, uses IK, generates, inspects renders,
validates, iterates, and exports. The user does not touch the tool.

Grounded in branch `report-plot` @ `e81734d`. All `file:line` citations refer
to that tree. Nothing tracked was modified; this file is the only output.

API facts used here were current as of 2026-10. Items marked **(verify)** depend
on SDK versions that are not installed in this checkout (`anthropic`, `mcp`,
`claude-agent-sdk` are absent locally), so check them against the pinned
versions before implementing.

---

## 0. Executive Summary

- GhostGUI already has most of the hard parts. It has a Qt-free detached
  working copy with checkpoints, provenance, and an atomic Accept
  (`AIEditSession`). It has 12 deterministic local motion operations (IK,
  Joint Angles, qpos manifold interpolation, primitives, repeat). It has strict
  JSON-schema tool validation (`ToolRegistry`), a provider-neutral LLM protocol
  with Mock and Replay providers, structural/kinematic validation, and a
  Qt-free frame-sampling contract. A bounded multi-turn tool loop
  (`GhostGUIAgent`) also exists, but only as legacy code.
- Five things block full autonomy:
  1. The live path is a **single-shot** structured-output planner. The model
     never sees the results of its own actions.
  2. **App-level actions live in Qt callbacks and viewer state.** This covers
     model load, project create/open/save, Commit Keyframe, Generate, and the
     export snapshot.
  3. **Rendering is only possible on the Qt GUI thread** (QOpenGL).
  4. The Anthropic adapter is **lossy for multi-turn tool use**. It drops
     thinking blocks and does no caching, effort control, or streaming.
  5. Accept is a **human-only gate**.
- Recommendation: build **one headless command layer** (`application/agent_api/`)
  as the single source of truth: about 29 typed tools over a `Workspace` port.
  Three front-ends use it:
  - **(B) an MCP server** `ghostgui-mcp` (stdio). It runs headless, or attaches
    to a running GUI over an authenticated loopback bridge. This makes GhostGUI
    drivable from Claude Code / Claude Desktop with no in-app LLM code.
  - **(A) an in-app multi-turn agent loop** that calls the same catalog
    in-process. It runs Claude Opus 5.5 by default with adaptive thinking,
    prompt caching, and vision self-critique.
  - **(C) a CLI** `ghostgui agent "prompt"` (headless workspace plus loop A).
  The GUI is an *observer*. The agent mutates the detached working copy on a
  worker thread and publishes immutable snapshots, which the Qt thread shows
  live as the Orange preview.
- Six phases. MCP (Phase 2) delivers value early because Claude Code supplies
  the agent loop. The in-app loop (Phase 4) and vision/metrics self-critique
  (Phase 5) follow. Copy-paste implementation prompts are in §7.

---

## 1. Current State Assessment

### 1.1 What the Motion Assistant does today

| Capability | Status | Where |
| --- | --- | --- |
| Natural-language create/edit of the **current document's motion** | Yes | `gui/ai_assistant_controller.py:171` `start_edit` |
| One structured-output provider request; ≤1 output-limit retry; ≤1 parse repair | Yes | `application/ai/trajectory_planner.py:113-261` |
| Compact spec of ≤16 operations across 12 operation types | Yes | `application/ai/trajectory_edit_spec.py:14-41` |
| Local deterministic execution with whole-spec rollback | Yes | `application/ai/trajectory_executor.py:68-132` |
| Structural/kinematic validation (FK consistency, limits, collisions as warnings) | Yes | `application/ai/motion_services.py:329` |
| Automatic ≤8 timestamped renders as *input* context | Yes (GUI thread only) | `application/ai/frame_capture.py:197-233`, `gui/ai_frame_capture.py:30-58` |
| Orange preview of the detached candidate; Accept/Discard/Refine preview | Yes | `gui/ai_assistant_controller.py:887-1100` |
| Model-owned primitive (`burpee` on G1) | Yes | `application/ai/motion_primitives.py:27`, `core/models/registry.py` `motion_primitives=("burpee",)` |
| Exact repeat/loop through the Motion Clip planner | Yes | `application/ai/trajectory_operations.py:352`, `application/motion_clipboard.py:185` |
| Multi-turn tool loop (legacy, developer-only) | Exists, not wired to the UI | `application/ai/agent.py:116-257`, `application/ai/semantic_tools.py:38` |
| Load model / new or open project / save / Commit Keyframe / Generate / export | **No** | Qt callbacks in `gui/main_window.py` |
| See its own result, critique, iterate | **No** | Single-shot topology |

### 1.2 Architecture today

```text
 Qt GUI thread                                         Worker thread (SerializedBackgroundJobs)
 ─────────────────────────────────────────────         ─────────────────────────────────────────────
 AIAssistantPanel ──submit_requested──▶ AIAssistantController._start_request (ai_assistant_controller.py:621)
                                          │ checkpoint session, build MotionAssistantContextBuilder context
                                          │ capture ≤8 frames via RobotViewerFrameRenderer (QOpenGL, GUI thread)
                                          └─ background_jobs.submit_cancellable(work) ───────▶ asyncio.run(
                                                                                                 CompactMotionWorkflow.run
                                                                                                   TrajectoryPlanner.plan ──▶ LLMProvider.generate
                                                                                                     (1 request, response_schema=TrajectoryEditSpec)
                                                                                                   TrajectorySpecExecutor.execute
                                                                                                     handlers(op) → AIEditSession.apply_ai(ReplaceMotionState…)
                                                                                                     GhostGUIMotionService.validate_motion
                                                                                               )
 _request_succeeded ◀──── queued signal ───────────────────────────────────────────────────── result
   Orange preview sampler ← session.working_document (detached ProjectDocument + RobotStateTimeline)
   Accept → AIEditSession.accept(committed_controller) → one history entry
```

### 1.3 Reusable assets (keep, do not rewrite)

| Asset | Why it matters for an agent | Citation |
| --- | --- | --- |
| `AIEditSession`: detached working copy, `checkpoint()` / `restore_checkpoint()`, provenance gate, validated-revision tracking, stale-commit guard, atomic `accept()` | This is already the transactional sandbox an autonomous agent needs. Every tool mutation goes here. | `application/ai/edit_session.py:78-367` (accept stale check `:294-297`, `can_accept` `:133-139`) |
| 12 operation handlers, each callable independently as `(TrajectoryOperation, TrajectoryExecutionContext) -> Mapping` | Each can be exposed as a fine-grained tool without re-implementing IK or interpolation | `application/ai/trajectory_operations.py:45-121` |
| `GhostGUIMotionService` (collision-aware IK, Joint Angles, validation) | Qt-free; built from an adapter alone | `application/ai/motion_services.py:95-113` |
| `ToolRegistry` strict allowlist + JSON-schema validation (closed objects, finite numbers, bounds) | Directly usable for both MCP and in-app tool dispatch | `application/ai/tool_registry.py:59-119` |
| `GhostGUIAgent` loop: turn/tool budgets, duplicate-ID guard, abort when the same failing call repeats, JSON-safe result bounds | Pattern to evolve, not throw away | `application/ai/agent.py:61-96`, `:200-246` |
| `LLMProvider` protocol, `MockProvider` (scripted steps), `ReplayProvider` (fingerprinted, fails closed) | Network-free CI for agent loops | `application/ai/providers/base.py:24-41`, `providers/mock.py:43`, `providers/replay.py:187-228` |
| `MotionFrameRenderer` protocol and `FrameSampler` | A headless renderer only has to implement one method | `application/ai/frame_capture.py:88-129` |
| `MotionAssistantContextBuilder` (robot capabilities, qpos layout, sampled states) | Becomes `get_robot_summary` / `get_pose` | `application/ai/context.py:251-450` |
| `EditorController` + `EditorEventBus` typed events | GUI already reacts to events, not widgets | `application/editor_controller.py:20-103` |
| `motion_clipboard`, `timeline_editing` planners | Repeat, loop, move, scale, insert time | `application/motion_clipboard.py:40-185`, `application/timeline_editing.py:68-492` |
| Export: `write_trajectory_csv`, `write_qpos_csv`, `export_mjlab_trajectory`, `export_dsms_trajectory`, `resample_trajectory_export` | Qt-free export contracts | `application/csv_io.py:110-136`, `application/trajectory_export_formats.py:99,259,399` |
| `GhostGUIProject.create/open/save_bundle`, `safe_project_path` | Qt-free project persistence and path safety | `application/project_manager.py:151,756,792,1020` |
| Headless adapter construction `MuJoCoRobotAdapter("g1")` | Model load needs no Qt (the GUI just wraps it in a `QThread`) | `gui/model_loading.py:20-23`, used headless in `tests/test_ai_motion_primitives.py:35-60` |
| Diagnostics with secret scrubbing; progress events | Transcript and audit for agent runs | `application/ai/diagnostics.py`, `application/ai/progress.py` |

### 1.4 What blocks full autonomy

| # | Blocker | Evidence |
| --- | --- | --- |
| B1 | **Single-shot topology.** One planning request, then local execution. The model never observes validation results, IK errors, or renders of its own output. "An IK or execution failure does not start an autonomous repair loop." | `trajectory_planner.py:113-261`; `docs/ai_assistant.md:104-110`; `docs/ai_security.md:77-84` ("receives no executable tool declarations"); `docs/architecture.md:249-255` |
| B2 | **Scope is the current document only.** No tool exists for model load, projects, Commit Keyframe, Generate, or export. These workflows live in Qt slots. | Commit Keyframe `gui/main_window.py:3560-3586` and `:3621-3656`; Generate `:3974-3999`; save `:1696-1739`; workspace payload read from widgets `:1794-1832` |
| B3 | **Motion state partly owned by the viewer.** The qpos timeline, committed state, and generated `robot_trajectory` live on `RobotViewer3D`. The export snapshot is built inside the viewer. | `gui/robot_viewer_3d.py:160` (`state_timeline`), `:2346-2391` (`_trajectory_export_snapshot`) |
| B4 | **Rendering is GUI-thread-only** (`QOpenGLWidget.grabFramebuffer`). Frames are captured *before* the worker starts, so a loop cannot request new views mid-run. | `gui/ai_frame_capture.py:32-33`; capture in `_start_request` `gui/ai_assistant_controller.py:621-694` |
| B5 | **The Anthropic adapter is lossy for multi-turn tool use.** `thinking` / `redacted_thinking` blocks are discarded on parse. `ProviderMessage` cannot carry opaque provider blocks. There is no `thinking` / `output_config.effort` / `cache_control` / streaming. `Usage` has no cache-token fields. `pause_turn` raises. On Opus 5.5 / Sonnet 5.5, thinking is always on and its blocks must be echoed back unchanged within a tool loop (append-only history), so this adapter cannot drive a modern multi-turn loop as-is. | `application/ai/providers/anthropic.py:116` (non-streaming `messages.create`), `:146-186` (request), `:266-341` (parse keeps only `text` / `tool_use`), `:314-321` (`pause_turn`); `application/ai/schemas.py:129-163`, `:190-197` |
| B6 | **Limits are sized for single-shot.** The output cap is 16,384 tokens. Requests time out at 180 s and carry at most 8 images. The legacy loop stops at 17 turns and 16 tools with a 60 s per-turn timeout. Adaptive thinking at high effort needs more output headroom per turn. | `application/ai/limits.py:3-19`; `application/ai/agent.py:61-73` |
| B7 | **Accept is human-only by design.** Mutating GUI controls are frozen while a session is unresolved. | `edit_session.py:133-139`; `gui/main_window.py:1323-1362`; `docs/ai_security.md:6`; `docs/ai_assistant.md:150-155` |
| B8 | **The background job runner is QtCore-based** (`QObject` signals), and `asyncio.run` is called once per job. Fine for one request; a persistent loop needs a non-Qt runner plus an event channel. | `application/background_jobs.py:11,103`; `gui/ai_assistant_controller.py:695-715` |
| B9 | **Stale model list.** The Anthropic registration offers `claude-sonnet-5`, `claude-sonnet-4-6`, and `claude-haiku-4-5-20251001`, with no Opus 5.5 / Sonnet 5.5. The default provider is Gemini. | `application/ai/provider_registry.py:131-158`; `providers/anthropic.py:40` |
| B10 | **Model-specific heuristics in the AI layer.** A burpee regex override sits in the planner, which conflicts with AGENTS.md ("Keep model-specific behavior in `core/models/registry.py`"). mjlab export is G1-29DoF-only, so the agent must check compatibility first. | `trajectory_planner.py:337-398`; `trajectory_export_formats.py:352-370` |
| B11 | **No motion-quality metrics beyond structural validity.** Useful checks (root continuity, Joint Angle speed, support proximity, anchor collisions) exist only as private burpee gates. | `motion_primitives.py:162-219`; `motion_services.py:329-490` |

---

## 2. Target Architecture

### 2.1 Options

| Criterion | (A) Extend in-app assistant into a multi-turn loop | (B) GhostGUI as MCP server (headless + GUI attach) | (C) Headless CLI via Claude Agent SDK |
| --- | --- | --- | --- |
| Drives *whole app* | Only if app-level tools exist | Only if app-level tools exist | Only if app-level tools exist |
| User can watch live | Yes, natively | Yes with GUI attach; no when headless | No (artifacts only) |
| Works without a display | No | Yes | Yes |
| LLM code we maintain | Loop, provider adapter, prompts, caching | **None**: Claude Code / Desktop is the harness | Thin wrapper; Agent SDK is the harness |
| Provider neutrality (Gemini support exists today) | Keeps it | MCP is model-agnostic | Claude only |
| Security surface | Smallest (in-process allowlist) | Local stdio, plus loopback bridge if attached | Agent SDK ships Bash/Write/Edit built-ins that must be disabled (`allowed_tools`) **(verify)** |
| Time to first value | Slow (UI, threading, adapter) | **Fast** (headless layer + ~300 lines of server) | Medium |
| Testability without network | Mock/Replay providers | MCP in-memory client; no LLM needed | Needs SDK mocking |

All three share one hard prerequisite: a **Qt-free command layer with app-level
tools**. Without it, each front-end would re-implement Commit Keyframe,
generation, and export, and drift.

### 2.2 Recommendation: one command layer, three thin front-ends

```text
                                ┌─────────────────────────────────────────────────────────────┐
                                │ application/agent_api/  (Qt-free; single source of truth)     │
                                │                                                               │
  Claude Code / Desktop ─stdio─▶│ mcp_server.py ─┐                                              │
   (any MCP client)             │                │   CommandCatalog (≈29 ToolSpecs, JSON schema)│
                                │ cli.py ────────┼──▶ ToolRuntime: validate (ToolRegistry) →    │
  `ghostgui agent "…"`  ───────▶│  + loop (A)    │    budget → path sandbox → checkpoint →       │
                                │                │    handler(workspace, args) → compact result │
  In-app Agent mode ───────────▶│ (in-process)   │    → AgentEvent(s)                            │
                                │                └──▶ Workspace (Protocol)                      │
                                │                      ├─ HeadlessWorkspace (core + application)│
                                │                      └─ GuiWorkspace  (gui/agent_bridge.py)   │
                                └─────────────────────────────────────────────────────────────┘
                                          │ reuses
     AIEditSession · trajectory_operations · GhostGUIMotionService · motion_clipboard · timeline_editing
     trajectory_generation · BackendInterface · csv_io · trajectory_export_formats · GhostGUIProject
     core.ik · core.models (registry, adapter, RobotStateTimeline) · core.trajectory
```

Rationale:

1. **One catalog, many harnesses.** MCP is the lowest-cost way to get a
   world-class agent loop. Claude Code already supplies planning, context
   management, approvals, and vision over tool-result images. The in-app loop
   (A) is still worth building for users without Claude Code and for a
   productised "prompt-only" UX. Both call the *same* `CommandCatalog`, so
   behaviour, validation, and safety cannot diverge.
2. **AIEditSession stays the transactional boundary.** Every mutating tool acts
   on the detached working copy. Commit to the user's document only happens
   through `accept_candidate`, which is subject to an approval policy (§4.6).
   Undo semantics, provenance, and the stale-commit guard
   (`edit_session.py:294-297`) carry over unchanged.
3. **The GUI is an observer.** The agent never touches widgets. Working-copy
   mutations run on the agent thread. After each mutating tool, an *immutable
   snapshot* (`capture_motion_state` → `detached_document`) is posted to the Qt
   thread, which swaps the Orange preview sampler, so the user watches live.
   App-level actions such as model switch, project open/save, and GUI-mode
   renders hop to the Qt main thread through an invoker.
4. **Headless parity.** `HeadlessWorkspace` owns exactly what the main window
   owns today: adapter, committed `ProjectDocument` + `RobotStateTimeline`,
   `BackendInterface`, `GhostGUIProject`, generated states, and a MuJoCo
   offscreen renderer. CI, MCP, and CLI need no display.
5. **AGENTS.md compliance.** App logic moves *out of* Qt callbacks into
   application services (`keyframe_service`, `trajectory_export_service`,
   `workspace_state`), which `gui/main_window.py` then also calls. New
   model-specific data (mirror pairs, camera presets, velocity limits, support
   frames, primitive aliases) goes into `RobotModelInfo` in
   `core/models/registry.py`.

Claude Agent SDK (option C) is not recommended as the primary path. It would
re-introduce general-purpose Bash/Write tools that GhostGUI's security model
explicitly excludes (`docs/ai_security.md:70-71`). Users who want a Claude Code
harness can already get it via MCP. `ghostgui agent` should run loop (A)
headless. An optional Agent-SDK recipe appears in §7.8.

### 2.3 Workspace contract

```python
# application/agent_api/workspace.py  (Qt-free)
class Workspace(Protocol):
    # identity / state
    @property
    def adapter(self) -> MuJoCoRobotAdapter | None: ...
    @property
    def committed(self) -> ProjectDocument | None: ...           # qpos_timeline attached
    @property
    def project(self) -> GhostGUIProject | None: ...
    @property
    def generated(self) -> GeneratedMotion | None: ...            # times, qposes, dt, source_revision
    def candidate(self, *, create: bool = False) -> AIEditSession | None: ...
    def metadata_service(self) -> MotionMetadataService: ...
    def motion_service(self) -> GhostGUIMotionService: ...
    def renderer(self) -> MotionFrameRenderer | None: ...          # headless: MuJoCo offscreen; GUI: Qt
    # app-level actions (GUI implementation marshals to the Qt thread)
    def load_model(self, model_key: str) -> None: ...
    def new_project(self, name: str, model_key: str, duration_s: float) -> GhostGUIProject: ...
    def open_project(self, name: str) -> GhostGUIProject: ...
    def save_project(self) -> None: ...
    def accept_candidate(self, description: str) -> AcceptOutcome: ...
    def discard_candidate(self) -> None: ...
    def set_generated(self, motion: GeneratedMotion | None) -> None: ...
    # observation
    def publish_candidate(self, snapshot: MotionStateSnapshot) -> None: ...  # GUI: Orange preview
    def emit(self, event: AgentEvent) -> None: ...
```

`HeadlessWorkspace` implements this directly. `GuiWorkspace` (Phase 3) adapts
`RobotGuiMainWindow` + `AIAssistantController`, after their logic has been
extracted into the application services listed in Phase 1.

### 2.4 Threading and live mirroring

```text
Agent thread (dedicated asyncio loop; one per run)        Qt main thread
───────────────────────────────────────────────           ───────────────────────────────────────
provider.generate()  (network, cancellable)
tool: solve_ik / author_keyframes / repeat …
   └─ mutate session.working_document  ── no Qt ──
   └─ snapshot = capture_motion_state(working)  ───post──▶ controller.present_agent_candidate(snapshot)
                                                            sampler ← detached_document(snapshot)
                                                            viewer.refresh(); panel.activity(row)
tool: render_views (GUI mode)  ──invoke(Future, timeout)──▶ RobotViewerFrameRenderer.render_frame(…)
   ◀────────────────────────── PNG bytes ──────────────────
tool: load_model / open_project / accept ──invoke──────────▶ main_window services (+ history entry)
cancellation: token checked between turns/tools; provider poll (anthropic.py:117-123 pattern)
```

Rules:

- **Working-copy mutation is agent-thread only.** Committed state, the viewer,
  the project, and history are Qt-thread only.
- **Snapshots are immutable value objects**, so there are no data races and no
  sampler detach dance. Today's code detaches the sampler during a worker run;
  this replaces that.
- Main-thread invocation uses a `QObject` with a queued signal carrying
  `(callable, concurrent.futures.Future)`. The agent waits on the future with a
  timeout and the cancellation token.
- Headless: all calls are direct. A single `asyncio.Lock` serializes tool
  execution. Parallel tool calls from Claude are executed **sequentially in the
  order received** for mutating tools. Read-only tools may run concurrently in
  a thread pool.
- The user stays in control. Mutating controls stay disabled during a run
  (`main_window.py:1323`), **Stop** cancels, and the result card offers
  Accept/Discard when the policy requires review.

### 2.5 GUI live-attach (MCP → running GUI)

- The GUI opens a `QTcpServer` on `127.0.0.1:0` only when the user enables
  **Settings → Allow external agent control** (off by default). It writes
  `{"port", "token", "pid", "version"}` to
  `ghostgui_user_data_dir()/agent-bridge.json` with mode `0600`, and deletes it
  on shutdown.
- Protocol: newline-delimited JSON-RPC 2.0. Method `tools/call` carries
  `{name, arguments}` for the **same catalog**. `events/subscribe` streams
  `AgentEvent`s. The first message must be `auth` with the token.
- `ghostgui-mcp --attach auto` reads the bridge file. If a GUI is reachable it
  proxies every tool call to the GUI process, where the catalog executes against
  `GuiWorkspace`. Otherwise it falls back to a private `HeadlessWorkspace`.
  `--attach never|required` force either mode.
- One client at a time. The panel shows **"External agent connected"** with a
  Disconnect button, and every tool call appears as an activity row.

---

## 3. Tool Catalogue (29 tools)

### 3.1 Conventions (apply to every tool)

- **Schema dialect.** The subset that `ToolRegistry` already enforces
  (`tool_registry.py:23-37`): closed objects (`additionalProperties: false`),
  `required`, `enum`, numeric/array/string bounds, and no `$ref` / `oneOf` /
  union types. Shorthands below expand at build time in
  `application/agent_api/schemas.py`:
  - `T` = `{"type":"number","minimum":0,"maximum":120}` (seconds; 120 s is the
    timeline cap, `limits.py:19`)
  - `V3` = `{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}`
  - `Name` = `{"type":"string","minLength":1,"maxLength":64}`
  - `Names(n)` = array of `Name`, `maxItems:n`
- **Names stay plain strings**, not per-model enums. The tool list stays
  byte-stable across model switches, which preserves the prompt cache and
  avoids MCP `tools/list_changed` churn. Handlers validate names against the
  loaded model and return a `did_you_mean` list (difflib) on error.
- **Units:** meters, radians, seconds, world frame, `wxyz` quaternions
  internally. Tools take **roll/pitch/yaw** for orientation so the LLM never
  writes quaternions.
- **Results are compact JSON.** Floats are rounded to 4 dp. Joint Angles are
  name→value only for requested or changed joints. Errors use
  `{"ok":false,"error":{"code","message","did_you_mean"?,"blocked"?}}` and
  set `is_error: true`.
- **Every mutating tool**:
  - takes optional `expected_revision` (optimistic concurrency; stale →
    `code:"stale_revision"`);
  - auto-creates a checkpoint (ring of 32 `AIEditSessionCheckpoint`s) and
    returns `checkpoint_id`;
  - invalidates validation;
  - publishes a snapshot (live Orange preview);
  - returns `{"ok", "revision", "checkpoint_id", "changed":{...}, "validated":false, "warnings":[...]}`.
- **Idempotency.** `commit_keyframe`, `set_joint_angles`, and `author_keyframes`
  are upserts. Repeating the same call yields the same state, and
  `changed:{}` is returned when nothing changed (the `CommandResult.changed`
  semantics in `editor_controller.py:51-55`).
- **Safety.**
  - No path arguments; only `file_name` / project `name`. These resolve under
    the project root `exports/` or `GHOSTGUI_AGENT_OUTPUT_DIR` via
    `safe_project_path` (`project_manager.py:151`).
  - No model import from arbitrary paths, no shell or code, no hardware.
  - Protection removal is impossible (`edit_session.py:267-268`).
- **Terminology** in descriptions follows AGENTS.md: Keyframe, Commit Keyframe,
  Orange preview, End Effector, Joint Angles.

### 3.2 Workspace and model

```jsonc
// 1 list_models  [read-only]
{"type":"object","properties":{},"additionalProperties":false}
// → {"models":[{"key":"g1","name":"Unitree G1","type":"humanoid","nq":36,
//      "end_effectors":["left_hand","right_hand","left_foot","right_foot"],
//      "primitives":["burpee"],"exports":["ghostgui_csv","qpos_csv","mjlab_csv","dsms"]}],"loaded":"g1"}

// 2 load_model  [app-level; requires no unresolved candidate]
{"type":"object","properties":{"model_key":Name},"required":["model_key"],"additionalProperties":false}
// → {"ok":true,"model_key":"g1","nq":36,"joints":29,"duration_s":5.0}

// 3 new_project  [app-level]
{"type":"object","properties":{"name":Name,"model_key":Name,
  "duration_s":{"type":"number","minimum":0.1,"maximum":120}},
 "required":["name"],"additionalProperties":false}
// → {"ok":true,"project":"g1-cartwheel","model_key":"g1","duration_s":6.0}

// 4 open_project  [app-level; projects dir / recent list only]
{"type":"object","properties":{"name":Name},"required":["name"],"additionalProperties":false}

// 5 save_project  [app-level; saves committed motion only]
{"type":"object","properties":{},"additionalProperties":false}
// → {"ok":true,"revision":42,"autosave_cleared":true}

// 6 get_status  [read-only; cheap; call any time]
{"type":"object","properties":{},"additionalProperties":false}
// → {"model":"g1","project":"g1-cartwheel","dirty":true,"duration_s":6.0,
//    "keyframes":{"logical_times":7,"qpos":7},
//    "candidate":{"state":"staged","revision":18,"validated":true,"checkpoints":["c12","c13"]},
//    "generated":{"available":true,"dt":0.02,"samples":301,"stale":false},
//    "approval_policy":"review","budget":{"turns_left":22,"images_left":30,"usd_spent":0.84}}
```

### 3.3 Inspection

```jsonc
// 7 get_robot_summary  [read-only; cacheable per model]
{"type":"object","properties":{"include":{"type":"array","maxItems":8,"items":{"type":"string",
  "enum":["joints","limits","groups","end_effectors","logical_frames","qpos_layout","home_pose","primitives","conventions"]}}},
 "additionalProperties":false}
// → {"model":"g1","conventions":{"up":"+z","forward":"+x","units":"m,rad,s"},
//    "joints":[["left_hip_pitch_joint",-2.53,2.88],...],              // name, lower, upper
//    "groups":{"left_arm":[...],...},"end_effectors":[...],"logical_frames":[...],
//    "root":{"free":true,"home_pos":[0,0,0.793],"home_rpy":[0,0,0]},
//    "home_pose_nonzero":{"left_knee_joint":0.3,...},"primitives":["burpee"],
//    "mirror_pairs":true,"exports":{"mjlab_csv":true}}

// 8 list_keyframes  [read-only]
{"type":"object","properties":{"start_s":T,"end_s":T,"source":{"type":"string","enum":["candidate","committed"]}},
 "additionalProperties":false}
// → {"keyframes":[{"t":0.0,"kind":"both","frames":6,"author":"ai","protected":false},...]}

// 9 get_pose  [read-only; samples the timeline, not just Keyframes]
{"type":"object","properties":{
  "times_s":{"type":"array","items":T,"minItems":1,"maxItems":16},
  "fields":{"type":"array","maxItems":6,"items":{"type":"string",
     "enum":["root","joints","end_effectors","com","contacts","limit_margins"]}},
  "joints":Names(64),
  "source":{"type":"string","enum":["candidate","committed","generated"]}},
 "required":["times_s"],"additionalProperties":false}
// → {"poses":[{"t":1.5,"root":{"p":[0.1,0,0.62],"rpy":[1.57,0,0]},
//    "ee":{"left_hand":{"p":[..],"rpy":[..],"z":0.0}},"com":[..],"contacts":["left_hand","right_hand"]}]}

// 10 render_views  [read-only; returns image content]
{"type":"object","properties":{
  "times_s":{"type":"array","items":T,"minItems":1,"maxItems":8},
  "cameras":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"string",
     "enum":["front","side","top","iso","follow"]}},
  "layout":{"type":"string","enum":["contact_sheet","separate"]},
  "size_px":{"type":"integer","minimum":256,"maximum":768},
  "source":{"type":"string","enum":["candidate","committed","generated"]},
  "overlays":{"type":"array","maxItems":3,"items":{"type":"string","enum":["com","contacts","ground_grid"]}}},
 "required":["times_s"],"additionalProperties":false}
// → content: [image/png (one contact sheet: rows=cameras, cols=times, each cell stamped "t=…s cam")],
//            text: {"cells":[{"t":0.0,"camera":"side"},...],"source":"candidate","revision":18}

// 11 compute_metrics  [read-only]
{"type":"object","properties":{
  "start_s":T,"end_s":T,"dt":{"type":"number","minimum":0.005,"maximum":0.1},
  "metrics":{"type":"array","maxItems":16,"items":{"type":"string","enum":[
    "joint_limit_margin","joint_velocity","joint_acceleration","root_continuity",
    "ground_penetration","foot_sliding","self_collision","com_support_margin",
    "contact_schedule","end_pose_error","loop_seam","root_rotation_total","duration"]}},
  "source":{"type":"string","enum":["candidate","committed","generated"]}},
 "additionalProperties":false}
// → {"metrics":{"ground_penetration":{"max_m":0.004,"worst_t":2.3,"limit":0.01,"pass":true},
//               "joint_velocity":{"max_rad_s":7.1,"joint":"left_knee_joint","worst_t":1.1,"limit":8.0,"pass":true},
//               "root_rotation_total":{"about_forward_rad":6.21}, ...},
//    "hard_fail":[],"soft_warn":["foot_sliding"],"note":"kinematic heuristics; not a dynamics proof"}
```

### 3.4 Authoring (all act on the candidate working copy)

```jsonc
// 12 set_joint_angles  → existing set_joint_target / set_joint_group_target handlers
{"type":"object","properties":{
  "time_s":T,
  "angles":{"type":"array","minItems":1,"maxItems":64,"items":{"type":"object",
     "properties":{"joint":Name,"angle_rad":{"type":"number"}},"required":["joint","angle_rad"],"additionalProperties":false}},
  "mode":{"type":"string","enum":["absolute","relative"]},
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["time_s","angles"],"additionalProperties":false}
// → {"ok":true,"revision":19,"checkpoint_id":"c14","changed":{"qpos_keyframes":[1.5],"logical_keyframes":4},
//    "clamped":[{"joint":"left_knee_joint","requested":2.9,"applied":2.88}]}

// 13 solve_ik  → set_end_effector_target / set_logical_frame_target(+locked_end_effectors)
{"type":"object","properties":{
  "start_s":T,"end_s":T,
  "targets":{"type":"array","minItems":1,"maxItems":6,"items":{"type":"object","properties":{
     "frame":Name,"mode":{"type":"string","enum":["absolute","relative"]},
     "position_m":V3,"orientation_rpy_rad":V3},
     "required":["frame","mode"],"additionalProperties":false}},
  "locked_end_effectors":Names(6),
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["start_s","end_s","targets"],"additionalProperties":false}
// → {"ok":true,...,"residuals":{"pelvis":{"pos_m":0.0004,"rot_rad":0.002}},"locks_held":true,"collision_warnings":[]}

// 14 commit_keyframe  (Commit Keyframe: capture all logical targets + qpos from the pose at time_s)
{"type":"object","properties":{"time_s":T,"expected_revision":{"type":"integer","minimum":0}},
 "required":["time_s"],"additionalProperties":false}

// 15 delete_keyframes
{"type":"object","properties":{"times_s":{"type":"array","items":T,"minItems":1,"maxItems":32},
  "expected_revision":{"type":"integer","minimum":0}},"required":["times_s"],"additionalProperties":false}

// 16 edit_timeline  → application.timeline_editing plan_* + ApplyTimelineEditPlan; retime → retime_interval op
{"type":"object","properties":{
  "action":{"type":"string","enum":["set_duration","insert_time","shift_all","move_range","scale_range"]},
  "start_s":T,"end_s":T,"at_s":T,"amount_s":{"type":"number","minimum":-120,"maximum":120},
  "scale":{"type":"number","minimum":0.05,"maximum":20},"duration_s":{"type":"number","minimum":0.1,"maximum":120},
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["action"],"additionalProperties":false}

// 17 author_keyframes  (sparse whole-body authoring → existing qpos_keyframes replace/patch path)
{"type":"object","properties":{
  "mode":{"type":"string","enum":["replace","patch"]},
  "duration_s":{"type":"number","minimum":0.1,"maximum":120},
  "start_s":T,"end_s":T,
  "base_pose":{"type":"string","enum":["home","previous_anchor","current_motion"]},
  "keyframes":{"type":"array","minItems":1,"maxItems":24,"items":{"type":"object","properties":{
     "t":T,
     "root_position_m":V3,"root_rpy_rad":V3,
     "joints":{"type":"array","maxItems":64,"items":{"type":"object",
        "properties":{"joint":Name,"angle_rad":{"type":"number"}},"required":["joint","angle_rad"],"additionalProperties":false}},
     "groups":{"type":"array","maxItems":8,"items":{"type":"object",
        "properties":{"group":Name,"preset":{"type":"string","enum":["home","mirror_of_other_side"]}},
        "required":["group","preset"],"additionalProperties":false}},
     "label":{"type":"string","maxLength":40}},
     "required":["t"],"additionalProperties":false}},
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["mode","keyframes"],"additionalProperties":false}
// Unspecified joints inherit from base_pose. This is the token-efficient part: only deltas are sent.
// Locally: rpy→quat, joint limits checked, complete qpos built, anchors spaced ≤120° of root rotation
// (manifold interpolation takes the shortest arc, so full turns need ≥3 anchors), then delegated to
// trajectory_operations._apply_qpos_keyframes (:790).
// → {"ok":true,...,"anchors":7,"clamped":[...],"rotation_gaps":[],"collision_warnings":[...]}

// 18 apply_motion_primitive  → motion_primitive op (model-advertised only)
{"type":"object","properties":{"primitive":Name,"duration_s":{"type":"number","minimum":0.5,"maximum":120}},
 "required":["primitive","duration_s"],"additionalProperties":false}

// 19 mirror_motion  (NEW; needs RobotModelInfo.mirror_joint_pairs/sign + mirror plane in registry)
{"type":"object","properties":{"start_s":T,"end_s":T,
  "mode":{"type":"string","enum":["swap_sides_in_place","mirror_copy_to"]},"destination_start_s":T,
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["start_s","end_s","mode"],"additionalProperties":false}

// 20 repeat_motion  → repeat_motion op → motion_clipboard.plan_repeat_motion (exact copies)
{"type":"object","properties":{"start_s":T,"end_s":T,
  "additional_copies":{"type":"integer","minimum":1,"maximum":32},"ping_pong":{"type":"boolean"},
  "expected_revision":{"type":"integer","minimum":0}},
 "required":["start_s","end_s","additional_copies"],"additionalProperties":false}

// 21 generate_trajectory  → trajectory_generation.generate_trajectory_status on a detached copy
{"type":"object","properties":{"dt":{"type":"number","minimum":0.005,"maximum":0.1},
  "smoothing":{"type":"number","minimum":0,"maximum":1},
  "source":{"type":"string","enum":["candidate","committed"]}},
 "additionalProperties":false}
// → {"ok":true,"samples":601,"dt":0.02,"backend":"MuJoCo IK","max_ik_pos_err_m":0.003,
//    "max_ik_rot_err_rad":0.01,"anchors":13,"collision":{"first_warning_t":2.4,"first_blocking_t":null}}

// 22 apply_edit_spec  (escape hatch: the existing compact TrajectoryEditSpec, deterministic, no nested LLM)
{"type":"object","properties":{"mode":{"type":"string","enum":["edit","generate"]},"summary":{"type":"string","maxLength":4000},
  "operations":{"type":"array","minItems":1,"maxItems":16,"items":{"type":"object","properties":{
     "type":{"type":"string","enum":["root_offset","hold_pose","retime_interval","repeat_motion","set_joint_target",
       "set_joint_group_target","set_end_effector_target","set_logical_frame_target","lock_end_effector",
       "sparse_keyframes","qpos_keyframes","motion_primitive"]},
     "arguments_json":{"type":"string","maxLength":60000}},"required":["type","arguments_json"],"additionalProperties":false}}},
 "required":["mode","summary","operations"],"additionalProperties":false}
// Reuses parse/validate (trajectory_edit_spec.py) + TrajectorySpecExecutor (atomic, validated).
```

### 3.5 Validation, history, and acceptance

```jsonc
// 23 validate_motion  → GhostGUIMotionService.validate_motion; marks the revision validated on success
{"type":"object","properties":{"source":{"type":"string","enum":["candidate","committed"]}},"additionalProperties":false}
// → {"valid":true,"scope":"structural_kinematic","issues":[],"warnings":["…collision…"],"revision":18}

// 24 simulate_rollout  (Phase 5, optional; MuJoCo PD tracking of the reference; heuristic)
{"type":"object","properties":{"start_s":T,"end_s":T,"kp_scale":{"type":"number","minimum":0.1,"maximum":5}},
 "additionalProperties":false}
// → {"fell":false,"fall_t":null,"tracking_rms_rad":0.08,"max_root_drift_m":0.12,"note":"heuristic, not hardware safety"}

// 25 undo  → AIEditSession.restore_checkpoint(ring[id]); default = previous checkpoint
{"type":"object","properties":{"checkpoint_id":{"type":"string","maxLength":16}},"additionalProperties":false}

// 26 accept_candidate  → policy-gated AIEditSession.accept (one history entry; Undo restores)
{"type":"object","properties":{"description":{"type":"string","maxLength":200}},"required":["description"],"additionalProperties":false}
// → {"status":"accepted"|"pending_user_review"|"denied","reason":"…"}

// 27 discard_candidate
{"type":"object","properties":{},"additionalProperties":false}
```

### 3.6 Export and high-level

```jsonc
// 28 export_trajectory  → csv_io / trajectory_export_formats; resamples uniformly first
{"type":"object","properties":{
  "format":{"type":"string","enum":["ghostgui_csv","qpos_csv","mjlab_csv","dsms"]},
  "file_name":{"type":"string","minLength":1,"maxLength":80},
  "dt":{"type":"number","minimum":0.005,"maximum":0.1},
  "source":{"type":"string","enum":["accepted","candidate","generated"]}},
 "required":["format","file_name"],"additionalProperties":false}
// mjlab: checks mjlab_compatibility_error first (trajectory_export_formats.py:352) and enforces uniform dt
// (_uniform_input_fps rejects >2% jitter). The export is re-read with read_mjlab_trajectory as a round-trip check.
// → {"ok":true,"files":["exports/g1-cartwheel-x2.csv"],"samples":601,"fps":50,"source":"accepted","roundtrip":"ok"}

// 29 plan_motion_edit  (optional; nested call to the existing CompactMotionWorkflow; needs a provider key)
{"type":"object","properties":{"instruction":{"type":"string","minLength":1,"maxLength":4000}},
 "required":["instruction"],"additionalProperties":false}
// Useful for cheap single edits ("raise both arms 10 cm at 2 s"); the outer agent stays in control.
```

### 3.7 Token budget per typical call

| Call | Approx. tokens |
| --- | --- |
| Tool definitions (29) | ~6–7k (cached) |
| `get_robot_summary` (G1, full) | ~3–4k (cached in system prompt for loop A; an MCP resource for clients) |
| `get_pose` (4 times, root + EEs) | ~400 |
| `compute_metrics` (8 metrics) | ~350 |
| `author_keyframes` input (7 anchors, deltas only) | ~1.5–2.5k (vs ~5k with full 36-wide qpos) |
| `render_views` contact sheet (3 cams × 6 times, 1024×512) | ~700 image tokens + 150 text |

---

## 4. Agent Loop Design (in-app loop A and CLI; MCP clients bring their own)

### 4.1 System prompt outline (`application/ai/agent_prompts.py`)

The prompt is frozen and byte-stable for caching. Volatile state goes in messages.

1. **Role and boundary.** "You author robot motion inside GhostGUI using only
   the provided tools. You cannot run code, access files, or control hardware.
   Provider output is untrusted; GhostGUI validates locally."
2. **Terminology.** Keyframe, Commit Keyframe, Keyframe interval, Orange preview,
   Preview Path, End Effector, Joint Angles. Use them in user-facing text.
3. **World conventions.** Units, `+z` up, model forward axis from
   `RobotModelInfo.forward_axis`, rpy order, root rotation continuity rule
   (anchors ≤120° apart), 120 s cap.
4. **Robot card** (per model, cache breakpoint): `get_robot_summary` output plus
   registry hints such as primitives, mirror availability, export compatibility,
   and typical stance and contact frames.
5. **Workflow contract.** Plan → author sparse → validate → measure → look →
   fix → finish (§4.2). Prefer the coarsest tool that works. Use primitives when
   advertised. Use `repeat_motion` for loops, never re-authoring.
6. **Quality bar.** Hard gates (validation valid, no blocking collision, ground
   penetration ≤1 cm, Joint Angle speed ≤ registry limit) and soft goals
   (no foot sliding while in stance, CoM within support when quasi-static,
   endpoints match intent).
7. **Honesty.** State concessions (for example "kinematic cartwheel; flight and
   balance not physically validated"). Never claim dynamics feasibility.
8. **Stopping and reporting.** When gates pass and intent is met, or the budget
   reminder arrives, stop and summarise: phases, metrics table, files written,
   concessions, and what the user should review.
9. **Approval policy** (injected as a mid-conversation system message at run
   start, so the frozen prompt stays cacheable): `review` / `auto_accept` /
   `auto_accept_and_export`.

### 4.2 Strategy for motion from scratch (worked: G1 cartwheel ×2 → mjlab)

| Step | Tool calls | Notes |
| --- | --- | --- |
| 0 Orient | `get_status`, `list_models` | `load_model(g1)` if needed |
| 1 Set up | `new_project("g1-cartwheel", g1, duration_s=6)` | |
| 2 Decompose | (thinking) | Phases: 0.0 stand → 0.8 lunge + arms up → 1.6 first hand plant (lateral) → 2.4 inverted, legs split, roll≈π → 3.2 second hand plant → 4.2 first foot down → 5.0 second foot down → 6.0 stand. The rotation is about the forward axis; anchors every ≤90° of roll. |
| 3 Author sparse | `author_keyframes(mode=replace, base_pose=home, keyframes=[8 anchors with root_position_m, root_rpy_rad, joint deltas])` | One call, about 2k tokens |
| 4 Contacts | `solve_ik(start_s=1.6, end_s=2.4, targets=[pelvis relative], locked_end_effectors=[left_hand])`, then the same for the right hand and feet | Puts hands and feet on the ground during support phases |
| 5 Check | `validate_motion`, `compute_metrics([ground_penetration, foot_sliding, joint_velocity, root_rotation_total, end_pose_error, self_collision])` | |
| 6 Look | `render_views(times=phase times, cameras=[front, side], layout=contact_sheet)` | Vision self-critique against the phase list: "is the body inverted at 2.4 s? are hands planted?" |
| 7 Fix | Targeted `set_joint_angles` / `solve_ik` / `author_keyframes(mode=patch)` / `undo` | ≤3 critique iterations without metric improvement, then stop |
| 8 Loop | `repeat_motion(start_s=0, end_s=6, additional_copies=1)` | "loop it twice" = 1 additional copy; duration becomes 12 s. Then `compute_metrics([loop_seam])`. |
| 9 Generate | `generate_trajectory(dt=0.02)` | Uniform samples for mjlab |
| 10 Commit | `accept_candidate` (policy) → `save_project` | In `review` mode the run ends here with a result card |
| 11 Export | `export_trajectory(mjlab_csv, "g1-cartwheel-x2", dt=0.02, source=accepted)` | Compatibility and round-trip checked |
| 12 Report | Final text | Metrics table, concessions, file path |

### 4.3 Loop mechanics (provider-neutral; `application/ai/agent_loop.py`)

```python
@dataclass(frozen=True)
class AgentRunLimits:
    max_provider_turns: int = 40
    max_tool_calls: int = 120
    max_images: int = 48                 # images returned to the model per run
    max_wall_seconds: float = 900.0
    max_cost_usd: float = 5.00           # from usage incl. cache read/write tokens
    request_timeout_seconds: float = 300.0
    max_output_tokens: int = 32_000      # thinking counts toward this; needs streaming
    max_consecutive_tool_errors: int = 4
    max_identical_failed_call: int = 2   # existing agent.py:228-236 behaviour

class MotionAgentLoop:
    def __init__(self, provider: LLMProvider, catalog: CommandCatalog, *, prompt: AgentPrompt,
                 limits: AgentRunLimits, policy: ApprovalPolicy, events: AgentEventSink,
                 pricing: ModelPricing): ...
    async def run(self, instruction: str, *, model: str, effort: str = "high",
                  cancellation_token: CancellationSignal | None = None) -> AgentRunResult: ...
```

- **Append-only transcript.** Assistant turns are stored with their **raw
  provider blocks** (thinking, text, tool_use) and replayed verbatim. History is
  never edited, which matches the preserved-thinking rules on Opus 5.5 /
  Sonnet 5.5.
- **Tool results.** One `user` message holds *all* `tool_result`s of a turn,
  including parallel calls. Images go inside `tool_result.content` as image
  blocks. Failures use `is_error: true`; nothing is dropped.
- **Budget pressure.** At 80% of any budget, append a mid-conversation
  `{"role":"system"}` message: "Budget nearly exhausted: finish, validate,
  report." This needs no cache invalidation and is supported on Opus 5.5 and
  Sonnet 5.5. Optionally pass `output_config.task_budget` (beta
  `task-budgets-2026-03-13`) **(verify)**.
- **Stop reasons.**
  - `end_turn`: finish.
  - `tool_use`: execute.
  - `max_tokens`: one retry with a higher cap, then fail the turn.
  - `pause_turn`: append and continue (today it raises; `anthropic.py:314`).
  - `refusal`: surface the category (already parsed at `anthropic.py:297-304`)
    and keep the best checkpoint.
- **Best-candidate tracking.** After each validated revision, compute a score
  (hard gates + soft metrics). On budget exhaustion or cancellation, restore the
  best checkpoint rather than the last one.

### 4.4 Model choice, thinking, caching

| Concern | Choice |
| --- | --- |
| Default loop model | `claude-opus-5-5`, `output_config.effort="high"` (Opus 5.5 defaults to `medium`, so set it explicitly) |
| Economy preset | `claude-sonnet-5-5`, `effort="medium"` (same tools and prompt) |
| Cheap side tasks | `claude-haiku-4-5` for intent routing (simple edit → existing single-shot `CompactMotionWorkflow`; motion-from-scratch → agent loop) and transcript titles. Never switch the main loop's model mid-run: caches are model-scoped. |
| Thinking | Opus 5.5 is always adaptive (omit `thinking` or send `{"type":"adaptive"}`; `disabled` / `budget_tokens` return 400). Use `display:"summarized"` to show reasoning summaries in the panel, or `display:"updates"` (beta `thinking-display-updates-2026-08-18`) for terse progress notes between tool calls **(verify on SDK 1.4.0)**. |
| Tool choice | `auto` only. Forced `any` / `tool` returns 400 on Opus 5.5 / Sonnet 5.5. |
| Sampling | No `temperature`. The provider already gates it per model (`anthropic.py:97-101`); keep the Opus/Sonnet 5.5 IDs out of `temperature_capable_models`. |
| Streaming | `client.messages.stream(...)` + `get_final_message()` for 32k output. Set `eager_input_streaming: true` on `author_keyframes` / `apply_edit_spec` (large inputs) and validate the parsed input against the schema before dispatch. |
| Prompt caching | Order: tools (sorted, frozen) → system [frozen prompt] + [robot card, `cache_control: {"type":"ephemeral","ttl":"1h"}`] → messages with top-level automatic `cache_control`. Verify `usage.cache_read_input_tokens > 0` from turn 2 onward in tests with a fake client. |
| Refusal fallback | Opt in by default on the Claude API: `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`. Configurable off. |
| Images | Contact sheets (one image per call) at ≤1024 px wide. Cap at 48 images per run. Prefer metrics first, eyes second. |

Rough cost for the cartwheel run on Opus 5.5 ($4 / $20 per MTok; cache reads
$0.20): ~25 turns, ~50k average context, mostly cache reads. That is about
1.2M cached-read tokens (~$0.25), ~150k uncached or cache-write tokens
(~$0.75), and ~60k output tokens including thinking (~$1.20), so **≈ $2–3**.
Sonnet 5.5 is roughly half. The default `max_cost_usd=5.00` covers it with
headroom. Re-baseline with `count_tokens` once the catalog exists.

### 4.5 Required provider-layer changes (Anthropic first, Gemini parity later)

- `schemas.py`:
  - add `ProviderMessage.provider_blocks: tuple[Mapping[str, Any], ...] = ()`
    (opaque, replayed verbatim on assistant turns);
  - add `ToolResult.content_parts` (text + images);
  - add `Usage.cache_read_input_tokens` / `cache_creation_input_tokens`;
  - add `ProviderRequest.effort`, `thinking_display`, `cache_breakpoints`,
    `stream: bool`, `fallbacks`;
  - add `StopReason.PAUSED`.
- `providers/anthropic.py`:
  - build requests with `output_config.effort` and the `thinking` display;
  - emit `cache_control` on system blocks plus top-level auto caching;
  - stream for large `max_tokens`;
  - **keep every raw content block** (thinking / redacted_thinking / text /
    tool_use) in `provider_blocks`;
  - replay them first on the next request;
  - parse cache usage;
  - continue on `pause_turn`;
  - put images inside `tool_result` content.
- `providers/replay.py`: include `provider_blocks` and tool-result image digests
  in the fingerprint and payload (`:231-366`).
- `provider_registry.py`:
  - models `("claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-4-5")`;
  - capability flags `supports_agent_loop` and `supports_prompt_cache`;
  - optionally default to Anthropic when only `ANTHROPIC_API_KEY` is present.
- `limits.py`: add `AGENT_MAX_OUTPUT_TOKENS = 32_000` and the other agent
  budgets, separate from the single-shot budgets (the pattern used at `:11-12`).

### 4.6 Accept gate and approval policy

| Policy | `accept_candidate` | `save_project` / `export_trajectory` | Default for |
| --- | --- | --- | --- |
| `review` | Returns `pending_user_review`. The panel shows the result card; the user clicks Accept. | Save allowed only after the user accepts. Export from `candidate` is allowed into the sandbox and labelled. | In-app Agent mode, GUI attach |
| `auto_accept` | Accepts if validation is current and hard gates pass. One history entry, Undo-able. | Allowed | Prompt-only mode (opt-in), CLI `--auto-accept` |
| `auto_accept_and_export` | As above | Export required before finishing | CLI scenarios, CI evals |

In every policy:

- Acceptance still goes through `AIEditSession.accept`, which rejects stale
  commits (`edit_session.py:294-297`).
- Protection and provenance rules are unchanged.
- Never auto-accept over protected content.
- `docs/ai_security.md` must be updated to describe the opt-in autonomous policy.

### 4.7 Cancellation, progress, transcript

- **Cancellation.**
  - `CancellationToken` (existing contract, `providers/base.py:16-21`) is
    checked before each provider turn, before each tool, and inside long tools
    (IK loops, generation, rollout).
  - In-flight provider requests use the existing poll-and-cancel pattern
    (`anthropic.py:117-123`).
  - On cancel: restore the last *validated* checkpoint, emit `cancelled`, and
    leave the candidate staged for review.
- **Progress.**
  - `AgentEvent` kinds: `turn_started`, `thinking_summary`, `text_delta`,
    `tool_started{name, args_summary}`, `tool_finished{ok, summary, ms}`,
    `candidate_updated{revision}`, `image_rendered{thumb}`, `budget{…}`,
    `awaiting_approval`, `done{summary, metrics}`, `failed{error}`.
  - These map onto the panel through the existing queued-signal pattern
    (`ai_assistant_panel.py:49` `progress_received`), onto MCP progress/log
    notifications, and onto CLI JSONL on stdout.
- **Transcript.**
  - `chat_transcript.py` holds the visible transcript.
  - `diagnostics.py` (`GHOSTGUI_AI_DEBUG=1`) records the per-run JSON
    transcript (blocks, tool calls, results with image hashes, usage, cost) with
    secrets scrubbed.
  - Development recording uses `RecordedProvider` (requires
    `development_mode=True` and a sanitizer). Replay uses `ReplayProvider`; it
    is deterministic because tool results are deterministic given the same
    code, model assets, and a **stub renderer** (see §5).

---

## 5. Evaluation

### 5.1 Test layers (all network-free in CI)

| Layer | What | How |
| --- | --- | --- |
| Tool contracts | Every tool's schema registers in `ToolRegistry`. Invalid args are rejected. Results are JSON-safe and size-bounded. Mutating tools checkpoint, invalidate validation, and honour `expected_revision`. Path sandbox rejects `..`, absolute paths, and symlink escape. | `tests/test_agent_api_commands.py`, `tests/test_agent_api_paths.py` with real `MuJoCoRobotAdapter("g1")` (already used headless in `tests/test_ai_motion_primitives.py:35`) |
| Workspace parity | Commit Keyframe, Generate, and export through the new services equal the old GUI results byte-for-byte on a fixture motion | `tests/test_keyframe_service.py`, `tests/test_trajectory_export_service.py` |
| Metrics | Synthetic trajectories with known foot sliding, penetration, seam error, and rotation | `tests/test_agent_api_metrics.py` |
| Rendering | `StubRenderer` (deterministic PNG from a qpos hash) by default. One real offscreen test, skipped when no GL (`MUJOCO_GL=egl` on Linux CI; `libegl1` is already installed in `.github/workflows/ci.yml`). | `tests/test_agent_api_rendering.py` |
| MCP server | `list_tools` returns the catalog. `call_tool` round-trips text and image content. Errors set `isError`. Runs over the MCP SDK's in-memory client/server pair **(verify name in the pinned `mcp`)** or a stdio subprocess. | `tests/test_agent_mcp_server.py` |
| Bridge | Loopback auth. Token required. One client. Snapshot reaches the Qt thread. User edit during a run → `stale_revision`. | `tests/test_agent_bridge.py` (`QT_QPA_PLATFORM=offscreen`) |
| Loop mechanics | `MockProvider` scripted steps: parallel tool results in one message; thinking blocks replayed verbatim; budgets; repeated-failure abort; cancel restores best checkpoint; approval policies; `pause_turn` continuation | `tests/test_ai_agent_loop.py` |
| Anthropic adapter | Fake async client asserts request shape (effort, thinking, cache_control order, no temperature, streaming, fallbacks header) and block preservation | extend `tests/test_ai_anthropic_provider.py` |
| Golden replays | Recorded sanitized sessions for 5 scenarios replayed end-to-end through `ReplayProvider` and `StubRenderer`, then the assertions below | `tests/test_agent_scenarios_replay.py`, fixtures under `tests/fixtures/agent_replays/` (new, synthetic, documented) |

### 5.2 Golden scenarios and automatic checks

| Scenario (G1 unless noted) | Hard checks | Soft checks |
| --- | --- | --- |
| "3-second squat, return to standing" | valid; duration 3 ± 0.05 s; `end_pose_error` < 1 cm / 0.02 rad; pelvis z dips ≥ 0.15 m | `foot_sliding` < 1 cm; `com_support_margin` > 0 |
| "Wave the right hand three times" | valid; right wrist/elbow oscillation has 3 peaks; left side within 0.05 rad of start | feet static |
| "Jump in place" (kinematic) | valid; root z rises ≥ 5 cm with both feet off the ground over an interval; lands at start pose | concession text mentions dynamics |
| "Five push-ups" (from the prone reference) | valid; uses the reference pose or primitive; 5 elbow cycles via `repeat_motion` (exact copies) | hands/feet planted (`foot_sliding` small) |
| "6 s cartwheel-like, end standing, loop twice, mjlab CSV" | valid; duration 12 ± 0.1 s; `root_rotation_total` about the forward axis ≥ 5.5 rad per cycle; `loop_seam` < 1 cm; file exists; `read_mjlab_trajectory` round-trip OK; uniform dt | hand plants detected in `contact_sheet` order L→R |
| Go2: "trot in place 4 s" | valid; 4 feet alternate contacts | — |
| Negative: "export mjlab for Go2" | the agent reports incompatibility (from `mjlab_compatibility_error`); no file written | — |

Per-scenario scores are recorded as JSON: pass/fail plus soft score plus
turns, tool calls, images, tokens, and cost. An optional **vision LLM judge**
(rubric over the final contact sheet) runs only in live evals.

### 5.3 Live evaluation

Extend `.github/workflows/live-ai-provider-smoke.yml` with an
`agent-scenarios` job:

- `workflow_dispatch`
- inputs: `scenarios`, `model`, `max_cost_usd`
- runs `ghostgui agent --scenario <id> --auto-accept --json --max-cost $X`
  headless with `MUJOCO_GL=egl`
- uploads artifacts: transcript without image bytes, contact sheets, metrics,
  exported CSV

Never run this on pull requests. The existing four-request contract smoke stays
as-is.

---

## 6. Phased Implementation Plan

Commands every phase must pass:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_docs.py
python3 scripts/check_architecture.py
```

### Phase 1: Headless command layer (foundation)

- **Add**
  - `application/agent_api/__init__.py`
  - `workspace.py`: `Workspace` protocol, `HeadlessWorkspace`, `GeneratedMotion`
  - `commands.py`: `build_command_catalog(workspace) -> CommandCatalog`,
    holding `ToolRegistry` + metadata: `read_only`, `mutates_candidate`,
    `app_level`, `returns_images`
  - `runtime.py`: `ToolRuntime.call(name, args) -> ToolOutcome`, covering
    validation, budgets, checkpoints, events, and error mapping
  - `schemas.py`: shorthand expansion
  - `checkpoints.py`: 32-entry ring over `AIEditSession.checkpoint()`
  - `sandbox.py`: output-root resolution using `safe_project_path`
  - `events.py`
  - `metrics.py`: the Phase-1 subset `joint_limit_margin`, `joint_velocity`,
    `ground_penetration`, `self_collision`, `end_pose_error`, `loop_seam`,
    `root_rotation_total`, `duration`
  - `rendering.py`: `OffscreenMotionRenderer(MotionFrameRenderer)` on
    `mujoco.Renderer`, camera presets from the registry, numpy contact-sheet
    tiling, zlib PNG encoder (same technique as
    `scripts/smoke_ai_provider.py:143-160`), `StubRenderer` for tests
- **Add application services, extracted from Qt callbacks**
  - `application/keyframe_service.py`: `commit_keyframe(document, adapter, time, *, phase="", selected_frame=None) -> CommandResult`,
    built from `main_window.py:3560-3656` + `timeslice_service.capture_timeslice_from_committed_pose`,
    and upserting the qpos state too
  - `application/trajectory_export_service.py`: `build_trajectory_export(adapter, timeline, generated, *, current_qpos=None, sample_dt=None) -> (TrajectoryExport, warning_report, blocking_report)`,
    from `robot_viewer_3d.py:2346-2391`
  - `application/workspace_state.py`: `default_workspace_payload(...)` /
    `merge_workspace_payload(...)` for headless saves, and a pure helper used by
    `capture_project_workspace`
- **Modify**
  - `gui/main_window.py`: `define_timeslice_from_committed_pose` and
    `on_generate_trajectory` call the services
  - `gui/robot_viewer_3d.py`: `_trajectory_export_snapshot` delegates
  - `core/models/registry.py`: new `RobotModelInfo` fields with safe defaults —
    `camera_presets`, `mirror_joint_pairs`, `mirror_joint_signs`,
    `joint_velocity_limits`, `support_frames`, `primitive_aliases`
  - `scripts/check_architecture.py`: forbid `PySide6` / `OpenGL` imports under
    `application/agent_api/` (stricter than `application/`, because
    `background_jobs.py` imports QtCore)
- **Dependencies:** none (numpy and mujoco are already required).
- **Tests:**
  - `tests/test_agent_api_workspace.py`, `test_agent_api_commands.py`,
    `test_agent_api_metrics.py`, `test_agent_api_paths.py`,
    `test_agent_api_rendering.py`
  - `tests/test_keyframe_service.py`, `test_trajectory_export_service.py`
  - plus an architecture test case
- **Docs:**
  - `docs/architecture.md` adds an "Agent command layer" section and ownership
    table rows
  - `docs/agent_integration.md` (new, developer-facing: catalog, conventions,
    safety)
  - link it from `docs/README.md`
- **AGENTS.md compliance:** services move out of Qt callbacks; model-specific
  data goes into the registry; no change to export contracts, fixtures, CSVs,
  or cache versions.

### Phase 2: MCP server (`ghostgui-mcp`, stdio)

- **Add**
  - `application/agent_api/mcp_server.py`: low-level MCP `Server` so the
    catalog's JSON schemas pass through verbatim. Handlers: `list_tools`,
    `call_tool` (text → `TextContent` JSON; images → `ImageContent` base64
    PNG); `list_prompts` (`ghostgui_author_motion` = system prompt §4.1);
    `list_resources` (`ghostgui://robot/{key}/summary`,
    `ghostgui://docs/terminology`). Server `instructions` carry the
    terminology and workflow contract. `main()` parses
    `--model`, `--project`, `--output-dir`, `--attach {auto,never,required}`,
    `--approval {review,auto_accept,auto_accept_and_export}`.
  - Import-guard `mcp` with a clear "install ghostgui[agent]" error.
- **Modify** `pyproject.toml`:
  - `[project.scripts] ghostgui-mcp = "application.agent_api.mcp_server:main"`
  - `[project.optional-dependencies] agent = ["mcp==<exact version validated in CI>"]`
    (pin exactly, like the `ai` extra)
- **Tests:** `tests/test_agent_mcp_server.py`, skipped when `mcp` is not
  installed. CI installs `.[agent]` in one matrix leg.
- **Docs:**
  - `docs/agent_integration.md`: Claude Code and Claude Desktop setup (§7.7),
    tool list, safety
  - `docs/ai_security.md`: new "External agent control" section covering
    stdio-only, the sandbox, no shell, approval policy, and data disclosed to the
    client's model
  - `docs/install.md`: the `agent` extra
  - `docs/README.md` index entry
  - `CHANGELOG.md`

### Phase 3: GUI live-attach

- **Add**
  - `gui/agent_bridge.py`: `AgentBridgeServer` (`QTcpServer` on 127.0.0.1,
    token file 0600), `GuiWorkspace(Workspace)`, `MainThreadInvoker`
  - `application/agent_api/bridge_protocol.py`: JSON-RPC types, Qt-free
  - `application/agent_api/bridge_client.py`: asyncio socket client, used by
    `mcp_server --attach`
- **Modify**
  - `gui/ai_assistant_controller.py`:
    - `begin_agent_session(owner)`, `present_agent_candidate(snapshot)` (sampler
      from an immutable snapshot), `end_agent_session(outcome)`;
    - route `accept_candidate` through the existing `accept()` (keeps the history
      transaction, `:929-990`)
  - `gui/panels/ai_assistant_panel.py` + `gui/widgets/ai_chat.py`: "External
    agent connected" banner, activity rows from `AgentEvent`, Disconnect
  - `gui/ai_frame_capture.py`: optional camera override (save/restore camera
    fields alongside `_CanvasPresentation`)
  - `gui/main_window.py`:
    - setting `agent/bridge_enabled` (default False), menu toggle **Allow
      external agent control**;
    - `load_model` / `open_project` / `new_project` / `save` invokable
      without dialogs; project transitions refuse while a candidate is
      unresolved
  - `gui/help/help_content.py` (+ `gui/tutorial/tutorial_steps.py` if a step
    references the assistant): "External agent control" copy
- **Tests:** `tests/test_agent_bridge.py` (offscreen Qt), extend
  `tests/test_ai_assistant_controller_chat.py`.
- **Docs:** `docs/agent_integration.md` attach section; `docs/user_guide.md`
  short how-to; `docs/ai_security.md` bridge threat model (loopback-only,
  token, opt-in, single client, visible indicator).

### Phase 4: In-app multi-turn agent loop (Anthropic tool use)

- **Add**
  - `application/ai/agent_loop.py` (`MotionAgentLoop`, `AgentRunLimits`,
    `ApprovalPolicy`, `AgentRunResult`, best-candidate tracking)
  - `application/ai/agent_prompts.py`
  - `application/ai/pricing.py` (per-model $/MTok table including cache rates;
    the single place to update)
  - `application/agent_api/cli.py` (`ghostgui agent "prompt" [--model g1]
    [--project NAME] [--auto-accept] [--export mjlab_csv] [--max-cost 5]
    [--json] [--scenario ID]`)
- **Modify**
  - `application/ai/schemas.py`, `providers/anthropic.py`, `providers/replay.py`,
    `providers/mock.py` (scripted raw blocks), `provider_registry.py`,
    `limits.py` (§4.5)
  - `application/launcher.py`: dispatch `agent` subcommand before importing Qt
  - `gui/ai_assistant_controller.py`: Agent-mode run as one cancellable job
    (`submit_cancellable`) running `asyncio.run(loop.run(...))` against a
    `GuiWorkspace` built in-process (no socket); events go through a queued
    signal
  - `gui/panels/ai_assistant_panel.py`: **Edit / Agent** mode switch,
    streaming text, budget meter, Stop
  - Gemini adapter: either preserve thought signatures equivalently or declare
    `supports_agent_loop=False` until it does
- **Dependencies:** bump `anthropic==` only if 1.4.0 lacks a needed feature;
  verify streaming helpers, `eager_input_streaming`, and the `fallbacks` param
  on 1.4.0 first.
- **Tests:** `tests/test_ai_agent_loop.py`, extended
  `test_ai_anthropic_provider.py` / `test_ai_provider_replay.py` /
  `test_ai_provider_registry.py`, `tests/test_agent_cli.py`.
- **Docs:**
  - `docs/ai_assistant.md`: Agent mode, budgets, approval policies, model
    presets; update the pinned baseline (`:18-22`)
  - `docs/ai_security.md`: multi-turn tool loop is now the opt-in path;
    executable tool declarations are allowlisted; revise the "one planning
    request" statements in `:77-84`
  - `docs/architecture.md:249-255`: the "no autonomous loop" paragraph becomes
    "the single-shot path has no loop; Agent mode is an opt-in bounded loop"
  - help copy
  - `CHANGELOG.md`

### Phase 5: Vision self-critique and metrics

- **Add**
  - `application/agent_api/metrics.py` (full set): foot sliding with stance
    detection, `com_support_margin` (quasi-static CoM ground projection vs
    support polygon from `support_frames`), `contact_schedule`, joint
    acceleration/jerk, `root_continuity`
  - `application/agent_api/rollout.py`: optional `simulate_rollout` (MuJoCo
    `mj_step`, PD tracking toward the reference qpos, gains from the registry;
    falls detected by root height/tilt; labelled heuristic)
  - camera presets + overlays in `rendering.py`
  - `critique_rubric` text in `agent_prompts.py`
- **Modify**
  - `core/models/registry.py`: populate G1/H2/Go2/Z1 camera presets, support
    frames, and velocity limits
  - move the burpee intent policy (`trajectory_planner.py:337-398`) to
    registry-declared `primitive_aliases`, with a focused test
  - `application/ai/visual_critique.py`: reuse its observation schema for the
    optional LLM judge
- **Tests:** metric unit tests with synthetic motions; rollout smoke on G1
  standing (no fall) and on a deliberately unbalanced pose (fall).
- **Docs:** `docs/motion_math.md` metric definitions; `docs/ai_security.md`
  "not a safety proof" language for rollout.

### Phase 6: Prompt-only mode UX

- **Add** `gui/panels/agent_prompt_panel.py`: start-screen "Describe a motion"
  box, model and approval presets, plan checklist (phases ticked as
  `AgentEvent`s arrive), live Orange preview auto-playback loop, final card with
  metrics table, Accept / Export / Open project buttons.
- **Modify**
  - `gui/project_browser.py`: "Start from a prompt" entry
  - `gui/main_window.py`: wire panel
  - `gui/tutorial/tutorial_steps.py` + `gui/help/help_content.py`: new tutorial
    step and help section
  - `docs/user_guide.md`
  - `docs/README.md`
- **Tests:** offscreen UI tests in the style of
  `tests/test_ai_assistant_ui.py`: prompt → scripted `MockProvider` run →
  result card → Accept → history entry.

---

## 7. One-Shot Implementation Prompts (copy-paste into Claude Code)

Each prompt is self-contained. Run them in order on a fresh branch per phase.

### 7.1 Phase 1: headless command layer

**Model / effort:** `claude-opus-5-5`, effort `xhigh`.

```text
Goal: Add a Qt-free agent command layer to GhostGUI so every motion-authoring workflow
(model load, project create/open/save, inspect, IK, Joint Angles, Commit Keyframe,
sparse Keyframe authoring, primitives, repeat, timeline edits, generate, validate,
metrics, render, undo, accept/discard, export) is callable headlessly as typed tools.

Read first: AGENTS.md, docs/architecture.md, docs/ai_security.md,
application/ai/edit_session.py, application/ai/trajectory_operations.py (handlers :45-121),
application/ai/trajectory_executor.py, application/ai/tool_registry.py,
application/ai/motion_services.py, application/ai/context.py (:251-450),
application/ai/frame_capture.py (:88-129), application/timeslice_service.py,
application/trajectory_generation.py, application/trajectory_export_formats.py,
application/csv_io.py, application/project_manager.py (:151, :756-1040),
gui/main_window.py (:1696-1832, :3560-3656, :3974-3999), gui/robot_viewer_3d.py (:2346-2391),
core/models/registry.py, tests/test_ai_motion_primitives.py (headless fixture pattern).

Create:
- application/agent_api/{__init__,workspace,commands,runtime,schemas,checkpoints,sandbox,events,metrics,rendering}.py
- application/keyframe_service.py: commit_keyframe(document, adapter, time_s, *, phase="", selected_frame=None) -> CommandResult
  (captures all logical targets from the qpos sampled at time_s via
  timeslice_service.capture_timeslice_from_committed_pose and upserts the qpos state; one UpsertKeyframes command).
- application/trajectory_export_service.py: build_trajectory_export(adapter, timeline, generated=None, *,
  current_qpos=None, sample_dt=None) -> TrajectoryExportBundle(export, warning_report, blocking_report).
- application/workspace_state.py: pure helpers for workspace payloads (headless defaults + merge).

Interfaces:
- class Workspace(Protocol) exactly as in review/03-claude-agent-integration.md §2.3.
- class HeadlessWorkspace(Workspace): __init__(self, *, projects_dir=None, output_dir=None,
  renderer_factory=OffscreenMotionRenderer, approval_policy="review").
- build_command_catalog(workspace) -> CommandCatalog; CommandCatalog.registry: ToolRegistry;
  CommandCatalog.meta[name] -> ToolMeta(read_only, mutates_candidate, app_level, returns_images).
- ToolRuntime(catalog, workspace, limits).call(name: str, arguments: dict) -> ToolOutcome(ok, payload: dict,
  images: tuple[EncodedFrame, ...], is_error: bool).
- Implement the 28 non-LLM tools in review/03-claude-agent-integration.md §3 (omit plan_motion_edit
  and simulate_rollout for now) with those schemas, restricted to ToolRegistry's supported keywords.
- Mutating tools: checkpoint (32-ring), accept expected_revision, invalidate validation,
  call workspace.publish_candidate(capture_motion_state(working_document)), return checkpoint_id.
- Names are plain strings validated per model with difflib suggestions. Round floats to 4 dp.
- File outputs only via sandbox.resolve_output(file_name) under <project>/exports or the output_dir;
  reuse project_manager.safe_project_path; reject absolute paths, '..', and symlink escapes.
- rendering.OffscreenMotionRenderer implements MotionFrameRenderer with mujoco.Renderer, camera presets
  from RobotModelInfo.camera_presets (fall back to forward_axis/up_axis), contact-sheet tiling in numpy,
  PNG via zlib/struct (no Pillow). Also StubRenderer (deterministic PNG from a qpos hash).
- metrics: joint_limit_margin, joint_velocity, ground_penetration, self_collision (core.ik.collision.
  trajectory_collision_reports), end_pose_error, loop_seam, root_rotation_total, duration.
- core/models/registry.py: add RobotModelInfo fields with empty defaults: camera_presets, mirror_joint_pairs,
  mirror_joint_signs, joint_velocity_limits, support_frames, primitive_aliases. Populate mirror pairs for g1.
- Refactor gui/main_window.py define_timeslice_from_committed_pose and on_generate_trajectory, and
  gui/robot_viewer_3d.py _trajectory_export_snapshot, to call the new services without behaviour change.
- scripts/check_architecture.py: forbid PySide6/PyQt/OpenGL imports in application/agent_api/.

Constraints: preserve unrelated working-tree changes; do not modify robot assets, CSV examples,
fixtures, cache versions, or export contracts; use canonical terminology (Keyframe, Commit Keyframe,
Orange preview, End Effector, Joint Angles) in tool descriptions.

Tests (unittest): tests/test_agent_api_{workspace,commands,metrics,paths,rendering}.py,
tests/test_keyframe_service.py, tests/test_trajectory_export_service.py, plus an architecture case.
Include an end-to-end headless test: new_project(g1) → author_keyframes (squat, 4 anchors) →
validate_motion → repeat_motion(1) → generate_trajectory(dt=0.02) → accept_candidate (auto_accept policy)
→ export_trajectory(mjlab_csv) → read back with read_mjlab_trajectory. Use StubRenderer by default; real
offscreen render test skips when GL is unavailable.

Docs: add docs/agent_integration.md (developer catalog + conventions + safety), link from docs/README.md,
add the agent command layer to docs/architecture.md ownership table.

Acceptance: all commands pass:
  python3 -m unittest discover -s tests -v
  python3 scripts/check_docs.py
  python3 scripts/check_architecture.py
and `python3 -c "from application.agent_api.workspace import HeadlessWorkspace"` imports without PySide6.
```

### 7.2 Phase 2: MCP server

**Model / effort:** `claude-sonnet-5-5`, effort `high`.

```text
Goal: Expose the Phase-1 CommandCatalog as a stdio MCP server `ghostgui-mcp` usable from Claude Code
and Claude Desktop, with no LLM code inside GhostGUI.

Read: application/agent_api/* (Phase 1), docs/agent_integration.md, docs/ai_security.md, pyproject.toml,
review/03-claude-agent-integration.md §2.5, §3, §7.7.

Create application/agent_api/mcp_server.py using the MCP Python SDK *low-level* Server (so catalog JSON
schemas pass through verbatim):
- list_tools → mcp.types.Tool(name, description, inputSchema) for every catalog tool (sorted by name).
- call_tool → ToolRuntime.call; payload as TextContent(JSON, compact); images as ImageContent(base64 PNG,
  mimeType "image/png"); errors return isError=True with the compact error JSON.
- list_prompts/get_prompt → "ghostgui_author_motion" (the system prompt outline in §4.1 of the review).
- list_resources/read_resource → ghostgui://robot/{model_key}/summary and ghostgui://docs/terminology.
- Server instructions: terminology + workflow contract (plan → author sparse → validate → metrics → look → fix → finish).
- main(): argparse --model, --project, --output-dir, --attach {never} (auto/required are added in Phase 3),
  --approval {review,auto_accept,auto_accept_and_export}; run over stdio; never print to stdout except
  protocol frames (log to stderr).
- Guard the `mcp` import with an actionable error: install `ghostgui[agent]`.
Verify the exact low-level API names (Server, stdio_server, types.Tool/TextContent/ImageContent,
create_initialization_options) against the pinned mcp version before writing code.

pyproject.toml: add [project.scripts] ghostgui-mcp = "application.agent_api.mcp_server:main" and
[project.optional-dependencies] agent = ["mcp==<exact validated version>"].

Tests: tests/test_agent_mcp_server.py (skip if mcp missing) using an in-memory client/server session or a
stdio subprocess: list_tools count/names; call get_robot_summary; call render_views → ImageContent;
invalid args → isError; path traversal rejected.

Docs: docs/agent_integration.md (Claude Code `claude mcp add`, .mcp.json, Claude Desktop config, tool table,
approval policies), docs/ai_security.md "External agent control" section, docs/install.md extra,
docs/README.md link, CHANGELOG.md.

Acceptance: test suite, scripts/check_docs.py, scripts/check_architecture.py pass; `pip install -e .[agent]`
then `ghostgui-mcp --help` works; a manual Claude Code session can list tools and render a contact sheet.
```

### 7.3 Phase 3: GUI live-attach

**Model / effort:** `claude-opus-5-5`, effort `xhigh` (threading).

```text
Goal: Let an MCP client drive the *running* GhostGUI so the user watches agent actions live as the
Orange preview, with explicit opt-in and Accept/Discard control.

Read: gui/ai_assistant_controller.py (whole file; esp. :171-200, :621-724, :887-1100), gui/main_window.py
(:1323-1362, :1558-1700, :2727-2800, :3036-3085), gui/ai_frame_capture.py, application/background_jobs.py,
application/agent_api/*, review/03-claude-agent-integration.md §2.3-§2.5, §4.6.

Create:
- application/agent_api/bridge_protocol.py (Qt-free JSON-RPC 2.0 types: auth, tools/call, events/subscribe).
- application/agent_api/bridge_client.py (asyncio TCP client; reads ghostgui_user_data_dir()/agent-bridge.json).
- gui/agent_bridge.py: AgentBridgeServer(QTcpServer on 127.0.0.1:0, 32-byte token, file mode 0600, removed on
  shutdown, single client), MainThreadInvoker(QObject; queued signal carrying (callable, Future)),
  GuiWorkspace(Workspace) adapting RobotGuiMainWindow + AIAssistantController.
Threading rule: working-copy mutations run on the bridge worker thread; committed state, viewer, project,
history and GUI renders run on the Qt thread via MainThreadInvoker with timeout + cancellation.
Live preview: publish_candidate(snapshot) → controller.present_agent_candidate(snapshot) sets the candidate
sampler to detached_document(snapshot) (immutable) and refreshes; never share the mutable working document.

Modify:
- gui/ai_assistant_controller.py: begin_agent_session(owner), present_agent_candidate(snapshot),
  end_agent_session(outcome); accept_candidate under policy "review" returns pending_user_review and shows
  the result card; Accept uses the existing accept() transaction.
- gui/panels/ai_assistant_panel.py, gui/widgets/ai_chat.py: "External agent connected" banner, activity rows from
  AgentEvent, Disconnect button.
- gui/main_window.py: setting agent/bridge_enabled (default False) + menu toggle "Allow external agent control";
  dialog-free invokable load_model/new_project/open_project/save; refuse project/model transitions while a
  candidate is unresolved.
- gui/ai_frame_capture.py: optional camera preset override with save/restore.
- application/agent_api/mcp_server.py: --attach auto|never|required (auto = bridge if reachable else headless).
- gui/help/help_content.py (+ tutorial copy if applicable): "External agent control" text with canonical terms.

Tests: tests/test_agent_bridge.py (QT_QPA_PLATFORM=offscreen): auth required; wrong token rejected; second
client refused; tools/call mutates candidate and the Qt-side sampler receives a snapshot; a user edit to
the committed document during a run makes accept fail stale; Stop/disconnect restores last validated
checkpoint. Extend tests/test_ai_assistant_controller_chat.py.

Docs: docs/agent_integration.md attach section; docs/user_guide.md how-to; docs/ai_security.md bridge threat model.
Acceptance: all three commands pass; manual: start GUI, enable toggle, run Claude Code with ghostgui-mcp
--attach required, ask for a squat; Orange preview updates after each tool; Accept creates one Undo entry.
```

### 7.4 Phase 4: in-app multi-turn agent loop

**Model / effort:** `claude-opus-5-5`, effort `xhigh`.

```text
Goal: Add an opt-in in-app Agent mode: a bounded, provider-neutral multi-turn tool loop over the
Phase-1 catalog, with a production Anthropic adapter (adaptive thinking, effort, prompt caching,
streaming, verbatim block replay, refusal fallback), plus `ghostgui agent "prompt"` headless CLI.

Read: application/ai/agent.py (legacy loop to evolve), application/ai/schemas.py, providers/{base,anthropic,
mock,replay}.py, provider_registry.py, limits.py, gui/ai_assistant_controller.py, gui/panels/ai_assistant_panel.py,
application/launcher.py, review/03-claude-agent-integration.md §4 (all) and §4.5.
Before coding, confirm against the pinned anthropic SDK: messages.stream + get_final_message, thinking
{"type":"adaptive","display":...}, output_config.effort, cache_control on system blocks and top-level,
eager_input_streaming on tools, beta fallbacks ("server-side-fallback-2026-07-01", fallbacks="default").
Opus 5.5 / Sonnet 5.5: no temperature, no forced tool_choice, thinking blocks must be replayed unchanged.

Change:
- schemas.py: ProviderMessage.provider_blocks (opaque, replayed verbatim), ToolResult.content_parts (text+images),
  Usage cache fields, ProviderRequest.effort/thinking_display/cache_breakpoints/stream/fallbacks, StopReason.PAUSED.
- providers/anthropic.py: build/stream requests as above; preserve all content blocks; parse cache usage;
  continue on pause_turn; images inside tool_result; keep existing error normalization and cancellation polling.
- providers/replay.py + mock.py: support provider_blocks and image digests in fingerprints/scripts.
- provider_registry.py: Anthropic models ("claude-opus-5-5","claude-sonnet-5-5","claude-sonnet-5","claude-haiku-4-5");
  capability flags supports_agent_loop, supports_prompt_cache.
- limits.py: AGENT_* budgets (output 32_000, turns 40, tool calls 120, images 48, wall 900 s, cost $5).
- New application/ai/agent_loop.py: MotionAgentLoop(provider, catalog, prompt, limits, policy, events, pricing)
  .run(instruction, model, effort, cancellation_token) -> AgentRunResult; append-only transcript; all tool
  results of one turn in one message; mid-conversation system budget reminder at 80%; best-candidate tracking;
  repeated-identical-failure abort; restore best validated checkpoint on cancel/budget.
- New application/ai/agent_prompts.py (frozen prompt §4.1 + per-model robot card), application/ai/pricing.py.
- New application/agent_api/cli.py and launcher dispatch for `ghostgui agent ...` before any Qt import.
- GUI: Edit/Agent mode switch; Agent runs as one submit_cancellable job (asyncio.run(loop.run)) against an
  in-process GuiWorkspace; AgentEvents → queued signal → activity rows, thinking summaries, budget meter; Stop.
- Gemini: set supports_agent_loop=False unless thought-signature preservation is implemented.

Tests: tests/test_ai_agent_loop.py (MockProvider: parallel results in one message, verbatim thinking replay,
budgets, cancel → best checkpoint, approval policies, pause_turn), extend test_ai_anthropic_provider.py (request
shape: effort, no temperature, cache_control order, stream, fallbacks; block preservation), test_ai_provider_replay.py,
test_ai_provider_registry.py, tests/test_agent_cli.py (headless, MockProvider, auto_accept_and_export → mjlab file).

Docs: docs/ai_assistant.md (Agent mode, presets, budgets, policies, pinned baseline), docs/ai_security.md (opt-in loop,
allowlisted tools; revise one-request statements), docs/architecture.md (:249-255 paragraph), help copy, CHANGELOG.md.
Acceptance: three commands pass; `ghostgui agent --help` works without a display; replay test of the squat
scenario passes offline.
```

### 7.5 Phase 5: vision self-critique and metrics

**Model / effort:** `claude-opus-5-5`, effort `high`.

```text
Goal: Give the agent reliable eyes and numbers: full metric set, camera presets/overlays, optional MuJoCo
rollout check, critique rubric, and registry-owned primitive aliases.

Read: application/agent_api/{metrics,rendering}.py, application/ai/motion_primitives.py (:162-266 gates to
generalize), core/ik/collision.py (:271-320), core/models/registry.py, application/ai/trajectory_planner.py
(:337-398 burpee policy), application/ai/visual_critique.py, docs/motion_math.md.

Implement:
- metrics: foot_sliding (stance = EE height < h_contact and |v| < v_contact; slide = horizontal travel while in stance),
  com_support_margin (quasi-static; CoM xy vs convex hull of support_frames in contact), contact_schedule,
  joint_acceleration, jerk, root_continuity; thresholds from RobotModelInfo with documented defaults.
- rendering: camera presets per model in registry; overlays (CoM marker, contact dots, ground grid) drawn in numpy.
- application/agent_api/rollout.py + tool simulate_rollout: mj_step PD tracking of the reference qpos, gains from
  registry, fall = root height/tilt thresholds; result labelled "heuristic, not hardware safety".
- agent_prompts.py: critique rubric (phase-by-phase checklist driven by contact sheets + metrics).
- Move burpee recognition to RobotModelInfo.primitive_aliases and make the planner policy registry-driven
  (behaviour-preserving; keep existing tests green, add a focused test).
- Populate g1/h2/go2/z1 camera presets, support frames, velocity limits in core/models/registry.py.

Tests: synthetic metric fixtures (known slide, penetration, seam, rotation); rollout on G1 home (no fall) and a
tilted pose (fall); registry alias test; planner regression tests unchanged.
Docs: docs/motion_math.md metric definitions; docs/ai_security.md rollout caveat; docs/agent_integration.md metrics table.
Acceptance: three commands pass.
```

### 7.6 Phase 6: prompt-only mode UX

**Model / effort:** `claude-sonnet-5-5`, effort `high`.

```text
Goal: A "Describe a motion" start experience: one prompt box → the agent creates the project, authors,
iterates, and shows a live Orange preview, ending with a result card (metrics, concessions, Accept/Export).

Read: gui/project_browser.py, gui/panels/ai_assistant_panel.py, gui/widgets/ai_chat.py, gui/tutorial/*,
gui/help/help_content.py, docs/user_guide.md, AGENTS.md terminology rules.
Create gui/panels/agent_prompt_panel.py: prompt box, model preset (Best = Opus 5.5 high, Balanced = Sonnet 5.5
medium), approval preset (Review / Auto-accept), plan checklist driven by AgentEvents, live preview auto-loop,
final card (metrics table, files, concessions, Accept, Export, Open project). Hook "Start from a prompt" into the
project browser. Update tutorial step + help copy + docs/user_guide.md + docs/README.md.
Tests: offscreen UI test with a scripted MockProvider: prompt → run → card → Accept → one history entry.
Acceptance: three commands pass; terminology checker passes.
```

### 7.7 Client configuration

**Claude Code, project scope** (`.mcp.json` in the user's working folder, not
committed to GhostGUI). Use the absolute path to the venv executable when it is
not on `PATH`:

```json
{
  "mcpServers": {
    "ghostgui": {
      "command": "ghostgui-mcp",
      "args": ["--attach", "auto", "--approval", "review", "--output-dir", "~/GhostGUI/agent-exports"],
      "env": {}
    }
  }
}
```

Or from the CLI: `claude mcp add ghostgui -- ghostgui-mcp --attach auto`. Add
`--scope project` to write `.mcp.json`. On headless Linux, add
`"env": {"MUJOCO_GL": "egl"}`. Claude Code limits MCP tool output size
(configurable via `MAX_MCP_OUTPUT_TOKENS`); contact sheets are designed to fit
**(verify current default)**.

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`
on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "ghostgui": {
      "command": "/Users/<you>/ghostgui-venv/bin/ghostgui-mcp",
      "args": ["--attach", "auto", "--approval", "review"]
    }
  }
}
```

**Example end-to-end prompt a user would type** (Claude Code / Desktop with the
server attached, or GhostGUI Agent mode):

```text
Using GhostGUI: design a 6-second Unitree G1 cartwheel-like motion that starts and ends standing at the
home pose, rotating about the forward axis with left hand, right hand, right foot, left foot contacts in order.
Keep ground penetration under 1 cm and Joint Angle speeds within limits. Show me side and front contact
sheets at each phase, fix anything that looks wrong, then loop it twice (12 s total), generate at 50 Hz,
accept it, save the project as "g1-cartwheel", and export an mjlab CSV named "g1-cartwheel-x2".
Report the metrics and any concessions.
```

### 7.8 Optional: Claude Agent SDK wrapper (not the primary path)

If a team wants a Claude-Code-style harness in a script, run the Agent SDK with
`ghostgui-mcp` as its only MCP server. Restrict tools to the GhostGUI MCP tools
(`mcp__ghostgui__<tool>`) and disable built-in Bash/Write/Edit. Confirm the
option names (`mcp_servers`, `allowed_tools` / `disallowed_tools`,
`permission_mode`) against the current `claude-agent-sdk` docs **(verify)**.
This adds a dependency and a second harness for little gain over loop A, so
keep it out of the product.

---

## 8. Risks and Open Questions

| Risk | Mitigation |
| --- | --- |
| Offscreen GL availability (macOS CGL vs Linux EGL/OSMesa; thread affinity) | One dedicated render thread per process; `MUJOCO_GL` documented per platform; fall back to text-only with a warning (existing `AutomaticMotionFrames.unavailable_reason` pattern) |
| Headless and GUI behaviour drift | Phase 1 refactors the GUI to call the same services; parity tests |
| Large qpos authoring inputs truncated mid-stream | `eager_input_streaming` + schema validation before dispatch; delta-from-base-pose format keeps inputs small |
| Root rotations > 180° between anchors interpolate the short way | `author_keyframes` rejects or warns on gaps > 150° and reports `rotation_gaps` |
| Autonomous accept over user work | `review` default; provenance gate unchanged; protected content never auto-accepted; stale-commit guard |
| Cost runaway | Per-run USD budget computed from usage including cache tokens; budget reminder; hard stop restores the best checkpoint |
| Kinematic "cartwheel" mistaken for physically feasible | Mandatory concession text; `simulate_rollout` is labelled heuristic; docs keep the dynamics-out-of-scope statements (`docs/ai_security.md`) |
| Gemini parity in Agent mode | Capability flag gates Agent mode until thought-signature replay is implemented |

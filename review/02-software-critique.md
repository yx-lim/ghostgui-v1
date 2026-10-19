# GhostGUI — Software Critique (principal-engineer / robotics-tooling review)

Reviewed: branch `report-plot` @ `e81734d` (2026-09-25), ~65k tracked lines of Python + 806 lines of C++.
Scope: the whole product. Visual/UI redesign and autonomous-agent integration are covered by other reviews and are touched only briefly here.

**How evidence was gathered**
- I read the code and docs directly.
- I ran the suites on the project `.venv`: Python 3.14.6, mujoco 3.11.0, PySide6 6.11.1, numpy 2.5.2.
- I wrote ad-hoc benchmarks and reproductions in `/tmp`. No tracked file was modified.
- Every `file:line` reference below was checked against this checkout.

---

## 1. Executive summary

**What it is.** GhostGUI is a desktop keyframe editor for MuJoCo robot models: Unitree G1, Go2, H2 and Z1 are bundled, and users can import URDF or MJCF models. The authoring loop:
1. Drag an End Effector gizmo (whole-body damped-least-squares IK), or set Joint Angles directly.
2. Inspect the **Orange preview**.
3. **Commit Keyframe**.
4. **Generate** a uniformly sampled qpos trajectory by per-sample IK.
5. Play it back.
6. Export MuJoCo CSV, DSMS, or mjlab CSV.

It also has transactional project files, Motion Clips (copy/paste/repeat/ping-pong), timeline retiming, and an optional LLM "Motion Assistant" (Gemini or Claude) that stages edits on a detached working copy.

**Who it is for.** Robotics/RL researchers who need short, hand-authored reference motions — mostly for the Unitree G1 — to feed motion-tracking pipelines (mjlab, DSMS).

**Strongest points**
- **Engineering discipline in the non-GUI layers.**
  - Layering is enforced by AST (`scripts/check_architecture.py:14-31`).
  - Saves are transactional with journal recovery (`application/project_manager.py:233-480`).
  - Schema migration and path-escape validation exist (`project_manager.py:81-195`).
  - The qpos contract is strict (`core/robotics/contracts.py`).
  - An installed-wheel smoke test and an Xvfb visual smoke run in CI (`.github/workflows/ci.yml`).
- **Correct robotics fundamentals where they exist.**
  - Interpolation is manifold-aware (`mj_differentiatePos`/`mj_integratePos`, `core/models/model.py:945-956`).
  - Generation uses a null-space posture objective under Cartesian targets (`model.py:676+`).
  - Collision policy is per model, with advisory vs blocking penetration (`core/ik/collision.py:59-155`).
  - URDF import includes deterministic collision-free home-pose repair (`core/models/adapter.py:462-603`).
- **Green, sizeable test suite:** 673 tests pass, 2 skipped, in 179 s.
- **Unusually thorough docs:** 17 pages, CI-validated links and terminology (`scripts/check_docs.py`).
- **The AI path is safe:**
  - It works on a detached working copy with checkpoint rollback.
  - Accept is one atomic undoable transaction (`application/ai/edit_session.py:280-331`).
  - The LLM never writes qpos directly; local code executes a compact spec.

**Five biggest risks**
1. **The motion data model is subtly wrong, and wrong in ways users cannot see.**
   - Every keyframe is stored twice: logical `TargetFrame`s with **Euler RPY** in `ProjectDocument`, plus qpos in a viewer-owned `RobotStateTimeline`.
   - **Merely navigating the timeline inserts invisible qpos keyframes** (`gui/robot_viewer_3d.py:1838-1844` → `core/models/model.py:912-920`). These permanently pin the interpolation.
   - Reproduced: with Keyframes at 0 s and 2 s, clicking t=1 s then editing the 2 s Keyframe leaves the left elbow at 0.80 rad at t=1. The correct value is 0.05 rad.
   - The phantom keyframes are not drawn as timeline markers, because markers come from target frames (`gui/main_window.py:4027-4029`). They are saved to the project (`model.py:964-973`).
2. **Exported motions are not RL-grade.**
   - Interpolation is piecewise-linear in qpos (C0 velocity), with per-segment SLERP.
   - There is no velocity, acceleration or torque check. There is no `mj_step`/`mj_inverse` anywhere in `core/`, `application/` or `gui/` — no dynamics feasibility at all.
   - There are no contact labels and no CoM/ZMP.
   - The same velocity kinks a tracking policy has to fight are baked into every export.
3. **GUI god objects with duplicated state.**
   - `RobotGuiMainWindow` has 178 methods in 4,062 lines. `RobotViewer3D` has 145 methods in 2,706 lines.
   - Time lives in ≥5 places. The generated trajectory lives in 3–4 copies.
   - Undo snapshots are assembled from widget state and deep-copy whole trajectories on every time change.
   - Regressions here are what will slow every future feature.
4. **Development effort has drifted to the AI subsystem while core authoring gaps remain.**
   - 51 of the last 60 commits touch AI. That is about +25.7k lines, against 12.3k lines of production AI code in 35 modules.
   - About 3.9k of those lines are unreachable from the UI (§3.7).
   - Recent work includes a regex-triggered, G1-only "burpee" special case (`application/ai/trajectory_planner.py:346-390`).
5. **Performance and blocking on the UI thread.**
   - **Generate** runs synchronously on the GUI thread (`gui/main_window.py:3974-3999`). Measured: **5.8 s freeze for a 10 s, 100 Hz G1 motion**, so a 60 s clip freezes for about 35 s.
   - Drag IK runs on every mouse event without coalescing.
   - The trajectory polyline is re-sampled on every `paintGL` (`gui/viewers/robot_canvas_3d.py:1025-1045`).
   - AI requests (up to about 3×180 s) share the **single** serialized worker with CSV export and model import (`gui/main_window.py:209, 399`).

**Scorecard (1–10)**

| Dimension | Score | Rationale |
| --- | :---: | --- |
| Feature completeness | 5 | The basics are solid (IK editing, keyframes, retime, clips, project files, 3 export formats). There is no curves, contacts, mirror, physics, retargeting import, batch CLI or dataset view. |
| Motion-authoring power | 4 | One keyframe channel per whole-body pose, linear qpos interpolation, no per-key tangents, no layers. Foot lock is a soft weighted task during drag only. |
| Architecture | 5 | Good core/application contracts and a CI-enforced direction. Undermined by GUI god objects, viewer-owned authoritative state, and G1 leakage into `application/`. |
| Code quality | 6 | Careful validation, dataclasses, explicit errors. No `logging` anywhere, 161 `getattr`/`hasattr`, 22 cross-class private accesses, no linter or type checker. |
| Testing | 7 | 673 green. But 112 tests each construct a full `MainWindow` (57 s alone), about 35% of AI test lines cover hidden paths, and there is no coverage measurement. |
| Docs | 7 | Extensive and CI-checked. Partly stale (§3.7), and heavy with internal phase records instead of tasks a user needs. |
| Usability | 4 | Five interacting time knobs, a mandatory but easily stale Generate step, a silent preview discard on time change, and a Preview Path that validates a motion that will never execute (§4). |
| Performance | 4 | Seconds-long UI freezes on Generate, O(history×trajectory) memory, repaint-time resampling. |
| Extensibility (new robots) | 6 | Registry plus `*.ghostgui.json` profile plus import repair is good. Exports, analytic backend, AI primitive and viewer script are G1-hardwired. |
| AI integration | 5 | Safe, bounded and well tested. Overbuilt (6 pipelines, 1 reachable), demo-overfit, and the temperature feature is dead code. |
| Packaging/install | 5 | Wheel and installed smoke are in CI. But `data-files` installs to `share/`, the installers skip the `[ai]` extra, core dependencies are unpinned, Linux is the only supported OS, and there are no releases or tags. |

---

## 2. Features

### 2.1 Inventory (what actually exists)

| Area | What exists | Where |
| --- | --- | --- |
| Models | G1, Go2, H2 and Z1 registry entries with logical frames, End Effectors, joint groups, home joints and collision policy. URDF/MJCF import with mesh rewriting, a profile JSON, and home-pose repair. | `core/models/registry.py:92-196`, `application/model_importer.py:509+`, `core/models/adapter.py:118-603` |
| Pose editing | World-aligned translate/rotate gizmo. Whole-body DLS IK with joint weights, posture, regularization, soft foot lock and soft root-upright. Joint Angle sliders. Collision-aware drag. | `gui/viewers/transform_gizmo.py`, `gui/robot_viewer_3d.py:1170-1331`, `core/ik/collision.py:341+` |
| Validation | Self/environment collision with advisory and blocking thresholds, joint limits, singularity metrics, Preview Path ghost sampling. | `core/ik/collision.py`, `core/models/model.py:433-447`, `gui/robot_viewer_3d.py:1366-1440` |
| Keyframes | Commit, Delete, Reset, Clear. Keyframe interval auto-advance. | `application/editor_commands.py`, `gui/main_window.py:3504-3973` |
| Timeline ops | Insert Time, Shift, Move Range, Scale Range, Copy/Paste/Paste-Reversed, Repeat (Forward or Ping-pong). | `application/timeline_editing.py`, `application/motion_clipboard.py` |
| Generation | Uniform-dt target sampling (linear or Hermite blend for positions, SLERP for orientation), per-sample hierarchical IK, exact qpos anchors. | `core/trajectory/model.py:324-590`, `application/trajectory_generation.py`, `application/backend_interface.py:513-850` |
| Playback | In-app ghost playback, plus an external `mujoco.viewer` subprocess (kinematic replay only). | `gui/viewers/mujoco_player.py`, `scripts/view_g1_mujoco.py` |
| Import/Export | Qpos CSV, MuJoCo time+qpos CSV, DSMS folder (with speed scaling), mjlab G1 CSV, named backend CSV. Imports for the same formats. | `application/csv_io.py`, `application/trajectory_export_formats.py`, `application/trajectory_import_formats.py` |
| Projects | `.ghostgui` folder, schema v2, transactional save and autosave, recovery, recent-projects list, session log. | `application/project_manager.py` |
| History | 100-deep snapshot undo/redo. | `application/history.py`, `gui/history.py`, `gui/main_window.py:2404-2614` |
| AI | Compact spec planner → local executor (12 op types, G1 burpee primitive), detached session, Orange candidate preview, Accept/Discard, keyring credentials. | `application/ai/*`, `gui/ai_assistant_controller.py` |
| Help | Help content and an interactive tutorial. | `gui/help/`, `gui/tutorial/` |

### 2.2 Gap analysis vs comparable tools

| Capability | GhostGUI | MuJoCo viewer | Cascadeur | Blender + robot add-ons | Isaac Sim/Lab tools | mjlab | Pinocchio/placo/mink editors | GMR / PHC-style retargeting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keyframe authoring on exact robot model | **Yes** | No | Character rigs, not MJCF | Yes (URDF import), no MuJoCo validity | Limited | No | Script-level | No |
| IK with hard limits/constraints (QP) | DLS + clamp, soft locks | No | AutoPosing | IK rigs | Yes (differential IK) | — | **Yes (QP, CoM, limits as constraints)** | Yes (retarget IK) |
| Curve/graph editor, tangents, easing | **No** | No | Yes | Yes | No | No | No | No |
| Layers / additive offsets | No | No | Yes | NLA | No | No | No | No |
| Mirror L/R pose | **No** | No | Yes | Yes | No | No | No | Symmetric maps |
| Contact / foot-plant labels | **No** (soft drag lock only) | No | Fulcrum points | Manual | Sensors | Consumes contacts | Contact tasks | Foot-contact heuristics |
| Physics feasibility | **None** (no `mj_step`/`mj_inverse`) | **Full sim** | AutoPhysics | No | Full sim | Training sim | Inverse dynamics available | PHC: physics imitation |
| CoM / ZMP / support polygon | **No** (no `subtree_com` use) | Optional render | Yes | No | Yes | — | Yes | — |
| Velocity/accel plots, limit checks | **No** | No | Yes | Graph editor | Yes | — | Yes | — |
| Retargeting import (SMPL/BVH/video) | **No** | No | Video→anim (paid) | Add-ons | Yes | No | No | **Core purpose** |
| Batch/headless CLI export | Two standalone converters only | — | — | Python API | Python API | CLI | Script | CLI |
| Dataset management | Single project = single clip | — | Scenes | Files | USD | Motion registry | — | Dataset folders |
| Natural-language editing | **Yes (unique)** | No | No | No | No | No | No | No |

**Positioning conclusion.** GhostGUI cannot out-animate Cascadeur or Blender, and cannot out-simulate Isaac or MuJoCo. Its defensible niche is the **"last mile" editor for robot reference motions**: take a retargeted or hand-blocked clip on the *exact* MJCF, fix contacts, collisions, limits and timing, check feasibility, and batch-export to the trainer's format. Most of what that niche needs is missing. The AI assistant is a differentiator, but only once the underlying motion is trustworthy.

### 2.3 Missing features ranked for RL reference-motion researchers

| # | Feature | Why it matters | Notes for implementation |
| --- | --- | --- | --- |
| 1 | **Smooth interpolation + velocity/acceleration plots + limit checks** | Linear qpos interpolation (`model.py:922-956`) gives step changes in joint velocity at every Keyframe, and tracking rewards punish them. Users cannot see the kinks today. | Cubic or quintic Hermite/Catmull-Rom on the manifold (tangents from neighbours, via `mj_differentiatePos`). Per-joint velocity/accel sparkline panel. Flag samples over `actuator`/registry limits. |
| 2 | **Contact / foot-lock tracks** | Foot skating is the #1 artifact in reference motions. Contact flags are also a direct training signal. | Per End Effector boolean track on the timeline. During Generate, contact frames become high-priority equality tasks (not weighted). Export a `contacts` column or NPZ key. |
| 3 | **Physics feasibility report** | Users discover infeasibility only after hours of training. | `mj_inverse` per sample → required joint torques vs `actuator_forcerange`/`ctrlrange`, root wrench magnitude (unactuated-base residual = "how much magic"), and contact force sign. Optional PD-tracking rollout with `mj_step` reporting divergence time. |
| 4 | **Retargeting import (GMR / SMPL-X / BVH outputs)** | Most RL reference motions start as mocap. GhostGUI as the cleanup tool is the strongest product story. | Start by importing GMR-style per-frame robot qpos (pkl/npz/csv) as a dense qpos track, plus **keyframe reduction** (Douglas–Peucker on qpos error) so it becomes editable. |
| 5 | **Batch / headless CLI** | Datasets are dozens to hundreds of clips. A GUI-only Generate does not scale. | `ghostgui export PROJECT --format mjlab|dsms|mujoco --dt …` reusing `application/` services. Needs Generate to be Qt-free (it already is) and off the GUI thread. |
| 6 | **Mirror pose / mirror clip L↔R** | Halves the authoring work for gait cycles and symmetric skills. | Registry `mirror_map` (joint pairs + sign flips + sagittal axis) in `RobotModelInfo`. Apply to the qpos and target frames. |
| 7 | **Loop/cycle tools** | Locomotion clips must be cyclic, with continuous root velocity. | "Make cyclic": match the last Keyframe to the first, modulo root x/y/yaw offset. Seam velocity matching. Builds on `motion_clipboard`. |
| 8 | **CoM / support-polygon / ZMP overlay** | Quick static-balance sanity check for humanoids. | `data.subtree_com[root]` projected onto the ground, plus a convex hull of the contact End Effectors. Cheap to render. |
| 9 | **Curve / graph editor** | Precise timing and easing beyond "Scale Time Range". | Large effort. Do it after #1 creates tangent data worth editing. |
| 10 | **IK constraint layering (true priorities)** | `solve_weighted_tasks` admits it "intentionally solves one weighted stack" (`model.py:562-566`). Foot lock is `required=False` (`gui/robot_viewer_3d.py:1195-1199`). | Swap in a QP IK (e.g. `mink` on MuJoCo) with hard joint limits and equality constraints for locked contacts. |
| 11 | **Dataset management** | Clip list, tags, robot, duration, validation status, export state. | A project-of-clips index. Later work. |
| 12 | **NPZ export with velocities/body poses (mjlab-native)** | The mjlab CSV still needs mjlab's external converter (`docs/data_formats.md`). | Emit `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` and so on directly from MuJoCo FK. |

---

## 3. Design & architecture

### 3.1 Layering

**What is good**
- `core <- application <- gui` is enforced by AST import checks (`scripts/check_architecture.py:14-31`) and passes.
- `core/` has zero Qt.
- Commands and events (`application/editor_controller.py:48-79`) are a sound kernel.

**Problems**
- **`application/` imports Qt.**
  - Affected files: `application/background_jobs.py:11` (PySide6 `QObject`/`Signal`), `application/file_dialog_helper.py`, `application/launcher.py`.
  - The checker only forbids `application → gui` (`check_architecture.py:14-18`), not `application → PySide6`.
  - This blocks the headless CLI (#5 above) and makes the "Qt-free application" claim untrue.
- **Model-specific behavior leaks out of `core/models/registry.py`.** This contradicts AGENTS.md and the architecture doc's statement that "G1-specific group names or joints do not live in the AI application modules" (`docs/architecture.md`, last section).

  | Leak | Location |
  | --- | --- |
  | G1 29-joint order in the generation backend | `application/backend_interface.py:48-82` (`LAB_JOINT_NAMES`), `:96` (`MODEL_PATH = g1_29dof.xml`), `:99-105` (`IK_TASKS` with `robot/left_palm`…) |
  | Duplicate G1 joint order for mjlab | `application/trajectory_export_formats.py:19-52` |
  | Hard-coded G1 crouch Joint Angles in AI | `application/ai/motion_primitives.py:118-135` (`_g1_crouch`) |
  | Regex rewrite of "burpee" prompts | `application/ai/trajectory_planner.py:346-390` |
  | G1-ish defaults in core | `core/trajectory/model.py:96-110` (`phase="crouch"`, `z=0.9`), `core/models/model.py:22` (`DEFAULT_MODEL_PATH = g1_29dof.xml`) |
  | G1 viewer reused for all models | `scripts/view_g1_mujoco.py` (launched for every model by `gui/viewers/mujoco_player.py:40`) |

### 3.2 GUI god objects

`gui/main_window.py` is one class (`RobotGuiMainWindow`, lines 119–4062) with **178 methods**:

| Method | Lines | Length | Notes |
| --- | --- | --- | --- |
| `__init__` | 136–424 | 289 | Includes a *synchronous* initial model load with g1 fallback at 208–242 |
| `build_menu_bar` | 447–770 | 324 | |
| `build_workflow_toolbar` | 783–923 | 141 | |
| `restore_project` | 1918–2025 | 108 | |
| `restore_history_snapshot` | 2474–2558 | 85 | |

`gui/robot_viewer_3d.py` has **145 methods** and mixes six roles:
- UI construction: `_build_ui` 275–495, 221 lines.
- IK/collision interaction.
- Timeline state.
- File dialogs and export assembly (1884–2420).
- Generated-trajectory model (2464–2507).
- Playback and AI candidate preview.

`gui/controls.py:100-465` is a single 366-line `build_ui`.

**Business logic in Qt callbacks** (each should be an application service):
- `on_generate_trajectory` runs sampling, batch IK and a CSV write inline (`main_window.py:3974-3999`).
- `on_delete_timeslice_requested` mutates `viewer_3d.state_timeline` directly, bypassing `EditorController` (`main_window.py:3588-3619`).
- The same five-subsystem "clear generated motion" sequence is copy-pasted. `backend_interface.clear_last_solution()` alone appears at `main_window.py:1654, 1938, 2507, 3492, 3887`.
- `restore_project` clears generated state *before* validations that can still fail (`main_window.py:1934-1987`). A failed open leaves the window half-restored.
- `activate_model_session` rebinds attributes with `setattr(self, name, value)` (`main_window.py:3047-3050`).

**The "RViz-inspired" visualization runtime is ceremony.**
- `application/visualization/` plus `gui/visualization/` add up to 691 lines of lifecycle machinery.
- That machinery registers five callbacks, which call straight back into the window's private methods (`gui/visualization/runtime.py:46-66` → `window._render_robot_scene`, `window._render_timeline_markers`, `window._refresh_status_panel`).
- It adds indirection without decoupling anything.

### 3.3 State management — no single source of truth

| Concept | Copies | Evidence |
| --- | --- | --- |
| Keyframes | Target frames in `ProjectDocument.trajectory` **and** qpos in `viewer.state_timeline` (attached, not owned, by the document) | `main_window.py:121-125, 262`, `application/project_document.py:31,56` |
| Robot pose | `committed_state`, `preview_state`, `playback_state`, plus the `robot_state` alias | `robot_viewer_3d.py:149-154` |
| Generated trajectory | `viewer.robot_trajectory`/`_times`, `backend.last_solution`, the playback-cache CSV, and `ghost_trajectory` | `robot_viewer_3d.py:174-175, 2605`, `backend_interface.py:478` |
| Current time | `document.current_time`, `viewer.current_time`, `viewer.display_time`, `controls.time_slider`, the timeslice slider/spinbox, and the MuJoCo player index | `main_window.py:3520-3537`, `robot_viewer_3d.py:1835-1837` |
| AI candidate | Reuses `preview_state`/`preview_active`, so the Orange preview means two different things | `robot_viewer_3d.py:1765-1767, 1452` |

The architecture doc admits that "`RobotViewer3D` currently owns distinct committed, preview, and playback MuJoCo states". The authoritative qpos timeline therefore lives in a widget.

**Consequences**
- The phantom-keyframe bug (§1 risk 1). `set_current_time` calls `state_timeline.set_state(...)` or `ensure_state(...)` on every navigation (`robot_viewer_3d.py:1838-1844`, `1797-1803`).
- It is asserted as intended behaviour in `tests/test_robot_viewer_timeline.py:125-126`.

**Orientation is stored as Euler RPY** in the persisted `TargetFrame` (`core/trajectory/model.py:96-110`). It is converted to quaternions for every solve (`core/ik/solver.py:100-102, 147-151`). This means:
- Gimbal-lock singularities at pitch ±90°, which happen when a humanoid goes prone.
- Lossy round-trips.
- Ambiguous interpolation endpoints.

Quaternions should be the stored form.

### 3.4 Undo / history

`HistoryStack` itself is clean (`application/history.py`), but what it stores is wrong:
- **The snapshot is built from widgets** (`gui/main_window.py:2404-2454`). It includes `controls.selected_row()`, the `show_ghosts` checkbox, ghost overlays and preview qpos. View state is mixed into document history.
- **Every snapshot deep-copies** the full qpos timeline, the generated trajectory and the ghost trajectory (`main_window.py:2415-2449`).
- **A snapshot is taken on every time change**: `on_time_changed` → `_refresh_history_baseline` → `capture_history_snapshot` (`main_window.py:3537, 2560-2563`).
- Rough cost: a 60 s, 100 Hz G1 clip (nq=36) is about 1.7 MB per array. Two arrays × 100 entries is about 350 MB of history. Every scrub release also pays a full copy.
- **Restore reaches into internals**: `state_timeline.states.clear()` (`main_window.py:2525`), `controls._suppress_pose_changed` (`2458, 1992`).

**Fix.** Make history document-level: `ProjectDocument` holds both target frames and the qpos timeline as immutable/persistent structures, or uses command-based undo. View state is excluded. No snapshot on navigation.

### 3.5 Threading / background jobs

**The worker itself is well engineered.** `SerializedBackgroundJobs` is a single worker thread with cancellation tokens and late-callback suppression (`application/background_jobs.py:103-341`).

**Problems**
- **Generate is synchronous on the GUI thread** (`main_window.py:3974-3999`). Measured at **5.79 s for 10 s @ 100 Hz** with 11 Keyframes on Apple Silicon (benchmark in `/tmp/ghostbench/bench.py`). There is no progress bar and no cancel.
- **Only one worker for everything.** The AI controller is given the same instance (`main_window.py:209, 399`). A planner request can take up to about 3×180 s (`application/ai/trajectory_planner.py:71, 171-240`), and during that time CSV export, model import and trajectory-import FK queue behind it.
- **Drag IK** (`CollisionAwareIKSolver.solve_drag`) runs per mouse-move event with no coalescing (`robot_viewer_3d.py:1224-1331, ~1263`).
- **Export preprocessing runs before the job is submitted**: snapshot, resample and collision checks (`robot_viewer_3d.py:2117-2122, 2153-2157, 2389`).
- **The initial model load is synchronous** (`main_window.py:208`). Later switches use `ModelLoadThread` (`main_window.py:2760`).

### 3.6 The C++ backend

`backend/robot_sim.{hpp,cpp}` and `backend/bindings.cpp` (806 lines):
- The header describes it as *"Takes pelvis TargetFrame… Keeps all joints fixed"* (`robot_sim.hpp:6-24`).
- There is **no build system**: no CMake, setup.py ext or pybind build in `pyproject.toml`.
- `BackendInterface` only selects it if a `robot_backend` module happens to be importable, and it is labelled "limited capabilities" (`backend_interface.py:39-46, 904-924, 975`).

The G1 "analytic" Python fallback (`PythonTrajectoryBackend`, `backend_interface.py:199-511`) is the same vestige. Together they are about 1.1k lines of dead or misleading code, plus a `FallbackPolicy` matrix to explain them. **Delete both**; MuJoCo IK is the only real backend.

### 3.7 AI subsystem — overbuilt and duplicated

12,262 lines across 35 modules, plus 1,842 lines of GUI. There are six pipelines, **only one of which a user can trigger**:

| Pipeline | Entry | Reachable? |
| --- | --- | --- |
| A. Compact spec: `TrajectoryPlanner` → `TrajectorySpecExecutor` → `trajectory_operations` | `application/ai/trajectory_workflow.py:71-172` | **Yes (default, hard-coded)** |
| B. Text/semantic: `TextMotionWorkflow` → `PlanExecutor` + `ToolRegistry` + `MotionRepairPlanner` | `application/ai/text_planner.py:257` | No. `_run_text_motion` (`gui/ai_assistant_controller.py:765-785`) is never called. |
| C. Tool-calling agent loop `GhostGUIAgent` | `application/ai/agent.py:116` | No. Only `provider_comparison.py` and the tests use it. |
| D–F. Visual refine / verify / critique | `visual_refinement.py:321`, `visual_critique.py:108` | Buttons are created then `hide()`'d and never laid out (`gui/panels/ai_assistant_panel.py:124-159`) |

**Duplication**
- Five planners.
- Seven separate `asyncio.wait_for` wrappers with inconsistent timeouts (180 s vs 60 s).
- `_raise_if_cancelled` copy-pasted 8 times.
- Two operation vocabularies: 12 `TrajectoryOperationType`s vs 9 semantic tools. The live path still imports `retime_segment` from the legacy `semantic_tools` (`trajectory_operations.py:34`).
- Two context builders (`context.py:101, 251`).

**Dead features**
- Commit `20f6391` "retain legacy semantic workflow" changes no files.
- Planner temperature (commit `fc933cc`) never activates. Anthropic `temperature_capable_models` defaults to empty (`providers/anthropic.py:63`), and the registry factory never passes any (`provider_registry.py:125-128`).
- Gemini's retry loop is configured for one attempt.

**Inconsistent transport behaviour.** Anthropic leaves SDK defaults in place (2 retries, 600 s timeout) inside a 180 s outer `wait_for`. Gemini does 1 attempt.

**Demo overfitting**
- A G1-only `burpee` primitive with hard-coded crouch angles (`motion_primitives.py:118-135`).
- A regex that rewrites provider output when the prompt says "burpee" (`trajectory_planner.py:346-390`).

**Validation gaps**
- `validate_motion` treats collisions as **warnings** (`application/ai/motion_services.py:472-485`).
- It checks only Keyframe poses, not interpolated samples, and has no velocity limits.

**Docs drift.** `docs/architecture.md` still describes `PlanExecutor`, text planning and visual refine as live paths.

**Verdict.** Pipeline A, at about 2.9k lines, is a reasonable vertical slice. Delete B, C and the hidden D–F buttons, about 3.9k production lines plus about 3.5k test lines. Move the G1 recipe into the registry or model layer as data. Unify the timeout/cancel helpers.

### 3.8 Data formats & versioning

**Good**
- Project schema v2 with migration, and refusal of newer versions (`project_manager.py:81-150`).
- NPZ with `schema_version` (`model.py:964-985`).
- Path containment.

**Weak**
- **Keyframe identity is a timestamp rounded to 1e-6** (`model.py:888-889`). AI provenance explicitly relies on a "timestamp-backed MVP resolver" (`docs/architecture.md`), so retiming invalidates identity.
- **Export source changes silently.** CSV/DSMS/mjlab export "the generated trajectory, or the editable qpos timeline when no generated trajectory is active" (`docs/data_formats.md`). Because edits clear the generated motion, the same button exports different motion depending on whether **Generate** was pressed.
- **Export files carry no metadata**: robot, nq order hash, dt, source project or GhostGUI version. Consumers cannot verify compatibility.
- **mjlab export is G1-only, and its output is a CSV that mjlab must convert again.**

### 3.9 Model registry extensibility

**Good.**
- `RobotModelInfo` is a data-only dataclass (`registry.py:64-89`).
- Imports can carry a `*.ghostgui.json` profile with frames, End Effectors, groups and contact tolerances (`docs/adding_models.md`).
- Ball joints are rejected explicitly with a clear message (`model.py:231-240`).

**Missing**
- No `mirror_map`.
- No contact End-Effector semantics beyond names.
- No actuator/velocity limits surfaced.
- No per-model export adapters.
- Adding a built-in robot still requires a code edit, because there is no plugin path for registry entries outside the checkout.

---

## 4. Workflow — the core authoring loop, step by step

| Step | What the user does | Friction found |
| --- | --- | --- |
| 1. Choose model and target frame | Robot menu, then target combo or double-click | Fine. "Advanced target" exposes raw bodies and sites. |
| 2. Edit | Drag the gizmo or move Joint Angle sliders, producing the **Orange preview** | The gizmo is world-aligned only (README "Project Status"). Foot lock and root-upright are *soft* (`robot_viewer_3d.py:1188-1212`), so planted feet drift during a large drag. |
| 3. **Preview Path** (optional) | Ghost samples, collisions in red | **Conceptually wrong.** It interpolates *committed pose at t* → *preview pose at t* (`robot_viewer_3d.py:1424-1430, 1366-1373`). It morphs the old version of this Keyframe into the new one, a motion the robot will never execute. The actual risky motion is *previous Keyframe → this Keyframe → next Keyframe*. Users will assume it validated the motion. |
| 4. **Commit Keyframe** | Writes target frames + qpos, auto-advances by the **Keyframe interval** | Changing time **silently discards** an uncommitted preview (`docs/concepts.md` "Orange Preview"), with no prompt and no undo. Auto-advance also lands you on a time where navigation creates a phantom qpos Keyframe (§3.3). |
| 5. Repeat | Scrub, edit, commit | Every scrub release snapshots history (§3.4) and may insert hidden Keyframes. |
| 6. **Generate** | Uniform-dt sampling, IK per sample | Blocks the UI for seconds (§3.5). Fails unless **every** Keyframe lies on the **Export interval** grid (`application/trajectory_generation.py:44-53`). Any later edit clears the result (`docs/concepts.md` "Timeline Retiming"), so Generate must be pressed again. |
| 7. Playback | Ghost playback in app, or external MuJoCo viewer | Two playback paths. The external one is kinematic replay only, with no physics. |
| 8. Export | File → Export → format | The source silently switches between generated and editable (§3.8). Five time-related knobs interact: **Keyframe interval**, **Export interval**, Playback speed, DSMS motion speed, Scale Time Range. |

**Is Generate necessary?**
- Not as a user-visible step. The editable qpos timeline is already a complete, playable, exportable trajectory: the AI candidate preview proves it ("Generate is not required", `docs/ai_assistant.md`), and exports already fall back to it.
- Generate adds one thing: End Effector targets become primary IK constraints between Keyframes, so hands and feet follow their interpolated Cartesian paths instead of the joint-space path.
- That is a **solver option on the motion**, not a mode. It should:
  - run automatically in the background on edit, per affected segment (Keyframe-bounded invalidation);
  - show per-segment status (solved / stale / failed with max IK error);
  - make export always use the solved path when it is current, and otherwise block with a clear choice ("export joint-space interpolation" vs "solve now").

**Preview / committed / generated states.** Three robots with three meanings, plus the Orange preview reused for AI candidates, is too much for users. "Everything is live" editors (Blender auto-key, Cascadeur) keep one pose per time and make the curve the truth.

The explicit commit has one real justification: blocking penetration must not be saved. That can be kept as a **validity gate on auto-key** — the pose turns red and refuses to key — rather than as a separate state the user must manage.

Recommended model:
- One authoritative motion document.
- Edits at a Keyframe time modify it immediately (undoable).
- Edits between Keyframes create a Keyframe, with the option shown in the UI.
- Orange is reserved for *proposals* (AI candidates, failed or blocked solves).

---

## 5. Usability (brief — visuals are covered by the UI review)

- **Install.**
  - Linux scripts use `apt` + `.venv` + `pip install -e .` (`scripts/install_linux.sh`). macOS and Windows are "experimental".
  - **The installers never install the `[ai]` extra**, so the advertised Motion Assistant fails until the user discovers `pip install -e '.[ai]'`.
  - Core dependencies are unpinned (`pyproject.toml` `dependencies`), while AI dependencies are pinned exactly (`anthropic==1.4.0`). A new mujoco or PySide6 major release can break a fresh install.
- **Running the tests.**
  - The documented command `python3 -m unittest discover -s tests -v` **fails outside the venv**. With system/miniconda Python: 331 tests ran, **28 import errors** (`No module named 'mujoco'`/`'PySide6'`), 37 skipped.
  - The docs should say "activate `.venv` first", or `scripts/run_test_suite.py` should be the documented entry point.
- **First run.** The initial model load is synchronous. If it fails, the app silently falls back to G1 (`main_window.py:221-242`).
- **Model import.** Good: staging, MuJoCo validation and home repair. But imports go into the checkout's `models/` by default, and a missing mesh prompts for a folder at run time.
- **Errors.**
  - Everything goes to a status label or Details panel. There is **no `logging` usage anywhere** in `core/`, `application/` or `gui/` (0 hits for `import logging`/`getLogger`).
  - Field failures cannot be diagnosed after the fact. The project "session log" records only user events (`project_manager.py:924-956`).
- **Big trajectories.**
  - Generate freezes, as above.
  - `refresh_display` rebuilds the whole keyframe `QTableWidget` on every pose or time change (`gui/controls.py:758-782`).
  - Imported dense trajectories add a marker per sample.
  - `TimesliceSlider.paintEvent` sorts markers every paint (`gui/widgets/timeline.py:47`).
- **Discoverability.**
  - Timeline operations sit in menus with long, legalistic help text (`gui/help/help_content.py:53`).
  - The Export-interval alignment error appears only at Generate time.

---

## 6. Code quality & reliability

### 6.1 Test results (actual)

| Command | Interpreter | Result |
| --- | --- | --- |
| `python3 -m unittest discover -s tests` | system miniconda 3.11 (no deps) | **Ran 331, FAILED (errors=28, skipped=37)** — every error is a missing `mujoco`/`PySide6` import |
| `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests` | `.venv` 3.14.6 | **Ran 673 tests in 178.8 s — OK (skipped=2)** |
| `.venv/bin/python scripts/check_docs.py` | | `Documentation validation passed.` |
| `.venv/bin/python scripts/check_architecture.py` | | `Architecture validation passed.` |

**Slowest modules** (run in isolation):

| Module | Time | Notes |
| --- | --- | --- |
| `test_robot_viewer_timeline` | 57.0 s | 112 tests in one 4,356-line class; each `setUp` builds a full `RobotGuiMainWindow` (`tests/test_robot_viewer_timeline.py:74-93`) |
| `test_robot_model_adapter` | 8.9 s | |
| `test_advanced_ik` | 5.4 s | |
| `test_pose_matching_contracts` | 3.5 s | |

All 68 modules pass individually. Offscreen runs emit "QOpenGLWidget is not supported" noise, so GL paths are untested outside the Xvfb CI job.

**Test-suite smells**
- About 35% of AI test lines target unreachable pipelines (§3.7).
- The GUI test file is coupled to `scripts/view_g1_mujoco.py` (`tests/test_robot_viewer_timeline.py:66`).
- Phantom-keyframe behaviour is *locked in* by a test (`:125-126`).
- No coverage measurement.
- No property tests for the interpolation and retime invariants, despite heavy invariant logic in `timeline_editing.py` and `motion_clipboard.py`.

### 6.2 Smells with locations

| Smell | Count / evidence |
| --- | --- |
| `except Exception` | application 33 (21 non-AI, 12 AI), gui 29 (14 in `gui/ai_assistant_controller.py`), core 1. Examples: `gui/main_window.py:1790` silently returns `False`; `application/model_importer.py:566`; `application/project_manager.py:228, 332`. Some are justified (transaction rollback, event isolation). |
| `getattr`/`hasattr` duck typing | application 91, gui 72, core 17. Includes init-order guards like `hasattr(self, "main_splitter")` (`main_window.py:951-1014`) and `getattr(self.viewer_3d, "robot_trajectory", [])` on its own field (`main_window.py:3667`). |
| Private access across classes | 22 in gui. Examples: `controls._suppress_pose_changed` (`main_window.py:2458`), `state_timeline._interpolate` (`robot_viewer_3d.py:1373, 1565`), `viewer._trajectory_export_snapshot` (`main_window.py:4005`), `host._refresh_history_baseline` (`gui/ai_assistant_controller.py:188`), `window._render_*` (`gui/visualization/runtime.py:47-53`). |
| "Compatibility/legacy" shims | 39 in gui. Example: `gui.controls` re-exports `gui/widgets/trajectory_controls.py`. There is no external caller to be compatible with — this is a v0.1.0 app. |
| Dead code | `backend/*.cpp` (no build); `PythonTrajectoryBackend` G1 analytic path (`backend_interface.py:199-511`); `FootLockTask`/`RootPoseTask` are bare priority aliases (`core/ik/tasks.py:142-149`); `JointRegularizationTask(PostureTask): pass` (`tasks.py:169-171`); AI pipelines B–F; temperature plumbing. |
| Duplication | G1 joint list ×2 (`backend_interface.py:48`, `trajectory_export_formats.py:19`); preview-reset sequence 7× in viewer plus 2× in window; four near-identical loaders (`robot_viewer_3d.py:2019-2078`); DSMS/mjlab save paths (`robot_viewer_3d.py:2117-2180`); `_raise_if_cancelled` ×8 in AI. |
| Hot-path inefficiency | `RobotStateTimeline.sample_state` sorts all keys per call (`model.py:929`); `_attach_qpos_references` is O(samples × anchors) with repeated `get_state` (`trajectory_generation.py:34-66`); `draw_trajectory` resamples per paint (`robot_canvas_3d.py:1025-1045`); immediate-mode GL (`glBegin`). |
| Magic numbers | ×100 time quantization repeated (`main_window.py:3553`, `robot_viewer_3d.py:681-801`, `gui/widgets/timeline.py:93`) alongside a 1e-6 `time_key` (`model.py:888-889`), giving two different time resolutions. |
| Typing / tooling | No ruff, flake8, mypy or pyright anywhere in `pyproject.toml`/CI. CI's only static gate is `compileall` (`ci.yml`). |
| Logging | None (see §5). |
| Packaging | `[tool.setuptools.data-files]` installs models, meshes, theme and docs to `<prefix>/share/ghostgui` (`pyproject.toml`). That is deprecated-style and breaks under `--user`/zipapps; use package data plus `importlib.resources`. The resource wheel is 63.7 MB (`docs/reliability.md`). Version 0.1.0 with no tag or release (`CHANGELOG.md`). |
| Repo hygiene | Untracked `build/`, `ghostgui.egg-info/` and `wheelhouse/` in the working tree are git-ignored (fine). `docs/reliability.md` "Phase Verification Record" and `docs/testing.md` request-count history are internal process logs in public docs, which runs against AGENTS.md "Do not commit one-off diagnosis notes". |

### 6.3 Reliability positives worth keeping

- Transactional project save with fsync and journal recovery (`project_manager.py:197-480`).
- Context-bound OpenGL cleanup and render coalescing (`docs/architecture.md` "Runtime And Resource Lifecycles").
- An isolated CI environment (`scripts/run_test_suite.py`), wheel inspection, and installed-package smoke.
- Explicit backend degradation reporting (`BackendSelection.degraded`).

---

## 7. Prioritized recommendations (top 15)

| # | Item | Impact | Effort | Risk | Files |
| --- | --- | --- | :---: | --- | --- |
| 1 | **Stop navigation from creating qpos Keyframes.** Sample without inserting; insert only on commit or edit. Migrate existing projects by dropping qpos-only states that equal the interpolation. | Correctness: no hidden pinning | S | Low–Med: changes a tested behaviour (`test_robot_viewer_timeline.py:125`) | `gui/robot_viewer_3d.py:1797-1845`, `core/models/model.py:912-920`, `gui/main_window.py:3520-3537` |
| 2 | **Background, cancellable Generate with progress.** Keep the editable path usable during it. | Removes multi-second freezes | S–M | Low | `gui/main_window.py:3974-3999`, `application/trajectory_generation.py`, `application/background_jobs.py` |
| 3 | **Smooth qpos interpolation plus a motion-quality report** (per-joint velocity/acceleration, limit violations, collision on *interpolated* samples). | RL-grade output; makes defects visible | M | Med: changes exported numbers, so needs a flag and docs | `core/models/model.py:922-956`, new `application/motion_quality.py`, `gui/panels/` |
| 4 | **Physics feasibility check** via `mj_inverse` (torque vs actuator limits, root residual wrench), plus an optional PD rollout. | Catches infeasible motion before training | M | Low (read-only analysis) | new `core/dynamics/feasibility.py`, `application/`, `registry.py` (limits) |
| 5 | **Contact tracks** per End Effector. Enforced in Generate as high-priority tasks; exported as labels. | Removes foot skating; adds a training signal | L | Med: touches the project schema → v3 migration | `core/trajectory/model.py`, `application/project_manager.py`, `application/trajectory_generation.py`, `core/ik/*`, exports |
| 6 | **Prune the AI subsystem** to pipeline A. Delete B, C and D–F plus their tests. Remove the burpee regex. Move the G1 crouch into registry data. Unify the timeout/cancel helpers. | −7k lines; less maintenance; honest docs | M | Low (unreachable code) | `application/ai/{agent,text_planner,repair_planner,plan_executor,motion_plan,provider_comparison,visual_*}.py`, `gui/panels/ai_assistant_panel.py:124-159`, `gui/ai_assistant_controller.py:208-400,487-619,765-785`, `docs/architecture.md` |
| 7 | **Remove G1 leakage and dead backends.** Delete `backend/` C++ and `PythonTrajectoryBackend`. Derive the mjlab joint order from a registry `export_profiles` entry. | Real extensibility; −1.1k lines | S–M | Low–Med: export contract, so needs tests and docs | `application/backend_interface.py`, `application/trajectory_export_formats.py:19-52`, `core/models/registry.py`, `backend/` |
| 8 | **Document-level history.** No snapshot on time change. Exclude view state. Store the qpos timeline in the document. | Memory and latency; fewer restore bugs | M | Med | `gui/history.py`, `gui/main_window.py:2404-2614`, `application/project_document.py`, `application/history.py` |
| 9 | **Single motion document** owning target frames (quaternions) and the qpos timeline. The viewer becomes a pure view. Time lives in one place. | Fixes the root of most state bugs | L | High (wide refactor) | `application/project_document.py`, `gui/robot_viewer_3d.py`, `gui/main_window.py`, `core/trajectory/model.py` |
| 10 | **Mirror pose / mirror range** via registry `mirror_map`. | Halves symmetric authoring | S–M | Low | `core/models/registry.py`, `application/motion_clipboard.py`, Timeline menu, help |
| 11 | **Headless CLI** (`ghostgui-cli generate/export/validate`) for batch datasets. Make `application/` Qt-free: move `background_jobs.py` and `file_dialog_helper.py` to `gui/`, and extend `check_architecture.py`. | Dataset-scale workflows | M | Low | `application/launcher.py`, new `application/cli.py`, `scripts/check_architecture.py`, `pyproject.toml` |
| 12 | **Fix Preview Path semantics.** Validate prev Keyframe → preview → next Keyframe at real timing. | The validation users rely on becomes real | S | Low (label unchanged) | `gui/robot_viewer_3d.py:1366-1440`, `docs/concepts.md`, help/tutorial |
| 13 | **Retargeted-motion import** (dense qpos npz/pkl, GMR-style) plus Keyframe reduction. | Unlocks the mocap-cleanup niche | M–L | Low–Med | `application/trajectory_import_formats.py`, new `application/keyframe_reduction.py` |
| 14 | **Hot-path performance.** Cache the line polyline by document revision, update the table incrementally, coalesce drag IK to one solve per frame, use bisect in `sample_state`, and add a single worker for AI separate from I/O. | Responsiveness | S–M | Low | `gui/viewers/robot_canvas_3d.py:1025`, `gui/controls.py:758`, `gui/robot_viewer_3d.py:1224`, `core/models/model.py:922`, `gui/main_window.py:209,399` |
| 15 | **Tooling and packaging.** Ruff and pyright (basic) in CI; `logging` with a rotating file in the config dir; installers offer `[ai]`; compatible-release pins for core deps; package data via `importlib.resources`; a shared-window fixture for GUI tests. | Fewer regressions; diagnosable field failures | S–M | Low | `pyproject.toml`, `.github/workflows/ci.yml`, `scripts/install_*`, `tests/test_robot_viewer_timeline.py` |

**Suggested order:** 1 → 2 → 12 → 6 → 7 → 14 → 3 → 8 → 4 → 10 → 11 → 5 → 13 → 9. Items 15a (logging, lint) can run in parallel at any time.

---

## 8. One-shot implementation prompts

Each prompt below is self-contained and can be pasted into Claude Code from the repo root.

### Prompt 1 — Stop timeline navigation from inserting hidden qpos Keyframes
*Recommended: Opus, high effort (subtle behavioural contract).*

```text
Goal: In GhostGUI (/Users/yitxuenglim/Downloads/ghostgui-v1), navigating the timeline must never add a qpos
state to RobotStateTimeline. Today RobotViewer3D.set_current_time (gui/robot_viewer_3d.py:1829-1852) calls
state_timeline.set_state(...) with a generated sample or ensure_keyframe_at_current_time() -> 
RobotStateTimeline.ensure_state (core/models/model.py:912-920). These hidden qpos-only states are not shown as
timeline markers (markers come from target frames, gui/main_window.py refresh_display) but they pin
interpolation: with Keyframes at 0 s and 2 s, visiting t=1 then editing the 2 s Keyframe leaves t=1 at the old midpoint.

Do:
1. Make navigation read-only: use sample_state() for display; only write to the timeline on Commit Keyframe,
   explicit Joint Angle/End Effector commits, Reset at time, timeline ops, imports and AI Accept.
   Audit every caller of ensure_state/ensure_keyframe_at_current_time/update_current_keyframe_from_robot_state
   and of set_current_time (grep gui/ application/) and decide per caller.
2. Ensure "edit at a non-Keyframe time" still works: the Orange preview starts from the sampled pose and the
   Keyframe is created on Commit Keyframe.
3. Add a project-load cleanup: on load, drop qpos states that have no logical target frames at the same time
   AND equal the interpolation of their neighbours within 1e-9 (these are the legacy phantom states). Do not
   drop qpos-only states that differ (they may be imported dense motion). Keep project schema v2 (no schema bump).
4. Update tests/test_robot_viewer_timeline.py (the assertion near line 125 that a keyframe exists after
   set_current_time) to the new contract and add a regression test reproducing the 0 s/2 s/1 s scenario above,
   plus a test that scrubbing does not change state_timeline.times().
5. Update docs/concepts.md ("Committed State": "Moving to a time without an existing keyframe creates an editable
   state…") and gui/help + gui/tutorial copy if they describe this behaviour.

Acceptance: scrubbing/clicking any time leaves state_timeline.times() unchanged; Commit Keyframe still creates
both target frames and qpos at that time; all tests pass.
Run: python3 -m unittest discover -s tests -v (inside .venv, QT_QPA_PLATFORM=offscreen) and python3 scripts/check_docs.py
and python3 scripts/check_architecture.py.
Constraints (AGENTS.md): preserve unrelated working-tree changes; keep model-specific behaviour in
core/models/registry.py; do not modify robot assets, example CSV data, project fixtures, cache versions or export
contracts without focused tests and docs; use canonical terms (Keyframe, Commit Keyframe, Keyframe interval,
Orange preview, Preview Path, End Effector, Joint Angles); update help/tutorial copy for any label change.
```

### Prompt 2 — Run Generate in the background with progress and cancel
*Recommended: Sonnet, medium effort.*

```text
Goal: GhostGUI's Generate (gui/main_window.py:3974-3999, on_generate_trajectory) runs
application.trajectory_generation.generate_trajectory_status synchronously on the GUI thread: ~5.8 s for a
10 s/100 Hz G1 motion. Move it to background work with progress and cancellation.

Do:
1. In application/trajectory_generation.py add an optional progress callback and a cancellation token parameter
   (CancellationToken from application/background_jobs.py) and thread them into the per-sample loop in
   application/backend_interface.py MujocoIKBackend.solve_grouped_trajectory (check token every N samples,
   report fraction). Keep the function Qt-free and keep existing call signatures working (new params keyword-only, optional).
2. Work on copies: snapshot Trajectory frames and the RobotStateTimeline states and create a fresh MuJoCo data
   (MjData) for the worker so the GUI thread can keep rendering; never touch viewer state from the worker.
3. In main_window, submit via background_jobs.submit_cancellable; disable Generate and show a determinate
   progress indication + Cancel while running; on success call viewer_3d.load_backend_states and record history
   exactly as today; on cancel leave the previous generated motion untouched; if the document revision changed
   during generation, discard the result and show a status message.
4. Tests: a unit test that cancellation stops generation and returns no partial result; a test that progress is
   monotonic and ends at 1.0; a GUI test that Generate leaves the window responsive (job submitted, not executed inline).
5. Update docs/user_guide.md (Workflow Toolbar → Generate) to mention progress/cancel.

Acceptance: no IK work on the GUI thread during Generate; identical CSV output to before for the existing
tests/test_model_agnostic_generation.py cases.
Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): preserve unrelated changes; reusable workflow logic belongs in application/, not Qt
callbacks; do not change export contracts; canonical terminology; update help/tutorial if any label changes.
```

### Prompt 3 — Make Preview Path validate the real motion
*Recommended: Sonnet, medium effort.*

```text
Goal: Preview Path (gui/robot_viewer_3d.py plan_preview ~1424 and _build_validated_preview_path ~1366)
interpolates committed-pose-at-t -> Orange-preview-at-t, i.e. the edit itself, which the robot never executes.
Change it to validate the motion that would result if the Orange preview were committed: previous Keyframe ->
Orange preview at t -> next Keyframe, sampled at the Export interval using the same interpolation as
RobotStateTimeline.sample_state.

Do:
1. Add a Qt-free function in application/ (e.g. application/preview_path.py) that takes the qpos timeline,
   current time, preview qpos, export dt, and a collision checker, and returns sampled qpos, times, per-sample
   collision flags, joint-limit violations and max per-joint velocity. Do not mutate the timeline (build a temporary copy).
2. Use it from plan_preview; draw ghosts for both adjacent segments; status reports first colliding time.
3. Keep the label "Preview Path". Update docs/concepts.md "Preview Path", docs/user_guide.md, gui/help content and
   gui/tutorial copy to describe the new semantics.
4. Tests: segment sampling covers both neighbours, preview at first/last Keyframe handles one side only, a
   colliding neighbour transition is flagged, and the timeline is unchanged afterwards.

Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): preserve unrelated changes; logic in application/ not Qt callbacks; model-specific
behaviour only in core/models/registry.py; canonical terminology; update help and tutorial copy.
```

### Prompt 4 — Prune the unreachable AI pipelines
*Recommended: Opus, medium effort (wide deletion with careful import surgery).*

```text
Goal: Reduce application/ai to the single reachable pipeline (CompactMotionWorkflow -> TrajectoryPlanner ->
TrajectorySpecExecutor -> trajectory_operations). Unreachable today: TextMotionWorkflow (text_planner.py;
GUI method gui/ai_assistant_controller.py _run_text_motion ~765-785 is never called), GhostGUIAgent (agent.py),
provider_comparison.py, repair_planner.py, plan_executor.py, motion_plan.py, and the hidden visual
refine/verify/critique buttons (gui/panels/ai_assistant_panel.py:124-159 are hide()'d and never laid out) with
their controller handlers (gui/ai_assistant_controller.py ~208-400 and ~487-619) plus visual_refinement.py and
visual_critique.py.

Do:
1. First move retime_segment (imported by trajectory_operations.py:34 from semantic_tools) and anything else
   the live path imports from legacy modules into the live modules. Then delete the legacy modules and
   tool_registry/semantic_tools if no longer referenced. Keep frame_capture.py if the live planner uses motion frames.
2. Remove the hidden panel widgets/signals and controller handlers; keep Accept/Discard/Refine preview behaviour unchanged.
3. Create one helper (e.g. application/ai/async_utils.py) for wait_for-with-timeout and _raise_if_cancelled and use
   it everywhere; delete the 8 duplicate copies. Remove dead temperature plumbing or wire it properly
   (providers/anthropic.py:63, provider_registry.py:125-128, trajectory_planner.py:~313) - prefer removal.
4. Remove the regex burpee rewrite (trajectory_planner.py:346-390). Move G1 crouch Joint Angles
   (motion_primitives.py:118-135) into core/models/registry.py as model data (e.g. a motion_reference_joint_poses
   mapping) so application/ai holds no G1 joint names.
5. Delete tests that only cover removed code; keep/adjust tests for the live path; shrink application/ai/__init__.py re-exports.
6. Update docs/architecture.md, docs/ai_assistant.md, docs/ai_security.md, docs/testing.md and docs/reliability.md
   so no removed component is described as live. Add nothing new to docs/README.md unless a page is added.

Acceptance: grep shows no references to removed modules; `grep -rn "left_hip\|left_knee\|burpee" application/ai`
returns only primitive-name plumbing, not joint values; GUI Motion Assistant still sends, refines, accepts, discards.
Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): preserve unrelated changes; model-specific behaviour in core/models/registry.py;
canonical terminology; update help/tutorial copy if any visible label changes.
```

### Prompt 5 — Remove dead backends and G1 hard-coding from `application/`
*Recommended: Sonnet, medium effort.*

```text
Goal: Delete the vestigial C++ backend (backend/robot_sim.{hpp,cpp}, backend/bindings.cpp - no build system
exists) and the G1-only PythonTrajectoryBackend analytic fallback (application/backend_interface.py:199-511),
and move hard-coded G1 joint orders out of application/.

Do:
1. Simplify BackendInterface to the MuJoCo IK backend only; remove BackendKind.CPP/ANALYTIC, FallbackPolicy
   approximate paths, LAB_JOINT_NAMES/IK_TASKS/MODEL_PATH constants (backend_interface.py:48-105) - keep a thin
   compatible API used by gui/ and tests (backend_name, solve_trajectory, export_last_solution_csv,
   has_last_solution, clear_last_solution). Keep the named backend CSV columns identical (export contract).
2. Add an `export_profiles` field to RobotModelInfo (core/models/registry.py), e.g.
   {"mjlab": ("left_hip_pitch_joint", ...)} for g1, and make application/trajectory_export_formats.py derive
   G1_JOINT_ORDER/mjlab compatibility from the active adapter's info instead of the module constant (lines 19-52, 352-370).
   Output bytes for G1 mjlab export must be unchanged - add a golden test comparing before/after on a small fixture.
3. Remove backend mentions from docs/architecture.md, docs/migration.md, README and update tests.
4. Extend scripts/check_architecture.py or add a test asserting no Unitree joint names appear in application/.

Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): preserve unrelated changes; do not change export contracts without focused tests and
docs (golden tests required); model-specific data only in core/models/registry.py; canonical terminology.
```

### Prompt 6 — Smooth interpolation and a motion-quality report
*Recommended: Opus, high effort (numerics + export-affecting).*

```text
Goal: Give GhostGUI RL-grade motion: C1-continuous qpos interpolation and a visible motion-quality report.
Today RobotStateTimeline._interpolate (core/models/model.py:945-956) is piecewise-linear on the manifold, so joint
velocity jumps at every Keyframe.

Do:
1. In core/models/model.py add an interpolation mode to RobotStateTimeline: "linear" (current, default for
   backwards compatibility) and "cubic" (cubic Hermite on the configuration manifold: tangents computed with
   mj_differentiatePos from neighbours, Catmull-Rom/monotone style, zero tangents at first/last Keyframe; integrate
   with mj_integratePos). Free-joint quaternions must stay normalized. Use bisect instead of sorting per call
   (model.py ~930). Exact values at Keyframes must be preserved.
2. Persist the mode in the project workspace (not a schema bump; optional key with default "linear"); expose it
   in planning controls. Generate's posture reference (application/trajectory_generation.py sample_state calls)
   must use the selected mode.
3. Add application/motion_quality.py (Qt-free): given times + qpos and the adapter, compute per-joint velocity and
   acceleration via finite differences on the manifold, joint-limit violations, velocity-limit violations
   (from MJCF actuator/joint data if present, else none), and collision on interpolated samples via
   core/ik/collision.py trajectory_collision_reports. Return a structured report.
4. Show the report in a small panel (worst joints, peak values with times, click to seek). Keep visuals minimal;
   another effort is redesigning the UI.
5. Tests: cubic passes through Keyframes exactly; velocity is continuous at interior Keyframes (finite-difference
   check) for a 3-Keyframe G1 case; quaternion norm stays 1; linear mode output is bit-identical to before.
6. Docs: docs/motion_math.md (interpolation), docs/concepts.md (Generation And Playback), docs/user_guide.md,
   help and tutorial copy.

Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): default behaviour and export contracts unchanged unless the user selects "cubic";
focused tests + docs for anything affecting exports; preserve unrelated changes; canonical terminology.
```

### Prompt 7 — Document-level history without view state
*Recommended: Opus, high effort.*

```text
Goal: Undo/redo currently snapshots widget and viewer state (gui/main_window.py capture_history_snapshot
~2404-2454, GuiHistorySnapshot in gui/history.py) including deep copies of the whole qpos timeline, generated
trajectory and ghost trajectory, and takes a new baseline on every time change (on_time_changed ~3537 ->
_refresh_history_baseline ~2560). Make history document-level and cheap.

Do:
1. Define an application-level MotionSnapshot (application/history.py or new module) containing only document
   data: target frames (immutable tuples), qpos timeline states (immutable mapping of time -> read-only ndarray,
   share unchanged arrays between snapshots instead of copying), timeline duration, motion metadata.
   Exclude: selected row, ghosts, show_ghosts, preview qpos/active, current time, generated trajectory.
2. Generated trajectory is derived data: on undo/redo mark it stale (clear it) instead of restoring a copy.
3. Remove the baseline refresh on navigation; capture "before" lazily at the start of a mutating action.
4. Restore through public APIs (no controls._suppress_pose_changed or state_timeline.states.clear() from the window).
5. Tests: memory/time test that 100 history entries over a 6000-state timeline share arrays (id/base checks);
   undo after commit, retime, paste, AI Accept restores document exactly; navigation creates no history entry.
6. Update docs/architecture.md ownership table.

Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py
Constraints (AGENTS.md): preserve unrelated changes; reusable logic in application/; no export-contract or
project-schema changes; canonical terminology.
```

### Prompt 8 — Tooling, logging and packaging hygiene
*Recommended: Sonnet, low–medium effort.*

```text
Goal: Add baseline engineering hygiene to GhostGUI without behaviour changes.

Do:
1. Add ruff (select E,F,B,UP,SIM with a conservative ignore list) and pyright in "basic" mode to pyproject.toml
   and CI (.github/workflows/ci.yml, after compileall). Fix only genuine errors (unused imports, undefined names);
   add per-file ignores rather than mass reformatting.
2. Introduce logging: a module-level logger per package (application, core, gui), configured in
   application/launcher.py with a RotatingFileHandler under the GhostGUI config dir (application/paths.py), plus
   GHOSTGUI_LOG_LEVEL env var. Log (not swallow) every `except Exception` in gui/main_window.py,
   gui/ai_assistant_controller.py, application/project_manager.py, application/model_importer.py with
   logger.exception. Never log API keys or prompt text (docs/ai_security.md).
3. Packaging: give core deps compatible-release pins (mujoco, PySide6, numpy, PyOpenGL) matching the CI-tested
   versions; make scripts/install_*.sh/.ps1 offer `--with-ai` that installs `.[ai]`.
4. Move application/background_jobs.py and application/file_dialog_helper.py (they import PySide6) into gui/ with
   re-export shims, and extend scripts/check_architecture.py to forbid PySide6/PyQt/OpenGL imports in application/
   except application/launcher.py.
5. Speed up tests/test_robot_viewer_timeline.py by reusing one RobotGuiMainWindow per class where tests don't
   depend on fresh state (reset via public APIs), targeting <50% of current 57 s.
6. Document the venv requirement for `python3 -m unittest discover -s tests -v` in docs/testing.md and CONTRIBUTING.md,
   and logging location in docs/operations.md and docs/troubleshooting.md.

Run: python3 -m unittest discover -s tests -v ; python3 scripts/check_docs.py ; python3 scripts/check_architecture.py ;
ruff check . ; pyright
Constraints (AGENTS.md): preserve unrelated changes; no behaviour, asset, fixture, cache-version or export changes;
canonical terminology; verify commands before documenting them.
```

---

### Appendix — reproduction snippets (in `/tmp`, not committed)

- `/tmp/ghostbench/bench.py`: G1 adapter load 0.34 s. Generate for 10 s @ 100 Hz with 11 Keyframes takes 5.79 s and yields 1001 states. `sample_state` takes 0.52 ms per call at 6,000 Keyframes.
- `/tmp/ghostbench/phantom.py`: Keyframes at [0.0, 2.0]. After `set_current_time(1.0)` they become [0.0, 1.0, 2.0]. After editing the 2 s Keyframe, the elbow at t=1 is 0.80 rad; the expected value is 0.05 rad.
- `/tmp/ghostbench/mod_times.txt`: per-module test timings.

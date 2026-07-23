# GhostGUI AI coding instructions

- This repo is a Python + PySide6 + MuJoCo desktop tool for editing robot trajectories and previewing IK against registered models.
- The app entry point is `application/launcher.py` (`ghostgui` console script). For dev runs use `python3 scripts/run_gui.py --model go2` or `--model g1`.
- `gui/main_window.py` is the central orchestrator. It creates one shared immutable `MuJoCoRobotAdapter` per model, then wires that into `BackendInterface`, `MujocoReferenceFrames`, and `RobotModelSession` objects.
- Treat `core/models/adapter.py` as the main model facade: it resolves source URDF/MJCF, builds the runtime MJCF cache, binds logical frames (`pelvis`, `left_hand`, `right_foot`, etc.), and exposes the mutable `mj_data` state used by the UI.
- The runtime model cache is versioned and path-based; for URDF imports it writes into `~/.cache/ghostgui/models/` or `GHOSTGUI_CACHE_DIR`. Preserve the cache-version and validation logic when changing model import behavior.
- `core/trajectory/model.py` is the source of truth for trajectory data. `Trajectory` stores multiple named per-frame tracks (`tracks`), not one flat list; `frames` is a derived compatibility view.
- The UI uses a “committed vs preview” workflow: the committed timeline state is the model-colored robot; the orange ghost is preview-only until `Accept Preview`. `Plan Preview` does not persist anything; `Cancel Preview` discards it.
- Changing the active timeline time discards an unaccepted preview. This is intentional and is the dominant state transition pattern in the viewer.
- Use `application.paths.PROJECT_ROOT` / `prepare_csv_save_path()` instead of hardcoding repo-relative paths when adding CSV or project code.
- Project persistence lives in `application/project_manager.py`. Projects are stored as folder-based `.ghostgui` workspaces with manifest files under `data/`, `autosave/`, and `workspace/`.
- Reuse the existing `RobotModelSession` pattern when adding cacheable model-specific UI state; do not create a second “current model” store outside the session map in `main_window.py`.
- Keep the live 3D viewer and batch backend solve separated: one immutable `MjModel` is shared, while each subsystem owns its own `MjData` state.
- Prefer `dataclass`-style plain Python objects over introducing new framework abstractions unless the code already has a clear pattern.
- Most tests are `unittest`-based and are run with `python3 -m unittest discover -s tests -v`.
- GUI tests may emit `QOpenGLWidget is not supported on this platform` in headless environments; that warning is expected in this repo’s current test setup and usually does not mean the test is broken.
- For model-related changes, follow the existing `RobotModelInfo`/`ROBOT_MODELS` registry flow in `core/models/registry.py` and keep `MuJoCoRobotAdapter` model-agnostic where possible.
- The repo already has explicit support for model-specific frame labels and end-effectors; if you add a new model, update the logical-frame mapping rather than inventing ad hoc code paths in the GUI.
- Existing workflows assume Linux/Ubuntu is the primary support target; shell scripts in `scripts/` are the canonical install/run entry points.
- When changing user-facing behavior, mirror the README’s documented semantics (`Reset 3D Pose`, `Accept Preview`, `Load qpos CSV`, `Save qpos CSV`) rather than introducing new UI metaphors.
- If you need to inspect or extend paths, look in `application/paths.py`, `application/model_importer.py`, and `core/models/assets.py` before writing new filesystem logic.

## Recommended automated remediation workflow

- Treat issue fixing as a staged pipeline, not a single “fix everything” pass.
- Phase 1: `scope lock` — read the repo guidance, then build a quick architecture snapshot from `gui/main_window.py`, `core/models/adapter.py`, `core/trajectory/model.py`, and `application/project_manager.py`.
- Phase 2: `audit` — identify logic holes, state-flow regressions, path-handling issues, cache/versioning problems, and security-sensitive input handling in `application/paths.py`, `application/project_manager.py`, `core/models/adapter.py`, `application/backend_interface.py`, and `gui/main_window.py`.
- Phase 3: `classify` — label each finding as a logic bug, state consistency bug, security weakness, or model/cache regression before changing code.
- Phase 4: `patch` — implement the smallest repo-native fix that preserves the existing `MuJoCoRobotAdapter` / `RobotModelSession` / committed-preview architecture.
- Phase 5: `verify` — rerun the repository’s test command, `python3 -m unittest discover -s tests -v`, and add targeted validation for the affected area when the broad suite is noisy in headless environments.

## Delegation pattern for AI agents

- Use a `planner` role to read instructions and produce a file-targeted execution plan.
- Use an `auditor` role to inspect only the likely problem files and produce a prioritized issue list with root cause and fix intent.
- Use a `fixer` role to apply minimal edits aligned with the repo’s patterns, such as preserving `dataclass` usage, not inventing a second “current model” store, and reusing `prepare_csv_save_path()` / `PROJECT_ROOT` for filesystem behavior.
- Use a `verifier` role to execute the test command, note environment-specific warnings like `QOpenGLWidget is not supported on this platform`, and confirm whether the patch addresses the actual regression.
- Require an approval gate before any code change: description of the issue, affected files, reason for the fix, and expected validation command.

## Repo-specific acceptance criteria

- Preserve the shared immutable `MjModel` + per-subsystem `MjData` split.
- Preserve the committed-vs-preview timeline behavior: preview edits must not silently alter the committed robot state.
- Preserve cache-version and path validation behavior for imported models.
- Prefer minimal edits over new abstractions unless the codebase already has an established pattern.
- Keep the fix aligned with the documented user semantics in `README.md` (`Reset 3D Pose`, `Accept Preview`, `Load qpos CSV`, `Save qpos CSV`).

# Changelog

All notable user-visible changes to GhostGUI are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project currently reports version `0.1.0`; no release tag has been created
in this repository.

## Unreleased

### Added

- Claude command bar (Ctrl+K): type a request and Claude operates GhostGUI with
  your own Anthropic key through the shared command registry, with streaming
  transcript, inline confirmations, Plan first, Stop, cost meter, Orange preview
  review and an offline command palette.
- Command bar **Undo run** (reverts the last run's accepted edit and its view
  changes in one step), Undo entries named "Claude: <prompt>", Up/Down prompt
  history (persisted owner-only, clearable, can be turned off) and an "Allow
  viewport images" setting.
- Command-bar evaluation scenarios with fail-closed replay fixtures and a
  replay/live evaluation script; Ask Claude help topic and shortcut rows.
- `ghostgui agent` headless command-bar runs, a manual live agent smoke check,
  prompt-injection red-team tests and a CI leg with the AI/agent extras.
- Shared Qt-free command registry used by GUI bindings, MCP and the command bar;
  every named menu/toolbar control is bound or explicitly exempt.
- Candidate timeline commands (insert, shift, move, duration, copy/paste, clear,
  reset pose, sandboxed CSV import) and settings/Preview Path/MuJoCo viewer
  commands for the command bar and attached MCP; MCP tool annotations and
  elicitation for destructive commands.

- Opt-in authenticated GUI attachment for external MCP agents, with immutable
  Orange preview, current-revision Accept, one-entry Undo, visible activity and
  Stop/Disconnect. Camera capture stays on the Qt thread and restores presentation.

- Optional `ghostgui-mcp` stdio server with shared tool schemas, resources, a
  motion-authoring prompt, structured errors, and bounded timestamped MuJoCo images.

- A Qt-free local agent command workspace with strict schemas, bounded
  checkpoints, provenance gates, sampled validation, and existing export writers.
- An explicit Export source selection for fresh generated or editable motion.

- Focused guides for concepts, data formats, models, model import, advanced IK,
  architecture, migration, operations, reliability, testing, and
  troubleshooting.
- A documentation index, contribution guide, citation metadata, and repository
  documentation checks.
- An MIT license for the project.
- Document, session, controller, command, and typed-event application
  contracts.
- An RViz-inspired visualization context with lifecycle-managed displays,
  tools, panels, and logical frame-pose services.
- Model-specific Motion Clips for copying, forward or reversed pasting, and
  Forward or Ping-pong repetition of committed motion ranges.
- Transaction recovery, cancellation, shutdown, architecture, packaging, and
  visual regression tests.
- An optional provider-neutral Motion Assistant with detached editing,
  semantic tools, Gemini and Claude adapters, timestamped visual critique,
  bounded visual refinement, and semantic provider comparison.
- Motion Assistant security and data-disclosure documentation with local
  request, response, tool-result, token, and rendered-frame limits.

### Changed

- Timeline navigation, save and export no longer insert qpos Keyframes.
- Commit Keyframe and automatic time advance form one reversible history action.
- Generated playback/export rejects stale motion; generation uses detached
  inputs and rejects late results when the source changes.

- Reworked the root README as a concise product landing page.
- Standardized the public workflow around **Preview Path**,
  **Commit Keyframe**, and **Keyframe interval**.
- Removed stale implementation investigations from the publishable repository.
- Made project saves and autosaves transactional with schema migration and
  project-contained path validation.
- Consolidated quaternion, coordinate, qpos, trajectory, collision, and IK
  contracts and made backend approximation explicit.
- Split history, playback, status, camera, loading, render progress, and IK
  panel ownership out of the large GUI facades.
- Added cancellable background work, deterministic session/process teardown,
  render coalescing, and context-bound OpenGL cleanup.
- Packaged model, mesh, theme, and help resources for installed wheels.

### Fixed

- Clarified the difference between raw qpos trajectory CSV and named backend
  CSV output.
- Clarified where imported model sources and generated runtime caches are
  stored.
- Corrected playback timing, logical-target edit restoration, and Go2 support
  contact acceptance defects from the documented baseline.

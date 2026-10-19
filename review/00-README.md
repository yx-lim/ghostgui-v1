# GhostGUI Review — Master Execution Plan

Three reports in this folder, each ending with copy-paste implementation prompts:

| File | What | Prompts |
| --- | --- | --- |
| `01-uiux-redesign.md` | UI inventory, ranked UX defects, new shell/timeline/inspector/Prompt Mode wireframes, design tokens, `main_window.py` decomposition. Screenshots in `screens/`. | UI-QW (Appendix A), UI-0, UI-1a, UI-1b, UI-2, UI-3, UI-4, UI-5 (§7) |
| `02-software-critique.md` | Scorecard, feature gaps vs Cascadeur/mjlab/GMR, architecture, workflow, code quality, actual test results, top-15 recommendations. | SC-1 … SC-8 (§8) |
| `03-claude-agent-integration.md` | Headless command layer (29 tools), MCP server, GUI live-attach, multi-turn Claude agent loop, eval plan. | AG-1 … AG-6 (§7), client config (§7.7) |

This file is the single order to run them in. It resolves conflicts between the
reports (marked **⚠ override**); where this file disagrees with a report, this file wins.

> `review/` is a working folder. Per AGENTS.md ("Do not commit one-off diagnosis
> notes"), do not commit it. Add `review/` to `.git/info/exclude` if you like.

---

## How to run each step

1. One step = one fresh Claude Code session on its own branch (`git switch -c <step-id>`).
2. Paste the **standard preamble** below, then the step's prompt from the referenced report
   section, then any **⚠ override** text from this file.
3. When it finishes: `/code-review high`, read the diff, run the app (`ghostgui --model g1`), commit, merge, next step.
4. Do not run two steps in parallel if both touch `gui/main_window.py`, `gui/robot_viewer_3d.py`, or `application/ai/`.

### Standard preamble (paste before every prompt)

```text
Repo: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md first and follow it.
- Use the project venv: `source .venv/bin/activate` (system python3 lacks mujoco/PySide6).
  Run GUI tests with QT_QPA_PLATFORM=offscreen.
- Line numbers quoted in the prompt come from a review of an older revision; treat them as
  hints and locate code by symbol name with grep before editing.
- Preserve unrelated working-tree files (plots/ and review/ are untracked; never modify or commit them).
- Canonical UI terms: Keyframe, Commit Keyframe, Keyframe interval, Orange preview, Preview Path,
  End Effector, Joint Angles. Any visible label change must update gui/help/* and gui/tutorial/* copy.
- Model-specific data/behaviour goes in core/models/registry.py, not application/ or gui/.
- Done means: `python3 -m unittest discover -s tests -v`, `python3 scripts/check_docs.py` and
  `python3 scripts/check_architecture.py` all pass; new public docs linked from docs/README.md.
- Finish with a short summary of what changed, what you could not verify, and any test you had
  to modify (and why).
```

---

## Execution order

Model key: **Opus-H** = `claude-opus-5-5`, high effort · **Opus-XH** = Opus, extra-high ·
**Sonnet** = `claude-sonnet-5-5`, medium effort.

### Wave 0 — Correctness bugs and quick wins (small, ship first)

| # | Step | Source | Model | Notes |
| --- | --- | --- | --- | --- |
| 0.1 | Stop timeline navigation from inserting hidden qpos Keyframes | SC-1 | Opus-H | Verified real: `gui/robot_viewer_3d.py` `set_current_time` calls `state_timeline.set_state`. Changes a tested contract; the test edit is expected. |
| 0.2 | UI quick wins 1–9 (stale generated motion, disabled-button styling, stale copy, Planning default, marker size, sidebar width, triple heading) | UI Appendix A | Sonnet | ⚠ override: item 2 (stale generated motion) is a correctness fix; add a test that Commit/Delete Keyframe marks generated motion stale and that export no longer uses stale motion silently. |
| 0.3 | Docs: AGENTS.md/testing.md must say to activate `.venv` | SC-8 (part) | Sonnet | Tiny; may fold into 0.2. |

### Wave 1 — Foundations (unblock both the redesign and the agent)

| # | Step | Source | Model | Notes |
| --- | --- | --- | --- | --- |
| 1.1 | Background, cancellable Generate with progress | SC-2 | Opus-H | |
| 1.2 | Preview Path validates the real prev → preview → next motion | SC-3 | Opus-H | |
| 1.3 | Remove dead C++/Python backends and G1 hard-coding from `application/` | SC-5 | Opus-H | Touches export contract; tests + docs required. Do before AG-1 (both add registry fields). |
| 1.4 | Prune unreachable AI pipelines (**reduced scope**) | SC-4 | Opus-H | ⚠ override below. |
| 1.5 | Design tokens + generated QSS | UI-0 | Sonnet | Independent; can run any time in Waves 1–2. |
| 1.6 | Extract controllers from `main_window.py`, no visual change | UI-1a | Opus-XH | Largest refactor. Must land before AG-1. |
| 1.7 | Tooling: ruff, pyright basic, `logging`, installer `[ai]` option | SC-8 | Sonnet | Independent. |

**⚠ override for 1.4 (SC-4).** Do **not** delete code the agent plan reuses. Keep:
`application/ai/tool_registry.py`, `agent.py`, `providers/mock.py`, `providers/replay.py`,
`frame_capture.py`, `context.py`, `edit_session.py`, `motion_services.py`, `visual_critique.py`.
Delete only: `text_planner.py`, `repair_planner.py`, `plan_executor.py`, `motion_plan.py`,
`provider_comparison.py`, `visual_refinement.py`, the hidden refine/verify/critique panel widgets and
their controller handlers, plus tests covering only those. Still do steps 3 (shared async helper,
remove dead temperature plumbing) and 4 (remove burpee regex; move G1 crouch Joint Angles into
`core/models/registry.py`). `semantic_tools.py` may be deleted only if nothing in the keep-list imports it.
Revisit whatever remains unreferenced after AG-4.

### Wave 2 — Claude can drive GhostGUI (fastest route to "prompt only")

After this wave you can open Claude Code in any folder and say *"design a 5-second G1 squat
that ends standing and export mjlab CSV"*, and Claude does the whole job headlessly through MCP.

| # | Step | Source | Model | Notes |
| --- | --- | --- | --- | --- |
| 2.1 | Headless command layer `application/agent_api/` + extracted services | AG-1 | Opus-XH | ⚠ override: after 1.6, Commit Keyframe/Generate/export live in `gui/editing/*` controllers, not `main_window.py`; extract services from there. Registry fields added in 1.3 should be extended, not duplicated. |
| 2.2 | MCP server `ghostgui-mcp` (stdio) + `.mcp.json` | AG-2 | Opus-H | Then try it for real: config from report 03 §7.7. |
| 2.3 | GUI live-attach (watch the agent edit in the running app) | AG-3 | Opus-XH | Opt-in, token-protected local socket. |

### Wave 3 — Redesigned editor + RL-grade motion

| # | Step | Source | Model | Notes |
| --- | --- | --- | --- | --- |
| 3.1 | New shell, view-state model, Orange preview bar, freshness | UI-1b | Opus-XH | |
| 3.2 | Document-level history (no snapshot per time change) | SC-7 | Opus-H | Before the dopesheet, which depends on cheap edits. |
| 3.3 | Timeline / dopesheet dock | UI-2 | Opus-XH | |
| 3.4 | Inspector, NumericField, Export dialog, Settings | UI-3 | Opus-H | |
| 3.5 | Smooth interpolation + motion-quality report | SC-6 | Opus-H | Changes exported numbers: behind a flag, with docs. Reuse `agent_api/metrics.py` from 2.1, do not duplicate it. |

### Wave 4 — In-app agent and Prompt Mode

| # | Step | Source | Model | Notes |
| --- | --- | --- | --- | --- |
| 4.1 | In-app multi-turn agent loop + Anthropic adapter upgrade (thinking replay, effort, caching, streaming, current model list) + `ghostgui agent "…"` CLI | AG-4 | Opus-XH | The prompt tells Claude to confirm API details against the installed `anthropic` SDK. That SDK is in `.venv`, so it can. |
| 4.2 | Vision self-critique + metrics loop | AG-5 | Opus-H | |
| 4.3 | Assistant dock + **Prompt Mode** (merged) | UI-4 **and** AG-6 | Opus-XH | ⚠ override: UI-4 and AG-6 both define Prompt Mode. Run them as one session: paste UI-4, then AG-6 as "additional requirements"; UI-4 wins on visuals, AG-6 wins on agent behaviour and approval policy. |
| 4.4 | Keyboard map, command palette, onboarding, a11y | UI-5 | Sonnet | Palette commands should come from the command registry (UI-1a) and, where relevant, the agent tool catalog. |

### Backlog (from the software review's top-15, not yet scheduled)

Physics feasibility via `mj_inverse` (#4), contact tracks + labels (#5), mirror pose via registry
`mirror_map` (#10; AG-1 already adds `mirror_joint_pairs`), headless batch CLI (#11; mostly
covered by 4.1), retargeted-motion import (#13), hot-path performance (#14), single motion document (#9).
Each of these becomes a natural agent tool once it exists.

---

## Realistic expectations

- This is about **22 sessions**, not one. Each prompt is sized to finish in a single session, and the
  test suite (673 tests, about 3 min in `.venv`) is the safety net between steps.
- The riskiest steps are 1.6, 2.1, 3.1 and 3.3. If a session runs out of room, ask it to commit what passes
  and write a TODO in the PR description. Then start a fresh session with "continue step X from the TODO".
- After Wave 2 you can prompt the agent from Claude Code. The in-app experience (Wave 4)
  needs the redesigned shell (Wave 3) to land cleanly.

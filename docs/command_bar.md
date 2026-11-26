# Claude Command Bar

The command bar is a second toolbar row ("Ask Claude to do anything in
GhostGUI…", Ctrl+K). You type a request in plain English and Claude, using
your own Anthropic API key, operates GhostGUI through the same command
registry that drives the menus, toolbar and MCP server.

## Setup

1. Click **Key…** and paste an Anthropic API key. By default it is stored in
   the system keychain; clear the checkbox to keep it for this session only.
   `ANTHROPIC_API_KEY` also works. Admin keys are refused.
2. Choose a model: Opus 5.5 (default, effort `medium`), Sonnet 5.5 (about
   half the cost) or Haiku 5.5 (cheapest, for simple commands). Changing the
   model starts a new conversation.
3. Choose the approval policy: **Review** (default; you Accept the Orange
   preview) or **Apply automatically** (this session only; a validated preview
   is accepted at the end of the run).

Without a key, or when Claude is unreachable, the bar works as a local command
palette: type `/` plus a command name, for example `/recent actions`.

## What Claude Can Do

- Read app state (`get_app_state`, a short digest is attached to every
  request), capture the viewport, list recent actions.
- Navigate (`go_to_time`), control playback, switch views and sidebars,
  change display options, camera and gizmo, select an End Effector, open Help.
- Edit motion on the detached Orange preview: author Keyframes, set Joint
  Angles, solve IK, insert time, shift or move ranges, scale, repeat, copy and
  paste (optionally reversed), set the duration, reset a pose to home, clear,
  generate and validate, then stage it for your review.
- Import a qpos or trajectory CSV from the project's `imports/` folder (plain
  file names only). DSMS, mjlab and robot-model imports stay in the File menu.
- Adjust export, generation and IK settings, validate Preview Path, and open
  or close the MuJoCo viewer (confirmation required).
- Create, open and save projects, load models and export trajectories.
- Ask you a short clarifying question.

## Safety

- Motion edits never change committed motion directly; one prompt produces at
  most one Undo entry, created when the preview is accepted. Because you review
  every preview, command-bar edits may change Keyframes you authored (never
  protected ones); external MCP agents keep the panel's override policy.
- Your own clicks on menu and toolbar commands are recorded in the same action
  log, so you can ask Claude to "do what I just did, but for the right hand".
- Every command carries a risk class. Destructive commands (for example GUI
  Undo/Redo) show an inline card: **Deny** (default), **Allow once** or
  **Allow for this run**. A card times out as a denial after 5 minutes, and
  more than 5 confirmations end the run.
- **Plan first** records UI, save, export, history and project actions as a
  plan for you to approve; read-only commands and preview edits still run.
- **Stop** (or Esc) cancels the stream and any queued tool call; the
  conversation stays valid and the preview is kept.
- Run limits: 30 turns, 100 tool calls, 12 images, 15 minutes, $1 per run and
  $10 per session by default. Costs are local estimates; set Console spend
  limits too.
- Text from projects, models and files is treated as untrusted data. Prompts
  that contain an API key are blocked before sending.
- Settings, keys, external-control policy and the bar itself are never exposed
  to Claude.

## Headless Use

`ghostgui agent "<request>" --projects-dir <folder> [--project NAME]` runs one
request without the GUI, using the same loop, registry and gates. GUI-only
commands are not offered; destructive commands ask on an interactive terminal
and are denied otherwise. Options: `--model`, `--effort`, `--approval
review|auto_accept`, `--plan`, `--max-cost` (default $1.00) and `--json`.

## Other Behaviour

- Actions by an attached external MCP agent appear in the transcript and the
  action log.
- Instruction-like text inside a tool result is flagged in the transcript as
  suspicious and still treated as data.
- Cache pre-warming (refreshing the 1-hour prompt cache when you return after
  an idle hour) exists but is off by default because it bills a cache write.
- `scripts/smoke_agent_loop.py --i-understand-this-spends` is the manual live
  check (models, two-turn loop, cache reads, spend cap). It also runs from the
  Live AI Provider Smoke workflow with `suite: agent`; it never runs on PRs.

## Evaluation

`tests/agent_scenarios.py` defines app-control scenarios (inspection, project
creation, a candidate timeline edit, plan-mode export, refusal handling,
injection containment, an unknown tool). `python scripts/eval_command_bar.py`
replays them from sanitised synthetic fixtures and reports the pass rate; the
replay test fails closed when the registry or prompt changes, and
`scripts/build_agent_replay_fixtures.py --write` refreshes the fixtures
deliberately. `--live --i-understand-this-spends` runs the same scenarios against
Claude with a spend cap; live and replay results are reported separately.

## Data Sent To Anthropic

Your prompt, the state digest, tool results (app and view state, project and
model names, sampled poses) and viewport captures when Claude requests them.
See [Motion Assistant Security](ai_security.md).

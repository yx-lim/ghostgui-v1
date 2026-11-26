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
- Edit motion on the detached Orange preview with the shipped motion tools,
  validate it and stage it for your review.
- Create, open and save projects, load models and export trajectories.
- Ask you a short clarifying question.

## Safety

- Motion edits never change committed motion directly; one prompt produces at
  most one Undo entry, created when the preview is accepted.
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

## Data Sent To Anthropic

Your prompt, the state digest, tool results (app and view state, project and
model names, sampled poses) and viewport captures when Claude requests them.
See [Motion Assistant Security](ai_security.md).

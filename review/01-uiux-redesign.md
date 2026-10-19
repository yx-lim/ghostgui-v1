# GhostGUI UI/UX Redesign Proposal

Status: proposal, not implemented. Branch inspected: `report-plot` (commit `e81734d`).
Author role: senior product designer + Qt/PySide6 engineer.
Scope: the PySide6 desktop editor (`gui/`), its theme, help/tutorial copy, and the
Motion Assistant surface. No tracked source file was modified to produce this report.

> Note on repository hygiene: AGENTS.md says "Do not commit one-off diagnosis
> notes." This file and `review/screens/` are untracked working material. Distill
> accepted decisions into `docs/` (e.g. a future `docs/ui_design_system.md`
> listed in `docs/README.md`) rather than committing this report as-is.

## 0. Evidence Base

Code read in full or in large chunks: `gui/main_window.py` (4,062 lines, 178
methods), `gui/robot_viewer_3d.py` (2,706 lines, 146 methods; layout and
state parts), `gui/controls.py`, `gui/app_sidebars.py`, `gui/theme.py`,
`gui/widgets/*`, `gui/panels/*`, `gui/viewers/ik_panels.py`,
`gui/viewers/robot_canvas_3d.py` (render/input parts),
`gui/viewers/trajectory_colors.py`, `gui/help/*`, `gui/tutorial/*`,
`gui/timeline_edit_dialog.py`, `gui/ai_settings_dialog.py`,
`gui/ai_assistant_controller.py` (UI-facing parts), plus `README.md`,
`docs/user_guide.md`, `docs/concepts.md`, `docs/architecture.md`,
`docs/ai_assistant.md`, `CHANGELOG.md`, `scripts/check_docs.py`,
`scripts/check_architecture.py`, and the GUI tests.

Screenshots were captured from the real app (`.venv/bin/python`, G1 model,
1400x860 logical, macOS, `QWidget.grab()`):

| File | State |
| --- | --- |
| `review/screens/main.png` | Cold start at ~12 s: modal "Rendering Unitree G1 49%" scrim |
| `review/screens/01_idle.png` | Idle after geometry ready (~25-30 s on this Mac) |
| `review/screens/02_orange_preview.png` | Orange preview after +X/+Z pose edit |
| `review/screens/03_preview_path.png` | After **Preview Path** |
| `review/screens/04_after_commits.png` | After three **Commit Keyframe** actions |
| `review/screens/05_planning_expanded.png` | Left "Planning" section expanded |
| `review/screens/06_joint_angles.png` | **Joint Angles** mode |
| `review/screens/07_ik_constraints.png` | Right "IK / Constraints" expanded |
| `review/screens/08_generated.png` | After **Generate** |
| `review/screens/09_help_center.png` | Help Center dialog |
| `review/screens/10_shortcuts.png` | Keyboard Shortcuts dialog |
| `review/screens/11_tutorial.png` | Tutorial step 4 of 9 |
| `review/screens/12_ai_running.png` | Motion Assistant panel, request running (standalone render) |
| `review/screens/13_ai_staged.png` | Motion Assistant panel, staged candidate result card |
| `review/screens/14_ai_error.png` | Motion Assistant panel, error after a refinement |
| `review/screens/dark_01_idle.png` | Dark color scheme, idle |

---

## 1. Current UI Inventory

### 1.1 Layout Today (1400x860, from `01_idle.png`)

```text
┌─ QMenuBar#appMenuBar (non-native even on macOS, main_window.py:451) ─────────────────────────┐
│ File  Robot  Timeline  View  Help                                                            │
├─ QToolBar#workflowToolbar (font shrunk 2pt, main_window.py:791-796) ──────────────────────────┤
│ [Preview Path][Commit Keyframe][Generate] | [Play][Reset][Clear] | [Move][Rotate][Gizmo] | [Undo][Redo] │
├────────────────────┬─┬───────────────────────────────────────────────────┬─┬─────────────────┤
│ AppLeftSidebar     │◂│ QTabWidget viewer_tabs (tab bar HIDDEN, :1276)    │▸│ AppRightSidebar │
│ ▼ Target           │ │  page 0 "3D Pose" = RobotViewer3D                 │ │ ▼ Status        │
│   Target robot     │ │  ┌─────────────────────────────────────────────┐  │ │  ✓ 3D geometry  │
│   frame [left_hand]│ │  │ RobotCanvas3D (OpenGL fixed-function)       │  │ │    ready.       │
│   Advanced target  │ │  │  committed robot (model colors)             │  │ │  ▸ Details      │
│   [left hand    ▾] │ │  │  Orange preview (alpha 0.65)                │  │ │ ▼ Motion Assist.│
│ ▼ Editing Mode     │ │  │  world-aligned gizmo, frame markers, ghosts │  │ │  Motion Assist. │
│  [End Effector|    │ │  │                                             │  │ │  claude-sonne…  │
│   Joint Angles]    │ │  └─────────────────────────────────────────────┘  │ │  [Settings]     │
│   X [m]  ███0.153 m│ │  ┌ Timeline (QGroupBox) ───────────────────────┐  │ │  transcript     │
│   Y [m]  ███0.227 m│ │  │ Time ○──────────────── [0.00 s] Frame —     │  │ │  context line   │
│   Z [m]  ██ 0.679 m│ │  │                         [Delete Keyframe]   │  │ │  [composer][↑]  │
│   Roll   ███ 16.3° │ │  └─────────────────────────────────────────────┘  │ │ ▸ IK/Constraints│
│   Pitch  ████44.4° │ │  page 1 "Simulation" = Mujoco3DViewerPanel        │ │  (Tasks|Weights │
│   Yaw    ███ 12.5° │ │  (reachable only via View menu)                   │ │   |Solver)      │
│ ▸ Planning (keyframe│ │                                                   │ │                 │
│   table + 10 knobs)│ │                                                   │ │                 │
├────────────────────┴─┴───────────────────────────────────────────────────┴─┴─────────────────┤
│ QStatusBar (appears lazily on first undoable action: "Commit keyframe; Ctrl+Z can undo.")    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Region Inventory

| Region | Contents | Source |
| --- | --- | --- |
| Menu bar | File (New, Open, Open Recent, Save, Import{Robot Model, Qpos, Trajectory{MuJoCo, DSMS, mjlab}}, Export{Qpos, Trajectory{MuJoCo, DSMS, mjlab}}); Robot (one checkable action per model); Timeline (Insert Time, Shift Entire Motion, Move Time Range, Scale Time Range, Copy Motion Range, Paste, Paste Reversed, Repeat); View (3D Pose, Simulation, Left/Right Sidebar, Use Model Colors, Show Keyframes, Show Trajectory Lines, Show Playback Poses); Help (Help Center F1, Start Tutorial, Keyboard Shortcuts) | `gui/main_window.py:447-770` |
| Workflow toolbar | 11 text+icon actions in 4 groups; `Move/Rotate` exclusive; `Gizmo` checkable | `gui/main_window.py:783-923` |
| Left sidebar | `Target` (frame combo + "Advanced target" body/site combo reparented from the viewer), `Editing Mode` (QTabBar `End Effector`/`Joint Angles` + 6 `InlineLabeledSlider` or 29 `JointControl`s for G1), `Planning` (collapsed) | `gui/app_sidebars.py:197-243`, `gui/controls.py:617-622`, `gui/robot_viewer_3d.py:351-377` |
| Planning section | "Trajectory keyframes" `QTableWidget` (9 columns, one row **per logical frame per Keyframe**), then a `QFormLayout` grab-bag: Smoothing, Collision substeps, Playback speed, Playback spacing, Playback opacity, Preview opacity, Keyframe interval, Max time, Export interval, DSMS motion speed | `gui/controls.py:423-463`, `gui/robot_viewer_3d.py:386-446`, `:549-618`, `:643-658` |
| Viewport | `RobotCanvas3D`: committed robot, Orange preview `(1.0,0.38,0.04)` override, ghosts at model colors x alpha 0.16, collision ghosts `(0.95,0.08,0.08)`, gizmo, per-frame markers | `gui/viewers/robot_canvas_3d.py:877-930` |
| Timeline strip | `QGroupBox("Timeline")`: label, `TimesliceSlider` with 2x6 px markers, time spinbox, "Frame —" readout, `Delete Keyframe` | `gui/robot_viewer_3d.py:505-638`, `gui/widgets/timeline.py` |
| Right sidebar | `Status` (`EditorStatusPanel`: severity glyph, title, message, Details toggle + QTextEdit), `Motion Assistant` (`AIAssistantPanel`), `IK / Constraints` (QTabWidget `Tasks/Weights/Solver`) | `gui/app_sidebars.py:246-266`, `gui/panels/status_panel.py`, `gui/panels/ai_assistant_panel.py`, `gui/viewers/ik_panels.py` |
| Status bar | Undo/redo/history toasts ("Undid …", "Commit keyframe; Ctrl+Z can undo.") | `gui/main_window.py:2577-2610` |
| Overlays | `RenderProgressOverlay` (modal scrim over viewport), tutorial overlay + card | `gui/render_progress.py`, `gui/tutorial/*` |

Dead or orphaned UI that still exists (built but never shown, or hidden for compatibility):

- `TrajectoryControlPanel.robot_panel` with `model_box`, `import_action_box`,
  `export_action_box` is built (`gui/controls.py:115-187`) but never added to
  a sidebar (`workflow_sections()` at `:617-622` omits it). Tests still drive
  `controls.import_action_box` (5 uses) and `model_box` is the source of truth
  for `select_robot_from_menu` (`gui/main_window.py:1169-1174`).
- `controls.time_slider` (`gui/controls.py:275-283`) is a hidden duplicate of
  the timeline; it is the value read by `current_frame()` (`:670-685`) and is
  used 29 times in tests.
- `Add Keyframe / Update / Delete / Clear Trajectory / Generate / Simulate`
  buttons (`gui/controls.py:403-418`) are never laid out, yet help still says
  "Click **Generate** or **Generate / Simulate**" (`gui/help/help_content.py:32`).
- `view_panel` (display toggles) exists only because `AppLeftSidebar` is
  created with `include_view=False` (`gui/main_window.py:320-323`); the toggles
  are reachable only through the View menu.
- Motion Assistant `Preview candidate`, `Refine`, `Visual refine`, `Verify
  visually`, `Critique` buttons, a proposal list and response label are built
  and hidden (`gui/panels/ai_assistant_panel.py:127-160`). Critique/visual
  verification therefore have **no user entry point** despite being documented
  features in `docs/architecture.md`.
- `RobotViewer3D.generate_button` "Demo trajectory", `play_button`,
  CSV load/save group boxes are created and hidden (`gui/robot_viewer_3d.py:310-349`, `:414-417`).

### 1.3 Key Flows Today

Manual authoring:

```text
Robot menu → choose model (render scrim ~25 s cold)
  → Target robot frame combo (or double-click body)
  → drag gizmo / drag InlineValueSlider / Joint Angles slider
      → Orange preview appears (status "ready to Plan, Accept, or Cancel")
  → [Preview Path] (toolbar)           → ghosts + status text only
  → [Commit Keyframe] (toolbar)        → writes N logical frames + qpos; time += Keyframe interval (0.10 s)
  → repeat (timeline markers are 2 px ticks)
  → open Planning ▸ set Export interval
  → [Generate] (toolbar)               → status "Generated uniformly sampled…"
  → [Play] (toolbar, far from the timeline)
  → File ▸ Export ▸ Trajectory ▸ MuJoCo/DSMS/mjlab → native file dialog
```

Steps that need a different region: 7 regions visited (menu, left sidebar,
viewport, toolbar, Planning section, timeline, menu again) and the status
feedback is in an eighth (right sidebar).

AI flow:

```text
Right sidebar ▸ Motion Assistant ▸ Settings (provider/model/key, Test Connection)
  → optionally select ≥2 rows in the Planning keyframe table (tooltip-only hint, controls.py:445-448)
  → type prompt, Enter
  → activity list → Claude message → result card "Motion updated / 5.0 s · 9 Keyframes"
     (the whole editor is frozen except viewport/timeline: main_window.py:1323-1362)
  → candidate auto-plays as Orange preview with a white reference robot
  → [Refine preview] toggles the *next* composer send into refinement mode
     (otherwise a new message silently replaces the candidate)
  → [Accept] / [Discard]
```

### 1.4 Counts

| Measure | Value |
| --- | --- |
| Interactive controls visible at once (idle, default sections) | ~41 (5 menus, 11 toolbar, 13 left, 3 timeline, 7 right, 2 splitter toggles) |
| With Planning expanded | ~52 (+ table + 10 numeric fields) |
| With IK / Constraints expanded | +3 tabs, +12 task controls (Tasks), or preset+Apply+29 joint weight sliders (Weights) |
| Joint sliders in Joint Angles mode (G1) | 29, flat list, snake_case names, no grouping/search (`06_joint_angles.png`) |
| Concurrent visual "robot states" | 6: committed, Orange preview, Preview Path ghosts, playback ghosts, generated playback pose, AI candidate (orange) + white reference |
| Distinct status surfaces | 4: right-sidebar Status, status bar, viewport overlay, Motion Assistant transcript |
| Modes | Editing mode (EE/Joint), gizmo (Move/Rotate/hidden), view (3D Pose/Simulation), edit time vs display time, AI-locked, candidate preview |
| Dialogs | Help Center, Keyboard Shortcuts, Project Browser, AI Settings, 6 timeline dialogs, 4 `QInputDialog`s, Unsaved Changes, Autosave restore, Clear confirm, 23 `QMessageBox.warning` sites in `main_window.py` |
| Keyboard shortcuts | 7 total: New/Open/Save, F1, Undo, Redo; T/R/E only when the canvas has focus (`robot_canvas_3d.py:1440-1461`) |
| Cold start to interactive viewport | ~25-30 s with a modal scrim (`main.png` at 12 s = 49%) |

---

## 2. Diagnosis (ranked by severity)

Severity: **S1** = causes wrong output or lost work; **S2** = blocks or
seriously slows core tasks; **S3** = friction/inconsistency; **S4** = polish.

### S1-1. Generated motion silently goes stale and keeps playing
After **Generate**, `preview_trajectory_time()` samples the generated
trajectory first and only falls back to the editable timeline when none exists
(`gui/robot_viewer_3d.py:1769`). **Commit Keyframe** does not clear it (only
timeline edits do, `gui/main_window.py:3490`). So after generating, then
committing a new Keyframe, scrubbing and playback show the *old* motion, and
DSMS/mjlab export samples "the current generated trajectory" (user guide).
There is no "out of date" indicator anywhere (`grep -ri stale gui` = 0 hits).
Users will export motion that does not contain their latest Keyframe.

### S1-2. "Orange" means two different scopes
The manual single-pose Orange preview and the whole-motion AI candidate use
the same orange override (`robot_canvas_3d.py:885,895`) and the same
viewport. The only differentiators are a status sentence
(`robot_viewer_3d.py:1669-1672`) and the frame readout text "AI candidate"
(`:1745`). In candidate mode **Commit Keyframe** is still enabled in the
toolbar and only explains itself via status text when pressed
(`robot_viewer_3d.py:808-813`, `:1451-1456`). Users cannot tell whether
Accept will commit one pose or a whole motion, or why Commit "does nothing".

### S1-3. Disabled primary buttons look enabled
`QPushButton#aiSubmitButton, QPushButton#aiResultAcceptButton` set a solid
accent background with no `:disabled` rule (`gui/theme.py:308-312`); the ID
selector out-specifies the generic `QPushButton:disabled` (`:228-232`).
In `14_ai_error.png` the superseded card's **Accept** is disabled but renders
as a bright blue primary button next to greyed Refine/Discard. Clicking a
"live-looking" Accept that does nothing during a failure is exactly the
moment users lose trust.

### S1-4. Destructive one-click actions sit in the primary toolbar
**Reset** immediately overwrites the committed pose at the active time and
records it as an edit (`robot_viewer_3d.py:1854-1882`) — no preview, no
confirmation — and sits between Play and Clear. **Clear** (trash icon) wipes
all Keyframes (confirmation exists, `main_window.py:3873`). Neither belongs in
an always-visible workflow strip next to Play.

### S2-1. Information architecture is organized by implementation, not task
- Planning mixes authoring (Keyframe interval, Max time), export (Export
  interval, DSMS motion speed), display (Playback spacing/opacity, Preview
  opacity), solver (Collision substeps) and generation (Smoothing) in one
  unlabeled form (`05_planning_expanded.png`; `robot_viewer_3d.py:386-446`,
  `:607-618`). It is collapsed by default (test asserts this,
  `tests/test_robot_viewer_timeline.py:2030`), so **Export interval** — which
  gates Generate alignment errors — is invisible.
- **Delete Keyframe** lives on the timeline; **Commit Keyframe** in the
  toolbar; **Play** in the toolbar; time scrubbing at the bottom; Keyframe list
  in the left sidebar; Keyframe retiming in a menu with typed ranges.
- Export is reachable only through File ▸ Export ▸ Trajectory ▸ format; the
  required settings (Export interval, DSMS motion speed) live in Planning.

### S2-2. The timeline is a slider, not a timeline
`TimesliceSlider` is a `QSlider` with 2x6 px markers in hard-coded blues
(`gui/widgets/timeline.py:15-18`, `:53`). With the defaults (5 s duration,
`robot_viewer_3d.py:159`; 0.10 s Keyframe interval, `:556`) three commits
occupy the first 4% of the track (`04_after_commits.png`). No zoom, no
per-track rows, no range selection, no drag-to-retime, no visible generated
range, no in/out markers, no transport (Play is at the top of the window).
All retiming is done by typing numbers into six modal dialogs
(`gui/timeline_edit_dialog.py`).

### S2-3. The Keyframe table is the wrong abstraction
One row per logical frame per Keyframe (`controls.py:758-782`) — G1 shows 6
rows for every commit; 9 columns in a ≤280 px sidebar are truncated to
"time / fram…" with a horizontal scrollbar (`05_planning_expanded.png`).
It is also the *only* way to give the Motion Assistant a time interval,
discoverable only via tooltip (`controls.py:445-448`).

### S2-4. Feedback is far from the action and parsed from strings
- Status lives in the top-right corner; the action buttons are top-left; the
  editing controls are left. Long messages are clipped at the sidebar edge
  ("Preview Path contains collisio…", `03_preview_path.png`).
- Severity is inferred by substring matching ("cannot", "collision", "no "…)
  in `status_event_from_text` (`gui/widgets/status.py:47-80`); a message
  starting with "No " is a warning by accident.
- Undo feedback goes to a separate, lazily created `QStatusBar`
  (`main_window.py:2590-2607`), which appears on first use and shifts the
  viewport layout by ~24 px (compare `01_idle.png` vs `03_preview_path.png`).
- Collision warnings appear for trivial moves (`02_orange_preview.png`) with
  raw MuJoCo body paths ("robot/right_wrist_pitch_link") and no highlighting
  link to the viewport.

### S2-5. Horizontal overflow in the right sidebar
At the default 270 px width the Motion Assistant header is clipped
("claude-sonne…"), the system bubble is cut on the right, and the IK tab bar
loses its third tab ("Se…") (`01_idle.png`, `08_generated.png`).
`AppSidebar.SECTION_MAX_WIDTH = 240` is overridden to `None` for both sidebars
(`app_sidebars.py:147,198,247`) while children keep fixed minimums
(transcript `setMinimumHeight(260)`, settings button, tab bar).

### S2-6. Motion Assistant has hidden modes and duplicated chrome
- "Motion Assistant" appears three times stacked: section header, panel title,
  first system bubble heading (`ai_assistant_panel.py:75-90`,
  `ai_chat.py:54-58`; `13_ai_staged.png`).
- One composer has two semantics: a normal send *replaces* the candidate; only
  after clicking **Refine preview** on the card does the next send build on it
  (`ai_assistant_panel.py:466-488`). The mode is shown only by placeholder text.
- Result card: title "Motion updated", no time range, no before/after, no link
  to the timeline; "Warnings and details" is a button that reveals bullet text.
- Error card has no **Retry** and leaves the previous candidate's actions in
  the confusing state from S1-3.
- While running or staged, the whole left sidebar, Robot/Timeline menus and
  most of the toolbar are disabled (`main_window.py:1323-1362`) with no banner
  explaining why.
- Critique / Verify visually exist in the controller but have no entry point.

### S2-7. Cold start and model switch block the whole viewport
The render-progress scrim covers the viewport for ~25-30 s on first launch
(`main.png`), while the side panels are interactive but meaningless (sliders
show values for a robot you cannot see). There is no skeleton/low-LOD robot
and no "you can start configuring" affordance.

### S3-1. Inline value sliders hurt legibility
`InlineValueSlider` paints the numeric value centered over the fill boundary
with two text colors (`inline_value_slider.py:245-300`), so values render as
"0.15|3 m" and "16.|3°" with the split mid-number (`01_idle.png`). The fill
starts at the range minimum, so a symmetric joint at 0° looks "half full".
Ranges are hard-coded (`X ∈ [-2, 2]`, `Y ∈ [-1, 1]`, `controls.py:285-303`),
not derived from the robot.

### S3-2. Units and vocabulary drift
- Joint sliders show degrees, but status says
  `Preview FK: name = +0.123 rad` (`robot_viewer_3d.py:1023,1028`).
- Status still says "ready to Plan, Accept, or Cancel"
  (`robot_viewer_3d.py:1341`) — legacy terms the docs checker would reject in
  docs but not in this string (it checks only listed patterns).
- "Target robot frame" vs "Advanced target" vs "End Effector" vs "TCP
  position" (`ik_panels.py:108-115`) for overlapping concepts.
- Window title "GhostGui" (`main_window.py:142,1385`) vs product "GhostGUI".
- Help says "Choose the robot model in the app toolbar" and "Export … from the
  app toolbar" (`help_content.py:25,33`) — there is no robot or export control
  in the toolbar.

### S3-3. Viewport color semantics collide
Reserved meanings overlap with frame colors (`trajectory_colors.py:7-14`):
`left_hand` amber `(0.95,0.70,0.10)` ≈ gizmo center sphere `(0.95,0.75,0.2)`
(`robot_canvas_3d.py:1141-1143`) ≈ Orange preview family; `right_hand`
`(0.95,0.15,0.12)` ≈ collision red `(0.95,0.08,0.08)` ≈ X-axis red. Preview
Path ghosts and playback ghosts use identical styling (model colors × alpha),
so "temporary validation" and "generated motion" look the same. The Orange
preview at alpha 0.65 almost fully hides the committed robot
(`02_orange_preview.png`).

### S3-4. Viewport framing wastes the hero area
The robot occupies roughly 8% of a 1390x900 px viewport (`01_idle.png`) with
no "frame selection"/"frame robot" command; the grid dominates.

### S3-5. Keyboard flow is minimal
Only 7 shortcuts exist; T/R/E require the canvas to have focus
(`robot_canvas_3d.py:1440-1461`); there is no key for Play, Commit Keyframe,
Preview Path, Generate, next/previous Keyframe, frame step, or focusing the
assistant. `KeyboardShortcutsDialog` lists 7 rows (`keyboard_shortcuts_dialog.py:13-21`).

### S3-6. Hidden 'Simulation' view
`viewer_tabs.tabBar().hide()` (`main_window.py:1276`) makes the MuJoCo
Simulation view reachable only via View ▸ Simulation, with no indicator of
which view is active.

### S3-7. Accessibility gaps
- Splitter collapse toggles are 12x140 px (`app_sidebars.py:27`) — below a
  24x24 minimum target and invisible against the border.
- Toolbar font is reduced by 2 pt (`main_window.py:791-796`).
- Muted text `#475569` on `#f1f5f9` passes AA (≈7:1), but `disabled_text`
  `#94a3b8` on white is ≈2.6:1 and is also used as a *meaningful* state.
- Severity is conveyed by a single Unicode glyph color (`●✓⚠✕`,
  `status.py:9-14`), timeline markers by color only.
- No visible focus ring style for `QToolButton`/`QTabBar` beyond border color;
  no `setAccessibleName` on toolbar buttons beyond text.

### S3-8. Theme is a flat palette, not a system
`Theme` (`theme.py:14-39`) has 24 role-ish fields named after widgets
(`section_header_hover_bg`, `quick_panel_border`), inline status colors
(`:136-138`), no spacing/type/radius/elevation tokens, and no robot-state
colors; GL colors are hard-coded in `robot_canvas_3d.py` and
`timeline.py`. The 3D viewport stays near-black in light mode, creating a
harsh light-chrome/dark-canvas seam (`01_idle.png`) while dark mode is
consistent (`dark_01_idle.png`).

### S4. Onboarding gaps
- No empty state: a new session opens straight into a full editor with 41
  controls and a loading scrim.
- The 9-step tutorial (`tutorial_steps.py`) has no Motion Assistant step and
  its card covers the Status panel it would want you to read (`11_tutorial.png`).
- Help Center content duplicates `docs/user_guide.md` in a separate source
  (`help_content.py`) and has already drifted (S3-2).

---

## 3. Redesign Vision

### 3.1 Design Principles

1. **One source of truth per state, one color per state.** Committed, Orange
   preview, Preview Path, generated, and AI candidate each get a unique,
   reserved visual encoding in the viewport *and* the timeline *and* chrome.
2. **The timeline is the spine.** Time, Keyframes, transport, generation
   status, ranges, and AI scope live in one bottom dock (Blender/Cascadeur).
3. **Act where you look.** Commit/Discard for an Orange preview appear
   *in the viewport HUD* next to the robot; status appears as an inline
   banner/toast at the point of action.
4. **Progressive disclosure by task.** The inspector shows what the current
   selection needs (Figma properties panel); solver/display/export settings
   move to their own homes.
5. **Never silently stale.** Derived data (generated motion, Preview Path)
   carries a freshness badge; exporting stale data requires a one-click
   regenerate.
6. **Keyboard-complete.** Every command is a registered action with a
   shortcut and a palette entry.
7. **AI is a collaborator with a visible scope.** The assistant always shows
   *what range* it will touch and *what will change* before Accept.
8. **Canonical words only:** Keyframe, Commit Keyframe, Keyframe interval,
   Orange preview, Preview Path, End Effector, Joint Angles.

### 3.2 Personas

| Persona | Goal | Today's pain | Design response |
| --- | --- | --- | --- |
| **Robotics researcher** (primary) | Precise, exportable qpos at known rates; reproducibility | Export interval hidden; stale generated data; no numeric Keyframe editing | Export panel with validation, freshness badge, numeric inspector, keyboard step |
| **Animator** (Cascadeur/Blender background) | Pose-to-pose blocking, timing, spacing | No dopesheet, no drag retime, tiny markers | Dopesheet with tracks, drag/scale, onion skin, transport |
| **Student** | Learn IK/keyframe concepts | 41 controls, jargon (TCP, qpos), no empty state | Guided empty state, tooltips with concept links, Simple inspector |
| **Prompt-only user** | "Make it do a burpee", review, export | Assistant is a 270 px side widget; whole editor freezes | Prompt Mode: chat as the primary surface, viewport/timeline as result review |

### 3.3 New Shell Layout (Editor Mode)

Model: Blender's 3D view + timeline, Figma's right properties panel,
Cursor's agent panel as a dockable right column that can take over.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ ◆ GhostGUI  ▸ my_project*  ·  Unitree G1 ▾        [Editor | Prompt]       ⌘K Search  ⚙  ?    │  Title/command bar (40px)
├────┬──────────────────────────────────────────────────────────────────┬──────────────────────┤
│ ⌖  │                                                                  │ INSPECTOR            │
│ ⟲  │  ┌ HUD top-left ───────────────┐            ┌ HUD top-right ┐    │ ┌──────────────────┐ │
│ ⎔  │  │ ● Committed @ 1.20 s        │            │ 3D ▾  Sim     │    │ │ left_hand  ▾  ⊕  │ │
│────│  └─────────────────────────────┘            └───────────────┘    │ │ [End Effector|   │ │
│ 👁 │                                                                  │ │  Joint Angles]   │ │
│ ◐  │                    (robot, framed to fill ~45% height)           │ ├ Position (m) ────┤ │
│    │                                                                  │ │ X [ 0.153 ] ↺    │ │
│    │                                                                  │ │ Y [ 0.227 ] ↺    │ │
│    │  ┌ Orange preview bar (only while preview exists) ─────────────┐ │ │ Z [ 0.679 ] ↺    │ │
│    │  │ ◍ Unsaved pose · IK err 0.4 mm · ⚠ 1 contact               │ │ ├ Rotation (°) ────┤ │
│    │  │      [Preview Path P]   [Discard Esc]   [Commit Keyframe K] │ │ │ R  16.3  P 44.4  │ │
│    │  └─────────────────────────────────────────────────────────────┘ │ │ Y  12.5          │ │
├────┴──────────────────────────────────────────────────────────────────┤ ├ Keyframe @1.20 s ┤ │
│ TIMELINE / DOPESHEET (dock, 180-320px, resizable)                     │ │ 6 targets · qpos │ │
│ ⏮ ◀◀ ▶ ▶▶ ⏭  1.20 s / 5.00 s  ⟳  1×   | Generated ✓ fresh  [Generate ⌘G]│ │ [Delete] [Go ▸] │ │
│ ─────────────┬──────────────────────────────────────────────────────── │ ├ IK ▸ (collapsed) ┤ │
│ Summary      │ ◆     ◆        ◇◍                ◆                     │ │ ASSISTANT ▸      │ │
│ left_hand    │ ◆     ◆        ◇                 ◆                     │ └──────────────────┘ │
│ right_hand   │ ◆              ◇                 ◆                     │                      │
│ pelvis       │ ◆     ◆                          ◆                     │                      │
│ Generated    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ stale after 1.2 s ░░░░             │                      │
├──────────────┴──────────────────────────────────────────────────────────────────────────────┤
│ ● Ready · 4 Keyframes · Export 0.01 s (100 Hz) · Backend: python   │  Undo: Commit Keyframe ↶ │  Status bar (24px, always present)
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Key changes:

- **Title/command bar** replaces menu+toolbar clutter: project name + dirty
  dot, model switcher, mode switch (Editor | Prompt), command palette entry,
  settings, help. On macOS, use the native menu bar (remove
  `setNativeMenuBar(False)`); keep menus for discoverability, but trimmed.
- **Left tool rail (40 px)**: Move (T), Rotate (R), Frame robot (F),
  gizmo visibility, Preview opacity popover, display toggles. Icons only,
  tooltips with shortcuts. Replaces 5 toolbar buttons + View toggles.
- **Viewport HUD**: state chip (top-left), view switch 3D/Simulation
  (top-right, replacing the hidden tab bar), and the **Orange preview bar**
  that appears only when a preview exists; it carries Preview Path, Discard,
  and the primary **Commit Keyframe**.
- **Timeline dock** (bottom) owns transport, Generate, freshness,
  Keyframes, ranges, Delete Keyframe, and retiming.
- **Inspector** (right) is selection-driven: target/End Effector/Joint
  Angles editing, then the selected Keyframe, then collapsed IK.
- **Assistant** is a dock that by default tabs with the Inspector
  (Inspector | Assistant), can be split side-by-side, or becomes the main
  surface in Prompt Mode.
- **Status bar** is permanent (no layout jump), shows readiness, Keyframe
  count, export rate, and the last undoable action with an undo button.
- Planning dissolves: Keyframe interval, Max time → timeline header popover;
  Export interval, DSMS motion speed → Export dialog (and project settings);
  Smoothing → Generate popover; Playback speed → transport; spacing/opacity →
  viewport display popover; Collision substeps → IK section.

### 3.4 Timeline / Dopesheet Wireframe

```text
┌ Timeline ───────────────────────────────────────────────────────────────────────────────────┐
│ ⏮ ◀| ▶ |▶ ⏭  [1.20 s]  of [5.00 s ▾]  ⟳ loop  [1.00× ▾]     Keyframe interval [0.10 s]  ⚙ │
│ Snap: ◉ Export grid (0.01 s)  ○ Keyframes  ○ Off        Generated: ✓ fresh   [Generate ⌘G] │
├──────────────┬──────────────────────────────────────────────────────────────────────────────┤
│              │0.0      0.5      1.0  ▼1.2   1.5      2.0      2.5      3.0      3.5    4.0 │  ruler (zoom: Ctrl+wheel,
│              │├────────┼────────┼────┃────┼────────┼────────┼────────┼────────┼────────┤ │   pan: Shift+wheel/MMB)
│ ▾ Summary    │◆        ◆          ◇┃◍                         ◆                  ⟦ AI ⟧   │  Summary row = qpos Keyframes
│   left_hand  │◆        ◆          ◇┃                          ◆                            │  per-target rows (expandable)
│   right_hand │◆                   ◇┃                          ◆                            │
│   left_foot  │◆        ◆           ┃                          ◆                            │
│   pelvis     │◆        ◆           ┃                          ◆                            │
│ ▾ Derived    │                     ┃                                                       │
│   Generated  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (hatched = stale)     │
│   Collisions │        ▲            ┃                                                       │  red ticks where samples collide
│   AI cand.   │                     ┃▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (orange hatched band)    │  only during AI review
└──────────────┴──────────────────────────────────────────────────────────────────────────────┘
Glyphs: ◆ committed Keyframe (filled, accent)   ◇ selected Keyframe (filled + 2px ring)
        ◍ Orange preview at playhead (hollow orange circle)   ┃ playhead   ⟦ ⟧ range brackets
```

Behaviors:

- Click empty ruler = move playhead (display time, live scrub). Release on a
  Keyframe snaps (existing `marker_snap_pixels = 10`).
- Click Keyframe = select (sets editing time). Shift-click = extend,
  Ctrl/Cmd-click = toggle. Drag in empty track area = marquee range select.
  Selected range drives: Copy (⌘C), Paste (⌘V), Paste Reversed (⌘⇧V),
  Repeat… (⌘R), Scale (S then drag, or right-click ▸ Scale…), Move (drag
  selection, G), Delete (⌫), and the Assistant's "Selected 0.50–1.20 s" chip.
- Alt-drag on ruler = Insert Time at that point (drag width = duration).
- All drags preview their result as ghost glyphs and validate live
  (red glyph = conflict, matching `TimelineEditPlan` preflight); release
  applies one Undo step. Typed dialogs remain for precision (double-click
  the selection badge).
- Keyframe interval, duration (Max time), and snap live in the header.
- Rows: Summary always; per-target rows collapsed by default for models with
  >6 logical frames.

### 3.5 Inspector Wireframe (Figma-style properties)

```text
┌ Inspector ───────────────────────────┐
│ Target                               │
│ [left_hand ▾]  ⊕ pick in viewport    │   ⊕ = click a body to target (replaces "Advanced target")
│ [ End Effector | Joint Angles ]  1/2 │   segmented control
├ Position · m ────────────── world ▾ ─┤
│ X  [ 0.153 ]  ‹›   ↺                 │   numeric field + drag-scrub on label (Blender style)
│ Y  [ 0.227 ]  ‹›   ↺                 │   ↺ reset to committed value (only when differs)
│ Z  [ 0.679 ]  ‹›   ↺                 │   changed fields get an orange left rule
├ Rotation · ° ────────────────────────┤
│ Roll [ 16.3 ]  Pitch [ 44.4 ]  Yaw [ 12.5 ]
├ Keyframe at 1.20 s ──────────────────┤   appears when playhead is on a Keyframe
│ ◆ Committed · 6 targets · qpos 36    │
│ [Go to prev ◀] [Go to next ▶] [Delete]│
├ ▸ IK & Constraints ──────────────────┤   collapsed; summary line "Position+Orientation · Feet planted"
├ ▸ Display ───────────────────────────┤   model colors, keyframe markers, paths, ghosts, opacities
└──────────────────────────────────────┘

Joint Angles mode:
┌ Joint Angles ────────── 🔍 filter ───┐
│ ▾ Left leg (6)                       │   groups from model metadata (hip/knee/ankle prefixes,
│   hip pitch   [-17.9°] ━━━━●━━━━     │   registry joint groups when available)
│   hip roll    [  0.0°] ━━━━●━━━━     │   centre-zero bar for symmetric limits, limit ticks
│   …                                  │
│ ▸ Right leg (6)  ▸ Waist (3)         │
│ ▸ Left arm (7)   ▸ Right arm (7)     │
│ Show only: ◉ all ○ changed ○ near limit
└──────────────────────────────────────┘
```

### 3.6 Prompt Mode Wireframe (AI-first)

Model: Cursor/Copilot agent panel taking the primary column, viewport as the
"diff view", timeline as the scope selector.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ ◆ GhostGUI ▸ burpee_study*  ·  Unitree G1 ▾        [Editor | Prompt]      ⌘K  ⚙  ?            │
├───────────────────────────────────────────┬──────────────────────────────────────────────────┤
│ ASSISTANT                 claude-sonnet-5 ▾│  Viewport — comparison                          │
│                                            │  ┌─────────────────────────────────────────────┐ │
│  You                                       │  │  ○ white = current motion                    │ │
│  Create a 5-second burpee                  │  │  ● orange = candidate (Orange preview)       │ │
│                                            │  │          [robot pair]                        │ │
│  ✓ Read current motion                     │  │                                              │ │
│  ✓ Planned 2 operations                    │  │  [Side-by-side | Overlay | Candidate only]   │ │
│  ✓ Applied burpee primitive                │  └─────────────────────────────────────────────┘ │
│  ✓ Validated (1 warning)                   │                                                  │
│                                            │  ┌ Candidate timeline ──────────────────────────┐ │
│  Claude                                    │  │ ▶ 2.40 s    current  ◆──◆───────◆           │ │
│  I built a no-jump burpee using the        │  │             candidate ◆─◆─◆─◆─◆─◆─◆─◆─◆      │ │
│  registered primitive.                     │  │             ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 0–5 s   │ │
│ ┌ Candidate ─────────────────────────────┐ │  │             ⚠ shallow foot contact @2.4 s    │ │
│ │ ◍ Burpee · 0.00–5.00 s · 9 Keyframes   │ │  └──────────────────────────────────────────────┘ │
│ │ Replaces current motion (3 Keyframes)  │ │                                                  │
│ │ ⚠ shallow foot contact at 2.4 s  [jump]│ │                                                  │
│ │ [Accept ⌘⏎]  [Discard]  [Open in Editor]│ │                                                  │
│ └────────────────────────────────────────┘ │                                                  │
│                                            │                                                  │
│ Scope: [Whole motion ▾] [left hand ×]      │                                                  │
│ ┌────────────────────────────────────────┐ │                                                  │
│ │ Refine the candidate…                   │ │                                                  │
│ └────────────────────────────────────────┘ │                                                  │
│ (●) Refine candidate  ( ) New request   ↑  │                                                  │
│ Suggestions: [slower] [keep feet planted]  │                                                  │
│              [loop 3×] [verify visually]   │                                                  │
├───────────────────────────────────────────┴──────────────────────────────────────────────────┤
│ ◍ Reviewing AI candidate — editor is read-only until you Accept or Discard   [Accept] [Discard]│
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Rules:

- **Explicit composer mode**: segmented `Refine candidate | New request`,
  defaulting to *Refine* while a candidate is staged (most follow-ups are
  refinements). Sending a *New request* while staged asks inline: "Replace the
  unaccepted candidate?" (one click, remembers per session). This reverses
  today's replace-by-default; it requires updating `docs/ai_assistant.md`,
  `help_content.py`, and `tests/test_ai_assistant_ui.py::test_staged_send_replaces_by_default_and_refine_is_explicit`.
  If the team wants to keep replace-by-default, keep the segmented control
  but default it to *New request* — the key fix is that the mode is a visible
  control, not a placeholder string.
- **Scope chips** come from the existing `_editor_selection_context`
  (`ai_assistant_controller.py:420-476`) and are editable: whole motion,
  selected range (from dopesheet), End Effector, Joint Angle.
- **Candidate card** shows range, Keyframe delta, replace/patch semantics,
  warnings as clickable timeline jumps, and three actions. Accept is primary.
- **Suggestion chips** expose hidden capabilities: "verify visually"
  (`start_visual_verification`), "critique" (`start_critique`), "visual
  refine" (`start_visual_refinement`).
- **Global review banner** (status bar region) replaces silent freezing.
- Retry button on error cards; errors show provider, short cause, and a
  "Copy details" link.

### 3.7 Onboarding / Empty State

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Start a motion                                       │
│  Robot   ( G1 humanoid ) ( Go2 quadruped ) ( H2 humanoid ) ( Z1 arm ) [Import…]│
│                                                                              │
│  ┌ Pose it yourself ───────────┐ ┌ Describe it ───────────────┐ ┌ Open ─────┐ │
│  │ Drag a hand, Commit         │ │ "a 5 s burpee"             │ │ Recent:   │ │
│  │ Keyframes, Generate.        │ │ [ Describe a motion…     ] │ │ • burpee  │ │
│  │ [Start editing]  [Tour 3m]  │ │ Needs an API key [Set up]  │ │ • wave    │ │
│  └─────────────────────────────┘ └────────────────────────────┘ │ [Browse…] │ │
│                                                                 └───────────┘ │
│  Import: [Trajectory CSV] [DSMS folder] [mjlab CSV]                           │
│  ☐ Show on startup                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

- Shown as a viewport overlay **while geometry loads** (turning the 25 s wait
  into setup time), with the loading progress as a thin bar at its bottom
  instead of a modal scrim. The editor becomes interactive as soon as the
  robot renders.
- "Describe it" switches to Prompt Mode with the text pre-filled.
- "Tour 3m" launches the existing `TutorialManager` with an updated 7-step
  script (3.11 below) that includes the Assistant.

### 3.8 Export Flow

```text
┌ Export Motion ─────────────────────────────────────────────────────────┐
│ Source   ◉ Generated motion  ✓ fresh (generated 14:02, 4 Keyframes)   │
│          ○ Editable Keyframe timeline (no IK resolve)                 │
│          ⚠ 1 Keyframe was committed after the last Generate.          │
│            [Generate now and export]                                   │
│ Format   [ MuJoCo CSV | DSMS folder | mjlab CSV (G1 29-DoF) | Qpos ]   │
│ Export interval  [0.01 s]  = 100 Hz   · 501 rows                       │
│ DSMS motion speed [1.00×]  (DSMS only; changes time.csv only)          │
│ Range    ◉ Whole motion  ○ Selected 0.50–1.20 s                        │
│ Preview  time,qpos0,qpos1,… (first 3 rows)                             │
│ Destination [ ~/projects/burpee/exports/burpee_mujoco.csv ] [Choose…] │
│                                             [Cancel]  [Export ⌘⏎]      │
└────────────────────────────────────────────────────────────────────────┘
```

- One dialog for all formats (replaces File ▸ Export ▸ Trajectory ▸ X and
  `export_action_box`); disabled formats explain why
  (`mjlab_export_compatibility_error`).
- Alignment check (Keyframe times on the Export grid) runs live; offending
  Keyframes are listed with "Snap to grid" fix.
- Orange preview is never exported; if one exists the dialog says so and
  offers **Commit Keyframe first**.
- "Repeat last export" (⌘⇧E) for iteration loops.
- Behavior stays in `application/trajectory_export_formats.py` and
  `application/csv_io.py`; the dialog only gathers options (AGENTS.md:
  keep workflow logic out of Qt callbacks).

### 3.9 Settings

```text
┌ Settings ────────────────────────────────────────────────────────────┐
│ General      │ Appearance: ◉ System ○ Light ○ Dark                   │
│ Appearance   │ Viewport background: ◉ Match theme ○ Always dark      │
│ Viewport     │ UI scale: [100% ▾]   Reduce motion ☐                  │
│ Timeline     │                                                       │
│ Assistant    │ Assistant: Provider [Anthropic ▾] Model [claude-…▾]   │
│ Keyboard     │ API key ••••••  ☑ Store in system keyring  [Test]     │
│ Advanced IK  │ Send rendered frames ☑   Max frames [8]               │
└──────────────┴───────────────────────────────────────────────────────┘
```

`AISettingsDialog` becomes the "Assistant" page of a single settings window;
keyboard page lists the command registry and allows rebinding later.

### 3.10 State Model (visually unambiguous)

| State | Meaning | Viewport | Timeline | Chrome | Allowed actions |
| --- | --- | --- | --- | --- | --- |
| **Committed** | Accepted pose at edit time | Model colors (or neutral grey `state.committed`), opaque | ◆ filled at Keyframes; playhead solid | HUD chip `● Committed @ t` (neutral) | Edit → Orange preview |
| **Orange preview** | Temporary single-pose edit | Orange `state.preview` at alpha 0.55 + 1 px orange silhouette outline; committed robot rendered as 35% grey "x-ray" so both are readable | Hollow orange ◍ at playhead | HUD chip `◍ Unsaved pose`, Orange preview bar with Commit Keyframe (primary), changed inspector fields get orange left rule, window dirty dot | Commit (K), Preview Path (P), Discard (Esc) |
| **Preview Path** | Temporary validation samples | Thin orange **outline-only** ghosts (no fill), collisions as solid red ghosts | Red ticks on "Collisions" lane | Bar shows "Path valid ✓" or "3/40 samples collide" with jump buttons | Same as Orange preview |
| **Generated** | Derived dense motion | During playback: committed robot animates; onion-skin ghosts in teal `state.generated` outline | "Generated" lane solid teal when fresh, hatched grey when stale | Badge `Generated ✓ fresh` / `⚠ stale — Generate` | Play, Export, Generate |
| **AI running** | Request in flight | Unchanged; subtle shimmer on viewport border | Scope range bracket pulses | Assistant activity list; status bar `Assistant working… Stop` | Stop; inspection only |
| **AI candidate** | Whole-motion staged edit | Same orange family (canonical "Orange preview") + white reference robot; viewport border 2 px orange and HUD chip `◍ AI candidate · 0–5 s` | Orange hatched "AI cand." lane over its range; current Keyframes dimmed | Review banner with Accept/Discard; toolbar Commit Keyframe hidden (not just inert) | Accept, Discard, Refine, scrub/play |

Decision: keep **orange for both** manual and AI previews because the
canonical term "Orange preview" is shared by the docs; disambiguate by
**scope cues** (single-pose bar vs whole-motion banner + lane + border) and
by removing actions that do not apply. Introducing a new candidate color
would force renaming public terminology.

Implementation: introduce an `EditorViewState` enum (Qt-free) in
`application/` with a pure reducer
`derive_view_state(preview_active, candidate_active, ai_running, generated_fresh)`,
published from `RobotGuiMainWindow` via one signal. HUD, timeline, inspector,
status bar and toolbar subscribe instead of each re-checking
`viewer.preview_active` / `candidate_preview_active`.

Freshness: add `generated_revision` alongside `ProjectDocument.revision`
(set when Generate succeeds); `generated_fresh = generated_revision ==
document.revision and qpos timeline revision unchanged`. This is an
application-layer change (no export contract change) with focused tests.

### 3.11 Tutorial Script (replacement, 7 steps)

1. Choose a robot (title-bar model switcher).
2. Pick a target (inspector ⊕ or double-click robot).
3. Move it → Orange preview appears (viewport bar).
4. Preview Path (P) validates; red means collision.
5. Commit Keyframe (K) saves at the playhead and advances by the Keyframe interval.
6. Generate (⌘G) and Play (Space) from the timeline; watch the freshness badge.
7. Export (⌘E) — or describe a motion in the Assistant (⌘L).

---

## 4. Design System

### 4.1 Tokens

Tokens live in `gui/design/tokens.py` as frozen dataclasses; QSS and painter
code read tokens, never hex literals.

**Color — primitives** (Tailwind-like scales, already close to current values):

| Primitive | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 950 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slate | #f8fafc | #f1f5f9 | #e2e8f0 | #cbd5e1 | #94a3b8 | #64748b | #475569 | #334155 | #1e293b | #0f172a | #020617 |
| blue (accent) | #eff6ff | #dbeafe | #bfdbfe | #93c5fd | #60a5fa | #3b82f6 | #2563eb | #1d4ed8 | | | |
| orange (preview) | #fff7ed | #ffedd5 | | #fdba74 | #fb923c | #f97316 | #ea580c | #c2410c | | | |
| red (danger/collision) | #fef2f2 | | | | #f87171 | #ef4444 | #dc2626 | #b91c1c | | | |
| amber (warning) | | | | | #fbbf24 | #f59e0b | #d97706 | #b45309 | | | |
| green (success) | | | | | #4ade80 | #22c55e | #16a34a | #15803d | | | |
| teal (generated) | | | | #5eead4 | #2dd4bf | #14b8a6 | #0d9488 | | | | |

**Color — semantic roles** (light / dark):

| Role | Light | Dark | Use |
| --- | --- | --- | --- |
| `bg.app` | slate-50 | slate-950 | window |
| `bg.panel` | white | slate-900 | inspector, docks |
| `bg.subtle` | slate-100 | slate-800 | section headers, tool rail |
| `bg.hover` | slate-200 | slate-700 | hover |
| `bg.viewport` | #e9edf2 (light grid) | #111418 | GL clear color (follows theme; "Always dark" option) |
| `fg.default` | slate-900 | slate-50 | text |
| `fg.muted` | slate-600 | slate-300 | secondary text (≥4.5:1 both) |
| `fg.disabled` | slate-400 | slate-500 | never the only carrier of meaning |
| `border.default` | slate-300 | slate-700 | |
| `border.focus` | blue-600 | blue-400 | 2 px focus ring |
| `accent.bg` / `accent.fg` | blue-600 / white | blue-500 / white | primary buttons |
| `accent.bg.disabled` | slate-200 | slate-700 | **fixes S1-3** |
| `success`, `warning`, `danger` | green-700, amber-700, red-700 | green-400, amber-400, red-400 | text/icons |
| `state.committed` | model colors or slate-400 | same | |
| `state.preview` | orange-500 (#f97316) | orange-400 | Orange preview, ◍ glyph, changed-field rule |
| `state.preview.outline` | orange-600 | orange-300 | path ghosts outline |
| `state.collision` | red-600 | red-400 | collision ghosts, ticks |
| `state.generated` | teal-600 | teal-400 | generated lane, onion skin |
| `state.stale` | slate-400 hatched | slate-500 hatched | stale generated |
| `state.keyframe` | blue-600 | blue-400 | ◆ glyphs |
| `state.reference` | white @ 0.9 / slate-200 | white @ 0.9 | AI reference robot |

**Reserved hue bands for the viewport** (enforced by a unit test): orange
20-35° = preview only; red 350-10° = collision/X-axis only; frame colors are
generated outside these bands. This requires changing `left_hand` and
`right_hand` in `FRAME_RGB_COLORS` and the focused assertion in
`tests/test_trajectory_colors.py::test_known_semantic_frame_colors_are_preserved`
(AGENTS.md: focused tests + docs).

**Spacing** (4 px base): `space.0=0, 1=4, 2=8, 3=12, 4=16, 5=20, 6=24, 8=32`.
Panel padding 12, section gap 16, control row height 28 (compact 24), tool
rail button 32x32, minimum hit target 24x24.

**Type scale** (relative to platform UI font, never hard-coded family; tests
assert the stylesheet does not override fonts, `tests/test_theme.py:82`):

| Token | Size | Weight | Use |
| --- | --- | --- | --- |
| `type.title` | base+4 | 600 | dialog titles, empty state |
| `type.heading` | base+1 | 600 | section headers |
| `type.body` | base | 400 | default |
| `type.label` | base-1 | 500 | field labels, toolbar text |
| `type.caption` | base-2 (min 10 px) | 400 | timeline ruler, badges |
| `type.mono` | base-1, `QFontDatabase.FixedFont` | 400 | numeric fields, time readouts |

Implement by setting `QFont` point sizes on widgets via a helper
(`apply_type(widget, "label")`), not QSS `font-size`, to keep the existing
"no font override" contract.

**Radius**: `radius.sm=3` (fields, chips), `radius.md=6` (buttons, cards),
`radius.lg=10` (dialogs, HUD bars), `radius.full=999` (pills/badges).

**Elevation** (Qt has no box-shadow in QSS; use `QGraphicsDropShadowEffect`
sparingly on overlays only):

| Level | Use | Treatment |
| --- | --- | --- |
| 0 | panels | border only |
| 1 | cards, HUD bars | `bg.panel` + border + 8 px/0.18 shadow |
| 2 | popovers, palette | 16 px/0.28 shadow |
| 3 | modal dialogs | scrim `rgba(2,6,23,0.45)` |

**Motion**: 120 ms for hover/press, 180 ms for panel reveal; honor a
"Reduce motion" setting.

### 4.2 Icons

- Keep the existing per-theme SVG approach (`theme_icon`,
  `gui/assets/theme/*-{light,dark}.svg`) but switch to a single
  `currentColor`-style SVG set recolored at load time
  (`QIcon` engine that replaces `#000` with the token color), removing the
  doubled asset files.
- Base set: Lucide (ISC license) at 16/20 px, 1.5 px stroke. Needed new
  glyphs: `keyframe` (diamond), `keyframe-add`, `preview-path`, `commit`,
  `discard`, `frame-target`, `pick`, `onion-skin`, `export`, `sparkles`
  (assistant), `stop`, `snap`, `loop`, `step-back/forward`, `jump-prev/next`.
- Icon-only buttons always have tooltip "Label (Shortcut)" and
  `setAccessibleName`.

### 4.3 Component Specs

| Component | Spec |
| --- | --- |
| **Button** | Variants `primary`, `secondary`, `ghost`, `danger`; sizes `sm 24`, `md 28`, `lg 32`; states default/hover/pressed/focus/disabled each defined for every variant (S1-3). Implemented via dynamic property `variant="primary"` + QSS `QPushButton[variant="primary"]:disabled`. Only one primary per surface. |
| **Segmented control** | `SegmentedControl(QWidget)` with exclusive `QButtonGroup` of checkable `QToolButton`s; 28 px; selected = `bg.panel` + 1 px border + `fg.default`; keyboard ←/→. Replaces `QTabBar#editingModeBar`, view switch, composer mode. |
| **Numeric field** | `NumericField` = `QLineEdit` + label that drag-scrubs (Blender): drag label horizontally to change, Shift = fine ×0.1, Ctrl = snap; wheel only when focused; Enter commits, Esc reverts; tabular mono digits; unit suffix in `fg.muted`; changed-from-committed left rule in `state.preview`. Replaces `InlineLabeledSlider` for X/Y/Z/RPY (fixes S3-1). |
| **Slider (joint)** | `JointSlider`: name row + `NumericField` + 4 px track; **fill from zero** for ranges spanning 0, else from min; limit ticks at ends; amber when within 5% of limit; value text never painted over fill. Keep `InlineValueSlider` API for compatibility and as the internal track. |
| **Timeline track** | Row height 22 px; header column 120 px; ruler 20 px with adaptive tick labels (0.01/0.1/0.5/1/5 s); playhead 2 px `accent` with time flag; zoom 0.25-200 px/0.1 s. |
| **Keyframe glyph** | 9 px diamond; committed = filled `state.keyframe`; selected = filled + 2 px `border.focus` ring; hover = 11 px; Orange preview at playhead = hollow circle `state.preview`; conflict during drag = `state.collision` outline; AI candidate Keyframes = small orange diamonds in the candidate lane. Shapes differ, not only colors (a11y). |
| **Badge/Chip** | 20 px pill, caption type; variants neutral/success/warning/danger/preview; removable chips have × with 16 px target. |
| **HUD bar** | Floating card bottom-center of viewport, elevation 1, max 640 px, appears with 180 ms slide; never overlaps gizmo center (repositions above if needed). |
| **Toast** | Bottom-right of viewport, auto-dismiss 4 s (errors persist), action button (e.g. Undo, Generate), stack max 3, also mirrored to status bar history. Replaces `statusBar().showMessage` for undo and most `show_status_message` calls. |
| **Inline banner** | Full-width in a panel, for persistent conditions (AI review, stale generated, model load warning). |
| **Result card** (Assistant) | Header (status icon + title + range), metrics row (duration, Keyframes Δ, warnings), warnings list with jump links, actions (primary Accept, secondary Discard, ghost Open in Editor). Resolved state collapses to one line "Accepted · 14:03". |
| **Section** | Replaces `CollapsibleSection` styling: header 28 px, chevron icon, optional summary text on the right when collapsed, no border box around every section (use 1 px separators). |

### 4.4 Restructuring `gui/theme.py`

Target structure:

```text
gui/design/__init__.py
gui/design/tokens.py      # Primitive + SemanticTokens dataclasses, LIGHT/DARK instances, GL color helpers
gui/design/qss.py         # build_stylesheet(tokens) -> str, composed from per-component functions
gui/design/components.py  # set_variant(button, "primary"), apply_type(widget, "label")
gui/design/icons.py       # themed icon engine (recolor SVG by token)
gui/theme.py              # compatibility shim
```

`gui/theme.py` keeps every public name tests import
(`Theme`, `LIGHT_THEME`, `DARK_THEME`, `application_stylesheet`,
`current_theme`, `ensure_application_theme`, `section_stylesheet`,
`tutorial_card_stylesheet`, `toolbar_stylesheet`, `render_progress_stylesheet`,
`help_button_stylesheet`, `theme_icon`, `is_dark_mode`) and maps the legacy
`Theme` fields onto semantic tokens (e.g. `Theme.section_header_bg =
tokens.bg_subtle`). `application_stylesheet()` becomes
`qss.build_stylesheet(current_tokens())`. `ThemeSynchronizer` stays.
GL code obtains colors via `tokens.gl("state.preview")` instead of literals in
`robot_canvas_3d.py:885,895,924,1141` and `timeline.py:53`.

---

## 5. Interaction Details

### 5.1 Keyboard Map

Single-key shortcuts are active only when focus is in the viewport, timeline,
or tool rail (not in text fields, not in `InlineValueSlider` which uses
arrows/Home/End). Implement with a `ShortcutScope` event filter on the
workspace container rather than `Qt.ApplicationShortcut`. Qt maps `Ctrl` to
`⌘` on macOS automatically.

| Key | Command | Notes |
| --- | --- | --- |
| `Ctrl+K` | Command palette | Cursor/VS Code convention |
| `Space` | Play / Pause | timeline/viewport scope |
| `K` | Commit Keyframe | mnemonic "Keyframe"; Blender uses I for insert, `I` also bound as alias |
| `P` | Preview Path | |
| `Esc` | Cancel drag; second Esc discards Orange preview | keeps existing E/Esc drag cancel |
| `T` / `R` / `E` | Move / Rotate / cancel drag | existing, widened scope |
| `F` / `Shift+F` | Frame target / frame robot | new camera helpers |
| `1` / `2` | End Effector / Joint Angles | |
| `←` / `→` | Step one Export interval | `Shift` = 10 steps |
| `↑` / `↓` | Jump to previous / next Keyframe | Blender convention |
| `Home` / `End` | Timeline start / end | |
| `Ctrl+G` | Generate | |
| `Ctrl+E` / `Ctrl+Shift+E` | Export… / Repeat last export | |
| `Delete` / `Backspace` | Delete selected Keyframe(s) | timeline scope; confirm if >1 |
| `Ctrl+C` / `Ctrl+V` / `Ctrl+Shift+V` | Copy Motion Range / Paste at playhead / Paste reversed | from dopesheet selection |
| `Ctrl+R` | Repeat Motion… | |
| `Ctrl+L` | Focus Assistant composer | Cursor convention |
| `Ctrl+Enter` | Accept AI candidate (when reviewing) | |
| `Ctrl+Shift+P` | Toggle Editor / Prompt mode | |
| `Ctrl+[` / `Ctrl+]` | Toggle left rail / right dock | |
| `Ctrl+Z` / `Ctrl+Shift+Z` (`Ctrl+Y` on Windows) | Undo / Redo | existing |
| `?` or `F1` | Shortcuts overlay / Help | |

### 5.2 Command Palette

- `CommandRegistry` (`gui/shell/commands.py`): each `Command` has `id`,
  canonical `label`, `category`, `shortcut`, `icon`, `enabled(state)`,
  `run()`. Menus, tool rail, HUD buttons, the shortcuts dialog, the tutorial,
  and the palette all read from it, so labels cannot drift (S3-2).
- Palette = frameless popup (elevation 2), fuzzy match over label + aliases
  ("slice" → Commit Keyframe is *not* an alias: legacy words stay out of UI
  copy per `check_docs.py`), shows shortcut, disabled commands show the reason
  ("Commit Keyframe — unavailable while reviewing an AI candidate").
- Also accepts `>`-prefixed natural language: "> make the wave slower" sends
  to the Assistant with current scope.

### 5.3 Undo Affordances

- Status bar shows `Undo: <description> ↶` permanently (clickable).
- Toast after destructive actions (Clear, Delete n Keyframes, Accept AI) with
  an **Undo** button.
- Edit ▸ History submenu lists the last 20 `HistoryStack` descriptions; click
  to jump (sequence of undos).
- Remove confirmation dialogs where Undo is reliable (Delete Keyframe), keep
  for Clear and for project-level transitions.

### 5.4 Drag / Scrub Behaviors

- Timeline scrub = display time only (current `preview_trajectory_time`);
  release on empty time sets edit time (current behavior); the playhead flag
  shows `1.234 s · frame 124` while dragging.
- Numeric field label scrub: 1 px = 1 step; Shift 0.1; Ctrl snaps to 10 steps;
  live Orange preview; release records nothing until Commit Keyframe (same as
  today).
- Keyframe drag: horizontal only, snaps per header Snap setting, Alt to
  duplicate (Paste), shows conflict state before release.
- Mouse wheel over timeline: zoom with Ctrl, pan with Shift, otherwise scroll
  rows (fixes accidental time changes from `TimesliceSlider.wheelEvent`).

### 5.5 Gizmo

- Keep world-aligned gizmo; add **Local/World** toggle in the inspector
  Position header when local frames land (README lists world-only as a
  limitation — show the toggle disabled with tooltip until then).
- Hover highlight + axis label ("X 0.153 m") near cursor while dragging;
  on-canvas readout of delta ("ΔZ +0.120 m").
- Gizmo center sphere color moves out of the preview orange band
  (`robot_canvas_3d.py:1141-1143`) to `fg.default` white/grey.
- During AI review the gizmo is hidden (not just non-interactive).

### 5.6 Multi-select Keyframes

- Selection model lives in a small Qt-free `TimelineSelection` (times set +
  optional range) in `application/`, consumed by dopesheet, inspector,
  assistant scope, and timeline edit planning (`application.timeline_editing`
  and `application.motion_clipboard` already accept ranges).
- Inspector with multiple Keyframes shows a summary: count, span, and bulk
  actions (Scale…, Move…, Repeat…, Delete).

### 5.7 Context Menus

| Target | Items |
| --- | --- |
| Viewport, robot body | Edit `<frame>` as target · Frame this body · Show only this limb's Joint Angles |
| Viewport, empty | Frame robot · Reset view · Display ▸ … |
| Orange preview bar | Commit Keyframe · Preview Path · Discard · Copy pose as qpos |
| Keyframe glyph | Go to · Delete Keyframe · Copy Motion Range (this Keyframe) · Recommit from current pose |
| Selected range | Copy · Paste at playhead · Paste Reversed · Repeat… · Scale Time Range… · Move Time Range… · Ask Assistant about this range |
| Ruler | Insert Time here… · Set Max time here · Shift Entire Motion… |
| Generated lane | Generate · Export… · Clear generated |
| Assistant message | Copy · Retry · Use as new request |

---

## 6. Implementation Plan

### 6.1 Ground Rules (from AGENTS.md and repository contracts)

- Canonical terms only in UI copy and docs: **Keyframe, Commit Keyframe,
  Keyframe interval, Orange preview, Preview Path, End Effector, Joint
  Angles**. `scripts/check_docs.py` rejects legacy "slice/plan preview/accept
  preview" words, but only in `UI_COPY_FILES` (`check_docs.py:28-34`). Every
  new module that contains UI strings must be **added to `UI_COPY_FILES`**
  (or switch it to a glob over `gui/**/*.py`) in the same change.
- Any public label change updates `gui/help/help_content.py`,
  `gui/tutorial/tutorial_steps.py`, `docs/user_guide.md`, and where relevant
  `docs/concepts.md` / `docs/ai_assistant.md` in the same change.
- New public docs (e.g. `docs/ui_design_system.md`, `docs/keyboard.md`) must
  be linked from `docs/README.md`.
- Dependency direction `core <- application <- gui` (enforced by
  `scripts/check_architecture.py` and `tests/test_architecture.py`): view-state
  reducers, freshness, selection models and command metadata that are Qt-free
  go in `application/`; widgets in `gui/`.
- Do not change export contracts, cache versions, robot assets, CSV
  fixtures, or project fixtures. Project schema additions (dock layout,
  appearance) go into `ui.ini` (`QSettings`), **not** `ghostgui_project.json`,
  unless a migration + tests are added.
- Preserve unrelated working-tree changes (`plots/` is untracked).
- Keep `RobotGuiMainWindow` attribute names used by tests as compatibility
  properties until tests are migrated (list below).

### 6.2 `main_window.py` Decomposition

Today `RobotGuiMainWindow` (178 methods) owns 14 concerns. Proposed split —
the facade keeps public attributes as forwarding properties:

| New module | Extract from `gui/main_window.py` | Lines (approx.) |
| --- | --- | --- |
| `gui/shell/commands.py` (`CommandRegistry`, `Command`) | action creation in `build_menu_bar`, `_toolbar_action`, `build_workflow_toolbar` | 447-923 |
| `gui/shell/menus.py` | menu assembly from registry; recent projects, robot menu | 447-770, 1146-1178, 1506-1553 |
| `gui/shell/workspace.py` (`WorkspaceLayout`) | splitter/dock build, sidebar collapse, persistence | 361-412, 950-1144 |
| `gui/shell/status_presenter.py` | `format_viewer_status`, `apply_status_event`, `show_status_message`, status bar/toasts | 2615-2726, 2588-2610 messages |
| `gui/shell/render_progress_controller.py` | render-progress orchestration | 2112-2227 |
| `gui/project/project_controller.py` | new/open/save/autosave/restore/transition prompts | 1364-2088 |
| `gui/editing/history_controller.py` | snapshot capture/restore, undo/redo | 2404-2614 |
| `gui/editing/model_session_controller.py` | model switch/load/import, session activation | 2727-3084 |
| `gui/editing/import_controller.py` | trajectory CSV import + dt prompts | 3085-3179, 2837-2931 |
| `gui/editing/timeline_edit_controller.py` | insert/shift/move/scale, Copy/Paste/Repeat, `_apply_timeline_edit` | 3180-3503 |
| `gui/editing/keyframe_controller.py` | pose/time handlers, commit/delete, target frame sync, generate | 3504-3999 |
| `gui/editing/scene_presenter.py` | `refresh_display`, render adapters | 4016-4062 |
| `gui/main_window.py` | composition root + compatibility facade (target < 600 lines) | — |

Each controller receives a narrow `host` protocol (document, controller,
viewer accessor, status presenter, history) rather than the whole window,
which makes them unit-testable without OpenGL.

Similarly, `RobotViewer3D._build_ui` should stop *building* sidebar widgets
(Planning form, target combo, IK panels) and only own canvas + state; the
inspector/timeline/settings panels build their own widgets and bind to the
viewer's state API. Keep the attribute names (`timeslice_step_input`,
`export_dt_input`, `delete_timeslice_button`, …) as forwarding properties
until tests move.

### 6.3 Tests Affected

| Test | Coupling | Action |
| --- | --- | --- |
| `tests/test_robot_viewer_timeline.py` (112 tests) | sidebar section titles `["Target","Editing Mode","Planning"]`/`["Status","Motion Assistant","IK / Constraints"]` and expanded states (`:2010-2035`); menu titles and Timeline menu items (`:2045-2058`); toolbar objectNames→text (`:2063-2076`: `planPreviewButton`, `sliceButton`, `quickGenerateButton`, `playbackToolbarButton`, `resetToolbarButton`, `clearToolbarButton`, `moveToolButton`, `rotateToolButton`, `gizmoVisibilityButton`); hidden tab bar (`:2061`); `timeslice_step_label "Keyframe interval"`, `"Max time"`, `"Delete Keyframe"`, `timeslice_timeline_group.title()=="Timeline"` (`:1360-1371`, `:2257`); `export_dt_label` (`:3850`); `playback_action.text()` and `viewer.play_button.text()` Play/Pause (`:1775-1780`); help dialog `section_list`/`browser`/`writtenGuideButton` and tutorial buttons (`:3549-3600`); tutorial target highlighting (`:3658+`) | Phase 1 rewrites structural asserts to new structure; keep objectNames on commands' buttons; keep viewer label attributes as properties |
| `tests/test_visual_smoke.py` | same section titles and toolbar texts | update with Phase 1 |
| `tests/test_robot_model_adapter.py` (54), `tests/test_advanced_ik.py` (12) | `window.controls.*`, `viewer_*` attributes | keep compatibility attributes |
| `tests/test_theme.py` | `LIGHT_THEME`/`DARK_THEME` field values in QSS, no font override, compact helpers, sidebar shell objectNames, tutorial card tokens | Phase 0: shim preserves names; extend with token tests |
| `tests/test_inline_value_slider.py` (9) | slider API | keep `InlineValueSlider`; add `NumericField` tests |
| `tests/test_status.py` (6) | `status_event_from_text` | keep; add typed `StatusEvent` emission tests |
| `tests/test_ai_assistant_ui.py` (18) | panel states, hidden compat buttons, Enter/Shift+Enter, replace-by-default, result card objectNames `aiResultAcceptButton` etc. | Phase 4 updates the replace/refine test if default flips |
| `tests/test_ai_assistant_controller_chat.py`, `tests/test_ai_v3_integration.py` | controller ↔ panel signals | keep signal names |
| `tests/test_trajectory_colors.py` | `left_foot`/`right_hand` RGB | Phase 5 if hue bands enforced |
| `tests/test_project_persistence.py` (17), `tests/test_lifecycle_reliability.py` (6) | project/close flows | must stay green through Phase 1 extraction |
| `tests/test_documentation.py`, `tests/test_architecture.py` | docs/architecture checkers | run every phase |

Compatibility attributes to retain on `RobotGuiMainWindow` until migrated:
`controls` (`time_slider`, `frame_box`, `table`, `import_action_box`,
`model_box`, `corner_smoothing_slider`, `show_lines_box`,
`show_keyframes_box`, `target_context_stack`, `set_position_values`,
`emit_time_changed`, `current_frame`), `viewer_3d`, `viewer_tabs`,
`app_toolbar`, all `*_action` attributes, `left_sidebar_content.sections`,
`right_sidebar_content.sections`, status label attributes from
`EditorStatusPanel.compatibility_widgets()`, `help_dialog`,
`tutorial_manager`, `undo_stack`, `redo_stack`.

### 6.4 Phases

| Phase | Goal | Create | Modify | Risk |
| --- | --- | --- | --- | --- |
| **0 Tokens/theme** | Token system, QSS generation, fix disabled-primary bug, themed GL colors | `gui/design/{__init__,tokens,qss,components,icons}.py`, `tests/test_design_tokens.py` | `gui/theme.py` (shim), `gui/widgets/timeline.py` (marker colors), `gui/viewers/robot_canvas_3d.py` (color literals → tokens), `gui/widgets/inline_value_slider.py` (token reads) | Low |
| **1 Shell layout** | Command registry, decomposed main window, title bar, tool rail, permanent status bar, dock layout, HUD + Orange preview bar, view-state reducer, freshness | `application/editor_view_state.py`, `gui/shell/*`, `gui/project/project_controller.py`, `gui/editing/*`, `gui/viewers/viewport_hud.py` | `gui/main_window.py`, `gui/app_sidebars.py`, `gui/robot_viewer_3d.py` (Commit hidden in candidate mode, stale generated), help/tutorial/docs, `scripts/check_docs.py` (`UI_COPY_FILES`), structural tests | High — split into 1a (extract, no visual change) and 1b (new shell) |
| **2 Timeline** | Dopesheet dock, transport, selection model, drag retime, generated lane | `application/timeline_selection.py`, `gui/timeline/{dopesheet,ruler,tracks,transport,timeline_dock}.py` | `gui/robot_viewer_3d.py` (remove `_build_timeslice_editor` UI, keep API), `gui/main_window.py`, help/tutorial/docs | Medium-high |
| **3 Inspector** | Selection-driven inspector, `NumericField`, grouped Joint Angles, Display/IK sections, Export dialog, Settings window | `gui/inspector/{inspector_dock,target_section,pose_section,joint_angles_section,keyframe_section,ik_section,display_section}.py`, `gui/widgets/numeric_field.py`, `gui/export_dialog.py`, `gui/settings_dialog.py` | `gui/controls.py` (becomes compatibility model), `gui/viewers/ik_panels.py`, `gui/ai_settings_dialog.py` (embedded page) | Medium |
| **4 Assistant** | Assistant dock + Prompt Mode, explicit composer mode, scope chips, richer result card, retry, review banner, suggestion chips for critique/verify | `gui/assistant/{assistant_dock,prompt_mode,result_card,scope_chips}.py` | `gui/panels/ai_assistant_panel.py`, `gui/widgets/ai_chat.py`, `gui/ai_assistant_controller.py` (UI hooks only), `docs/ai_assistant.md`, help | Medium |
| **5 Polish/a11y** | Keyboard map, palette, shortcuts overlay, empty state/onboarding, tutorial rewrite, accessibility names, contrast test, hue-band enforcement, frame-robot camera | `gui/shell/command_palette.py`, `gui/onboarding/empty_state.py`, `docs/keyboard.md` | `gui/help/*`, `gui/tutorial/*`, `gui/viewers/trajectory_colors.py`, `gui/viewers/camera.py`, tests | Low-medium |

---

## 7. One-Shot Implementation Prompts

Each prompt is self-contained and sized for one Claude Code session. Run them
in order; each assumes the previous phase merged.

### Prompt 0 — Design tokens and themed QSS

Recommended: Claude Opus, effort medium.

```text
You are working in /Users/yitxuenglim/Downloads/ghostgui-v1 (PySide6 app GhostGUI).
Read AGENTS.md, docs/architecture.md, gui/theme.py, tests/test_theme.py,
gui/widgets/timeline.py, gui/widgets/inline_value_slider.py, and the color
literals in gui/viewers/robot_canvas_3d.py (lines ~880-930, ~985-1100, ~1130-1260).

GOAL: introduce a token-based design system without changing layout.

1. Create gui/design/__init__.py, gui/design/tokens.py, gui/design/qss.py,
   gui/design/components.py.
   - tokens.py: frozen dataclasses `Palette` (primitive scales slate, blue,
     orange, red, amber, green, teal) and `SemanticTokens` with roles:
     bg_app, bg_panel, bg_subtle, bg_hover, bg_viewport, fg_default, fg_muted,
     fg_disabled, border_default, border_focus, accent_bg, accent_fg,
     accent_bg_disabled, accent_fg_disabled, success, warning, danger,
     state_preview, state_preview_outline, state_collision, state_generated,
     state_keyframe, state_keyframe_current, state_reference; plus spacing
     (4px scale), radius (sm=3, md=6, lg=10, full=999), and type roles as
     relative point offsets. Provide LIGHT_TOKENS and DARK_TOKENS and
     `gl_rgb(hex)` / `gl_rgba(hex, alpha)` helpers.
   - Keep the current light/dark hex values where an equivalent role exists so
     the app looks the same; state_preview must equal today's preview orange
     (1.0, 0.38, 0.04) when converted, state_collision today's (0.95, 0.08, 0.08).
   - qss.py: `build_stylesheet(tokens) -> str` composed of per-component
     functions (window, sidebar, buttons, inputs, tabs, menu, toolbar,
     sections, ai chat, render progress, tutorial card, help button).
     Add a generic button variant system: `QPushButton[variant="primary"]`
     with explicit :hover, :pressed, :disabled, :focus rules. Every
     ID-specific accent rule (e.g. #aiSubmitButton, #aiResultAcceptButton)
     MUST have a matching :disabled rule using accent_bg_disabled /
     accent_fg_disabled. Do not set font-family or font-size in QSS.
   - components.py: `set_variant(button, name)` (sets property + repolish).
2. Turn gui/theme.py into a compatibility shim: keep every currently exported
   name (Theme, LIGHT_THEME, DARK_THEME, current_theme, is_dark_mode,
   application_stylesheet, section_stylesheet, toolbar_stylesheet,
   render_progress_stylesheet, tutorial_card_stylesheet,
   help_button_stylesheet, theme_icon, ensure_application_theme,
   apply_application_theme, ThemeSynchronizer) with identical signatures.
   LIGHT_THEME/DARK_THEME field values must be derived from the tokens.
   Add `current_tokens(widget=None)`.
3. Replace hard-coded colors with token lookups in gui/widgets/timeline.py
   (marker colors) and gui/viewers/robot_canvas_3d.py (preview override,
   collision ghost red, gizmo center sphere) using current_tokens(). Do not
   change any rendered value in this phase except the gizmo sphere, which
   becomes a neutral light grey from fg tokens so it no longer resembles the
   Orange preview.
4. Mark #aiSubmitButton and #aiResultAcceptButton as variant="primary" in
   gui/panels/ai_assistant_panel.py and gui/widgets/ai_chat.py.
5. Add tests/test_design_tokens.py: (a) light and dark tokens define every
   role; (b) fg_default and fg_muted on bg_panel and bg_app meet WCAG 4.5:1
   (implement relative-luminance contrast in the test); (c) the generated
   stylesheet contains a ':disabled' rule for every selector that sets
   accent_bg; (d) a disabled QPushButton with objectName aiResultAcceptButton
   resolves a different background color than an enabled one (use
   QApplication with QT_QPA_PLATFORM=offscreen and widget.grab() pixel sample).

CONSTRAINTS: no layout or copy changes; keep tests/test_theme.py passing
unchanged; core/ and application/ must not import gui (run
python3 scripts/check_architecture.py). Do not touch export code, assets,
fixtures, or ghostgui_project.json.

ACCEPTANCE: app looks the same except the disabled Accept/Send buttons now
look disabled and the gizmo center is neutral; all tests pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 1a — Extract controllers from `main_window.py` (no visual change)

Recommended: Claude Opus, effort high.

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
docs/architecture.md ("UI Ownership During Migration"), gui/main_window.py
fully, and grep tests/ for `window.` attribute usage.

GOAL: reduce gui/main_window.py to a composition root + compatibility facade
WITHOUT any visible UI change.

Create (each with a narrow host Protocol argument, not the whole window):
- gui/shell/commands.py: `Command` dataclass (id, label, icon, shortcut,
  tooltip, checkable, object_name) and `CommandRegistry` that creates QActions.
  Re-implement build_menu_bar/build_workflow_toolbar on top of it so every
  existing QAction, objectName, text, tooltip, shortcut, and menu order is
  byte-identical (tests/test_robot_viewer_timeline.py:2010-2076 assert these).
- gui/shell/workspace.py: splitter construction and left/right sidebar
  collapse/persistence (main_window.py ~361-412, 950-1144).
- gui/shell/status_presenter.py: format_viewer_status, apply_status_event,
  show_status_message, undo/redo status-bar messages (~2588-2726).
- gui/project/project_controller.py: project new/open/save/autosave/restore/
  transition prompts (~1364-2088).
- gui/editing/history_controller.py (~2404-2614),
  gui/editing/model_session_controller.py (~2727-3084),
  gui/editing/import_controller.py (~2837-2931, 3085-3179),
  gui/editing/timeline_edit_controller.py (~3180-3503),
  gui/editing/keyframe_controller.py (~3504-3999),
  gui/editing/scene_presenter.py (~4016-4062).
RobotGuiMainWindow keeps every public method and attribute name currently
used by tests (e.g. on_add_keyframe, on_viewer_timeslice_time_changed,
undo_last_action, create_project_at, open_project_path, copy_motion_range,
paste_motion_at_current_time, set_ai_motion_controls_enabled, *_action,
undo_stack, redo_stack, status_text, viewer_status_label, ...) as thin
delegating methods/properties. AIAssistantController and TutorialManager must
keep working with the window as host.

Also: add every new gui module that contains user-facing strings to
UI_COPY_FILES in scripts/check_docs.py (or replace the tuple with a sorted
glob of gui/**/*.py excluding __pycache__) and keep check_docs passing.
Update the "UI Ownership During Migration" table in docs/architecture.md.

CONSTRAINTS: no copy, layout, or behavior changes; no changes to application/
contracts, exports, assets, fixtures; preserve unrelated working-tree files
(plots/ is untracked). gui/main_window.py target < 900 lines.

ACCEPTANCE: full test suite passes unchanged (no test edits allowed in this
phase except imports if a test imports a moved private helper);
python3 scripts/check_architecture.py passes; app launches with
`ghostgui --model g1` and looks identical.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 1b — New shell, view state, Orange preview bar, freshness

Recommended: Claude Opus, effort high.

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
review/01-uiux-redesign.md sections 3.3, 3.10, 4.3, 5.1, 6.1-6.3,
gui/main_window.py and the gui/shell/ + gui/editing/ modules from Phase 1a,
gui/robot_viewer_3d.py (preview_trajectory_time ~1722-1790, accept_timeslice
~803-836, begin/end_candidate_preview ~1627-1720), docs/concepts.md.

GOAL: new shell chrome and an unambiguous state model.

1. application/editor_view_state.py (Qt-free): enum EditorViewState
   {COMMITTED, ORANGE_PREVIEW, AI_RUNNING, AI_CANDIDATE} plus a frozen
   `ViewStateSnapshot(state, preview_path_shown, generated_status)` where
   generated_status ∈ {NONE, FRESH, STALE}; pure function
   `derive_view_state(...)`. Add tests.
2. Generated freshness: record the document revision and qpos-timeline
   revision at successful Generate (KeyframeController.generate). When either
   changes, status becomes STALE. When STALE, preview_trajectory_time must
   sample the editable state_timeline instead of the stale generated
   trajectory (fixes robot_viewer_3d.py:1769 behavior), and exports that
   would use the generated trajectory must show a QMessageBox offering
   "Generate and export" / "Export Keyframe timeline" / "Cancel". Add focused
   tests: commit after generate -> STALE; scrub shows the new Keyframe pose.
   Document the rule in docs/concepts.md ("Generation And Playback").
3. Shell: permanent QStatusBar created at startup (no layout jump) with
   ready text, Keyframe count, Export interval/Hz, and "Undo: <description>"
   button. Viewport HUD widget gui/viewers/viewport_hud.py overlaid on the
   canvas workspace: state chip (top-left) and an "Orange preview bar"
   (bottom-center) visible only in ORANGE_PREVIEW with buttons Preview Path,
   Discard, Commit Keyframe (primary variant). In AI_CANDIDATE show a
   banner "Reviewing AI candidate — Accept or Discard in the Assistant" and
   HIDE (not just disable) Commit Keyframe and Delete Keyframe in toolbar/HUD.
   Add a 3D / Simulation segmented control in the HUD top-right and keep
   viewer_tabs (tab bar may stay hidden).
4. Toolbar: keep the same QActions and objectNames, but move Reset and Clear
   out of the toolbar into an Edit menu (new top-level "Edit" between File
   and Robot containing Undo, Redo, separator, Reset Pose at Current Time,
   Clear All Keyframes…). Keep reset_action/clear_action attributes.
   Use the native macOS menu bar (remove setNativeMenuBar(False)).
   Window title "GhostGUI" (was "GhostGui").
5. Fix stale copy: robot_viewer_3d.py "ready to Plan, Accept, or Cancel" ->
   "ready to Preview Path, Commit Keyframe, or Discard"; joint status in
   degrees to match sliders.
6. Update help_content.py (remove "app toolbar" and "Generate / Simulate"
   references), tutorial_steps.py targets/copy, docs/user_guide.md
   "Interface" diagram and toolbar section, README "Basic Workflow" if
   labels change. Update structural tests (menu titles now include Edit;
   toolbar no longer contains reset/clear buttons) in
   tests/test_robot_viewer_timeline.py and tests/test_visual_smoke.py.

CONSTRAINTS: canonical terms only (Keyframe, Commit Keyframe, Keyframe
interval, Orange preview, Preview Path, End Effector, Joint Angles); no
export format/contract changes; Qt-free logic in application/; keep the
Phase 1a compatibility attributes.

ACCEPTANCE: making an edit shows the Orange preview bar; Esc discards; K
or the bar commits; AI candidate hides Commit; committing after Generate
shows "Generated ⚠ stale" in the status bar and scrub shows the new pose;
all tests and checkers pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 2 — Timeline / dopesheet dock

Recommended: Claude Opus, effort high.

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
review/01-uiux-redesign.md sections 3.4, 4.3 (timeline track, keyframe
glyph), 5.4, 5.6, 5.7; gui/widgets/timeline.py; gui/robot_viewer_3d.py
_build_timeslice_editor and the timeslice_* API (~505-840);
application/timeline_editing.py; application/motion_clipboard.py;
gui/editing/timeline_edit_controller.py; tests that reference timeslice_*.

GOAL: replace the bottom QSlider strip with a dopesheet dock.

1. application/timeline_selection.py (Qt-free): TimelineSelection with
   selected Keyframe times and optional [start, end] range; helpers to
   derive a range for motion_clipboard / timeline_editing plans. Tests.
2. gui/timeline/: ruler.py (adaptive ticks, zoom/pan), tracks.py (Summary
   row of qpos Keyframes, collapsible per-logical-target rows from
   ProjectDocument, "Generated" lane solid when FRESH / hatched when STALE,
   "Collisions" ticks from Preview Path flags, "AI candidate" orange hatched
   lane during candidate review), dopesheet.py (QWidget painting with
   QPainter; tokens from gui.design), transport.py (⏮ ◀| ▶ |▶ ⏭, time
   field, duration field = Max time, loop, playback speed, Keyframe
   interval, snap mode, Generate button + freshness badge),
   timeline_dock.py composing them in a QDockWidget at the bottom.
3. Interactions: click ruler = scrub display time (reuse
   RobotViewer3D.preview_trajectory_time); release = set edit time (reuse
   timeslice_time_changed path); click Keyframe = select + set edit time;
   Shift/Ctrl multi-select; marquee range; drag selection = Move Time Range
   plan with live conflict preview; S + drag = Scale Time Range; Alt+drag on
   ruler = Insert Time; Delete = delete selected Keyframes (one Undo step);
   Ctrl+C/V/Shift+V/Ctrl+R = Copy/Paste/Paste Reversed/Repeat using the
   selection. Context menus per section 5.7. Every applied edit goes through
   TimelineEditController so it is one history transition and clears
   generated motion exactly as today.
4. Keep RobotViewer3D's public timeslice API and attributes
   (timeslice_slider, timeslice_time_input, timeslice_step_input,
   timeslice_duration_input, export_dt_input, delete_timeslice_button,
   timeslice_timeline_group, labels) as hidden compatibility widgets or
   properties bound to the dock so existing tests pass; update only tests
   that assert the old strip is visible.
5. The Motion Assistant time interval now comes from TimelineSelection (keep
   controls.selected_time_interval() as a fallback adapter).
6. Docs/help/tutorial: update docs/user_guide.md (Timeline Editing,
   Planning And Timeline Controls), help_content.py, tutorial step for
   Commit/Generate; mention drag retime and shortcuts.

CONSTRAINTS: retiming semantics, validation, and rejection rules must be
exactly those of application.timeline_editing / motion_clipboard (no new
math in gui); canonical terms; no export changes.

ACCEPTANCE: three commits at 0.0/0.1/0.2 s are clearly visible after "zoom
to fit"; dragging a selected range moves Keyframes with one Undo step;
conflicts are shown before release and rejected exactly as the dialogs do;
all tests and checkers pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 3 — Inspector, numeric fields, Export dialog, Settings

Recommended: Claude Opus, effort medium-high.

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
review/01-uiux-redesign.md sections 3.5, 3.8, 3.9, 4.3 (numeric field,
joint slider); gui/controls.py; gui/widgets/{inline_value_slider,
joint_controls,trajectory_controls}.py; gui/viewers/ik_panels.py;
gui/robot_viewer_3d.py widget-building parts (~275-500);
application/trajectory_export_formats.py; gui/ai_settings_dialog.py.

GOAL: selection-driven right Inspector dock; dissolve "Planning".

1. gui/widgets/numeric_field.py: NumericField (QLineEdit + drag-scrub label,
   Shift fine, Ctrl snap, Enter commit, Esc revert, unit suffix, tabular
   digits, `changed` property painting a state_preview left rule). Tests
   modeled on tests/test_inline_value_slider.py.
2. gui/inspector/: inspector_dock.py (QDockWidget right), target_section.py
   (logical frame combo + "pick in viewport" toggle replacing Advanced
   target combo; keep target_box as compatibility), pose_section.py (X/Y/Z
   and Roll/Pitch/Yaw NumericFields; ranges derived from model reach where
   available), joint_angles_section.py (groups by joint-name prefix or
   registry groups, filter box, show changed/near-limit, centre-zero fill),
   keyframe_section.py (selected Keyframe summary, prev/next, delete),
   ik_section.py (wrap ik_panels; rename "TCP position/orientation" to
   "End Effector position/orientation"; move Collision substeps here),
   display_section.py (model colors, keyframe markers, trajectory lines,
   playback poses, playback spacing/opacity, Preview opacity).
3. TrajectoryControlPanel stays as the state/compatibility object (signals,
   time_slider, frame_box, table, set_position_values, current_frame) but is
   no longer shown; Inspector widgets bind to it. Remove the "Planning"
   section: Keyframe interval/Max time already live in the timeline dock;
   Smoothing moves into a Generate popover on the transport; Export interval
   and DSMS motion speed move to the Export dialog (and are still persisted
   via the same viewer inputs/project workspace keys).
4. gui/export_dialog.py: single dialog per section 3.8 (source with
   freshness, format segmented control with disabled reasons from
   mjlab_export_compatibility_error, Export interval with Hz and row count,
   DSMS motion speed, range, destination, first-rows preview). It calls the
   existing RobotViewer3D save paths / application services; no new file
   writing in Qt callbacks. File ▸ Export… (Ctrl+E) opens it; keep the
   existing per-format QActions working (they open the dialog preselected).
5. gui/settings_dialog.py with pages General/Appearance/Viewport/Assistant
   (embed AISettingsDialog content)/Keyboard (read-only list from
   CommandRegistry).
6. Update tests asserting sidebar section titles to the new docks; update
   help_content.py, tutorial_steps.py (targets), docs/user_guide.md
   (Interface, Left Sidebar, Planning sections), docs/README.md if a new
   doc is added.

CONSTRAINTS: canonical terms; persisted project keys unchanged (project
workspace export_dt, dsms speed, smoothing, display flags); no export
contract changes.

ACCEPTANCE: no truncated text at 1280x800; Joint Angles for G1 grouped
into ≤6 groups with filter; Export interval visible in the Export dialog;
full suite + checkers pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 4 — Assistant dock and Prompt Mode

Recommended: Claude Opus, effort high.

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
review/01-uiux-redesign.md sections 2 (S1-2, S1-3, S2-6), 3.6, 3.10,
4.3 (result card); docs/ai_assistant.md; gui/panels/ai_assistant_panel.py;
gui/widgets/ai_chat.py; gui/ai_assistant_controller.py (start_edit, refine,
start_critique, start_visual_refinement, start_visual_verification, preview,
accept, reject, _editor_selection_context, _motion_result_summary);
tests/test_ai_assistant_ui.py; tests/test_ai_assistant_controller_chat.py.

GOAL: make the Assistant a first-class, unambiguous surface.

1. gui/assistant/assistant_dock.py: QDockWidget tabbed with the Inspector.
   Header shows provider/model as one elided label (no duplicated "Motion
   Assistant" titles; first system bubble has no heading).
2. Composer: SegmentedControl "Refine candidate | New request" visible only
   while staged; default = Refine candidate. Sending "New request" while
   staged shows an inline confirm row ("Replace the unaccepted candidate?
   Replace / Cancel", remember for session). Update
   tests/test_ai_assistant_ui.py::test_staged_send_replaces_by_default_and_refine_is_explicit
   and docs/ai_assistant.md + help_content.py accordingly. Keep the
   refine_requested / submit_requested signals.
3. Scope chips above the composer from _editor_selection_context (range,
   End Effector, Joint Angle); chips are removable and clicking the range
   chip selects it in the timeline dock.
4. gui/assistant/result_card.py replacing MotionResultCard (keep objectNames
   aiResultAcceptButton, aiResultDiscardButton, aiResultRefineButton): title
   with range ("Burpee · 0.00–5.00 s"), metrics (duration, Keyframes Δ vs
   current, warnings), replace vs patch line, warnings as buttons that seek
   the playhead, actions Accept (primary) / Discard / Open in Editor.
   Resolved cards collapse to one line. Error cards get Retry (re-emits the
   last instruction with the same mode) and "Copy details".
5. Suggestion chips after a staged result: "Verify visually"
   (visual_verify_requested), "Critique" (critique_requested), "Visual
   refine" (visual_refine_requested) — enabled only when the provider
   capabilities support images.
6. Review banner: while AI_RUNNING or AI_CANDIDATE show a status-bar/
   viewport banner explaining that editing is paused, with Stop or
   Accept/Discard; Ctrl+Enter accepts when reviewing.
7. gui/assistant/prompt_mode.py: Editor | Prompt switch in the title bar
   (Ctrl+Shift+P). Prompt mode = assistant column left (min 420 px),
   viewport right with comparison mode segmented control (Overlay |
   Candidate only), compact candidate timeline under it; inspector hidden.
   Persist the mode in ui.ini.

CONSTRAINTS: no provider/protocol changes, no new provider turns, keep
application/ai contracts; Qt-free logic stays in application/; canonical
terms (the candidate is shown as the Orange preview).

ACCEPTANCE: disabled actions look disabled; superseded cards have no live
buttons; error has Retry; critique/verify reachable from UI; Prompt Mode
usable at 1280x800; full suite + checkers pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

### Prompt 5 — Keyboard, palette, onboarding, accessibility

Recommended: Claude Sonnet, effort medium (Opus if the palette fuzzy matcher
and scope filter need debugging).

```text
Repository: /Users/yitxuenglim/Downloads/ghostgui-v1. Read AGENTS.md,
review/01-uiux-redesign.md sections 3.7, 3.11, 4.1 (hue bands), 5.1-5.3;
gui/shell/commands.py; gui/help/*; gui/tutorial/*;
gui/viewers/trajectory_colors.py; gui/viewers/camera.py;
tests/test_trajectory_colors.py.

GOAL: keyboard-complete, discoverable, accessible.

1. Register every shortcut in section 5.1 via CommandRegistry with a
   ShortcutScope event filter so single-key shortcuts never fire inside text
   inputs or InlineValueSlider/NumericField editing. T/R/E keep working.
2. gui/shell/command_palette.py (Ctrl+K): fuzzy search over command labels
   and categories, shows shortcuts and disabled reasons; '>' prefix sends
   the text to the Assistant with current scope.
3. Rebuild KeyboardShortcutsDialog from the registry (group by category);
   add '?' overlay. Add docs/keyboard.md and link it from docs/README.md
   and docs/user_guide.md.
4. gui/onboarding/empty_state.py: overlay shown during initial geometry load
   and for sessions with no Keyframes (setting "Show on startup"); robot
   cards, "Start editing", "Describe it" (opens Prompt Mode prefilled),
   recent projects, import buttons. Replace the modal render scrim with a
   thin progress bar on the empty state.
5. Tutorial: rewrite FIRST_MOTION_TUTORIAL to the 7 steps in section 3.11
   targeting the new objectNames; include the Assistant step; ensure the
   card never covers its target or the status bar.
6. Camera: F frames the selected target, Shift+F frames the robot (fit
   bounding sphere to ~45% of viewport height).
7. Colors: move FRAME_RGB_COLORS and generated hues out of the reserved
   bands (orange 20-35°, red 350-10°); update
   tests/test_trajectory_colors.py focused assertions; add a test that all
   bundled models' frame colors avoid reserved bands.
8. Accessibility: setAccessibleName on all icon-only buttons; minimum 24x24
   hit targets (fix sidebar toggles 12x140 in app_sidebars.py); visible
   2px focus ring for QToolButton/QTabBar/segmented controls; add a test
   that iterates the main window's buttons and asserts non-empty
   accessibleName or text and size ≥ 24 in both axes.

CONSTRAINTS: canonical terms; update help_content.py and tutorial copy with
any label change; add new docs to docs/README.md; no export changes.

ACCEPTANCE: every command in menus has a shortcut or palette entry;
tutorial runs end-to-end; full suite + checkers pass.

RUN: QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v;
python3 scripts/check_docs.py; python3 scripts/check_architecture.py.
```

---

## Appendix A. Quick Wins (can ship before the redesign)

Each is < 1 hour and low risk:

1. Add `:disabled` rules for `#aiSubmitButton` / `#aiResultAcceptButton`
   (`gui/theme.py:308-312`). Fixes S1-3.
2. Clear or flag generated motion on Commit Keyframe / Delete Keyframe (or
   sample `state_timeline` when the document revision changed). Fixes S1-1.
3. Hide `slice_action` while `candidate_preview_active`. Mitigates S1-2.
4. Fix stale copy: `robot_viewer_3d.py:1341`, `:1023,1028` (rad→°),
   `help_content.py:25,32,33`, window title "GhostGUI".
5. Create the `QStatusBar` at startup to stop the layout jump.
6. Default the `Planning` section to expanded or move **Export interval**
   next to Generate. (Test `tests/test_robot_viewer_timeline.py:2030`
   asserts collapsed; update it.)
7. Increase timeline markers to 9 px diamonds and add "zoom to Keyframes".
8. Remove the triple "Motion Assistant" heading.
9. Set right sidebar default width to 320 px and elide the provider label.

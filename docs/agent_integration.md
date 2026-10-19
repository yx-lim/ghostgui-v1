# Local Agent Command Layer

GhostGUI exposes a Qt-free Python command layer for deterministic project and
motion workflows. It does not call an LLM. MCP transport, GUI attachment, the
in-app autonomous loop, rendering tools, and an agent CLI are later milestones
in the [implementation plan](implementation_plan.md).

## Entry Points

`HeadlessWorkspace` owns the compiled model, committed document, project,
detached `AIEditSession`, provenance, candidate checkpoints, generated samples,
and acceptance history. Caller code configures the project root, optional
output root, registered models, and approval policy.

`build_command_catalog(workspace)` provides closed JSON schemas through the
existing `ToolRegistry`. `ToolRuntime.call(name, arguments)` is the untrusted
dispatch boundary. Front ends must use the runtime rather than calling registry
handlers or workspace methods directly.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
from application.agent_api.workspace import HeadlessWorkspace

with TemporaryDirectory() as directory:
    workspace = HeadlessWorkspace(Path(directory), approval="review")
    runtime = ToolRuntime(build_command_catalog(workspace), workspace)
    outcome = runtime.call("new_project", {"name": "example", "duration_s": 1.0})
    assert outcome.ok, outcome.payload
    home = workspace.adapter.home_qpos.tolist()
    outcome = runtime.call("author_keyframes", {
        "duration_s": 1.0,
        "keyframes": [
            {"time_seconds": 0.0, "qpos": home},
            {"time_seconds": 1.0, "qpos": home},
        ],
    })
    assert outcome.ok, outcome.payload
    report = runtime.call("validate_motion", {"dt": 0.02})
    assert report.ok and report.payload["valid"], report.payload
    pending = runtime.call("accept_candidate", {})
    assert pending.payload["status"] == "pending_user_review"
    # A trusted user interface supplies approval; this argument is not a tool input.
    workspace.accept_candidate(user_approved=True)
    assert runtime.call("save_project", {}).ok
```

All work in this example stays inside the temporary project directory and
requires neither provider credentials nor API calls. For installed setup and
the existing GUI command, see [Installation](install.md).

## Implemented Catalog

Schemas, including required fields and limits, are returned by
`catalog.registry.definitions()`. These 23 tools are implemented:

| Tool | Inputs / purpose |
| --- | --- |
| `list_models` | Registered model keys and display names |
| `load_model` | `model_key`; refuses unresolved candidates or unsaved committed edits |
| `new_project` | `name`, optional `model_key`, `duration_s`; never overwrites a project |
| `open_project` | `name` within the caller's project root |
| `save_project` | Saves accepted motion and preserves opaque workspace fields |
| `get_status` | Model, revision, duration, candidate/project/generated presence |
| `get_robot_summary` | Compiled qpos width, Joint Angles, logical frames and End Effectors |
| `list_keyframes` | Optional `source` and `limit`; bounded logical frames/qpos times and totals |
| `get_pose` | `time_s`, optional candidate/committed source; read-only qpos sample |
| `author_keyframes` | 2–16 complete `{time_seconds, qpos}` anchors; optional duration; replaces candidate motion |
| `commit_keyframe` | `time_s`, optional phase; captures all logical targets and qpos atomically |
| `set_joint_angles` | `time_s`, unique `{joint, angle_rad}` values |
| `solve_ik` | Logical frame, time, position in meters, optional RPY in radians and locked End Effectors |
| `repeat_motion` | Start/end, additional copies, optional ping-pong; exact local repetition |
| `delete_keyframes` | Times; logical targets and qpos are deleted together |
| `edit_timeline` | Start/end and duration scale; delegates to the existing time-range planner |
| `apply_edit_spec` | Bounded compact-spec JSON text; parsed/validated locally, never evaluated as code |
| `generate_trajectory` | Optional dt/smoothing; existing exact backend, detached inputs and late-result rejection |
| `validate_motion` | Optional dt; structural/FK, sampled limits, root quaternions and blocking collisions |
| `undo` | Optional checkpoint ID; candidate restoration, not committed-document Undo |
| `accept_candidate` | Requires current validation, unchanged committed source and caller-configured approval |
| `discard_candidate` | Drops staged motion and its checkpoints without changing committed motion |
| `export_trajectory` | Format, plain file name, optional dt/source/DSMS speed; accepted motion only |

Mutating candidate tools accept optional `expected_revision`. Successful calls
return a revision and the checkpoint ID for the state before that call. Use the
revision from the latest outcome: rollback/restoration can advance the revision
even when motion values are restored. Checkpoints are opaque and session-owned;
Accept, Discard, or a project/model transition expires them.

`ToolOutcome.ok` means dispatch succeeded. For validation, inspect the separate
`payload["valid"]` field. An invalid motion report is an observation; it does
not authorize acceptance. Unknown tools and invalid arguments return structured
errors, not executable requests or arbitrary exception objects.

## Transactions And Approval

Every candidate command executes against a detached working copy. Runtime
preflight/rollback preserves earlier staged work after a failure, including its
provenance and previously validated generated path. Mutations invalidate current
validation and derived samples. Protected or user-owned affected values reject
the whole command; an application may explicitly configure user overrides for
an authorized workflow, but protection remains blocking.

Approval defaults to `review`; `accept_candidate` then reports
`pending_user_review`. Only trusted application code can supply the user's
approval. `auto_accept` and `auto_accept_and_export` are caller-selected policies,
not model-selected arguments. The latter allows an automated caller to accept
and then export; the caller still must invoke and verify the export tool before
reporting its overall run complete. W1 supplies no autonomous run orchestrator.
Accept uses the existing atomic session transaction and creates one history
transition. Late protection changes and committed motion edits reject acceptance.

Content identities include the document/session, revision, compiled model and
target/qpos contents, so legacy edits that bypass document revision tracking
still invalidate results. GUI generated-motion identity also tracks generation
settings. Headless generation binds its explicit dt/smoothing to the produced
sample array; a new generation replaces that artifact and requires revalidation.

## Sampling And File Contracts

Validation samples the selected path at a declared interval and includes its
exact anchors/endpoints. A generated candidate is validated on its generated
samples; otherwise the editable qpos path is used. Blocking interpolated
collisions and limit violations prevent acceptance. The report names unavailable
velocity-limit, dynamics, and continuous-collision evidence. Finite sampling is
not a proof of collision freedom between samples or physical feasibility.

Generated output must match its source. In the GUI, **Export source** in
Planning offers Automatic, editable Keyframes, and generated motion. Automatic
refuses stale generated output; choose editable explicitly or run Generate again.
Playback also avoids stale generated samples. Navigation/export/save do not
insert Keyframes. Old persisted qpos-only anchors are retained.

Agent output names resolve under the caller's output root, or the current
project's `exports` directory. Absolute paths, directory traversal, separators,
and escaping symlinks are rejected. Existing outputs require a new name rather
than implicit replacement. Project open/create take names within the project
root. Arbitrary model imports, shell commands and hardware operations are not
tools. Trusted Python callers may supply imported model registrations.

Exports reuse the existing Qpos, MuJoCo, DSMS and G1 mjlab writers; column order,
precision, quaternion conventions and filenames are unchanged. mjlab remains
G1-specific and rejects incompatible robots before writing output. DSMS speed
uses the explicit value or the saved project's display setting and affects
elapsed timestamps only. See [Data Formats](data_formats.md).

Default runtime bounds are 256 calls, 256 KiB request JSON, 64 KiB successful
result JSON, 32 candidate checkpoints and 10,001 motion samples. The runtime
serializes commands and checks cancellation around local calls; cancellation
inside lengthy IK/generation work and responsive GUI progress remain later
milestones. Errors are bounded summaries. There is no network listener or
provider execution in this layer.

## Verification And Remaining Scope

Run the isolated suite as described in [Testing](testing.md). Focused modules
are `test_agent_api.py`, `test_shared_motion_services.py`, and the two workflow
compatibility modules. A subprocess test rejects all Qt/GUI/OpenGL imports
while constructing and using the headless runtime. Static architecture checks
also forbid UI/rendering imports in `application/agent_api/`.

The full catalog still defers `render_views`, richer `compute_metrics`, mirror,
optional rollout and nested planning. F6 adds the remaining agent surfaces and
render/critique capabilities; F7/F9 add velocity/contact/dynamics quality work.
No unimplemented tools are registered. MCP packaging, client setup and GUI
attachment belong to W2/W3 rather than this Python API milestone.

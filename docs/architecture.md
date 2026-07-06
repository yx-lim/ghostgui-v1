# GhostGUI architecture

GhostGUI uses three dependency layers:

```text
gui -> application -> core
              \-> backend
```

- `core/` contains MuJoCo model loading, immutable metadata, mutable robot
  state, weighted IK tasks/solving, collision validation, asset conversion,
  quaternion helpers, and trajectory data. It does not import Qt widgets.
- `application/` owns project documents, per-model sessions, asynchronous model
  loading, and committed/preview timeline state.
- `gui/` contains the main window plus focused `panels/`, `viewers/`, and
  `widgets/`. Compatibility modules at the old import paths re-export renamed
  classes for downstream callers.
- `backend/` contains trajectory-generation adapters and the retained native C++
  sources. The MuJoCo trajectory backend delegates to the same weighted IK
  solver used by live 3D dragging; the analytic fallback is explicitly G1-only.

Robot-specific frame semantics belong only in `core.models.registry`. Target
tracks and backend bindings are generated from the selected adapter. A
`ProjectDocument` relates the target-frame trajectory and qpos timeline while
keeping their distinct meanings. Preview state remains transient.

The low-level OpenGL canvas renders and emits interactions. The high-level 3D
editor panel owns Qt composition, while extracted IK controls and preview state
controllers keep solver configuration and state transitions out of the canvas.


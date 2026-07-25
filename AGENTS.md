# Repository Instructions

## Project Map

- `application/` owns use cases, background jobs, projects, import/export, and
  trajectory generation.
- `core/` owns models, MuJoCo state, IK, collision, and trajectory math.
- `gui/` owns PySide6 widgets, interaction, help, and tutorials.
- `models/` contains bundled and user-imported model sources and assets.
- `scripts/` contains supported installation and launch entry points.
- `tests/` contains the automated contract and GUI suite.
- `docs/` contains current public documentation.

Read `docs/architecture.md` before changing ownership across these layers.

## User-Facing Terminology

Use **Keyframe**, **Commit Keyframe**, **Keyframe interval**, **Orange preview**,
**Preview Path**, **End Effector**, and **Joint Angles** in UI copy and public
documentation. Internal legacy identifiers do not need opportunistic renaming.

## Supported Commands

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_docs.py
ghostgui --model g1
```

Use the platform scripts in `scripts/` for installation. Verify commands before
adding them to documentation.

## Change Rules

- Preserve unrelated working-tree changes.
- Keep model-specific behavior in `core/models/registry.py` or the model layer.
- Keep reusable file and workflow logic out of Qt callbacks when an application
  service is appropriate.
- Do not modify robot assets, example CSV data, project fixtures, cache
  versions, or export contracts without focused tests and documentation.
- Update help and tutorial copy with any public workflow-label change.
- Add new public documentation to `docs/README.md`.
- Do not commit one-off diagnosis notes. Distill useful current behavior into
  the focused public documentation.

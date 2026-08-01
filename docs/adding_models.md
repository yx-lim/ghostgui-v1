# Adding Models

GhostGUI can import URDF and MuJoCo XML models into the checkout's model library.
For stable, named logical targets, developers can also add an explicit registry
entry.

## Import Through The Application

1. Open **File → Import → Model**.
2. Select a `.urdf` or `.xml` source.
3. If mesh resolution fails, select the external mesh folder when prompted.
4. Wait for GhostGUI to copy, prepare, and validate the model and its home pose.
5. Select the newly registered model.

The import is staged in a temporary directory and validated with MuJoCo before
the source and assets are moved into `models/`.

## Import Output

For a source named `example.urdf`, the default output resembles:

```text
models/
├── example.urdf
├── example.ghostgui.json   # only when the home pose was repaired
└── assets-example/
    └── copied-mesh-files
```

If that name already exists, GhostGUI appends a numeric suffix. Imported models
are discovered from the `models/` directory on later launches.

## Home-Pose Safety

URDF does not define a standard initial joint pose, so MuJoCo normally starts
URDF joints at zero. GhostGUI resolves a home pose, grounds the robot, and then
checks that pose for self-collision and disallowed environment collision before
the import is accepted.

When a generic imported model starts in collision, GhostGUI searches its joint
limits deterministically for a nearby collision-free pose. A successful repair
is stored in `<model>.ghostgui.json`; the source robot asset is not rewritten.
The stored pose is validated again whenever the model is loaded.

If no collision-free pose is found, the import fails instead of silently
allowing the initial contacts. Provide an MJCF `home` keyframe, correct the
collision geometry, or add a maintained registry entry with `home_joints`.

## Geometry Names And Collision Labels

GhostGUI does not require every vendor asset to hand-name every geometry. It
preserves source geom names and assigns each unnamed geometry a deterministic
effective identity based on its owning body, role, and ordinal, for example
`left_forearm__contact_1` or `link02__visual_1`.

For URDF imports, these names are written into the generated runtime MJCF cache.
For direct MJCF models, unnamed source geoms receive the same effective names at
runtime without rewriting the imported asset. Updating the cache naming schema
invalidates and regenerates old URDF runtime caches automatically.

Collision and preview messages preserve the model's original body-frame names,
such as `FL_thigh ↔ base` or `link02 ↔ link06`. A world-owned environment geom
uses its own source name, so the generated `ground` geom is reported as
`ground`, not `world`. Exact geom identities and penetration depth remain in
the technical status details. Multiple MuJoCo contact points between the same
pair of geoms are reported as one collision.

`RobotModelInfo.collision_blocking_penetration_m` sets the model-scale boundary
between advisory and blocking penetration. Optional
`allowed_contact_body_pairs` entries contain two original body-frame names and
a maximum allowed penetration in metres. Add an entry only after auditing the
specific collision geometry; broad adjacency exclusions can hide real contact.

Other user-facing model views can infer friendly body labels by splitting
underscores, namespaces, camel case, numbered links, and common quadruped
abbreviations (`FL`, `FR`, `RL`, and `RR`). Maintained models can use
`RobotModelInfo.body_labels` for ambiguous vendor terms; collision messages do
not replace the original frame names with these aliases.

## Mesh Resolution

GhostGUI tries:

- absolute and source-relative paths;
- ROS `package://` paths inferred from nearby directories;
- common `assets`, `dae`, and `meshes` directories;
- the optional mesh folder selected in the application;
- matching filenames below the selected mesh folder.

Supported mesh extensions are `.stl`, `.obj`, `.msh`, and convertible `.dae`.
COLLADA inputs must contain triangulated geometry and use `Z_UP`.

## Generic Discovery

An imported model without a registry entry uses generic hints:

- root candidates such as `base`, `base_link`, `trunk`, `pelvis`, `link00`,
  `root`, or `world`;
- root-joint candidates such as `floating_base`, `root`, or `freejoint`;
- bodies and sites exposed by the compiled MuJoCo model.

This is enough for direct joint editing and advanced target selection, but it
may not produce the preferred names or target set.

## Add A Registry Entry

For a maintained model, add a `RobotModelInfo` entry to
`core/models/registry.py` containing:

- a stable key and display name;
- model type and source path;
- root body and root joint candidates;
- logical frame aliases;
- optional body-label overrides for ambiguous source names;
- a model-scale blocking-penetration threshold and any audited contact-pair
  tolerances;
- any ROS package-to-directory mappings;
- optional home-joint values.

Add tests for model loading, logical-frame binding, qpos dimensions, and
representative visual assets. Update [Models](models.md) only after those tests
pass.

## Import Limitations

- Import writes into the repository checkout, so that directory must be
  writable.
- Texture images are not copied as a general material pipeline; prefer mesh
  geometry with direct material colors.
- Complex COLLADA features beyond triangulated geometry may require conversion
  to OBJ or STL before import.
- Logical frame semantics cannot always be inferred from arbitrary names.

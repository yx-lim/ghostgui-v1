"""Scene graph primitives for multi-actor GhostGUI projects."""

from .model import (
    ACTOR_KIND_OBJECT,
    ACTOR_KIND_ROBOT,
    SCENE_SCHEMA_VERSION,
    Actor,
    ActorRegistry,
    Constraint,
    ConstraintEndpoint,
    ConstraintGraph,
    Scene,
    SceneSelection,
    SceneTimeline,
    TrackRegistry,
    Transform,
    TransformKeyframe,
)
from .mesh import (
    MeshGeometry,
    SUPPORTED_OBJECT_MESH_EXTENSIONS,
    load_mesh_geometry,
    load_obj_geometry,
    load_stl_geometry,
)
from .runtime import (
    ComposedSceneMJCF,
    SceneBuildError,
    SceneRuntime,
    SceneRuntimePlan,
)

__all__ = [
    "ACTOR_KIND_OBJECT",
    "ACTOR_KIND_ROBOT",
    "SCENE_SCHEMA_VERSION",
    "Actor",
    "ActorRegistry",
    "Constraint",
    "ConstraintEndpoint",
    "ConstraintGraph",
    "ComposedSceneMJCF",
    "MeshGeometry",
    "Scene",
    "SceneBuildError",
    "SceneRuntime",
    "SceneRuntimePlan",
    "SceneSelection",
    "SceneTimeline",
    "SUPPORTED_OBJECT_MESH_EXTENSIONS",
    "TrackRegistry",
    "Transform",
    "TransformKeyframe",
    "load_mesh_geometry",
    "load_obj_geometry",
    "load_stl_geometry",
]

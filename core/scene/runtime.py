"""Runtime planning helpers for composed multi-actor scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import posixpath
import re
import xml.etree.ElementTree as ET

from .model import ACTOR_KIND_OBJECT, ACTOR_KIND_ROBOT, WORLD_FRAME_ID


MJCF_SECTION_ORDER = (
    "default",
    "asset",
    "worldbody",
    "contact",
    "equality",
    "tendon",
    "actuator",
    "sensor",
)

REFERENCE_ATTRIBUTES = {
    "body",
    "body1",
    "body2",
    "camera",
    "geom",
    "geom1",
    "geom2",
    "hfield",
    "joint",
    "joint1",
    "joint2",
    "material",
    "mesh",
    "name1",
    "name2",
    "objname",
    "refname",
    "site",
    "site1",
    "site2",
    "skin",
    "tendon",
    "texture",
}


@dataclass(frozen=True)
class SceneRuntimePlan:
    robot_actors: tuple
    object_actors: tuple
    namespaces: dict
    constraints: tuple

    def actor_namespace(self, actor_id):
        return self.namespaces[str(actor_id)]

    def namespaced_name(self, actor_id, frame_name):
        namespace = self.actor_namespace(actor_id)
        name = str(frame_name)
        if name.startswith(f"{namespace}/"):
            return name
        return f"{namespace}/{name}"


@dataclass(frozen=True)
class ComposedSceneMJCF:
    xml: str
    assets: dict
    plan: SceneRuntimePlan
    actor_bodies: dict
    frame_bodies: dict
    frame_sites: dict


class SceneBuildError(ValueError):
    def __init__(self, message, actor_id=None, constraint_id=None):
        self.actor_id = actor_id
        self.constraint_id = constraint_id
        parts = []
        if actor_id is not None:
            parts.append(f"actor_id={actor_id}")
        if constraint_id is not None:
            parts.append(f"constraint_id={constraint_id}")
        detail = f" ({', '.join(parts)})" if parts else ""
        super().__init__(f"{message}{detail}")


@dataclass(frozen=True)
class _ResolvedRobotModel:
    path: Path
    frame_bindings: dict


class SceneRuntime:
    """Builds deterministic runtime namespaces for scene composition.

    The current renderer still owns one live robot adapter. This planner keeps
    actor IDs, robot/object namespaces, and constraints explicit so future
    MuJoCo composition can share one collision/runtime contract.
    """

    def __init__(self, scene):
        self.scene = scene

    @staticmethod
    def namespace_for_actor(actor):
        compact_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(actor.id)).strip("_")
        return f"actor_{compact_id or 'unnamed'}"

    def build_plan(self):
        namespaces = {
            actor.id: self.namespace_for_actor(actor)
            for actor in self.scene.actors
        }
        return SceneRuntimePlan(
            robot_actors=tuple(
                actor for actor in self.scene.actors
                if actor.kind == ACTOR_KIND_ROBOT
            ),
            object_actors=tuple(
                actor for actor in self.scene.actors
                if actor.kind == ACTOR_KIND_OBJECT
            ),
            namespaces=namespaces,
            constraints=tuple(self.scene.constraints.constraints.values()),
        )

    def compose_mjcf(self, project_root=None, robot_model_resolver=None, time=None):
        """Return a namespaced MJCF document plus virtual assets.

        Robot actors are resolved to MJCF files and copied under a deterministic
        actor namespace. Object actors become MuJoCo bodies with free joints.
        The generated XML is intentionally inspectable so migrations and
        collision/runtime changes can be acceptance-tested before the editor
        swaps away from the current single live robot adapter.
        """

        plan = self.build_plan()
        composer = _MJCFComposer(
            scene=self.scene,
            plan=plan,
            project_root=project_root,
            robot_model_resolver=robot_model_resolver,
            time=self.scene.timeline.current_time if time is None else time,
        )
        return composer.compose()

    def build_mjcf(self, project_root=None, robot_model_resolver=None, time=None):
        return self.compose_mjcf(
            project_root=project_root,
            robot_model_resolver=robot_model_resolver,
            time=time,
        ).xml

    def compile_model(self, project_root=None, robot_model_resolver=None, time=None):
        composition = self.compose_mjcf(
            project_root=project_root,
            robot_model_resolver=robot_model_resolver,
            time=time,
        )
        import mujoco

        try:
            return mujoco.MjModel.from_xml_string(
                composition.xml,
                assets=composition.assets or None,
            )
        except Exception as exc:
            raise SceneBuildError(f"MuJoCo scene compilation failed: {exc}") from exc

    def active_collision_actor_ids(self):
        return tuple(
            actor.id for actor in self.scene.actors
            if actor.visible and not actor.locked
        )


class _MJCFComposer:
    def __init__(
        self,
        scene,
        plan,
        project_root=None,
        robot_model_resolver=None,
        time=0.0,
    ):
        self.scene = scene
        self.plan = plan
        self.project_root = (
            None if project_root is None else Path(project_root).expanduser().resolve()
        )
        self.robot_model_resolver = robot_model_resolver
        self.time = float(time)
        self.sections = {tag: ET.Element(tag) for tag in MJCF_SECTION_ORDER}
        self.compiler_attrs = {"autolimits": "true"}
        self.assets = {}
        self._asset_keys_by_path = {}
        self.actor_bodies = {}
        self.frame_bodies = {}
        self.frame_sites = {}
        self._body_elements_by_name = {}
        self._endpoint_site_names = {}

    def compose(self):
        for actor in self.plan.robot_actors:
            self._append_robot_actor(actor)
        for actor in self.plan.object_actors:
            self._append_object_actor(actor)
        self._append_constraints()

        root = ET.Element("mujoco", {"model": "ghostgui_scene"})
        ET.SubElement(root, "compiler", self.compiler_attrs)
        ET.SubElement(root, "option", {"timestep": "0.002"})
        for tag in MJCF_SECTION_ORDER:
            section = self.sections[tag]
            if len(section) or tag == "worldbody":
                root.append(section)
        return ComposedSceneMJCF(
            xml=ET.tostring(root, encoding="unicode"),
            assets=dict(self.assets),
            plan=self.plan,
            actor_bodies=dict(self.actor_bodies),
            frame_bodies=dict(self.frame_bodies),
            frame_sites=dict(self.frame_sites),
        )

    def _append_robot_actor(self, actor):
        try:
            resolved_model = self._resolve_robot_model(actor)
            model_path = resolved_model.path
            source_root = ET.parse(model_path).getroot()
        except SceneBuildError:
            raise
        except Exception as exc:
            raise SceneBuildError(str(exc), actor_id=actor.id) from exc
        if source_root.tag != "mujoco":
            raise SceneBuildError(
                "Composed MJCF currently requires MJCF robot sources; "
                f"{model_path} is not an MJCF file.",
                actor_id=actor.id,
            )

        namespace = self.plan.actor_namespace(actor.id)
        name_map, class_map, body_map, site_map = self._name_maps(source_root, namespace)
        source_asset_dirs = self._source_asset_dirs(source_root, model_path)
        root_default_class = None
        compiler = source_root.find("compiler")
        if compiler is not None:
            self._merge_compiler_attrs(compiler, actor)

        for section in source_root:
            if section.tag == "asset":
                for child in section:
                    copied = copy.deepcopy(child)
                    self._namespace_tree(
                        copied,
                        namespace,
                        name_map,
                        class_map,
                        model_path,
                        source_asset_dirs,
                    )
                    self.sections["asset"].append(copied)
            elif section.tag == "default":
                copied = copy.deepcopy(section)
                root_default_class = copied.get("class") or namespace
                if not copied.get("class"):
                    copied.set("class", root_default_class)
                self._namespace_tree(
                    copied,
                    namespace,
                    name_map,
                    class_map,
                    model_path,
                    source_asset_dirs,
                )
                root_default_class = copied.get("class")
                self.sections["default"].append(copied)
            elif section.tag == "worldbody":
                self._append_robot_worldbody(
                    actor,
                    section,
                    namespace,
                    root_default_class,
                    name_map,
                    class_map,
                    body_map,
                    site_map,
                    resolved_model.frame_bindings,
                    model_path,
                    source_asset_dirs,
                )
            elif section.tag in self.sections and section.tag not in {"default", "equality"}:
                copied = copy.deepcopy(section)
                self._namespace_tree(
                    copied,
                    namespace,
                    name_map,
                    class_map,
                    model_path,
                    source_asset_dirs,
                )
                for child in list(copied):
                    self.sections[section.tag].append(child)
            elif section.tag == "equality":
                copied = copy.deepcopy(section)
                self._namespace_tree(
                    copied,
                    namespace,
                    name_map,
                    class_map,
                    model_path,
                    source_asset_dirs,
                )
                for child in list(copied):
                    self.sections["equality"].append(child)

    def _append_robot_worldbody(
        self,
        actor,
        worldbody,
        namespace,
        root_default_class,
        name_map,
        class_map,
        body_map,
        site_map,
        frame_bindings,
        model_path,
        source_asset_dirs,
    ):
        frame = ET.Element(
            "frame",
            {
                "name": f"{namespace}/anchor",
                "pos": _float_text(actor.world_transform.position),
                "quat": _float_text(actor.world_transform.quaternion),
            },
        )
        top_level_body_names = []
        for child in worldbody:
            if child.tag != "body":
                continue
            copied = copy.deepcopy(child)
            self._namespace_tree(
                copied,
                namespace,
                name_map,
                class_map,
                model_path,
                source_asset_dirs,
            )
            if root_default_class:
                if copied.get("class") is None:
                    copied.set("class", root_default_class)
                if copied.get("childclass") is None:
                    copied.set("childclass", root_default_class)
            body_name = copied.get("name")
            if body_name:
                top_level_body_names.append(body_name)
            frame.append(copied)
            self._register_body_elements(copied)
        self.sections["worldbody"].append(frame)
        if not top_level_body_names:
            raise SceneBuildError(
                f"Robot model has no worldbody body elements: {actor.name}",
                actor_id=actor.id,
            )
        self.actor_bodies[actor.id] = top_level_body_names[0]
        self.frame_bodies[(actor.id, WORLD_FRAME_ID)] = top_level_body_names[0]
        for original, namespaced in body_map.items():
            self.frame_bodies[(actor.id, original)] = namespaced
            self._add_frame_alias(self.frame_bodies, actor.id, original, namespaced)
        for original, namespaced in site_map.items():
            self.frame_sites[(actor.id, original)] = namespaced
            self._add_frame_alias(self.frame_sites, actor.id, original, namespaced)
        self._apply_logical_frame_bindings(actor, body_map, site_map, frame_bindings)

    def _append_object_actor(self, actor):
        namespace = self.plan.actor_namespace(actor.id)
        transform = self.plan_actor_transform(actor)
        reference = actor.model_reference or {}
        body = ET.SubElement(
            self.sections["worldbody"],
            "body",
            {
                "name": namespace,
                "pos": _float_text(transform.position),
                "quat": _float_text(transform.quaternion),
            },
        )
        ET.SubElement(body, "freejoint", {"name": f"{namespace}/freejoint"})
        root_site_name = f"{namespace}/world"
        ET.SubElement(
            body,
            "site",
            {
                "name": root_site_name,
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )
        self.actor_bodies[actor.id] = namespace
        self.frame_bodies[(actor.id, WORLD_FRAME_ID)] = namespace
        self.frame_sites[(actor.id, WORLD_FRAME_ID)] = root_site_name
        self._body_elements_by_name[namespace] = body

        reference_type = reference.get("type")
        if reference_type == "primitive":
            try:
                geom_type = self._primitive_geom_type(reference)
                geom_size = self._primitive_geom_size(reference)
            except Exception as exc:
                raise SceneBuildError(str(exc), actor_id=actor.id) from exc
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{namespace}/geom",
                    "type": geom_type,
                    "size": _float_text(geom_size),
                    "rgba": _float_text(_rgba(reference.get("rgba"))),
                },
            )
        elif reference_type == "mesh":
            mesh_name = f"{namespace}/mesh"
            asset_path = str(reference.get("asset_path") or "")
            try:
                resolved_path = self._resolve_project_asset(asset_path)
                self.assets[asset_path] = resolved_path.read_bytes()
                scale = _scale(reference.get("scale"))
                if any(value <= 0.0 for value in scale):
                    raise ValueError("Mesh scale values must be positive.")
            except SceneBuildError:
                raise
            except Exception as exc:
                raise SceneBuildError(str(exc), actor_id=actor.id) from exc
            ET.SubElement(
                self.sections["asset"],
                "mesh",
                {
                    "name": mesh_name,
                    "file": asset_path,
                    "scale": _float_text(scale),
                },
            )
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{namespace}/geom",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "rgba": _float_text(_rgba(reference.get("rgba"))),
                },
            )
        else:
            raise SceneBuildError(
                f"Unsupported object model reference type for {actor.name}: "
                f"{reference_type}",
                actor_id=actor.id,
            )

    def plan_actor_transform(self, actor):
        return self.scene.tracks.object_transform_at(actor, self.time)

    def _append_constraints(self):
        for constraint in self.plan.constraints:
            if not constraint.enabled:
                continue
            if constraint.kind not in {"attachment", "weld"}:
                continue
            source_site = self._site_for_endpoint(constraint.source)
            target_site = self._site_for_endpoint(constraint.target)
            if source_site is None or target_site is None:
                raise SceneBuildError(
                    "Constraint endpoint could not be resolved.",
                    constraint_id=constraint.id,
                )
            ET.SubElement(
                self.sections["equality"],
                "weld",
                {
                    "name": f"constraint/{constraint.id}",
                    "site1": source_site,
                    "site2": target_site,
                },
            )

    def _site_for_endpoint(self, endpoint):
        key = (endpoint.actor_id, endpoint.frame_id or WORLD_FRAME_ID)
        if key in self.frame_sites:
            return self.frame_sites[key]
        body_name = self.frame_bodies.get(key)
        if body_name is None:
            return None
        return self._site_for_body(
            body_name,
            endpoint.actor_id,
            endpoint.frame_id or WORLD_FRAME_ID,
        )

    def _site_for_body(self, body_name, actor_id, frame_id):
        key = (str(actor_id), str(frame_id), str(body_name))
        if key in self._endpoint_site_names:
            return self._endpoint_site_names[key]
        body = self._body_elements_by_name.get(body_name)
        if body is None:
            return None
        namespace = self.plan.actor_namespace(actor_id)
        site_name = _unique_site_name(
            self.frame_sites,
            f"{body_name}/{_site_token(frame_id)}",
            namespace,
        )
        ET.SubElement(
            body,
            "site",
            {
                "name": site_name,
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )
        self.frame_sites[(str(actor_id), str(frame_id))] = site_name
        self._endpoint_site_names[key] = site_name
        return site_name

    def _resolve_robot_model(self, actor):
        if self.robot_model_resolver is not None:
            try:
                resolved = self.robot_model_resolver(actor)
            except TypeError:
                resolved = self.robot_model_resolver(actor.model_reference or {})
            if resolved is not None:
                return _resolved_model(resolved)
        reference = actor.model_reference or {}
        model_path = reference.get("model_path")
        if model_path:
            return _ResolvedRobotModel(Path(model_path).expanduser().resolve(), {})
        model_key = reference.get("model_key")
        try:
            from core.models import ROBOT_MODELS
        except Exception as exc:
            raise ValueError("Robot model registry is unavailable.") from exc
        model_info = ROBOT_MODELS.get(model_key)
        if model_info is None:
            raise SceneBuildError(
                f"Unknown robot model for actor {actor.name}: {model_key}",
                actor_id=actor.id,
            )
        return _ResolvedRobotModel(Path(model_info.model_path).expanduser().resolve(), {})

    def _resolve_project_asset(self, asset_path):
        if not asset_path:
            raise ValueError("Mesh object is missing an asset path.")
        path = Path(asset_path)
        if path.is_absolute():
            return path
        if self.project_root is None:
            raise SceneBuildError(
                f"Project root is required to resolve mesh asset: {asset_path}",
            )
        resolved = self.project_root / path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _source_asset_dirs(self, source_root, model_path):
        source_dir = Path(model_path).parent
        dirs = [source_dir]
        compiler = source_root.find("compiler")
        if compiler is not None:
            for key in ("assetdir", "meshdir", "texturedir"):
                value = compiler.get(key)
                if value:
                    dirs.append((source_dir / value).resolve())
        return tuple(dict.fromkeys(dirs))

    def _name_maps(self, source_root, namespace):
        name_map = {}
        class_map = {}
        body_map = {}
        site_map = {}
        for element in source_root.iter():
            name = element.get("name")
            if name:
                namespaced = _namespaced(namespace, name)
                name_map[name] = namespaced
                if element.tag == "body":
                    body_map[name] = namespaced
                elif element.tag == "site":
                    site_map[name] = namespaced
            if element.tag == "default":
                class_name = element.get("class")
                if class_name:
                    class_map[class_name] = _namespaced(namespace, class_name)
        return name_map, class_map, body_map, site_map

    def _namespace_tree(
        self,
        element,
        namespace,
        name_map,
        class_map,
        model_path,
        source_asset_dirs,
    ):
        for item in element.iter():
            if item.get("name"):
                item.set("name", _map_name(item.get("name"), name_map, namespace))
            if item.get("file"):
                item.set(
                    "file",
                    self._register_source_asset(
                        item.get("file"),
                        namespace,
                        model_path,
                        source_asset_dirs,
                    ),
                )
            for attr, value in list(item.attrib.items()):
                if attr == "name" or attr == "file":
                    continue
                if attr in {"class", "childclass"}:
                    item.set(attr, class_map.get(value, value))
                elif attr in REFERENCE_ATTRIBUTES:
                    item.set(attr, _map_reference_name(value, name_map))

    def _merge_compiler_attrs(self, compiler, actor):
        for key, value in compiler.attrib.items():
            if key in {"assetdir", "meshdir", "texturedir"}:
                continue
            current = self.compiler_attrs.get(key)
            if current is None:
                self.compiler_attrs[key] = value
            elif current != value:
                raise SceneBuildError(
                    f"Conflicting MJCF compiler setting {key!r}: "
                    f"{current!r} vs {value!r}",
                    actor_id=actor.id,
                )

    def _register_body_elements(self, element):
        for item in element.iter("body"):
            name = item.get("name")
            if name:
                self._body_elements_by_name[name] = item

    @staticmethod
    def _add_frame_alias(mapping, actor_id, original, namespaced):
        if "/" not in str(original):
            return
        short_name = str(original).rsplit("/", 1)[-1]
        mapping.setdefault((actor_id, short_name), namespaced)

    def _apply_logical_frame_bindings(self, actor, body_map, site_map, frame_bindings):
        for logical_name, binding in (frame_bindings or {}).items():
            kind, frame_name = _normal_frame_binding(binding)
            if kind == "site" and frame_name in site_map:
                self.frame_sites[(actor.id, str(logical_name))] = site_map[frame_name]
            elif kind == "body" and frame_name in body_map:
                self.frame_bodies[(actor.id, str(logical_name))] = body_map[frame_name]

    def _register_source_asset(
        self,
        file_value,
        namespace,
        model_path,
        source_asset_dirs,
    ):
        path = Path(file_value)
        if path.is_absolute():
            resolved = path
        else:
            resolved = None
            for directory in source_asset_dirs:
                candidate = directory / file_value
                if candidate.is_file():
                    resolved = candidate
                    break
            if resolved is None:
                resolved = Path(model_path).parent / file_value
        resolved = resolved.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved in self._asset_keys_by_path:
            return self._asset_keys_by_path[resolved]
        key = _unique_asset_key(
            self.assets,
            posixpath.join(namespace, Path(file_value).name),
        )
        self.assets[key] = resolved.read_bytes()
        self._asset_keys_by_path[resolved] = key
        return key

    @staticmethod
    def _primitive_geom_type(reference):
        shape = str(reference.get("shape") or "box").lower()
        if shape in {"box", "sphere", "cylinder"}:
            return shape
        raise ValueError(f"Unsupported primitive object shape: {shape}")

    @staticmethod
    def _primitive_geom_size(reference):
        shape = str(reference.get("shape") or "box").lower()
        size = _size3(reference.get("size"))
        if shape == "box":
            return [max(0.001, value * 0.5) for value in size]
        if shape == "sphere":
            return [max(0.001, size[0])]
        if shape == "cylinder":
            return [max(0.001, size[0]), max(0.001, size[2] * 0.5)]
        raise ValueError(f"Unsupported primitive object shape: {shape}")


def _float_text(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def _size3(values):
    values = list(values or (0.2, 0.2, 0.2))[:3]
    while len(values) < 3:
        values.append(values[-1] if values else 0.2)
    return [float(value) for value in values]


def _scale(values):
    values = list(values or (1.0, 1.0, 1.0))[:3]
    while len(values) < 3:
        values.append(1.0)
    return [float(value) for value in values]


def _rgba(values):
    values = list(values or (0.20, 0.58, 0.88, 1.0))[:4]
    while len(values) < 4:
        values.append(1.0)
    return [float(value) for value in values]


def _namespaced(namespace, name):
    name = str(name)
    if name.startswith(f"{namespace}/"):
        return name
    return f"{namespace}/{name}"


def _map_name(value, name_map, namespace):
    if value in name_map:
        return name_map[value]
    return _namespaced(namespace, value)


def _map_reference_name(value, name_map):
    return name_map.get(value, value)


def _unique_asset_key(existing, desired):
    key = str(desired).replace("\\", "/")
    if key not in existing:
        return key
    stem, suffix = posixpath.splitext(key)
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in existing:
            return candidate
        counter += 1


def _resolved_model(value):
    frame_bindings = {}
    if isinstance(value, (str, Path)):
        return _ResolvedRobotModel(Path(value).expanduser().resolve(), frame_bindings)
    frame_bindings = getattr(value, "frame_bindings", None) or getattr(
        value,
        "logical_frame_bindings",
        None,
    ) or {}
    for attr in ("runtime_mjcf_path", "runtime_model_path", "model_path", "path"):
        path = getattr(value, attr, None)
        if path:
            return _ResolvedRobotModel(
                Path(path).expanduser().resolve(),
                dict(frame_bindings),
            )
    if isinstance(value, dict):
        frame_bindings = value.get("frame_bindings") or value.get(
            "logical_frame_bindings"
        ) or {}
        for key in ("runtime_mjcf_path", "runtime_model_path", "model_path", "path"):
            path = value.get(key)
            if path:
                return _ResolvedRobotModel(
                    Path(path).expanduser().resolve(),
                    dict(frame_bindings),
                )
    raise TypeError(f"Robot resolver returned an unsupported value: {value!r}")


def _normal_frame_binding(binding):
    if isinstance(binding, (list, tuple)) and len(binding) >= 2:
        return str(binding[0]), str(binding[1])
    if isinstance(binding, dict):
        return str(binding.get("kind") or binding.get("type") or "body"), str(
            binding.get("name") or binding.get("frame") or ""
        )
    return "body", str(binding)


def _site_token(value):
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return token or "frame"


def _unique_site_name(existing_sites, desired, namespace):
    existing = {name for (_actor_id, _frame_id), name in existing_sites.items()}
    site_name = desired
    if site_name not in existing:
        return site_name
    counter = 2
    while True:
        site_name = f"{namespace}/{_site_token(desired)}_{counter}"
        if site_name not in existing:
            return site_name
        counter += 1

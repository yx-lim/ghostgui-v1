"""Plain-Python scene model used by projects and future multi-actor UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import uuid

from core.trajectory import TargetFrame, Trajectory


SCENE_SCHEMA_VERSION = 1
ACTOR_KIND_ROBOT = "robot"
ACTOR_KIND_OBJECT = "object"
WORLD_FRAME_ID = "world"


def new_actor_id():
    return str(uuid.uuid4())


def _json_safe_dict(value):
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _float_tuple(values, length, default):
    if not isinstance(values, (list, tuple)):
        return default
    result = []
    for index in range(length):
        try:
            result.append(float(values[index]))
        except (IndexError, TypeError, ValueError):
            result.append(float(default[index]))
    return tuple(result)


def _normal_quaternion(values):
    quat = _float_tuple(values, 4, (1.0, 0.0, 0.0, 0.0))
    norm = math.sqrt(sum(value * value for value in quat))
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quat)


def _lerp(a, b, alpha):
    return a + (b - a) * alpha


def _multiply_quaternions(left, right):
    w1, x1, y1, z1 = _normal_quaternion(left)
    w2, x2, y2, z2 = _normal_quaternion(right)
    return _normal_quaternion((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def _rotate_vector(quaternion, vector):
    quaternion = _normal_quaternion(quaternion)
    inverse = (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    rotated = _multiply_quaternions_raw(
        _multiply_quaternions_raw(quaternion, (0.0, *vector)),
        inverse,
    )
    return tuple(float(value) for value in rotated[1:])


def _multiply_quaternions_raw(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


@dataclass(frozen=True)
class Transform:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return cls.identity()
        return cls(
            position=_float_tuple(
                data.get("position"), 3, (0.0, 0.0, 0.0)
            ),
            quaternion=_normal_quaternion(data.get("quaternion")),
        )

    def to_dict(self):
        return {
            "position": [float(value) for value in self.position],
            "quaternion": [float(value) for value in self.quaternion],
        }

    def interpolate(self, other, alpha):
        alpha = max(0.0, min(1.0, float(alpha)))
        position = tuple(
            _lerp(float(a), float(b), alpha)
            for a, b in zip(self.position, other.position)
        )
        q0 = self.quaternion
        q1 = other.quaternion
        if sum(a * b for a, b in zip(q0, q1)) < 0.0:
            q1 = tuple(-value for value in q1)
        quaternion = _normal_quaternion(
            tuple(_lerp(a, b, alpha) for a, b in zip(q0, q1))
        )
        return Transform(position=position, quaternion=quaternion)

    def compose(self, other):
        rotated = _rotate_vector(self.quaternion, other.position)
        return Transform(
            position=tuple(
                float(origin) + float(offset)
                for origin, offset in zip(self.position, rotated)
            ),
            quaternion=_multiply_quaternions(
                self.quaternion,
                other.quaternion,
            ),
        )

    def inverse(self):
        inverse_quaternion = (
            self.quaternion[0],
            -self.quaternion[1],
            -self.quaternion[2],
            -self.quaternion[3],
        )
        inverse_position = _rotate_vector(
            inverse_quaternion,
            tuple(-float(value) for value in self.position),
        )
        return Transform(inverse_position, inverse_quaternion)


@dataclass
class Actor:
    id: str
    kind: str
    name: str
    model_reference: dict = field(default_factory=dict)
    world_transform: Transform = field(default_factory=Transform.identity)
    visible: bool = True
    locked: bool = False
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind,
        name,
        model_reference=None,
        world_transform=None,
        actor_id=None,
        visible=True,
        locked=False,
        metadata=None,
    ):
        if kind not in {ACTOR_KIND_ROBOT, ACTOR_KIND_OBJECT}:
            raise ValueError(f"Unsupported actor kind: {kind}")
        return cls(
            id=str(actor_id or new_actor_id()),
            kind=str(kind),
            name=str(name or kind),
            model_reference=_json_safe_dict(model_reference),
            world_transform=world_transform or Transform.identity(),
            visible=bool(visible),
            locked=bool(locked),
            metadata=_json_safe_dict(metadata),
        )

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Scene actor data must be a JSON object.")
        return cls.create(
            kind=data.get("kind"),
            name=data.get("name"),
            actor_id=data.get("id"),
            model_reference=data.get("model_reference"),
            world_transform=Transform.from_dict(data.get("world_transform")),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            metadata=data.get("metadata"),
        )

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "model_reference": self.model_reference,
            "world_transform": self.world_transform.to_dict(),
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "metadata": self.metadata,
        }

    def duplicate(self, name=None):
        return Actor.create(
            kind=self.kind,
            name=name or f"{self.name} copy",
            model_reference=dict(self.model_reference),
            world_transform=self.world_transform,
            visible=self.visible,
            locked=False,
            metadata=dict(self.metadata),
        )


class ActorRegistry:
    def __init__(self, actors=None):
        self.actors = {}
        for actor in actors or []:
            self.add(actor)

    def __contains__(self, actor_id):
        return str(actor_id) in self.actors

    def __iter__(self):
        return iter(self.actors.values())

    def add(self, actor):
        if actor.id in self.actors:
            raise ValueError(f"Duplicate actor id: {actor.id}")
        self.actors[actor.id] = actor
        return actor

    def require(self, actor_id):
        try:
            return self.actors[str(actor_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown scene actor: {actor_id}") from exc

    def get(self, actor_id, default=None):
        return self.actors.get(str(actor_id), default)

    def remove(self, actor_id):
        return self.actors.pop(str(actor_id))

    def robots(self):
        return [
            actor for actor in self.actors.values()
            if actor.kind == ACTOR_KIND_ROBOT
        ]

    def objects(self):
        return [
            actor for actor in self.actors.values()
            if actor.kind == ACTOR_KIND_OBJECT
        ]

    def duplicate(self, actor_id, name=None):
        actor = self.require(actor_id).duplicate(name=name)
        self.add(actor)
        return actor

    @classmethod
    def from_list(cls, items):
        return cls(Actor.from_dict(item) for item in (items or []))

    def to_list(self):
        return [actor.to_dict() for actor in self.actors.values()]


@dataclass(frozen=True)
class TransformKeyframe:
    time: float
    transform: Transform
    interpolation: str = "linear"

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Transform keyframe data must be a JSON object.")
        return cls(
            time=float(data.get("time", 0.0)),
            transform=Transform.from_dict(data.get("transform")),
            interpolation=str(data.get("interpolation") or "linear"),
        )

    def to_dict(self):
        return {
            "time": float(self.time),
            "transform": self.transform.to_dict(),
            "interpolation": self.interpolation,
        }


class TrackRegistry:
    """Actor-qualified animation tracks.

    Robot actors store named target-frame tracks. Object actors store transform
    tracks. Additional joint/qpos tracks can be layered in without changing
    actor identity or project migration rules.
    """

    def __init__(
        self,
        robot_targets=None,
        object_transforms=None,
        joint_tracks=None,
    ):
        self.robot_targets = robot_targets or {}
        self.object_transforms = object_transforms or {}
        self.joint_tracks = joint_tracks or {}

    def set_robot_trajectory(self, actor_id, trajectory):
        tracks = {}
        for frame_name, frames in trajectory.tracks.items():
            tracks[str(frame_name)] = [
                TargetFrame.from_dict(frame.to_dict()) for frame in frames
            ]
        self.robot_targets[str(actor_id)] = tracks

    def robot_trajectory(self, actor_id):
        trajectory = Trajectory()
        tracks = self.robot_targets.get(str(actor_id), {})
        trajectory.tracks = {str(name): [] for name in tracks}
        if not trajectory.tracks:
            trajectory = Trajectory()
        for frame_name, frames in tracks.items():
            trajectory.ensure_track(frame_name)
            for frame in frames:
                trajectory.add_frame(TargetFrame.from_dict(frame.to_dict()))
        return trajectory

    def set_robot_tracks_from_dict(self, actor_id, data):
        trajectory = Trajectory()
        trajectory.load_project_dict(data)
        self.set_robot_trajectory(actor_id, trajectory)

    def add_object_transform_keyframe(self, actor_id, keyframe):
        actor_id = str(actor_id)
        track = self.object_transforms.setdefault(actor_id, [])
        track[:] = [
            item for item in track
            if abs(float(item.time) - float(keyframe.time)) > 1e-9
        ]
        track.append(keyframe)
        track.sort(key=lambda item: item.time)
        return keyframe

    def set_robot_qpos_timeline(self, actor_id, timeline):
        actor_id = str(actor_id)
        if timeline is None:
            self.joint_tracks.pop(actor_id, None)
            return
        self.joint_tracks[actor_id] = {
            "qpos": [
                {
                    "time": float(time),
                    "values": [float(value) for value in timeline.get_state(time)],
                }
                for time in timeline.times()
            ]
        }

    def load_robot_qpos_timeline(self, actor_id, timeline):
        actor_tracks = self.joint_tracks.get(str(actor_id), {})
        keyframes = actor_tracks.get("qpos", []) if isinstance(actor_tracks, dict) else []
        if not keyframes or timeline is None:
            return False
        timeline.states.clear()
        for keyframe in keyframes:
            timeline.set_state(
                float(keyframe.get("time", 0.0)),
                keyframe.get("values", []),
            )
        return True

    def object_transform_at(self, actor, time):
        track = self.object_transforms.get(str(actor.id), [])
        if not track:
            return actor.world_transform
        time = float(time)
        if time <= track[0].time:
            return track[0].transform
        if time >= track[-1].time:
            return track[-1].transform
        for previous, next_item in zip(track, track[1:]):
            if previous.time <= time <= next_item.time:
                duration = next_item.time - previous.time
                alpha = 0.0 if abs(duration) < 1e-12 else (
                    (time - previous.time) / duration
                )
                return previous.transform.interpolate(next_item.transform, alpha)
        return track[-1].transform

    def remove_actor(self, actor_id):
        actor_id = str(actor_id)
        self.robot_targets.pop(actor_id, None)
        self.object_transforms.pop(actor_id, None)
        self.joint_tracks.pop(actor_id, None)

    def duplicate_actor_tracks(self, source_actor_id, target_actor_id):
        source_actor_id = str(source_actor_id)
        target_actor_id = str(target_actor_id)
        if source_actor_id in self.robot_targets:
            trajectory = self.robot_trajectory(source_actor_id)
            self.set_robot_trajectory(target_actor_id, trajectory)
        if source_actor_id in self.object_transforms:
            self.object_transforms[target_actor_id] = [
                TransformKeyframe(item.time, item.transform, item.interpolation)
                for item in self.object_transforms[source_actor_id]
            ]
        if source_actor_id in self.joint_tracks:
            self.joint_tracks[target_actor_id] = {
                str(name): [dict(item) for item in values]
                for name, values in self.joint_tracks[source_actor_id].items()
            }

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        robot_targets = {}
        for actor_id, tracks in data.get("robot_targets", {}).items():
            robot_targets[str(actor_id)] = {}
            for frame_name, frames in (tracks or {}).items():
                robot_targets[str(actor_id)][str(frame_name)] = [
                    TargetFrame.from_dict(dict(frame)) for frame in frames
                ]
        object_transforms = {}
        for actor_id, frames in data.get("object_transforms", {}).items():
            object_transforms[str(actor_id)] = [
                TransformKeyframe.from_dict(frame) for frame in (frames or [])
            ]
        return cls(
            robot_targets=robot_targets,
            object_transforms=object_transforms,
            joint_tracks=_json_safe_dict(data.get("joint_tracks")),
        )

    def to_dict(self):
        return {
            "robot_targets": {
                actor_id: {
                    frame_name: [frame.to_dict() for frame in frames]
                    for frame_name, frames in tracks.items()
                }
                for actor_id, tracks in self.robot_targets.items()
            },
            "object_transforms": {
                actor_id: [frame.to_dict() for frame in frames]
                for actor_id, frames in self.object_transforms.items()
            },
            "joint_tracks": self.joint_tracks,
        }


@dataclass(frozen=True)
class ConstraintEndpoint:
    actor_id: str
    frame_id: str = WORLD_FRAME_ID

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Constraint endpoint must be a JSON object.")
        return cls(
            actor_id=str(data.get("actor_id") or ""),
            frame_id=str(data.get("frame_id") or WORLD_FRAME_ID),
        )

    def to_dict(self):
        return {"actor_id": self.actor_id, "frame_id": self.frame_id}


@dataclass
class Constraint:
    id: str
    kind: str
    source: ConstraintEndpoint
    target: ConstraintEndpoint
    enabled: bool = True
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, kind, source, target, constraint_id=None, metadata=None):
        return cls(
            id=str(constraint_id or uuid.uuid4()),
            kind=str(kind),
            source=source,
            target=target,
            enabled=True,
            metadata=_json_safe_dict(metadata),
        )

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Constraint data must be a JSON object.")
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            kind=str(data.get("kind") or "attachment"),
            source=ConstraintEndpoint.from_dict(data.get("source")),
            target=ConstraintEndpoint.from_dict(data.get("target")),
            enabled=bool(data.get("enabled", True)),
            metadata=_json_safe_dict(data.get("metadata")),
        )

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "enabled": bool(self.enabled),
            "metadata": self.metadata,
        }


class ConstraintGraph:
    def __init__(self, constraints=None):
        self.constraints = {}
        for constraint in constraints or []:
            self.add(constraint)

    def add(self, constraint):
        self.constraints[constraint.id] = constraint
        return constraint

    def add_attachment(self, source_actor_id, source_frame_id, target_actor_id, target_frame_id):
        return self.add(Constraint.create(
            "attachment",
            ConstraintEndpoint(str(source_actor_id), str(source_frame_id)),
            ConstraintEndpoint(str(target_actor_id), str(target_frame_id)),
        ))

    def add_weld(self, source_actor_id, source_frame_id, target_actor_id, target_frame_id):
        return self.add(Constraint.create(
            "weld",
            ConstraintEndpoint(str(source_actor_id), str(source_frame_id)),
            ConstraintEndpoint(str(target_actor_id), str(target_frame_id)),
        ))

    def remove(self, constraint_id):
        return self.constraints.pop(str(constraint_id))

    def remove_actor(self, actor_id):
        actor_id = str(actor_id)
        for constraint_id in list(self.constraints):
            constraint = self.constraints[constraint_id]
            if (
                constraint.source.actor_id == actor_id
                or constraint.target.actor_id == actor_id
            ):
                del self.constraints[constraint_id]

    def for_actor(self, actor_id):
        actor_id = str(actor_id)
        return [
            constraint for constraint in self.constraints.values()
            if (
                constraint.source.actor_id == actor_id
                or constraint.target.actor_id == actor_id
            )
        ]

    @classmethod
    def from_list(cls, items):
        return cls(Constraint.from_dict(item) for item in (items or []))

    def to_list(self):
        return [constraint.to_dict() for constraint in self.constraints.values()]


@dataclass
class SceneSelection:
    actor_id: str | None = None
    frame_id: str | None = None
    track_id: str | None = None

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        return cls(
            actor_id=data.get("actor_id"),
            frame_id=data.get("frame_id"),
            track_id=data.get("track_id"),
        )

    def to_dict(self):
        return {
            "actor_id": self.actor_id,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
        }


@dataclass
class SceneTimeline:
    current_time: float = 0.0
    duration: float = 5.0
    selected_track_id: str | None = None

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        return cls(
            current_time=float(data.get("current_time", 0.0)),
            duration=float(data.get("duration", 5.0)),
            selected_track_id=data.get("selected_track_id"),
        )

    def to_dict(self):
        return {
            "current_time": float(self.current_time),
            "duration": float(self.duration),
            "selected_track_id": self.selected_track_id,
        }


class Scene:
    def __init__(
        self,
        actors=None,
        tracks=None,
        constraints=None,
        selection=None,
        timeline=None,
        metadata=None,
    ):
        self.actors = actors or ActorRegistry()
        self.tracks = tracks or TrackRegistry()
        self.constraints = constraints or ConstraintGraph()
        self.selection = selection or SceneSelection()
        self.timeline = timeline or SceneTimeline()
        self.metadata = _json_safe_dict(metadata)

    @classmethod
    def create(cls):
        return cls()

    @classmethod
    def single_robot(cls, model_key, model_name=None, trajectory=None, actor_id=None):
        scene = cls.create()
        robot = scene.add_robot(model_key, model_name=model_name, actor_id=actor_id)
        if trajectory is not None:
            scene.tracks.set_robot_trajectory(robot.id, trajectory)
        return scene

    @classmethod
    def from_legacy(cls, model_key, model_name=None, trajectory_data=None, workspace=None):
        scene = cls.single_robot(model_key, model_name=model_name)
        robot_id = scene.active_robot_id()
        if trajectory_data is not None and robot_id is not None:
            scene.tracks.set_robot_tracks_from_dict(robot_id, trajectory_data)
        workspace = workspace if isinstance(workspace, dict) else {}
        scene.timeline.current_time = float(workspace.get("current_time", 0.0))
        scene.timeline.duration = float(workspace.get("timeline_duration", 5.0))
        selected_frame = workspace.get("selected_frame")
        if selected_frame:
            scene.selection.frame_id = str(selected_frame)
        scene.selection.actor_id = robot_id
        return scene

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        version = int(data.get("schema_version", 0) or 0)
        if version not in (0, SCENE_SCHEMA_VERSION):
            raise ValueError(f"Unsupported GhostGUI scene schema: {version}")
        scene = cls(
            actors=ActorRegistry.from_list(data.get("actors", [])),
            tracks=TrackRegistry.from_dict(data.get("tracks", {})),
            constraints=ConstraintGraph.from_list(data.get("constraints", [])),
            selection=SceneSelection.from_dict(data.get("selection")),
            timeline=SceneTimeline.from_dict(data.get("timeline")),
            metadata=data.get("metadata"),
        )
        if scene.selection.actor_id not in scene.actors.actors:
            scene.selection.actor_id = scene.active_robot_id()
        return scene

    def to_dict(self):
        return {
            "schema_version": SCENE_SCHEMA_VERSION,
            "actors": self.actors.to_list(),
            "tracks": self.tracks.to_dict(),
            "constraints": self.constraints.to_list(),
            "selection": self.selection.to_dict(),
            "timeline": self.timeline.to_dict(),
            "metadata": self.metadata,
        }

    def add_robot(self, model_key, model_name=None, actor_id=None, name=None):
        actor = Actor.create(
            ACTOR_KIND_ROBOT,
            name or model_name or model_key,
            actor_id=actor_id,
            model_reference={
                "type": "robot_model",
                "model_key": str(model_key),
                "model_name": str(model_name or model_key),
            },
        )
        self.actors.add(actor)
        self.tracks.robot_targets.setdefault(actor.id, {})
        if self.selection.actor_id is None:
            self.selection.actor_id = actor.id
        return actor

    def add_object(
        self,
        name="Object",
        shape="box",
        size=None,
        color=None,
        transform=None,
        actor_id=None,
        locked=False,
    ):
        size = size or [0.20, 0.20, 0.20]
        color = color or [0.20, 0.58, 0.88, 1.0]
        actor = Actor.create(
            ACTOR_KIND_OBJECT,
            name,
            actor_id=actor_id,
            model_reference={
                "type": "primitive",
                "shape": str(shape),
                "size": [float(value) for value in size],
                "rgba": [float(value) for value in color],
            },
            world_transform=transform or Transform.identity(),
            locked=locked,
        )
        self.actors.add(actor)
        self.tracks.object_transforms.setdefault(actor.id, [])
        return actor

    def add_mesh_object(
        self,
        name="Object",
        asset_path=None,
        mesh_format=None,
        scale=None,
        color=None,
        transform=None,
        actor_id=None,
        locked=False,
        metadata=None,
    ):
        if not asset_path:
            raise ValueError("Mesh object actors require an asset path.")
        scale = list(scale or [1.0, 1.0, 1.0])[:3]
        while len(scale) < 3:
            scale.append(1.0)
        scale = [float(value) for value in scale]
        if any(value <= 0.0 for value in scale):
            raise ValueError("Mesh object scale values must be positive.")
        color = color or [0.20, 0.58, 0.88, 1.0]
        actor = Actor.create(
            ACTOR_KIND_OBJECT,
            name,
            actor_id=actor_id,
            model_reference={
                "type": "mesh",
                "asset_path": str(asset_path),
                "mesh_format": str(mesh_format or "").lstrip("."),
                "scale": scale,
                "rgba": [float(value) for value in color],
            },
            world_transform=transform or Transform.identity(),
            locked=locked,
            metadata=metadata,
        )
        self.actors.add(actor)
        self.tracks.object_transforms.setdefault(actor.id, [])
        return actor

    def active_robot_id(self):
        selected = self.actors.get(self.selection.actor_id)
        if selected is not None and selected.kind == ACTOR_KIND_ROBOT:
            return selected.id
        robots = self.actors.robots()
        return robots[0].id if robots else None

    def active_robot(self):
        actor_id = self.active_robot_id()
        return self.actors.get(actor_id) if actor_id else None

    def select_actor(self, actor_id, frame_id=None, track_id=None):
        self.actors.require(actor_id)
        self.selection = SceneSelection(
            actor_id=str(actor_id),
            frame_id=frame_id,
            track_id=track_id,
        )
        return self.selection

    def set_actor_visibility(self, actor_id, visible):
        actor = self.actors.require(actor_id)
        actor.visible = bool(visible)
        return actor

    def set_actor_locked(self, actor_id, locked):
        actor = self.actors.require(actor_id)
        actor.locked = bool(locked)
        return actor

    def duplicate_actor(self, actor_id, name=None):
        actor = self.actors.duplicate(actor_id, name=name)
        self.tracks.duplicate_actor_tracks(actor_id, actor.id)
        return actor

    def delete_actor(self, actor_id):
        actor_id = str(actor_id)
        actor = self.actors.remove(actor_id)
        self.tracks.remove_actor(actor_id)
        self.constraints.remove_actor(actor_id)
        if self.selection.actor_id == actor_id:
            self.selection.actor_id = self.active_robot_id()
            self.selection.frame_id = None
            self.selection.track_id = None
        return actor

    def set_object_transform_keyframe(self, actor_id, time, transform):
        actor = self.actors.require(actor_id)
        if actor.kind != ACTOR_KIND_OBJECT:
            raise ValueError("Only object actors can store transform tracks.")
        if actor.locked:
            raise ValueError(f"Actor is locked: {actor.name}")
        keyframe = TransformKeyframe(float(time), transform)
        return self.tracks.add_object_transform_keyframe(actor_id, keyframe)

    def attach(self, source_actor_id, source_frame_id, target_actor_id, target_frame_id):
        self.actors.require(source_actor_id)
        self.actors.require(target_actor_id)
        return self.constraints.add_attachment(
            source_actor_id,
            source_frame_id,
            target_actor_id,
            target_frame_id,
        )

    def weld(self, source_actor_id, source_frame_id, target_actor_id, target_frame_id):
        self.actors.require(source_actor_id)
        self.actors.require(target_actor_id)
        return self.constraints.add_weld(
            source_actor_id,
            source_frame_id,
            target_actor_id,
            target_frame_id,
        )

    def active_robot_trajectory(self):
        actor_id = self.active_robot_id()
        if actor_id is None:
            return Trajectory()
        return self.tracks.robot_trajectory(actor_id)

    def set_active_robot_trajectory(self, trajectory):
        actor_id = self.active_robot_id()
        if actor_id is None:
            raise ValueError("Scene has no robot actor.")
        self.tracks.set_robot_trajectory(actor_id, trajectory)
        return actor_id

    def visible_object_actors(self):
        return [
            actor for actor in self.actors.objects()
            if actor.visible
        ]

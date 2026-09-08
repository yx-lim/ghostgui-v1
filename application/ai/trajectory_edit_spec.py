"""Compact provider-neutral motion intent for local trajectory execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping


MAX_TRAJECTORY_OPERATIONS = 16
MAX_SPARSE_KEYFRAMES = 16
MAX_SPEC_SUMMARY_CHARACTERS = 4_000


class TrajectoryEditSpecError(RuntimeError):
    """A compact provider response is malformed or outside local limits."""


class TrajectoryEditMode(str, Enum):
    EDIT = "edit"
    GENERATE = "generate"


class TrajectoryOperationType(str, Enum):
    ROOT_OFFSET = "root_offset"
    HOLD_POSE = "hold_pose"
    RETIME_INTERVAL = "retime_interval"
    SET_JOINT_TARGET = "set_joint_target"
    SET_JOINT_GROUP_TARGET = "set_joint_group_target"
    SET_END_EFFECTOR_TARGET = "set_end_effector_target"
    LOCK_END_EFFECTOR = "lock_end_effector"
    SPARSE_KEYFRAMES = "sparse_keyframes"


@dataclass(frozen=True)
class TrajectoryOperation:
    operation_type: TrajectoryOperationType
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.operation_type, TrajectoryOperationType):
            object.__setattr__(
                self,
                "operation_type",
                TrajectoryOperationType(self.operation_type),
            )
        if not isinstance(self.arguments, Mapping):
            raise TypeError("trajectory operation arguments must be an object")
        normalized = dict(self.arguments)
        _validate_arguments(self.operation_type, normalized)
        object.__setattr__(self, "arguments", normalized)


@dataclass(frozen=True)
class TrajectoryEditSpec:
    mode: TrajectoryEditMode
    summary: str
    operations: tuple[TrajectoryOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TrajectoryEditMode):
            object.__setattr__(self, "mode", TrajectoryEditMode(self.mode))
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("trajectory edit summary must not be empty")
        if len(self.summary) > MAX_SPEC_SUMMARY_CHARACTERS:
            raise ValueError("trajectory edit summary exceeds the local size limit")
        if not self.operations:
            raise ValueError("trajectory edit spec requires at least one operation")
        if len(self.operations) > MAX_TRAJECTORY_OPERATIONS:
            raise ValueError("trajectory edit spec exceeds the operation limit")
        has_sparse_generation = any(
            operation.operation_type is TrajectoryOperationType.SPARSE_KEYFRAMES
            for operation in self.operations
        )
        if has_sparse_generation != (self.mode is TrajectoryEditMode.GENERATE):
            raise ValueError(
                "generate mode requires sparse_keyframes and edit mode forbids it"
            )


def trajectory_edit_spec_response_schema() -> dict[str, Any]:
    """Return the common Claude/Gemini structured-output envelope.

    Operation arguments remain compact JSON text because the current Gemini
    endpoint rejects the realistic nested union. The local parser validates the
    complete type-specific contract before execution.
    """

    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": [item.value for item in TrajectoryEditMode]},
            "summary": {"type": "string"},
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_TRAJECTORY_OPERATIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [item.value for item in TrajectoryOperationType],
                        },
                        "arguments": {
                            "type": "string",
                            "description": (
                                "Compact JSON object matching the selected "
                                "operation contract supplied in the prompt."
                            ),
                        },
                    },
                    "required": ["type", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["mode", "summary", "operations"],
        "additionalProperties": False,
    }


def trajectory_operation_argument_contracts() -> dict[str, Any]:
    """Describe the exact compact arguments included in planner context."""

    time_scope = {
        "start_time": "finite seconds >= 0",
        "end_time": "finite seconds >= start_time",
    }
    return {
        "root_offset": {**time_scope, "translation_m": "[x, y, z]"},
        "hold_pose": {
            "source_time": "finite seconds >= 0",
            **time_scope,
            "body_scope": "whole_body|joint|joint_group|end_effector",
            "body_name": "empty only for whole_body",
        },
        "retime_interval": {**time_scope, "scale": "finite number > 0"},
        "set_joint_target": {
            "joint": "named joint",
            "time_seconds": "finite seconds >= 0",
            "angle_rad": "finite radians",
        },
        "set_joint_group_target": {
            "joint_group": "named semantic group",
            "time_seconds": "finite seconds >= 0",
            "joint_angles_rad": "non-empty [{joint, angle_rad}]",
        },
        "set_end_effector_target": {
            "end_effector": "named End Effector",
            **time_scope,
            "mode": "absolute|relative",
            "position_m": "[x, y, z]",
        },
        "lock_end_effector": {
            "end_effectors": "non-empty unique names",
            "source_time": "finite seconds >= 0",
            **time_scope,
        },
        "sparse_keyframes": {
            "duration_seconds": "finite seconds > 0",
            "keyframes": (
                "4-16 chronological objects containing time_seconds, nullable "
                "root_position_m, nullable torso_rpy_rad, end_effector_targets, "
                "and joint_targets; first=0 and last=duration"
            ),
        },
    }


def parse_trajectory_edit_spec(text: str) -> TrajectoryEditSpec:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise TrajectoryEditSpecError(
            "provider returned malformed trajectory-edit JSON"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"mode", "summary", "operations"}:
        raise TrajectoryEditSpecError("trajectory edit spec has invalid fields")
    if not isinstance(payload["mode"], str) or not isinstance(payload["summary"], str):
        raise TrajectoryEditSpecError("trajectory edit mode and summary must be text")
    if not isinstance(payload["operations"], list):
        raise TrajectoryEditSpecError("trajectory edit operations must be a list")
    operations = []
    try:
        for value in payload["operations"]:
            if not isinstance(value, dict) or set(value) != {"type", "arguments"}:
                raise ValueError("trajectory operation has invalid fields")
            if not isinstance(value["type"], str) or not isinstance(value["arguments"], str):
                raise ValueError("trajectory operation type and arguments must be text")
            arguments = json.loads(value["arguments"])
            operations.append(TrajectoryOperation(value["type"], arguments))
        return TrajectoryEditSpec(
            mode=payload["mode"],
            summary=payload["summary"],
            operations=tuple(operations),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise TrajectoryEditSpecError(str(error)) from error


def _validate_arguments(operation_type: TrajectoryOperationType, values: dict[str, Any]) -> None:
    validators = {
        TrajectoryOperationType.ROOT_OFFSET: _validate_root_offset,
        TrajectoryOperationType.HOLD_POSE: _validate_hold_pose,
        TrajectoryOperationType.RETIME_INTERVAL: _validate_retime,
        TrajectoryOperationType.SET_JOINT_TARGET: _validate_joint,
        TrajectoryOperationType.SET_JOINT_GROUP_TARGET: _validate_joint_group,
        TrajectoryOperationType.SET_END_EFFECTOR_TARGET: _validate_end_effector,
        TrajectoryOperationType.LOCK_END_EFFECTOR: _validate_lock,
        TrajectoryOperationType.SPARSE_KEYFRAMES: _validate_sparse_keyframes,
    }
    validators[operation_type](values)


def _exact_fields(values, required):
    if set(values) != set(required):
        raise ValueError("trajectory operation arguments have invalid fields")


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its minimum")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _name(value, field, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be a non-empty name")
    return value.strip()


def _vector(value, field):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three values")
    return tuple(_number(item, field) for item in value)


def _time_scope(values):
    start = _number(values["start_time"], "start_time", minimum=0.0)
    end = _number(values["end_time"], "end_time", minimum=0.0)
    if end < start:
        raise ValueError("end_time must not precede start_time")


def _validate_root_offset(values):
    _exact_fields(values, ("start_time", "end_time", "translation_m"))
    _time_scope(values)
    _vector(values["translation_m"], "translation_m")


def _validate_hold_pose(values):
    _exact_fields(values, ("source_time", "start_time", "end_time", "body_scope", "body_name"))
    _number(values["source_time"], "source_time", minimum=0.0)
    _time_scope(values)
    scopes = {"whole_body", "joint", "joint_group", "end_effector"}
    if values["body_scope"] not in scopes:
        raise ValueError("body_scope is invalid")
    name = _name(values["body_name"], "body_name", allow_empty=True)
    if values["body_scope"] != "whole_body" and not name:
        raise ValueError("body_name is required for the selected body_scope")


def _validate_retime(values):
    _exact_fields(values, ("start_time", "end_time", "scale"))
    _time_scope(values)
    _number(values["scale"], "scale", positive=True)


def _validate_joint(values):
    _exact_fields(values, ("joint", "time_seconds", "angle_rad"))
    _name(values["joint"], "joint")
    _number(values["time_seconds"], "time_seconds", minimum=0.0)
    _number(values["angle_rad"], "angle_rad")


def _validate_joint_group(values):
    _exact_fields(values, ("joint_group", "time_seconds", "joint_angles_rad"))
    _name(values["joint_group"], "joint_group")
    _number(values["time_seconds"], "time_seconds", minimum=0.0)
    targets = values["joint_angles_rad"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("joint_angles_rad must be a non-empty list")
    _validate_joint_targets(targets)


def _validate_end_effector(values):
    _exact_fields(values, ("end_effector", "start_time", "end_time", "mode", "position_m"))
    _name(values["end_effector"], "end_effector")
    _time_scope(values)
    if values["mode"] not in {"absolute", "relative"}:
        raise ValueError("End Effector target mode is invalid")
    _vector(values["position_m"], "position_m")


def _validate_lock(values):
    _exact_fields(values, ("end_effectors", "source_time", "start_time", "end_time"))
    names = values["end_effectors"]
    if not isinstance(names, list) or not names:
        raise ValueError("end_effectors must be a non-empty list")
    normalized = [_name(value, "end_effector") for value in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("end_effectors must not contain duplicates")
    _number(values["source_time"], "source_time", minimum=0.0)
    _time_scope(values)


def _validate_sparse_keyframes(values):
    _exact_fields(values, ("duration_seconds", "keyframes"))
    duration = _number(values["duration_seconds"], "duration_seconds", positive=True)
    keyframes = values["keyframes"]
    if not isinstance(keyframes, list) or not 4 <= len(keyframes) <= MAX_SPARSE_KEYFRAMES:
        raise ValueError("sparse_keyframes requires 4-16 Keyframes")
    times = []
    required = (
        "time_seconds", "root_position_m", "torso_rpy_rad",
        "end_effector_targets", "joint_targets",
    )
    for keyframe in keyframes:
        if not isinstance(keyframe, dict):
            raise ValueError("sparse Keyframe must be an object")
        _exact_fields(keyframe, required)
        times.append(_number(keyframe["time_seconds"], "time_seconds", minimum=0.0))
        if keyframe["root_position_m"] is not None:
            _vector(keyframe["root_position_m"], "root_position_m")
        if keyframe["torso_rpy_rad"] is not None:
            _vector(keyframe["torso_rpy_rad"], "torso_rpy_rad")
        _validate_end_effector_targets(keyframe["end_effector_targets"])
        _validate_joint_targets(keyframe["joint_targets"])
        if (
            keyframe["root_position_m"] is None
            and keyframe["torso_rpy_rad"] is None
            and not keyframe["end_effector_targets"]
            and not keyframe["joint_targets"]
        ):
            raise ValueError("sparse Keyframe requires at least one target")
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("sparse Keyframe times must be strictly increasing")
    if abs(times[0]) > 1e-9 or abs(times[-1] - duration) > 1e-9:
        raise ValueError("sparse Keyframes must span zero through duration_seconds")


def _validate_end_effector_targets(values):
    if not isinstance(values, list):
        raise ValueError("end_effector_targets must be a list")
    names = []
    for target in values:
        if not isinstance(target, dict):
            raise ValueError("End Effector target must be an object")
        _exact_fields(target, ("end_effector", "position_m"))
        names.append(_name(target["end_effector"], "end_effector"))
        _vector(target["position_m"], "position_m")
    if len(set(names)) != len(names):
        raise ValueError("End Effector targets must not contain duplicates")


def _validate_joint_targets(values):
    if not isinstance(values, list):
        raise ValueError("joint_targets must be a list")
    names = []
    for target in values:
        if not isinstance(target, dict):
            raise ValueError("joint target must be an object")
        _exact_fields(target, ("joint", "angle_rad"))
        names.append(_name(target["joint"], "joint"))
        _number(target["angle_rad"], "angle_rad")
    if len(set(names)) != len(names):
        raise ValueError("joint targets must not contain duplicates")

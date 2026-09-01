"""Strict allowlist and argument validation for AI-callable tools."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Callable, Mapping

from application.ai.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolValidationError,
)
from application.ai.schemas import ToolDefinition


_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SUPPORTED_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SUPPORTED_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}


class ToolCategory(str, Enum):
    INSPECT = "inspect"
    EDIT = "edit"
    TEST = "test"


ToolHandler = Callable[[Mapping[str, Any], Any], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler
    category: ToolCategory
    mutates_working_copy: bool = False


class ToolRegistry:
    """The only dispatch path from provider tool calls to application code."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not _TOOL_NAME.fullmatch(spec.name):
            raise ToolRegistrationError(
                "tool name must be lowercase snake_case and at most 64 characters"
            )
        if spec.name in self._tools:
            raise ToolRegistrationError(f"tool already registered: {spec.name}")
        if not spec.description.strip():
            raise ToolRegistrationError("tool description must not be empty")
        if not callable(spec.handler):
            raise ToolRegistrationError("tool handler must be callable")
        schema = deepcopy(dict(spec.input_schema))
        _validate_schema_definition(schema, path="$", require_object=True)
        self._tools[spec.name] = ToolSpec(
            name=spec.name,
            description=spec.description,
            input_schema=schema,
            handler=spec.handler,
            category=spec.category,
            mutates_working_copy=spec.mutates_working_copy,
        )

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            ToolDefinition(tool.name, tool.description, deepcopy(tool.input_schema))
            for tool in self._tools.values()
        )

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(f"tool is not allowlisted: {name}") from error

    def validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        tool = self.get(name)
        if not isinstance(arguments, Mapping):
            raise ToolValidationError(f"{name} arguments must be an object")
        _validate_value(dict(arguments), tool.input_schema, path="$", tool_name=name)

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        context: Any = None,
    ) -> Any:
        tool = self.get(name)
        self.validate_arguments(name, arguments)
        try:
            return tool.handler(dict(arguments), context)
        except ToolExecutionError:
            raise
        except Exception as error:
            raise ToolExecutionError(f"tool {name} failed: {error}") from error


def _validate_schema_definition(
    schema: Mapping[str, Any],
    *,
    path: str,
    require_object: bool = False,
) -> None:
    unknown = set(schema) - _SUPPORTED_KEYS
    if unknown:
        raise ToolRegistrationError(
            f"unsupported schema keyword at {path}: {sorted(unknown)[0]}"
        )
    schema_type = schema.get("type")
    if schema_type not in _SUPPORTED_TYPES:
        raise ToolRegistrationError(f"schema at {path} has unsupported or missing type")
    if require_object and schema_type != "object":
        raise ToolRegistrationError("tool input schema must have type object")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise ToolRegistrationError(f"enum at {path} must be a non-empty list")

    if schema_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ToolRegistrationError(f"object schema at {path} requires properties")
        if schema.get("additionalProperties") is not False:
            raise ToolRegistrationError(
                f"object schema at {path} must set additionalProperties to false"
            )
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ToolRegistrationError(f"required at {path} must be a string list")
        missing = set(required) - set(properties)
        if missing:
            raise ToolRegistrationError(
                f"required property is not declared at {path}: {sorted(missing)[0]}"
            )
        for name, child in properties.items():
            if not isinstance(name, str) or not isinstance(child, Mapping):
                raise ToolRegistrationError(f"invalid property schema at {path}")
            _validate_schema_definition(child, path=f"{path}.{name}")
    elif schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise ToolRegistrationError(f"array schema at {path} requires items")
        _validate_schema_definition(items, path=f"{path}[]")


def _validate_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    tool_name: str,
) -> None:
    expected = schema["type"]
    valid = {
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }[expected](value)
    if not valid:
        raise ToolValidationError(f"{tool_name} {path} must be {expected}")
    if expected in {"number", "integer"} and not math.isfinite(value):
        raise ToolValidationError(f"{tool_name} {path} must be finite")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{tool_name} {path} is not an allowed value")

    if expected == "object":
        properties = schema["properties"]
        unknown = set(value) - set(properties)
        if unknown:
            raise ToolValidationError(
                f"{tool_name} {path} contains unknown property {sorted(unknown)[0]!r}"
            )
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ToolValidationError(
                f"{tool_name} {path} is missing required property {sorted(missing)[0]!r}"
            )
        for name, child_value in value.items():
            _validate_value(
                child_value,
                properties[name],
                path=f"{path}.{name}",
                tool_name=tool_name,
            )
    elif expected == "array":
        _check_bound(len(value), schema, "minItems", "maxItems", path, tool_name)
        for index, child_value in enumerate(value):
            _validate_value(
                child_value,
                schema["items"],
                path=f"{path}[{index}]",
                tool_name=tool_name,
            )
    elif expected == "string":
        _check_bound(len(value), schema, "minLength", "maxLength", path, tool_name)
    elif expected in {"number", "integer"}:
        _check_bound(value, schema, "minimum", "maximum", path, tool_name)


def _check_bound(
    value: float,
    schema: Mapping[str, Any],
    minimum_key: str,
    maximum_key: str,
    path: str,
    tool_name: str,
) -> None:
    if minimum_key in schema and value < schema[minimum_key]:
        raise ToolValidationError(f"{tool_name} {path} is below {minimum_key}")
    if maximum_key in schema and value > schema[maximum_key]:
        raise ToolValidationError(f"{tool_name} {path} exceeds {maximum_key}")

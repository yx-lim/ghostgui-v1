"""Derive closed tool-dialect JSON schemas from annotated handler signatures (stdlib only)."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import types
import typing
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class Range:
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Len:
    minimum: int = 1
    maximum: int = 80


@dataclass(frozen=True)
class Items:
    minimum: int = 0
    maximum: int = 128


@dataclass(frozen=True)
class Unit:
    unit: str


@dataclass(frozen=True)
class Doc:
    text: str


_MISSING = inspect.Parameter.empty
_CONTEXT_PARAMETERS = frozenset({"ctx", "context"})


def _strip_optional(annotation):
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [item for item in get_args(annotation) if item is not type(None)]
        if len(members) == 1 and len(get_args(annotation)) == 2:
            return members[0], True
    return annotation, False


def type_schema(annotation) -> dict[str, Any]:
    """Map one annotation (possibly Annotated) to the restricted schema dialect."""
    metadata = ()
    if get_origin(annotation) is Annotated:
        annotation, *extra = get_args(annotation)
        metadata = tuple(extra)
    annotation, _ = _strip_optional(annotation)
    if get_origin(annotation) is Annotated:
        nested, *extra = get_args(annotation)
        annotation, metadata = nested, tuple(extra) + metadata
    origin = get_origin(annotation)
    schema: dict[str, Any]
    if annotation is bool:
        schema = {"type": "boolean"}
    elif annotation is int:
        schema = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif annotation is str:
        schema = {"type": "string", "minLength": 1, "maxLength": 80}
    elif origin is Literal:
        values = list(get_args(annotation))
        if not values or not all(isinstance(value, str) for value in values):
            raise TypeError("Literal parameters must be non-empty string choices")
        schema = {"type": "string", "enum": values}
    elif origin in (tuple, list):
        arguments = get_args(annotation)
        item = arguments[0] if arguments else str
        schema = {"type": "array", "items": type_schema(item), "minItems": 0, "maxItems": 128}
    elif isinstance(annotation, type) and typing.is_typeddict(annotation):
        hints = get_type_hints(annotation, include_extras=True)
        required = sorted(getattr(annotation, "__required_keys__", hints))
        schema = {
            "type": "object",
            "properties": {name: type_schema(hint) for name, hint in hints.items()},
            "required": required,
            "additionalProperties": False,
        }
    else:
        raise TypeError(f"unsupported parameter annotation: {annotation!r}")
    descriptions = []
    for item in metadata:
        if isinstance(item, Range):
            if schema["type"] not in {"number", "integer"}:
                raise TypeError("Range applies only to numbers")
            if item.minimum is not None:
                schema["minimum"] = item.minimum
            if item.maximum is not None:
                schema["maximum"] = item.maximum
        elif isinstance(item, Len):
            if schema["type"] != "string":
                raise TypeError("Len applies only to strings")
            schema["minLength"], schema["maxLength"] = item.minimum, item.maximum
        elif isinstance(item, Items):
            if schema["type"] != "array":
                raise TypeError("Items applies only to arrays")
            schema["minItems"], schema["maxItems"] = item.minimum, item.maximum
        elif isinstance(item, Unit):
            descriptions.append(f"Unit: {item.unit}.")
        elif isinstance(item, Doc):
            descriptions.insert(0, item.text)
    if descriptions:
        schema["description"] = " ".join(descriptions)
    return schema


def derive_parameters(handler) -> dict[str, Any]:
    """Build a closed object schema from keyword parameters after the context argument.

    Defaults are documented in the description because the dialect has no ``default``.
    """
    signature = inspect.signature(handler)
    hints = get_type_hints(handler, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in _CONTEXT_PARAMETERS:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("command handlers must declare explicit parameters")
        if name not in hints:
            raise TypeError(f"command parameter {name!r} needs an annotation")
        schema = type_schema(hints[name])
        _, optional = _strip_optional(
            get_args(hints[name])[0] if get_origin(hints[name]) is Annotated else hints[name]
        )
        if parameter.default is _MISSING and not optional:
            required.append(name)
        elif parameter.default is not _MISSING and parameter.default is not None:
            note = f"Default: {parameter.default!r}."
            schema["description"] = f"{schema['description']} {note}" if "description" in schema else note
        properties[name] = schema
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def handler_description(handler) -> str:
    return " ".join(inspect.cleandoc(handler.__doc__ or "").split())

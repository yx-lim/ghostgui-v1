"""The command registry and its projections onto tool catalogs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from application.ai.tool_registry import _validate_schema_definition

from .schema import derive_parameters, handler_description
from .spec import (
    AGENT_SURFACES, Affinity, CommandSpec, Effect, LEGACY_SURFACES, Origin, Surface,
)


class CommandUnavailable(RuntimeError):
    """Raised when a command needs a port the current host does not provide."""


@dataclass
class CommandContext:
    """Everything a handler may touch. Ports are host-provided capability objects."""

    workspace: Any = None
    renderer: Any = None
    ports: Mapping[str, Any] = field(default_factory=dict)
    origin: Origin = Origin.AGENT_MCP
    gui: Callable[[Callable[[], Any]], Any] | None = None
    action_log: Any = None
    run_id: str = ""

    def port(self, name: str):
        try:
            return self.ports[name]
        except KeyError as error:
            raise CommandUnavailable(f"this GhostGUI host does not provide '{name}'") from error

    def on_qt(self, callback: Callable[[], Any]):
        """Run on the Qt main thread when hosted by the GUI; inline otherwise."""
        return callback() if self.gui is None else self.gui(callback)


class CommandRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, CommandSpec] = {}
        self._tools: dict[str, str] = {}

    def register(self, spec: CommandSpec) -> CommandSpec:
        if spec.id in self._specs:
            raise ValueError(f"command already registered: {spec.id}")
        if spec.tool in self._tools:
            raise ValueError(f"tool name already registered: {spec.tool}")
        params = deepcopy(dict(spec.params))
        _validate_schema_definition(params, path="$", require_object=True)
        spec = replace(spec, params=params)
        for example in spec.examples:
            from application.ai.tool_registry import _validate_value
            _validate_value(dict(example), params, path="$", tool_name=spec.tool)
        self._specs[spec.id] = spec
        self._tools[spec.tool] = spec.id
        return spec

    def command(self, id: str, *, title: str, effect: Effect, tool: str | None = None,
                description: str | None = None, params: Mapping | None = None, **options):
        """Decorator: derive the tool schema from the handler's annotated signature."""
        def decorate(handler):
            self.register(CommandSpec(
                id=id, tool=tool or id.split(".")[-1], title=title,
                description=description or handler_description(handler), handler=handler,
                params=params if params is not None else derive_parameters(handler),
                effect=effect, **options,
            ))
            return handler
        return decorate

    def get(self, command_id: str) -> CommandSpec:
        return self._specs[command_id]

    def by_tool(self, tool: str) -> CommandSpec:
        return self._specs[self._tools[tool]]

    def has_tool(self, tool: str) -> bool:
        return tool in self._tools

    def specs(self) -> tuple[CommandSpec, ...]:
        return tuple(self._specs.values())

    def projection(self, surface: Surface, *, ports: Iterable[str] | None = None,
                   sort: bool = True) -> tuple[CommandSpec, ...]:
        """Commands exposed on a surface; commands needing an absent port are omitted."""
        available = None if ports is None else frozenset(ports)
        selected = [
            spec for spec in self._specs.values()
            if surface in spec.surfaces and spec.effect is not Effect.FORBIDDEN
            and (available is None or spec.requires <= available)
        ]
        return tuple(sorted(selected, key=lambda spec: spec.tool)) if sort else tuple(selected)

    def tool_definitions(self, surface: Surface, *, ports=None, sort=True) -> list[dict]:
        return [{"name": spec.tool, "description": spec.description, "input_schema": deepcopy(dict(spec.params))}
                for spec in self.projection(surface, ports=ports, sort=sort)]

    def tool_hash(self, surface: Surface, *, ports=None) -> str:
        encoded = json.dumps(self.tool_definitions(surface, ports=ports), sort_keys=True,
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def bindings(self) -> dict[str, tuple[CommandSpec, Mapping]]:
        """objectName → (spec, fixed args) for the GUI binder and coverage test."""
        result = {}
        for spec in self._specs.values():
            for binding in spec.gui:
                if binding.object_name in result:
                    raise ValueError(f"widget bound twice: {binding.object_name}")
                result[binding.object_name] = (spec, dict(binding.args))
        return result


def build_tool_catalog(registry: CommandRegistry, context: CommandContext, surface: Surface,
                       *, ports=None, sort=False):
    """Project registry commands into the legacy ``CommandCatalog`` used by ``ToolRuntime``."""
    from application.agent_api.commands import CommandCatalog, ToolMeta
    from application.ai.tool_registry import ToolCategory, ToolSpec

    catalog = CommandCatalog()
    for spec in registry.projection(surface, ports=ports, sort=sort):
        def handler(args, _context, _spec=spec):
            values = {key: value for key, value in args.items() if key != "expected_revision"}
            return _spec.handler(context, **values)
        mutate = spec.mutates_candidate
        catalog.registry.register(ToolSpec(
            spec.tool, spec.description, spec.params, handler,
            ToolCategory.EDIT if mutate or spec.app_level else ToolCategory.INSPECT,
            mutates_working_copy=mutate,
        ))
        catalog.meta[spec.tool] = ToolMeta(
            read_only=not mutate and not spec.app_level,
            mutates_candidate=mutate, app_level=spec.app_level, returns_images=spec.returns_images,
        )
        catalog.specs[spec.tool] = spec
    catalog.context = context
    return catalog

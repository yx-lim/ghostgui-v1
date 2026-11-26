"""Qt-free command registry: one source of truth for GUI, in-app agent and MCP surfaces."""

from functools import lru_cache

from .registry import CommandContext, CommandRegistry, CommandUnavailable, build_tool_catalog
from .spec import (
    Affinity, Availability, Binding, CommandSpec, Effect, Origin, Surface,
)

__all__ = [
    "Affinity", "Availability", "Binding", "CommandContext", "CommandRegistry", "CommandSpec",
    "CommandUnavailable", "Effect", "Origin", "Surface", "build_tool_catalog", "default_registry",
]


@lru_cache(maxsize=1)
def default_registry() -> CommandRegistry:
    """The process-wide registry. Specs are immutable, so sharing it is safe."""
    from .catalog import app as app_catalog
    from .catalog import motion as motion_catalog
    from .catalog.legacy import register_legacy

    registry = CommandRegistry()
    register_legacy(registry)
    app_catalog.register(registry)
    motion_catalog.register(registry)
    return registry

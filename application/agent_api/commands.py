"""Closed-schema deterministic command catalog; no model or transport calls.

The catalog is a projection of ``application.commands.default_registry()``. Tool names,
schemas and order are pinned by ``tests/fixtures/legacy_agent_tools.json``.
"""

from dataclasses import dataclass

from application.ai.tool_registry import ToolRegistry


@dataclass(frozen=True)
class ToolMeta:
    read_only: bool = False
    mutates_candidate: bool = False
    app_level: bool = False
    returns_images: bool = False


class CommandCatalog:
    def __init__(self):
        self.registry = ToolRegistry()
        self.meta = {}
        self.specs = {}
        self.context = None


def build_command_catalog(workspace, *, renderer=None, surface=None, ports=None, origin=None,
                          registry=None, action_log=None, gui=None):
    """Compatibility wrapper: project the shared registry onto a ``ToolRuntime`` catalog.

    The default surface is headless MCP, which reproduces the shipped 24-tool catalog.
    """
    from application.commands import CommandContext, Origin, Surface, build_tool_catalog, default_registry

    context = CommandContext(
        workspace=workspace, renderer=renderer, ports=dict(ports or {}),
        origin=origin or Origin.AGENT_MCP, action_log=action_log, gui=gui,
    )
    return build_tool_catalog(
        registry or default_registry(), context, surface or Surface.MCP_HEADLESS,
        ports=None if ports is None else tuple(ports),
    )

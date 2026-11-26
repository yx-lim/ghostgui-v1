"""Command specifications: the single source of truth for GUI, agent and MCP surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Mapping


class Effect(str, Enum):
    """Risk class enforced identically for every agent origin."""

    READ = "read"
    UI = "ui"
    CANDIDATE = "candidate"
    COMMIT = "commit"
    PERSIST = "persist"
    TRANSITION = "transition"
    DESTRUCTIVE = "destructive"
    FORBIDDEN = "forbidden"


class Affinity(str, Enum):
    """Where a handler must execute."""

    QT_MAIN = "qt_main"
    AGENT_WORKER = "agent_worker"
    ANY = "any"


class Surface(str, Enum):
    GUI = "gui"
    PALETTE = "palette"
    AGENT_INAPP = "agent_inapp"
    MCP_ATTACHED = "mcp_attached"
    MCP_HEADLESS = "mcp_headless"


class Origin(str, Enum):
    USER_GUI = "user_gui"
    AGENT_INAPP = "agent_inapp"
    AGENT_MCP = "agent_mcp"


AGENT_SURFACES = frozenset({Surface.AGENT_INAPP, Surface.MCP_ATTACHED, Surface.MCP_HEADLESS})
ALL_SURFACES = frozenset(Surface)
LEGACY_SURFACES = frozenset({Surface.MCP_ATTACHED, Surface.MCP_HEADLESS, Surface.AGENT_INAPP})
# Effects that need a human decision before an agent may execute them.
CONFIRM_EFFECTS = frozenset({Effect.DESTRUCTIVE})
# Effects deferred by plan mode (READ and CANDIDATE run normally, CANDIDATE is already a preview).
PLAN_DEFERRED_EFFECTS = frozenset({
    Effect.UI, Effect.PERSIST, Effect.COMMIT, Effect.DESTRUCTIVE, Effect.TRANSITION,
})

_TOOL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True)
class Binding:
    """A widget or QAction (by objectName) that triggers this command in the GUI."""

    object_name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Availability:
    enabled: bool
    reason: str = ""


@dataclass(frozen=True)
class CommandSpec:
    id: str
    tool: str
    title: str
    description: str
    handler: Callable[..., Any]
    params: Mapping[str, Any]
    effect: Effect
    affinity: Affinity = Affinity.ANY
    surfaces: frozenset = LEGACY_SURFACES
    requires: frozenset = frozenset()
    availability: Callable[[Any], Availability] | None = None
    preview: Callable[[Any, Mapping[str, Any]], str] | None = None
    examples: tuple = ()
    gui: tuple = ()
    shortcut: str | None = None
    idempotent: bool = False
    exposure_reason: str | None = None
    # Legacy catalog flags, preserved byte-for-byte for the MCP projection.
    mutates_candidate: bool = False
    app_level: bool = False
    returns_images: bool = False
    aliases: tuple = ()

    def __post_init__(self):
        if not _ID.fullmatch(self.id):
            raise ValueError(f"command id must be dotted snake_case: {self.id!r}")
        if not _TOOL.fullmatch(self.tool):
            raise ValueError(f"tool name must be snake_case and at most 64 characters: {self.tool!r}")
        if not self.title.strip() or not self.description.strip():
            raise ValueError(f"{self.id}: title and description are required")
        if self.effect is Effect.FORBIDDEN and self.surfaces & AGENT_SURFACES:
            raise ValueError(f"{self.id}: forbidden commands are never projected to agent surfaces")
        if not self.surfaces & AGENT_SURFACES and self.effect is not Effect.FORBIDDEN and not self.exposure_reason:
            raise ValueError(f"{self.id}: exposure_reason is required when agent surfaces are omitted")

    @property
    def read_only(self) -> bool:
        return self.effect is Effect.READ

    def summarize(self, arguments: Mapping[str, Any], limit: int = 80) -> str:
        """Short argument summary for chips and logs (values only, scrubbed by the caller)."""
        parts = []
        for key, value in arguments.items():
            if key == "expected_revision":
                continue
            text = value if isinstance(value, str) else repr(value)
            parts.append(f"{key}={text}")
        summary = ", ".join(parts)
        return summary if len(summary) <= limit else summary[: limit - 1] + "…"

"""Bounded activity values suitable for future GUI and MCP adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    tool: str
    revision: int
    summary: str = ""

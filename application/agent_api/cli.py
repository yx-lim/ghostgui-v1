"""Headless ``ghostgui agent`` command: run one command-bar prompt against a project folder.

Uses the same loop, registry and gates as the in-app bar, without Qt. GUI-only
commands are not offered. Destructive commands ask on an interactive terminal and are
denied otherwise. Motion edits stay on the candidate unless ``--approval auto_accept``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from application.ai.agent_events import RunOutcome
from application.ai.agent_loop import AgentLoop, ExecOutcome, LoopHooks, LoopPolicy
from application.ai.agent_wire import Effort
from application.ai.limits import AgentRunLimits

EXIT_CODES = {RunOutcome.COMPLETED: 0, RunOutcome.PLAN_READY: 0, RunOutcome.STOPPED: 130}


class HeadlessExecutor:
    def __init__(self, runtime, registry, action_log=None):
        self.runtime, self.registry, self.log = runtime, registry, action_log

    def spec_for(self, tool):
        return self.runtime.catalog.specs.get(tool) if tool in self.runtime.catalog.specs else (
            self.registry.by_tool(tool) if tool == "ask_user" else None)

    async def execute(self, tool, arguments, cancel):
        outcome = await asyncio.to_thread(self.runtime.call, tool, dict(arguments))
        if self.log is not None:
            self.log.record(origin="agent_inapp", command=tool, arguments=arguments,
                            status="ok" if outcome.ok else "error", code=str(outcome.payload.get("code", "")))
        return ExecOutcome(outcome.ok, outcome.payload, tuple(outcome.images))


def parser():
    result = argparse.ArgumentParser(prog="ghostgui agent",
                                     description="Run one Claude command-bar prompt headlessly.")
    result.add_argument("prompt", help="What Claude should do")
    result.add_argument("--projects-dir", type=Path, required=True, help="Allowed project root")
    result.add_argument("--project", help="Open this project (by name) before the prompt")
    result.add_argument("--output-dir", type=Path, help="Optional allowed export root")
    result.add_argument("--robot", default="g1", help="Registered model key (default: g1)")
    result.add_argument("--model", default="claude-opus-5-5",
                        choices=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"))
    result.add_argument("--effort", default="medium", choices=[effort.value for effort in Effort])
    result.add_argument("--approval", default="review", choices=("review", "auto_accept"))
    result.add_argument("--plan", action="store_true", help="Plan first: defer UI/persist/commit actions")
    result.add_argument("--max-cost", type=float, default=1.0, help="Run spend cap in USD (default 1.00)")
    result.add_argument("--json", action="store_true", help="Print the run result as JSON")
    return result


def _interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


async def _confirm(spec, arguments, preview, prompt):
    if not _interactive():
        return "deny"
    answer = await asyncio.to_thread(input, f"Allow {preview}? [y/N] ")
    return "once" if answer.strip().lower() in ("y", "yes") else "deny"


async def _ask(question, options):
    if not _interactive():
        return None
    hint = f" ({' / '.join(options)})" if options else ""
    return (await asyncio.to_thread(input, f"Claude asks: {question}{hint} ")).strip() or None


def _emit(event):
    if event.kind == "tool_finished":
        mark = "✓" if event.ok else "✗"
        print(f"{mark} {event.title} · {event.summary}"[:200], file=sys.stderr)
    elif event.kind == "notice":
        print(event.text[:300], file=sys.stderr)


def run(argv=None, *, provider=None) -> int:
    args = parser().parse_args(argv)
    from application.agent_api.commands import build_command_catalog
    from application.agent_api.runtime import ToolRuntime
    from application.agent_api.workspace import HeadlessWorkspace
    from application.commands import Origin, Surface, default_registry
    from application.commands.action_log import ActionLog

    workspace = HeadlessWorkspace(args.projects_dir, output_dir=args.output_dir, model=args.robot,
                                  approval="auto_accept" if args.approval == "auto_accept" else "review")
    if args.project:
        workspace.open_project(args.project)
    log = ActionLog()
    registry = default_registry()
    catalog = build_command_catalog(workspace, surface=Surface.AGENT_INAPP, ports={}, origin=Origin.AGENT_INAPP,
                                    action_log=log)
    runtime = ToolRuntime(catalog, workspace)
    executor = HeadlessExecutor(runtime, registry, log)
    tools = [tool for tool in registry.tool_definitions(Surface.AGENT_INAPP, ports=())]
    if provider is None:
        from application.ai.credentials import default_credential_source
        from application.ai.provider_registry import DEFAULT_PROVIDER_REGISTRY
        from application.ai.secrets import SecretStr

        key = default_credential_source().get_secret("anthropic")
        if not key:
            print("Set ANTHROPIC_API_KEY or store a key with the GhostGUI command bar.", file=sys.stderr)
            return 2
        provider = DEFAULT_PROVIDER_REGISTRY.get("anthropic").create_agent(api_key=SecretStr(key))
    loop = AgentLoop(provider, executor, model=args.model, tools=tuple(tools), effort=Effort(args.effort),
                     hooks=LoopHooks(emit=_emit, confirm=_confirm, ask=_ask),
                     limits=AgentRunLimits(max_run_cost_usd=args.max_cost),
                     policy=LoopPolicy(approval="auto" if args.approval == "auto_accept" else "review",
                                       plan_mode=args.plan))

    async def main():
        try:
            return await loop.run(args.prompt, workspace.status())
        finally:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    result = asyncio.run(main())
    if args.json:
        print(json.dumps({"outcome": result.outcome.value, "text": result.text, "turns": result.turns,
                          "actions": result.actions, "cost_usd": result.cost_usd, "message": result.message,
                          "deferred": [call.preview for call in result.deferred]}, sort_keys=True))
    else:
        print(result.text or result.message or result.outcome.value)
    return EXIT_CODES.get(result.outcome, 1)


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())

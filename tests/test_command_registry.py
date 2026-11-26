"""Shared command registry: dialect conformance, golden legacy snapshot, Qt-free import."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Annotated, Literal

from application.agent_api.commands import build_command_catalog
from application.agent_api.workspace import HeadlessWorkspace
from application.commands import (
    CommandRegistry, CommandSpec, Effect, Surface, default_registry,
)
from application.commands.action_log import ActionLog, scrub
from application.commands.schema import Range, Unit, derive_parameters
from application.commands.spec import AGENT_SURFACES

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "legacy_agent_tools.json"


class GoldenSnapshotTests(unittest.TestCase):
    def test_mcp_catalog_is_byte_identical_to_shipped_tools(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            catalog = build_command_catalog(HeadlessWorkspace(Path(directory)))
            tools = catalog.registry.definitions()
            meta = {name: {"app_level": m.app_level, "mutates_candidate": m.mutates_candidate,
                           "read_only": m.read_only, "returns_images": m.returns_images}
                    for name, m in catalog.meta.items()}
        encode = lambda value: json.dumps(value, sort_keys=True)  # noqa: E731
        self.assertEqual(encode([{"name": t.name, "description": t.description,
                                       "input_schema": t.input_schema} for t in tools]), encode(golden["tools"]))
        self.assertEqual(encode(meta), encode(golden["meta"]))


class RegistryTests(unittest.TestCase):
    def test_unique_ids_tools_and_valid_examples(self):
        registry = default_registry()
        specs = registry.specs()
        self.assertEqual(len({spec.id for spec in specs}), len(specs))
        self.assertEqual(len({spec.tool for spec in specs}), len(specs))
        for spec in specs:
            self.assertLessEqual(len(spec.tool), 64)
            self.assertTrue(spec.description.strip(), spec.id)

    def test_forbidden_never_projected(self):
        registry = CommandRegistry()
        with self.assertRaises(ValueError):
            CommandSpec(id="danger.shell", tool="shell", title="Shell", description="Run shell.",
                        handler=lambda ctx: None, params={"type": "object", "properties": {},
                        "required": [], "additionalProperties": False}, effect=Effect.FORBIDDEN,
                        surfaces=AGENT_SURFACES)
        registry.register(CommandSpec(
            id="danger.quit", tool="quit_app", title="Quit", description="Quit.",
            handler=lambda ctx: None, params={"type": "object", "properties": {}, "required": [],
            "additionalProperties": False}, effect=Effect.FORBIDDEN, surfaces=frozenset({Surface.GUI})))
        for surface in Surface:
            self.assertEqual(registry.projection(surface), ())

    def test_duplicate_tool_rejected(self):
        registry = CommandRegistry()
        params = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        registry.register(CommandSpec(id="a.one", tool="one", title="One", description="One.",
                                      handler=lambda ctx: None, params=params, effect=Effect.READ))
        with self.assertRaises(ValueError):
            registry.register(CommandSpec(id="a.two", tool="one", title="Two", description="Two.",
                                          handler=lambda ctx: None, params=params, effect=Effect.READ))

    def test_derived_schema_is_closed_and_bounded(self):
        def handler(ctx, time_s: Annotated[float, Range(0, 120), Unit("s")],
                    mode: Literal["a", "b"] = "a", flag: bool | None = None):
            """Do a thing."""
        schema = derive_parameters(handler)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["time_s"])
        self.assertEqual(schema["properties"]["time_s"]["maximum"], 120)
        self.assertEqual(schema["properties"]["mode"]["enum"], ["a", "b"])

    def test_tool_hash_is_deterministic(self):
        registry = default_registry()
        self.assertEqual(registry.tool_hash(Surface.AGENT_INAPP), registry.tool_hash(Surface.AGENT_INAPP))
        tools = [tool["name"] for tool in registry.tool_definitions(Surface.AGENT_INAPP)]
        self.assertEqual(tools, sorted(tools))
        self.assertNotIn("get_status", tools)

    def test_action_log_scrubs_and_bounds(self):
        log = ActionLog(capacity=2)
        for index in range(3):
            log.record(origin="agent_inapp", command="x", status="ok",
                       arguments={"key": "sk-ant-abcdefghijkl", "n": index, "expected_revision": 4})
        tail = log.tail()
        self.assertEqual(len(tail), 2)
        self.assertNotIn("sk-ant", json.dumps(tail))
        self.assertNotIn("expected_revision", tail[0]["arguments"])
        self.assertEqual(scrub("a" * 300)[-1], "…")


class HeadlessImportTests(unittest.TestCase):
    def test_registry_imports_without_qt(self):
        code = ("import sys; import application.commands as c; c.default_registry(); "
                "bad=[m for m in sys.modules if m.startswith(('PySide6','PyQt'))]; "
                "assert not bad, bad")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()


class DialogManifestTests(unittest.TestCase):
    def test_every_gui_dialog_is_classified(self):
        import ast

        manifest = json.loads((ROOT / "tests" / "fixtures" / "dialog_manifest.json").read_text())
        found = set()
        dialog_names = {"QDialog"}
        for _ in range(3):  # follow subclasses of dialogs defined under gui/
            for path in (ROOT / "gui").rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.ClassDef) and any(
                            getattr(base, "id", getattr(base, "attr", "")) in dialog_names for base in node.bases):
                        found.add(node.name)
            dialog_names |= found
        self.assertEqual(sorted(found - set(manifest)), [], "classify new dialogs in dialog_manifest.json")
        self.assertEqual(sorted(set(manifest) - found), [], "remove stale manifest entries")

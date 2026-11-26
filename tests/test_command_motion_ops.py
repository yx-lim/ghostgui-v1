"""M3: candidate-side timeline/clipboard/pose/import commands on a headless workspace."""
import asyncio
from pathlib import Path
import tempfile
import unittest

import numpy as np

from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
from application.agent_api.workspace import HeadlessWorkspace
from application.commands import Effect, Surface, default_registry


class CandidateOpsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = HeadlessWorkspace(self.root, approval="auto_accept")
        # In-app projection without a GUI port: motion commands only.
        self.runtime = ToolRuntime(build_command_catalog(self.workspace, surface=Surface.AGENT_INAPP, ports={}),
                                   self.workspace)
        self.call("new_project", {"name": "motion", "duration_s": 0.6})
        home = self.workspace.adapter.home_qpos.copy()
        middle = home.copy()
        middle[-1] += 0.05
        self.call("author_keyframes", {"duration_s": 0.6, "keyframes": [
            {"time_seconds": time, "qpos": pose.tolist()}
            for time, pose in ((0, home), (0.2, middle), (0.4, middle), (0.6, home))]})
        self.call("validate_motion")
        self.call("accept_candidate")
        self.committed_times = list(self.workspace.committed.qpos_timeline.times())

    def call(self, name, arguments=None, ok=True):
        outcome = self.runtime.call(name, arguments or {})
        self.assertEqual(outcome.ok, ok, outcome.payload)
        return outcome.payload

    def candidate_times(self):
        return [round(time, 6) for time in self.workspace.document("candidate").qpos_timeline.times()]

    def test_gui_commands_absent_without_port(self):
        names = {tool.name for tool in self.runtime.catalog.registry.definitions()}
        self.assertIn("insert_time", names)
        self.assertNotIn("set_view", names)
        self.assertNotIn("get_status", names)

    def test_insert_shift_move_duration(self):
        self.call("insert_time", {"at_s": 0.2, "duration_s": 0.3})
        times = self.candidate_times()
        self.assertEqual(times[-1], 0.9)
        self.assertFalse([time for time in times if 0.2 < time < 0.5])  # the held interval
        self.assertEqual(self.workspace.committed.qpos_timeline.times(), self.committed_times)
        self.call("set_motion_duration", {"duration_s": 2.0})
        self.call("shift_motion", {"offset_s": 0.5})
        self.assertEqual(self.candidate_times()[0], 0.5)
        self.call("move_time_range", {"start_s": 1.2, "end_s": 1.4, "destination_s": 1.6})
        self.assertEqual(self.candidate_times()[-1], 1.8)
        self.call("set_motion_duration", {"duration_s": 1.0}, ok=False)

    def test_copy_paste_clear_reset(self):
        self.call("copy_motion_range", {"start_s": 0.0, "end_s": 0.6, "source": "committed"})
        self.call("set_motion_duration", {"duration_s": 1.2})
        self.call("paste_motion", {"at_s": 0.6, "reverse": True})
        self.assertEqual(self.candidate_times()[-1], 1.2)
        self.call("reset_pose", {"time_s": 0.2})
        state = self.workspace.document("candidate").qpos_timeline.get_state(0.2)
        np.testing.assert_allclose(state, self.workspace.adapter.home_qpos)
        self.call("clear_motion")
        self.assertEqual(self.candidate_times(), [0.0])
        self.assertEqual(self.workspace.document("candidate").trajectory.frames, [])

    def test_paste_without_copy_fails_and_rolls_back(self):
        before = self.workspace.identity()
        self.call("paste_motion", {"at_s": 0.0}, ok=False)
        self.assertEqual(self.workspace.identity(), before)

    def test_list_projects_and_import_sandbox(self):
        self.assertEqual(self.call("list_projects")["projects"], ["motion"])
        imports = self.workspace.project.root_dir / "imports"
        imports.mkdir()
        nq = self.workspace.adapter.mj_model.nq
        home = self.workspace.adapter.home_qpos
        rows = []
        for index in range(11):
            pose = home.copy()
            pose[-1] += 0.01 * index
            rows.append(f"{index * 0.05}," + ",".join(str(value) for value in pose))
        (imports / "wave.csv").write_text("\n".join(rows) + "\n")
        result = self.call("import_file", {"kind": "trajectory", "file_name": "wave"})
        self.assertEqual(result["keyframes"], 11)
        self.assertAlmostEqual(self.workspace.document("candidate").timeline_duration, 0.5)
        for bad in ("../wave", "/etc/passwd", "missing"):
            self.call("import_file", {"kind": "trajectory", "file_name": bad}, ok=False)

    def test_effects_are_classified(self):
        registry = default_registry()
        self.assertIs(registry.by_tool("simulation_viewer").effect, Effect.DESTRUCTIVE)
        self.assertIs(registry.by_tool("history_undo").effect, Effect.DESTRUCTIVE)
        for tool in ("insert_time", "paste_motion", "clear_motion", "import_file"):
            self.assertTrue(registry.by_tool(tool).mutates_candidate, tool)


class MCPAnnotationTests(unittest.TestCase):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp extra not installed")

    def handlers(self, server):
        from mcp import types
        return (server.request_handlers[types.ListToolsRequest], server.request_handlers[types.CallToolRequest])

    def test_headless_annotations_and_attached_destructive_denied(self):
        from mcp import types
        from application.mcp_server import create_server

        with tempfile.TemporaryDirectory() as directory:
            server = create_server(HeadlessWorkspace(Path(directory)))
            list_tools, _ = self.handlers(server)
            tools = asyncio.run(list_tools(types.ListToolsRequest(method="tools/list"))).root.tools
            self.assertEqual(len(tools), 24)
            by_name = {tool.name: tool for tool in tools}
            self.assertTrue(by_name["get_pose"].annotations.readOnlyHint)
            self.assertFalse(by_name["author_keyframes"].annotations.readOnlyHint)

        class FakeBridge:
            info = {"workspace": "gui", "attach": "required", "projects_dir": "x", "approval": "review"}
            calls = []

            def call(self, name, arguments):
                self.calls.append(name)
                return {"ok": True, "payload": {"ok": True}, "is_error": False, "images": []}

        bridge = FakeBridge()
        server = create_server(bridge_client=bridge)
        list_tools, call_tool = self.handlers(server)
        tools = {tool.name: tool for tool in asyncio.run(list_tools(types.ListToolsRequest(method="tools/list"))).root.tools}
        self.assertIn("set_view", tools)
        self.assertTrue(tools["history_undo"].annotations.destructiveHint)
        request = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(
            name="history_undo", arguments={}))
        result = asyncio.run(call_tool(request)).root
        self.assertTrue(result.isError)
        self.assertEqual(bridge.calls, [])


if __name__ == "__main__":
    unittest.main()

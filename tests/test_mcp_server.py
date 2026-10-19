"""Actual SDK stdio transport plus bounded, detached offscreen rendering."""
import asyncio
import base64
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from application.agent_rendering import (
    MuJoCoViewRenderer, RenderedImage, RenderingUnavailable, png_rgb,
)
from application.agent_api.commands import build_command_catalog
from application.agent_api.runtime import ToolRuntime
from application.agent_api.workspace import HeadlessWorkspace

ROOT = Path(__file__).resolve().parents[1]
MCP_AVAILABLE = importlib.util.find_spec('mcp') is not None


class StubRenderer:
    def render(self, model, samples, views, width, height):
        print('deterministic renderer diagnostic')
        from application.agent_rendering import label_pixels
        return tuple(RenderedImage(png_rgb(label_pixels(
            np.full((height, width, 3), 80 + index * 10, dtype=np.uint8), f'{time:.3f}s')))
            for index, (time, _) in enumerate(samples) for view in views)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = HeadlessWorkspace(self.temp.name)
        self.runtime = ToolRuntime(build_command_catalog(self.workspace, renderer=StubRenderer()), self.workspace)

    def test_detached_bounded_images_and_labels(self):
        before = self.workspace.identity()
        result = self.runtime.call('render_views', {'times_s': [0, 0.5]})
        self.assertTrue(result.ok, result.payload)
        self.assertTrue(result.payload['available'])
        self.assertEqual(len(result.images), 4)
        self.assertNotEqual(result.images[0].data, result.images[2].data)
        self.assertEqual(self.workspace.identity(), before)
        self.assertIsNone(self.workspace.candidate())
        for image in result.images:
            self.assertTrue(image.data.startswith(b'\x89PNG\r\n\x1a\n'))
        for args in ({'times_s': [0, 1, 2]}, {'width': 641}, {'times_s': [6]},
                     {'views': ['front', 'front']}, {'views': ['overhead']}):
            self.assertFalse(self.runtime.call('render_views', args).ok, args)
        self.assertEqual(self.workspace.identity(), before)

    def test_unavailable_and_oversized_images(self):
        with patch.object(StubRenderer, 'render', side_effect=RenderingUnavailable('no GL')):
            outcome = self.runtime.call('render_views', {})
            self.assertTrue(outcome.ok)
            self.assertFalse(outcome.payload['available'])
            self.assertEqual(outcome.payload['code'], 'rendering_unavailable')
            self.assertFalse(outcome.images)
        bad = RenderedImage(b'\x89PNG\r\n\x1a\n' + b'x' * (2 * 1024 * 1024))
        with patch.object(StubRenderer, 'render', return_value=(bad, bad)):
            self.assertFalse(self.runtime.call('render_views', {}).ok)

    def test_real_mujoco_offscreen_render(self):
        runtime = ToolRuntime(build_command_catalog(self.workspace, renderer=MuJoCoViewRenderer()), self.workspace)
        with redirect_stdout(io.StringIO()):
            result = runtime.call('render_views', {'views': ['front'], 'width': 160, 'height': 120})
        self.assertTrue(result.ok, result.payload)
        if not result.payload['available']:
            self.skipTest(result.payload['message'])
        self.assertEqual(len(result.images), 1)
        self.assertGreater(len(result.images[0].data), 100)

    def test_generated_source_matches_rendered_values(self):
        self.assertTrue(self.runtime.call('new_project', {'name': 'render', 'duration_s': 0.2}).ok)
        home = self.workspace.adapter.home_qpos.tolist()
        self.assertTrue(self.runtime.call('author_keyframes', {'duration_s': 0.2, 'keyframes': [
            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.2, 'qpos': home}]}).ok)
        self.assertTrue(self.runtime.call('generate_trajectory', {}).ok)
        result = self.runtime.call('render_views', {'times_s': [0.1]})
        self.assertTrue(result.ok, result.payload)
        self.assertEqual(result.payload['motion_source'], 'generated')


@unittest.skipUnless(MCP_AVAILABLE, 'optional agent extra is not installed')
class MCPTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_client_workflow_resources_images_errors_and_shutdown(self):
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode='w+') as errors:
            params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve()),
                '--serve-stub', directory], cwd=str(ROOT))
            with anyio.fail_after(60):
                async with stdio_client(params, errlog=errors) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        self.assertIn('headless', initialized.instructions)
                        tools = await session.list_tools()
                        self.assertEqual(len(tools.tools), 24)
                        names = {tool.name for tool in tools.tools}
                        self.assertIn('render_views', names)
                        for tool in tools.tools:
                            self.assertFalse(tool.inputSchema['additionalProperties'])
                        resources = await session.list_resources()
                        self.assertEqual(len(resources.resources), 3)
                        robot = await session.read_resource('ghostgui://robot')
                        summary = json.loads(robot.contents[0].text)
                        self.assertGreater(summary['nq'], 0)
                        self.assertIn('Keyframe', (await session.read_resource('ghostgui://terminology')).contents[0].text)
                        prompts = await session.list_prompts()
                        self.assertEqual(prompts.prompts[0].name, 'author_motion')
                        self.assertIn('validate_motion', (await session.get_prompt('author_motion')).messages[0].content.text)

                        async def call(name, args=None, ok=True):
                            result = await session.call_tool(name, args or {})
                            self.assertEqual(result.isError, not ok, result)
                            self.assertEqual(result.structuredContent['ok'], ok)
                            return result

                        for name, args in [('missing_tool', {}), ('new_project', {'name': '../escape'}),
                                           ('get_status', {'unknown': True}), ('author_keyframes', {'keyframes': []})]:
                            result = await call(name, args, ok=False)
                            self.assertIn('code', result.structuredContent)
                        await call('new_project', {'name': 'motion', 'duration_s': 0.4})
                        home = (await call('get_pose', {'time_s': 0})).structuredContent['qpos']
                        changed = list(home)
                        # G1's final wrist coordinate gives a nonconstant, bounded path.
                        changed[-1] += 0.05
                        await call('author_keyframes', {'duration_s': 0.4, 'keyframes': [
                            {'time_seconds': 0, 'qpos': home}, {'time_seconds': 0.2, 'qpos': changed},
                            {'time_seconds': 0.4, 'qpos': home}]})
                        await call('repeat_motion', {'start_s': 0, 'end_s': 0.4, 'additional_copies': 1})
                        await call('generate_trajectory', {'dt': 0.02})
                        validated = await call('validate_motion', {'dt': 0.02})
                        self.assertTrue(validated.structuredContent['valid'], validated)
                        images = await call('render_views', {'times_s': [0, 0.2]})
                        blocks = [block for block in images.content if block.type == 'image']
                        self.assertEqual(len(blocks), 4)
                        self.assertTrue(base64.b64decode(blocks[0].data).startswith(b'\x89PNG'))
                        await call('render_views', {'width': 9000}, ok=False)
                        await call('accept_candidate')
                        await call('save_project')
                        exported = await call('export_trajectory', {'format': 'mujoco_csv', 'file_name': 'motion'})
                        output = Path(exported.structuredContent['paths'][0])
                        self.assertTrue(output.resolve().is_relative_to(Path(directory).resolve()))
                        from application.csv_io import read_trajectory_csv
                        loaded = read_trajectory_csv(output, summary['nq'])
                        self.assertGreater(len(loaded.times), 3)
                        self.assertTrue(any(not np.array_equal(q, loaded.qposes[0]) for q in loaded.qposes))
                        await call('export_trajectory', {'format': 'mujoco_csv', 'file_name': '../escape'}, ok=False)
                        await call('export_trajectory', {'format': 'unsupported', 'file_name': 'bad'}, ok=False)
                        await call('load_model', {'model_key': 'go2'})
                        await call('new_project', {'name': 'quadruped', 'duration_s': 0.2})
                        unsupported = await call('export_trajectory', {'format': 'mjlab_csv', 'file_name': 'unsupported'}, ok=False)
                        self.assertIn('requires G1', unsupported.structuredContent['message'])
                        self.assertFalse(list(Path(directory).rglob('unsupported.csv')))
            errors.seek(0)
            diagnostics = errors.read()
            self.assertNotIn('Traceback', diagnostics)
            self.assertNotIn('Failed to parse', diagnostics)
            self.assertIn('deterministic renderer diagnostic', diagnostics)

    async def test_real_mujoco_images_over_stdio(self):
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode='w+') as errors:
            with anyio.fail_after(30):
                async with stdio_client(StdioServerParameters(command=sys.executable, args=[
                    '-m', 'application.mcp_server', '--projects-dir', directory]), errlog=errors) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool('render_views', {'views': ['front'], 'width': 160, 'height': 120})
                        self.assertFalse(result.isError, result)
                        unavailable = not result.structuredContent['available']
                        if not unavailable:
                            block = next(block for block in result.content if block.type == 'image')
                            self.assertTrue(base64.b64decode(block.data).startswith(b'\x89PNG'))
            if unavailable:
                self.skipTest(result.structuredContent['message'])

    async def test_installed_entry_point_and_review_policy(self):
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        command = Path(sys.executable).parent / ('ghostgui-mcp.exe' if os.name == 'nt' else 'ghostgui-mcp')
        self.assertTrue(command.exists(), 'install package with agent extra to test console entry point')
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode='w+') as errors:
            with anyio.fail_after(30):
                async with stdio_client(StdioServerParameters(command=str(command), args=[
                    '--projects-dir', directory], cwd=str(ROOT)), errlog=errors) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        resource = await session.read_resource('ghostgui://workspace')
                        self.assertEqual(json.loads(resource.contents[0].text)['approval'], 'review')
                        self.assertEqual(len((await session.list_tools()).tools), 24)
            errors.seek(0)
            self.assertIn('workspace=headless attach=never', errors.read())
        with tempfile.TemporaryDirectory() as directory:
            closed = subprocess.run([str(command), '--projects-dir', directory], input='',
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(closed.returncode, 0, closed.stderr)
            self.assertFalse(closed.stdout)
        for mode in ['required', 'auto']:
            result = subprocess.run([str(command), '--projects-dir', '.', '--attach', mode],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(result.stdout)
            self.assertIn('unavailable', result.stderr)


if __name__ == '__main__' and '--serve-stub' in sys.argv:
    from application.mcp_server import create_server, serve
    with redirect_stdout(sys.stderr):
        workspace = HeadlessWorkspace(sys.argv[-1], approval='auto_accept')
        server = create_server(workspace, renderer=StubRenderer())
    asyncio.run(serve(server))
elif __name__ == '__main__':
    unittest.main()

import os
import time
import unittest
from collections import deque
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.models import MuJoCoRobotAdapter
from gui.viewers.robot_canvas_3d import (
    GEOMETRY_FACE_CHUNK_SIZE,
    IndexedMeshBuffer,
    RobotCanvas3D,
)


class GeometryQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.adapter = MuJoCoRobotAdapter("g1")

    def test_scene_geometry_is_queued_without_opengl_compilation(self):
        canvas = RobotCanvas3D()
        try:
            canvas.scene_robot_adapters = {"robot": self.adapter}
            with patch(
                "gui.viewers.robot_canvas_3d.GL.glGenLists"
            ) as generate_list:
                canvas._queue_scene_robot_geometries()

            self.assertFalse(generate_list.called)
            self.assertGreater(len(canvas._geometry_queue), 35)
            mesh_tasks = [
                task for task in canvas._geometry_queue if task[3] is not None
            ]
            self.assertTrue(mesh_tasks)
            self.assertTrue(
                all(task[4] <= GEOMETRY_FACE_CHUNK_SIZE for task in mesh_tasks)
            )
        finally:
            canvas.close()

    def test_compile_callback_stops_after_time_budget(self):
        canvas = RobotCanvas3D()
        bucket = []
        canvas._geometry_queue = deque(
            [(object(), index, bucket, None, None) for index in range(10)]
        )
        canvas._geometry_total = 10
        try:
            with (
                patch.object(canvas, "isValid", return_value=True),
                patch.object(canvas, "isVisible", return_value=True),
                patch.object(canvas, "makeCurrent"),
                patch.object(canvas, "doneCurrent"),
                patch.object(
                    canvas,
                    "_draw_local_geom",
                    side_effect=lambda *_args, **_kwargs: time.sleep(0.004),
                ),
                patch("gui.viewers.robot_canvas_3d.GL.glGenLists", return_value=1),
                patch("gui.viewers.robot_canvas_3d.GL.glNewList"),
                patch("gui.viewers.robot_canvas_3d.GL.glEndList"),
                patch(
                    "gui.viewers.robot_canvas_3d.GEOMETRY_TIME_BUDGET_MS",
                    5.0,
                ),
            ):
                canvas._compile_next_geometry()

            self.assertGreater(canvas._geometry_completed, 0)
            self.assertLess(canvas._geometry_completed, 10)
            self.assertTrue(canvas._geometry_queue)
        finally:
            canvas.close()

    def test_mesh_chunk_builds_indexed_interleaved_geometry(self):
        model = self.adapter.mj_model
        mesh_geom = next(
            geom_id
            for geom_id in RobotCanvas3D.render_geom_ids(model)
            if int(model.geom_dataid[geom_id]) >= 0
        )

        vertices, indices = RobotCanvas3D._mesh_chunk_arrays(
            model,
            mesh_geom,
            0,
            min(8, int(model.mesh_facenum[int(model.geom_dataid[mesh_geom])])),
        )

        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 6)
        self.assertEqual(indices.ndim, 1)
        self.assertEqual(indices.size % 3, 0)
        self.assertLess(int(indices.max()), len(vertices))

    def test_mesh_task_uploads_vbo_and_ebo_instead_of_display_list(self):
        canvas = RobotCanvas3D()
        bucket = []
        canvas._geometry_queue = deque(
            [(object(), 3, bucket, 0, 10)]
        )
        canvas._geometry_total = 1
        resource = IndexedMeshBuffer(11, 12, 30)
        try:
            with (
                patch.object(canvas, "isValid", return_value=True),
                patch.object(canvas, "isVisible", return_value=True),
                patch.object(canvas, "makeCurrent"),
                patch.object(canvas, "doneCurrent"),
                patch.object(canvas, "_upload_mesh_chunk", return_value=resource),
                patch("gui.viewers.robot_canvas_3d.GL.glGenLists") as generate_list,
            ):
                canvas._compile_next_geometry()

            self.assertEqual(bucket, [resource])
            self.assertFalse(generate_list.called)
        finally:
            canvas.close()

    def test_secondary_detail_mode_can_skip_mesh_queue(self):
        canvas = RobotCanvas3D()
        try:
            canvas.scene_robot_adapters = {"robot": self.adapter}
            canvas.secondary_robot_detail = "skeleton"
            canvas._queue_scene_robot_geometries()

            self.assertFalse(canvas._geometry_queue)
            self.assertFalse(canvas._scene_robot_geom_lists)
        finally:
            canvas.close()

    def test_secondary_detail_mode_defers_existing_secondary_tasks(self):
        canvas = RobotCanvas3D()
        active_bucket = []
        secondary_bucket = []
        canvas._geom_lists = [active_bucket]
        canvas._geometry_queue = deque(
            [
                (object(), 1, secondary_bucket, None, None),
                (object(), 2, active_bucket, None, None),
            ]
        )
        canvas._geometry_total = 2
        canvas.secondary_robot_detail = "proxy"
        try:
            with (
                patch.object(canvas, "isValid", return_value=True),
                patch.object(canvas, "isVisible", return_value=True),
                patch.object(canvas, "makeCurrent"),
                patch.object(canvas, "doneCurrent"),
                patch.object(canvas, "_draw_local_geom"),
                patch("gui.viewers.robot_canvas_3d.GL.glGenLists", return_value=9),
                patch("gui.viewers.robot_canvas_3d.GL.glNewList"),
                patch("gui.viewers.robot_canvas_3d.GL.glEndList"),
            ):
                canvas._compile_next_geometry()

            self.assertEqual(active_bucket, [9])
            self.assertFalse(secondary_bucket)
            self.assertEqual(len(canvas._geometry_queue), 1)
            self.assertIs(canvas._geometry_queue[0][2], secondary_bucket)
        finally:
            canvas.close()

    def test_cleanup_releases_each_shared_gpu_resource_once(self):
        canvas = RobotCanvas3D()
        mesh = IndexedMeshBuffer(21, 22, 30)
        shared_bucket = [mesh]
        canvas._geom_lists = [shared_bucket, [31]]
        canvas._scene_robot_geom_lists = {"same": canvas._geom_lists}
        canvas._scene_mesh_display_lists = {"object": 32}
        canvas._quadric = object()
        try:
            with (
                patch.object(canvas, "isValid", return_value=True),
                patch.object(canvas, "makeCurrent"),
                patch.object(canvas, "doneCurrent"),
                patch("gui.viewers.robot_canvas_3d.GL.glDeleteBuffers") as delete_buffer,
                patch("gui.viewers.robot_canvas_3d.GL.glDeleteLists") as delete_list,
                patch("gui.viewers.robot_canvas_3d.GLU.gluDeleteQuadric") as delete_quadric,
            ):
                canvas.cleanup_gl_resources()

            self.assertEqual(delete_buffer.call_count, 2)
            self.assertEqual(delete_list.call_count, 2)
            delete_quadric.assert_called_once()
            self.assertFalse(canvas._geom_lists)
            self.assertFalse(canvas._scene_robot_geom_lists)
        finally:
            canvas.close()

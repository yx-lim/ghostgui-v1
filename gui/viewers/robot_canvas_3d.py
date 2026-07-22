"""
viewer_3d.py

Purpose:
    OpenGL 3D viewer/editor for the target reference frame and trajectory.

The editor shares the same small contract as the 2D side view:
    - update_scene(trajectory, active_frame)
    - target_dragged(x, z)
"""

import math
from pathlib import Path

import numpy as np

from OpenGL import GL
from OpenGL import GLU
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QMatrix4x4, QSurfaceFormat, QVector3D
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .trajectory_colors import gl_color_for_frame
from core.trajectory import rpy_to_quat
from core.scene import Transform
from core.scene.mesh import load_mesh_geometry
from .transform_gizmo import (
    GizmoInteractionState,
    TransformGizmo,
    normalize_quaternion,
)

TRAJECTORY_LINE_DT = 0.02


class RobotCanvas3D(QOpenGLWidget):
    target_dragged = Signal(float, float)
    target_pose_dragged = Signal(float, float, float)
    target_transform_dragged = Signal(object, object)
    transform_drag_finished = Signal()
    transform_drag_cancel_requested = Signal()
    scene_actor_transform_dragged = Signal(str, object, object)
    scene_actor_transform_drag_finished = Signal(str, object, object)
    gizmo_mode_changed = Signal(str)
    geometry_progress = Signal(int, int)
    body_double_clicked = Signal(str)
    scene_robot_body_double_clicked = Signal(str, str)
    camera_changed = Signal()

    def __init__(self):
        super().__init__()

        # This widget is an opaque scene. Request no composited alpha channel
        # and tell Qt not to expose the desktop through transparent GL pixels.
        surface_format = QSurfaceFormat(self.format())
        surface_format.setAlphaBufferSize(0)
        self.setFormat(surface_format)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.setMinimumSize(650, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.trajectory = None
        self.scene = None
        self.scene_asset_root = None
        self.scene_edit_actor_id = None
        self.scene_edit_target = None
        self.active_robot_actor_id = None
        self.scene_robot_adapters = {}
        self.scene_robot_states = {}
        self.show_trajectory_lines = True
        self.show_keyframes = True
        self.trajectory_smoothing = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.9
        self.target_yaw = 0.0

        self.camera_distance = 5.0
        self.camera_yaw = 38.0
        self.camera_pitch = 24.0
        self.camera_center = np.array([0.0, 0.0, 0.75], dtype=float)

        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None

        self._model_view = QMatrix4x4()
        self._projection = QMatrix4x4()
        self._viewport = QRect(0, 0, 1, 1)
        self.robot_state = None
        self.preview_state = None
        self.preview_visible = False
        self.preview_alpha = 0.65
        self.ghost_renderer = None
        self.show_ghosts = False
        self.ghost_alpha = 0.18
        self.use_model_colors = True
        self.selected_target_kind = None
        self.selected_target_name = None
        self.selected_body_id = None
        self._geom_lists = []
        self._mesh_display_lists = {}
        self._scene_mesh_display_lists = {}
        self._scene_robot_geom_lists = {}
        self._scene_robot_mesh_display_lists = {}
        self._quadric = None
        self.gizmo = TransformGizmo(
            (self.target_x, self.target_y, self.target_z)
        )
        self._geometry_build_count = 0
        self._geometry_queue = []
        self._geometry_total = 0
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setInterval(0)
        self._geometry_timer.timeout.connect(self._compile_next_geometry)

    def set_robot_state(self, robot_state, ghost_renderer=None):
        self.set_robot_states(robot_state, None, ghost_renderer)

    def set_robot_states(
        self, committed_state, preview_state=None, ghost_renderer=None
    ):
        self.robot_state = committed_state
        self.preview_state = preview_state
        self.ghost_renderer = ghost_renderer
        if self.isValid():
            self._build_robot_geometry()
        self.update()

    def set_preview_visible(self, visible):
        self.preview_visible = bool(visible and self.preview_state is not None)
        self.update()

    def set_preview_alpha(self, alpha):
        self.preview_alpha = max(0.1, min(1.0, float(alpha)))
        self.update()

    def set_ghost_options(self, visible, alpha=0.18):
        self.show_ghosts = bool(visible)
        self.ghost_alpha = max(0.02, min(0.8, float(alpha)))
        self.update()

    def set_use_model_colors(self, enabled):
        self.use_model_colors = bool(enabled)
        self.update()

    def set_selected_target(self, kind=None, name=None, owner_body_id=None):
        self.selected_target_kind = kind
        self.selected_target_name = name
        self.selected_body_id = (
            None if owner_body_id is None else int(owner_body_id)
        )
        self.update()

    def set_target_pose(self, position, quaternion=None):
        if self._scene_edit_kind() == "object":
            return
        position, quaternion = self._active_robot_pose_to_world(
            position,
            quaternion,
        )
        self.target_x, self.target_y, self.target_z = map(float, position)
        self.gizmo.set_pose(position, quaternion)
        self.update()

    # ============================================================
    # Scene API
    # ============================================================

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        scene=None,
        show_trajectory_lines=True,
        trajectory_smoothing=0.0,
        show_keyframes=True,
        scene_asset_root=None,
        scene_edit_actor_id=None,
        scene_robot_adapters=None,
        scene_edit_target=None,
        active_robot_actor_id=None,
        scene_robot_states=None,
    ):
        self.trajectory = trajectory
        self.scene = scene
        self.scene_asset_root = (
            None if scene_asset_root is None else Path(scene_asset_root)
        )
        self.scene_edit_actor_id = (
            None if scene_edit_actor_id is None else str(scene_edit_actor_id)
        )
        self.scene_edit_target = (
            dict(scene_edit_target) if isinstance(scene_edit_target, dict) else None
        )
        self.active_robot_actor_id = (
            None if active_robot_actor_id is None else str(active_robot_actor_id)
        )
        self.scene_robot_adapters = dict(scene_robot_adapters or {})
        self.scene_robot_states = dict(scene_robot_states or {})
        self.show_trajectory_lines = show_trajectory_lines
        self.show_keyframes = show_keyframes
        self.trajectory_smoothing = max(0.0, min(1.0, float(trajectory_smoothing)))

        if active_frame is not None and self._scene_edit_kind() != "object":
            position, quaternion = self._active_robot_pose_to_world(
                (active_frame.x, active_frame.y, active_frame.z),
                rpy_to_quat(
                    active_frame.roll,
                    active_frame.pitch,
                    active_frame.yaw,
                ),
            )
            self.target_x, self.target_y, self.target_z = map(float, position)
            self.target_yaw = active_frame.yaw
            self.gizmo.set_pose(position, quaternion)

        self._sync_scene_edit_actor_pose()
        self.update()

    # ============================================================
    # OpenGL lifecycle
    # ============================================================

    def initializeGL(self):
        GL.glClearColor(0.08, 0.09, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(8.0)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.28, 0.28, 0.28, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.78, 0.78, 0.78, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        self._quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self._quadric, GLU.GLU_SMOOTH)
        # Display lists belong to this OpenGL context. A restored/cached widget
        # keeps them; a genuinely new context rebuilds them incrementally.
        self._geom_lists = []
        self._mesh_display_lists = {}
        self._scene_mesh_display_lists = {}
        self._scene_robot_geom_lists = {}
        self._scene_robot_mesh_display_lists = {}
        self._geometry_queue = []
        self._build_robot_geometry()

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def paintGL(self):
        # Reassert an opaque framebuffer on every frame. Transparent robot
        # passes preserve destination alpha rather than changing window alpha.
        GL.glClearColor(0.08, 0.09, 0.10, 1.0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.configure_camera()

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(self.matrix_values(self._projection))

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(self.matrix_values(self._model_view))

        self.draw_ground_grid()
        self.draw_world_axes()
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (2.0, -3.0, 5.0, 1.0))
        self.draw_scene_objects()
        self.draw_scene_robots()
        self.draw_robot()
        self.draw_trajectory_ghosts()
        self.draw_preview_robot()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_LIGHT0)
        self.draw_trajectory()
        self.draw_selected_target_marker()
        self.draw_transform_gizmo()

    def _build_robot_geometry(self):
        """Queue local geometry so Qt can repaint between expensive meshes."""
        if self.robot_state is None or self._geom_lists:
            return
        model = self.robot_state.mj_model
        self._geom_lists = [None] * model.ngeom
        render_ids = self.render_geom_ids(model)
        # Native MJCF models usually use visual group 2; imported URDF visuals
        # use non-colliding group 1. render_geom_ids handles both conventions.
        self._geometry_queue = sorted(render_ids)
        self._geometry_total = len(self._geometry_queue)
        self.geometry_progress.emit(0, self._geometry_total)
        if self._geometry_queue:
            self._geometry_timer.start()

    def _compile_next_geometry(self):
        if not self._geometry_queue:
            self._geometry_timer.stop()
            return
        if not self.isValid() or self.robot_state is None:
            return
        geom_id = self._geometry_queue.pop(0)
        model = self.robot_state.mj_model
        import mujoco
        mesh_id = (
            int(model.geom_dataid[geom_id])
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            else None
        )
        if mesh_id is not None and mesh_id in self._mesh_display_lists:
            self._geom_lists[geom_id] = self._mesh_display_lists[mesh_id]
            self._finish_geometry_item()
            return
        self.makeCurrent()
        try:
            list_id = GL.glGenLists(1)
            GL.glNewList(list_id, GL.GL_COMPILE)
            self._draw_local_geom(model, geom_id)
            GL.glEndList()
            self._geom_lists[geom_id] = list_id
            if mesh_id is not None:
                self._mesh_display_lists[mesh_id] = list_id
        finally:
            self.doneCurrent()
        self._finish_geometry_item()

    def _finish_geometry_item(self):
        complete = self._geometry_total - len(self._geometry_queue)
        self.geometry_progress.emit(complete, self._geometry_total)
        self.update()
        if not self._geometry_queue:
            self._geometry_timer.stop()
            self._geometry_build_count += 1

    @staticmethod
    def render_geom_ids(model):
        visual_ids = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == 2
        }
        if not visual_ids:
            # MuJoCo imports URDF <visual> elements as non-colliding group-1
            # geoms. Prefer those over the primitive collision shapes.
            visual_ids = {
                geom_id for geom_id in range(model.ngeom)
                if int(model.geom_group[geom_id]) == 1
                and int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            }
        return visual_ids or set(range(model.ngeom))

    def _draw_local_geom(self, model, geom_id):
        import mujoco

        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            vertices = model.mesh_vert[
                vertex_start:vertex_start + int(model.mesh_vertnum[mesh_id])
            ]
            normal_start = int(model.mesh_normaladr[mesh_id])
            normals = model.mesh_normal[
                normal_start:normal_start + int(model.mesh_normalnum[mesh_id])
            ]
            faces = model.mesh_face[
                face_start:face_start + int(model.mesh_facenum[mesh_id])
            ]
            face_normals = model.mesh_facenormal[
                face_start:face_start + int(model.mesh_facenum[mesh_id])
            ]
            GL.glBegin(GL.GL_TRIANGLES)
            for face, normal_ids in zip(faces, face_normals):
                fallback_normal = np.cross(
                    vertices[int(face[1])] - vertices[int(face[0])],
                    vertices[int(face[2])] - vertices[int(face[0])],
                )
                fallback_norm = float(np.linalg.norm(fallback_normal))
                if fallback_norm > 1e-12:
                    fallback_normal /= fallback_norm
                for vertex_id, normal_id in zip(face, normal_ids):
                    if int(normal_id) >= 0:
                        GL.glNormal3fv(normals[int(normal_id)])
                    elif fallback_norm > 1e-12:
                        GL.glNormal3fv(fallback_normal)
                    GL.glVertex3fv(vertices[int(vertex_id)])
            GL.glEnd()
            return
        if self._quadric is None:
            return
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            GLU.gluSphere(self._quadric, float(size[0]), 12, 8)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            GL.glPushMatrix()
            GL.glScalef(float(size[0]), float(size[1]), float(size[2]))
            GLU.gluSphere(self._quadric, 1.0, 12, 8)
            GL.glPopMatrix()
        elif geom_type in (
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        ):
            radius, half_length = float(size[0]), float(size[1])
            GL.glTranslatef(0.0, 0.0, -half_length)
            GLU.gluCylinder(self._quadric, radius, radius, 2.0 * half_length, 12, 1)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                GLU.gluSphere(self._quadric, radius, 12, 8)
                GL.glTranslatef(0.0, 0.0, 2.0 * half_length)
                GLU.gluSphere(self._quadric, radius, 12, 8)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            x, y, z = (float(v) for v in size)
            vertices = [
                (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
            ]
            faces = [
                ((0, 1, 2, 3), (0, 0, -1)),
                ((4, 7, 6, 5), (0, 0, 1)),
                ((0, 4, 5, 1), (0, -1, 0)),
                ((1, 5, 6, 2), (1, 0, 0)),
                ((2, 6, 7, 3), (0, 1, 0)),
                ((4, 0, 3, 7), (-1, 0, 0)),
            ]
            GL.glBegin(GL.GL_QUADS)
            for face, normal in faces:
                GL.glNormal3fv(normal)
                for vertex in face:
                    GL.glVertex3fv(vertices[vertex])
            GL.glEnd()

    @staticmethod
    def _transform_matrix(position, rotation):
        matrix = [0.0] * 16
        rotation = rotation.reshape(3, 3)
        for row in range(3):
            for column in range(3):
                matrix[column * 4 + row] = float(rotation[row, column])
        matrix[15] = 1.0
        matrix[12:15] = [float(v) for v in position]
        return matrix

    def _draw_robot_transforms(
        self, positions, rotations, alpha_scale=1.0, color_override=None
    ):
        if self.robot_state is None or not self._geom_lists:
            return
        model = self.robot_state.mj_model
        self._draw_model_transforms(
            model,
            positions,
            rotations,
            self._geom_lists,
            lambda geom_id: self.robot_state.robot_model.get_geom_rgba(geom_id),
            alpha_scale=alpha_scale,
            color_override=color_override,
            selected_body_id=self.selected_body_id,
        )

    def _draw_model_transforms(
        self,
        model,
        positions,
        rotations,
        display_lists,
        rgba_lookup,
        alpha_scale=1.0,
        color_override=None,
        selected_body_id=None,
    ):
        for geom_id, list_id in enumerate(display_lists):
            if list_id is None:
                continue
            selected = (
                color_override is None
                and selected_body_id is not None
                and int(model.geom_bodyid[geom_id]) == int(selected_body_id)
            )
            if color_override is not None:
                rgba = color_override
            elif self.use_model_colors:
                rgba = rgba_lookup(geom_id)
            else:
                rgba = (0.55, 0.55, 0.55, 1.0)
            if selected:
                rgba = self._selected_body_rgba(rgba)
            self._apply_geom_material(model, geom_id, selected=selected)
            GL.glColor4f(float(rgba[0]), float(rgba[1]), float(rgba[2]),
                         float(rgba[3]) * alpha_scale)
            GL.glPushMatrix()
            GL.glMultMatrixf(self._transform_matrix(positions[geom_id], rotations[geom_id]))
            GL.glCallList(list_id)
            GL.glPopMatrix()

    def _geom_is_selected_body(self, model, geom_id):
        return (
            self.selected_body_id is not None
            and int(model.geom_bodyid[geom_id]) == int(self.selected_body_id)
        )

    @staticmethod
    def _selected_body_rgba(rgba):
        rgba = np.asarray(rgba, dtype=float)
        bright = np.clip(
            rgba[:3] * 1.35 + np.array([0.18, 0.14, 0.03]),
            0.0,
            1.0,
        )
        return (float(bright[0]), float(bright[1]), float(bright[2]), float(rgba[3]))

    @staticmethod
    def _apply_geom_material(model, geom_id, selected=False):
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            specular = float(model.mat_specular[material_id])
            shininess = float(model.mat_shininess[material_id]) * 128.0
            emission = float(model.mat_emission[material_id])
        else:
            specular, shininess, emission = 0.2, 32.0, 0.0
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SPECULAR,
            (specular, specular, specular, 1.0),
        )
        GL.glMaterialf(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SHININESS,
            max(0.0, min(128.0, shininess)),
        )
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_EMISSION,
            (
                min(1.0, emission + (0.18 if selected else 0.0)),
                min(1.0, emission + (0.14 if selected else 0.0)),
                min(1.0, emission + (0.04 if selected else 0.0)),
                1.0,
            ),
        )

    def draw_robot(self):
        if self.robot_state is None:
            return
        actor = (
            None
            if self.scene is None or self.active_robot_actor_id is None
            else self.scene.actors.get(self.active_robot_actor_id)
        )
        if actor is not None and not actor.visible:
            return
        data = self.robot_state.mj_data
        self._push_actor_transform(actor)
        try:
            self._draw_robot_transforms(data.geom_xpos, data.geom_xmat)
        finally:
            self._pop_actor_transform(actor)

    def draw_scene_robots(self):
        scene = self.scene
        if scene is None or self._quadric is None:
            return
        for actor in scene.actors.robots():
            if not actor.visible:
                continue
            state = self.scene_robot_states.get(actor.id)
            adapter = self.scene_robot_adapters.get(actor.id)
            if state is None:
                continue
            robot_model = getattr(state, "robot_model", None)
            if robot_model is None:
                robot_model = adapter
            model = getattr(state, "mj_model", None)
            if model is None:
                model = getattr(adapter, "mj_model", None)
            data = getattr(state, "mj_data", None)
            if model is None or robot_model is None:
                continue
            display_lists = self._ensure_scene_robot_geometry(robot_model)
            if not display_lists:
                continue
            transform = actor.world_transform
            GL.glPushMatrix()
            rotation = self._quaternion_rotation_matrix(transform.quaternion)
            GL.glMultMatrixf(self._transform_matrix(transform.position, rotation))
            try:
                self._draw_model_transforms(
                    model,
                    data.geom_xpos,
                    data.geom_xmat,
                    display_lists,
                    robot_model.get_geom_rgba,
                    alpha_scale=0.82 if actor.locked else 1.0,
                )
            finally:
                GL.glPopMatrix()

    @staticmethod
    def _pop_actor_transform(actor):
        if actor is not None:
            GL.glPopMatrix()

    def _push_actor_transform(self, actor):
        if actor is None:
            return
        transform = actor.world_transform
        GL.glPushMatrix()
        rotation = self._quaternion_rotation_matrix(transform.quaternion)
        GL.glMultMatrixf(self._transform_matrix(transform.position, rotation))

    def _ensure_scene_robot_geometry(self, adapter):
        if self._quadric is None:
            return None
        model = adapter.mj_model
        cache_key = self._scene_robot_geometry_key(adapter)
        display_lists = self._scene_robot_geom_lists.get(cache_key)
        if display_lists is not None and len(display_lists) == model.ngeom:
            return display_lists

        mesh_lists = {}
        display_lists = [None] * model.ngeom
        render_ids = sorted(self.render_geom_ids(model))
        import mujoco
        for geom_id in render_ids:
            mesh_id = (
                int(model.geom_dataid[geom_id])
                if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                else None
            )
            if mesh_id is not None and mesh_id in mesh_lists:
                display_lists[geom_id] = mesh_lists[mesh_id]
                continue
            list_id = GL.glGenLists(1)
            GL.glNewList(list_id, GL.GL_COMPILE)
            self._draw_local_geom(model, geom_id)
            GL.glEndList()
            display_lists[geom_id] = list_id
            if mesh_id is not None:
                mesh_lists[mesh_id] = list_id
        self._scene_robot_geom_lists[cache_key] = display_lists
        self._scene_robot_mesh_display_lists[cache_key] = mesh_lists
        return display_lists

    @staticmethod
    def _scene_robot_geometry_key(adapter):
        path = getattr(adapter, "runtime_model_path", None) or getattr(
            adapter,
            "model_path",
            None,
        )
        return str(Path(path).expanduser().resolve()) if path else str(id(adapter))

    def draw_scene_objects(self):
        scene = self.scene
        if scene is None or self._quadric is None:
            return
        time = float(getattr(scene.timeline, "current_time", 0.0))
        for actor in scene.visible_object_actors():
            reference = actor.model_reference or {}
            transform = scene.tracks.object_transform_at(actor, time)
            rgba = self._scene_object_rgba(actor, reference)
            if actor.locked:
                rgba = [
                    min(1.0, float(rgba[0]) * 0.7 + 0.22),
                    min(1.0, float(rgba[1]) * 0.7 + 0.22),
                    min(1.0, float(rgba[2]) * 0.7 + 0.22),
                    float(rgba[3]),
                ]
            transparent = float(rgba[3]) < 0.999
            if transparent:
                self._begin_transparent_pass()
            try:
                GL.glColor4f(
                    float(rgba[0]), float(rgba[1]),
                    float(rgba[2]), float(rgba[3])
                )
                GL.glMaterialfv(
                    GL.GL_FRONT_AND_BACK,
                    GL.GL_SPECULAR,
                    (0.22, 0.22, 0.22, 1.0),
                )
                GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 34.0)
                GL.glMaterialfv(
                    GL.GL_FRONT_AND_BACK,
                    GL.GL_EMISSION,
                    (0.02, 0.02, 0.02, 1.0),
                )
                GL.glPushMatrix()
                rotation = self._quaternion_rotation_matrix(transform.quaternion)
                GL.glMultMatrixf(self._transform_matrix(transform.position, rotation))
                reference_type = reference.get("type")
                if reference_type == "primitive":
                    self._draw_scene_primitive(
                        str(reference.get("shape") or "box"),
                        reference.get("size") or (0.2, 0.2, 0.2),
                    )
                elif reference_type == "mesh":
                    scale = self._scene_mesh_scale(reference)
                    GL.glScalef(float(scale[0]), float(scale[1]), float(scale[2]))
                    self._draw_scene_mesh(reference)
                GL.glPopMatrix()
            finally:
                if transparent:
                    self._end_transparent_pass()

    @staticmethod
    def _scene_object_rgba(actor, reference):
        rgba = list(reference.get("rgba") or (0.20, 0.58, 0.88, 1.0))[:4]
        while len(rgba) < 4:
            rgba.append(1.0)
        return [float(value) for value in rgba]

    @staticmethod
    def _scene_mesh_scale(reference):
        scale = list(reference.get("scale") or (1.0, 1.0, 1.0))[:3]
        while len(scale) < 3:
            scale.append(1.0)
        return [max(0.0001, float(value)) for value in scale]

    def _draw_scene_mesh(self, reference):
        path = self._resolve_scene_mesh_path(reference)
        if path is None:
            return
        try:
            stat = path.stat()
            cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            return
        list_id = self._scene_mesh_display_lists.get(cache_key)
        if list_id is None:
            try:
                geometry = load_mesh_geometry(path)
            except (OSError, ValueError):
                return
            list_id = GL.glGenLists(1)
            GL.glNewList(list_id, GL.GL_COMPILE)
            self._compile_scene_mesh_geometry(geometry)
            GL.glEndList()
            self._scene_mesh_display_lists[cache_key] = list_id
        GL.glCallList(list_id)

    def _resolve_scene_mesh_path(self, reference):
        asset_path = reference.get("asset_path")
        if not asset_path:
            return None
        path = Path(asset_path).expanduser()
        if path.is_absolute():
            return path
        if self.scene_asset_root is None:
            return None
        return self.scene_asset_root / path

    @staticmethod
    def _compile_scene_mesh_geometry(geometry):
        GL.glBegin(GL.GL_TRIANGLES)
        for face in geometry.faces:
            vertices = [
                np.asarray(geometry.vertices[int(index)], dtype=float)
                for index in face
            ]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                normal /= norm
                GL.glNormal3fv(normal)
            for vertex in vertices:
                GL.glVertex3fv(vertex)
        GL.glEnd()

    def _sync_scene_edit_actor_pose(self):
        if self.gizmo.is_dragging:
            return
        actor = self._scene_edit_actor()
        if actor is None:
            return
        time = float(getattr(self.scene.timeline, "current_time", 0.0))
        transform = self.scene.tracks.object_transform_at(actor, time)
        self.target_x, self.target_y, self.target_z = map(
            float,
            transform.position,
        )
        self.gizmo.set_pose(transform.position, transform.quaternion)

    def _scene_edit_kind(self):
        if not isinstance(self.scene_edit_target, dict):
            return None
        return self.scene_edit_target.get("kind")

    def _gizmo_enabled(self):
        return self._scene_edit_kind() in {"object", "robot"}

    def _active_robot_transform(self):
        actor = self._active_robot_actor()
        return None if actor is None else actor.world_transform

    def _active_robot_actor(self):
        if self.scene is None or self.active_robot_actor_id is None:
            return None
        return self.scene.actors.get(self.active_robot_actor_id)

    def _active_robot_pose_to_world(self, position, quaternion=None):
        transform = self._active_robot_transform()
        if transform is None:
            return np.asarray(position, dtype=float), quaternion
        if quaternion is None:
            quaternion = (1.0, 0.0, 0.0, 0.0)
            include_quaternion = False
        else:
            include_quaternion = True
        world = transform.compose(
            Transform(tuple(map(float, position)), tuple(map(float, quaternion)))
        )
        return np.asarray(world.position, dtype=float), (
            np.asarray(world.quaternion, dtype=float)
            if include_quaternion else None
        )

    def _world_pose_to_active_robot(self, position, quaternion):
        transform = self._active_robot_transform()
        if transform is None:
            return np.asarray(position, dtype=float), normalize_quaternion(quaternion)
        local = transform.inverse().compose(
            Transform(
                tuple(map(float, position)),
                tuple(map(float, quaternion)),
            )
        )
        return (
            np.asarray(local.position, dtype=float),
            np.asarray(local.quaternion, dtype=float),
        )

    def _scene_edit_actor(self):
        if self._scene_edit_kind() != "object":
            return None
        actor_id = self.scene_edit_target.get("actor_id")
        if self.scene is None or actor_id is None:
            return None
        actor = self.scene.actors.get(str(actor_id))
        if actor is None or actor.kind != "object" or actor.locked:
            return None
        return actor

    def _draw_scene_primitive(self, shape, size):
        size = [float(value) for value in list(size)[:3]]
        while len(size) < 3:
            size.append(size[-1] if size else 0.2)
        shape = shape.lower()
        if shape == "sphere":
            GLU.gluSphere(self._quadric, max(0.001, size[0]), 24, 16)
            return
        if shape == "cylinder":
            radius = max(0.001, size[0])
            height = max(0.001, size[2])
            GL.glTranslatef(0.0, 0.0, -height * 0.5)
            GLU.gluCylinder(self._quadric, radius, radius, height, 24, 1)
            GLU.gluDisk(self._quadric, 0.0, radius, 24, 1)
            GL.glTranslatef(0.0, 0.0, height)
            GLU.gluDisk(self._quadric, 0.0, radius, 24, 1)
            return
        x, y, z = (max(0.001, value * 0.5) for value in size)
        vertices = [
            (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
            (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
        ]
        faces = [
            ((0, 1, 2, 3), (0, 0, -1)),
            ((4, 7, 6, 5), (0, 0, 1)),
            ((0, 4, 5, 1), (0, -1, 0)),
            ((1, 5, 6, 2), (1, 0, 0)),
            ((2, 6, 7, 3), (0, 1, 0)),
            ((4, 0, 3, 7), (-1, 0, 0)),
        ]
        GL.glBegin(GL.GL_QUADS)
        for face, normal in faces:
            GL.glNormal3fv(normal)
            for vertex in face:
                GL.glVertex3fv(vertices[vertex])
        GL.glEnd()

    def draw_preview_robot(self):
        if not self.preview_visible or self.preview_state is None:
            return
        data = self.preview_state.mj_data
        actor = self._active_robot_actor()
        self._push_actor_transform(actor)
        if self.preview_alpha >= 0.999:
            try:
                self._draw_robot_transforms(
                    data.geom_xpos,
                    data.geom_xmat,
                    color_override=(1.0, 0.38, 0.04, 1.0),
                )
            finally:
                self._pop_actor_transform(actor)
            return
        self._begin_transparent_pass()
        try:
            self._draw_robot_transforms(
                data.geom_xpos,
                data.geom_xmat,
                alpha_scale=self.preview_alpha,
                color_override=(1.0, 0.38, 0.04, 1.0),
            )
        finally:
            self._end_transparent_pass()
            self._pop_actor_transform(actor)

    def draw_trajectory_ghosts(self):
        if not self.show_ghosts or self.ghost_renderer is None:
            return
        self._begin_transparent_pass()
        actor = self._active_robot_actor()
        self._push_actor_transform(actor)
        try:
            for positions, rotations in self.ghost_renderer.transforms:
                # The animated main pose may exactly equal one cached waypoint.
                # Skipping that duplicate avoids coincident transparent surfaces.
                if self.robot_state is not None and np.allclose(
                    positions,
                    self.robot_state.mj_data.geom_xpos,
                    atol=1e-8,
                    rtol=0.0,
                ):
                    continue
                self._draw_robot_transforms(positions, rotations, self.ghost_alpha)
        finally:
            self._pop_actor_transform(actor)
            self._end_transparent_pass()

    @staticmethod
    def _begin_transparent_pass():
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ZERO,
            GL.GL_ONE,
        )
        GL.glDepthMask(GL.GL_FALSE)
        # Preserve the opaque alpha written by glClear. Standard blending also
        # blends destination alpha, which made QOpenGLWidget composite desktop
        # pixels through the preview on some window managers.
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)

    @staticmethod
    def _end_transparent_pass():
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def configure_camera(self):
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height

        eye_array = self._camera_eye()
        eye = QVector3D(*map(float, eye_array))
        center = QVector3D(*map(float, self.camera_center))
        up = QVector3D(0.0, 0.0, 1.0)

        self._projection = QMatrix4x4()
        self._projection.perspective(45.0, aspect, 0.05, 100.0)

        self._model_view = QMatrix4x4()
        self._model_view.lookAt(eye, center, up)

        self._viewport = QRect(0, 0, width, height)

    def matrix_values(self, matrix):
        data = matrix.data()
        return [data[i] for i in range(16)]

    # ============================================================
    # Drawing helpers
    # ============================================================

    def draw_ground_grid(self):
        GL.glLineWidth(1.0)
        GL.glColor3f(0.30, 0.33, 0.35)

        GL.glBegin(GL.GL_LINES)
        for i in range(-10, 11):
            v = i * 0.25
            GL.glVertex3f(-2.5, v, 0.0)
            GL.glVertex3f(2.5, v, 0.0)
            GL.glVertex3f(v, -2.5, 0.0)
            GL.glVertex3f(v, 2.5, 0.0)
        GL.glEnd()

        GL.glLineWidth(2.0)
        GL.glColor3f(0.55, 0.58, 0.60)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(-2.5, 0.0, 0.0)
        GL.glVertex3f(2.5, 0.0, 0.0)
        GL.glVertex3f(0.0, -2.5, 0.0)
        GL.glVertex3f(0.0, 2.5, 0.0)
        GL.glEnd()

    def draw_world_axes(self):
        GL.glLineWidth(3.0)
        GL.glBegin(GL.GL_LINES)

        GL.glColor3f(0.90, 0.15, 0.12)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.6, 0.0, 0.02)

        GL.glColor3f(0.20, 0.75, 0.25)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.6, 0.02)

        GL.glColor3f(0.20, 0.45, 0.95)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.0, 0.6)

        GL.glEnd()

    def draw_trajectory(self):
        if self.trajectory is None or len(self.trajectory.frames) == 0:
            return

        actor = self._active_robot_actor()
        self._push_actor_transform(actor)
        try:
            self._draw_robot_trajectory_local()
        finally:
            self._pop_actor_transform(actor)

    def _draw_robot_trajectory_local(self):
        if self.show_trajectory_lines:
            samples = self.trajectory.sample_tracks_uniform_dt(
                dt=TRAJECTORY_LINE_DT,
                smoothing=self.trajectory_smoothing,
            )
            frames_by_name = {}
            for sample in samples:
                for frame_name, target in sample["targets"].items():
                    frames_by_name.setdefault(frame_name, []).append(target)

            GL.glLineWidth(2.0)

            for frame_name, targets in frames_by_name.items():
                if len(targets) < 2:
                    continue

                GL.glColor3f(*gl_color_for_frame(frame_name))
                GL.glBegin(GL.GL_LINE_STRIP)
                for target in targets:
                    GL.glVertex3f(target.x, target.y, target.z)
                GL.glEnd()

        if self.show_keyframes:
            GL.glPointSize(7.0)
            GL.glBegin(GL.GL_POINTS)
            for frame in self.trajectory.frames:
                GL.glColor3f(*gl_color_for_frame(frame.frame_name))
                GL.glVertex3f(frame.x, frame.y, frame.z)
            GL.glEnd()

    def draw_target_frame(self):
        x = self.target_x
        y = self.target_y
        z = self.target_z

        GL.glPointSize(12.0)
        GL.glColor3f(1.0, 0.12, 0.08)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(x, y, z)
        GL.glEnd()

    def draw_selected_target_marker(self):
        if self.selected_target_name is None or self._scene_edit_kind() != "robot":
            return
        origin = np.asarray(self.gizmo.position, dtype=float)
        rotation = self._quaternion_rotation_matrix(self.gizmo.quaternion)
        length = max(self.gizmo.sphere_radius * 2.6, 0.045)
        colors = (
            ((1.0, 0.10, 0.08), rotation[:, 0]),
            ((0.10, 0.85, 0.25), rotation[:, 1]),
            ((0.22, 0.50, 1.0), rotation[:, 2]),
        )

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(4.0)
        GL.glBegin(GL.GL_LINES)
        for color, axis in colors:
            endpoint = origin + np.asarray(axis, dtype=float) * length
            GL.glColor3f(*color)
            GL.glVertex3fv(origin)
            GL.glVertex3fv(endpoint)
        GL.glEnd()
        GL.glPointSize(7.0)
        GL.glColor3f(1.0, 0.92, 0.18)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3fv(origin)
        GL.glEnd()
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def _quaternion_rotation_matrix(quaternion):
        w, x, y, z = np.asarray(quaternion, dtype=float)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-12:
            return np.eye(3)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ], dtype=float)

    def draw_transform_gizmo(self):
        if not self._gizmo_enabled():
            return
        self._sync_gizmo_screen_scale()
        origin = self.gizmo.position
        show_sphere, translation_axes, rotation_axes = (
            self.gizmo.visible_handles()
        )
        colors = {
            "x": (0.9, 0.1, 0.1),
            "y": (0.1, 0.8, 0.2),
            "z": (0.2, 0.45, 1.0),
        }
        GL.glDisable(GL.GL_DEPTH_TEST)
        self._draw_gizmo_drag_guide(colors)
        sphere_hovered = (
            self.gizmo.state == GizmoInteractionState.HOVER_TRANSLATE_FREE
        )
        sphere_color = (
            (1.0, 0.9, 0.15) if sphere_hovered else (0.95, 0.75, 0.2)
        )
        GL.glColor3f(*sphere_color)
        if show_sphere and self._quadric is not None:
            GL.glPushMatrix()
            GL.glTranslatef(*map(float, origin))
            GLU.gluSphere(self._quadric, self.gizmo.sphere_radius, 16, 10)
            GL.glPopMatrix()

        for axis, delta in (("x", (self.gizmo.arrow_length, 0.0, 0.0)),
                            ("y", (0.0, self.gizmo.arrow_length, 0.0)),
                            ("z", (0.0, 0.0, self.gizmo.arrow_length))):
            if axis not in translation_axes:
                continue
            self._draw_gizmo_arrow(
                origin,
                axis,
                self._gizmo_color(axis, "TRANSLATE", colors[axis]),
            )

        self._begin_transparent_pass()
        GL.glLineWidth(4.0 if self.gizmo.is_dragging else 3.2)
        for axis in rotation_axes:
            self._draw_gizmo_ring(
                axis,
                self._gizmo_color(axis, "ROTATE", colors[axis]),
            )
        self._end_transparent_pass()
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _gizmo_color(self, axis, operation, base_color):
        state_name = self.gizmo.state.name
        if (
            state_name.startswith("HOVER_")
            and operation in state_name
            and state_name.endswith(axis.upper())
        ):
            return (1.0, 0.9, 0.15)
        return base_color

    def _sync_gizmo_screen_scale(self):
        self.configure_camera()
        origin = np.asarray(self.project_point(*self.gizmo.position), dtype=float)
        sample = 0.1
        pixels_per_unit = []
        for vector in ((sample, 0.0, 0.0), (0.0, sample, 0.0), (0.0, 0.0, sample)):
            point = self.gizmo.position + np.asarray(vector, dtype=float)
            distance = float(np.linalg.norm(np.asarray(self.project_point(*point)) - origin))
            if distance > 1e-6:
                pixels_per_unit.append(distance / sample)
        if pixels_per_unit:
            self.gizmo.set_screen_scale(1.0 / float(np.median(pixels_per_unit)))

    def _draw_gizmo_arrow(self, origin, axis, color):
        if self._quadric is None:
            return
        vector = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }[axis]
        shaft_start = self.gizmo.sphere_radius * 1.25
        cone_length = min(self.gizmo.arrow_length * 0.32, self.gizmo.sphere_radius * 1.85)
        shaft_length = max(0.01, self.gizmo.arrow_length - shaft_start - cone_length)
        shaft_radius = max(self.gizmo.sphere_radius * 0.18, self.gizmo.arrow_length * 0.018)
        cone_radius = max(self.gizmo.sphere_radius * 0.42, shaft_radius * 2.2)

        GL.glColor3f(*color)
        GL.glPushMatrix()
        GL.glTranslatef(*map(float, np.asarray(origin) + vector * shaft_start))
        self._orient_z_to_axis(axis)
        GLU.gluCylinder(self._quadric, shaft_radius, shaft_radius, shaft_length, 12, 1)
        GL.glTranslatef(0.0, 0.0, shaft_length)
        GLU.gluCylinder(self._quadric, cone_radius, 0.0, cone_length, 18, 1)
        GL.glPopMatrix()

    def _draw_gizmo_ring(self, axis, color):
        eye = self._camera_eye()
        origin = np.asarray(self.gizmo.position, dtype=float)
        points = self.gizmo.ring_points(axis)
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
            facing = float(np.dot(midpoint - origin, eye - origin))
            alpha = 0.88 if facing >= 0.0 else 0.32
            GL.glColor4f(float(color[0]), float(color[1]), float(color[2]), alpha)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3fv(start)
            GL.glVertex3fv(end)
            GL.glEnd()

    def _draw_gizmo_drag_guide(self, colors):
        if not self.gizmo.is_dragging:
            return
        state = self.gizmo.state
        axis = getattr(self.gizmo, "_drag_axis", None)
        if axis not in colors:
            return
        self._begin_transparent_pass()
        try:
            color = colors[axis]
            GL.glColor4f(float(color[0]), float(color[1]), float(color[2]), 0.28)
            GL.glLineWidth(2.0)
            if "TRANSLATE" in state.name:
                vector = {
                    "x": np.array([1.0, 0.0, 0.0]),
                    "y": np.array([0.0, 1.0, 0.0]),
                    "z": np.array([0.0, 0.0, 1.0]),
                }[axis]
                origin = np.asarray(self.gizmo.position, dtype=float)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3fv(origin - vector * 2.0)
                GL.glVertex3fv(origin + vector * 2.0)
                GL.glEnd()
            elif "ROTATE" in state.name:
                GL.glLineWidth(6.0)
                self._draw_gizmo_ring(axis, color)
        finally:
            self._end_transparent_pass()

    @staticmethod
    def _orient_z_to_axis(axis):
        if axis == "x":
            GL.glRotatef(90.0, 0.0, 1.0, 0.0)
        elif axis == "y":
            GL.glRotatef(-90.0, 1.0, 0.0, 0.0)

    def _camera_eye(self):
        return self.camera_center + self._camera_offset()

    def _camera_offset(self):
        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)
        return np.array([
            self.camera_distance * math.cos(pitch) * math.sin(yaw),
            -self.camera_distance * math.cos(pitch) * math.cos(yaw),
            self.camera_distance * math.sin(pitch),
        ], dtype=float)

    def _camera_basis(self):
        forward = self.camera_center - self._camera_eye()
        forward /= max(1e-12, float(np.linalg.norm(forward)))
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= float(np.linalg.norm(right))
        up = np.cross(right, forward)
        up /= max(1e-12, float(np.linalg.norm(up)))
        return right, up, forward

    def _orbit_camera(self, dx, dy):
        self.camera_yaw -= float(dx) * 0.4
        self.camera_pitch = max(
            -85.0,
            min(85.0, self.camera_pitch + float(dy) * 0.3),
        )

    def _pan_camera(self, dx, dy):
        right, up, _ = self._camera_basis()
        view_height = 2.0 * self.camera_distance * math.tan(math.radians(45.0) * 0.5)
        units_per_pixel = view_height / max(1, self.height())
        self.camera_center += (
            -right * float(dx) * units_per_pixel
            + up * float(dy) * units_per_pixel
        )

    def _zoom_camera(self, amount):
        self.camera_distance = max(
            1.8,
            min(10.0, self.camera_distance + float(amount)),
        )

    # ============================================================
    # Mouse editing
    # ============================================================

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.last_mouse_pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton:
            self._sync_gizmo_screen_scale()
            if self._gizmo_enabled() and self.gizmo.begin_drag(
                event.position().x(),
                event.position().y(),
                self.project_point,
                self.screen_ray,
            ):
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
                return
            self.rotating_camera = True
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.panning_camera = True
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.zooming_camera = True
            return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self.pick_scene_robot_body(
                event.position().x(), event.position().y()
            )
            if hit:
                actor_id, body_name = hit
                if actor_id is not None:
                    self.scene_robot_body_double_clicked.emit(actor_id, body_name)
                else:
                    self.body_double_clicked.emit(body_name)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.gizmo.is_dragging:
            self._sync_gizmo_screen_scale()
            modifiers = event.modifiers()
            position, quaternion = self.gizmo.drag(
                event.position().x(),
                event.position().y(),
                self.project_point,
                self.screen_ray,
                fine=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
                snap=bool(modifiers & Qt.KeyboardModifier.ControlModifier),
            )
            self.target_x, self.target_y, self.target_z = map(float, position)
            if self._scene_edit_actor() is not None:
                self.scene_actor_transform_dragged.emit(
                    str(self.scene_edit_target["actor_id"]),
                    position,
                    quaternion,
                )
            else:
                local_position, local_quaternion = (
                    self._world_pose_to_active_robot(position, quaternion)
                )
                self.target_transform_dragged.emit(
                    local_position,
                    local_quaternion,
                )
            self.update()
            return

        if self.rotating_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._orbit_camera(delta.x(), delta.y())
            self.last_mouse_pos = event.position()
            self.update()
            return

        if self.panning_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._pan_camera(delta.x(), delta.y())
            self.last_mouse_pos = event.position()
            self.update()
            return

        if self.zooming_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._zoom_camera(-delta.y() * 0.02)
            self.last_mouse_pos = event.position()
            self.update()
            return

        old_state = self.gizmo.state
        self._sync_gizmo_screen_scale()
        new_state = self.gizmo.hover(
            event.position().x(), event.position().y(), self.project_point
        )
        if new_state != old_state:
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if new_state != GizmoInteractionState.NONE
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        transform_was_dragging = self.gizmo.is_dragging
        camera_was_interacting = (
            self.rotating_camera or self.panning_camera or self.zooming_camera
        )
        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None
        self.gizmo.end_drag()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        if transform_was_dragging:
            if self._scene_edit_actor() is not None:
                self.scene_actor_transform_drag_finished.emit(
                    str(self.scene_edit_target["actor_id"]),
                    self.gizmo.position.copy(),
                    self.gizmo.quaternion.copy(),
                )
            else:
                self.transform_drag_finished.emit()
            return
        if camera_was_interacting:
            self.camera_changed.emit()
            return
        super().mouseReleaseEvent(event)

    def cancel_transform_drag(self, emit_cancelled=False):
        """Cancel interaction state without emitting a completed edit."""
        self.gizmo.end_drag()
        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        if emit_cancelled:
            self.transform_drag_cancel_requested.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_T:
            self.gizmo.set_mode("translate")
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.gizmo_mode_changed.emit("translate")
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_R:
            self.gizmo.set_mode("rotate")
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.gizmo_mode_changed.emit("rotate")
            self.update()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_E, Qt.Key.Key_Escape):
            self.cancel_transform_drag(emit_cancelled=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self._zoom_camera(steps * 0.35)
        self.update()
        self.camera_changed.emit()

    # ============================================================
    # Projection helpers
    # ============================================================

    def project_point(self, x, y, z):
        self.configure_camera()
        point = QVector3D(x, y, z)
        screen = point.project(self._model_view, self._projection, self._viewport)
        return screen.x(), self.height() - screen.y()

    def screen_ray(self, sx, sy):
        self.configure_camera()
        near = QVector3D(sx, self.height() - sy, 0.0).unproject(
            self._model_view, self._projection, self._viewport
        )
        far = QVector3D(sx, self.height() - sy, 1.0).unproject(
            self._model_view, self._projection, self._viewport
        )
        origin = np.array([near.x(), near.y(), near.z()], dtype=float)
        direction = np.array(
            [far.x() - near.x(), far.y() - near.y(), far.z() - near.z()],
            dtype=float,
        )
        direction /= max(1e-12, float(np.linalg.norm(direction)))
        return origin, direction

    def pick_robot_body(self, sx, sy):
        return self.pick_robot_body_from_ray(*self.screen_ray(sx, sy))

    def pick_scene_robot_body(self, sx, sy):
        return self.pick_scene_robot_body_from_ray(*self.screen_ray(sx, sy))

    def pick_robot_body_from_ray(self, origin, direction):
        hit = self.pick_scene_robot_body_from_ray(origin, direction)
        if hit is None:
            return None
        actor_id, body_name = hit
        if self.active_robot_actor_id is None or actor_id == self.active_robot_actor_id:
            return body_name
        return None

    def pick_scene_robot_body_from_ray(self, origin, direction):
        """Return ``(actor_id, body_name)`` for the closest visible robot hit."""
        if self.robot_state is None:
            return None
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction /= max(1e-12, float(np.linalg.norm(direction)))
        active_data = (
            self.preview_state.mj_data
            if self.preview_visible and self.preview_state is not None
            else self.robot_state.mj_data
        )
        candidates = []
        active_actor = self._active_robot_actor()
        if active_actor is None or active_actor.visible:
            candidates.append((
                self.active_robot_actor_id,
                self.robot_state.mj_model,
                active_data,
                self._active_robot_transform(),
            ))
        if self.scene is not None:
            for actor in self.scene.actors.robots():
                if actor.id == self.active_robot_actor_id or not actor.visible:
                    continue
                state = self.scene_robot_states.get(actor.id)
                adapter = self.scene_robot_adapters.get(actor.id)
                if state is None:
                    continue
                model = getattr(state, "mj_model", None)
                if model is None:
                    model = getattr(adapter, "mj_model", None)
                data = getattr(state, "mj_data", None)
                if model is not None and data is not None:
                    candidates.append((actor.id, model, data, actor.world_transform))

        best_by_actor = {}
        for actor_id, model, data, transform in candidates:
            rotation = (
                np.eye(3)
                if transform is None
                else self._quaternion_rotation_matrix(transform.quaternion)
            )
            translation = (
                np.zeros(3)
                if transform is None
                else np.asarray(transform.position, dtype=float)
            )
            for geom_id in self.render_geom_ids(model):
                body_id = int(model.geom_bodyid[geom_id])
                if body_id == 0:
                    continue
                center = rotation @ np.asarray(
                    data.geom_xpos[geom_id], dtype=float
                ) + translation
                radius = max(0.015, float(model.geom_rbound[geom_id]))
                offset = origin - center
                b = float(np.dot(offset, direction))
                c = float(np.dot(offset, offset) - radius * radius)
                discriminant = b * b - c
                if discriminant < 0.0:
                    continue
                root = math.sqrt(discriminant)
                distance = -b - root
                if distance < 0.0:
                    distance = -b + root
                if distance < 0.0:
                    continue
                closest = origin + direction * max(0.0, -b)
                ray_distance = float(np.linalg.norm(closest - center))
                ray_distance_bucket = int(ray_distance / 0.005)
                candidate = (
                        ray_distance_bucket,
                        distance,
                        actor_id,
                        body_id,
                        model,
                    )
                actor_key = actor_id or "__active_robot__"
                current = best_by_actor.get(actor_key)
                if current is None or candidate[:2] < current[:2]:
                    best_by_actor[actor_key] = candidate
        if not best_by_actor:
            return None
        best = min(best_by_actor.values(), key=lambda candidate: candidate[1])
        import mujoco
        return best[2], mujoco.mj_id2name(
            best[4], mujoco.mjtObj.mjOBJ_BODY, best[3]
        )

    def screen_to_edit_plane(self, sx, sy):
        """
        Convert a screen point to the Y=0 edit plane.

        The reference-frame editor currently exposes X/Z controls, so the 3D
        editor drags the target within that same side-view plane.
        """

        self.configure_camera()

        near = QVector3D(sx, self.height() - sy, 0.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )
        far = QVector3D(sx, self.height() - sy, 1.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )

        direction = far - near
        if abs(direction.y()) < 1e-6:
            return self.target_x, self.target_z

        t = -near.y() / direction.y()
        point = near + direction * t
        return point.x(), point.z()

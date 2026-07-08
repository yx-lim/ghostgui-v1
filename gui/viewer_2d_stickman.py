# gui/viewer_2d_stickman.py

"""
2D stickman viewer with simple body-part target attachment.

Purpose:
    - Display simplified stickman
    - Attach red target frame to selected body part
    - Drag selected target frame
    - Move stickman with simple 2D kinematics
    - Keep this separate from the real 3D MuJoCo robot

Important:
    This is NOT the real robot IK.
    This is only a visual/debugging model for the 2D GUI.
"""

import math
import mujoco
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPen, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene

from .trajectory_colors import qt_color_for_frame


# ============================================================
# Small 2D vector helpers
# ============================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def distance(a, b):
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    return math.sqrt(dx * dx + dz * dz)


def two_link_ik(root, target, l1, l2, bend_sign=1.0):
    """
    Fixed-length 2D two-link IK.

    root:
        shoulder or hip point, (x, z)

    target:
        desired hand or foot point, (x, z)

    l1:
        upper limb length

    l2:
        lower limb length

    bend_sign:
        controls elbow/knee bend direction

    Returns:
        joint point, clamped endpoint point

    Important:
        This version never stretches the limb.
        If target is outside the reachable radius, the endpoint is clamped.
    """

    rx, rz = root
    tx, tz = target

    dx = tx - rx
    dz = tz - rz

    raw_d = math.sqrt(dx * dx + dz * dz)

    if raw_d < 1e-9:
        # Avoid division by zero.
        dx = l1 + l2
        dz = 0.0
        raw_d = l1 + l2

    ux = dx / raw_d
    uz = dz / raw_d

    max_reach = l1 + l2 - 1e-6
    min_reach = abs(l1 - l2) + 1e-6

    # Clamp distance to valid two-link range.
    d = clamp(raw_d, min_reach, max_reach)

    # Clamp actual endpoint too.
    tx = rx + ux * d
    tz = rz + uz * d

    # Perpendicular direction.
    px = -uz
    pz = ux

    # Law of cosines.
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h_sq = max(0.0, l1 * l1 - a * a)
    h = math.sqrt(h_sq)

    joint_x = rx + a * ux + bend_sign * h * px
    joint_z = rz + a * uz + bend_sign * h * pz

    return (joint_x, joint_z), (tx, tz)

# ============================================================
# Stickman pose model
# ============================================================

class StickmanPose:
    """
    Simplified 2D stickman model.

    Coordinates:
        x = forward/horizontal
        z = vertical

    This stores body points in world coordinates.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Body anchor points.
        self.pelvis = (0.0, 0.80)
        self.torso = (0.0, 1.15)
        self.head = (0.0, 1.38)

        self.left_shoulder = (-0.12, 1.15)
        self.right_shoulder = (0.12, 1.15)

        self.left_hip = (-0.10, 0.80)
        self.right_hip = (0.10, 0.80)

        self.left_foot = (-0.18, 0.00)
        self.right_foot = (0.18, 0.00)

        self.left_hand = (-0.42, 0.85)
        self.right_hand = (0.42, 0.85)

        # IK intermediate joints.
        self.left_knee = (-0.18, 0.40)
        self.right_knee = (0.18, 0.40)

        self.left_elbow = (-0.30, 1.00)
        self.right_elbow = (0.30, 1.00)

        # Segment lengths for the simplified drawing.
        self.upper_leg_len = 0.42
        self.lower_leg_len = 0.42

        self.upper_arm_len = 0.28
        self.lower_arm_len = 0.28
        self.torso_len = distance(self.pelvis, self.torso)

        self.update_all_ik()

    def translate_body_raw(self, dx, dz):
        """
        Move all main body points without immediately recomputing IK.

        This is used when a dragged hand/foot pulls the body along.
        """

        names = [
            "pelvis",
            "torso",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ]

        for name in names:
            x, z = getattr(self, name)
            setattr(self, name, (x + dx, z + dz))


    def translate_body(self, dx, dz):
        """
        Move the whole stickman and recompute IK.
        """

        self.translate_body_raw(dx, dz)
        self.update_all_ik()


    def pull_body_until_reachable(self, root_name, target, max_reach):
        """
        If target is too far from the selected limb root,
        move the whole stickman so the target becomes reachable.

        Example:
            dragging left hand too far
                -> left shoulder follows
                -> torso/pelvis/legs also follow
                -> arm length stays fixed
        """

        root = getattr(self, root_name)

        rx, rz = root
        tx, tz = target

        dx = tx - rx
        dz = tz - rz

        d = math.sqrt(dx * dx + dz * dz)

        if d <= max_reach:
            return

        if d < 1e-9:
            return

        ux = dx / d
        uz = dz / d

        # New root should be exactly max_reach away from target.
        desired_root_x = tx - ux * max_reach
        desired_root_z = tz - uz * max_reach

        move_x = desired_root_x - rx
        move_z = desired_root_z - rz

        self.translate_body_raw(move_x, move_z)

    def set_pelvis(self, x, z):
        old_x, old_z = self.pelvis
        dx = x - old_x
        dz = z - old_z

        self.translate_body(dx, dz)

    def set_torso(self, x, z):
        """
        Move torso/head/shoulders.

        If the torso target is out of range from the pelvis, pull the whole
        body along so the pelvis-to-torso length stays fixed.
        """

        pelvis_x, pelvis_z = self.pelvis
        dx = x - pelvis_x
        dz = z - pelvis_z
        d = math.sqrt(dx * dx + dz * dz)

        if d > self.torso_len:
            ux = dx / d
            uz = dz / d

            desired_pelvis_x = x - ux * self.torso_len
            desired_pelvis_z = z - uz * self.torso_len

            move_x = desired_pelvis_x - pelvis_x
            move_z = desired_pelvis_z - pelvis_z

            self.translate_body_raw(move_x, move_z)

        self.torso = (x, z)
        self.head = (x, z + 0.23)

        self.left_shoulder = (x - 0.12, z)
        self.right_shoulder = (x + 0.12, z)

        self.update_all_ik()

    def set_left_hand(self, x, z):
        target = (x, z)

        max_reach = self.upper_arm_len + self.lower_arm_len
        self.pull_body_until_reachable(
            root_name="left_shoulder",
            target=target,
            max_reach=max_reach,
        )

        self.left_hand = target
        self.update_all_ik()


    def set_right_hand(self, x, z):
        target = (x, z)

        max_reach = self.upper_arm_len + self.lower_arm_len
        self.pull_body_until_reachable(
            root_name="right_shoulder",
            target=target,
            max_reach=max_reach,
        )

        self.right_hand = target
        self.update_all_ik()


    def set_left_foot(self, x, z):
        target = (x, z)

        max_reach = self.upper_leg_len + self.lower_leg_len
        self.pull_body_until_reachable(
            root_name="left_hip",
            target=target,
            max_reach=max_reach,
        )

        self.left_foot = target
        self.update_all_ik()


    def set_right_foot(self, x, z):
        target = (x, z)

        max_reach = self.upper_leg_len + self.lower_leg_len
        self.pull_body_until_reachable(
            root_name="right_hip",
            target=target,
            max_reach=max_reach,
        )

        self.right_foot = target
        self.update_all_ik()

    def update_all_ik(self):
        self.update_left_leg_ik()
        self.update_right_leg_ik()
        self.update_left_arm_ik()
        self.update_right_arm_ik()

    def update_left_leg_ik(self):
        self.left_knee, self.left_foot = two_link_ik(
            root=self.left_hip,
            target=self.left_foot,
            l1=self.upper_leg_len,
            l2=self.lower_leg_len,
            bend_sign=-1.0,
        )

    def update_right_leg_ik(self):
        self.right_knee, self.right_foot = two_link_ik(
            root=self.right_hip,
            target=self.right_foot,
            l1=self.upper_leg_len,
            l2=self.lower_leg_len,
            bend_sign=1.0,
        )

    def update_left_arm_ik(self):
        self.left_elbow, self.left_hand = two_link_ik(
            root=self.left_shoulder,
            target=self.left_hand,
            l1=self.upper_arm_len,
            l2=self.lower_arm_len,
            bend_sign=1.0,
        )

    def update_right_arm_ik(self):
        self.right_elbow, self.right_hand = two_link_ik(
            root=self.right_shoulder,
            target=self.right_hand,
            l1=self.upper_arm_len,
            l2=self.lower_arm_len,
            bend_sign=-1.0,
        )

    def get_body_point(self, frame_name):
        """
        Return the current world position of a body frame.
        Used when the GUI selects a new target frame.
        """

        if frame_name in ["pelvis", "base", "root"]:
            return self.pelvis

        if frame_name == "torso":
            return self.torso

        if frame_name == "left_foot":
            return self.left_foot

        if frame_name == "right_foot":
            return self.right_foot

        if frame_name == "left_hand":
            return self.left_hand

        if frame_name == "right_hand":
            return self.right_hand

        # Default fallback.
        return self.pelvis

    def apply_target_frame(self, frame):
        """
        Move the stickman according to the active TargetFrame.
        """

        name = frame.frame_name

        if name in ["pelvis", "base", "root"]:
            self.set_pelvis(frame.x, frame.z)

        elif name == "torso":
            self.set_torso(frame.x, frame.z)

        elif name == "left_foot":
            self.set_left_foot(frame.x, frame.z)

        elif name == "right_foot":
            self.set_right_foot(frame.x, frame.z)

        elif name == "left_hand":
            self.set_left_hand(frame.x, frame.z)

        elif name == "right_hand":
            self.set_right_hand(frame.x, frame.z)

    def apply_target_snapshot(self, targets):
        """
        Rebuild the simplified stickman from a same-time target snapshot while
        preserving the same reach constraints used during dragging.
        """
        self.reset()

        for name in [
            "pelvis",
            "base",
            "root",
            "torso",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ]:
            frame = targets.get(name)
            if frame is not None:
                self.apply_target_frame(frame)


# ============================================================
# 2D viewer widget
# ============================================================

class Stickman2DViewer(QGraphicsView):
    target_dragged = Signal(float, float)

    def __init__(self, adapter=None):
        super().__init__()

        self.adapter = adapter
        self.skeleton_state = adapter.create_state() if adapter is not None else None

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setMinimumSize(650, 500)
        self.setSceneRect(-325, -250, 650, 500)

        self.scale_pixels_per_meter = 180

        self.pose = StickmanPose()

        self.selected_frame_name = "pelvis"
        self.target_x = 0.0
        self.target_z = 0.8
        self.target_yaw = 0.0

        self.dragging_target = False
        self._skeleton_targets = {}
        self._last_projected_positions = {}
        self._last_editable_projected_points = {}

    def set_adapter(self, adapter):
        self.adapter = adapter
        self.skeleton_state = adapter.create_state() if adapter is not None else None
        self.pose.reset()
        self._skeleton_targets = {}
        self._last_editable_projected_points = {}

    # ============================================================
    # Coordinate transforms
    # ============================================================

    def world_to_screen(self, x, z):
        sx = x * self.scale_pixels_per_meter
        sy = -z * self.scale_pixels_per_meter + 180
        return sx, sy

    def screen_to_world(self, sx, sy):
        x = sx / self.scale_pixels_per_meter
        z = -(sy - 180) / self.scale_pixels_per_meter
        return x, z

    # ============================================================
    # External API used by main_window.py
    # ============================================================

    def get_body_point(self, frame_name):
        """
        Return current position of a body part.
        Used to attach the target marker when frame selector changes.
        """

        if self.adapter is None:
            return self.pose.get_body_point(frame_name)
        binding = self.adapter.resolve_logical_frame(frame_name)
        if binding is None:
            return 0.0, 0.0
        kind, name = binding
        try:
            position, _ = (
                self.adapter.get_site_pose(name)
                if kind == "site" else self.adapter.get_body_pose(name)
            )
        except KeyError:
            return 0.0, 0.0
        return self.project_position(position)

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        apply_active_frame=True,
        show_trajectory_lines=True,
    ):
        """
        Redraw viewer.

        If active_frame is given:
            - move simplified stickman based on active_frame
            - attach red target marker to selected body part
        """

        if active_frame is not None:
            self.selected_frame_name = active_frame.frame_name

            if apply_active_frame and self.adapter is None:
                targets = trajectory.targets_at_time(active_frame.time)
                existing = targets.get(active_frame.frame_name)
                if (
                    existing is None
                    or abs(existing.x - active_frame.x) > 0.005
                    or abs(existing.z - active_frame.z) > 0.005
                    or abs(existing.yaw - active_frame.yaw) > 0.005
                ):
                    targets[active_frame.frame_name] = active_frame
                self.pose.apply_target_snapshot(targets)

            if apply_active_frame and self.adapter is None:
                point_x, point_z = self.pose.get_body_point(active_frame.frame_name)
            else:
                point_x = active_frame.x
                point_z = active_frame.z

            self.target_x = point_x
            self.target_z = point_z
            self.target_yaw = active_frame.yaw

            if self.adapter is not None:
                self._skeleton_targets = {}
                if apply_active_frame:
                    self._skeleton_targets = trajectory.targets_at_time(
                        active_frame.time
                    )
                    self._skeleton_targets[active_frame.frame_name] = active_frame
                self.solve_generated_skeleton()

        self.scene.clear()

        self.draw_ground()
        self.draw_stickman()
        self.draw_trajectory(trajectory, show_lines=show_trajectory_lines)
        self.draw_target_frame()
        self.draw_legend()

    # ============================================================
    # Drawing
    # ============================================================

    def draw_ground(self):
        ground_y = self.world_to_screen(0.0, 0.0)[1]
        self.scene.addLine(
            -325,
            ground_y,
            325,
            ground_y,
            QPen(Qt.GlobalColor.black, 2),
        )

    def draw_stickman(self):
        if self.adapter is not None:
            self.draw_generated_skeleton()
            return
        pen_body = QPen(Qt.GlobalColor.black, 4)
        pen_joint = QPen(Qt.GlobalColor.black, 2)
        brush_joint = QBrush(Qt.GlobalColor.white)

        def line(a, b):
            ax, ay = self.world_to_screen(a[0], a[1])
            bx, by = self.world_to_screen(b[0], b[1])
            self.scene.addLine(ax, ay, bx, by, pen_body)

        def joint(p, radius=5):
            x, y = self.world_to_screen(p[0], p[1])
            self.scene.addEllipse(
                x - radius,
                y - radius,
                2 * radius,
                2 * radius,
                pen_joint,
                brush_joint,
            )

        p = self.pose

        # Torso / head
        line(p.pelvis, p.torso)
        line(p.torso, p.head)

        # Shoulders
        line(p.left_shoulder, p.right_shoulder)

        # Left arm
        line(p.left_shoulder, p.left_elbow)
        line(p.left_elbow, p.left_hand)

        # Right arm
        line(p.right_shoulder, p.right_elbow)
        line(p.right_elbow, p.right_hand)

        # Pelvis line
        line(p.left_hip, p.right_hip)

        # Left leg
        line(p.left_hip, p.left_knee)
        line(p.left_knee, p.left_foot)

        # Right leg
        line(p.right_hip, p.right_knee)
        line(p.right_knee, p.right_foot)

        # Head circle
        hx, hy = self.world_to_screen(p.head[0], p.head[1])
        self.scene.addEllipse(
            hx - 12,
            hy - 12,
            24,
            24,
            QPen(Qt.GlobalColor.black, 2),
            QBrush(Qt.GlobalColor.white),
        )

        # Draw important joints
        for body_point in [
            p.pelvis,
            p.torso,
            p.left_foot,
            p.right_foot,
            p.left_hand,
            p.right_hand,
            p.left_knee,
            p.right_knee,
            p.left_elbow,
            p.right_elbow,
        ]:
            joint(body_point)

    def project_position(self, position):
        """Readable side projection with a small lateral separation cue."""
        x, y, z = map(float, position)
        lateral = 0.22 if self.adapter.model_type == "humanoid" else 0.12
        return x + lateral * y, z

    def solve_generated_skeleton(self):
        """Apply target snapshots through joint-limited MuJoCo kinematics."""
        if self.skeleton_state is None:
            return
        self.skeleton_state.set_qpos(self.adapter.home_qpos)

        def is_root_target(frame):
            binding = self.adapter.resolve_logical_frame(frame.frame_name)
            return bool(binding and binding[1] == self.adapter.root_body)

        targets = sorted(
            self._skeleton_targets.values(), key=lambda frame: not is_root_target(frame)
        )
        for frame in targets:
            binding = self.adapter.resolve_logical_frame(frame.frame_name)
            if binding is None:
                continue
            kind, name = binding
            target = np.array([frame.x, frame.y, frame.z], dtype=float)
            self.skeleton_state.solve_ik(
                name,
                target,
                kind=kind,
                target_quaternion=None,
                orientation_weight=0.0,
                tolerance=0.003,
                max_iterations=100,
            )
            current, _ = self.skeleton_state.get_body_pose(name, kind)
            residual = target - current
            # Preserve rigid link lengths for out-of-reach targets by pulling
            # the floating root and the rest of the robot along, matching the
            # original 2D editor's whole-body follow behavior.
            if (
                np.linalg.norm(residual) > 0.005
                and name != self.adapter.root_body
                and self.adapter.free_joints_by_body
            ):
                free_joint = next(iter(self.adapter.free_joints_by_body.values()))
                qpos = self.skeleton_state.get_qpos()
                address = free_joint.qpos_address
                qpos[address:address + 3] += residual
                self.skeleton_state.set_qpos(qpos)
                self.skeleton_state.solve_ik(
                    name,
                    target,
                    kind=kind,
                    target_quaternion=None,
                    orientation_weight=0.0,
                    tolerance=0.003,
                    max_iterations=40,
                )

    def _editable_projected_points(self, projected):
        """
        Return editable logical-frame points keyed by the GUI frame name.

        The displayed label uses the original MuJoCo body/site name so the 2D
        skeleton reflects the source XML/URDF object instead of the friendly
        logical alias used by the trajectory editor.
        """
        editable = {}
        if self.adapter is None:
            return editable
        frame_bindings = self.adapter.logical_frame_bindings.items()
        for logical_name, (kind, object_name) in frame_bindings:
            point = projected.get(object_name)
            if point is not None:
                editable[logical_name] = (kind, object_name, point)
        return editable

    def draw_generated_skeleton(self):
        state = self.skeleton_state
        positions = {}
        for body_id in range(self.adapter.mj_model.nbody):
            name = mujoco.mj_id2name(
                self.adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                positions[name] = state.mj_data.xpos[body_id].copy()
        ignored = tuple(
            token.lower() for token in self.adapter.info.ignored_body_tokens
        )

        def relevant(name):
            return name == "world" or not any(token in name.lower() for token in ignored)

        site_parents = {}
        for logical, (kind, name) in self.adapter.logical_frame_bindings.items():
            if kind != "site":
                continue
            site_id = mujoco.mj_name2id(
                self.adapter.mj_model, mujoco.mjtObj.mjOBJ_SITE, name
            )
            if site_id < 0:
                continue
            positions[name] = state.mj_data.site_xpos[site_id].copy()
            body_id = int(self.adapter.mj_model.site_bodyid[site_id])
            site_parents[name] = mujoco.mj_id2name(
                self.adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )

        projected = {
            name: self.project_position(position)
            for name, position in positions.items() if relevant(name)
        }
        edges = []
        for parent, child in self.adapter.get_kinematic_edges():
            if child not in projected:
                continue
            effective_parent = parent
            while effective_parent not in projected and effective_parent:
                effective_parent = self.adapter.get_parent_body(effective_parent)
            if effective_parent in projected and effective_parent != child:
                edges.append((effective_parent, child))

        # Logical sites are useful endpoints even when the MuJoCo body tree
        # ends at an ankle/palm link.
        for name, parent in site_parents.items():
            if parent in projected:
                edges.append((parent, name))

        self._last_projected_positions = dict(projected)
        editable_points = self._editable_projected_points(projected)
        self._last_editable_projected_points = dict(editable_points)

        pen_body = QPen(Qt.GlobalColor.black, 4)
        pen_joint = QPen(Qt.GlobalColor.black, 2)
        brush_joint = QBrush(Qt.GlobalColor.white)
        for parent, child in edges:
            ax, ay = self.world_to_screen(*projected[parent])
            bx, by = self.world_to_screen(*projected[child])
            self.scene.addLine(ax, ay, bx, by, pen_body)

        drawn_bindings = set()
        for logical_name, (_, object_name, point) in editable_points.items():
            if object_name in drawn_bindings:
                continue
            drawn_bindings.add(object_name)
            x, y = self.world_to_screen(*point)
            radius = 5 if object_name != self.adapter.root_body else 7
            handle = self.scene.addEllipse(
                x - radius, y - radius, 2 * radius, 2 * radius,
                pen_joint, brush_joint,
            )
            handle.setData(0, "editable_handle")
            handle.setData(1, logical_name)
            handle.setData(2, object_name)
            label = self.scene.addText(object_name)
            label.setData(0, "editable_label")
            label.setData(1, logical_name)
            label.setData(2, object_name)
            label.setPos(x + radius + 4, y - radius - 12)

    def draw_trajectory(self, trajectory, show_lines=True):
        """
        Draw stored keyframe target positions.
        """

        if len(trajectory.frames) == 0:
            return

        previous_by_frame = {}

        for frame in trajectory.frames:
            x, y = self.world_to_screen(frame.x, frame.z)
            color = qt_color_for_frame(frame.frame_name)
            pen_line = QPen(color, 2)
            pen_point = QPen(color, 2)
            brush_point = QBrush(color)

            self.scene.addEllipse(
                x - 5,
                y - 5,
                10,
                10,
                pen_point,
                brush_point,
            )

            self.scene.addText(
                f"{frame.frame_name}\n{frame.time:.1f}s"
            ).setPos(x + 6, y - 24)

            previous = previous_by_frame.get(frame.frame_name)
            if show_lines and previous is not None:
                px, py = self.world_to_screen(previous.x, previous.z)
                self.scene.addLine(px, py, x, y, pen_line)

            previous_by_frame[frame.frame_name] = frame

    def draw_target_frame(self):
        """
        Draw red target frame attached to selected body part.
        """

        x, y = self.world_to_screen(self.target_x, self.target_z)

        pen_target = QPen(Qt.GlobalColor.red, 3)
        brush_target = QBrush(Qt.GlobalColor.red)

        self.scene.addEllipse(
            x - 9,
            y - 9,
            18,
            18,
            pen_target,
            brush_target,
        )

        arrow_len = 45
        x2 = x + arrow_len * math.cos(self.target_yaw)
        y2 = y - arrow_len * math.sin(self.target_yaw)

        self.scene.addLine(x, y, x2, y2, pen_target)

        self.scene.addText(
            f"target: {self.selected_frame_name}"
        ).setPos(x + 12, y + 12)

    def draw_legend(self):
        label = (
            f"Black = {self.adapter.model_name} kinematic skeleton"
            if self.adapter is not None else "Black = simplified stickman"
        )
        self.scene.addText(label).setPos(-315, -240)
        self.scene.addText("Red = selected target frame").setPos(-315, -215)
        self.scene.addText("Colored = stored per-frame keyframes").setPos(-315, -190)

    # ============================================================
    # Mouse dragging
    # ============================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())

            target_sx, target_sy = self.world_to_screen(
                self.target_x,
                self.target_z,
            )

            dx = scene_pos.x() - target_sx
            dy = scene_pos.y() - target_sy

            if math.sqrt(dx * dx + dy * dy) < 25:
                self.dragging_target = True
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_target:
            scene_pos = self.mapToScene(event.position().toPoint())

            x, z = self.screen_to_world(scene_pos.x(), scene_pos.y())

            # Prevent dragging below ground.
            z = max(0.0, z)

            self.target_x = x
            self.target_z = z

            self.target_dragged.emit(x, z)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging_target = False
        super().mouseReleaseEvent(event)

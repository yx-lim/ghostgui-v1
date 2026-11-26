"""Qt implementation of the ``gui`` command port. Every method runs on the Qt main thread.

UI-state commands call the same widgets and slots a user click reaches, so dirty
marking and signal reactions match manual use. Nothing here opens a modal dialog.
"""

from __future__ import annotations

import numpy as np

from application.commands.sanitize import untrusted

CAMERA_PRESETS = {
    "front": {"yaw": 90.0, "pitch": 10.0},
    "side": {"yaw": 180.0, "pitch": 10.0},
    "back": {"yaw": 270.0, "pitch": 10.0},
    "top": {"yaw": 90.0, "pitch": 89.0},
    "default": {"yaw": 38.0, "pitch": 24.0, "distance": 5.0},
}
DIGEST_SECTIONS = ("project", "motion", "candidate", "playback", "view", "selection", "history")


class GuiCommandPort:
    def __init__(self, window):
        self.window = window

    # ------------------------------------------------------------------ state
    def _viewer(self):
        return self.window.viewer_3d

    def _keyframe_times(self):
        document = self.window.document
        times = {round(float(frame.time), 6) for frame in document.trajectory.frames}
        times.update(round(float(time), 6) for time in document.qpos_timeline.times())
        return sorted(times)

    def app_state(self, sections=()):
        w, viewer = self.window, self._viewer()
        wanted = set(sections) or {"project", "motion", "candidate", "playback", "view", "selection",
                                   "history", "camera", "display", "commands"}
        state = {}
        if "project" in wanted:
            project = w.current_project
            state["project"] = {
                "name": None if project is None else untrusted(project.project_name),
                "open": project is not None, "unsaved_changes": bool(w.project_dirty),
            }
        if "motion" in wanted:
            times = self._keyframe_times()
            state["motion"] = {
                "model_key": untrusted(w.model_key), "duration_s": round(float(w.document.timeline_duration), 4),
                "current_time_s": round(float(viewer.get_current_time()), 4),
                "keyframe_count": len(times), "keyframe_times_s": times[:32],
                "revision": int(w.document.revision),
            }
        if "candidate" in wanted:
            bridge = w.agent_bridge_controller
            state["candidate"] = {"orange_preview_staged": bool(bridge.has_candidate),
                                  "external_control": bridge.server is not None}
        if "playback" in wanted:
            state["playback"] = {"playing": bool(viewer.play_timer.isActive()),
                                 "speed": round(float(viewer.playback_speed.value()), 2)}
        if "view" in wanted:
            tabs = w.viewer_tabs
            state["view"] = {
                "active_view": tabs.tabText(tabs.currentIndex()),
                "views": [tabs.tabText(index) for index in range(tabs.count())],
                "left_sidebar": not w._left_sidebar_collapsed, "right_sidebar": not w._right_sidebar_collapsed,
            }
        if "selection" in wanted:
            box = w.controls.frame_box
            state["selection"] = {
                "frame": untrusted(box.currentText()),
                "frames": [untrusted(box.itemText(index)) for index in range(min(box.count(), 32))],
                "gizmo_visible": bool(getattr(w, "gizmo_visibility_action", None)
                                      and w.gizmo_visibility_action.isChecked()),
            }
        if "history" in wanted:
            state["history"] = {
                "undo": untrusted(w.undo_stack[-1].description) if w.undo_stack else None,
                "redo": untrusted(w.redo_stack[-1].description) if w.redo_stack else None,
            }
        if "camera" in wanted:
            canvas = viewer.canvas
            state["camera"] = {"yaw": round(float(canvas.camera_yaw), 2), "pitch": round(float(canvas.camera_pitch), 2),
                               "distance": round(float(canvas.camera_distance), 3)}
        if "display" in wanted:
            state["display"] = {
                "model_colors": bool(viewer.model_colors_box.isChecked()),
                "show_keyframes": bool(w.controls.show_keyframes_box.isChecked()),
                "show_trajectory_lines": bool(w.controls.show_lines_box.isChecked()),
                "show_playback_poses": bool(viewer.show_ghosts.isChecked()),
            }
        if "commands" in wanted:
            unavailable = {}
            if not w.undo_stack:
                unavailable["history_undo"] = "nothing to undo"
            if not w.redo_stack:
                unavailable["history_redo"] = "nothing to redo"
            if w.agent_bridge_controller.server is not None:
                unavailable["motion_edits"] = "external control owns the document"
            state["commands"] = {"unavailable": unavailable}
        return state

    def digest(self):
        return self.app_state(DIGEST_SECTIONS)

    def capture_viewport(self, max_size):
        from PySide6.QtCore import Qt
        from gui.ai_frame_capture import _encode_png

        w = self.window
        if w.viewer_tabs.currentWidget() is not w.viewer_3d_stack:
            return None
        image = self._viewer().canvas.grabFramebuffer()
        if image.isNull():
            return None
        if max(image.width(), image.height()) > max_size:
            image = image.scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        return _encode_png(image)

    # -------------------------------------------------------------- navigation
    def go_to_time(self, time_s=None, keyframe=None):
        w, viewer = self.window, self._viewer()
        current = float(viewer.get_current_time())
        if keyframe is not None:
            times = self._keyframe_times()
            if not times:
                raise ValueError("there are no Keyframes yet")
            if keyframe == "first":
                time_s = times[0]
            elif keyframe == "last":
                time_s = times[-1]
            elif keyframe == "next":
                later = [time for time in times if time > current + 1e-6]
                if not later:
                    raise ValueError("no Keyframe after the current time")
                time_s = later[0]
            else:
                earlier = [time for time in times if time < current - 1e-6]
                if not earlier:
                    raise ValueError("no Keyframe before the current time")
                time_s = earlier[-1]
        duration = float(w.document.timeline_duration)
        if time_s > duration + 1e-9:
            raise ValueError(f"time is past the motion duration ({duration:.2f} s)")
        if viewer.play_timer.isActive():
            viewer.pause_playback(commit_time=False)
        slider = w.controls.time_slider
        slider.blockSignals(True)
        try:
            slider.set_value(time_s)
        finally:
            slider.blockSignals(False)
        w.on_time_changed(float(time_s))
        return {"time_s": round(float(viewer.get_current_time()), 4)}

    def playback(self, action, speed=None):
        viewer = self._viewer()
        if speed is not None:
            viewer.playback_speed.setValue(float(speed))
        playing = viewer.play_timer.isActive()
        if action == "play" and not playing:
            viewer.start_playback()
        elif action in ("pause", "stop") and playing:
            viewer.pause_playback(commit_time=True)
        elif action == "toggle":
            viewer.toggle_playback()
        if action == "stop":
            self.window.on_time_changed(0.0)
        return {"playing": bool(viewer.play_timer.isActive()),
                "speed": round(float(viewer.playback_speed.value()), 2)}

    # ------------------------------------------------------------------- view
    def set_view(self, active_view=None, left_sidebar=None, right_sidebar=None):
        w = self.window
        if active_view is not None:
            tabs = w.viewer_tabs
            titles = [tabs.tabText(index) for index in range(tabs.count())]
            match = [index for index, title in enumerate(titles)
                     if title.replace("&", "").lower() == active_view.lower()]
            if not match:
                raise ValueError(f"unknown view; choose one of {titles}")
            tabs.setCurrentIndex(match[0])
        if left_sidebar is not None:
            w.set_left_sidebar_visible(bool(left_sidebar))
        if right_sidebar is not None:
            w.set_right_sidebar_visible(bool(right_sidebar))
        return self.app_state(("view",))["view"]

    def set_display(self, **values):
        w, viewer = self.window, self._viewer()
        widgets = {"model_colors": viewer.model_colors_box, "show_keyframes": w.controls.show_keyframes_box,
                   "show_trajectory_lines": w.controls.show_lines_box, "show_playback_poses": viewer.show_ghosts}
        for key, value in values.items():
            if value is not None:
                widgets[key].setChecked(bool(value))
        w.sync_display_actions()
        return self.app_state(("display",))["display"]

    def set_camera(self, preset=None, yaw=None, pitch=None, distance=None):
        canvas = self._viewer().canvas
        values = dict(CAMERA_PRESETS[preset]) if preset else {}
        for key, value in (("yaw", yaw), ("pitch", pitch), ("distance", distance)):
            if value is not None:
                values[key] = float(value)
        if not values:
            raise ValueError("pass a preset or at least one of yaw, pitch, distance")
        if "yaw" in values:
            canvas.camera_yaw = values["yaw"]
        if "pitch" in values:
            canvas.camera_pitch = values["pitch"]
        if "distance" in values:
            canvas.camera_distance = values["distance"]
        if preset == "default":
            canvas.camera_center = np.asarray([0.0, 0.0, 0.75], dtype=float)
        canvas.update()
        return self.app_state(("camera",))["camera"]

    def set_gizmo(self, visible=None, mode=None):
        w = self.window
        action = w.gizmo_visibility_action
        if visible is not None:
            if not action.isEnabled() and visible:
                raise ValueError("the gizmo is unavailable in this view or while motion controls are locked")
            action.setChecked(bool(visible))
            w.sync_transform_gizmo_state()
        if mode is not None:
            w.set_gizmo_mode(mode)
        return {"visible": bool(action.isChecked()), "mode": mode}

    def select_target(self, frame):
        box = self.window.controls.frame_box
        names = [box.itemText(index) for index in range(box.count())]
        if frame not in names:
            raise ValueError(f"unknown frame; choose one of {names[:24]}")
        box.setCurrentText(frame)
        return {"frame": untrusted(box.currentText())}

    # ---------------------------------------------------------------- history
    def history(self, direction):
        w = self.window
        stack = w.undo_stack if direction == "undo" else w.redo_stack
        if not stack:
            raise ValueError(f"nothing to {direction}")
        if w.agent_bridge_controller.has_candidate:
            raise ValueError("Accept or Discard the Orange preview first")
        description = stack[-1].description
        result = w.undo_last_action() if direction == "undo" else w.redo_last_action()
        if result is False:
            raise ValueError(f"{direction} is blocked right now")
        return {direction: untrusted(description)}

    def show_help(self, topic="help_center"):
        w = self.window
        {"help_center": w.show_help_center, "keyboard_shortcuts": w.show_keyboard_shortcuts,
         "tutorial": w.start_first_motion_tutorial}[topic]()
        return {"opened": topic}

    # --------------------------------------------------------------- settings
    def preview_path(self):
        viewer = self._viewer()
        if not viewer.preview_active:
            raise ValueError("there is no manual preview pose to validate")
        viewer.plan_preview()
        return {"status": viewer.status_label.text()[:200]}

    def set_export_settings(self, sample_interval_s=None, motion_speed=None):
        viewer = self._viewer()
        if sample_interval_s is not None:
            viewer.set_export_dt(float(sample_interval_s))
        if motion_speed is not None:
            viewer.set_dsms_motion_speed(float(motion_speed))
        return {"sample_interval_s": float(viewer.export_dt()), "motion_speed": float(viewer.dsms_motion_speed())}

    def set_generation_settings(self, smoothing):
        slider = self.window.controls.corner_smoothing_slider
        slider.set_value(float(smoothing))
        return {"smoothing": float(self.window.controls.corner_smoothing())}

    def set_ik_settings(self, **values):
        viewer = self._viewer()
        widgets = {"damping": viewer.ik_damping, "max_iterations": viewer.ik_max_iterations,
                   "step_size": viewer.ik_step_size, "max_step": viewer.ik_max_step,
                   "position_tolerance": viewer.ik_position_tolerance,
                   "orientation_tolerance": viewer.ik_orientation_tolerance}
        for key, value in values.items():
            if value is not None:
                widgets[key].setValue(value)
        return {key: widget.value() for key, widget in widgets.items()}

    def simulation_viewer(self, action):
        panel = self.window.viewer_3d_mujoco
        if action == "open":
            panel.open_viewer()
        else:
            panel.close_viewer()
        return {"action": action}

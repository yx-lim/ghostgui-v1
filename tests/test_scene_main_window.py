import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from core.scene import Scene, Transform
from gui.main_window import RobotGuiMainWindow


class SceneMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config_dir = tempfile.TemporaryDirectory()
        self.config_patch = patch.dict(
            os.environ,
            {
                "GHOSTGUI_CONFIG_DIR": str(
                    Path(self.config_dir.name) / "config"
                ),
                "GHOSTGUI_PROJECTS_DIR": str(
                    Path(self.config_dir.name) / "projects"
                ),
            },
        )
        self.config_patch.start()
        self.window = RobotGuiMainWindow()

    def tearDown(self):
        self.window.close()
        self.config_patch.stop()
        self.config_dir.cleanup()

    def test_object_actor_persists_through_undo_save_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            actor = self.window.add_scene_object(
                name="Box",
                shape="box",
                size=[0.3, 0.2, 0.1],
                transform=Transform(position=(0.4, 0.0, 0.1)),
            )
            self.assertEqual(len(self.window.scene.actors.objects()), 1)
            self.assertEqual(
                self.window.scene_tree.currentItem().data(0, Qt.ItemDataRole.UserRole),
                actor.id,
            )

            self.window.set_scene_actor_visibility(actor.id, False)
            self.assertFalse(self.window.scene.actors.require(actor.id).visible)

            self.window.undo_last_action()
            self.assertTrue(self.window.scene.actors.require(actor.id).visible)

            project_root = Path(directory) / "scene_reopen.ghostgui"
            project = self.window.create_project_at(project_root, "scene_reopen")
            saved_scene = Scene.from_dict(project.read_scene_dict())
            self.assertEqual(len(saved_scene.actors.objects()), 1)

            self.window.scene.delete_actor(actor.id)
            self.assertEqual(len(self.window.scene.actors.objects()), 0)

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ):
                self.assertTrue(self.window.open_project_path(project_root))

            self.assertEqual(len(self.window.scene.actors.objects()), 1)
            restored = self.window.scene.actors.objects()[0]
            self.assertEqual(restored.name, "Box")

    def test_extra_robot_selection_does_not_corrupt_editor_robot_tracks(self):
        editor_robot_id = self.window.editor_robot_actor_id
        extra = self.window.add_scene_robot(
            model_key="custom-model",
            model_name="Custom Robot",
        )
        self.window.scene.select_actor(extra.id)

        scene = Scene.from_dict(self.window.capture_project_scene())

        self.assertEqual(
            scene.actors.require(extra.id).model_reference["model_key"],
            "custom-model",
        )
        self.assertEqual(
            scene.actors.require(editor_robot_id).model_reference["model_key"],
            self.window.model_key,
        )
        self.assertEqual(
            scene.metadata["editor_robot_actor_id"],
            editor_robot_id,
        )

    def test_add_robot_button_adds_actor_from_model_registry(self):
        label = next(
            item_label
            for item_label, key, _display_name
            in self.window.available_scene_robot_choices()
            if key == self.window.model_key
        )
        initial_count = len(self.window.scene.actors.robots())

        with patch(
            "gui.main_window.QInputDialog.getItem",
            return_value=(label, True),
        ):
            self.window.on_add_scene_robot_clicked()

        self.assertEqual(
            len(self.window.scene.actors.robots()),
            initial_count + 1,
        )
        self.window.set_project_dirty(False)

    def test_added_robot_is_offset_and_passed_to_3d_scene_renderer(self):
        extra = self.window.add_scene_robot(
            model_key=self.window.model_key,
            model_name=self.window.current_model_display_name(),
        )
        self.assertNotEqual(extra.world_transform.position, (0.0, 0.0, 0.0))

        sentinel_adapter = object()
        captured = {}
        self.window.scene_extra_robot_adapters = lambda: {
            extra.id: sentinel_adapter
        }

        def capture_update_scene(*_args, **kwargs):
            captured.update(kwargs)

        self.window.viewer_3d.update_scene = capture_update_scene
        self.window.refresh_display(apply_stickman_frame=False)

        self.assertIs(
            captured["scene_robot_adapters"][extra.id],
            sentinel_adapter,
        )
        self.window.set_project_dirty(False)

    def test_import_mesh_button_creates_project_local_mesh_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "external" / "fixture.obj"
            source.parent.mkdir()
            source.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            project = self.window.create_project_at(
                root / "mesh_project.ghostgui",
                "mesh_project",
            )

            with patch(
                "gui.main_window.QFileDialog.getOpenFileName",
                return_value=(str(source), ""),
            ), patch("gui.main_window.ObjectMeshImportDialog") as dialog_class, patch(
                "gui.main_window.QMessageBox.warning"
            ) as warning:
                dialog = dialog_class.return_value
                dialog.exec.return_value = QDialog.DialogCode.Accepted
                dialog.object_name.return_value = "Fixture"
                dialog.uniform_scale.return_value = [2.0, 2.0, 2.0]

                self.window.on_import_scene_object_clicked()

            self.assertFalse(warning.called, warning.call_args)

            objects = self.window.scene.actors.objects()
            mesh_actor = next(
                actor for actor in objects
                if actor.model_reference.get("type") == "mesh"
            )
            copied = project.root_dir / mesh_actor.model_reference["asset_path"]

            self.assertEqual(mesh_actor.name, "Fixture")
            self.assertEqual(mesh_actor.model_reference["scale"], [2.0, 2.0, 2.0])
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.window.set_project_dirty(False)

    def test_open_project_refreshes_meshes_against_new_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_project = self.window.create_project_at(
                root / "old_project.ghostgui",
                "old_project",
            )
            self.window.set_project_dirty(False)

            source = root / "external" / "fixture.obj"
            source.parent.mkdir()
            source.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            new_project = self.window.create_project_at(
                root / "mesh_project.ghostgui",
                "mesh_project",
            )
            imported = new_project.import_object_mesh(source)
            self.window.add_scene_mesh_object(
                name="Fixture",
                model_reference=imported.model_reference(),
            )
            self.window.save_current_project(show_status=False, capture_snapshot=False)

            self.window.current_project = old_project
            self.window.set_project_dirty(False)
            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ):
                self.assertTrue(self.window.open_project_path(new_project.root_dir))

            self.assertEqual(
                self.window.viewer_3d.canvas.scene_asset_root,
                new_project.root_dir,
            )
            self.window.set_project_dirty(False)

    def test_scene_object_tracks_follow_sidebar_time_changes(self):
        actor = self.window.add_scene_object(name="Box")
        self.window.scene.set_object_transform_keyframe(
            actor.id,
            0.0,
            Transform(position=(0.0, 0.0, 0.0)),
        )
        self.window.scene.set_object_transform_keyframe(
            actor.id,
            1.0,
            Transform(position=(1.0, 0.0, 0.0)),
        )

        self.window.on_time_changed(0.5)

        self.assertAlmostEqual(self.window.scene.timeline.current_time, 0.5)
        transform = self.window.scene.tracks.object_transform_at(actor, 0.5)
        self.assertAlmostEqual(transform.position[0], 0.5)
        self.window.set_project_dirty(False)

    def test_selected_object_is_draggable_scene_edit_actor(self):
        actor = self.window.add_scene_object(name="Draggable")
        captured = {}

        def capture_update_scene(*_args, **kwargs):
            captured.update(kwargs)

        self.window.viewer_3d.update_scene = capture_update_scene
        self.window.refresh_display(apply_stickman_frame=False)

        self.assertEqual(captured["scene_edit_actor_id"], actor.id)

        self.window._refresh_history_baseline()
        self.window.on_scene_actor_transform_dragged(
            actor.id,
            (0.25, 0.10, 0.35),
            (1.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            self.window.scene.actors.require(actor.id).world_transform.position,
            (0.25, 0.10, 0.35),
        )

        self.window.on_scene_actor_transform_drag_finished(
            actor.id,
            (0.40, 0.10, 0.35),
            (1.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(
            self.window.scene.actors.require(actor.id).world_transform.position,
            (0.40, 0.10, 0.35),
        )

        self.window.undo_last_action()
        self.assertNotEqual(
            self.window.scene.actors.require(actor.id).world_transform.position,
            (0.40, 0.10, 0.35),
        )
        self.window.set_project_dirty(False)


if __name__ == "__main__":
    unittest.main()

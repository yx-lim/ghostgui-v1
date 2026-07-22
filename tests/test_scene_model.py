import tempfile
import unittest
import uuid
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from application.project_manager import GhostGUIProject
from core.scene import Scene, SceneRuntime, Transform, load_mesh_geometry
from core.trajectory import TargetFrame, Trajectory


class SceneModelTests(unittest.TestCase):
    def make_trajectory(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(time=0.0, frame_name="left_hand", x=0.1))
        trajectory.add_frame(TargetFrame(time=0.5, frame_name="left_hand", x=0.4))
        return trajectory

    def test_scene_roundtrip_keeps_actor_ids_and_robot_tracks(self):
        trajectory = self.make_trajectory()
        scene = Scene.single_robot("g1", model_name="Unitree G1", trajectory=trajectory)
        robot = scene.active_robot()

        uuid.UUID(robot.id)
        robot.name = "Display label can change"

        restored = Scene.from_dict(scene.to_dict())
        restored_robot = restored.active_robot()
        self.assertEqual(restored_robot.id, robot.id)
        self.assertEqual(restored_robot.name, "Display label can change")

        restored_trajectory = restored.active_robot_trajectory()
        self.assertEqual(len(restored_trajectory.frames), 2)
        self.assertEqual(restored_trajectory.frames[0].frame_name, "left_hand")
        self.assertAlmostEqual(restored_trajectory.frames[1].x, 0.4)

    def test_object_tracks_visibility_locking_duplicate_and_constraints(self):
        scene = Scene.single_robot("g1")
        robot = scene.active_robot()
        box = scene.add_object(
            name="Box",
            size=[0.3, 0.2, 0.1],
            transform=Transform(position=(0.0, 0.0, 0.1)),
        )
        scene.set_object_transform_keyframe(
            box.id,
            0.0,
            Transform(position=(0.0, 0.0, 0.1)),
        )
        scene.set_object_transform_keyframe(
            box.id,
            1.0,
            Transform(position=(1.0, 0.0, 0.1)),
        )

        midpoint = scene.tracks.object_transform_at(box, 0.5)
        self.assertAlmostEqual(midpoint.position[0], 0.5)

        duplicate = scene.duplicate_actor(box.id)
        self.assertNotEqual(duplicate.id, box.id)
        self.assertEqual(len(scene.tracks.object_transforms[duplicate.id]), 2)

        scene.set_actor_visibility(box.id, False)
        self.assertNotIn(box.id, [actor.id for actor in scene.visible_object_actors()])

        scene.set_actor_locked(box.id, True)
        with self.assertRaisesRegex(ValueError, "locked"):
            scene.set_object_transform_keyframe(box.id, 2.0, Transform.identity())

        constraint = scene.attach(robot.id, "left_hand", duplicate.id, "world")
        self.assertEqual(
            scene.constraints.constraints[constraint.id].source.actor_id,
            robot.id,
        )
        scene.delete_actor(duplicate.id)
        self.assertEqual(scene.constraints.for_actor(robot.id), [])

    def test_runtime_plan_namespaces_each_actor(self):
        scene = Scene.single_robot("g1")
        robot = scene.active_robot()
        box = scene.add_object(name="Box")
        plan = SceneRuntime(scene).build_plan()

        self.assertEqual(len(plan.robot_actors), 1)
        self.assertEqual(len(plan.object_actors), 1)
        self.assertTrue(plan.namespaced_name(robot.id, "pelvis").endswith("/pelvis"))
        self.assertNotEqual(plan.actor_namespace(robot.id), plan.actor_namespace(box.id))

    def test_mesh_object_roundtrip_keeps_project_relative_reference(self):
        scene = Scene.single_robot("g1")
        mesh = scene.add_mesh_object(
            name="Fixture",
            asset_path="assets/objects/cube.obj",
            mesh_format="obj",
            actor_id="object-fixture",
            scale=[1.5, 1.0, 0.5],
            transform=Transform(position=(0.2, 0.3, 0.4)),
            locked=True,
        )
        scene.set_actor_visibility(mesh.id, False)

        restored = Scene.from_dict(scene.to_dict())
        actor = restored.actors.require("object-fixture")

        self.assertEqual(actor.kind, "object")
        self.assertEqual(actor.model_reference["type"], "mesh")
        self.assertEqual(
            actor.model_reference["asset_path"],
            "assets/objects/cube.obj",
        )
        self.assertEqual(actor.model_reference["scale"], [1.5, 1.0, 0.5])
        self.assertEqual(actor.world_transform.position, (0.2, 0.3, 0.4))
        self.assertFalse(actor.visible)
        self.assertTrue(actor.locked)


class SceneRuntimeComposerTests(unittest.TestCase):
    def write_robot_mjcf(self, path):
        path.write_text(
            """
<mujoco model="fixture_robot">
  <worldbody>
    <light name="source_light" pos="0 0 2"/>
    <geom name="source_floor" type="plane" size="1 1 .01"/>
    <body name="base">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="shell" type="box" size=".1 .1 .1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="slide"/>
  </actuator>
</mujoco>
""".strip(),
            encoding="utf-8",
        )

    def write_tetra_obj(self, path):
        path.write_text(
            """
v 0 0 0
v 0.1 0 0
v 0 0.1 0
v 0 0 0.1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
""".strip(),
            encoding="utf-8",
        )

    def test_composed_mjcf_namespaces_robot_primitive_mesh_and_weld(self):
        with tempfile.TemporaryDirectory() as directory:
            root_dir = Path(directory)
            robot_path = root_dir / "robot.xml"
            mesh_path = root_dir / "assets" / "objects" / "tetra.obj"
            mesh_path.parent.mkdir(parents=True)
            self.write_robot_mjcf(robot_path)
            self.write_tetra_obj(mesh_path)

            scene = Scene.single_robot("fixture", actor_id="robot-alpha")
            robot = scene.active_robot()
            box = scene.add_object(name="Box", actor_id="object-box")
            mesh = scene.add_mesh_object(
                name="Fixture",
                asset_path="assets/objects/tetra.obj",
                actor_id="object-mesh",
                scale=[2.0, 1.0, 1.0],
            )
            scene.weld(robot.id, "base", box.id, "world")

            runtime = SceneRuntime(scene)
            composition = runtime.compose_mjcf(
                root_dir,
                robot_model_resolver=lambda _actor: robot_path,
            )
            xml_root = ET.fromstring(composition.xml)
            robot_ns = runtime.namespace_for_actor(robot)
            box_ns = runtime.namespace_for_actor(box)
            mesh_ns = runtime.namespace_for_actor(mesh)

            self.assertIsNotNone(
                xml_root.find(f".//body[@name='{robot_ns}/base']")
            )
            self.assertIsNone(xml_root.find(".//geom[@name='source_floor']"))
            self.assertIsNotNone(xml_root.find(f".//body[@name='{box_ns}']"))
            self.assertIsNotNone(
                xml_root.find(f".//freejoint[@name='{box_ns}/freejoint']")
            )
            self.assertIsNotNone(xml_root.find(f".//body[@name='{mesh_ns}']"))
            self.assertIsNotNone(xml_root.find(f".//mesh[@name='{mesh_ns}/mesh']"))
            self.assertIsNotNone(xml_root.find(f".//geom[@mesh='{mesh_ns}/mesh']"))
            self.assertIsNotNone(
                xml_root.find(
                    f".//weld[@site1='{robot_ns}/base/base'][@site2='{box_ns}/world']"
                )
            )
            self.assertIn("assets/objects/tetra.obj", composition.assets)

            model = runtime.compile_model(
                root_dir,
                robot_model_resolver=lambda _actor: robot_path,
            )
            self.assertGreaterEqual(model.nbody, 4)
            self.assertGreaterEqual(model.njnt, 3)

    def test_object_transform_tracks_drive_composed_initial_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root_dir = Path(directory)
            robot_path = root_dir / "robot.xml"
            self.write_robot_mjcf(robot_path)

            scene = Scene.single_robot("fixture", actor_id="robot-alpha")
            box = scene.add_object(name="Box", actor_id="object-box")
            scene.set_object_transform_keyframe(
                box.id,
                0.0,
                Transform(position=(0.0, 0.0, 0.0)),
            )
            scene.set_object_transform_keyframe(
                box.id,
                1.0,
                Transform(position=(1.0, 0.0, 0.0)),
            )

            xml_text = SceneRuntime(scene).build_mjcf(
                root_dir,
                robot_model_resolver=lambda _actor: robot_path,
                time=0.5,
            )
            xml_root = ET.fromstring(xml_text)
            box_ns = SceneRuntime.namespace_for_actor(box)
            body = xml_root.find(f".//body[@name='{box_ns}']")
            self.assertEqual(body.get("pos"), "0.5 0 0")

    def test_bundled_floating_base_models_compile_with_disjoint_namespaces(self):
        from core.models import MuJoCoRobotAdapter, ROBOT_MODELS

        adapter_cache = {}

        def resolve(actor):
            model_key = actor.model_reference["model_key"]
            if model_key not in adapter_cache:
                adapter = MuJoCoRobotAdapter(ROBOT_MODELS[model_key])
                adapter_cache[model_key] = {
                    "runtime_model_path": adapter.runtime_model_path,
                    "logical_frame_bindings": dict(adapter.logical_frame_bindings),
                }
            return adapter_cache[model_key]

        scene = Scene.single_robot("g1", actor_id="g1_actor")
        go2 = scene.add_robot("go2", actor_id="go2_actor")
        box = scene.add_object(actor_id="box_actor")

        model = SceneRuntime(scene).compile_model(robot_model_resolver=resolve)

        self.assertGreater(model.nbody, 40)
        self.assertNotEqual(
            SceneRuntime.namespace_for_actor(scene.active_robot()),
            SceneRuntime.namespace_for_actor(go2),
        )
        self.assertIn(box.id, SceneRuntime(scene).build_plan().namespaces)

    def test_duplicate_bundled_robot_defaults_do_not_collide(self):
        from core.models import MuJoCoRobotAdapter, ROBOT_MODELS

        adapter = MuJoCoRobotAdapter(ROBOT_MODELS["go2"])
        resolver_value = {
            "runtime_model_path": adapter.runtime_model_path,
            "logical_frame_bindings": dict(adapter.logical_frame_bindings),
        }
        scene = Scene.single_robot("go2", actor_id="go2_a")
        scene.add_robot("go2", actor_id="go2_b")

        model = SceneRuntime(scene).compile_model(
            robot_model_resolver=lambda _actor: resolver_value,
        )

        self.assertGreater(model.nbody, 20)

    def test_logical_frame_weld_uses_namespaced_robot_site(self):
        from core.models import MuJoCoRobotAdapter, ROBOT_MODELS

        adapter = MuJoCoRobotAdapter(ROBOT_MODELS["g1"])
        resolver_value = {
            "runtime_model_path": adapter.runtime_model_path,
            "logical_frame_bindings": dict(adapter.logical_frame_bindings),
        }
        scene = Scene.single_robot("g1", actor_id="g1_actor")
        robot = scene.active_robot()
        box = scene.add_object(actor_id="box_actor")
        scene.weld(robot.id, "left_hand", box.id, "world")

        composition = SceneRuntime(scene).compose_mjcf(
            robot_model_resolver=lambda _actor: resolver_value,
        )
        xml_root = ET.fromstring(composition.xml)
        robot_ns = SceneRuntime.namespace_for_actor(robot)

        self.assertEqual(
            composition.frame_sites[(robot.id, "left_hand")],
            f"{robot_ns}/robot/left_palm",
        )
        self.assertIsNotNone(
            xml_root.find(f".//weld[@site1='{robot_ns}/robot/left_palm']")
        )
        model = SceneRuntime(scene).compile_model(
            robot_model_resolver=lambda _actor: resolver_value,
        )
        self.assertEqual(model.neq, 1)


class ProjectSchemaV2Tests(unittest.TestCase):
    def test_create_writes_scene_file_and_v2_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            project = GhostGUIProject.create(
                Path(directory) / "scene_project.ghostgui",
                "scene_project",
                "g1",
                "Unitree G1",
            )

            self.assertEqual(project.metadata["schema_version"], 2)
            self.assertTrue(project.paths.scene.exists())
            scene = Scene.from_dict(project.read_scene_dict())
            self.assertEqual(scene.active_robot().model_reference["model_key"], "g1")

    def test_project_imports_object_mesh_to_relative_asset_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "outside" / "cube.obj"
            source.parent.mkdir()
            source.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            project = GhostGUIProject.create(
                root / "asset_project.ghostgui",
                "asset_project",
                "g1",
                "Unitree G1",
            )

            imported = project.import_object_mesh(source)
            copied = project.root_dir / imported.asset_path

            self.assertFalse(Path(imported.asset_path).is_absolute())
            self.assertTrue(copied.is_file())
            self.assertTrue(copied.is_relative_to(project.root_dir))
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.assertEqual(load_mesh_geometry(copied).faces, ((0, 1, 2),))

    def test_v1_project_metadata_migrates_without_rewriting_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy.ghostgui"
            root.mkdir()
            (root / "ghostgui_project.json").write_text(
                """
{
  "schema_version": 1,
  "project_name": "legacy",
  "created_at": "2025-01-01T00:00:00Z",
  "modified_at": "2025-01-01T00:00:00Z",
  "application": {
    "name": "GhostGUI",
    "project_format": "ghostgui.project.v1"
  },
  "robot": {
    "model_key": "g1",
    "model_name": "Unitree G1"
  },
  "files": {
    "target_trajectory": "data/target_frames.json",
    "qpos_timeline": "data/qpos_timeline.npz",
    "workspace": "workspace/workspace.json",
    "last_snapshot": "snapshots/last_workspace.png",
    "session_log": "metadata/session_log.jsonl"
  }
}
""".strip(),
                encoding="utf-8",
            )

            project = GhostGUIProject.open(root)

            self.assertEqual(project.metadata["schema_version"], 2)
            self.assertEqual(
                project.metadata["application"]["project_format"],
                "ghostgui.project.v2",
            )
            self.assertEqual(project.model_key, "g1")
            self.assertEqual(project.paths.scene, root / "data" / "scene.json")
            self.assertEqual(
                project.autosave_paths.scene,
                root / "autosave" / "scene.autosave.json",
            )
            scene = Scene.from_dict(project.read_scene_dict())
            self.assertEqual(scene.active_robot().model_reference["model_key"], "g1")
            self.assertFalse(project.paths.scene.exists())

    def test_v1_autosave_without_scene_still_counts_as_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            project = GhostGUIProject.create(
                Path(directory) / "autosave_legacy.ghostgui",
                "autosave_legacy",
                "g1",
                "Unitree G1",
            )
            for path in (
                project.autosave_paths.manifest,
                project.autosave_paths.target_trajectory,
                project.autosave_paths.qpos_timeline,
                project.autosave_paths.workspace,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            project.autosave_paths.scene.unlink(missing_ok=True)

            old_time = 1_700_000_000
            new_time = old_time + 60
            for path in (
                project.project_file,
                project.paths.target_trajectory,
                project.paths.qpos_timeline,
                project.paths.workspace,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("{}", encoding="utf-8")
                os.utime(path, (old_time, old_time))
            for path in (
                project.autosave_paths.manifest,
                project.autosave_paths.target_trajectory,
                project.autosave_paths.qpos_timeline,
                project.autosave_paths.workspace,
            ):
                os.utime(path, (new_time, new_time))

            self.assertTrue(project.autosave_exists())
            self.assertTrue(project.is_autosave_newer())


if __name__ == "__main__":
    unittest.main()

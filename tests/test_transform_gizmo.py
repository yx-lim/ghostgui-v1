import math
import unittest

import numpy as np

from gui.transform_gizmo import (
    GizmoInteractionState,
    TransformGizmo,
)


def project_isometric(x, y, z):
    return 200.0 + 100.0 * x + 30.0 * y, 200.0 - 100.0 * z + 20.0 * y


def unused_ray(sx, sy):
    return np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0])


class TransformGizmoTests(unittest.TestCase):
    def test_center_sphere_picks_and_drags_freely_in_camera_plane(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.5))

        def project(x, y, z):
            return 200.0 + 100.0 * x, 200.0 + 100.0 * y

        def ray(sx, sy):
            return (
                np.array([(sx - 200.0) / 100.0, (sy - 200.0) / 100.0, 5.0]),
                np.array([0.0, 0.0, -1.0]),
            )

        state, handle = gizmo.pick(200.0, 200.0, project)
        self.assertEqual(state, GizmoInteractionState.HOVER_TRANSLATE_FREE)
        self.assertEqual(handle, "free")
        original_quaternion = gizmo.quaternion.copy()
        self.assertTrue(gizmo.begin_drag(200.0, 200.0, project, ray))
        position, quaternion = gizmo.drag(220.0, 210.0, project, ray)
        np.testing.assert_allclose(position, [0.2, 0.1, 0.5], atol=1e-8)
        np.testing.assert_allclose(quaternion, original_quaternion)

    def test_x_arrow_pick_and_miss(self):
        gizmo = TransformGizmo((0.0, 0.0, 0.0))
        endpoint = project_isometric(gizmo.arrow_length, 0.0, 0.0)
        state, axis = gizmo.pick(*endpoint, project_isometric)
        self.assertEqual(state, GizmoInteractionState.HOVER_TRANSLATE_X)
        self.assertEqual(axis, "x")
        self.assertEqual(
            gizmo.pick(20.0, 20.0, project_isometric),
            (GizmoInteractionState.NONE, None),
        )

    def test_each_translation_changes_only_selected_axis(self):
        cases = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        for axis, vector in cases.items():
            with self.subTest(axis=axis):
                gizmo = TransformGizmo((0.0, 0.0, 0.5))
                endpoint = project_isometric(*(gizmo.position + gizmo.arrow_length * vector))
                self.assertTrue(gizmo.begin_drag(*endpoint, project_isometric, unused_ray))
                origin_screen = np.array(project_isometric(*gizmo.position))
                endpoint_screen = np.array(endpoint)
                drag_screen = endpoint_screen + 20.0 * (
                    endpoint_screen - origin_screen
                ) / np.linalg.norm(endpoint_screen - origin_screen)
                position, _ = gizmo.drag(
                    *drag_screen, project_isometric, unused_ray
                )
                changed = np.abs(position - np.array([0.0, 0.0, 0.5])) > 1e-8
                expected = np.array([axis == "x", axis == "y", axis == "z"])
                np.testing.assert_array_equal(changed, expected)

    def test_rotation_ring_constrains_quaternion_axis(self):
        camera_setups = {
            "x": (
                lambda x, y, z: (200 + 100 * y, 200 - 100 * z),
                lambda sx, sy: (
                    np.array([1.0, (sx - 200) / 100, -(sy - 200) / 100]),
                    np.array([-1.0, 0.0, 0.0]),
                ),
                (214.142, 185.858), (185.858, 185.858), 1,
            ),
            "y": (
                lambda x, y, z: (200 + 100 * x, 200 - 100 * z),
                lambda sx, sy: (
                    np.array([(sx - 200) / 100, 1.0, -(sy - 200) / 100]),
                    np.array([0.0, -1.0, 0.0]),
                ),
                (214.142, 185.858), (185.858, 185.858), 2,
            ),
            "z": (
                lambda x, y, z: (200 + 100 * x, 200 + 100 * y),
                lambda sx, sy: (
                    np.array([(sx - 200) / 100, (sy - 200) / 100, 1.0]),
                    np.array([0.0, 0.0, -1.0]),
                ),
                (214.142, 214.142), (185.858, 214.142), 3,
            ),
        }
        for axis, (project, ray, start, end, quaternion_index) in camera_setups.items():
            with self.subTest(axis=axis):
                gizmo = TransformGizmo((0.0, 0.0, 0.0))
                self.assertTrue(gizmo.begin_drag(*start, project, ray))
                self.assertIn("ROTATE", gizmo.state.name)
                _, quaternion = gizmo.drag(*end, project, ray)
                other_vector_parts = np.delete(quaternion[1:], quaternion_index - 1)
                np.testing.assert_allclose(other_vector_parts, 0.0, atol=1e-7)
                self.assertAlmostEqual(abs(quaternion[quaternion_index]), math.sqrt(0.5), places=5)


if __name__ == "__main__":
    unittest.main()

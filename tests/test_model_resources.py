import tempfile
import unittest
from pathlib import Path

from application.model_resources import ModelResourcePool


class FakeAdapter:
    def __init__(self, model_path):
        self.model_path = Path(model_path)


class ModelResourcePoolTests(unittest.TestCase):
    def test_canonical_path_reuses_first_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text("<mujoco/>", encoding="utf-8")
            first = FakeAdapter(path)
            duplicate = FakeAdapter(path.parent / "." / path.name)
            pool = ModelResourcePool()

            self.assertIs(pool.register(first), first)
            self.assertIs(pool.register(duplicate), first)
            self.assertIs(pool.get(model_path=path), first)
            self.assertEqual(len(pool), 1)

"""Project-local asset importing for scene object actors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

from core.scene.mesh import SUPPORTED_OBJECT_MESH_EXTENSIONS, load_mesh_geometry


OBJECT_ASSET_DIR = Path("assets") / "objects"


@dataclass(frozen=True)
class ImportedObjectMesh:
    asset_path: str
    mesh_format: str
    source_name: str
    vertex_count: int
    face_count: int

    def model_reference(self, scale=None, rgba=None):
        scale = scale or (1.0, 1.0, 1.0)
        rgba = rgba or (0.20, 0.58, 0.88, 1.0)
        return {
            "type": "mesh",
            "asset_path": self.asset_path,
            "mesh_format": self.mesh_format,
            "source_name": self.source_name,
            "scale": [float(value) for value in scale],
            "rgba": [float(value) for value in rgba],
        }


def import_object_mesh(source_path, project_root):
    source_path = Path(source_path).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_OBJECT_MESH_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_OBJECT_MESH_EXTENSIONS))
        raise ValueError(f"Unsupported mesh extension; choose one of {allowed}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    geometry = load_mesh_geometry(source_path)
    project_root = Path(project_root).expanduser().resolve()
    asset_dir = project_root / OBJECT_ASSET_DIR
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_asset_path(asset_dir, _slug(source_path.stem), suffix)
    shutil.copy2(source_path, target)
    return ImportedObjectMesh(
        asset_path=target.relative_to(project_root).as_posix(),
        mesh_format=suffix.lstrip("."),
        source_name=source_path.name,
        vertex_count=len(geometry.vertices),
        face_count=len(geometry.faces),
    )


def resolve_project_asset(project_root, asset_path):
    asset_path = Path(asset_path)
    if asset_path.is_absolute():
        return asset_path
    return Path(project_root).expanduser().resolve() / asset_path


def _slug(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return text or "object"


def _unique_asset_path(directory, stem, suffix):
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

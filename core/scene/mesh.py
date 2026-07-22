"""Small mesh loading helpers for scene object actors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


SUPPORTED_OBJECT_MESH_EXTENSIONS = {".obj", ".stl"}


@dataclass(frozen=True)
class MeshGeometry:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]

    def triangles(self):
        return tuple(
            tuple(self.vertices[index] for index in face)
            for face in self.faces
        )


def load_mesh_geometry(path):
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj_geometry(path)
    if suffix == ".stl":
        return load_stl_geometry(path)
    allowed = ", ".join(sorted(SUPPORTED_OBJECT_MESH_EXTENSIONS))
    raise ValueError(f"Unsupported mesh extension; choose one of {allowed}")


def load_obj_geometry(path):
    path = Path(path).expanduser()
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(_float3(parts[1:4], f"OBJ vertex in {path}"))
            elif parts[0] == "f" and len(parts) >= 4:
                face = [
                    _obj_vertex_index(token, len(vertices), path)
                    for token in parts[1:]
                ]
                for index in range(1, len(face) - 1):
                    faces.append((face[0], face[index], face[index + 1]))
    return _validated_geometry(vertices, faces, path)


def load_stl_geometry(path):
    path = Path(path).expanduser()
    payload = path.read_bytes()
    if payload.lstrip().lower().startswith(b"solid"):
        try:
            return _load_ascii_stl(payload, path)
        except ValueError:
            pass
    return _load_binary_stl(payload, path)


def _load_ascii_stl(payload, path):
    vertices = []
    faces = []
    triangle = []
    text = payload.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            triangle.append(_float3(parts[1:4], f"STL vertex in {path}"))
            if len(triangle) == 3:
                base = len(vertices)
                vertices.extend(triangle)
                faces.append((base, base + 1, base + 2))
                triangle = []
    return _validated_geometry(vertices, faces, path)


def _load_binary_stl(payload, path):
    if len(payload) < 84:
        raise ValueError(f"Mesh file is empty or truncated: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_length = 84 + triangle_count * 50
    if len(payload) < expected_length:
        raise ValueError(f"Binary STL is truncated: {path}")
    vertices = []
    faces = []
    offset = 84
    for _index in range(triangle_count):
        values = struct.unpack_from("<12fH", payload, offset)
        offset += 50
        base = len(vertices)
        vertices.extend(
            (
                (float(values[3]), float(values[4]), float(values[5])),
                (float(values[6]), float(values[7]), float(values[8])),
                (float(values[9]), float(values[10]), float(values[11])),
            )
        )
        faces.append((base, base + 1, base + 2))
    return _validated_geometry(vertices, faces, path)


def _float3(values, context):
    try:
        return tuple(float(value) for value in values[:3])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {context}") from exc


def _obj_vertex_index(token, vertex_count, path):
    index_text = str(token).split("/", 1)[0]
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Invalid OBJ face index in {path}: {token}") from exc
    if index == 0:
        raise ValueError(f"OBJ face index cannot be zero in {path}")
    if index < 0:
        index = vertex_count + index + 1
    if index < 1 or index > vertex_count:
        raise ValueError(f"OBJ face index out of range in {path}: {token}")
    return index - 1


def _validated_geometry(vertices, faces, path):
    if not vertices or not faces:
        raise ValueError(f"Mesh has no triangle geometry: {path}")
    return MeshGeometry(
        vertices=tuple(tuple(float(value) for value in vertex) for vertex in vertices),
        faces=tuple(tuple(int(index) for index in face) for face in faces),
    )

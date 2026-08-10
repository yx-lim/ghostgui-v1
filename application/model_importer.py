"""Import user-selected robot models into a persistent model library."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from core.models import (
    PROJECT_ROOT,
    ROBOT_MODELS,
    ResolvedMeshAsset,
    RobotModelInfo,
    resolve_mesh_path,
)
from core.resources import is_source_checkout
from application.paths import ghostgui_user_data_dir


SUPPORTED_MODEL_EXTENSIONS = {".urdf", ".xml"}
MODEL_PROFILE_SCHEMA_VERSION = 2
MODEL_PROFILE_SUFFIX = ".ghostgui.json"
BUILT_IN_MODEL_FILENAMES = {
    "g1_29dof.urdf",
    "g1_29dof.xml",
    "go2.xml",
    "go2_description.urdf",
    "h2.urdf",
    "z1.urdf",
}
UNITREE_MODEL_CODES = {
    "a1": "A1",
    "b1": "B1",
    "b2": "B2",
    "g1": "G1",
    "go1": "Go1",
    "go2": "Go2",
    "h1": "H1",
    "h2": "H2",
    "z1": "Z1",
}
DESCRIPTOR_TOKENS = {"description", "robot", "model", "urdf", "mjcf", "xml"}


def default_model_library_root():
    if is_source_checkout(PROJECT_ROOT):
        return PROJECT_ROOT / "models"
    return ghostgui_user_data_dir() / "models"


def _slug(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._").lower()
    return slug or "robot"


def _display_name(value):
    return _slug(value).replace("-", " ").replace("_", " ").title()


def _name_tokens(value):
    return [
        token
        for token in re.split(r"[^A-Za-z0-9]+", str(value or "").lower())
        if token and token not in DESCRIPTOR_TOKENS
    ]


def _unitree_model_code(*values):
    for value in values:
        tokens = _name_tokens(value)
        for token in tokens:
            if token in UNITREE_MODEL_CODES:
                return token
        if tokens and tokens[-1].isdigit():
            base = "-".join(tokens[:-1])
            if base in UNITREE_MODEL_CODES:
                return base
    return None


def _display_name_from_hints(value, *hints):
    code = _unitree_model_code(value, *hints)
    if code is not None:
        return f"Unitree {UNITREE_MODEL_CODES[code]}"
    return _display_name(value)


def model_profile_path(model_path):
    model_path = Path(model_path).expanduser().resolve()
    return model_path.with_name(f"{model_path.stem}{MODEL_PROFILE_SUFFIX}")


def _load_model_profile(model_path):
    profile_path = model_profile_path(model_path)
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(profile, dict):
        return {}
    if profile.get("schema_version") not in (1, MODEL_PROFILE_SCHEMA_VERSION):
        return {}
    return profile


def _load_profile_home_joints(model_path):
    profile = _load_model_profile(model_path)
    values = profile.get("home_joints")
    if not isinstance(values, dict):
        return {}
    home_joints = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return {}
        home_joints[name] = float(value)
    return home_joints


def _profile_string_tuple(profile, key, default=()):
    values = profile.get(key, default)
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, (list, tuple)) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        return tuple(default)
    return tuple(value.strip() for value in values)


def _profile_string_mapping(profile, key):
    values = profile.get(key, {})
    if not isinstance(values, dict):
        return {}
    result = {}
    for name, candidates in values.items():
        if not isinstance(name, str) or not name.strip():
            return {}
        if isinstance(candidates, str):
            candidates = (candidates,)
        if not isinstance(candidates, (list, tuple)) or not all(
            isinstance(candidate, str) and candidate.strip()
            for candidate in candidates
        ):
            return {}
        result[name.strip()] = tuple(
            candidate.strip() for candidate in candidates
        )
    return result


def _profile_body_labels(profile):
    values = profile.get("body_labels", {})
    if not isinstance(values, dict) or not all(
        isinstance(name, str)
        and isinstance(label, str)
        and name.strip()
        and label.strip()
        for name, label in values.items()
    ):
        return {}
    return {name.strip(): label.strip() for name, label in values.items()}


def _profile_contact_pairs(profile):
    values = profile.get("allowed_contact_body_pairs", ())
    if not isinstance(values, (list, tuple)):
        return ()
    pairs = []
    for value in values:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
            or isinstance(value[2], bool)
            or not isinstance(value[2], (int, float))
            or not math.isfinite(float(value[2]))
            or float(value[2]) < 0.0
        ):
            return ()
        pairs.append((value[0], value[1], float(value[2])))
    return tuple(pairs)


def _write_generated_home_profile(model_path, adapter):
    profile_path = model_profile_path(model_path)
    profile = _load_model_profile(model_path)
    profile.update({
        "schema_version": MODEL_PROFILE_SCHEMA_VERSION,
        "home_source": "collision_repair",
        "home_joints": adapter.home_joint_values(),
    })
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return profile_path


def _unique_stem(directory, stem, suffix):
    candidate = stem
    index = 2
    while (
        (directory / f"{candidate}{suffix}").exists()
        or (directory / f"assets-{candidate}").exists()
        or (directory / f"{candidate}{MODEL_PROFILE_SUFFIX}").exists()
    ):
        candidate = f"{stem}-{index}"
        index += 1
    return candidate


def _copy_mesh(source, asset_dir, copied_by_path, used_names):
    source = Path(source).resolve()
    existing = copied_by_path.get(source)
    if existing is not None:
        return existing

    target_name = source.name
    if target_name in used_names:
        stem = source.stem
        suffix = source.suffix
        index = 2
        while f"{stem}-{index}{suffix}" in used_names:
            index += 1
        target_name = f"{stem}-{index}{suffix}"
    used_names.add(target_name)

    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / target_name
    shutil.copy2(source, target)
    copied_by_path[source] = target_name
    return target_name


def _normalized_mesh_roots(mesh_roots):
    return [
        Path(root).expanduser().resolve()
        for root in (mesh_roots or [])
        if root
    ]


def _mesh_root_candidates(reference, mesh_roots):
    reference = str(reference or "").strip()
    if not reference:
        return []
    roots = _normalized_mesh_roots(mesh_roots)
    path = Path(reference)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    elif reference.startswith("package://"):
        payload = reference[len("package://"):]
        package, separator, relative = payload.partition("/")
        if separator:
            relative_path = Path(relative)
            for root in roots:
                candidates.extend((
                    root / relative_path,
                    root / package / relative_path,
                    root / relative_path.name,
                ))
                if relative_path.parts and relative_path.parts[0].lower() in {
                    "dae", "meshes", "assets"
                }:
                    candidates.append(root.joinpath(*relative_path.parts[1:]))
    else:
        for root in roots:
            candidates.extend((root / path, root / path.name))
    return candidates


def _resolve_mesh(reference, model_dir, package_map=None, mesh_roots=None):
    resolved = resolve_mesh_path(reference, model_dir, package_map)
    if resolved.error is None:
        return resolved
    first_error = resolved.error
    for candidate in _mesh_root_candidates(reference, mesh_roots):
        candidate = candidate.resolve()
        if candidate.is_file():
            return ResolvedMeshAsset(reference, candidate)
    reference_name = Path(str(reference).removeprefix("package://")).name
    basename = reference_name.lower()
    stem = Path(reference_name).stem.lower()
    for root in _normalized_mesh_roots(mesh_roots):
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() == basename:
                return ResolvedMeshAsset(reference, candidate.resolve())
        for candidate in root.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix.lower() == ".stl"
                and candidate.stem.lower() == stem
            ):
                return ResolvedMeshAsset(reference, candidate.resolve())
    if mesh_roots:
        roots = ", ".join(str(root) for root in _normalized_mesh_roots(mesh_roots))
        return ResolvedMeshAsset(
            reference,
            None,
            f"{first_error}; also searched chosen mesh folder(s): {roots}",
        )
    return resolved if first_error else ResolvedMeshAsset(reference, None, first_error)


def _package_names(root):
    names = set()
    for mesh in root.findall(".//mesh"):
        reference = mesh.get("filename") or mesh.get("file") or ""
        if reference.startswith("package://"):
            payload = reference[len("package://"):]
            package, separator, _ = payload.partition("/")
            if separator and package:
                names.add(package)
    return names


def _infer_package_map(source_path, packages):
    source_path = Path(source_path).resolve()
    candidates_by_package = {}
    ancestors = [source_path.parent, *source_path.parents]
    for package in packages:
        candidates = []
        for ancestor in ancestors:
            if ancestor.name == package:
                candidates.append(ancestor)
            package_child = ancestor / package
            if package_child.is_dir():
                candidates.append(package_child)
        if source_path.parent.name.lower() in {"urdf", "mjcf", "xml"}:
            candidates.append(source_path.parent.parent)
        for candidate in candidates:
            if candidate.is_dir():
                candidates_by_package[package] = candidate.resolve()
                break
    return candidates_by_package


def _rewrite_urdf_meshes(root, source_path, asset_dir, mesh_roots=None):
    package_map = _infer_package_map(source_path, _package_names(root))
    copied_by_path = {}
    used_names = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        resolved = _resolve_mesh(
            filename, source_path.parent, package_map, mesh_roots
        )
        if resolved.error:
            raise RuntimeError(resolved.error)
        copied_name = _copy_mesh(
            resolved.path, asset_dir, copied_by_path, used_names
        )
        mesh.set("filename", f"{asset_dir.name}/{copied_name}")


def _mjcf_mesh_root(root, source_path):
    model_dir = source_path.parent
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    return (model_dir / meshdir).resolve() if meshdir else model_dir.resolve()


def _rewrite_mjcf_meshes(root, source_path, asset_dir, mesh_roots=None):
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    mesh_root = _mjcf_mesh_root(root, source_path)
    compiler.set("meshdir", asset_dir.name)

    copied_by_path = {}
    used_names = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        resolved = _resolve_mesh(filename, mesh_root, mesh_roots=mesh_roots)
        if resolved.error:
            raise RuntimeError(resolved.error)
        copied_name = _copy_mesh(
            resolved.path, asset_dir, copied_by_path, used_names
        )
        mesh.set("file", copied_name)


def _model_info_for_path(path, key=None, name_hints=()):
    path = Path(path).expanduser().resolve()
    stem = _slug(key or path.stem)
    profile = _load_model_profile(path)
    default_root_bodies = (
        "base", "base_link", "trunk", "pelvis", "link00", "root", "world"
    )
    default_root_joints = ("floating_base", "root", "freejoint")
    root_body_candidates = _profile_string_tuple(
        profile, "root_body_candidates", default_root_bodies
    )
    root_joint_candidates = _profile_string_tuple(
        profile, "root_joint_candidates", default_root_joints
    )
    floating_base = profile.get("floating_base")
    if not isinstance(floating_base, bool):
        floating_base = None
    model_type = profile.get("model_type", "generic")
    if not isinstance(model_type, str) or not model_type.strip():
        model_type = "generic"
    blocking_depth = profile.get("collision_blocking_penetration_m", 0.001)
    if (
        isinstance(blocking_depth, bool)
        or not isinstance(blocking_depth, (int, float))
        or not math.isfinite(float(blocking_depth))
        or float(blocking_depth) < 0.0
    ):
        blocking_depth = 0.001
    return RobotModelInfo(
        key=stem,
        display_name=_display_name_from_hints(stem, path.stem, *name_hints),
        model_type=model_type.strip(),
        model_path=path,
        root_body_candidates=root_body_candidates,
        root_joint_candidates=root_joint_candidates,
        logical_frames=_profile_string_mapping(profile, "logical_frames"),
        ignored_body_tokens=_profile_string_tuple(
            profile,
            "ignored_body_tokens",
            RobotModelInfo.__dataclass_fields__["ignored_body_tokens"].default,
        ),
        floating_base=floating_base,
        end_effector_frames=_profile_string_tuple(profile, "end_effectors"),
        joint_groups=_profile_string_mapping(profile, "joint_groups"),
        passive_joints=_profile_string_tuple(profile, "passive_joints"),
        home_joints=_load_profile_home_joints(path),
        body_labels=_profile_body_labels(profile),
        collision_blocking_penetration_m=float(blocking_depth),
        allowed_contact_body_pairs=_profile_contact_pairs(profile),
    )


def _validate_model_file(model_path):
    from core.models import MuJoCoRobotAdapter

    cache_root = model_path.parent / ".cache"
    old_cache_root = os.environ.get("GHOSTGUI_CACHE_DIR")
    os.environ["GHOSTGUI_CACHE_DIR"] = str(cache_root)
    try:
        return MuJoCoRobotAdapter(_model_info_for_path(model_path))
    finally:
        if old_cache_root is None:
            os.environ.pop("GHOSTGUI_CACHE_DIR", None)
        else:
            os.environ["GHOSTGUI_CACHE_DIR"] = old_cache_root


def _built_in_model_paths():
    paths = {info.model_path.resolve() for info in ROBOT_MODELS.values()}
    paths.update(
        (PROJECT_ROOT / "models" / filename).resolve()
        for filename in BUILT_IN_MODEL_FILENAMES
    )
    return paths


def discover_imported_models(library_root=None, excluded_paths=None):
    """
    Return model infos for files previously saved in the user model library.

    Discovery is intentionally light-weight: it checks the filename, extension,
    and optional small home-pose profile. Actual MuJoCo parsing still happens
    lazily when the user selects a model.
    """
    library_root = Path(library_root or default_model_library_root()).expanduser()
    if not library_root.is_dir():
        return {}

    excluded = (
        _built_in_model_paths()
        if excluded_paths is None
        else {Path(path).expanduser().resolve() for path in excluded_paths}
    )
    discovered = {}
    for path in sorted(library_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
            continue
        if path.name in BUILT_IN_MODEL_FILENAMES:
            continue
        if path.resolve() in excluded:
            continue
        info = _model_info_for_path(path)
        discovered[info.key] = info
    return discovered


def import_robot_model(source_path, library_root=None, mesh_roots=None):
    """
    Copy a URDF/MJCF model and its referenced meshes into the model library.

    Imported files are stored as ``<library>/<model>.urdf|xml`` and meshes are
    stored beside them in ``<library>/assets-<model>/``.
    """
    source_path = Path(source_path).expanduser().resolve()
    if source_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_EXTENSIONS))
        raise ValueError(f"Unsupported model extension; choose one of {allowed}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    library_root = Path(library_root or default_model_library_root()).expanduser()
    library_root.mkdir(parents=True, exist_ok=True)

    source_stem = _slug(source_path.stem)
    source_profile_path = model_profile_path(source_path)
    suffix = source_path.suffix.lower()
    stem = _unique_stem(library_root, source_stem, suffix)
    model_path = library_root / f"{stem}{suffix}"
    asset_dir = library_root / f"assets-{stem}"
    profile_path = model_profile_path(model_path)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".import-{stem}-", dir=library_root
        ) as staging:
            staging = Path(staging)
            staged_model_path = staging / model_path.name
            staged_asset_dir = staging / asset_dir.name
            staged_asset_dir.mkdir(parents=True, exist_ok=True)

            tree = ET.parse(source_path)
            root = tree.getroot()
            name_hints = [root.get("name"), root.get("model")]
            name_hints.extend(_package_names(root))
            if root.tag == "robot":
                _rewrite_urdf_meshes(root, source_path, staged_asset_dir, mesh_roots)
            else:
                _rewrite_mjcf_meshes(root, source_path, staged_asset_dir, mesh_roots)
            tree.write(staged_model_path, encoding="unicode")
            staged_profile_path = None
            if source_profile_path.is_file():
                staged_profile_path = model_profile_path(staged_model_path)
                shutil.copy2(source_profile_path, staged_profile_path)
            adapter = _validate_model_file(staged_model_path)
            if adapter.home_pose_was_repaired:
                staged_profile_path = _write_generated_home_profile(
                    staged_model_path, adapter
                )

            shutil.move(str(staged_model_path), str(model_path))
            shutil.move(str(staged_asset_dir), str(asset_dir))
            if staged_profile_path is not None:
                shutil.move(str(staged_profile_path), str(profile_path))
    except Exception:
        if model_path.exists():
            model_path.unlink()
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        if profile_path.exists():
            profile_path.unlink()
        raise

    return _model_info_for_path(model_path, key=stem, name_hints=name_hints)

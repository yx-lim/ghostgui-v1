"""GhostGUI project folder metadata and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

from application.paths import PROJECT_ROOT


PROJECT_FILENAME = "ghostgui_project.json"
PROJECT_SCHEMA_VERSION = 1
RECENT_PROJECTS_FILENAME = "recent_projects.json"
PROJECTS_DIR_ENV = "GHOSTGUI_PROJECTS_DIR"
PROJECTS_FOLDER_NAME = "projects"
MAX_RECENT_PROJECTS = 10

DEFAULT_PROJECT_FILES = {
    "target_trajectory": "data/target_frames.json",
    "qpos_timeline": "data/qpos_timeline.npz",
    "workspace": "workspace/workspace.json",
    "last_snapshot": "snapshots/last_workspace.png",
    "session_log": "metadata/session_log.jsonl",
}

DEFAULT_AUTOSAVE_FILES = {
    "manifest": "autosave/autosave_manifest.json",
    "target_trajectory": "autosave/target_frames.autosave.json",
    "qpos_timeline": "autosave/qpos_timeline.autosave.npz",
    "workspace": "autosave/workspace.autosave.json",
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def sanitized_project_folder_name(project_name):
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", project_name.strip())
    name = re.sub(r"\s+", "_", name).strip("._-")
    if not name:
        name = "ghostgui_project"
    if not name.endswith(".ghostgui"):
        name = f"{name}.ghostgui"
    return name


def project_root_from_name(parent_dir, project_name):
    return Path(parent_dir).expanduser() / sanitized_project_folder_name(project_name)


def ghostgui_projects_dir():
    override = os.environ.get(PROJECTS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / PROJECTS_FOLDER_NAME


def default_project_root_from_name(project_name):
    return project_root_from_name(ghostgui_projects_dir(), project_name)


def available_project_root_from_name(parent_dir, project_name):
    base = project_root_from_name(parent_dir, project_name)
    if not base.exists():
        return base
    suffix = ".ghostgui"
    stem = base.name[:-len(suffix)] if base.name.endswith(suffix) else base.name
    for index in range(2, 1000):
        candidate = base.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available project folder for {base}")


def available_default_project_root_from_name(project_name):
    return available_project_root_from_name(ghostgui_projects_dir(), project_name)


def ghostgui_config_dir():
    override = os.environ.get("GHOSTGUI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GhostGUI"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "GhostGUI"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "ghostgui"
    return Path.home() / ".config" / "ghostgui"


def recent_projects_path(config_dir=None):
    root = Path(config_dir).expanduser() if config_dir else ghostgui_config_dir()
    return root / RECENT_PROJECTS_FILENAME


def _project_root_from_recent_path(path):
    project_path = Path(path).expanduser()
    if project_path.name == PROJECT_FILENAME:
        return project_path.parent
    return project_path


def _recent_project_key(path):
    return str(_project_root_from_recent_path(path).resolve())


def _is_project_root(path):
    return (_project_root_from_recent_path(path) / PROJECT_FILENAME).exists()


def _recent_project_entry(project):
    return {
        "path": str(project.root_dir),
        "project_name": project.project_name,
        "model_key": project.model_key,
        "last_opened_at": utc_now_iso(),
    }


def _clean_recent_project_entries(entries, limit=MAX_RECENT_PROJECTS):
    cleaned = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not raw_path:
            continue
        project_root = _project_root_from_recent_path(raw_path)
        if not _is_project_root(project_root):
            continue
        key = _recent_project_key(project_root)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "path": key,
                "project_name": str(
                    entry.get("project_name") or project_root.stem
                ),
                "model_key": str(entry.get("model_key") or ""),
                "last_opened_at": str(entry.get("last_opened_at") or ""),
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def load_recent_projects(config_dir=None, limit=MAX_RECENT_PROJECTS):
    path = recent_projects_path(config_dir)
    try:
        payload = _read_json(path)
    except FileNotFoundError:
        return []
    except (OSError, TypeError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        entries = payload.get("projects", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        return []
    return _clean_recent_project_entries(entries, limit=limit)


def save_recent_projects(entries, config_dir=None, limit=MAX_RECENT_PROJECTS):
    cleaned = _clean_recent_project_entries(entries, limit=limit)
    _write_json(
        recent_projects_path(config_dir),
        {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "projects": cleaned,
        },
    )
    return cleaned


def remember_recent_project(project, config_dir=None, limit=MAX_RECENT_PROJECTS):
    entry = _recent_project_entry(project)
    entry_key = _recent_project_key(entry["path"])
    entries = [entry]
    entries.extend(
        recent
        for recent in load_recent_projects(config_dir, limit=limit)
        if _recent_project_key(recent["path"]) != entry_key
    )
    return save_recent_projects(entries, config_dir=config_dir, limit=limit)


def forget_recent_project(path, config_dir=None, limit=MAX_RECENT_PROJECTS):
    remove_key = _recent_project_key(path)
    entries = [
        entry
        for entry in load_recent_projects(config_dir, limit=limit)
        if _recent_project_key(entry["path"]) != remove_key
    ]
    return save_recent_projects(entries, config_dir=config_dir, limit=limit)


def project_preview_from_project(project, last_opened_at=""):
    robot = project.metadata.get("robot", {})
    snapshot_path = project.paths.last_snapshot
    return {
        "path": str(project.root_dir),
        "project_name": project.project_name,
        "model_key": project.model_key,
        "model_name": str(robot.get("model_name") or project.model_key),
        "modified_at": str(project.metadata.get("modified_at") or ""),
        "last_opened_at": str(last_opened_at or ""),
        "snapshot_path": str(snapshot_path) if snapshot_path.exists() else "",
    }


def project_preview_from_recent_entry(entry):
    project = GhostGUIProject.open(entry["path"])
    return project_preview_from_project(
        project,
        last_opened_at=entry.get("last_opened_at"),
    )


def project_roots_in_dir(projects_dir=None):
    root = Path(projects_dir).expanduser() if projects_dir else ghostgui_projects_dir()
    if not root.exists():
        return []
    roots = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / PROJECT_FILENAME).exists()
    ]
    return sorted(
        roots,
        key=lambda path: (path / PROJECT_FILENAME).stat().st_mtime,
        reverse=True,
    )


def load_recent_project_previews(config_dir=None, limit=MAX_RECENT_PROJECTS):
    previews = []
    for entry in load_recent_projects(config_dir, limit=limit):
        try:
            previews.append(project_preview_from_recent_entry(entry))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return previews


def load_project_browser_previews(
    config_dir=None,
    projects_dir=None,
    limit=MAX_RECENT_PROJECTS,
):
    previews = []
    seen = set()
    for preview in load_recent_project_previews(config_dir, limit=limit):
        key = _recent_project_key(preview["path"])
        if key in seen:
            continue
        seen.add(key)
        previews.append(preview)
    for root in project_roots_in_dir(projects_dir):
        key = _recent_project_key(root)
        if key in seen:
            continue
        try:
            previews.append(project_preview_from_project(GhostGUIProject.open(root)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        seen.add(key)
    return previews


@dataclass(frozen=True)
class ProjectPaths:
    project_file: Path
    target_trajectory: Path
    qpos_timeline: Path
    workspace: Path
    last_snapshot: Path
    session_log: Path


@dataclass(frozen=True)
class AutosavePaths:
    manifest: Path
    target_trajectory: Path
    qpos_timeline: Path
    workspace: Path


@dataclass
class GhostGUIProject:
    root_dir: Path
    metadata: dict

    @classmethod
    def create(cls, root_dir, project_name, model_key, model_name=None):
        root_dir = Path(root_dir).expanduser().resolve()
        now = utc_now_iso()
        metadata = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_name": project_name,
            "created_at": now,
            "modified_at": now,
            "application": {
                "name": "GhostGUI",
                "project_format": "ghostgui.project.v1",
            },
            "robot": {
                "model_key": model_key,
                "model_name": model_name or model_key,
            },
            "files": dict(DEFAULT_PROJECT_FILES),
        }
        project = cls(root_dir=root_dir, metadata=metadata)
        project.ensure_directories()
        project.save_metadata(update_modified=False)
        project.append_session_event(
            "project_created",
            {
                "project_name": project_name,
                "model_key": model_key,
                "model_name": model_name or model_key,
            },
        )
        return project

    @classmethod
    def open(cls, path):
        path = Path(path).expanduser()
        project_file = path / PROJECT_FILENAME if path.is_dir() else path
        project_file = project_file.resolve()
        metadata = _read_json(project_file)
        schema_version = int(metadata.get("schema_version", 0))
        if schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GhostGUI project schema: {schema_version}"
            )
        return cls(root_dir=project_file.parent, metadata=metadata)

    @property
    def project_file(self):
        return self.root_dir / PROJECT_FILENAME

    @property
    def project_name(self):
        return self.metadata.get("project_name") or self.root_dir.stem

    @property
    def model_key(self):
        return self.metadata.get("robot", {}).get("model_key", "g1")

    @property
    def paths(self):
        return ProjectPaths(
            project_file=self.project_file,
            target_trajectory=self.resolve_file("target_trajectory"),
            qpos_timeline=self.resolve_file("qpos_timeline"),
            workspace=self.resolve_file("workspace"),
            last_snapshot=self.resolve_file("last_snapshot"),
            session_log=self.resolve_file("session_log"),
        )

    @property
    def autosave_paths(self):
        return AutosavePaths(
            manifest=self.resolve_autosave_file("manifest"),
            target_trajectory=self.resolve_autosave_file("target_trajectory"),
            qpos_timeline=self.resolve_autosave_file("qpos_timeline"),
            workspace=self.resolve_autosave_file("workspace"),
        )

    def resolve_file(self, key):
        relative = self.metadata.get("files", {}).get(key)
        if relative is None:
            relative = DEFAULT_PROJECT_FILES[key]
            self.metadata.setdefault("files", {})[key] = relative
        return self.root_dir / relative

    def resolve_autosave_file(self, key):
        relative = (
            self.metadata.get("autosave_files", {}).get(key)
            or DEFAULT_AUTOSAVE_FILES[key]
        )
        self.metadata.setdefault("autosave_files", {})[key] = relative
        return self.root_dir / relative

    def ensure_directories(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for path in self.paths.__dict__.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        for path in self.autosave_paths.__dict__.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "exports").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "logs").mkdir(parents=True, exist_ok=True)

    def update_robot(self, model_key, model_name=None):
        self.metadata.setdefault("robot", {})
        self.metadata["robot"]["model_key"] = model_key
        self.metadata["robot"]["model_name"] = model_name or model_key

    def save_metadata(self, update_modified=True):
        self.ensure_directories()
        if update_modified:
            self.metadata["modified_at"] = utc_now_iso()
        _write_json(self.project_file, self.metadata)

    def append_session_event(self, event, details=None):
        self.ensure_directories()
        record = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "event": str(event),
        }
        if details:
            record["details"] = _json_safe(details)
        with self.paths.session_log.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        return record

    def read_session_log(self, limit=None):
        records = []
        try:
            with self.paths.session_log.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except FileNotFoundError:
            return []
        if limit is None:
            return records
        return records[-int(limit):]

    def write_trajectory(self, trajectory):
        _write_json(self.paths.target_trajectory, trajectory.to_project_dict())

    def read_trajectory_dict(self, autosave=False):
        path = (
            self.autosave_paths.target_trajectory
            if autosave else self.paths.target_trajectory
        )
        return _read_json(path)

    def save_qpos_timeline(self, timeline):
        timeline.save_npz(self.paths.qpos_timeline)

    def load_qpos_timeline(self, timeline, autosave=False):
        path = (
            self.autosave_paths.qpos_timeline
            if autosave else self.paths.qpos_timeline
        )
        timeline.load_npz(path)

    def write_workspace(self, workspace_state):
        _write_json(self.paths.workspace, workspace_state)

    def read_workspace(self, autosave=False):
        path = self.autosave_paths.workspace if autosave else self.paths.workspace
        return _read_json(path)

    def write_autosave(
        self,
        trajectory,
        timeline,
        workspace_state,
        model_key,
        model_name=None,
    ):
        self.ensure_directories()
        _write_json(self.autosave_paths.target_trajectory, trajectory.to_project_dict())
        timeline.save_npz(self.autosave_paths.qpos_timeline)
        _write_json(self.autosave_paths.workspace, workspace_state)
        _write_json(
            self.autosave_paths.manifest,
            {
                "schema_version": PROJECT_SCHEMA_VERSION,
                "autosaved_at": utc_now_iso(),
                "robot": {
                    "model_key": model_key,
                    "model_name": model_name or model_key,
                },
            },
        )

    def read_autosave_manifest(self):
        return _read_json(self.autosave_paths.manifest)

    def autosave_exists(self):
        paths = self.autosave_paths
        return (
            paths.manifest.exists()
            and paths.target_trajectory.exists()
            and paths.qpos_timeline.exists()
            and paths.workspace.exists()
        )

    def autosave_model_key(self):
        if not self.autosave_paths.manifest.exists():
            return self.model_key
        manifest = self.read_autosave_manifest()
        return manifest.get("robot", {}).get("model_key", self.model_key)

    def is_autosave_newer(self):
        if not self.autosave_exists():
            return False
        try:
            autosave_mtime = max(
                path.stat().st_mtime
                for path in self.autosave_paths.__dict__.values()
            )
            saved_mtime = max(
                path.stat().st_mtime
                for path in (
                    self.project_file,
                    self.paths.target_trajectory,
                    self.paths.qpos_timeline,
                    self.paths.workspace,
                )
                if path.exists()
            )
        except (OSError, ValueError):
            return False
        return autosave_mtime > saved_mtime

    def clear_autosave(self):
        for path in self.autosave_paths.__dict__.values():
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

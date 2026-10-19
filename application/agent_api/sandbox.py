"""Resolve untrusted leaf names within caller-configured local roots."""

from pathlib import Path
import re

from application.project_manager import safe_project_path


def safe_leaf(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,79}", name):
        raise ValueError("name must be a plain file name, without path separators")
    if name.endswith((".", " ")):
        raise ValueError("name cannot end with a dot or space")
    return name


class OutputSandbox:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def resolve_output(self, file_name, *, suffix=""):
        name = safe_leaf(file_name)
        if suffix and not name.lower().endswith(suffix):
            name += suffix
        return safe_project_path(self.root, name, label="agent output")

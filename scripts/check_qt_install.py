#!/usr/bin/env python3
"""Validate that an environment uses GhostGUI's lightweight Qt runtime."""

from __future__ import annotations

import argparse
from importlib import metadata
import re
import sys


REQUIRED_DISTRIBUTION = "pyside6-essentials"
FORBIDDEN_DISTRIBUTIONS = frozenset({"pyside6", "pyside6-addons"})


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip()).lower()


def installed_distribution_names() -> set[str]:
    names = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            names.add(normalize_distribution_name(name))
    return names


def validate_distribution_names(
    names,
    *,
    require_essentials: bool = True,
) -> list[str]:
    normalized = {normalize_distribution_name(name) for name in names}
    issues = []
    forbidden = sorted(normalized & FORBIDDEN_DISTRIBUTIONS)
    if forbidden:
        issues.append(
            "legacy full-Qt distributions are installed: " + ", ".join(forbidden)
        )
    if require_essentials and REQUIRED_DISTRIBUTION not in normalized:
        issues.append("PySide6-Essentials is not installed")
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="allow Essentials to be absent before dependency installation",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    issues = validate_distribution_names(
        installed_distribution_names(),
        require_essentials=not args.preflight,
    )
    if issues:
        print("GhostGUI Qt dependency check failed:")
        for issue in issues:
            print(f"- {issue}")
        if any("legacy full-Qt" in issue for issue in issues):
            print(
                "Recreate GhostGUI's dedicated .venv; uninstalling overlapping "
                "PySide6 wheels in place is not supported."
            )
        return 1
    print("GhostGUI Qt dependency check passed: Essentials only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

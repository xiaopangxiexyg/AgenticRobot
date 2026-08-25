from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import pytest

from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_ROOT = REPO_ROOT / "hardware"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _registered_submodule_roots() -> tuple[Path, ...]:
    parser = ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    return tuple(
        REPO_ROOT / parser.get(section, "path")
        for section in parser.sections()
    )


SUBMODULE_ROOTS = _registered_submodule_roots()


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
            and not any(
                submodule_root in path.parents
                for submodule_root in SUBMODULE_ROOTS
            )
        )
    )


def _violations(root: Path, forbidden_roots: set[str]) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for path in _python_files(root):
        forbidden = _absolute_import_roots(path) & forbidden_roots
        if forbidden:
            violations[path.relative_to(REPO_ROOT).as_posix()] = sorted(forbidden)
    return violations


def test_hardware_does_not_depend_on_application_runtime_or_model_layers() -> None:
    assert _violations(
        HARDWARE_ROOT,
        {
            "collection_runtime",
            "control_station",
            "dataset_interface",
            "deployments",
            "model",
            "sim",
            "teleop_runtime",
        },
    ) == {}


def test_robot_motion_hardware_dependency_debt_cannot_spread() -> None:
    # Tool-profile JSON still has a stable, externally referenced location below
    # hardware/rm65.  Keep that existing adapter explicit until the artifact-path
    # migration is handled as its own compatibility slice.
    assert _violations(REPO_ROOT / "robot_motion", {"hardware"}) == {
        "robot_motion/rm65/tool_profiles.py": ["hardware"],
    }

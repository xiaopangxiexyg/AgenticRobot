from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_ROOT = REPO_ROOT / "hardware"
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _violations(root: Path, forbidden_roots: set[str]) -> dict[str, list[str]]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): sorted(
            _absolute_import_roots(path) & forbidden_roots
        )
        for path in _python_files(root)
        if _absolute_import_roots(path) & forbidden_roots
    }


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

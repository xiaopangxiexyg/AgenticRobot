from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_model_directory_is_artifact_only() -> None:
    model_root = REPO_ROOT / "model"

    assert (model_root / "README.md").is_file()
    assert sorted(model_root.rglob("*.py")) == []


def test_policy_learning_does_not_depend_on_runtime_or_motion_owners() -> None:
    forbidden = {
        "collection_runtime",
        "control_composition",
        "control_station",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    }
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_roots(path) & forbidden
        )
        for path in sorted((REPO_ROOT / "policy_learning").rglob("*.py"))
        if _absolute_import_roots(path) & forbidden
    }

    assert violations == {}


def test_lower_owners_do_not_import_policy_learning() -> None:
    lower_owners = (
        "collection_runtime",
        "control_composition",
        "dataset_contracts",
        "dataset_interface",
        "hardware",
        "robot_motion",
        "sim",
    )
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            root
            for root in _absolute_import_roots(path)
            if root == "policy_learning"
        )
        for owner in lower_owners
        for path in sorted((REPO_ROOT / owner).rglob("*.py"))
        if "policy_learning" in _absolute_import_roots(path)
    }

    assert violations == {}

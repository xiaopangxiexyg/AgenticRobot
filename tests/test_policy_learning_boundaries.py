from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _registered_model_source_submodule_roots() -> tuple[Path, ...]:
    parser = ConfigParser()
    parser.read(REPO_ROOT / ".gitmodules", encoding="utf-8")
    model_root = REPO_ROOT / "model"
    return tuple(
        REPO_ROOT / parser.get(section, "path")
        for section in parser.sections()
        if (REPO_ROOT / parser.get(section, "path")).parent == model_root
    )


MODEL_SOURCE_SUBMODULE_ROOTS = _registered_model_source_submodule_roots()


def _is_model_source_submodule_path(path: Path) -> bool:
    return any(root in path.parents for root in MODEL_SOURCE_SUBMODULE_ROOTS)


def test_model_directory_is_artifact_only() -> None:
    model_root = REPO_ROOT / "model"

    assert (model_root / "README.md").is_file()
    assert sorted(
        path
        for path in model_root.rglob("*.py")
        if not _is_model_source_submodule_path(path)
    ) == []


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

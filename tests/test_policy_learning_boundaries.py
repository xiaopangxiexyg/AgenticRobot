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


def test_experimental_policy_learning_has_a_narrow_dependency_surface() -> None:
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
    violations: dict[str, list[str]] = {}
    for path in sorted(
        (REPO_ROOT / "experiments" / "policy_learning").rglob("*.py")
    ):
        forbidden_imports = _absolute_import_roots(path) & forbidden
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)

    assert violations == {}


def test_production_owners_do_not_import_experiments() -> None:
    production_owners = (
        "agent_gateway",
        "collection_runtime",
        "control_composition",
        "control_station",
        "dataset_contracts",
        "dataset_interface",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    )
    violations: dict[str, list[str]] = {}
    for owner in production_owners:
        for path in sorted((REPO_ROOT / owner).rglob("*.py")):
            imports = _absolute_import_roots(path)
            if "experiments" in imports:
                violations[str(path.relative_to(REPO_ROOT))] = sorted(
                    root for root in imports if root == "experiments"
                )

    assert violations == {}


def test_top_level_policy_learning_package_is_removed() -> None:
    assert not (REPO_ROOT / "policy_learning").exists()


def test_experiments_are_discoverable_without_becoming_a_core_dependency() -> None:
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '    "experiments*",' in project
    assert '    "policy_learning*",' not in project


def test_arm_hold_compatibility_launchers_stay_thin() -> None:
    launchers = (
        "probe_arm_hold_front_state_batch_benchmark.py",
        "probe_arm_hold_one_episode_overfit.py",
        "probe_arm_hold_state_only_train_validation.py",
    )

    for name in launchers:
        path = REPO_ROOT / "scripts" / "probes" / name
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 18
        assert "from experiments.policy_learning.arm_hold.commands" in source


def test_experiments_do_not_add_catch_all_modules() -> None:
    forbidden_names = {"common.py", "helpers.py", "misc.py", "utils.py"}

    assert {
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "experiments").rglob("*.py")
        if path.name in forbidden_names
    } == set()

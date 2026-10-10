from __future__ import annotations

import subprocess
from configparser import ConfigParser
from pathlib import Path

from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _registered_model_source_submodule_roots(
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, ...]:
    parser = ConfigParser()
    parser.read(repo_root / ".gitmodules", encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", "model"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gitlinks = {
        path
        for entry in tracked.split("\0")
        if entry
        for header, path in [entry.split("\t", 1)]
        if header.split()[0] == "160000" and header.split()[2] == "0"
    }
    model_root = repo_root / "model"
    return tuple(
        repo_root / parser.get(section, "path")
        for section in parser.sections()
        if parser.get(section, "path") in gitlinks
        and model_root in (repo_root / parser.get(section, "path")).parents
    )


MODEL_SOURCE_SUBMODULE_ROOTS = _registered_model_source_submodule_roots()


def _is_model_source_submodule_path(path: Path) -> bool:
    return any(root in path.parents for root in MODEL_SOURCE_SUBMODULE_ROOTS)


def test_model_source_boundary_requires_registered_gitlinks(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text(
        '[submodule "nested"]\npath = model/perception/vendor\n'
        '[submodule "untracked"]\npath = model/untracked\n'
        '[submodule "ordinary"]\npath = model/ordinary\n'
        '[submodule "hardware"]\npath = hardware/vendor\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for path in ("model/perception/vendor", "model/unregistered", "hardware/vendor"):
        subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", "160000", "1" * 40, path],
            cwd=tmp_path,
            check=True,
        )
    ordinary = tmp_path / "model" / "ordinary"
    ordinary.parent.mkdir()
    ordinary.write_text("not a submodule", encoding="utf-8")
    subprocess.run(["git", "add", "model/ordinary"], cwd=tmp_path, check=True)

    assert _registered_model_source_submodule_roots(tmp_path) == (
        tmp_path / "model/perception/vendor",
    )


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

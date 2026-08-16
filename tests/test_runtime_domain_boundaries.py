from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
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


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _violations(
    root: Path,
    forbidden_roots: set[str],
    *,
    excluded: frozenset[Path] = frozenset(),
) -> dict[str, list[str]]:
    return {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_roots(path) & forbidden_roots
        )
        for path in _python_files(root)
        if path not in excluded and _absolute_import_roots(path) & forbidden_roots
    }


def test_collection_frontend_does_not_absorb_deploy_or_composition() -> None:
    assert _violations(
        REPO_ROOT / "teleop_runtime" / "collection",
        {"control_composition", "deployments"},
    ) == {}


def test_general_real_deploy_does_not_absorb_composition_or_simulation() -> None:
    assert _violations(
        REPO_ROOT / "teleop_runtime" / "deploy",
        {"collection_runtime", "control_composition", "control_station", "sim"},
    ) == {}


def test_model_integrations_do_not_own_application_or_backend_runtime() -> None:
    compatibility_entrypoint = (
        REPO_ROOT / "deployments" / "openpi" / "client" / "real.py"
    )
    assert _violations(
        REPO_ROOT / "deployments",
        {
            "collection_runtime",
            "control_station",
            "hardware",
            "sim",
            "teleop_runtime",
        },
        excluded=frozenset({compatibility_entrypoint}),
    ) == {}

    tree = ast.parse(
        compatibility_entrypoint.read_text(encoding="utf-8"),
        filename=str(compatibility_entrypoint),
    )
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }
    assert imported_modules == {"teleop_runtime.deploy.openpi_application"}


def test_shared_deploy_and_composition_cores_remain_backend_neutral() -> None:
    assert _violations(
        REPO_ROOT / "robot_motion" / "deploy",
        {
            "collection_runtime",
            "control_composition",
            "control_station",
            "deployments",
            "hardware",
            "sim",
            "teleop_runtime",
        },
    ) == {}
    assert _violations(
        REPO_ROOT / "control_composition",
        {
            "collection_runtime",
            "control_station",
            "dataset_interface",
            "deployments",
            "hardware",
            "sim",
            "teleop_runtime",
        },
    ) == {}


def test_simulation_does_not_reverse_depend_on_real_or_application_packages() -> None:
    assert _violations(
        REPO_ROOT / "sim",
        {
            "collection_runtime",
            "control_station",
            "deployments",
            "hardware",
            "teleop_runtime",
        },
    ) == {}


def test_optional_sim_deploy_integrations_do_not_spread_into_simulation_core() -> None:
    integration_root = REPO_ROOT / "sim" / "mujoco" / "deploy"
    integration_files = frozenset(_python_files(integration_root))
    assert _violations(
        REPO_ROOT / "sim",
        {"control_composition"},
        excluded=integration_files,
    ) == {}


def test_mujoco_target_runtime_delegates_composition_runtime_construction() -> None:
    target_runtime = REPO_ROOT / "sim" / "mujoco" / "deploy" / "runtime.py"
    composition_binding = (
        REPO_ROOT / "sim" / "mujoco" / "deploy" / "composition_binding.py"
    )
    implementation_constructors = {
        "CompositionControlTraceEmitter",
        "ControlCompositionRuntime",
        "TimeWindowRetentionSpec",
    }

    assert _called_names(target_runtime).isdisjoint(implementation_constructors)
    assert implementation_constructors <= _called_names(composition_binding)

    runtime_tree = ast.parse(
        target_runtime.read_text(encoding="utf-8"),
        filename=str(target_runtime),
    )
    imported_symbols = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "build_mujoco_composition_runtime_binding" in imported_symbols
    assert imported_symbols.isdisjoint(implementation_constructors)

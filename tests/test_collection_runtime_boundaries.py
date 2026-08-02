from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_RUNTIME_ROOT = REPO_ROOT / "collection_runtime"
LEGACY_COLLECTION_ROOT = REPO_ROOT / "teleop_runtime" / "collection"
MIGRATED_MODULES = (
    "async_writer",
    "dataset_episode_maintenance",
    "immutable_canonical_store",
    "stream_sink",
    "v25p_component_action",
    "v25p_native_store",
    "v25p_parquet_store",
    "v25p_producer",
    "v25p_sidecar",
)
VERSIONED_RUNTIME_SHIMS = frozenset(
    f"collection_runtime.{module_name}"
    for module_name in (
        "v25p_component_action",
        "v25p_native_store",
        "v25p_parquet_store",
        "v25p_producer",
        "v25p_sidecar",
    )
)
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


def _absolute_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_collection_runtime_does_not_reverse_depend_on_application_planes() -> None:
    forbidden = {
        "cameras",
        "control_station",
        "deployments",
        "hardware",
        "rm65_inspire_tac",
        "sim",
        "teleop_runtime",
    }
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_roots(path) & forbidden
        )
        for path in _python_files(COLLECTION_RUNTIME_ROOT)
        if _absolute_import_roots(path) & forbidden
    }
    assert violations == {}


@pytest.mark.parametrize("module_name", MIGRATED_MODULES)
def test_legacy_collection_module_is_a_thin_reexport(module_name: str) -> None:
    path = LEGACY_COLLECTION_ROOT / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    executable_nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    assert executable_nodes
    assert all(isinstance(node, ast.ImportFrom) for node in executable_nodes)
    assert {
        node.module for node in executable_nodes if isinstance(node, ast.ImportFrom)
    } == {f"collection_runtime.{module_name}"}


def test_collection_runtime_package_import_is_side_effect_free() -> None:
    code = """
import sys
import collection_runtime

loaded_submodules = sorted(
    name for name in sys.modules if name.startswith("collection_runtime.")
)
assert loaded_submodules == [], loaded_submodules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_production_callers_use_capability_named_collection_runtime_modules() -> None:
    allowed_compatibility_shims = {
        LEGACY_COLLECTION_ROOT / f"{module_name}.py"
        for module_name in (
            "v25p_component_action",
            "v25p_native_store",
            "v25p_parquet_store",
            "v25p_producer",
            "v25p_sidecar",
        )
    }
    violations = {
        str(path.relative_to(REPO_ROOT)): sorted(
            _absolute_import_modules(path) & VERSIONED_RUNTIME_SHIMS
        )
        for path in _python_files(REPO_ROOT / "teleop_runtime")
        if path not in allowed_compatibility_shims
        and _absolute_import_modules(path) & VERSIONED_RUNTIME_SHIMS
    }
    assert violations == {}


def test_collection_runtime_is_in_package_discovery() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"collection_runtime*"' in pyproject


def test_legacy_real_teleop_dataset_layout_is_a_thin_reexport() -> None:
    path = REPO_ROOT / "teleop_runtime" / "real_teleop" / "dataset_layout.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    executable_nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    assert executable_nodes
    assert all(isinstance(node, ast.ImportFrom) for node in executable_nodes)
    assert {
        node.module for node in executable_nodes if isinstance(node, ast.ImportFrom)
    } == {"collection_runtime.dataset_layout"}


def test_umi_runtime_bundle_is_separate_from_default_concrete_devices() -> None:
    umi_root = REPO_ROOT / "teleop_runtime" / "collection" / "umi"
    bundle_source = (umi_root / "components.py").read_text(encoding="utf-8")
    default_source = (umi_root / "default_components.py").read_text(encoding="utf-8")

    concrete_names = {
        "AngleTriggerSource",
        "DMTacGripperBackend",
        "RealSenseRGBSource",
        "VivePoseSource",
    }
    assert all(name not in bundle_source for name in concrete_names)
    assert all(name in default_source for name in concrete_names)


def test_umi_controller_does_not_import_default_or_concrete_device_adapters() -> None:
    path = (
        REPO_ROOT
        / "teleop_runtime"
        / "collection"
        / "umi"
        / "controller.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "default_components" not in source
    assert "hardware." not in source
    assert "DMTacGripperBackend" not in source
    assert "VivePoseSource" not in source
    assert "RealSenseRGBSource" not in source
    assert "AngleTriggerSource" not in source


def test_umi_recorder_uses_canonical_collection_runtime_owners() -> None:
    path = (
        REPO_ROOT
        / "teleop_runtime"
        / "collection"
        / "umi"
        / "recorder.py"
    )
    modules = _absolute_import_modules(path)
    assert "collection_runtime.async_writer" in modules
    assert "collection_runtime.dataset_episode_maintenance" in modules

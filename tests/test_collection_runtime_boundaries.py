from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from tests.support.repository_imports import (
    absolute_import_modules as _absolute_import_modules,
    absolute_import_roots as _absolute_import_roots,
    python_files as _python_files,
)

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


def test_collection_runtime_does_not_reverse_depend_on_application_planes() -> None:
    forbidden = {
        "cameras",
        "control_station",
        "deployments",
        "hardware",
        "rm65_inspire_tac",
        "sim",
        "teleop_runtime",
        "utils",
    }
    violations: dict[str, list[str]] = {}
    for path in _python_files(COLLECTION_RUNTIME_ROOT):
        forbidden_imports = _absolute_import_roots(path) & forbidden
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)
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
    violations: dict[str, list[str]] = {}
    for path in _python_files(REPO_ROOT / "teleop_runtime"):
        if path in allowed_compatibility_shims:
            continue
        forbidden_imports = _absolute_import_modules(path) & VERSIONED_RUNTIME_SHIMS
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)
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


def test_dataset_postprocess_modality_storage_is_a_thin_reexport() -> None:
    path = REPO_ROOT / "utils" / "dataset_postprocess" / "modality_storage.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    executable_nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        and not (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
    ]
    assert executable_nodes
    assert all(isinstance(node, ast.ImportFrom) for node in executable_nodes)
    assert {
        node.module for node in executable_nodes if isinstance(node, ast.ImportFrom)
    } == {"collection_runtime.modality_storage"}


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


def test_control_station_real_teleop_entrypoint_uses_canonical_collection_runtime_owners() -> None:
    path = REPO_ROOT / "control_station" / "real_teleop_entrypoint.py"
    modules = _absolute_import_modules(path)
    assert "collection_runtime.async_writer" in modules
    assert "collection_runtime.dataset_episode_maintenance" in modules
    assert "teleop_runtime.collection.async_writer" not in modules
    assert "teleop_runtime.collection.dataset_episode_maintenance" not in modules

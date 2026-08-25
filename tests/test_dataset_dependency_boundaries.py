from __future__ import annotations

import ast
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dataset_contracts import parquet as parquet_contract
from dataset_contracts.stats import EpisodeStatsAccumulator
from dataset_contracts.tactile import ChunkedTactileStream
from dataset_contracts.wrist_depth import DepthRangeMaskConfig, mask_manifest_payload
from tests.support.repository_imports import (
    absolute_import_roots as _absolute_import_roots,
    dynamic_import_modules as _dynamic_imports,
    imported_modules as _imported_modules,
    python_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parsed_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_files(root: Path) -> list[Path]:
    return list(python_files(root))


@lru_cache(maxsize=None)
def _is_declared_diagnostic_bridge(path: Path) -> bool:
    tree = _parsed_tree(path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "DIAGNOSTIC_RUNTIME_BRIDGE" for target in node.targets):
            continue
        return isinstance(node.value, ast.Constant) and node.value.value is True
    return False


def test_dataset_contracts_has_no_application_or_tooling_dependencies() -> None:
    forbidden = {
        "teleop_runtime",
        "robot_motion",
        "hardware",
        "cameras",
        "sim",
        "utils",
        "rm65_inspire_tac",
    }
    violations: dict[str, list[str]] = {}
    for path in _python_files(REPO_ROOT / "dataset_contracts"):
        imported_roots = {name.split(".", 1)[0] for name in _imported_modules(path)}
        forbidden_imports = imported_roots & forbidden
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)
    assert violations == {}


def test_dataset_interface_has_no_application_or_tooling_dependencies() -> None:
    forbidden = {
        "collection_runtime",
        "control_station",
        "deployments",
        "hardware",
        "model",
        "sim",
        "teleop_runtime",
        "utils",
    }
    violations: dict[str, list[str]] = {}
    for path in _python_files(REPO_ROOT / "dataset_interface"):
        imported_roots = {name.split(".", 1)[0] for name in _imported_modules(path)}
        forbidden_imports = imported_roots & forbidden
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)
    assert violations == {}


def test_offline_postprocess_has_no_static_runtime_dependency() -> None:
    forbidden = {
        "teleop_runtime",
        "robot_motion",
        "hardware",
        "cameras",
        "sim",
        "rm65_inspire_tac",
    }
    violations: dict[str, list[str]] = {}
    for path in _python_files(REPO_ROOT / "utils/dataset_postprocess"):
        if _is_declared_diagnostic_bridge(path):
            continue
        forbidden_imports = _absolute_import_roots(path) & forbidden
        if forbidden_imports:
            violations[str(path.relative_to(REPO_ROOT))] = sorted(forbidden_imports)
    assert violations == {}


def test_postprocess_runtime_bridge_is_explicit_and_limited_to_camera_smoke_cli() -> None:
    dynamic_runtime_imports: dict[str, list[str]] = {}
    for path in _python_files(REPO_ROOT / "utils/dataset_postprocess"):
        runtime_imports = sorted(
            name
            for name in _dynamic_imports(path)
            if name.split(".", 1)[0] in {"teleop_runtime", "robot_motion"}
        )
        if runtime_imports:
            dynamic_runtime_imports[str(path.relative_to(REPO_ROOT))] = runtime_imports
    expected_path = "utils/dataset_postprocess/wrist_cam_adjust/smoke_wrist_depth_capture.py"
    assert dynamic_runtime_imports == {
        expected_path: ["teleop_runtime.sources.camera"],
    }
    assert _is_declared_diagnostic_bridge(REPO_ROOT / expected_path)


def test_runtime_and_sim_do_not_import_offline_postprocess_implementation() -> None:
    violations: list[str] = []
    for package in ("teleop_runtime", "robot_motion", "sim"):
        for path in _python_files(REPO_ROOT / package):
            if any(
                name == "utils.dataset_postprocess"
                or name.startswith("utils.dataset_postprocess.")
                for name in _imported_modules(path)
            ):
                violations.append(str(path.relative_to(REPO_ROOT)))
    assert sorted(set(violations)) == []


def test_legacy_data_paths_reexport_the_shared_contract_objects() -> None:
    from sim.mujoco.replay.dataset_writer import \
        DepthRangeMaskConfig as SimWriterDepthConfig
    from teleop_runtime.collection import schema_v21
    from teleop_runtime.collection.stats import \
        EpisodeStatsAccumulator as RuntimeStats
    from teleop_runtime.collection.writer_v21 import \
        DepthRangeMaskConfig as WriterDepthConfig
    from teleop_runtime.replay.media import _ChunkedTactileStream
    from utils.dataset_postprocess.wrist_cam_adjust.depth import \
        DepthRangeMaskConfig as OfflineDepthConfig

    assert RuntimeStats is EpisodeStatsAccumulator
    assert WriterDepthConfig is DepthRangeMaskConfig
    assert SimWriterDepthConfig is DepthRangeMaskConfig
    assert OfflineDepthConfig is DepthRangeMaskConfig
    assert _ChunkedTactileStream is ChunkedTactileStream
    assert schema_v21.write_rows_to_parquet is parquet_contract.write_rows_to_parquet
    assert schema_v21._FLOAT32_LIST_COLUMNS is parquet_contract.FLOAT32_LIST_COLUMNS
    assert schema_v21._STRING_LIST_COLUMNS is parquet_contract.STRING_LIST_COLUMNS
    assert schema_v21._INT64_LIST_COLUMNS is parquet_contract.INT64_LIST_COLUMNS
    assert schema_v21._ARM_MODE_FLOAT64_COLUMNS is parquet_contract.ARM_MODE_FLOAT64_COLUMNS
    assert schema_v21._ARM_MODE_INT64_COLUMNS is parquet_contract.ARM_MODE_INT64_COLUMNS
    assert schema_v21._ARM_MODE_STRING_COLUMNS is parquet_contract.ARM_MODE_STRING_COLUMNS


def test_legacy_random_background_paths_reexport_dataset_owned_objects() -> None:
    from dataset_interface.random_background import \
        RandomBackgroundAugmentConfig as CanonicalAugmentConfig
    from dataset_interface.random_background.mask_io import \
        BitpackMaskReader as CanonicalMaskReader
    from utils.random_background import \
        RandomBackgroundAugmentConfig as LegacyAugmentConfig
    from utils.random_background.mask_io import \
        BitpackMaskReader as LegacyMaskReader

    assert LegacyAugmentConfig is CanonicalAugmentConfig
    assert LegacyMaskReader is CanonicalMaskReader


def test_collection_policy_io_layout_loader_reexports_dataset_interface_owner() -> None:
    path = REPO_ROOT / "teleop_runtime" / "collection" / "policy_io.py"
    tree = _parsed_tree(path)
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
    } == {"dataset_interface.policy_io_layout"}

    from dataset_interface.policy_io_layout import load_policy_io_layout as canonical
    from teleop_runtime.collection.policy_io import load_policy_io_layout as compatibility

    assert compatibility is canonical


def test_policy_io_layout_loader_callers_use_dataset_interface_owner() -> None:
    for relative_path in (
        "control_station/umi_application.py",
        "teleop_runtime/collection/policy_io_runtime.py",
        "teleop_runtime/collection/umi/cli.py",
    ):
        modules = _imported_modules(REPO_ROOT / relative_path)
        assert "dataset_interface.policy_io_layout" in modules
        assert "teleop_runtime.collection.policy_io" not in modules


def test_shared_manifest_does_not_embed_an_offline_implementation_path() -> None:
    payload = mask_manifest_payload(
        camera_key="left_wrist",
        relative_dir="wrist_depth/mask/left_wrist",
        stream_path="mask.bitpack.bin",
        frame_count=0,
        height=2,
        width=2,
        records=[],
        config=DepthRangeMaskConfig(),
    )
    note = str(payload["semantics"]["postprocess_note"])
    assert "utils.dataset_postprocess" not in note


def test_scene_layout_legacy_path_reexports_robot_motion_contract() -> None:
    from robot_motion.scene_layout import rm65_base_y_for_side as canonical
    from teleop_runtime.scene_layout import rm65_base_y_for_side as legacy

    assert legacy is canonical


def test_rm65_support_package_is_archived() -> None:
    assert _python_files(REPO_ROOT / "rm65_inspire_tac") == []


@pytest.mark.integration
def test_data_packages_import_without_loading_runtime_or_hardware() -> None:
    code = """
import sys
import dataset_contracts
import utils.dataset_postprocess

forbidden = ("teleop_runtime", "robot_motion", "hardware", "rm65_inspire_tac")
loaded = sorted(
    name for name in sys.modules
    if name in forbidden or name.startswith(tuple(prefix + "." for prefix in forbidden))
)
assert loaded == [], loaded
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

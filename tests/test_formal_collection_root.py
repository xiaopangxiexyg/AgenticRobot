from __future__ import annotations

import json
from pathlib import Path

import project_paths
from teleop_runtime.collection.umi.cli import (
    DEFAULT_UMI_FORMAL_DATASET_ROOT,
    apply_dataset_root_selection,
    build_parser,
)
from teleop_runtime.collection.umi.config import load_umi_collection_config


EXPECTED_FORMAL_COLLECTION_ROOT = Path(
    "/home/user/zc/datasets/RealmanDMTAC/FormalCollection"
)


def test_formal_collection_default_preserves_local_override(monkeypatch) -> None:
    monkeypatch.setattr(project_paths, "local_config", lambda: {})
    assert project_paths.DEFAULT_FORMAL_COLLECTION_ROOT == EXPECTED_FORMAL_COLLECTION_ROOT
    assert project_paths.default_dataset_root() == EXPECTED_FORMAL_COLLECTION_ROOT
    assert project_paths.real_robot_dataset_root() == EXPECTED_FORMAL_COLLECTION_ROOT

    override = Path("/tmp/dexteleop-formal-collection-override")
    monkeypatch.setattr(
        project_paths,
        "local_config",
        lambda: {
            "dataset_root": str(override),
            "real_robot_dataset_root": str(override),
        },
    )
    assert project_paths.default_dataset_root() == override
    assert project_paths.real_robot_dataset_root() == override


def test_default_umi_launch_uses_dedicated_dataset_below_formal_root() -> None:
    assert build_parser().parse_args([]).config is None
    config = load_umi_collection_config(
        "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    selected = apply_dataset_root_selection(
        config,
        default_profile_implicit=True,
        dataset_root=None,
    )
    assert DEFAULT_UMI_FORMAL_DATASET_ROOT == (
        EXPECTED_FORMAL_COLLECTION_ROOT / "umi_dual_wrist"
    )
    assert selected.dataset_root.resolve() == DEFAULT_UMI_FORMAL_DATASET_ROOT


def test_explicit_umi_profile_root_is_preserved_unless_explicitly_overridden(
    tmp_path,
) -> None:
    config = load_umi_collection_config(
        "teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json"
    )
    preserved = apply_dataset_root_selection(
        config,
        default_profile_implicit=False,
        dataset_root=None,
    )
    assert preserved is config

    override = tmp_path / "explicit_dataset"
    selected = apply_dataset_root_selection(
        config,
        default_profile_implicit=False,
        dataset_root=override,
    )
    assert selected.dataset_root == override


def test_legacy_umi_profile_without_dataset_root_keeps_original_fallback(
    tmp_path,
) -> None:
    source = Path("teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("dataset_root", None)
    profile = tmp_path / "legacy_without_dataset_root.json"
    profile.write_text(json.dumps(payload), encoding="utf-8")

    config = load_umi_collection_config(profile)
    assert config.dataset_root.resolve() == (tmp_path / "datasets/umi").resolve()

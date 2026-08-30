from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.dataset_postprocess.wrist_cam_adjust import (
    stage1_final_mask_pipeline as pipeline,
)


def _write_namespace(root: Path, name: str) -> Path:
    namespace = root / name
    for camera in ("left_wrist", "right_wrist"):
        manifest = namespace / "episode_000001" / camera / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
    return namespace


def _migration_fixture(tmp_path: Path) -> dict[str, object]:
    dataset_root = tmp_path / "transport-view"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text(
        json.dumps({"dataset_name": "canonical-dataset"}),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    source_receipt_path = tmp_path / "source-receipt.json"
    source_receipt = {
        "intermediate_retention": {
            "postprocessed_depth_keep": "retained",
        }
    }
    source_receipt_path.write_text(
        json.dumps(source_receipt),
        encoding="utf-8",
    )
    post_root = tmp_path / "post" / "source-id"
    final_root = tmp_path / "final" / "source-id"
    post_legacy = _write_namespace(post_root, dataset_root.name)
    final_legacy = _write_namespace(final_root, dataset_root.name)
    return {
        "dataset_root": dataset_root,
        "profile_path": profile_path,
        "source_receipt_path": source_receipt_path,
        "source_receipt": source_receipt,
        "post_root": post_root,
        "post_legacy": post_legacy,
        "final_root": final_root,
        "final_legacy": final_legacy,
        "receipt_path": tmp_path / "layout-migration.json",
    }


def _migrate(values: dict[str, object]) -> dict[str, object]:
    summary = {
        "episode_count": 1,
        "camera_count": 2,
        "frame_count": 3,
        "camera_frame_count": 6,
        "stream_bytes": 12,
    }
    return pipeline._migrate_final_mask_namespace(
        receipt_path=values["receipt_path"],
        profile_path=values["profile_path"],
        binding_hash="binding-hash",
        source_id="source-id",
        dataset_root=values["dataset_root"],
        source_receipt_path=values["source_receipt_path"],
        source_receipt=values["source_receipt"],
        canonical_dataset_name="canonical-dataset",
        post_root=values["post_root"],
        final_root=values["final_root"],
        postprocessed_summary=summary,
        final_summary=summary,
    )


def test_canonical_dataset_name_is_fail_closed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "view"
    (dataset_root / "meta").mkdir(parents=True)
    info_path = dataset_root / "meta/info.json"
    info_path.write_text(
        json.dumps({"dataset_name": "canonical-dataset"}),
        encoding="utf-8",
    )
    assert pipeline._canonical_dataset_name(dataset_root) == "canonical-dataset"

    for unsafe in ("", ".", "..", "../escape", "a/b", "a\\b", " padded "):
        info_path.write_text(
            json.dumps({"dataset_name": unsafe}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="safe path component"):
            pipeline._canonical_dataset_name(dataset_root)


def test_final_mask_namespace_migration_preserves_inode_and_is_idempotent(
    tmp_path: Path,
) -> None:
    values = _migration_fixture(tmp_path)
    legacy = values["final_legacy"]
    before = legacy.stat()

    receipt = _migrate(values)

    canonical = values["final_root"] / "canonical-dataset"
    assert not legacy.exists()
    assert canonical.is_dir()
    assert canonical.stat().st_dev == before.st_dev
    assert canonical.stat().st_ino == before.st_ino
    assert values["post_legacy"].is_dir()
    assert receipt["content_policy"] == {
        "mask_stream_bytes_modified": False,
        "manifest_bytes_modified": False,
        "manifest_paths_remain_generation_time_provenance": True,
        "stream_resolution": "relative_to_manifest_location",
    }
    assert receipt["rollback"]["from"] == str(canonical)
    assert receipt["rollback"]["to"] == str(legacy)
    assert _migrate(values) == receipt


def test_final_mask_namespace_migration_refuses_target_collision(
    tmp_path: Path,
) -> None:
    values = _migration_fixture(tmp_path)
    _write_namespace(values["final_root"], "canonical-dataset")

    with pytest.raises(FileExistsError, match="without a matching"):
        _migrate(values)
    assert values["final_legacy"].is_dir()


def test_final_mask_namespace_migration_rolls_back_receipt_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _migration_fixture(tmp_path)
    legacy = values["final_legacy"]
    before = legacy.stat()

    def fail_receipt(_path: Path, _payload: object) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(pipeline, "_write_receipt", fail_receipt)
    with pytest.raises(OSError, match="injected"):
        _migrate(values)

    canonical = values["final_root"] / "canonical-dataset"
    assert legacy.is_dir()
    assert not canonical.exists()
    assert legacy.stat().st_dev == before.st_dev
    assert legacy.stat().st_ino == before.st_ino

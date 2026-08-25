from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest import mock

import pytest

from collection_runtime import modality_storage as storage
from collection_runtime.tactile_recompression import (
    recompress_tactile_zstd_in_place,
)
from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
from dataset_interface.episode_curation import source_dataset_base_fingerprint
from dataset_interface.modality_artifacts import (
    DETACHED_VERIFIED,
    SELF_CONTAINED,
    DatasetCaptureWritableError,
    assert_dataset_capture_writable,
    build_modality_artifact_inventory,
    resolve_modality_availability,
)
from dataset_interface.modality_artifacts import admission as capture_admission
from utils.dataset_postprocess.cli import build_parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _dataset(root: Path) -> Path:
    _write_json(root / "meta/info.json", {"episode_count": 1, "task": "unit"})
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "unit"}) + "\n",
        encoding="utf-8",
    )
    _write_json(root / "meta/modality_validity.json", {"tactile": "valid"})
    manifest_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/manifest.json"
    )
    payload_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L_RAW.uint8.bin"
    )
    payload = bytes(range(24))
    payload_path = root / payload_relative
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    _write_json(
        root / manifest_relative,
        {
            "format": "binary_stream_v1",
            "side": "left",
            "frame_count": 4,
            "streams": {
                "L_RAW": {
                    "path": payload_relative,
                    "dtype": "uint8",
                    "shape": [2, 3],
                    "frame_count": 4,
                    "encoding": "raw_c_order",
                    "compression": "none",
                }
            },
            "frames": [
                {"sample_frame_index": index, "tactile_frame_index": index}
                for index in range(4)
            ],
        },
    )
    (root / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 4,
                "tactile_paths": {"left": manifest_relative},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _multi_group_dataset(root: Path, *, episode_count: int = 10) -> Path:
    _write_json(root / "meta/info.json", {"episode_count": episode_count, "task": "unit"})
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "unit"}) + "\n",
        encoding="utf-8",
    )
    _write_json(root / "meta/modality_validity.json", {"tactile": "valid"})
    episodes: list[str] = []
    for episode_index in range(episode_count):
        tactile_paths: dict[str, str] = {}
        for side in ("left", "right"):
            directory = (
                f"tactile/observation.hand.{side}.tactile/chunk-000/"
                f"episode_{episode_index:06d}"
            )
            manifest_relative = f"{directory}/manifest.json"
            sensor = f"{side[0].upper()}_{episode_index:03d}"
            payload_relative = f"{directory}/{sensor}.uint8.bin"
            frame_count = 8 if episode_index == episode_count // 2 else 4
            payload = bytes(
                value % 256 for value in range(frame_count * 6)
            )
            payload_path = root / payload_relative
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(payload)
            _write_json(
                root / manifest_relative,
                {
                    "format": "binary_stream_v1",
                    "side": side,
                    "frame_count": frame_count,
                    "streams": {
                        sensor: {
                            "path": payload_relative,
                            "dtype": "uint8",
                            "shape": [2, 3],
                            "frame_count": frame_count,
                            "encoding": "raw_c_order",
                            "compression": "none",
                        }
                    },
                    "frames": [
                        {
                            "sample_frame_index": index,
                            "tactile_frame_index": index,
                        }
                        for index in range(frame_count)
                    ],
                },
            )
            tactile_paths[side] = manifest_relative
        episodes.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "frame_count": 4,
                    "tactile_paths": tactile_paths,
                },
                sort_keys=True,
            )
        )
    (root / "meta/episodes.jsonl").write_text(
        "\n".join(episodes) + "\n",
        encoding="utf-8",
    )
    return root


def _artifact_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "tactile").rglob("*"))
        if path.is_file()
    }


def test_fsync_retries_transient_network_filesystem_error() -> None:
    transient = BlockingIOError(errno.EAGAIN, "try again")
    with (
        mock.patch.object(storage.os, "fsync", side_effect=[transient, None]) as fsync,
        mock.patch.object(storage.time, "sleep") as sleep,
    ):
        storage._fsync_with_retry(17)

    assert fsync.call_count == 2
    sleep.assert_called_once_with(storage.FSYNC_INITIAL_RETRY_DELAY_SEC)


def test_fsync_does_not_retry_non_transient_error() -> None:
    permanent = OSError(errno.EIO, "I/O error")
    with (
        mock.patch.object(storage.os, "fsync", side_effect=permanent) as fsync,
        mock.patch.object(storage.time, "sleep") as sleep,
        pytest.raises(OSError, match="I/O error"),
    ):
        storage._fsync_with_retry(17)

    fsync.assert_called_once_with(17)
    sleep.assert_not_called()


def test_replace_retries_transient_cifs_busy_error() -> None:
    transient = OSError(errno.EBUSY, "device or resource busy")
    source = Path("partial")
    target = Path("complete")
    with (
        mock.patch.object(storage.os, "replace", side_effect=[transient, None]) as replace,
        mock.patch.object(storage.time, "sleep") as sleep,
    ):
        storage._replace_with_retry(source, target)

    assert replace.call_args_list == [mock.call(source, target), mock.call(source, target)]
    sleep.assert_called_once_with(storage.REPLACE_INITIAL_RETRY_DELAY_SEC)


def test_replace_does_not_retry_non_transient_error() -> None:
    permanent = OSError(errno.EIO, "I/O error")
    source = Path("partial")
    target = Path("complete")
    with (
        mock.patch.object(storage.os, "replace", side_effect=permanent) as replace,
        mock.patch.object(storage.time, "sleep") as sleep,
        pytest.raises(OSError, match="I/O error"),
    ):
        storage._replace_with_retry(source, target)

    replace.assert_called_once_with(source, target)
    sleep.assert_not_called()


def test_cold_staging_namespace_is_path_safe_and_preserves_default_names() -> None:
    artifact_root = Path("/cold/fingerprint/inventory")

    assert storage._cold_partial_root(artifact_root) == Path(
        "/cold/fingerprint/.inventory.partial"
    )
    assert storage._cold_partial_root(
        artifact_root,
        staging_namespace="cifs-recovery-20260809",
    ) == Path("/cold/fingerprint/.inventory.partial.cifs-recovery-20260809")
    with pytest.raises(storage.ModalityStorageError, match="path-safe ASCII"):
        storage._cold_partial_root(
            artifact_root,
            staging_namespace="../escape",
        )


@pytest.fixture
def offline_validation() -> object:
    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(),
    ):
        yield


def _detach(
    root: Path,
    cold: Path,
    *,
    detachment_scope: str = "payload_only",
) -> storage.ModalityStorageResult:
    plan = storage.plan_modality_detach(
        root,
        cold,
        detachment_scope=detachment_scope,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert plan.ready, plan.errors
    return storage.detach_modality(
        root,
        cold,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        detachment_scope=detachment_scope,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )


def test_export_preserves_hot_files_and_accepts_read_only_hardlinked_source(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    payload = next(
        path
        for path in (root / "tactile").rglob("*.bin")
        if path.is_file()
    )
    alias = tmp_path / "historical-hardlink.bin"
    os.link(payload, alias)
    before = _artifact_snapshot(root)
    source_stat = payload.stat()

    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert plan.ready, plan.errors
    result = storage.export_modality(
        root,
        cold,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert result.operation == "export"
    assert result.final_status == SELF_CONTAINED
    assert result.released_hot_bytes == 0
    assert _artifact_snapshot(root) == before
    assert payload.stat().st_ino == source_stat.st_ino
    assert payload.stat().st_nlink == source_stat.st_nlink
    inventory = build_modality_artifact_inventory(root, require_single_link=False)
    artifact_root = Path(result.cold_artifact_root)
    storage.verify_modality_artifact_inventory(
        root,
        inventory,
        artifact_root=artifact_root,
    )
    assert (artifact_root / payload.relative_to(root)).stat().st_ino != payload.stat().st_ino
    assert (artifact_root / "export_receipt.json").is_file()
    assert not (root / "meta/modality_artifacts").exists()

    with pytest.raises(storage.ModalityStorageError, match="link_count=1"):
        storage.plan_modality_detach(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
        )


def test_namespaced_export_preserves_existing_default_partial_tree(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root, require_single_link=False)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    default_stage = storage._cold_partial_root(artifact_root)
    storage._prepare_partial_export_stage(default_stage, inventory)
    payload = next(item for item in inventory.artifacts if item.role == "payload")
    old_partial = storage._partial_file_path(default_stage / payload.logical_path)
    old_partial.parent.mkdir(parents=True)
    old_partial.write_bytes((root / payload.logical_path).read_bytes()[:11])
    old_partial_before = old_partial.read_bytes()

    namespace = "cifs-recovery-20260809"
    with mock.patch.object(
        storage.shutil,
        "disk_usage",
        return_value=SimpleNamespace(free=inventory.total_bytes - 1),
    ):
        default_plan = storage.plan_modality_export(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
        )
        recovery_capacity_plan = storage.plan_modality_export(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
            cold_staging_namespace=namespace,
        )
    assert default_plan.ready, default_plan.errors
    assert not recovery_capacity_plan.ready
    assert "target filesystem has insufficient free space" in (
        recovery_capacity_plan.errors
    )

    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        cold_staging_namespace=namespace,
    )
    assert plan.ready, plan.errors
    result = storage.export_modality(
        root,
        cold,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
        cold_staging_namespace=namespace,
    )

    assert Path(result.cold_artifact_root).is_dir()
    assert default_stage.is_dir()
    assert old_partial.read_bytes() == old_partial_before
    assert not storage._cold_partial_root(
        artifact_root,
        staging_namespace=namespace,
    ).exists()
    assert (artifact_root / payload.logical_path).read_bytes() == (
        root / payload.logical_path
    ).read_bytes()
    assert (artifact_root / "export_receipt.json").is_file()
    assert not (root / "meta/modality_artifacts").exists()


def test_legacy_export_is_explicit_and_still_rejects_invalid_tactile_payload(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "legacy-dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    strict = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    legacy = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_source=True,
    )
    assert strict.ready, strict.errors
    assert strict.source_validation_mode == "tactile_inventory"
    assert legacy.ready, legacy.errors
    assert legacy.source_validation_mode == "legacy_inventory"
    assert strict.plan_hash != legacy.plan_hash

    payload = next((root / "tactile").rglob("*.bin"))
    with payload.open("ab") as handle:
        handle.write(b"corrupt trailing byte")
    invalid = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_source=True,
    )
    assert not invalid.ready
    assert any("raw tactile byte count mismatch" in error for error in invalid.errors)


def test_detach_and_rehydrate_round_trip_preserves_every_artifact_byte(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)

    detached = _detach(root, cold)

    inventory = storage.load_modality_inventory(root, detached.inventory_hash)
    assert detached.final_status == DETACHED_VERIFIED
    assert resolve_modality_availability(root, inventory).status == DETACHED_VERIFIED
    assert _artifact_snapshot(root) == {
        path: payload for path, payload in before.items() if path.endswith("manifest.json")
    }
    status = storage.modality_storage_status(
        root,
        inventory_hash=detached.inventory_hash,
    )
    assert status["production_detached_verified"] is True
    assert status["core_training_eligible"] is True
    assert status["tactile_read_eligible"] is False
    assert set(status["receipt_operations"]) == {
        "export_shadow",
        "restore_drill",
        "retire_hot_payloads",
    }
    with pytest.raises(DatasetCaptureWritableError, match="not physically self-contained"):
        assert_dataset_capture_writable(root)

    plan = storage.plan_modality_rehydrate(
        root,
        cold,
        detached.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert plan.ready, plan.errors
    restored = storage.rehydrate_modality(
        root,
        cold,
        detached.inventory_hash,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert restored.final_status == SELF_CONTAINED
    assert _artifact_snapshot(root) == before
    assert Path(detached.cold_artifact_root).is_dir()
    manually_moved = tmp_path / "manually-moved-tactile"
    os.replace(root / "tactile", manually_moved)
    after_manual_move = storage.modality_storage_status(
        root,
        inventory_hash=inventory.semantic_content_hash,
    )
    assert after_manual_move["status"] == "INCOMPLETE_OR_CORRUPT"
    assert after_manual_move["production_detached_verified"] is False
    os.replace(manually_moved, root / "tactile")
    assert resolve_modality_availability(root, inventory).status == SELF_CONTAINED
    assert_dataset_capture_writable(root)


def test_whole_prefix_detach_and_rehydrate_moves_the_complete_tactile_tree(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)

    detached = _detach(
        root,
        cold,
        detachment_scope="whole_prefix",
    )

    inventory = storage.load_modality_inventory(root, detached.inventory_hash)
    assert not (root / "tactile").exists()
    assert detached.released_hot_bytes == inventory.total_bytes
    assert resolve_modality_availability(
        root,
        inventory,
        detachment_scope="whole_prefix",
    ).status == DETACHED_VERIFIED
    status = storage.modality_storage_status(root)
    assert status["status"] == DETACHED_VERIFIED
    assert status["detachment_scope"] == "whole_prefix"
    assert status["production_detached_verified"] is True
    assert set(status["receipt_operations"]) == {
        "export_shadow",
        "restore_drill",
        "retire_hot_module_tree",
    }
    with pytest.raises(DatasetCaptureWritableError):
        assert_dataset_capture_writable(root)

    plan = storage.plan_modality_rehydrate(
        root,
        cold,
        detached.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert plan.ready, plan.errors
    assert plan.detachment_scope == "whole_prefix"
    restored = storage.rehydrate_modality(
        root,
        cold,
        detached.inventory_hash,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert restored.final_status == SELF_CONTAINED
    assert _artifact_snapshot(root) == before
    assert Path(detached.cold_artifact_root).is_dir()


def test_whole_prefix_smoke_keeps_visual_numeric_loadable_and_restores_tactile(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    parquet = root / "data/chunk-000/episode_000000.parquet"
    video = root / "videos/front/chunk-000/episode_000000.mp4"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"numeric-parquet-smoke")
    video.write_bytes(b"visual-mp4-smoke")
    catalog_path = root / "meta/episodes.jsonl"
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    record["parquet_path"] = parquet.relative_to(root).as_posix()
    record["video_paths"] = {"front": video.relative_to(root).as_posix()}
    catalog_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    tactile_before = _artifact_snapshot(root)
    catalog_before = catalog_path.read_bytes()
    parquet_before = parquet.read_bytes()
    video_before = video.read_bytes()
    cold = tmp_path / "cold"
    cold.mkdir()

    detached = _detach(root, cold, detachment_scope="whole_prefix")
    detached_record = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog_path.read_bytes() == catalog_before
    assert (root / detached_record["parquet_path"]).read_bytes() == parquet_before
    assert (
        root / detached_record["video_paths"]["front"]
    ).read_bytes() == video_before
    assert not (root / "tactile").exists()

    plan = storage.plan_modality_rehydrate(
        root,
        cold,
        detached.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    storage.rehydrate_modality(
        root,
        cold,
        detached.inventory_hash,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert _artifact_snapshot(root) == tactile_before
    assert catalog_path.read_bytes() == catalog_before
    assert parquet.read_bytes() == parquet_before
    assert video.read_bytes() == video_before


def test_capture_admission_allows_blank_and_rejects_unreceipted_missing_tactile(
    tmp_path: Path,
) -> None:
    assert_dataset_capture_writable(tmp_path / "blank")
    root = _dataset(tmp_path / "manual-missing")
    inventory = build_modality_artifact_inventory(root)
    payload = next(
        item for item in inventory.artifacts if item.role == "payload"
    )
    (root / payload.logical_path).unlink()

    with pytest.raises(
        DatasetCaptureWritableError,
        match="no complete current hot artifact graph",
    ):
        assert_dataset_capture_writable(root)

    root = _dataset(tmp_path / "orphan-active-episode")
    (root / "tactile/orphan-active-episode.bin").write_bytes(b"partial")
    with pytest.raises(DatasetCaptureWritableError, match="unowned files"):
        assert_dataset_capture_writable(root)


def test_capture_admission_filters_historical_inventories_without_payload_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset(tmp_path / "tracked-history")
    current = build_modality_artifact_inventory(root)
    catalog = root / "meta/episodes.jsonl"
    historical = ModalityArtifactInventoryV1(
        modality_key="tactile",
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(root),
        source_episode_catalog_sha256=hashlib.sha256(
            catalog.read_bytes()
        ).hexdigest(),
        source_episode_count=1,
        logical_prefixes=("tactile",),
        artifacts=current.artifacts,
    )
    inventory_dir = root / "meta/modality_artifacts/tactile/inventories"
    _write_json(
        inventory_dir / f"{historical.semantic_content_hash}.json",
        historical.to_dict(),
    )

    # The new scoped inventory may coexist with a legacy whole-dataset-bound
    # inventory for the exact same tactile graph. An unrelated catalog change
    # invalidates only the legacy fast path; admission safely coalesces them.
    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["video_paths"] = {"front": "videos/front/repacked.mp4"}
    catalog.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(
        inventory_dir / f"{current.semantic_content_hash}.json",
        current.to_dict(),
    )

    hashed_paths: list[str] = []
    original_sha256_file = capture_admission._sha256_file

    def traced_sha256_file(path: Path) -> str:
        hashed_paths.append(Path(path).relative_to(root).as_posix())
        return original_sha256_file(path)

    monkeypatch.setattr(capture_admission, "_sha256_file", traced_sha256_file)
    capture_admission.assert_dataset_capture_writable(root)

    payload_relative = next(
        item.logical_path for item in current.artifacts if item.role == "payload"
    )
    manifest_relative = next(
        item.logical_path for item in current.artifacts if item.role == "manifest"
    )
    assert "meta/episodes.jsonl" not in hashed_paths
    assert manifest_relative in hashed_paths
    assert payload_relative not in hashed_paths


def test_wrong_confirmation_and_tactile_change_cannot_reach_retirement(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    with pytest.raises(storage.ModalityStorageError, match="plan hash changed"):
        storage.detach_modality(
            root,
            cold,
            expected_plan_hash="0" * 64,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
        )
    assert _artifact_snapshot(root) == before
    assert not (root / "meta/modality_artifacts/tactile/inventories").exists()

    manifest = next((root / "tactile").rglob("manifest.json"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["tactile_revision"] = 2
    _write_json(manifest, manifest_payload)
    changed_before_apply = _artifact_snapshot(root)
    with pytest.raises(storage.ModalityStorageError, match="plan hash changed"):
        storage.detach_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
        )
    assert _artifact_snapshot(root) == changed_before_apply


def test_missing_reader_confirmation_is_zero_write_rejection(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    with pytest.raises(
        storage.ModalityStorageError,
        match="no dataset readers are active",
    ):
        storage.detach_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
        )

    assert _artifact_snapshot(root) == before
    assert not (root / "meta/modality_artifacts").exists()
    assert not (root.parent / f".{root.name}.modality-maintenance.lock").exists()


def test_production_plan_rejects_cold_root_on_same_filesystem(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()

    plan = storage.plan_modality_detach(root, cold, min_free_bytes=0)

    assert plan.ready is False
    assert plan.source_device == plan.target_device
    assert "cold storage must be on a distinct filesystem/device" in plan.errors


def test_plan_and_managed_move_reject_hardlinked_source_artifacts(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    payload = next(item for item in inventory.artifacts if item.role == "payload")
    source = root / payload.logical_path
    sibling_link = tmp_path / "derived-dataset-shared-payload.bin"
    os.link(source, sibling_link)
    before = source.read_bytes()

    with pytest.raises(storage.ModalityStorageError, match="link_count=1"):
        storage.plan_modality_detach(
            root,
            cold,
            detachment_scope="whole-prefix",
            require_distinct_device=False,
            min_free_bytes=0,
        )
    with pytest.raises(storage.ModalityStorageError, match="link_count=1"):
        storage.move_modality(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
        )
    assert source.read_bytes() == before
    assert sibling_link.read_bytes() == before
    assert int(source.stat().st_nlink) == 2
    assert not (root / "meta/modality_artifacts").exists()


def test_explicit_legacy_hardlink_detach_preserves_sibling_and_is_rehydratable(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    payload = next((root / "tactile").rglob("*.bin"))
    sibling_link = tmp_path / "derived-dataset-shared-payload.bin"
    os.link(payload, sibling_link)
    expected = payload.read_bytes()
    manifest = next((root / "tactile").rglob("manifest.json"))
    manifest_relative = manifest.relative_to(root)
    manifest_sibling = tmp_path / "derived-dataset-shared-manifest.json"
    os.link(manifest, manifest_sibling)
    expected_manifest = manifest.read_bytes()

    with pytest.raises(storage.ModalityStorageError, match="link_count=1"):
        storage.plan_modality_detach(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
        )

    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_hardlinked_source=True,
    )
    assert plan.ready, plan.errors
    assert plan.source_validation_mode == "legacy_inventory"
    result = storage.detach_modality(
        root,
        cold,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
        allow_legacy_hardlinked_source=True,
    )

    assert result.final_status == DETACHED_VERIFIED
    assert not payload.exists()
    assert sibling_link.read_bytes() == expected
    assert sibling_link.stat().st_nlink == 1
    retained_manifest = root / manifest_relative
    assert retained_manifest.read_bytes() == expected_manifest
    assert retained_manifest.stat().st_nlink == 1
    assert manifest_sibling.read_bytes() == expected_manifest
    assert manifest_sibling.stat().st_nlink == 1
    assert retained_manifest.stat().st_ino != manifest_sibling.stat().st_ino

    rehydrate = storage.plan_modality_rehydrate(
        root,
        cold,
        result.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    assert rehydrate.ready, rehydrate.errors
    restored = storage.rehydrate_modality(
        root,
        cold,
        result.inventory_hash,
        expected_plan_hash=rehydrate.plan_hash,
        confirm_dataset_fingerprint=rehydrate.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    assert restored.final_status == SELF_CONTAINED
    assert payload.read_bytes() == expected
    assert payload.stat().st_ino != sibling_link.stat().st_ino


def test_detach_plans_use_tactile_reader_validation_only(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()

    legacy = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_hardlinked_source=True,
    )
    strict = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    assert legacy.ready, legacy.errors
    assert legacy.source_validation_mode == "legacy_inventory"
    assert strict.ready, strict.errors
    assert strict.source_validation_mode == "tactile_inventory"


def test_sampled_revalidation_is_explicit_deterministic_and_side_balanced(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _multi_group_dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)

    first = storage._sampled_manifest_groups(inventory)
    second = storage._sampled_manifest_groups(inventory)
    manifest_paths = {
        item.logical_path
        for group in first
        for item in group
        if item.role == "manifest"
    }

    assert first == second
    assert len(first) == storage.SAMPLED_REVALIDATION_MANIFEST_COUNT
    assert any("hand.left" in path for path in manifest_paths)
    assert any("hand.right" in path for path in manifest_paths)
    assert any("episode_000000" in path for path in manifest_paths)
    assert any("episode_000009" in path for path in manifest_paths)

    full = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    sampled = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        revalidation_profile=storage.SAMPLED_REVALIDATION_V1,
    )
    assert full.revalidation_profile == storage.FULL_REVALIDATION
    assert full.revalidation_manifest_count == 20
    assert sampled.revalidation_profile == storage.SAMPLED_REVALIDATION_V1
    assert sampled.revalidation_manifest_count == 16
    assert "revalidation_profile" not in full.semantic_payload()
    assert sampled.semantic_payload()["revalidation_profile"] == "sampled_v1"
    assert sampled.plan_hash != full.plan_hash


def test_sampled_detach_keeps_exact_copy_and_marks_sampled_receipts(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _multi_group_dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        revalidation_profile=storage.SAMPLED_REVALIDATION_V1,
    )

    result = storage.detach_modality(
        root,
        cold,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
        revalidation_profile=storage.SAMPLED_REVALIDATION_V1,
    )

    assert result.final_status == DETACHED_VERIFIED
    assert any("restore_drill.sampled_v1" in path for path in result.receipt_paths)
    assert any("retire_hot_payloads.sampled_v1" in path for path in result.receipt_paths)
    inventory = storage.load_modality_inventory(root, result.inventory_hash)
    storage.verify_modality_artifact_inventory(
        root,
        inventory,
        artifact_root=Path(result.cold_artifact_root),
    )
    restored = {
        artifact.logical_path: (Path(result.cold_artifact_root) / artifact.logical_path).read_bytes()
        for artifact in inventory.artifacts
    }
    assert restored == before


def test_sampled_detach_aborts_when_source_stat_changes_after_export(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    payload = next((root / "tactile").rglob("*.bin"))
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        revalidation_profile=storage.SAMPLED_REVALIDATION_V1,
    )
    original_drill = storage._restore_drill

    def drill_then_touch(*args: object, **kwargs: object) -> Path:
        receipt = original_drill(*args, **kwargs)  # type: ignore[arg-type]
        stat = payload.stat()
        os.utime(
            payload,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )
        return receipt

    with (
        mock.patch.object(storage, "_restore_drill", side_effect=drill_then_touch),
        pytest.raises(storage.ModalityStorageError, match="changed after sampled"),
    ):
        storage.detach_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
            revalidation_profile=storage.SAMPLED_REVALIDATION_V1,
        )

    assert payload.exists()
    assert not any(
        path.name.startswith("retire_hot_payloads")
        for path in (root / "meta/modality_artifacts/tactile/receipts").glob("*.json")
    )


def test_legacy_scan_detach_binds_actual_bytes_when_sealed_hash_is_stale(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    payload = next((root / "tactile").rglob("*.bin"))
    sibling_link = tmp_path / "derived-dataset-shared-payload.bin"
    os.link(payload, sibling_link)
    expected = payload.read_bytes()
    manifest_path = next((root / "tactile").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stream = manifest["streams"]["L_RAW"]
    stream["stored_size_bytes"] = len(expected)
    stream["stored_sha256"] = "0" * 64
    stream["stored_hash_profile"] = "stored_bytes_sha256_v1"
    _write_json(manifest_path, manifest)

    stale_plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_hardlinked_source=True,
    )
    assert stale_plan.ready, stale_plan.errors
    with pytest.raises(
        storage.ModalityStorageError,
        match="hash differs from frozen inventory",
    ):
        storage.detach_modality(
            root,
            cold,
            expected_plan_hash=stale_plan.plan_hash,
            confirm_dataset_fingerprint=stale_plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
            allow_legacy_hardlinked_source=True,
        )
    assert payload.read_bytes() == expected
    assert sibling_link.read_bytes() == expected

    scan_plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_hardlinked_source=True,
        legacy_scan_payload_hashes=True,
    )
    assert scan_plan.ready, scan_plan.errors
    assert scan_plan.source_validation_mode == "legacy_scan_inventory"
    result = storage.detach_modality(
        root,
        cold,
        expected_plan_hash=scan_plan.plan_hash,
        confirm_dataset_fingerprint=scan_plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
        allow_legacy_hardlinked_source=True,
        legacy_scan_payload_hashes=True,
    )

    assert result.final_status == DETACHED_VERIFIED
    assert not payload.exists()
    assert sibling_link.read_bytes() == expected


def test_restore_drill_failure_keeps_hot_payload_and_never_retires(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    with mock.patch.object(
        storage,
        "_restore_drill",
        side_effect=ValueError("injected drill failure"),
    ):
        with pytest.raises(ValueError, match="injected drill failure"):
            storage.detach_modality(
                root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert _artifact_snapshot(root) == before
    inventory = build_modality_artifact_inventory(root)
    operations = {
        receipt.operation
        for receipt in storage._complete_receipts(
            root, inventory.semantic_content_hash
        )
    }
    assert operations == {"export_shadow"}


def test_unrelated_catalog_change_after_restore_drill_does_not_abort_retirement(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    original_drill = storage._restore_drill

    def drill_then_change_catalog(*args: object, **kwargs: object) -> Path:
        receipt = original_drill(*args, **kwargs)  # type: ignore[arg-type]
        catalog = root / "meta/episodes.jsonl"
        catalog.write_text(
            catalog.read_text(encoding="utf-8").rstrip() + " \n",
            encoding="utf-8",
        )
        return receipt

    with mock.patch.object(storage, "_restore_drill", side_effect=drill_then_change_catalog):
        result = storage.detach_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
        )

    assert result.final_status == DETACHED_VERIFIED


def test_tactile_manifest_change_after_restore_drill_aborts_before_retirement(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    original_drill = storage._restore_drill

    def drill_then_change_manifest(*args: object, **kwargs: object) -> Path:
        receipt = original_drill(*args, **kwargs)  # type: ignore[arg-type]
        manifest = next((root / "tactile").rglob("manifest.json"))
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["tactile_revision"] = 2
        _write_json(manifest, payload)
        return receipt

    with mock.patch.object(
        storage,
        "_restore_drill",
        side_effect=drill_then_change_manifest,
    ):
        with pytest.raises(storage.ModalityStorageError, match="changed after shadow"):
            storage.detach_modality(
                root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert (root / "tactile").is_dir()
    assert set(_artifact_snapshot(root)) == set(before)


def test_partial_cold_target_is_rejected_in_read_only_plan(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    artifact_root.mkdir(parents=True)
    storage._atomic_write_json(artifact_root / "inventory.json", inventory.to_dict())
    first = inventory.artifacts[0]
    target = artifact_root / first.logical_path
    target.parent.mkdir(parents=True)
    shutil.copyfile(root / first.logical_path, target)
    before = _artifact_snapshot(root)

    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    assert plan.ready is False
    assert any("cold artifact root is not reusable" in error for error in plan.errors)
    assert _artifact_snapshot(root) == before


@pytest.mark.parametrize("detachment_scope", ("payload_only", "whole_prefix"))
def test_retirement_swap_failure_rolls_back_to_self_contained(
    tmp_path: Path,
    offline_validation: object,
    detachment_scope: str,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    plan = storage.plan_modality_detach(
        root,
        cold,
        detachment_scope=detachment_scope,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    original_publish = storage._publish_receipt

    def fail_retire_receipt(
        dataset_root: Path,
        receipt: object,
    ) -> Path:
        if getattr(receipt, "operation", "") in {
            "retire_hot_payloads",
            "retire_hot_module_tree",
        }:
            raise OSError("injected receipt failure")
        return original_publish(dataset_root, receipt)  # type: ignore[arg-type]

    with mock.patch.object(storage, "_publish_receipt", side_effect=fail_retire_receipt):
        with pytest.raises(OSError, match="injected receipt failure"):
            storage.detach_modality(
                root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                detachment_scope=detachment_scope,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert _artifact_snapshot(root) == before
    inventory = build_modality_artifact_inventory(root)
    assert resolve_modality_availability(root, inventory).status == SELF_CONTAINED
    assert not storage._journal_path(root).exists()


@pytest.mark.parametrize("detachment_scope", ("payload_only", "whole_prefix"))
def test_rehydrate_validation_failure_rolls_back_to_detached(
    tmp_path: Path,
    offline_validation: object,
    detachment_scope: str,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    detached = _detach(root, cold, detachment_scope=detachment_scope)
    inventory = storage.load_modality_inventory(root, detached.inventory_hash)
    plan = storage.plan_modality_rehydrate(
        root,
        cold,
        detached.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    with mock.patch.object(
        storage,
        "_verify_tactile_reader",
        side_effect=storage.ModalityStorageError("injected tactile validation failure"),
    ):
        with pytest.raises(storage.ModalityStorageError, match="injected"):
            storage.rehydrate_modality(
                root,
                cold,
                detached.inventory_hash,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert resolve_modality_availability(
        root,
        inventory,
        detachment_scope=detachment_scope,
    ).status == DETACHED_VERIFIED
    assert (root / "tactile").exists() is (detachment_scope == "payload_only")
    assert not storage._journal_path(root).exists()


@pytest.mark.parametrize("detachment_scope", ("payload_only", "whole_prefix"))
def test_durable_retire_receipt_makes_post_commit_cleanup_recoverable(
    tmp_path: Path,
    offline_validation: object,
    detachment_scope: str,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    storage._publish_inventory(root, inventory)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    storage._export_shadow(
        root,
        artifact_root,
        inventory,
        require_distinct_device=False,
        source_storage_id="hot.local",
        target_storage_id="cold.archive",
    )
    storage._restore_drill(
        root,
        artifact_root,
        inventory,
        source_storage_id="cold.archive",
        target_storage_id="hot.local",
    )
    real_rmtree = shutil.rmtree
    failed = False

    def fail_first_quarantine(path: object, *args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed and str(path).endswith(".quarantine"):
            failed = True
            raise OSError("injected post-commit cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    with mock.patch.object(storage.shutil, "rmtree", side_effect=fail_first_quarantine):
        with pytest.raises(OSError, match="post-commit cleanup"):
            retire = (
                storage._retire_hot_module_tree
                if detachment_scope == "whole_prefix"
                else storage._retire_hot_payloads
            )
            retire(
                root,
                inventory,
                artifact_root=artifact_root,
                source_storage_id="hot.local",
                target_storage_id="cold.archive",
            )

    assert resolve_modality_availability(
        root,
        inventory,
        detachment_scope=detachment_scope,
    ).status == DETACHED_VERIFIED
    assert storage._journal_path(root).is_file()
    assert storage.recover_modality_storage_transaction(root) is True
    assert resolve_modality_availability(
        root,
        inventory,
        detachment_scope=detachment_scope,
    ).status == DETACHED_VERIFIED
    assert not storage._journal_path(root).exists()


def test_unreceipted_crash_journal_is_rolled_back_only_by_apply_recovery(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    before = _artifact_snapshot(root)
    inventory = build_modality_artifact_inventory(root)
    operation_id = "retire_hot_payloads.crashfixture"
    stage, quarantine = storage._swap_paths(root, operation_id)
    stage.mkdir()
    storage._copy_inventory_artifacts(
        root,
        stage,
        inventory,
        roles=frozenset({"manifest"}),
    )
    storage._write_journal(
        root,
        operation_id=operation_id,
        operation="retire_hot_payloads",
        inventory_hash=inventory.semantic_content_hash,
        stage=stage,
        quarantine=quarantine,
    )
    os.replace(root / "tactile", quarantine)
    os.replace(stage / "tactile", root / "tactile")

    status = storage.modality_storage_status(root)
    assert status["pending_transaction"] is True
    assert _artifact_snapshot(root) != before

    assert storage.recover_modality_storage_transaction(root) is True
    assert _artifact_snapshot(root) == before
    assert not storage._journal_path(root).exists()


def test_unreceipted_recovery_never_clears_journal_when_both_trees_are_missing(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)
    operation_id = "retire_hot_module_tree.missingfixture"
    stage, quarantine = storage._swap_paths(root, operation_id)
    stage.mkdir()
    storage._write_journal(
        root,
        operation_id=operation_id,
        operation="retire_hot_module_tree",
        inventory_hash=inventory.semantic_content_hash,
        stage=stage,
        quarantine=quarantine,
    )
    os.replace(root / "tactile", quarantine)
    emergency_backup = tmp_path / "emergency-backup"
    os.replace(quarantine, emergency_backup)

    with pytest.raises(storage.ModalityStorageError, match="both active and quarantine"):
        storage.recover_modality_storage_transaction(root)

    assert emergency_backup.is_dir()
    assert stage.is_dir()
    assert storage._journal_path(root).is_file()


def test_committed_retirement_recovery_keeps_quarantine_if_cold_is_unavailable(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    storage._publish_inventory(root, inventory)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    storage._export_shadow(
        root,
        artifact_root,
        inventory,
        require_distinct_device=False,
        source_storage_id="hot.local",
        target_storage_id="cold.archive",
    )
    storage._restore_drill(
        root,
        artifact_root,
        inventory,
        source_storage_id="cold.archive",
        target_storage_id="hot.local",
    )
    real_rmtree = shutil.rmtree
    failed = False

    def fail_first_quarantine(path: object, *args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed and str(path).endswith(".quarantine"):
            failed = True
            raise OSError("injected post-commit cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    with mock.patch.object(storage.shutil, "rmtree", side_effect=fail_first_quarantine):
        with pytest.raises(OSError, match="post-commit cleanup"):
            storage._retire_hot_module_tree(
                root,
                inventory,
                artifact_root=artifact_root,
                source_storage_id="hot.local",
                target_storage_id="cold.archive",
            )

    journal = storage._read_json(storage._journal_path(root))
    quarantine = Path(str(journal["quarantine"]))
    assert quarantine.is_dir()
    offline_artifact = artifact_root.with_name(f"{artifact_root.name}.offline")
    os.replace(artifact_root, offline_artifact)

    with pytest.raises(Exception):
        storage.recover_modality_storage_transaction(root)

    assert quarantine.is_dir()
    assert storage._journal_path(root).is_file()
    assert offline_artifact.is_dir()

    os.replace(offline_artifact, artifact_root)
    assert storage.recover_modality_storage_transaction(root) is True
    assert not quarantine.exists()
    assert not storage._journal_path(root).exists()


def test_corrupt_transaction_journal_cannot_target_sibling_directory(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    marker = sibling / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    journal = storage._journal_path(root)
    _write_json(
        journal,
        {
            "schema": "dexteleop.modality_storage_transaction.v1",
            "operation_id": "retire_hot_payloads.corruptfixture",
            "operation": "retire_hot_payloads",
            "inventory_hash": "0" * 64,
            "stage": str(sibling),
            "quarantine": str(tmp_path / "another-unrelated"),
        },
    )

    with pytest.raises(storage.ModalityStorageError, match="paths do not match"):
        storage.recover_modality_storage_transaction(root)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert journal.is_file()


def test_managed_move_resumes_verified_partial_file_without_recopy(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    stage = storage._cold_partial_root(artifact_root)
    stage.mkdir(parents=True)
    storage._atomic_write_json(stage / "inventory.json", inventory.to_dict())
    payload = next(item for item in inventory.artifacts if item.role == "payload")
    source = root / payload.logical_path
    source_before = source.read_bytes()
    target = stage / payload.logical_path
    partial = storage._partial_file_path(target)
    partial.parent.mkdir(parents=True)
    prefix = source_before[:11]
    partial.write_bytes(prefix)
    partial_inode = partial.stat().st_ino

    result = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    final_payload = Path(result.cold_artifact_root) / payload.logical_path
    assert result.final_status == DETACHED_VERIFIED
    assert not (root / "tactile").exists()
    assert final_payload.read_bytes() == source_before
    assert final_payload.stat().st_ino == partial_inode
    assert not stage.exists()


def test_managed_move_reuses_completed_files_after_interruption(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    original_copy = storage._copy_file_resumable
    copied: list[Path] = []

    def copy_then_interrupt(source: Path, target: Path, artifact: object) -> None:
        original_copy(source, target, artifact)
        copied.append(target)
        raise KeyboardInterrupt("injected transfer interruption")

    with mock.patch.object(
        storage,
        "_copy_file_resumable",
        side_effect=copy_then_interrupt,
    ):
        with pytest.raises(KeyboardInterrupt, match="injected"):
            storage.move_modality(
                root,
                cold,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert (root / "tactile").is_dir()
    assert len(copied) == 1
    completed_stage_file = copied[0]
    assert completed_stage_file.is_file()
    completed_inode = completed_stage_file.stat().st_ino
    relative = completed_stage_file.relative_to(
        storage._cold_partial_root(artifact_root)
    )

    result = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    final_completed_file = Path(result.cold_artifact_root) / relative
    assert result.final_status == DETACHED_VERIFIED
    assert final_completed_file.stat().st_ino == completed_inode
    assert not storage._cold_partial_root(artifact_root).exists()


def test_managed_move_discards_corrupt_partial_prefix_before_append(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    inventory = build_modality_artifact_inventory(root)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    stage = storage._cold_partial_root(artifact_root)
    stage.mkdir(parents=True)
    storage._atomic_write_json(stage / "inventory.json", inventory.to_dict())
    payload = next(item for item in inventory.artifacts if item.role == "payload")
    expected = (root / payload.logical_path).read_bytes()
    partial = storage._partial_file_path(stage / payload.logical_path)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"corrupt-prefix")

    result = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    final_payload = Path(result.cold_artifact_root) / payload.logical_path
    assert final_payload.read_bytes() == expected
    assert not (root / "tactile").exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd CLI is required")
def test_managed_move_round_trips_current_chunked_zstd_tactile_format(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    recompress_tactile_zstd_in_place(
        root,
        level=1,
        chunk_frames=2,
        min_free_bytes=0,
        delete_originals=True,
        backend="cli",
        enforce_global_collection_exclusion=False,
    )
    before = _artifact_snapshot(root)

    moved = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    inventory = storage.load_modality_inventory(root, moved.inventory_hash)
    storage._verify_tactile_reader(Path(moved.cold_artifact_root), inventory)
    plan = storage.plan_modality_rehydrate(
        root,
        cold,
        moved.inventory_hash,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    restored = storage.rehydrate_modality(
        root,
        cold,
        moved.inventory_hash,
        expected_plan_hash=plan.plan_hash,
        confirm_dataset_fingerprint=plan.dataset_fingerprint,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert restored.final_status == SELF_CONTAINED
    assert _artifact_snapshot(root) == before


def test_managed_move_rejects_apply_without_reader_confirmation(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()

    with pytest.raises(storage.ModalityStorageError, match="no dataset readers"):
        storage.move_modality(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
        )

    assert (root / "tactile").is_dir()


def test_managed_move_rejects_active_collection_without_touching_tactile(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=("pid=123 fixture",),
    ):
        with pytest.raises(storage.ModalityStorageError, match="formal collection is active"):
            storage.move_modality(
                root,
                cold,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert _artifact_snapshot(root) == before
    assert not (root / "meta/modality_artifacts").exists()
    assert tuple(cold.iterdir()) == ()


def test_managed_move_rechecks_collection_after_final_inventory_hash(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    process_snapshots = [(), (), (), (), ("pid=456 late fixture",)]

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        side_effect=process_snapshots,
    ):
        with pytest.raises(storage.ModalityStorageError, match="formal collection is active"):
            storage.move_modality(
                root,
                cold,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert _artifact_snapshot(root) == before
    inventory = build_modality_artifact_inventory(root)
    operations = {
        receipt.operation
        for receipt in storage._complete_receipts(
            root,
            inventory.semantic_content_hash,
        )
    }
    assert operations == {"export_shadow", "restore_drill"}


def test_managed_move_never_follows_symlinked_partial_artifact(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    inventory = build_modality_artifact_inventory(root)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    stage = storage._cold_partial_root(artifact_root)
    stage.mkdir(parents=True)
    storage._atomic_write_json(stage / "inventory.json", inventory.to_dict())
    payload = next(item for item in inventory.artifacts if item.role == "payload")
    partial = storage._partial_file_path(stage / payload.logical_path)
    partial.parent.mkdir(parents=True)
    partial.symlink_to(root / payload.logical_path)

    with pytest.raises(storage.ModalityStorageError, match="plan is not ready"):
        storage.move_modality(
            root,
            cold,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
        )

    assert _artifact_snapshot(root) == before
    assert partial.is_symlink()
    assert not artifact_root.exists()


def test_managed_move_recovers_unreceipted_retirement_before_replanning(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    inventory = build_modality_artifact_inventory(root)
    storage._publish_inventory(root, inventory)
    artifact_root = storage._cold_artifact_root(cold, inventory)
    storage._export_shadow(
        root,
        artifact_root,
        inventory,
        require_distinct_device=False,
        source_storage_id="hot.local",
        target_storage_id="cold.archive",
    )
    storage._restore_drill(
        root,
        artifact_root,
        inventory,
        source_storage_id="cold.archive",
        target_storage_id="hot.local",
    )
    operation_id = "retire_hot_module_tree.crashfixture"
    stage, quarantine = storage._swap_paths(root, operation_id)
    stage.mkdir()
    storage._write_journal(
        root,
        operation_id=operation_id,
        operation="retire_hot_module_tree",
        inventory_hash=inventory.semantic_content_hash,
        stage=stage,
        quarantine=quarantine,
        active_was_present=True,
    )
    os.replace(root / "tactile", quarantine)
    assert not (root / "tactile").exists()

    result = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert result.final_status == DETACHED_VERIFIED
    assert not (root / "tactile").exists()
    assert not storage._journal_path(root).exists()
    assert _artifact_snapshot(Path(result.cold_artifact_root)) == before


def test_managed_move_retry_after_committed_cleanup_failure_is_idempotent(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    before = _artifact_snapshot(root)
    real_rmtree = shutil.rmtree
    failed = False

    def fail_first_quarantine(path: object, *args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed and str(path).endswith(".quarantine"):
            failed = True
            raise OSError("injected post-commit cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    with mock.patch.object(storage.shutil, "rmtree", side_effect=fail_first_quarantine):
        with pytest.raises(OSError, match="post-commit cleanup"):
            storage.move_modality(
                root,
                cold,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert not (root / "tactile").exists()
    assert storage._journal_path(root).is_file()
    result = storage.move_modality(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert result.final_status == DETACHED_VERIFIED
    assert not storage._journal_path(root).exists()
    assert not (root / "tactile").exists()
    assert _artifact_snapshot(Path(result.cold_artifact_root)) == before
    assert set(
        storage.modality_storage_status(
            root,
            inventory_hash=result.inventory_hash,
        )["receipt_operations"]
    ) == {
        "export_shadow",
        "restore_drill",
        "retire_hot_module_tree",
    }


def test_managed_move_source_change_during_copy_keeps_hot_tree(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    payload = next(
        path
        for path in (root / "tactile").rglob("*")
        if path.is_file() and path.name.endswith(".bin")
    )
    original_payload = payload.read_bytes()
    real_assert = storage._assert_regular_single_link
    mutated = False

    def mutate_after_source_stat(path: Path, *, label: str) -> os.stat_result:
        nonlocal mutated
        result = real_assert(path, label=label)
        if label == "source artifact" and path == payload and not mutated:
            mutated = True
            path.write_bytes(bytes(reversed(original_payload)))
        return result

    with mock.patch.object(
        storage,
        "_assert_regular_single_link",
        side_effect=mutate_after_source_stat,
    ):
        with pytest.raises(storage.ModalityStorageError, match="source artifact"):
            storage.move_modality(
                root,
                cold,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
            )

    assert (root / "tactile").is_dir()
    assert payload.read_bytes() == bytes(reversed(original_payload))
    assert not any(
        receipt.operation == "retire_hot_module_tree"
        for receipt in storage._complete_receipts(
            root,
            build_modality_artifact_inventory(root).semantic_content_hash,
        )
    )


def test_export_can_explicitly_allow_disjoint_active_collection(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    root = _dataset(tmp_path / "dataset")
    active_root = tmp_path / "formal_collection"
    active_root.mkdir()
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(f"pid=123 python collector --dataset-root {active_root}",),
    ):
        result = storage.export_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
            allow_disjoint_active_collection=True,
        )

    assert result.final_status == SELF_CONTAINED


def test_detach_can_narrow_active_collection_base_to_verified_task_leaf(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    dataset_base = tmp_path / "datasets"
    root = _dataset(dataset_base / "archived_dataset")
    active_root = dataset_base / "2026-08-11_active_task"
    active_root.mkdir()
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_detach(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
        allow_legacy_hardlinked_source=True,
    )
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name active_task"
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        result = storage.detach_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
            allow_legacy_hardlinked_source=True,
            allow_disjoint_active_collection=True,
            active_collection_dataset_roots=(active_root,),
        )

    assert result.final_status == DETACHED_VERIFIED


def test_export_can_narrow_active_collection_base_to_verified_task_leaf(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    dataset_base = tmp_path / "datasets"
    root = _dataset(dataset_base / "archived_dataset")
    active_root = dataset_base / "2026-08-07_active_task"
    active_root.mkdir()
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name active_task"
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        result = storage.export_modality(
            root,
            cold,
            expected_plan_hash=plan.plan_hash,
            confirm_dataset_fingerprint=plan.dataset_fingerprint,
            require_distinct_device=False,
            min_free_bytes=0,
            confirm_no_active_readers=True,
            allow_disjoint_active_collection=True,
            active_collection_dataset_roots=(active_root,),
        )

    assert result.final_status == SELF_CONTAINED


def test_active_collection_dataset_roots_for_date_resolves_exact_live_leaf(
    tmp_path: Path,
) -> None:
    dataset_base = tmp_path / "datasets"
    active_root = dataset_base / "2026-08-07_active_task"
    active_root.mkdir(parents=True)
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name active_task"
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        assert storage.active_collection_dataset_roots_for_date("2026-08-07") == (
            active_root,
        )


def test_active_collection_dataset_roots_for_date_fails_closed(
    tmp_path: Path,
) -> None:
    dataset_base = tmp_path / "datasets"
    dataset_base.mkdir()
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name active_task"
    )

    with pytest.raises(storage.ModalityStorageError, match="YYYY-MM-DD"):
        storage.active_collection_dataset_roots_for_date("today")
    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        assert storage.active_collection_dataset_roots_for_date("2026-08-07") == ()


@pytest.mark.parametrize(
    ("task_name", "active_leaf_name"),
    (
        ("different_task", "2026-08-07_active_task"),
        ("active_task", "not-a-dated-task-leaf"),
    ),
)
def test_export_rejects_unproven_active_collection_leaf_narrowing(
    tmp_path: Path,
    offline_validation: object,
    task_name: str,
    active_leaf_name: str,
) -> None:
    dataset_base = tmp_path / "datasets"
    root = _dataset(dataset_base / "archived_dataset")
    active_root = dataset_base / active_leaf_name
    active_root.mkdir()
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name {task_name}"
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        with pytest.raises(storage.ModalityStorageError, match="cannot prove"):
            storage.export_modality(
                root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
                allow_disjoint_active_collection=True,
                active_collection_dataset_roots=(active_root,),
            )

    assert not tuple(cold.iterdir())


def test_export_rejects_verified_active_collection_leaf_itself(
    tmp_path: Path,
    offline_validation: object,
) -> None:
    dataset_base = tmp_path / "datasets"
    active_root = _dataset(dataset_base / "2026-08-07_active_task")
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_export(
        active_root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    command = (
        "pid=123 python collector "
        f"--dataset-root {dataset_base} --collection-task-name active_task"
    )

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        with pytest.raises(storage.ModalityStorageError, match="cannot prove"):
            storage.export_modality(
                active_root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
                allow_disjoint_active_collection=True,
                active_collection_dataset_roots=(active_root,),
            )

    assert not tuple(cold.iterdir())


@pytest.mark.parametrize(
    "active_command",
    (
        "pid=123 python collector",
        "pid=123 python collector --dataset-root {dataset_root}",
    ),
)
def test_export_disjoint_override_fails_closed_when_not_provable(
    tmp_path: Path,
    offline_validation: object,
    active_command: str,
) -> None:
    root = _dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    plan = storage.plan_modality_export(
        root,
        cold,
        require_distinct_device=False,
        min_free_bytes=0,
    )
    command = active_command.format(dataset_root=root)

    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(command,),
    ):
        with pytest.raises(storage.ModalityStorageError, match="cannot prove"):
            storage.export_modality(
                root,
                cold,
                expected_plan_hash=plan.plan_hash,
                confirm_dataset_fingerprint=plan.dataset_fingerprint,
                require_distinct_device=False,
                min_free_bytes=0,
                confirm_no_active_readers=True,
                allow_disjoint_active_collection=True,
            )

    assert not tuple(cold.iterdir())


def test_cli_is_dry_run_by_default_and_apply_requires_confirmations() -> None:
    parser = build_parser()
    detach = parser.parse_args(
        ["modality-detach", "/tmp/hot", "--cold-root", "/tmp/cold"]
    )
    assert detach.apply is False
    assert detach.plan_hash == ""
    assert detach.detach_scope == "payload-only"
    whole = parser.parse_args(
        [
            "modality-detach",
            "/tmp/hot",
            "--cold-root",
            "/tmp/cold",
            "--detach-scope",
            "whole-prefix",
        ]
    )
    assert whole.detach_scope == "whole-prefix"
    move = parser.parse_args(
        ["modality-move", "/tmp/hot", "--cold-root", "/tmp/cold"]
    )
    assert move.apply is False
    assert move.confirm_no_active_readers is False
    rehydrate = parser.parse_args(
        [
            "modality-rehydrate",
            "/tmp/hot",
            "--cold-root",
            "/tmp/cold",
            "--inventory-hash",
            "0" * 64,
        ]
    )
    assert rehydrate.apply is False

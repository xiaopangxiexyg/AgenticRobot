from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from dataset_contracts.modality_artifacts import (
    MODALITY_SOURCE_BINDING_PROFILE_SCOPED_V1,
    ModalityArtifactContractError,
    ModalityArtifactInventoryV1,
    ModalityStorageTransferReceiptV1,
)
from dataset_interface.modality_artifacts import (
    DETACHED_VERIFIED,
    EXTERNAL_BOUND,
    INCOMPLETE_OR_CORRUPT,
    ModalityArtifactInventoryError,
    ModalityStorageEvidenceError,
    PAYLOAD_HASH_PREFER_SEALED,
    PAYLOAD_HASH_REQUIRE_SEALED,
    SELF_CONTAINED,
    STALE_INVENTORY,
    WHOLE_PREFIX_DETACH,
    build_modality_artifact_inventory,
    resolve_modality_availability,
    verify_modality_artifact_inventory,
    verified_detachment_scope,
)
from dataset_interface.episode_curation import source_dataset_base_fingerprint


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _dataset(root: Path) -> Path:
    _write_json(root / "meta" / "info.json", {"episode_count": 1, "task": "unit"})
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "unit"}) + "\n",
        encoding="utf-8",
    )
    _write_json(root / "meta" / "modality_validity.json", {"tactile": "valid"})
    manifest_rel = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/manifest.json"
    )
    raw_rel = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/raw.bin"
    )
    zstd_rel = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/compressed.zst"
    )
    raw_path = root / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"raw-frame-0raw-frame-1")
    (root / zstd_rel).write_bytes(b"zstd-payload-fixture")
    _write_json(
        root / manifest_rel,
        {
            "format": "dexteleop_tactile_stream_v1",
            "side": "left",
            "frame_count": 2,
            "streams": {
                "L_RAW": {
                    "path": raw_rel,
                    "dtype": "uint8",
                    "shape": [11],
                    "frame_count": 2,
                    "encoding": "raw_c_order",
                    "compression": "none",
                },
                "L_ZSTD": {
                    "path": zstd_rel,
                    "dtype": "uint8",
                    "shape": [10],
                    "frame_count": 2,
                    "encoding": "chunked_zstd",
                    "compression": "zstd",
                },
            },
            "frames": [{"frame_index": 0}, {"frame_index": 1}],
        },
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 2,
                "tactile_paths": {"left": manifest_rel},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_tactile_inventory_round_trip_and_mixed_codec_graph(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)

    assert inventory.modality_key == "tactile"
    assert (
        inventory.source_binding_profile
        == MODALITY_SOURCE_BINDING_PROFILE_SCOPED_V1
    )
    assert inventory.file_count == 3
    assert [item.role for item in inventory.artifacts].count("manifest") == 1
    assert [item.role for item in inventory.artifacts].count("payload") == 2
    assert ModalityArtifactInventoryV1.from_dict(inventory.to_dict()) == inventory
    verify_modality_artifact_inventory(root, inventory)


def test_sealed_payload_hash_mode_is_explicit_and_byte_equivalent(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    manifest_path = next((root / "tactile").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for spec in manifest["streams"].values():
        payload = root / spec["path"]
        spec["stored_size_bytes"] = payload.stat().st_size
        spec["stored_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
        spec["stored_hash_profile"] = "stored_bytes_sha256_v1"
    _write_json(manifest_path, manifest)

    scanned = build_modality_artifact_inventory(root)
    preferred = build_modality_artifact_inventory(
        root,
        payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
    )
    required = build_modality_artifact_inventory(
        root,
        payload_hash_mode=PAYLOAD_HASH_REQUIRE_SEALED,
    )
    assert preferred == scanned == required

    first_spec = next(iter(manifest["streams"].values()))
    first_spec.pop("stored_sha256")
    _write_json(manifest_path, manifest)
    with pytest.raises(ModalityArtifactInventoryError, match="incomplete"):
        build_modality_artifact_inventory(
            root,
            payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
        )


def test_inventory_verifies_an_explicit_independent_artifact_root(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)
    cold = tmp_path / "cold"
    shutil.copytree(root / "tactile", cold / "tactile")

    verify_modality_artifact_inventory(root, inventory, artifact_root=cold)


def test_availability_distinguishes_hot_detached_and_external_bound(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)

    hot = resolve_modality_availability(root, inventory)
    assert hot.status == SELF_CONTAINED
    assert hot.core_training_eligible is True
    assert hot.tactile_read_eligible is True

    cold = tmp_path / "cold"
    shutil.copytree(root / "tactile", cold / "tactile")
    for artifact in inventory.artifacts:
        if artifact.role == "payload":
            (root / artifact.logical_path).unlink()

    detached = resolve_modality_availability(root, inventory)
    assert detached.status == DETACHED_VERIFIED
    assert detached.core_training_eligible is True
    assert detached.tactile_read_eligible is False
    assert detached.present_payload_count == 0
    assert detached.missing_payload_count == 2

    external = resolve_modality_availability(root, inventory, artifact_root=cold)
    assert external.status == EXTERNAL_BOUND
    assert external.core_training_eligible is True
    assert external.tactile_read_eligible is True


def test_whole_prefix_availability_requires_explicit_scope_and_complete_absence(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)
    cold = tmp_path / "cold"
    shutil.copytree(root / "tactile", cold / "tactile")
    shutil.rmtree(root / "tactile")

    legacy = resolve_modality_availability(root, inventory)
    assert legacy.status == INCOMPLETE_OR_CORRUPT

    detached = resolve_modality_availability(
        root,
        inventory,
        detachment_scope=WHOLE_PREFIX_DETACH,
    )
    assert detached.status == DETACHED_VERIFIED
    external = resolve_modality_availability(
        root,
        inventory,
        artifact_root=cold,
        detachment_scope=WHOLE_PREFIX_DETACH,
    )
    assert external.status == EXTERNAL_BOUND

    (root / "tactile").mkdir()
    partial = resolve_modality_availability(
        root,
        inventory,
        detachment_scope=WHOLE_PREFIX_DETACH,
    )
    assert partial.status == INCOMPLETE_OR_CORRUPT


def test_availability_fails_closed_for_partial_extra_and_stale_states(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "partial")
    inventory = build_modality_artifact_inventory(root)
    first_payload = next(
        artifact for artifact in inventory.artifacts if artifact.role == "payload"
    )
    (root / first_payload.logical_path).unlink()
    partial = resolve_modality_availability(root, inventory)
    assert partial.status == INCOMPLETE_OR_CORRUPT
    assert partial.core_training_eligible is False

    root = _dataset(tmp_path / "extra")
    inventory = build_modality_artifact_inventory(root)
    (root / "tactile" / "orphan.bin").write_bytes(b"orphan")
    extra = resolve_modality_availability(root, inventory)
    assert extra.status == INCOMPLETE_OR_CORRUPT
    assert "unowned" in extra.reason

    root = _dataset(tmp_path / "stale")
    inventory = build_modality_artifact_inventory(root)
    catalog = root / "meta" / "episodes.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["tactile_paths"]["left"] = "tactile/changed/manifest.json"
    catalog.write_text(json.dumps(record) + "\n", encoding="utf-8")
    stale = resolve_modality_availability(root, inventory)
    assert stale.status == STALE_INVENTORY
    assert stale.core_training_eligible is False


def test_inventory_rejects_missing_and_unowned_tactile_files(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "missing")
    (root / inventory_payload_path(root)).unlink()
    with pytest.raises(ModalityArtifactInventoryError, match="missing"):
        build_modality_artifact_inventory(root)

    root = _dataset(tmp_path / "extra")
    (root / "tactile" / "orphan.bin").write_bytes(b"orphan")
    with pytest.raises(ModalityArtifactInventoryError, match="unowned"):
        build_modality_artifact_inventory(root)


def inventory_payload_path(root: Path) -> str:
    manifest = next((root / "tactile").rglob("manifest.json"))
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return str(value["streams"]["L_RAW"]["path"])


def test_inventory_rejects_path_escape_before_opening_file(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    catalog = root / "meta" / "episodes.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["tactile_paths"]["left"] = "tactile/../outside.json"
    catalog.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ModalityArtifactInventoryError, match="normalized path"):
        build_modality_artifact_inventory(root)


def test_inventory_rejects_symlink_and_hardlink_artifacts(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "symlink")
    payload = root / inventory_payload_path(root)
    backing = root / "backing.bin"
    payload.rename(backing)
    payload.symlink_to(backing)
    with pytest.raises(ModalityArtifactInventoryError, match="symlink"):
        build_modality_artifact_inventory(root)

    root = _dataset(tmp_path / "hardlink")
    payload = root / inventory_payload_path(root)
    payload.with_name("hardlink-copy.bin").hardlink_to(payload)
    with pytest.raises(ModalityArtifactInventoryError, match="link_count=1"):
        build_modality_artifact_inventory(root)

    root = _dataset(tmp_path / "symlink-directory")
    external = tmp_path / "external-directory"
    external.mkdir()
    (root / "tactile" / "unowned-link").symlink_to(
        external,
        target_is_directory=True,
    )
    with pytest.raises(ModalityArtifactInventoryError, match="symlink entries"):
        build_modality_artifact_inventory(root)


def test_inventory_ignores_unrelated_metadata_but_detects_tactile_mutation(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)
    catalog = root / "meta" / "episodes.jsonl"
    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["video_paths"] = {"front": "videos/front/new.mp4"}
    record["frame_count"] = 99
    catalog.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(root / "meta/info.json", {"episode_count": 1, "task": "renamed"})
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "renamed"}) + "\n",
        encoding="utf-8",
    )
    _write_json(root / "meta/modality_validity.json", {"video": "updated"})

    assert build_modality_artifact_inventory(root) == inventory
    verify_modality_artifact_inventory(root, inventory)

    record["tactile_paths"]["left"] = "tactile/changed/manifest.json"
    catalog.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ModalityArtifactInventoryError, match="reference binding"):
        verify_modality_artifact_inventory(root, inventory)

    root = _dataset(tmp_path / "payload")
    inventory = build_modality_artifact_inventory(root)
    payload = root / inventory_payload_path(root)
    original = payload.read_bytes()
    payload.write_bytes(bytes([original[0] ^ 0x01]) + original[1:])
    with pytest.raises(ModalityArtifactInventoryError, match="hash changed"):
        verify_modality_artifact_inventory(root, inventory)


def test_legacy_inventory_keeps_its_hash_and_uses_safe_scoped_fallback(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    scoped = build_modality_artifact_inventory(root)
    catalog = root / "meta/episodes.jsonl"
    legacy = ModalityArtifactInventoryV1(
        modality_key="tactile",
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(root),
        source_episode_catalog_sha256=hashlib.sha256(
            catalog.read_bytes()
        ).hexdigest(),
        source_episode_count=1,
        logical_prefixes=("tactile",),
        artifacts=scoped.artifacts,
    )
    legacy_payload = legacy.to_dict()
    legacy_hash = legacy.semantic_content_hash
    assert "source_binding_profile" not in legacy_payload
    assert ModalityArtifactInventoryV1.from_dict(legacy_payload) == legacy
    assert ModalityArtifactInventoryV1.from_dict(
        legacy_payload
    ).semantic_content_hash == legacy_hash

    record = json.loads(catalog.read_text(encoding="utf-8"))
    record["video_paths"] = {"front": "videos/front/repacked.mp4"}
    catalog.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(root / "meta/info.json", {"episode_count": 1, "task": "renamed"})
    verify_modality_artifact_inventory(root, legacy)

    record["tactile_paths"]["left"] = "tactile/changed/manifest.json"
    catalog.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ModalityArtifactInventoryError, match="reference binding"):
        verify_modality_artifact_inventory(root, legacy)


def test_inventory_contract_rejects_tampered_summary_hash(tmp_path: Path) -> None:
    inventory = build_modality_artifact_inventory(_dataset(tmp_path / "dataset"))
    payload = inventory.to_dict()
    payload["total_bytes"] = int(payload["total_bytes"]) + 1
    with pytest.raises(ModalityArtifactContractError, match="total_bytes mismatch"):
        ModalityArtifactInventoryV1.from_dict(payload)


def test_transfer_receipt_is_path_free_and_hash_verified(tmp_path: Path) -> None:
    inventory = build_modality_artifact_inventory(_dataset(tmp_path / "dataset"))
    receipt = ModalityStorageTransferReceiptV1(
        operation_id="export.unit.001",
        operation="export_shadow",
        inventory_hash=inventory.semantic_content_hash,
        source_storage_id="hot.local",
        target_storage_id="cold.archive",
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        status="complete",
        independent_copy_verified=True,
        completed_at="2026-08-04T12:00:00Z",
    )
    payload = receipt.to_dict()
    assert "/" not in payload["source_storage_id"]
    assert ModalityStorageTransferReceiptV1.from_dict(payload) == receipt
    payload["file_count"] = int(payload["file_count"]) + 1
    with pytest.raises(ModalityArtifactContractError, match="hash mismatch"):
        ModalityStorageTransferReceiptV1.from_dict(payload)


def test_verified_detachment_scope_requires_a_valid_durable_retirement_receipt(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    inventory = build_modality_artifact_inventory(root)
    assert verified_detachment_scope(root, inventory.semantic_content_hash) is None
    receipt = ModalityStorageTransferReceiptV1(
        operation_id="retire_hot_module_tree.fixture",
        operation="retire_hot_module_tree",
        inventory_hash=inventory.semantic_content_hash,
        source_storage_id="hot.local",
        target_storage_id="cold.archive",
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        status="complete",
        independent_copy_verified=True,
        completed_at="2026-08-05T00:00:00Z",
    )
    receipt_path = (
        root
        / "meta/modality_artifacts/tactile/receipts"
        / f"{receipt.operation_id}.json"
    )
    _write_json(receipt_path, receipt.to_dict())
    assert (
        verified_detachment_scope(root, inventory.semantic_content_hash)
        == WHOLE_PREFIX_DETACH
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["total_bytes"] += 1
    _write_json(receipt_path, payload)
    with pytest.raises(ModalityStorageEvidenceError):
        verified_detachment_scope(root, inventory.semantic_content_hash)

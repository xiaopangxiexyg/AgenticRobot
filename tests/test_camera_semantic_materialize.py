from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from dataset_contracts.modality_artifacts import (
    ModalityArtifactInventoryV1,
    ModalityArtifactRecordV1,
    ModalityStorageTransferReceiptV1,
)
from dataset_interface.camera_semantic_prepared_view import (
    load_camera_semantic_prepared_view,
)
from dataset_interface.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
)
from utils.dataset_postprocess.camera_semantic_materialize import (
    materialize_camera_semantics,
    plan_camera_semantic_materialization,
)
from utils.dataset_postprocess.camera_semantic_prepared_view import (
    materialize_camera_semantic_prepared_view,
    plan_camera_semantic_prepared_view,
    plan_invalid_camera_prepared_view_quarantine,
    quarantine_invalid_camera_prepared_view,
)

LEFT_SERIAL = "260522278055"
RIGHT_SERIAL = "261122273924"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {"serial": LEFT_SERIAL},
                    "right_wrist": {"serial": RIGHT_SERIAL},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_video(path: Path, *, frame_count: int, color: str) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=16x16:r=15",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "mpeg4",
            "-y",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _write_source_dataset(root: Path) -> tuple[Path, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta = root / "meta"
    devices = meta / "devices"
    devices.mkdir(parents=True)
    (devices / "realsense.json").write_text(
        json.dumps(
            {
                "left_wrist": {
                    "camera_key": "left_wrist",
                    "serial": RIGHT_SERIAL,
                    "serial_number": RIGHT_SERIAL,
                },
                "right_wrist": {
                    "camera_key": "right_wrist",
                    "serial": LEFT_SERIAL,
                    "serial_number": LEFT_SERIAL,
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "episode_count": 1,
                "frame_count": 3,
                "features": {
                    "observation.images.left_wrist": {"shape": [16, 16, 3]},
                    "observation.images.right_wrist": {"shape": [16, 16, 3]},
                },
            }
        ),
        encoding="utf-8",
    )
    left_video = Path(
        "videos/chunk-000/observation.images.left_wrist/episode_000000.mp4"
    )
    right_video = Path(
        "videos/chunk-000/observation.images.right_wrist/episode_000000.mp4"
    )
    _write_video(root / left_video, frame_count=3, color="red")
    _write_video(root / right_video, frame_count=2, color="blue")

    left_mask = Path(
        "wrist_depth/mask/observation.images.left_wrist/"
        "episode_000000/manifest.json"
    )
    right_mask = Path(
        "wrist_depth/mask/observation.images.right_wrist/"
        "episode_000000/manifest.json"
    )
    for role, manifest, count in (
        ("left_wrist", left_mask, 3),
        ("right_wrist", right_mask, 2),
    ):
        path = root / manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "frames.bin").write_bytes(bytes(range(count)))
        path.write_text(
            json.dumps(
                {
                    "camera_key": role,
                    "stream": {
                        "path": "frames.bin",
                        "frame_count": count,
                    },
                }
            ),
            encoding="utf-8",
        )

    parquet = Path("data/chunk-000/episode_000000.parquet")
    (root / parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1, 2],
                "observation.images.left_wrist.video_frame_index": [0, 1, 2],
                "observation.images.right_wrist.video_frame_index": [0, None, 1],
                "observation.images.left_wrist.source_frame_index": [10, 11, 12],
                "observation.images.right_wrist.source_frame_index": [20, None, 21],
                "observation.images.left_wrist.path": [left_video.as_posix()] * 3,
                "observation.images.right_wrist.path": [right_video.as_posix()] * 3,
                "observation.tactile.left.payload_path": [
                    "tactile/episode_000000/left/frames.bin"
                ]
                * 3,
                "observation.wrist_camera.left.pose_base": [[1.0] * 7] * 3,
                "observation.wrist_camera.right.pose_base": [[2.0] * 7] * 3,
            }
        ),
        root / parquet,
    )
    episode = {
        "episode_index": 0,
        "frame_count": 3,
        "parquet_path": parquet.as_posix(),
        "video_paths": {
            "left_wrist": left_video.as_posix(),
            "right_wrist": right_video.as_posix(),
        },
        "wrist_depth_mask_paths": {
            "left_wrist": left_mask.as_posix(),
            "right_wrist": right_mask.as_posix(),
        },
        "summary": {
            "video_frame_stats": {
                "left_wrist": {"written": 3, "path": left_video.as_posix()},
                "right_wrist": {"written": 2, "path": right_video.as_posix()},
            }
        },
    }
    (meta / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )
    return root / left_video, root / right_video


def _duplicate_episode_zero_as_one(root: Path) -> None:
    def replace_episode_path(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): replace_episode_path(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [replace_episode_path(item) for item in value]
        if isinstance(value, str):
            return value.replace("episode_000000", "episode_000001")
        return value

    episodes_path = root / "meta/episodes.jsonl"
    source_episode = json.loads(episodes_path.read_text(encoding="utf-8"))
    duplicate = replace_episode_path(source_episode)
    assert isinstance(duplicate, dict)
    duplicate["episode_index"] = 1
    for key in ("parquet_path", "video_paths", "wrist_depth_mask_paths"):
        raw = source_episode[key]
        values = raw.values() if isinstance(raw, Mapping) else (raw,)
        for source_relative in values:
            source_path = root / str(source_relative)
            target_path = root / str(
                replace_episode_path(str(source_relative))
            )
            if source_path.name == "manifest.json":
                shutil.copytree(source_path.parent, target_path.parent)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
    episodes_path.write_text(
        json.dumps(source_episode) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["episode_count"] = 2
    info["frame_count"] = 6
    info_path.write_text(json.dumps(info), encoding="utf-8")


def _add_detached_tactile_evidence(root: Path) -> tuple[Path, str]:
    episode_path = root / "meta/episodes.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    manifest_relative = Path("tactile/episode_000000/left/manifest.json")
    payload_relative = Path("tactile/episode_000000/left/frames.bin")
    episode["tactile_paths"] = {"left": manifest_relative.as_posix()}
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    manifest = root / manifest_relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"stream": {"path": payload_relative.name}}),
        encoding="utf-8",
    )
    payload_bytes = b"detached-tactile-payload"
    inventory = ModalityArtifactInventoryV1(
        modality_key="tactile",
        source_dataset_base_fingerprint="1" * 64,
        source_episode_catalog_sha256="2" * 64,
        source_episode_count=1,
        logical_prefixes=("tactile",),
        artifacts=tuple(
            sorted(
                (
                    ModalityArtifactRecordV1(
                        logical_path=manifest_relative.as_posix(),
                        role="manifest",
                        size_bytes=manifest.stat().st_size,
                        sha256=_sha256(manifest),
                        episode_index=0,
                        side="left",
                    ),
                    ModalityArtifactRecordV1(
                        logical_path=payload_relative.as_posix(),
                        role="payload",
                        size_bytes=len(payload_bytes),
                        sha256=hashlib.sha256(payload_bytes).hexdigest(),
                        episode_index=0,
                        side="left",
                    ),
                ),
                key=lambda item: item.logical_path,
            )
        ),
    )
    sidecar = root / "meta/modality_artifacts/tactile"
    inventory_path = sidecar / "inventories" / f"{inventory.semantic_content_hash}.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(json.dumps(inventory.to_dict()), encoding="utf-8")
    receipt = ModalityStorageTransferReceiptV1(
        operation_id="retire_hot_payloads.camera_test",
        operation="retire_hot_payloads",
        inventory_hash=inventory.semantic_content_hash,
        source_storage_id="hot.test",
        target_storage_id="cold.test",
        file_count=inventory.file_count,
        total_bytes=inventory.total_bytes,
        status="complete",
        independent_copy_verified=True,
        completed_at="2026-08-13T00:00:00+00:00",
    )
    receipt_path = sidecar / "receipts" / f"{receipt.operation_id}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    return manifest, inventory.semantic_content_hash


def _replace_with_whole_prefix_retirement(root: Path, inventory_hash: str) -> None:
    shutil.rmtree(root / "tactile")
    receipt_dir = root / "meta/modality_artifacts/tactile/receipts"
    shutil.rmtree(receipt_dir)
    receipt_dir.mkdir(parents=True)
    receipt = ModalityStorageTransferReceiptV1(
        operation_id="retire_hot_module_tree.camera_test",
        operation="retire_hot_module_tree",
        inventory_hash=inventory_hash,
        source_storage_id="hot.test",
        target_storage_id="cold.test",
        file_count=2,
        total_bytes=1,
        status="complete",
        independent_copy_verified=True,
        completed_at="2026-08-13T00:00:01+00:00",
    )
    (receipt_dir / f"{receipt.operation_id}.json").write_text(
        json.dumps(receipt.to_dict()), encoding="utf-8"
    )


def _add_equivalent_stale_scope_evidence(root: Path, inventory_hash: str) -> str:
    sidecar = root / "meta/modality_artifacts/tactile"
    source_inventory = json.loads(
        (sidecar / "inventories" / f"{inventory_hash}.json").read_text(
            encoding="utf-8"
        )
    )
    source_inventory["source_dataset_base_fingerprint"] = "3" * 64
    source_inventory.pop("semantic_content_hash")
    equivalent = ModalityArtifactInventoryV1.from_dict(
        {
            **source_inventory,
            "semantic_content_hash": ModalityArtifactInventoryV1(
                modality_key=source_inventory["modality_key"],
                source_dataset_base_fingerprint=source_inventory[
                    "source_dataset_base_fingerprint"
                ],
                source_episode_catalog_sha256=source_inventory[
                    "source_episode_catalog_sha256"
                ],
                source_episode_count=source_inventory["source_episode_count"],
                logical_prefixes=tuple(source_inventory["logical_prefixes"]),
                artifacts=tuple(
                    ModalityArtifactRecordV1.from_dict(item)
                    for item in source_inventory["artifacts"]
                ),
            ).semantic_content_hash,
        }
    )
    equivalent_hash = equivalent.semantic_content_hash
    source_inventory = equivalent.to_dict()
    (sidecar / "inventories" / f"{equivalent_hash}.json").write_text(
        json.dumps(source_inventory), encoding="utf-8"
    )
    receipt = ModalityStorageTransferReceiptV1(
        operation_id="retire_hot_module_tree.stale_equivalent",
        operation="retire_hot_module_tree",
        inventory_hash=equivalent_hash,
        source_storage_id="hot.test",
        target_storage_id="cold.test",
        file_count=2,
        total_bytes=1,
        status="complete",
        independent_copy_verified=True,
        completed_at="2026-08-12T00:00:00+00:00",
    )
    (sidecar / "receipts" / f"{receipt.operation_id}.json").write_text(
        json.dumps(receipt.to_dict()), encoding="utf-8"
    )
    return equivalent_hash


@pytest.mark.optional
def test_materialization_swaps_complete_camera_families_and_preserves_gaps(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source"
    source.mkdir()
    source_left_video, source_right_video = _write_source_dataset(source)
    source_video_hashes = (_sha256(source_left_video), _sha256(source_right_video))
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    plan = plan_camera_semantic_materialization(source, output, registry)
    assert plan.status_counts == {"swapped": 1}
    assert plan.remapped_mismatched_row_count == 2
    result = materialize_camera_semantics(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )

    assert result["validated_video_count"] == 2
    assert result["validated_decoded_video_frame_count"] == 5
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table.column(
        "observation.images.left_wrist.video_frame_index"
    ).to_pylist() == [0, None, 1]
    assert table.column(
        "observation.images.right_wrist.video_frame_index"
    ).to_pylist() == [0, 1, 2]
    assert table.column(
        "observation.images.left_wrist.source_frame_index"
    ).to_pylist() == [20, None, 21]
    # Robot/arm semantic fields are deliberately outside the camera namespace.
    assert table.column(
        "observation.wrist_camera.left.pose_base"
    ).to_pylist() == [[1.0] * 7] * 3

    episode = json.loads(
        (output / "meta/episodes.jsonl").read_text(encoding="utf-8")
    )
    output_left = output / episode["video_paths"]["left_wrist"]
    output_right = output / episode["video_paths"]["right_wrist"]
    assert _sha256(output_left) == source_video_hashes[1]
    assert _sha256(output_right) == source_video_hashes[0]
    assert output_left.stat().st_ino != source_right_video.stat().st_ino
    assert output_right.stat().st_ino != source_left_video.stat().st_ino

    left_manifest = json.loads(
        (output / episode["wrist_depth_mask_paths"]["left_wrist"]).read_text(
            encoding="utf-8"
        )
    )
    right_manifest = json.loads(
        (output / episode["wrist_depth_mask_paths"]["right_wrist"]).read_text(
            encoding="utf-8"
        )
    )
    assert left_manifest["camera_key"] == "left_wrist"
    assert left_manifest["stream"]["frame_count"] == 2
    assert right_manifest["camera_key"] == "right_wrist"
    assert right_manifest["stream"]["frame_count"] == 3

    binding = load_camera_role_binding(registry)
    resolver = DatasetCameraSemanticResolver(output, binding)
    assert resolver.resolution_for_episode(0).status == "identity"
    assert resolver.require_frame_index_alignment()["remapped_episode_count"] == 0
    assert (output / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH).is_file()
    assert (_sha256(source_left_video), _sha256(source_right_video)) == source_video_hashes


@pytest.mark.optional
def test_apply_requires_frozen_plan_and_never_overwrites(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    with pytest.raises(Exception, match="expected-plan-hash"):
        materialize_camera_semantics(
            source,
            output,
            registry,
            expected_plan_hash="wrong",
        )

    output.mkdir()
    plan = plan_camera_semantic_materialization(source, output, registry)
    with pytest.raises(Exception, match="already exists"):
        materialize_camera_semantics(
            source,
            output,
            registry,
            expected_plan_hash=plan.plan_hash,
        )


@pytest.mark.optional
def test_plan_rejects_declared_source_video_frame_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _, source_right_video = _write_source_dataset(source)
    # Parquet still declares two right-wrist frames.  Replacing the payload
    # models a truncated inode propagated across a legacy hardlink lineage.
    _write_video(source_right_video, frame_count=1, color="blue")
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    with pytest.raises(
        Exception,
        match=(
            "Declared/decoded video frame count mismatch: "
            "episode=0 role=right_wrist declared=2 decoded=1"
        ),
    ):
        plan_camera_semantic_materialization(source, output, registry)

    assert not output.exists()

    _write_video(source_right_video, frame_count=1, color="blue")
    with pytest.raises(
        Exception, match="Every source episode failed video-integrity admission"
    ):
        plan_camera_semantic_prepared_view(source, output, registry)
    assert not output.exists()
    assert not list(
        tmp_path.glob(f".{output.name}.camera-semantic-stage-*")
    )


@pytest.mark.optional
def test_prepared_view_marks_and_excludes_corrupt_video_episode(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _, source_right_video = _write_source_dataset(source)
    _duplicate_episode_zero_as_one(source)
    _write_video(source_right_video, frame_count=1, color="blue")
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "prepared"

    plan = plan_camera_semantic_prepared_view(source, output, registry)
    assert plan.source_episode_count == 2
    assert plan.episode_count == 1
    assert plan.frame_count == 3
    assert plan.excluded_episode_indices == (0,)
    assert plan.excluded_episode_count == 1
    assert plan.video_integrity_exclusions_hash is not None
    assert [item.to_dict() for item in plan.video_integrity_issues] == [
        {
            "episode_index": 0,
            "camera_key": "right_wrist",
            "relative_path": (
                "videos/chunk-000/observation.images.right_wrist/"
                "episode_000000.mp4"
            ),
            "reason": "declared_decoded_frame_count_mismatch",
            "declared_frame_count": 2,
            "decoded_frame_count": 1,
            "detail": (
                "Parquet video-frame indices and the fully decoded payload "
                "report different frame counts."
            ),
        }
    ]

    result = materialize_camera_semantic_prepared_view(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )
    assert result["validated_video_count"] == 2
    manifest = load_camera_semantic_prepared_view(
        output, verify_payload_hashes=True
    )
    assert manifest.episode_indices == (1,)
    assert manifest.excluded_episode_indices == (0,)
    assert (
        manifest.video_integrity_exclusions_hash
        == plan.video_integrity_exclusions_hash
    )
    output_episodes = [
        json.loads(line)
        for line in (output / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["episode_index"] for row in output_episodes] == [1]
    output_info = json.loads(
        (output / "meta/info.json").read_text(encoding="utf-8")
    )
    assert output_info["episode_count"] == 1
    assert output_info["frame_count"] == 3
    receipt_path = (
        output
        / "meta/prepared_views/camera_semantics_video_integrity_exclusions.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["included_episode_indices"] == [1]
    assert receipt["excluded_episode_indices"] == [0]
    assert receipt["receipt_hash"] == plan.video_integrity_exclusions_hash
    assert not (
        output
        / "videos/chunk-000/observation.images.right_wrist/episode_000000.mp4"
    ).exists()

    from deployments.openpi.corrected_train import _camera_audits

    factory = type(
        "Factory",
        (),
        {
            "camera_slots": ("left_wrist_0_rgb", "right_wrist_0_rgb"),
            "dataset_sources": (
                type("Source", (), {"dataset_root": output})(),
            ),
        },
    )()
    audits = _camera_audits(
        factory,
        registry,
        allow_camera_prepared_view=True,
    )
    audit = audits[0]["camera_prepared_view"]
    assert audit["included_episode_count"] == 1
    assert audit["excluded_corrupt_video_episode_count"] == 1
    assert audit["excluded_corrupt_video_episode_indices"] == [0]

    receipt["issues"][0]["decoded_frame_count"] = 2
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(Exception, match="receipt hash mismatch"):
        load_camera_semantic_prepared_view(output, verify_payload_hashes=False)


@pytest.mark.optional
def test_materialization_preserves_detached_tactile_graph_without_cold_reads(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    source_manifest, inventory_hash = _add_detached_tactile_evidence(source)
    source_manifest_hash = _sha256(source_manifest)
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    plan = plan_camera_semantic_materialization(source, output, registry)

    assert plan.tactile_detach_audit == {
        "status": "detached_verified",
        "safe_for_camera_materialization": True,
        "inventory_hashes": [inventory_hash],
        "detachment_scope": "payload_only",
        "manifest_count": 1,
        "payload_count": 1,
        "present_hot_payload_count": 0,
        "unexpected_hot_path_count": 0,
        "missing_hot_manifest_count": 0,
        "manifest_mismatch_count": 0,
        "cold_payloads_read": False,
    }
    result = materialize_camera_semantics(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )

    output_episode = json.loads(
        (output / "meta/episodes.jsonl").read_text(encoding="utf-8")
    )
    assert output_episode["tactile_paths"] == {
        "left": "tactile/episode_000000/left/manifest.json"
    }
    output_manifest = output / output_episode["tactile_paths"]["left"]
    assert _sha256(output_manifest) == source_manifest_hash
    assert not (output / "tactile/episode_000000/left/frames.bin").exists()
    assert result["tactile_detach_audit"]["cold_payloads_read"] is False


@pytest.mark.optional
def test_materialization_refuses_tactile_that_is_not_retired(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    _, _ = _add_detached_tactile_evidence(source)
    receipt_dir = source / "meta/modality_artifacts/tactile/receipts"
    for receipt in receipt_dir.glob("*.json"):
        receipt.unlink()
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    plan = plan_camera_semantic_materialization(source, output, registry)

    assert plan.tactile_detach_audit["status"] == "no_verified_retirement"
    assert plan.tactile_detach_audit["safe_for_camera_materialization"] is False
    with pytest.raises(Exception, match="already retired"):
        materialize_camera_semantics(
            source,
            output,
            registry,
            expected_plan_hash=plan.plan_hash,
        )


@pytest.mark.optional
def test_materialization_preserves_whole_prefix_retirement(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    _, inventory_hash = _add_detached_tactile_evidence(source)
    _replace_with_whole_prefix_retirement(source, inventory_hash)
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    plan = plan_camera_semantic_materialization(source, output, registry)

    assert plan.tactile_detach_audit["status"] == "detached_verified"
    assert plan.tactile_detach_audit["detachment_scope"] == "whole_prefix"
    result = materialize_camera_semantics(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )
    assert not (output / "tactile").exists()
    assert result["tactile_detach_audit"]["cold_payloads_read"] is False


@pytest.mark.optional
def test_materialization_selects_retirement_matching_current_hot_scope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    _, inventory_hash = _add_detached_tactile_evidence(source)
    stale_hash = _add_equivalent_stale_scope_evidence(source, inventory_hash)
    registry = tmp_path / "registry.json"
    _write_registry(registry)

    plan = plan_camera_semantic_materialization(
        source, tmp_path / "output", registry
    )

    assert plan.tactile_detach_audit["detachment_scope"] == "payload_only"
    assert plan.tactile_detach_audit["inventory_hashes"] == [inventory_hash]
    assert stale_hash not in plan.tactile_detach_audit["inventory_hashes"]


@pytest.mark.optional
def test_materialization_rejects_tampered_tactile_inventory(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    _, inventory_hash = _add_detached_tactile_evidence(source)
    inventory_path = (
        source
        / "meta/modality_artifacts/tactile/inventories"
        / f"{inventory_hash}.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["artifacts"][0]["size_bytes"] += 1
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    registry = tmp_path / "registry.json"
    _write_registry(registry)

    with pytest.raises(Exception, match="Invalid tactile inventory contract"):
        plan_camera_semantic_materialization(
            source, tmp_path / "output", registry
        )


@pytest.mark.optional
def test_prepared_view_corrects_mismatched_indices_without_reading_tactile(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source"
    source.mkdir()
    source_left_video, source_right_video = _write_source_dataset(source)
    source_manifest, _ = _add_detached_tactile_evidence(source)
    tactile_payload = source_manifest.parent / "frames.bin"
    tactile_payload.write_bytes(b"hot-tactile-must-not-be-copied")
    receipt_dir = source / "meta/modality_artifacts/tactile/receipts"
    shutil.rmtree(receipt_dir)
    tactile_recompression = source / "meta/tactile_recompression/run.json"
    tactile_recompression.parent.mkdir(parents=True)
    tactile_recompression.write_text(
        json.dumps({"tactile_payload": "must-not-be-projected"}),
        encoding="utf-8",
    )
    runtime_lock = source / "meta/.modality_validity.lock"
    runtime_lock.write_text("", encoding="utf-8")
    (source / "meta/modality_validity.json").write_text(
        json.dumps(
            {
                "capture": {
                    "ignored_modalities": {"camera": [], "tactile": []},
                    "missing_value_policy": {
                        "camera": "frame_absent_valid_false",
                        "tactile": "value_absent_valid_false",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    info_path = source / "meta/info.json"
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    source_info["nested_capture_contract"] = {
        "modality_source_clock": {
            "camera": "receiver_wall_time",
            "tactile": "local_wall_time",
        }
    }
    info_path.write_text(json.dumps(source_info), encoding="utf-8")
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "prepared"

    full_plan = plan_camera_semantic_materialization(
        source, tmp_path / "full", registry
    )
    assert full_plan.tactile_detach_audit["safe_for_camera_materialization"] is False

    plan = plan_camera_semantic_prepared_view(source, output, registry)
    assert plan.tactile_reference_episode_count == 1
    assert plan.remapped_mismatched_row_count == 2
    assert plan.to_dict()["tactile_payloads_inspected"] is False
    result = materialize_camera_semantic_prepared_view(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )

    assert result["validated_video_count"] == 2
    assert not (output / "tactile").exists()
    assert not (output / "meta/modality_artifacts/tactile").exists()
    assert not (output / "meta/tactile_recompression").exists()
    assert not (output / "meta/.modality_validity.lock").exists()
    manifest = load_camera_semantic_prepared_view(
        output, verify_payload_hashes=True
    )
    assert manifest.episode_indices == (0,)
    assert manifest.plan_hash == plan.plan_hash
    assert not any(
        part.endswith(".lock")
        for record in manifest.files
        for part in record.relative_path.split("/")
    )

    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table.column(
        "observation.images.left_wrist.video_frame_index"
    ).to_pylist() == [0, None, 1]
    assert table.column(
        "observation.images.right_wrist.video_frame_index"
    ).to_pylist() == [0, 1, 2]
    assert not any("tactile" in name for name in table.column_names)
    episode = json.loads(
        (output / "meta/episodes.jsonl").read_text(encoding="utf-8")
    )
    assert "tactile_paths" not in episode
    output_info = json.loads(
        (output / "meta/info.json").read_text(encoding="utf-8")
    )
    assert output_info["nested_capture_contract"]["modality_source_clock"] == {
        "camera": "receiver_wall_time"
    }
    output_validity = json.loads(
        (output / "meta/modality_validity.json").read_text(encoding="utf-8")
    )
    assert output_validity == {
        "capture": {
            "ignored_modalities": {"camera": []},
            "missing_value_policy": {
                "camera": "frame_absent_valid_false"
            },
        }
    }
    output_left = output / episode["video_paths"]["left_wrist"]
    output_right = output / episode["video_paths"]["right_wrist"]
    assert _sha256(output_left) == _sha256(source_right_video)
    assert _sha256(output_right) == _sha256(source_left_video)
    assert output_left.stat().st_ino != source_right_video.stat().st_ino
    assert output_right.stat().st_ino != source_left_video.stat().st_ino

    from deployments.openpi.corrected_train import _camera_audits

    factory = type(
        "Factory",
        (),
        {
            "camera_slots": ("left_wrist_0_rgb", "right_wrist_0_rgb"),
            "dataset_sources": (
                type("Source", (), {"dataset_root": output})(),
            ),
        },
    )()
    with pytest.raises(ValueError, match="--allow-camera-prepared-view"):
        _camera_audits(factory, registry)
    audits = _camera_audits(
        factory, registry, allow_camera_prepared_view=True
    )
    assert audits[0]["camera_prepared_view"]["full_v23_dataset"] is False
    assert audits[0]["frame_index_alignment"]["remapped_episode_count"] == 0

    injected_lock = output / "meta/runtime.lock"
    injected_lock.write_text("", encoding="utf-8")
    with pytest.raises(Exception, match="runtime lock file"):
        load_camera_semantic_prepared_view(output, verify_payload_hashes=True)
    injected_lock.unlink()

    output_left.write_bytes(b"tampered")
    with pytest.raises(Exception, match="file (size|hash) mismatch"):
        load_camera_semantic_prepared_view(output, verify_payload_hashes=True)


@pytest.mark.optional
def test_prepared_view_rejects_corrupt_video_and_stale_plan(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _, source_right_video = _write_source_dataset(source)
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "prepared"

    plan = plan_camera_semantic_prepared_view(source, output, registry)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["concurrent_change"] = True
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(Exception, match="expected-plan-hash"):
        materialize_camera_semantic_prepared_view(
            source,
            output,
            registry,
            expected_plan_hash=plan.plan_hash,
        )
    assert not output.exists()


def test_invalid_prepared_view_quarantine_is_exact_and_recoverable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared_views/bad_trial"
    manifest = source / "meta/prepared_views/camera_semantics.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"manifest_hash": "a" * 64, "schema": "invalid"}),
        encoding="utf-8",
    )
    (source / "meta/tactile_recompression").mkdir(parents=True)
    (source / "meta/tactile_recompression/run.json").write_text(
        "{}", encoding="utf-8"
    )
    audit = tmp_path / "prepared_views/_invalid_audit"

    plan = plan_invalid_camera_prepared_view_quarantine(source, audit)
    assert "unsupported" in plan.invalid_reason.lower()
    with pytest.raises(Exception, match="expected-plan-hash mismatch"):
        quarantine_invalid_camera_prepared_view(
            source,
            audit,
            expected_plan_hash="b" * 64,
        )
    assert source.is_dir()

    applied = quarantine_invalid_camera_prepared_view(
        source,
        audit,
        expected_plan_hash=plan.plan_hash,
    )
    assert not source.exists()
    destination = Path(applied.destination_root)
    assert destination.is_dir()
    assert (destination / "meta/tactile_recompression/run.json").is_file()


def test_invalid_prepared_view_quarantine_does_not_treat_io_failure_as_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "prepared_views/transient_failure"
    manifest = source / "meta/prepared_views/camera_semantics.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"manifest_hash": "c" * 64, "schema": "invalid"}),
        encoding="utf-8",
    )

    def fail_io(*_args: object, **_kwargs: object) -> object:
        raise OSError("transient read failure")

    monkeypatch.setattr(
        "utils.dataset_postprocess.camera_semantic_prepared_view."
        "load_camera_semantic_prepared_view",
        fail_io,
    )
    with pytest.raises(OSError, match="transient read failure"):
        plan_invalid_camera_prepared_view_quarantine(
            source,
            tmp_path / "prepared_views/_invalid_audit",
        )
    assert source.is_dir()

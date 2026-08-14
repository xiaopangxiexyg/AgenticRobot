from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from collection_runtime import modality_storage as storage
from collection_runtime.tactile_postprocess import (
    TACTILE_TARGET_COMPRESSED_HOT,
    TACTILE_TARGET_DETACHED,
    TACTILE_TARGET_HOT,
    apply_tactile_postprocess,
    plan_tactile_postprocess,
)
from collection_runtime.tactile_recompression import (
    recompress_tactile_zstd_in_place,
)
from dataset_contracts.tactile import tactile_stream_compression
from utils.dataset_postprocess.cli import build_parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _mixed_dataset(root: Path) -> Path:
    _write_json(root / "meta/info.json", {"episode_count": 2, "task": "unit"})
    records: list[dict[str, object]] = []
    for episode_index in range(2):
        token = f"episode_{episode_index:06d}"
        directory = (
            root
            / "tactile/observation.hand.left.tactile/chunk-000"
            / token
        )
        payload_path = directory / "L_RAW.uint8.bin"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(bytes(range(24)))
        manifest_path = directory / "manifest.json"
        _write_json(
            manifest_path,
            {
                "format": "binary_stream_v1",
                "side": "left",
                "frame_count": 4,
                "streams": {
                    "L_RAW": {
                        "path": payload_path.relative_to(root).as_posix(),
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
        records.append(
            {
                "episode_index": episode_index,
                "frame_count": 4,
                "tactile_paths": {
                    "left": manifest_path.relative_to(root).as_posix()
                },
            }
        )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    recompress_tactile_zstd_in_place(
        root,
        level=1,
        chunk_frames=2,
        min_free_bytes=0,
        max_manifests=1,
        delete_originals=True,
        backend="auto",
        enforce_global_collection_exclusion=False,
    )
    return root


def _compressions(root: Path) -> list[str]:
    values: list[str] = []
    for path in sorted((root / "tactile").rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        values.extend(
            tactile_stream_compression(spec)
            for spec in manifest["streams"].values()
        )
    return values


@pytest.fixture
def offline_storage_validation() -> object:
    with mock.patch.object(
        storage,
        "_active_collection_processes",
        return_value=(),
    ):
        yield


def test_mixed_raw_and_zstd_plan_only_converts_remaining_raw(tmp_path: Path) -> None:
    root = _mixed_dataset(tmp_path / "dataset")
    assert _compressions(root) == ["zstd", "none"]

    plan = plan_tactile_postprocess(
        root,
        target=TACTILE_TARGET_COMPRESSED_HOT,
        min_free_bytes=0,
    )

    assert plan.ready
    assert not plan.target_satisfied
    assert plan.raw_manifest_count == 1
    assert plan.raw_stream_count == 1
    assert plan.compressed_stream_count == 1
    assert [step["name"] for step in plan.actions] == ["recompress-raw-zstd"]

    result = apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_COMPRESSED_HOT,
        min_free_bytes=0,
        backend="auto",
    )
    assert result.final_plan.target_satisfied
    assert _compressions(root) == ["zstd", "zstd"]

    retry = apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_COMPRESSED_HOT,
        min_free_bytes=0,
    )
    assert retry.operations == ()


def test_detached_state_is_noop_and_rehydrates_before_hot_target(
    tmp_path: Path,
    offline_storage_validation: object,
) -> None:
    root = _mixed_dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()

    detached = apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_DETACHED,
        cold_root=cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    assert detached.final_plan.target_satisfied
    assert not (root / "tactile").exists()
    assert [item["name"] for item in detached.operations] == [
        "recompress-raw-zstd",
        "move-whole-tactile-prefix",
    ]

    already_detached = plan_tactile_postprocess(
        root,
        target=TACTILE_TARGET_DETACHED,
        min_free_bytes=0,
    )
    assert already_detached.ready
    assert already_detached.target_satisfied
    assert already_detached.actions == ()

    missing_cold = plan_tactile_postprocess(
        root,
        target=TACTILE_TARGET_COMPRESSED_HOT,
        min_free_bytes=0,
    )
    assert not missing_cold.ready
    assert "--cold-root" in " ".join(missing_cold.errors)

    restored = apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_HOT,
        cold_root=cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )
    assert restored.final_plan.target_satisfied
    assert _compressions(root) == ["zstd", "zstd"]


def test_compressed_hot_rehydrates_then_finishes_mixed_detached_dataset(
    tmp_path: Path,
    offline_storage_validation: object,
) -> None:
    root = _mixed_dataset(tmp_path / "dataset")
    cold = tmp_path / "cold"
    cold.mkdir()
    apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_DETACHED,
        cold_root=cold,
        require_distinct_device=False,
        min_free_bytes=0,
        compress_before_detach=False,
        confirm_no_active_readers=True,
    )

    result = apply_tactile_postprocess(
        root,
        target=TACTILE_TARGET_COMPRESSED_HOT,
        cold_root=cold,
        require_distinct_device=False,
        min_free_bytes=0,
        confirm_no_active_readers=True,
    )

    assert [item["name"] for item in result.operations] == [
        "rehydrate",
        "recompress-raw-zstd",
    ]
    assert result.final_plan.target_satisfied
    assert _compressions(root) == ["zstd", "zstd"]


def test_missing_payload_and_no_tactile_are_not_confused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    empty_plan = plan_tactile_postprocess(empty)
    assert empty_plan.ready
    assert empty_plan.current_status == "NO_EXTERNAL_TACTILE"
    assert empty_plan.target_satisfied

    unowned_root = tmp_path / "unowned"
    unowned_payload = unowned_root / "tactile/orphan.bin"
    unowned_payload.parent.mkdir(parents=True)
    unowned_payload.write_bytes(b"orphan")
    unowned = plan_tactile_postprocess(unowned_root)
    assert not unowned.ready
    assert unowned.current_status == "UNOWNED_TACTILE_TREE"

    root = _mixed_dataset(tmp_path / "broken")
    next((root / "tactile").rglob("*.bin")).unlink()
    broken = plan_tactile_postprocess(root, min_free_bytes=0)
    assert not broken.ready
    assert broken.current_status == "INCOMPLETE_OR_CORRUPT"


def test_tactile_maintain_cli_is_dry_run_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["tactile-maintain", "/tmp/dataset"])
    assert args.target == TACTILE_TARGET_COMPRESSED_HOT
    assert args.apply is False
    assert args.compress_before_detach is True

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from collection_runtime import dataset_capture_recovery as recovery
from collection_runtime.modality_inventory import (
    build_and_publish_modality_inventory,
)
from dataset_interface.modality_artifacts import (
    DatasetCaptureWritableError,
    UnownedTactileFilesError,
    assert_dataset_capture_writable,
)
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _finalized_tactile_dataset(root: Path) -> Path:
    manifest_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/manifest.json"
    )
    payload_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L.uint8.bin"
    )
    payload = root / payload_relative
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"sealed-tactile")
    _write_json(
        root / manifest_relative,
        {
            "frame_count": 14,
            "streams": {
                "L": {
                    "path": payload_relative,
                    "dtype": "uint8",
                    "shape": [1],
                    "frame_count": 14,
                }
            }
        },
    )
    catalog = root / "meta/episodes.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 14,
                "tactile_paths": {"left": manifest_relative},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _write_abandoned_tail(root: Path, episode_index: int = 1) -> dict[str, bytes]:
    token = f"episode_{episode_index:06d}"
    payloads = {
        (
            "tactile/observation.hand.left.tactile/chunk-000/"
            f"{token}/L.uint8.bin"
        ): b"abandoned-tactile",
        f"videos/observation.images.front/chunk-000/{token}.mp4": b"partial-video",
        (
            "wrist_depth/mask/observation.images.left_wrist/chunk-000/"
            f"{token}/stream.bitpack.bin"
        ): b"partial-depth",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payloads


def test_prepare_quarantines_only_abandoned_next_episode_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    root = _finalized_tactile_dataset(tmp_path / "dataset")
    finalized_before = (
        root
        / "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L.uint8.bin"
    ).read_bytes()
    abandoned = _write_abandoned_tail(root)

    with pytest.warns(RuntimeWarning, match="quarantined abandoned"):
        result = recovery.prepare_dataset_capture_writable(root)

    assert result.recovered is True
    assert result.episode_index == 1
    assert result.receipt_path is not None
    assert Path(result.receipt_path).is_file()
    quarantine = Path(result.quarantine_root or "")
    for relative, payload in abandoned.items():
        assert not (root / relative).exists()
        assert (quarantine / "artifacts" / relative).read_bytes() == payload
    assert (
        root
        / "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L.uint8.bin"
    ).read_bytes() == finalized_before
    assert_dataset_capture_writable(root)
    assert recovery.prepare_dataset_capture_writable(root).recovered is False


def test_prepare_resumes_partially_applied_recovery_transaction(tmp_path: Path) -> None:
    root = _finalized_tactile_dataset(tmp_path / "dataset")
    abandoned = _write_abandoned_tail(root)
    with pytest.raises(DatasetCaptureWritableError):
        assert_dataset_capture_writable(root)
    with pytest.raises(UnownedTactileFilesError) as captured:
        assert_dataset_capture_writable(root)
    transaction = recovery._plan_transaction(
        root,
        unowned_tactile_paths=captured.value.paths,
    )
    journal = (
        root
        / recovery.CAPTURE_RECOVERY_RELATIVE_ROOT
        / "active_transaction.json"
    )
    recovery._atomic_write_json(journal, transaction)
    first = transaction["moves"][0]
    source = root / first["source_relative"]
    target = root / first["target_relative"]
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)

    with pytest.warns(RuntimeWarning, match="resumed abandoned"):
        result = recovery.prepare_dataset_capture_writable(root)

    assert result.recovered is True
    assert not journal.exists()
    quarantine = Path(result.quarantine_root or "")
    for relative, payload in abandoned.items():
        assert (quarantine / "artifacts" / relative).read_bytes() == payload
    assert_dataset_capture_writable(root)


def test_recovery_journal_fails_closed_if_finalized_catalog_changes(
    tmp_path: Path,
) -> None:
    root = _finalized_tactile_dataset(tmp_path / "dataset")
    abandoned = _write_abandoned_tail(root)
    with pytest.raises(UnownedTactileFilesError) as captured:
        assert_dataset_capture_writable(root)
    transaction = recovery._plan_transaction(
        root,
        unowned_tactile_paths=captured.value.paths,
    )
    journal = (
        root
        / recovery.CAPTURE_RECOVERY_RELATIVE_ROOT
        / "active_transaction.json"
    )
    recovery._atomic_write_json(journal, transaction)
    with (root / "meta/episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(
        recovery.DatasetCaptureRecoveryError,
        match="catalog changed",
    ):
        recovery.prepare_dataset_capture_writable(root)

    assert journal.is_file()
    for relative, payload in abandoned.items():
        assert (root / relative).read_bytes() == payload


def test_prepare_rejects_unknown_or_multiple_unfinalized_artifacts(tmp_path: Path) -> None:
    unknown = _finalized_tactile_dataset(tmp_path / "unknown")
    stray = unknown / "tactile/unowned.bin"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"unknown")
    with pytest.raises(
        recovery.DatasetCaptureRecoveryError,
        match="not attributable to one Episode",
    ):
        recovery.prepare_dataset_capture_writable(unknown)
    assert stray.read_bytes() == b"unknown"

    multiple = _finalized_tactile_dataset(tmp_path / "multiple")
    tail = _write_abandoned_tail(multiple, episode_index=1)
    extra = multiple / "videos/front/chunk-000/episode_000002.mp4"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"later")
    with pytest.raises(
        recovery.DatasetCaptureRecoveryError,
        match="not confined to the unique next Episode",
    ):
        recovery.prepare_dataset_capture_writable(multiple)
    assert extra.read_bytes() == b"later"
    for relative, payload in tail.items():
        assert (multiple / relative).read_bytes() == payload


def test_prepare_rejects_recovery_while_collection_process_is_active(
    tmp_path: Path,
) -> None:
    root = _finalized_tactile_dataset(tmp_path / "dataset")
    abandoned = _write_abandoned_tail(root)
    with mock.patch.object(
        recovery,
        "_active_collection_processes",
        return_value=("pid=42 python -m teleop_runtime.collection",),
    ):
        with pytest.raises(
            recovery.DatasetCaptureRecoveryError,
            match="formal collection is still active",
        ):
            recovery.prepare_dataset_capture_writable(root)
    for relative, payload in abandoned.items():
        assert (root / relative).read_bytes() == payload


def test_tracked_inventory_uses_the_same_tail_recovery_path(tmp_path: Path) -> None:
    root = _finalized_tactile_dataset(tmp_path / "dataset")
    inventory = build_and_publish_modality_inventory(
        root,
        payload_hash_mode="scan",
    )
    abandoned = _write_abandoned_tail(root)

    with pytest.raises(UnownedTactileFilesError):
        assert_dataset_capture_writable(root)
    with pytest.warns(RuntimeWarning, match="episode=1"):
        result = recovery.prepare_dataset_capture_writable(root)

    assert result.episode_index == 1
    assert inventory.semantic_content_hash in str(
        next(
            (root / "meta/modality_artifacts/tactile/inventories").glob("*.json")
        )
    )
    for relative in abandoned:
        assert not (root / relative).exists()
    assert_dataset_capture_writable(root)


def test_prepare_recovers_interrupted_first_episode_with_empty_catalog(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/episodes.jsonl").touch()
    abandoned = _write_abandoned_tail(root, episode_index=0)

    with pytest.warns(RuntimeWarning, match="episode=0"):
        result = recovery.prepare_dataset_capture_writable(root)

    assert result.episode_index == 0
    quarantine = Path(result.quarantine_root or "")
    for relative, payload in abandoned.items():
        assert (quarantine / "artifacts" / relative).read_bytes() == payload
    assert_dataset_capture_writable(root)


def test_native_prepare_runs_full_admission_once_without_recovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    identity = root / "meta/tactile_bundle_v1/dataset_identity.cjson"
    identity.parent.mkdir(parents=True)
    identity.write_bytes(b"native-identity-fixture")
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()

    with (
        mock.patch.object(
            recovery,
            "assert_dataset_capture_writable",
        ) as admission,
        mock.patch.object(
            recovery,
            "_plan_native_tactile_core_tail",
            return_value=None,
        ) as plan_tail,
    ):
        result = recovery.prepare_dataset_capture_writable(
            root,
            tactile_bundle_root=tactile_root,
        )

    assert result.recovered is False
    admission.assert_called_once_with(
        root.resolve(),
        tactile_bundle_root=tactile_root,
    )
    plan_tail.assert_called_once_with(
        root.resolve(),
        tactile_bundle_root=tactile_root,
    )


def test_native_prepare_revalidates_after_tail_recovery(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    identity = root / "meta/tactile_bundle_v1/dataset_identity.cjson"
    identity.parent.mkdir(parents=True)
    identity.write_bytes(b"native-identity-fixture")
    tactile_root = tmp_path / "tactile"
    tactile_root.mkdir()
    transaction = {"schema": recovery.CAPTURE_RECOVERY_SCHEMA_V1}
    recovered = recovery.DatasetCaptureRecoveryResult(
        recovered=True,
        episode_index=7,
        quarantine_root=str(root / "meta/capture_recovery/abandoned_episodes/fixture"),
    )

    with (
        mock.patch.object(
            recovery,
            "assert_dataset_capture_writable",
        ) as admission,
        mock.patch.object(
            recovery,
            "_plan_native_tactile_core_tail",
            return_value=transaction,
        ),
        mock.patch.object(recovery, "_atomic_write_json") as write_journal,
        mock.patch.object(
            recovery,
            "_apply_transaction",
            return_value=recovered,
        ) as apply_transaction,
        pytest.warns(RuntimeWarning, match="native-tactile core Episode"),
    ):
        result = recovery.prepare_dataset_capture_writable(
            root,
            tactile_bundle_root=tactile_root,
        )

    assert result == recovered
    assert admission.call_count == 2
    assert admission.call_args_list == [
        mock.call(root.resolve(), tactile_bundle_root=tactile_root),
        mock.call(root.resolve(), tactile_bundle_root=tactile_root),
    ]
    write_journal.assert_called_once_with(
        root.resolve()
        / recovery.CAPTURE_RECOVERY_RELATIVE_ROOT
        / "active_transaction.json",
        transaction,
    )
    apply_transaction.assert_called_once_with(root.resolve(), transaction)


def test_legacy_writer_startup_uses_recovery_before_reusing_episode_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/episodes.jsonl").touch()
    abandoned = _write_abandoned_tail(root, episode_index=0)

    with pytest.warns(RuntimeWarning, match="episode=0"):
        writer = LeRobotV21Writer(
            LeRobotV21WriterConfig(
                root_dir=root,
                write_parquet=False,
                write_video=False,
            )
        )

    assert writer.next_episode_index == 0
    for relative in abandoned:
        assert not (root / relative).exists()
    assert_dataset_capture_writable(root)

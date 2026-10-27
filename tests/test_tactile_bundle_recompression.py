from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

import dataset_interface.tactile_bundle as tactile_bundle_module
import collection_runtime.tactile_bundle_recompression as native_recompression_module
from collection_runtime.tactile_bundle_recompression import (
    recompress_native_tactile_bundle_zstd_in_place,
    resolve_native_bundle_dataset_root,
    run_idle_native_tactile_recompression_worker,
)
from collection_runtime.tactile_bundle_store import NativeTactileBundleStore
from collection_runtime.tactile_recompression import TactileRecompressionCancelled
from collection_runtime.tactile_recompression import run_idle_tactile_recompression_worker
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.tactile import ChunkedTactileStream
from dataset_interface.tactile_bundle import attach_tactile_bundle


DATASET_UUID = uuid.UUID("3ef8a7b7-22f5-44e7-812c-79d2f66f69d1")


def _native_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    core = tmp_path / "core"
    storage = tmp_path / "storage"
    core.mkdir()
    storage.mkdir()
    store = NativeTactileBundleStore(
        core_dataset_root=core,
        tactile_storage_root=storage,
        dataset_label="native-recompression-test",
        require_distinct_device=False,
        uuid_factory=lambda: DATASET_UUID,
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="episode-capture-uuid-0",
        legacy_episode_index=0,
        lifecycle_owner_id="tests.native_recompression",
        prepared_at=12.5,
        prepared_clock_domain_id="test_clock",
    )
    store.prepare_episode(identity)
    relative_dir = "tactile/chunk-000/episode_000000/left"
    payload_relative = f"{relative_dir}/M_RAW.uint8.bin"
    manifest_relative = f"{relative_dir}/manifest.json"
    payload = bytes(range(8))
    payload_path = store.tactile_artifact_path(payload_relative)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    frames = [
        {"sample_frame_index": index, "tactile_frame_index": index}
        for index in range(2)
    ]
    manifest_path = store.tactile_artifact_path(manifest_relative)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "binary_stream_v1",
                "side": "left",
                "relative_dir": relative_dir,
                "frame_count": 2,
                "sensor_keys": ["M_RAW"],
                "streams": {
                    "M_RAW": {
                        "path": payload_relative,
                        "dtype": "uint8",
                        "shape": [4],
                        "frame_count": 2,
                        "encoding": "raw_c_order",
                        "compression": "none",
                        "chunk_frame_count": None,
                        "chunks": [],
                        "stored_size_bytes": len(payload),
                        "stored_sha256": hashlib.sha256(payload).hexdigest(),
                        "stored_hash_profile": "stored_bytes_sha256_v1",
                    }
                },
                "frames": frames,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "frame_index": index,
            "observation.hand.left.tactile_path": manifest_relative,
            "observation.hand.left.tactile_frame_index": index,
        }
        for index in range(2)
    ]
    report = store.seal_episode(
        identity=identity,
        rows=rows,
        tactile_paths={"left": manifest_relative},
    )
    core_catalog = core / "meta" / "episodes.jsonl"
    core_catalog.parent.mkdir(parents=True, exist_ok=True)
    core_catalog.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "episode_id": identity.episode_uuid,
                "frame_count": 2,
                "tactile_paths": {"left": manifest_relative},
                "tactile_bundle": report.episode_binding_payload(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return core, storage, payload_path


def _append_native_episode(
    core: Path,
    storage: Path,
    *,
    episode_index: int,
) -> Path:
    store = NativeTactileBundleStore(
        core_dataset_root=core,
        tactile_storage_root=storage,
        dataset_label="native-recompression-test",
        require_distinct_device=False,
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid=f"episode-capture-uuid-{episode_index}",
        legacy_episode_index=episode_index,
        lifecycle_owner_id="tests.native_recompression",
        prepared_at=12.5 + episode_index,
        prepared_clock_domain_id="test_clock",
    )
    store.prepare_episode(identity)
    relative_dir = (
        f"tactile/chunk-{episode_index:03d}/episode_{episode_index:06d}/left"
    )
    payload_relative = f"{relative_dir}/M_RAW.uint8.bin"
    manifest_relative = f"{relative_dir}/manifest.json"
    payload = bytes(range(8 + episode_index, 16 + episode_index))
    payload_path = store.tactile_artifact_path(payload_relative)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(payload)
    frames = [
        {"sample_frame_index": index, "tactile_frame_index": index}
        for index in range(2)
    ]
    manifest_path = store.tactile_artifact_path(manifest_relative)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "binary_stream_v1",
                "side": "left",
                "relative_dir": relative_dir,
                "frame_count": 2,
                "sensor_keys": ["M_RAW"],
                "streams": {
                    "M_RAW": {
                        "path": payload_relative,
                        "dtype": "uint8",
                        "shape": [4],
                        "frame_count": 2,
                        "encoding": "raw_c_order",
                        "compression": "none",
                        "chunk_frame_count": None,
                        "chunks": [],
                        "stored_size_bytes": len(payload),
                        "stored_sha256": hashlib.sha256(payload).hexdigest(),
                        "stored_hash_profile": "stored_bytes_sha256_v1",
                    }
                },
                "frames": frames,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "frame_index": index,
            "observation.hand.left.tactile_path": manifest_relative,
            "observation.hand.left.tactile_frame_index": index,
        }
        for index in range(2)
    ]
    report = store.seal_episode(
        identity=identity,
        rows=rows,
        tactile_paths={"left": manifest_relative},
    )
    catalog_path = core / "meta" / "episodes.jsonl"
    catalog_rows = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog_rows.append(
        {
            "episode_index": episode_index,
            "episode_id": identity.episode_uuid,
            "frame_count": 2,
            "tactile_paths": {"left": manifest_relative},
            "tactile_bundle": report.episode_binding_payload(),
        }
    )
    catalog_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            for row in catalog_rows
        ),
        encoding="utf-8",
    )
    return payload_path


def _episode_metadata_paths(core: Path, storage: Path, episode_index: int = 0) -> dict[str, Path]:
    attached = attach_tactile_bundle(core, storage)
    commit = attached.episodes[episode_index].commit
    bundle = storage / "datasets" / str(DATASET_UUID)
    component = bundle / commit.tactile_component_ref
    return {
        "component": component,
        "tactile_commit": component.with_name("bundle_commit.cjson"),
        "core_commit": (core / commit.core_component_ref).with_name("bundle_commit.cjson"),
        "catalog": core / "meta/episodes.jsonl",
    }


def _write_with_fault(write, path, payload, *, stage, error):
    real_sync, real_replace = os.fsync, os.replace

    def sync(fd):
        if stage == "directory_sync" and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise error
        real_sync(fd)

    def replace(source, target):
        if stage == "replace" and Path(target) == path:
            raise error
        real_replace(source, target)

    with (
        mock.patch.object(os, "fsync", side_effect=sync),
        mock.patch.object(os, "replace", side_effect=replace),
    ):
        write(path, payload)


def _assert_compressed_payload(bundle: Path, manifest: bytes, expected: bytes) -> None:
    spec = json.loads(manifest)["streams"]["M_RAW"]
    assert spec["compression"] == "zstd"
    reader = ChunkedTactileStream(dataset_root=bundle, spec=spec)
    assert b"".join(reader.frame(index).tobytes() for index in range(2)) == expected


@pytest.mark.parametrize("failed_file", ["manifest", "component", "tactile_commit", "core_commit", "catalog"])
def test_native_publication_failure_restores_attachment_before_deleting_payload(
    tmp_path, monkeypatch, failed_file
):
    core, storage, raw = _native_dataset(tmp_path)
    manifest = raw.with_name("manifest.json")
    paths = _episode_metadata_paths(core, storage)
    paths["manifest"] = manifest
    before = {path: path.read_bytes() for path in paths.values()}
    raw_before = raw.read_bytes()
    injected = OSError(errno.EIO, "publication directory sync")
    entrypoint = "_atomic_write_json" if failed_file == "manifest" else "_atomic_write_bytes"
    original_write = getattr(native_recompression_module, entrypoint)
    faulted = False

    def publish(path, payload):
        nonlocal faulted
        if path == paths[failed_file] and not faulted:
            faulted = True
            return _write_with_fault(
                original_write, path, payload, stage="directory_sync", error=injected
            )
        return original_write(path, payload)

    monkeypatch.setattr(native_recompression_module, entrypoint, publish)
    with pytest.raises(OSError) as caught:
        recompress_native_tactile_bundle_zstd_in_place(
            core, storage, min_free_bytes=0, backend="python"
        )
    assert caught.value is injected
    assert faulted
    assert {path: path.read_bytes() for path in before} == before
    assert raw.read_bytes() == raw_before
    assert not raw.with_name(f"{raw.name}.tdelta8.zst").exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    attach_tactile_bundle(core, storage)
    result = recompress_native_tactile_bundle_zstd_in_place(
        core, storage, min_free_bytes=0, backend="python"
    )
    assert result["episode_count"] == 1
    assert not raw.exists()
    attach_tactile_bundle(core, storage)
    _assert_compressed_payload(
        storage / "datasets" / str(DATASET_UUID), manifest.read_bytes(), raw_before
    )


@pytest.mark.parametrize("restore_file", ["component", "tactile_commit", "core_commit", "catalog"])
@pytest.mark.parametrize("stage", ["replace", "directory_sync"])
def test_native_metadata_restore_failure_stops_payload_cleanup(
    tmp_path, monkeypatch, restore_file, stage
):
    core, storage, raw = _native_dataset(tmp_path)
    paths = _episode_metadata_paths(core, storage)
    before = {path: path.read_bytes() for path in paths.values()}
    manifest = raw.with_name("manifest.json")
    raw_before = raw.read_bytes()
    published = {}
    restored = []
    validation_error = OSError(errno.EIO, "post-publication validation")
    restore_error = OSError(errno.EIO, "metadata restore failed")
    original_write = native_recompression_module._atomic_write_bytes
    original_attach = native_recompression_module.attach_tactile_bundle
    attachments = 0

    def attach(*args, **kwargs):
        nonlocal attachments
        result = original_attach(*args, **kwargs)
        attachments += 1
        if attachments == 2:
            published.update({path: path.read_bytes() for path in before})
            published[manifest] = manifest.read_bytes()
            raise validation_error
        return result

    def write(path, payload):
        if published:
            restored.append(path)
            if path == paths[restore_file]:
                assert payload == before[path]
                return _write_with_fault(
                    original_write, path, payload, stage=stage, error=restore_error
                )
        return original_write(path, payload)

    monkeypatch.setattr(native_recompression_module, "attach_tactile_bundle", attach)
    monkeypatch.setattr(native_recompression_module, "_atomic_write_bytes", write)
    with pytest.raises(OSError) as caught:
        recompress_native_tactile_bundle_zstd_in_place(
            core, storage, min_free_bytes=0, backend="python"
        )
    assert caught.value is restore_error
    assert restore_error.__context__ is validation_error
    assert attachments == 2
    assert raw.read_bytes() == raw_before
    assert raw.with_name(f"{raw.name}.tdelta8.zst").is_file()
    assert manifest.read_bytes() == published[manifest]
    ordered = list(paths.values())
    failed_index = ordered.index(paths[restore_file])
    assert restored == ordered[:failed_index + 1]
    for index, path in enumerate(ordered):
        was_restored = index < failed_index or (index == failed_index and stage == "directory_sync")
        assert path.read_bytes() == (before[path] if was_restored else published[path])
    _assert_compressed_payload(
        storage / "datasets" / str(DATASET_UUID), published[manifest], raw_before
    )
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.parametrize("verify_history", [True, False])
def test_native_multiple_episodes_keep_cumulative_catalog_updates(tmp_path, verify_history):
    core, storage, first_raw = _native_dataset(tmp_path)
    second_raw = _append_native_episode(core, storage, episode_index=1)
    raw_bytes = {path: path.read_bytes() for path in (first_raw, second_raw)}
    catalog = core / "meta/episodes.jsonl"
    rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metadata"] = {"tactile_bundle_episode": dict(row["tactile_bundle"])}
    catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = attach_tactile_bundle(core, storage)
    result = recompress_native_tactile_bundle_zstd_in_place(
        core, storage, min_free_bytes=0, backend="python",
        verify_historical_payload_hashes=verify_history,
    )
    assert result["episode_count"] == result["manifest_count"] == result["stream_count"] == 2
    assert result["removed_original_bytes"] == 16
    attached = attach_tactile_bundle(core, storage)
    rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()]
    for index, raw in enumerate((first_raw, second_raw)):
        assert not raw.exists()
        assert attached.episodes[index].commit != before.episodes[index].commit
        expected_hash = attached.episodes[index].commit.semantic_content_hash
        assert rows[index]["tactile_bundle"]["commit_hash"] == expected_hash
        assert rows[index]["metadata"]["tactile_bundle_episode"]["commit_hash"] == expected_hash
        _assert_compressed_payload(
            storage / "datasets" / str(DATASET_UUID),
            raw.with_name("manifest.json").read_bytes(), raw_bytes[raw],
        )


def test_second_episode_rollback_keeps_the_first_episode_committed(tmp_path, monkeypatch):
    core, storage, first_raw = _native_dataset(tmp_path)
    second_raw = _append_native_episode(core, storage, episode_index=1)
    first_bytes, second_bytes = first_raw.read_bytes(), second_raw.read_bytes()
    before = attach_tactile_bundle(core, storage)
    paths = _episode_metadata_paths(core, storage, 1)
    original_write = native_recompression_module._atomic_write_bytes
    injected = OSError(errno.EIO, "second episode commit sync")
    catalog_before_second = None

    def publish(path, payload):
        nonlocal catalog_before_second
        if path == paths["component"] and catalog_before_second is None:
            catalog_before_second = paths["catalog"].read_bytes()
            return _write_with_fault(
                original_write, path, payload, stage="directory_sync", error=injected
            )
        return original_write(path, payload)

    monkeypatch.setattr(native_recompression_module, "_atomic_write_bytes", publish)
    with pytest.raises(OSError) as caught:
        recompress_native_tactile_bundle_zstd_in_place(
            core, storage, min_free_bytes=0, backend="python"
        )
    assert caught.value is injected
    assert paths["catalog"].read_bytes() == catalog_before_second
    attached = attach_tactile_bundle(core, storage)
    assert attached.episodes[0].commit != before.episodes[0].commit
    assert attached.episodes[1].commit == before.episodes[1].commit
    assert not first_raw.exists()
    assert second_raw.read_bytes() == second_bytes
    assert not second_raw.with_name(f"{second_raw.name}.tdelta8.zst").exists()
    _assert_compressed_payload(
        storage / "datasets" / str(DATASET_UUID),
        first_raw.with_name("manifest.json").read_bytes(), first_bytes,
    )


@pytest.mark.parametrize("failed_read", ["catalog", "core_commit"])
def test_early_second_episode_failure_does_not_reuse_previous_recovery_state(
    tmp_path, monkeypatch, failed_read
):
    core, storage, first_raw = _native_dataset(tmp_path)
    second_raw = _append_native_episode(core, storage, episode_index=1)
    before = attach_tactile_bundle(core, storage)
    paths = _episode_metadata_paths(core, storage, 1)
    original_read = Path.read_bytes
    original_write = native_recompression_module._atomic_write_bytes
    injected = OSError(errno.EIO, "second episode backup read")
    faulted = False
    restored = []

    def read(path):
        nonlocal faulted
        if path == paths[failed_read] and not first_raw.exists() and not faulted:
            faulted = True
            raise injected
        return original_read(path)

    def write(path, payload):
        if faulted:
            restored.append(path)
        return original_write(path, payload)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(native_recompression_module, "_atomic_write_bytes", write)
    with pytest.raises(OSError) as caught:
        recompress_native_tactile_bundle_zstd_in_place(
            core, storage, min_free_bytes=0, backend="python"
        )
    assert caught.value is injected
    assert faulted
    assert restored == ([] if failed_read == "catalog" else [second_raw.with_name("manifest.json")])
    attached = attach_tactile_bundle(core, storage)
    assert attached.episodes[0].commit != before.episodes[0].commit
    assert attached.episodes[1].commit == before.episodes[1].commit
    assert not first_raw.exists()
    assert second_raw.read_bytes() == bytes(range(9, 17))
    assert not second_raw.with_name(f"{second_raw.name}.tdelta8.zst").exists()


def test_raw_removal_failure_keeps_the_verified_new_metadata(tmp_path, monkeypatch):
    core, storage, raw = _native_dataset(tmp_path)
    raw_before = raw.read_bytes()
    before = attach_tactile_bundle(core, storage)
    original_unlink = Path.unlink
    injected = PermissionError(errno.EACCES, "raw removal blocked")

    def unlink(path, *args, **kwargs):
        if path == raw:
            raise injected
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(PermissionError) as caught:
        recompress_native_tactile_bundle_zstd_in_place(
            core, storage, min_free_bytes=0, backend="python"
        )
    assert caught.value is injected
    assert raw.read_bytes() == raw_before
    attached = attach_tactile_bundle(core, storage)
    assert attached.episodes[0].commit != before.episodes[0].commit
    _assert_compressed_payload(
        storage / "datasets" / str(DATASET_UUID),
        raw.with_name("manifest.json").read_bytes(), raw_before,
    )


def test_native_idle_recompression_updates_bundle_commit_and_attachment() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, raw_payload = _native_dataset(Path(temporary))
        before = attach_tactile_bundle(core, storage)
        result = recompress_native_tactile_bundle_zstd_in_place(
            core,
            storage,
            min_free_bytes=0,
            backend="python",
        )

        assert result["outcome"] == "completed"
        assert result["removed_original_bytes"] == 8
        assert not raw_payload.exists()
        attached = attach_tactile_bundle(core, storage)
        assert attached.report.payload_bytes > 0
        assert attached.report.payload_bytes != before.report.payload_bytes
        manifest_path = (
            storage
            / "datasets"
            / str(DATASET_UUID)
            / "tactile/chunk-000/episode_000000/left/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["streams"]["M_RAW"]["compression"] == "zstd"


def test_bounded_idle_recompression_does_not_rehash_historical_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, episode_zero_raw = _native_dataset(Path(temporary))
        recompress_native_tactile_bundle_zstd_in_place(
            core,
            storage,
            min_free_bytes=0,
            backend="python",
        )
        assert not episode_zero_raw.exists()
        episode_one_raw = _append_native_episode(core, storage, episode_index=1)
        payload_hash_paths: list[Path] = []
        original_sha256 = tactile_bundle_module._sha256_file

        def record_payload_hash(path: Path) -> str:
            if path.name != "manifest.json":
                payload_hash_paths.append(path)
            return original_sha256(path)

        monkeypatch.setattr(
            tactile_bundle_module,
            "_sha256_file",
            record_payload_hash,
        )
        result = recompress_native_tactile_bundle_zstd_in_place(
            core,
            storage,
            min_free_bytes=0,
            backend="python",
            max_episodes=1,
            verify_historical_payload_hashes=False,
        )

        assert result["outcome"] == "completed"
        assert result["episode_count"] == 1
        assert result["max_episodes"] == 1
        assert result["historical_payload_hashes_verified"] is False
        assert not episode_one_raw.exists()
        assert payload_hash_paths
        assert all("episode_000001" in str(path) for path in payload_hash_paths)
        attach_tactile_bundle(core, storage)


def test_metadata_attachment_can_verify_only_selected_payload_episodes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, payload = _native_dataset(Path(temporary))
        attached = attach_tactile_bundle(
            core,
            storage,
            verify_payload_hashes=False,
        )

        assert attached.verify_payload_hashes_for_episodes((0,)) == (0,)
        payload.write_bytes(b"corrupt!")
        with pytest.raises(RuntimeError, match="hash mismatch"):
            attached.verify_payload_hashes_for_episodes((0,))


def test_idle_native_worker_uses_bounded_incremental_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_recompress(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        calls.update(kwargs)
        return {"outcome": "no_work"}

    monkeypatch.setattr(
        native_recompression_module,
        "recompress_native_tactile_bundle_zstd_in_place",
        fake_recompress,
    )

    class _StopEvent:
        def is_set(self) -> bool:
            return False

    result = run_idle_native_tactile_recompression_worker(
        "/tmp/core",
        _StopEvent(),
        "/tmp/storage",
        min_free_bytes=0,
    )

    assert result == {"outcome": "no_work"}
    assert calls["max_episodes"] == 1
    assert calls["verify_historical_payload_hashes"] is False


def test_native_idle_recompression_cancel_is_before_bundle_commit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, raw_payload = _native_dataset(Path(temporary))
        with pytest.raises(TactileRecompressionCancelled):
            recompress_native_tactile_bundle_zstd_in_place(
                core,
                storage,
                min_free_bytes=0,
                backend="python",
                cancel_requested=lambda: True,
            )
        assert raw_payload.exists()
        attach_tactile_bundle(core, storage)


def test_native_idle_recompression_cleans_uncommitted_target_without_raw_loss() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, raw_payload = _native_dataset(Path(temporary))
        orphan = raw_payload.with_name(f"{raw_payload.name}.tdelta8.zst")
        orphan.write_bytes(b"orphan")
        with pytest.raises(TactileRecompressionCancelled):
            recompress_native_tactile_bundle_zstd_in_place(
                core,
                storage,
                min_free_bytes=0,
                backend="python",
                cancel_requested=lambda: True,
            )
        assert raw_payload.exists()
        assert not orphan.exists()


def test_native_bundle_resolution_is_bound_to_core_dataset_uuid() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, _raw_payload = _native_dataset(Path(temporary))
        resolved = resolve_native_bundle_dataset_root(
            core,
            storage,
        )
        assert resolved == (storage / "datasets" / str(DATASET_UUID)).resolve()


def test_legacy_idle_worker_entry_point_dispatches_native_bundle() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        core, storage, _raw_payload = _native_dataset(Path(temporary))

        class _StopEvent:
            def is_set(self) -> bool:
                return False

        result = run_idle_tactile_recompression_worker(
            str(core),
            _StopEvent(),
            str(storage),
            native_min_free_bytes=0,
        )
        assert result["outcome"] == "completed"

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from collection_runtime.v25p_native_store import (
    V25pComponentInput,
    V25pNativeEpisodeStore,
)
from collection_runtime.v25p_producer import V25pProducedSample
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.v25p_native_storage import V25P_PARQUET_CHUNK_LAYOUT_V1
from dataset_interface.v25p_native import V25pNativeReadError, load_v25p_native_episode
from dataset_interface.v25p_v23_adapter import project_native_episode_to_v23_rows
from teleop_runtime.collection import v25p_dual_write_shadow as shadow_module
from teleop_runtime.collection.v25p_capture_profiles import (
    V25P_REAL_TELEOP_SOURCE,
    build_policy_io_v25p_capture_profiles,
)
from teleop_runtime.collection.v25p_dual_write_shadow import (
    V25pDualWriteShadow,
    append_shadow_from_legacy_jobs,
    finalize_shadow_after_legacy,
)

POSE_HASH = canonical_content_hash(
    {"name": "shadow_test_pose_frames"},
    domain="test.v25p.dual_write_shadow.v1",
)


class _Producer:
    def __init__(self, bundle) -> None:
        self.bundle = bundle

    def produce(self, sample: dict[str, object]) -> V25pProducedSample:
        timestamp = float(sample["timestamp"])
        offset = int(sample["frame_index"]) / 16.0
        inputs = {}
        for stream in self.bundle.streams.streams:
            if stream.representation_family in {"xyz_quat_xyzw", "cartesian_pose"}:
                side_offset = 0.0 if stream.stream_id.endswith(".left") else 1.0
                values = np.asarray(
                    [side_offset + offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            elif stream.representation_family == "joint_position":
                values = np.arange(6, dtype=np.float32) + offset
            else:  # pragma: no cover - the no-hand test layout has no other streams
                raise AssertionError(stream.stream_id)
            inputs[stream.stream_id] = V25pComponentInput(
                values=values,
                effective_at=timestamp,
                available_at=timestamp,
                effective_clock_domain_id="test_clock",
                available_clock_domain_id="test_clock",
            )
        return V25pProducedSample(
            sample_index=int(sample["frame_index"]),
            cut_at=timestamp,
            cut_clock_domain_id="test_clock",
            values_by_stream=inputs,
        )


def _setup(
    tmp_path: Path,
    *,
    physical_storage_layout: str = "record_cjson_v1",
    parquet_chunk_rows: int = 128,
):
    bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_REAL_TELEOP_SOURCE,
        layout=PolicyIOLayout(name="shadow_no_end_effector"),
        pose_frame_contract_hash=POSE_HASH,
    )
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="shadow-dataset",
        embodiment_profile=bundle.embodiment,
        component_stream_profile=bundle.streams,
        compatibility_projection_profiles=(bundle.v23_projection,),
        defer_sample_file_writes=(physical_storage_layout == "record_cjson_v1"),
        physical_storage_layout=physical_storage_layout,
        parquet_chunk_rows=parquet_chunk_rows,
    )
    shadow = V25pDualWriteShadow(
        store=store,
        bundle=bundle,
        producer=_Producer(bundle),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="shadow-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="test.authoritative_owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    return bundle, shadow, identity


def _legacy_row(
    *,
    frame_index: int = 0,
    timestamp: float = 3.0,
    drift: bool = False,
) -> dict[str, object]:
    offset = frame_index / 16.0
    left_x = offset + (0.25 if drift else 0.0)
    return {
        "episode_index": 0,
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "action.left_arm.pose": [left_x, 0, 0, 0, 0, 0, 1],
        "observation.left_arm.pose": [offset, 0, 0, 0, 0, 0, 1],
        "action.right_arm.pose": [1 + offset, 0, 0, 0, 0, 0, 1],
        "observation.right_arm.pose": [1 + offset, 0, 0, 0, 0, 0, 1],
    }


def _publish_legacy_episode_record(root: Path, row: dict[str, object]) -> None:
    path = root / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("drift", [False, True])
def test_owner_helpers_preserve_native_capture_and_projection_receipt(
    tmp_path: Path, drift: bool,
) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(identity, bound_at=2.0, bound_clock_domain_id="test_clock")
    row = _legacy_row(drift=drift)
    job = SimpleNamespace(kind="lerobot_v21.row", payload={"row": row})

    assert append_shadow_from_legacy_jobs(
        shadow, identity,
        source_sample={"frame_index": 0, "timestamp": 3.0}, jobs=(job,),
    ) == ""
    _publish_legacy_episode_record(tmp_path, row)
    report, error = finalize_shadow_after_legacy(
        shadow, identity, dataset_root=tmp_path, episode_index=0,
    )

    assert report is not None
    assert report.receipt.status == ("failed" if drift else "complete")
    assert error == (report.receipt.message if drift else None)
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.identity == identity
    assert loaded.native_capture_manifest.sample_count == 1
    assert loaded.compatibility_projection_receipts == (report.receipt,)
    assert json.loads((tmp_path / "meta/episodes.jsonl").read_text()) == row


@pytest.mark.parametrize("failure", [None, RuntimeError("append failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("close_failure", [None, OSError("close failed")])
def test_append_helper_observes_exact_planned_row_and_preserves_failures(
    failure, close_failure,
) -> None:
    identity, sample, row = object(), object(), {"frame_index": 7}
    shadow = Mock()
    shadow.append.side_effect = failure
    shadow.close_unsealed.side_effect = close_failure
    jobs = (
        SimpleNamespace(kind="video", payload={}),
        SimpleNamespace(kind="lerobot_v21.row", payload={"row": row}),
    )

    error = append_shadow_from_legacy_jobs(
        shadow, identity, source_sample=sample, jobs=jobs,
    )

    shadow.append.assert_called_once_with(identity, source_sample=sample, legacy_row=row)
    assert shadow.append.call_args.kwargs["legacy_row"] is row
    if failure is None:
        assert error == ""
        shadow.close_unsealed.assert_not_called()
    else:
        expected = f"append:{type(failure).__name__}: {failure}"
        if close_failure is not None:
            expected += f";close_unsealed:OSError: {close_failure}"
        assert error == expected
        shadow.close_unsealed.assert_called_once_with(identity)


@pytest.mark.parametrize("row_count", [0, 2])
def test_append_helper_rejects_ambiguous_jobs_without_appending(row_count: int) -> None:
    shadow, identity = Mock(), object()
    job = SimpleNamespace(kind="lerobot_v21.row", payload={"row": {}})
    error = append_shadow_from_legacy_jobs(
        shadow, identity, source_sample=object(), jobs=(job,) * row_count,
    )
    assert error == f"append:ValueError: Expected exactly one legacy row job, found {row_count}."
    shadow.append.assert_not_called()
    shadow.close_unsealed.assert_called_once_with(identity)


@pytest.mark.parametrize(
    ("status", "message", "fallback", "expected"),
    [
        ("complete", "", "unused", None),
        ("failed", "projection differs", "unused", "projection differs"),
        ("failed", "", "v2.5-p compatibility projection failed", "v2.5-p compatibility projection failed"),
        ("failed", "", "", ""),
    ],
)
def test_finalize_helper_preserves_receipt_and_owner_message_policy(
    tmp_path: Path, status, message, fallback, expected,
) -> None:
    record = {"episode_index": 7, "length": 1}
    _publish_legacy_episode_record(tmp_path, record)
    report = SimpleNamespace(receipt=SimpleNamespace(status=status, message=message))
    shadow, identity = Mock(), object()
    shadow.finalize_after_legacy.return_value = report
    published_reports = []
    published_errors = []

    actual, error = finalize_shadow_after_legacy(
        shadow, identity, dataset_root=tmp_path, episode_index=7,
        incomplete_message=fallback,
        record_report=published_reports.append,
        record_error=published_errors.append,
    )

    assert actual is report
    assert published_reports == [report]
    assert published_errors == ([] if expected is None else [expected])
    assert error == expected
    shadow.finalize_after_legacy.assert_called_once()
    args, kwargs = shadow.finalize_after_legacy.call_args
    assert args == (identity,)
    assert kwargs["legacy_episode_record"] == record
    assert kwargs["completed_clock_domain_id"] == "local_wall_clock"
    shadow.close_unsealed.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["load", "finalize"])
@pytest.mark.parametrize("same_identity", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
@pytest.mark.parametrize("publish_error", [False, True])
def test_finalize_helper_cleans_only_failed_identity_after_recording_timing(
    tmp_path: Path, monkeypatch, failure_stage, same_identity, close_failure, publish_error,
) -> None:
    identity = object()
    shadow = Mock(active_identity=identity if same_identity else object())
    events = []

    def close(active_identity):
        assert active_identity is identity
        events.append("close")
        if close_failure:
            raise OSError("cleanup failed")

    shadow.close_unsealed.side_effect = close
    shadow.finalize_after_legacy.side_effect = RuntimeError("seal failed")
    loader = Mock(return_value={"episode_index": 7})
    if failure_stage == "load":
        loader.side_effect = ValueError("record missing")
    monkeypatch.setattr(shadow_module, "load_finalized_legacy_episode_record", loader)
    monkeypatch.setattr(shadow_module.time, "perf_counter", Mock(side_effect=[10.0, 10.25]))

    report, error = finalize_shadow_after_legacy(
        shadow, identity, dataset_root=tmp_path, episode_index=7,
        error_prefix="post_finalize",
        record_finalize_elapsed=lambda elapsed: events.append(("timing", elapsed)),
        record_error=(lambda error: events.append(("error", error))) if publish_error else None,
    )

    assert report is None
    expected = "post_finalize:" + (
        "ValueError: record missing" if failure_stage == "load" else "RuntimeError: seal failed"
    )
    expected_events = [("timing", 0.25)]
    if publish_error:
        expected_events.append(("error", expected))
    if same_identity:
        expected_events.append("close")
    if same_identity and close_failure:
        expected += ";close_unsealed:OSError: cleanup failed"
        if publish_error:
            expected_events.append(("error", expected))
    assert error == expected
    assert events == expected_events
    assert shadow.finalize_after_legacy.call_count == int(failure_stage == "finalize")
    assert shadow.close_unsealed.call_count == int(same_identity)


def _assert_persisted_projection(loaded, bundle, *, sample_count: int) -> None:
    projected = project_native_episode_to_v23_rows(loaded, bundle.v23_projection)
    assert len(projected.rows) == sample_count
    for frame_index, row in enumerate(projected.rows):
        expected = _legacy_row(
            frame_index=frame_index, timestamp=3.0 + frame_index / 15.0
        )
        assert row.sample_index == frame_index
        assert row.cut_at == expected["timestamp"]
        assert row.cut_clock_domain_id == "test_clock"
        assert dict(row.fields) == {
            field: tuple(expected[field])
            for field in (
                "action.left_arm.pose",
                "observation.left_arm.pose",
                "action.right_arm.pose",
                "observation.right_arm.pose",
            )
        }


def test_shadow_uses_owner_identity_and_commits_complete_receipt(tmp_path: Path) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="test_clock",
    )
    row = _legacy_row()
    append = shadow.append(
        identity,
        source_sample={"frame_index": 0, "timestamp": 3.0},
        legacy_row=row,
    )
    assert append.equivalent
    _publish_legacy_episode_record(tmp_path, row)
    finalized = shadow.finalize_after_legacy(
        identity,
        legacy_episode_record=row,
        completed_at=4.0,
        completed_clock_domain_id="test_clock",
    )
    assert finalized.receipt.status == "complete"
    assert finalized.receipt.lossy
    assert set(finalized.receipt.omitted_stream_ids) == {
        "real_teleop.state.qpos.left",
        "real_teleop.state.qpos.right",
    }
    assert finalized.commit.v24_capture_commit_hash is None
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.identity == identity
    assert loaded.compatibility_projection_receipts == (finalized.receipt,)


def test_shadow_mismatch_keeps_native_capture_and_publishes_failed_receipt(
    tmp_path: Path,
) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="test_clock",
    )
    row = _legacy_row(drift=True)
    append = shadow.append(
        identity,
        source_sample={"frame_index": 0, "timestamp": 3.0},
        legacy_row=row,
    )
    assert not append.equivalent
    assert "action.left_arm.pose" in append.mismatch_fields
    _publish_legacy_episode_record(tmp_path, row)
    finalized = shadow.finalize_after_legacy(
        identity,
        legacy_episode_record=row,
        completed_at=4.0,
        completed_clock_domain_id="test_clock",
    )
    assert finalized.receipt.status == "failed"
    assert finalized.receipt.target_episode_locator is None
    assert finalized.mismatched_sample_count == 1
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.native_capture_manifest.sample_count == 1


@pytest.mark.slow_sim
def test_shadow_multiframe_episode_preserves_persisted_projection_for_every_sample(
    tmp_path: Path,
) -> None:
    bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="test_clock",
    )
    # Exercise zero/one-based records and the 9/10 filename transition with
    # distinct payloads. Large-volume chunk coverage remains in the Parquet test.
    sample_count = 12
    for frame_index in range(sample_count):
        timestamp = 3.0 + frame_index / 15.0
        append = shadow.append(
            identity,
            source_sample={
                "frame_index": frame_index,
                "timestamp": timestamp,
            },
            legacy_row=_legacy_row(
                frame_index=frame_index,
                timestamp=timestamp,
            ),
        )
        assert append.equivalent

    legacy_record = {
        "episode_index": identity.legacy_episode_index,
        "length": sample_count,
    }
    _publish_legacy_episode_record(tmp_path, legacy_record)
    finalized = shadow.finalize_after_legacy(
        identity,
        legacy_episode_record=legacy_record,
        completed_at=21.0,
        completed_clock_domain_id="test_clock",
    )
    assert finalized.receipt.status == "complete"
    assert finalized.equivalent_sample_count == sample_count
    assert finalized.mismatched_sample_count == 0
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.native_capture_manifest.sample_count == sample_count
    assert len(loaded.samples) == sample_count
    _assert_persisted_projection(loaded, bundle, sample_count=sample_count)


@pytest.mark.parametrize("mismatch_index", (5, 11), ids=("middle", "last"))
def test_shadow_late_mismatch_preserves_capture_and_resets_next_episode(
    tmp_path: Path,
    mismatch_index: int,
) -> None:
    bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity, bound_at=2.0, bound_clock_domain_id="test_clock"
    )
    for frame_index in range(12):
        timestamp = 3.0 + frame_index / 15.0
        report = shadow.append(
            identity,
            source_sample={"frame_index": frame_index, "timestamp": timestamp},
            legacy_row=_legacy_row(
                frame_index=frame_index,
                timestamp=timestamp,
                drift=frame_index == mismatch_index,
            ),
        )
        assert report.sample_index == frame_index
        assert report.equivalent is (frame_index != mismatch_index)
        assert report.mismatch_fields == (
            ("action.left_arm.pose",) if frame_index == mismatch_index else ()
        )
    legacy_record = {"episode_index": 0, "length": 12}
    _publish_legacy_episode_record(tmp_path, legacy_record)
    finalized = shadow.finalize_after_legacy(
        identity,
        legacy_episode_record=legacy_record,
        completed_at=5.0,
        completed_clock_domain_id="test_clock",
    )
    assert finalized.receipt.status == "failed"
    assert finalized.receipt.target_episode_locator is None
    assert finalized.equivalent_sample_count == 11
    assert finalized.mismatched_sample_count == 1
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.native_capture_manifest.sample_count == 12
    _assert_persisted_projection(loaded, bundle, sample_count=12)
    assert shadow.active_identity is None
    assert shadow.store.active_identity is None

    # Reuse the actual wrapper/store: a failed parity report must not leak into
    # the next episode, and its sample index must restart from zero.
    second = replace(
        identity, episode_uuid="shadow-next", legacy_episode_index=1, prepared_at=6.0
    )
    shadow.prepare_episode(second)
    shadow.bind_prepared_episode(
        second, bound_at=7.0, bound_clock_domain_id="test_clock"
    )
    row = _legacy_row(timestamp=8.0)
    row["episode_index"] = 1
    report = shadow.append(
        second,
        source_sample={"frame_index": 0, "timestamp": 8.0},
        legacy_row=row,
    )
    assert report.equivalent
    _publish_legacy_episode_record(tmp_path, row)
    next_result = shadow.finalize_after_legacy(
        second,
        legacy_episode_record=row,
        completed_at=9.0,
        completed_clock_domain_id="test_clock",
    )
    assert next_result.receipt.status == "complete"
    assert next_result.equivalent_sample_count == 1
    assert next_result.mismatched_sample_count == 0
    next_loaded = load_v25p_native_episode(tmp_path, second.episode_uuid)
    assert next_loaded.identity == second
    assert len(next_loaded.samples) == 1
    assert next_loaded.samples[0].sample_index == 0


@pytest.mark.parametrize("fail_directory", (False, True), ids=("file", "directory"))
def test_shadow_seal_sync_failure_never_publishes_complete_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_directory: bool,
) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity, bound_at=2.0, bound_clock_domain_id="test_clock"
    )
    row = _legacy_row()
    assert shadow.append(
        identity,
        source_sample={"frame_index": 0, "timestamp": 3.0},
        legacy_row=row,
    ).equivalent
    _publish_legacy_episode_record(tmp_path, row)
    real_fsync = os.fsync
    sync_count = 0

    def fail_second_sync(fd: int) -> None:
        nonlocal sync_count
        if stat.S_ISDIR(os.fstat(fd).st_mode) == fail_directory:
            sync_count += 1
            if sync_count == 2:
                raise OSError("injected native seal sync failure")
        real_fsync(fd)

    # Leave earlier writes real; fail after a durable prefix has been published.
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_second_sync)
        with pytest.raises(OSError, match="injected native seal sync failure"):
            shadow.finalize_after_legacy(
                identity,
                legacy_episode_record=row,
                completed_at=4.0,
                completed_clock_domain_id="test_clock",
            )
    assert sync_count == 2
    episode_dir = tmp_path / "meta/v25p/native/episodes" / identity.episode_uuid
    assert tuple((episode_dir / "streams").glob("*/*.cjson"))
    assert not (episode_dir / "native_capture_manifest.cjson").exists()
    assert not (episode_dir / "episode_manifest.cjson").exists()
    assert not (episode_dir / "compatibility_projections").exists()
    assert shadow.active_identity == identity
    assert shadow.store.active_identity == identity
    with pytest.raises(V25pNativeReadError, match="missing"):
        load_v25p_native_episode(tmp_path, identity.episode_uuid)


@pytest.mark.optional
def test_parquet_shadow_long_episode_preserves_v23_projection_and_chunks(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    bundle, shadow, identity = _setup(
        tmp_path,
        physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        parquet_chunk_rows=64,
    )
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="test_clock",
    )
    sample_count = 256
    for frame_index in range(sample_count):
        timestamp = 3.0 + frame_index / 15.0
        append = shadow.append(
            identity,
            source_sample={"frame_index": frame_index, "timestamp": timestamp},
            legacy_row=_legacy_row(
                frame_index=frame_index,
                timestamp=timestamp,
            ),
        )
        assert append.equivalent

    legacy_record = {
        "episode_index": identity.legacy_episode_index,
        "length": sample_count,
    }
    _publish_legacy_episode_record(tmp_path, legacy_record)
    finalized = shadow.finalize_after_legacy(
        identity,
        legacy_episode_record=legacy_record,
        completed_at=21.0,
        completed_clock_domain_id="test_clock",
    )

    assert finalized.receipt.status == "complete"
    assert finalized.mismatched_sample_count == 0
    assert finalized.seal.native_storage_manifest_hash is not None
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert len(loaded.samples) == sample_count
    episode_dir = tmp_path / "meta/v25p/native/episodes" / identity.episode_uuid
    assert len(tuple((episode_dir / "chunks").glob("*.cjson"))) == 4
    _assert_persisted_projection(loaded, bundle, sample_count=sample_count)


def test_shadow_discard_is_subordinate_and_close_unsealed_preserves_audit(
    tmp_path: Path,
) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
    shadow.prepare_episode(identity)
    shadow.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="test_clock",
    )
    shadow.discard_episode(identity)
    episode_dir = tmp_path / "meta" / "v25p" / "native" / "episodes" / identity.episode_uuid
    assert not episode_dir.exists()

    second = EpisodeCaptureIdentity(
        episode_uuid="shadow-interrupted",
        legacy_episode_index=1,
        lifecycle_owner_id="test.authoritative_owner",
        prepared_at=5.0,
        prepared_clock_domain_id="test_clock",
    )
    shadow.prepare_episode(second)
    shadow.bind_prepared_episode(
        second,
        bound_at=6.0,
        bound_clock_domain_id="test_clock",
    )
    shadow.close_unsealed(second)
    interrupted = episode_dir.parent / second.episode_uuid
    assert (interrupted / "identity.cjson").is_file()
    assert not (interrupted / "episode_manifest.cjson").exists()

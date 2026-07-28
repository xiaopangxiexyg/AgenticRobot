from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.v25p_native_storage import V25P_PARQUET_CHUNK_LAYOUT_V1
from dataset_interface.v25p_native import load_v25p_native_episode
from teleop_runtime.collection.v25p_capture_profiles import (
    V25P_REAL_TELEOP_SOURCE,
    build_policy_io_v25p_capture_profiles,
)
from teleop_runtime.collection.v25p_dual_write_shadow import V25pDualWriteShadow
from teleop_runtime.collection.v25p_native_store import (
    V25pComponentInput,
    V25pNativeEpisodeStore,
)
from teleop_runtime.collection.v25p_producer import V25pProducedSample


POSE_HASH = canonical_content_hash(
    {"name": "shadow_test_pose_frames"},
    domain="test.v25p.dual_write_shadow.v1",
)


class _Producer:
    def __init__(self, bundle) -> None:
        self.bundle = bundle

    def produce(self, sample: dict[str, object]) -> V25pProducedSample:
        timestamp = float(sample["timestamp"])
        inputs = {}
        for stream in self.bundle.streams.streams:
            if stream.representation_family in {"xyz_quat_xyzw", "cartesian_pose"}:
                side_offset = 0.0 if stream.stream_id.endswith(".left") else 1.0
                values = np.asarray(
                    [side_offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            elif stream.representation_family == "joint_position":
                values = np.arange(6, dtype=np.float32)
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
    left_x = 0.25 if drift else 0.0
    return {
        "episode_index": 0,
        "frame_index": int(frame_index),
        "timestamp": float(timestamp),
        "action.left_arm.pose": [left_x, 0, 0, 0, 0, 0, 1],
        "observation.left_arm.pose": [0, 0, 0, 0, 0, 0, 1],
        "action.right_arm.pose": [1, 0, 0, 0, 0, 0, 1],
        "observation.right_arm.pose": [1, 0, 0, 0, 0, 0, 1],
    }


def _publish_legacy_episode_record(root: Path, row: dict[str, object]) -> None:
    path = root / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


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


def test_shadow_long_episode_preserves_exact_projection_for_every_sample(
    tmp_path: Path,
) -> None:
    _bundle, shadow, identity = _setup(tmp_path)
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


def test_parquet_shadow_long_episode_preserves_v23_projection_and_chunks(
    tmp_path: Path,
) -> None:
    _bundle, shadow, identity = _setup(
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

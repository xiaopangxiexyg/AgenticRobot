from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from dataset_contracts.embodiment import (
    CaptureProjectionComponent,
    CaptureProjectionProfile,
    EmbodimentCaptureProfile,
    EmbodimentComponentCaptureSpec,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonSubtask,
    LongHorizonTaskProfile,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v25p import v25p_base_episode_manifest_hash
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfile,
    V25pComponentStreamSpec,
    V25pExternalPayloadReference,
)
from dataset_contracts.v25p_projection import V25pCompatibilityProjectionReceipt
from dataset_contracts.v25p_native_storage import V25P_PARQUET_CHUNK_LAYOUT_V1
from dataset_interface.v25p_native import (
    V25P_NATIVE_READINESS_WITH_COMPATIBILITY,
    V25pNativeReadError,
    load_v25p_native_episode,
)
from dataset_interface.v25p_parquet_storage import load_v25p_parquet_sample
from dataset_interface.v25p_v23_adapter import (
    V25pV23ProjectionError,
    plan_v23_compatibility_projection,
    project_native_episode_to_v23_rows,
)
from collection_runtime.v25p_native_store import (
    V25pComponentInput,
    V25pNativeEpisodeStore,
)
from collection_runtime.v25p_parquet_store import (
    V25pParquetChunkEpisodeWriter,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _profiles():
    embodiment = EmbodimentCaptureProfile(
        profile_id="arbitrary_mobile_manipulator",
        profile_version="1",
        topology_id="arm_gripper",
        component_catalog_contract_id="component_catalog.v2",
        component_catalog_contract_hash=HASH_A,
        target_capability_contract_id="target_capability.v1",
        target_capability_contract_hash=HASH_B,
        components=(
            EmbodimentComponentCaptureSpec(
                component_id="arm.primary",
                component_kind="manipulator",
                controlled_entity_id="robot.arm",
                command_payload_schema_hash=HASH_C,
                measurement_payload_schema_hash=HASH_D,
            ),
        ),
    )
    native = V25pComponentStreamProfile(
        profile_id="native_components",
        profile_version="1",
        embodiment_profile_hash=embodiment.semantic_content_hash,
        streams=(
            V25pComponentStreamSpec(
                stream_id="arm.primary.accepted",
                component_id="arm.primary",
                dataset_role="action",
                value_stage="accepted_reference",
                payload_schema_hash=HASH_C,
                representation_family="joint_position",
                dtype="float64",
                shape=(3,),
                coordinate_names=("j1", "j2", "j3"),
                units=("rad",),
            ),
            V25pComponentStreamSpec(
                stream_id="arm.primary.measured",
                component_id="arm.primary",
                dataset_role="state",
                value_stage="causal_measured",
                payload_schema_hash=HASH_D,
                representation_family="joint_position",
                dtype="float64",
                shape=(3,),
                coordinate_names=("j1", "j2", "j3"),
                units=("rad",),
            ),
        ),
    )
    v23_projection = CaptureProjectionProfile(
        projection_id="legacy_v23",
        projection_version="1",
        temporal_alignment_contract_id="causal.v1",
        temporal_alignment_contract_hash=HASH_A,
        row_encoder_contract_id="row.v23",
        row_encoder_contract_hash=HASH_B,
        policy_io_layout_hash=HASH_C,
        pose_frame_contract_hash=HASH_D,
        causal_snapshot_contract_hash=HASH_E,
        components=(
            CaptureProjectionComponent(
                "arm.primary", "action", "action", HASH_C
            ),
            CaptureProjectionComponent(
                "arm.primary", "state", "observation.state", HASH_D
            ),
        ),
    )
    return embodiment, native, v23_projection


def _identity() -> EpisodeCaptureIdentity:
    return EpisodeCaptureIdentity(
        episode_uuid="native-episode",
        legacy_episode_index=7,
        lifecycle_owner_id="test.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="capture",
    )


def _sample_values(timestamp: float):
    common = {
        "effective_at": timestamp - 0.01,
        "available_at": timestamp - 0.005,
        "effective_clock_domain_id": "capture",
        "available_clock_domain_id": "capture",
    }
    return {
        "arm.primary.accepted": V25pComponentInput(
            values=(0.1, 0.2, 0.3), **common
        ),
        "arm.primary.measured": V25pComponentInput(
            values=(0.0, 0.1, 0.2), **common
        ),
    }


@pytest.mark.integration
def test_native_store_seals_without_any_v23_episode_record(tmp_path: Path) -> None:
    embodiment, native, projection = _profiles()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        compatibility_projection_profiles=(projection,),
        clock=lambda: 1.0,
        clock_domain_id="capture",
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.1,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )

    seal = store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    receipt = V25pCompatibilityProjectionReceipt(
        projection_id="legacy-v23-output",
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=identity.legacy_episode_index,
        episode_identity_hash=identity.semantic_content_hash,
        native_capture_manifest_hash=seal.native_capture_manifest_hash,
        projection_profile_hash=projection.semantic_content_hash,
        target_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        status="failed",
        completed_at=3.1,
        completed_clock_domain_id="capture",
        message="legacy writer intentionally absent in native-only test",
    )
    commit = store.commit_episode(
        identity,
        committed_at=3.2,
        committed_clock_domain_id="capture",
        compatibility_projection_receipts=(receipt,),
    )

    assert seal.sample_count == 1
    assert commit.native_capture_manifest_hash == seal.native_capture_manifest_hash
    assert commit.compatibility_projection_receipt_hashes == (
        receipt.semantic_content_hash,
    )
    assert not (tmp_path / "meta/episodes.jsonl").exists()
    assert (
        tmp_path
        / "meta/v25p/native/episodes/native-episode/episode_manifest.cjson"
    ).is_file()
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.identity == identity
    assert loaded.native_capture_manifest.semantic_content_hash == (
        seal.native_capture_manifest_hash
    )
    assert loaded.readiness.classification == (
        V25P_NATIVE_READINESS_WITH_COMPATIBILITY
    )
    assert tuple(loaded.records_by_stream) == (
        "arm.primary.accepted",
        "arm.primary.measured",
    )
    assert loaded.samples[0].record_hash_by_stream == {
        stream_id: records[0].semantic_content_hash
        for stream_id, records in loaded.records_by_stream.items()
    }
    projected = project_native_episode_to_v23_rows(loaded, projection)
    assert projected.plan.lossy is False
    assert dict(projected.rows[0].fields) == {
        "action": (0.1, 0.2, 0.3),
        "observation.state": (0.0, 0.1, 0.2),
    }


@pytest.mark.integration
def test_native_store_can_defer_sample_files_until_seal(tmp_path: Path) -> None:
    embodiment, native, _ = _profiles()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        defer_sample_file_writes=True,
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )

    episode_dir = tmp_path / "meta/v25p/native/episodes/native-episode"
    assert (episode_dir / "identity.cjson").is_file()
    assert not (episode_dir / "streams").exists()
    assert not (episode_dir / "samples").exists()

    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )

    assert (
        episode_dir / "streams/arm.primary.accepted/00000001.cjson"
    ).is_file()
    assert (
        episode_dir / "streams/arm.primary.measured/00000001.cjson"
    ).is_file()
    assert (episode_dir / "samples/00000000.cjson").is_file()
    assert (episode_dir / "native_capture_manifest.cjson").is_file()


@pytest.mark.optional
def test_parquet_chunks_round_trip_exactly_matches_record_cjson(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    embodiment, native, projection = _profiles()
    identity = _identity()

    def capture(root: Path, *, layout: str):
        store = V25pNativeEpisodeStore(
            dataset_root=root,
            dataset_id="native-dataset",
            embodiment_profile=embodiment,
            component_stream_profile=native,
            compatibility_projection_profiles=(projection,),
            physical_storage_layout=layout,
            parquet_chunk_rows=2,
            parquet_compression="none",
        )
        store.prepare_episode(identity)
        store.bind_prepared_episode(
            identity,
            bound_at=1.0,
            bound_clock_domain_id="capture",
        )
        for sample_index in range(5):
            timestamp = 2.0 + sample_index * 0.1
            store.append_sample(
                identity,
                sample_index=sample_index,
                cut_at=timestamp,
                cut_clock_domain_id="capture",
                values_by_stream=_sample_values(timestamp),
            )
        seal = store.seal_native_episode(
            identity,
            sealed_at=3.0,
            sealed_clock_domain_id="capture",
        )
        store.commit_episode(
            identity,
            committed_at=3.1,
            committed_clock_domain_id="capture",
        )
        return seal, load_v25p_native_episode(root, identity.episode_uuid)

    cjson_seal, cjson = capture(tmp_path / "cjson", layout="record_cjson_v1")
    parquet_seal, parquet = capture(
        tmp_path / "parquet",
        layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
    )

    assert cjson_seal.native_storage_manifest_hash is None
    assert parquet_seal.native_storage_manifest_hash is not None
    assert parquet.native_capture_manifest.semantic_content_hash == (
        cjson.native_capture_manifest.semantic_content_hash
    )
    assert {
        stream_id: tuple(record.to_payload() for record in records)
        for stream_id, records in parquet.records_by_stream.items()
    } == {
        stream_id: tuple(record.to_payload() for record in records)
        for stream_id, records in cjson.records_by_stream.items()
    }
    assert tuple(sample.to_payload() for sample in parquet.samples) == tuple(
        sample.to_payload() for sample in cjson.samples
    )
    assert project_native_episode_to_v23_rows(
        parquet, projection
    ).rows == project_native_episode_to_v23_rows(cjson, projection).rows
    episode_dir = tmp_path / "parquet/meta/v25p/native/episodes/native-episode"
    assert len(tuple((episode_dir / "chunks").glob("*.cjson"))) == 3
    assert len(tuple((episode_dir / "samples").glob("*.parquet"))) == 3
    assert not tuple(episode_dir.glob("streams/**/*.cjson"))
    selected_sample, selected_records = load_v25p_parquet_sample(
        episode_dir=episode_dir,
        identity=identity,
        profile=native,
        sample_index=3,
        stream_ids=("arm.primary.measured",),
    )
    assert selected_sample == parquet.samples[3]
    assert tuple(selected_records) == ("arm.primary.measured",)
    assert selected_records["arm.primary.measured"] == (
        parquet.records_by_stream["arm.primary.measured"][3]
    )


@pytest.mark.optional
def test_parquet_unsealed_interruption_keeps_chunks_but_publishes_no_episode(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    embodiment, native, _ = _profiles()
    identity = _identity()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        parquet_chunk_rows=2,
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    for sample_index in range(2):
        timestamp = 2.0 + sample_index * 0.1
        store.append_sample(
            identity,
            sample_index=sample_index,
            cut_at=timestamp,
            cut_clock_domain_id="capture",
            values_by_stream=_sample_values(timestamp),
        )

    active = store._active
    assert active is not None
    assert active.sample_count == 2
    assert active.samples == []
    assert all(not values for values in active.records_by_stream.values())
    store.close_unsealed(identity)

    episode_dir = tmp_path / "meta/v25p/native/episodes/native-episode"
    assert (episode_dir / "chunks/chunk_000000.cjson").is_file()
    assert not (episode_dir / "native_storage_manifest.cjson").exists()
    assert not (episode_dir / "native_capture_manifest.cjson").exists()
    assert not (episode_dir / "episode_manifest.cjson").exists()
    with pytest.raises(V25pNativeReadError, match="missing"):
        load_v25p_native_episode(tmp_path, identity.episode_uuid)


@pytest.mark.optional
def test_parquet_full_chunk_publication_runs_off_the_append_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    original_write_chunk = V25pParquetChunkEpisodeWriter._write_chunk
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocked_write_chunk(writer, **kwargs):
        worker_started.set()
        if not release_worker.wait(timeout=5.0):
            raise TimeoutError("test did not release the Parquet worker")
        return original_write_chunk(writer, **kwargs)

    monkeypatch.setattr(
        V25pParquetChunkEpisodeWriter,
        "_write_chunk",
        blocked_write_chunk,
    )
    embodiment, native, _ = _profiles()
    identity = _identity()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        parquet_chunk_rows=2,
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    for sample_index in range(2):
        timestamp = 2.0 + sample_index * 0.1
        store.append_sample(
            identity,
            sample_index=sample_index,
            cut_at=timestamp,
            cut_clock_domain_id="capture",
            values_by_stream=_sample_values(timestamp),
        )

    assert worker_started.wait(timeout=1.0)
    assert not release_worker.is_set()
    release_worker.set()
    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )


@pytest.mark.optional
@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_parquet_chunk_compression_preserves_logical_content(
    tmp_path: Path,
    compression: str,
) -> None:
    pytest.importorskip("pyarrow")
    embodiment, native, _ = _profiles()
    identity = _identity()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        parquet_compression=compression,
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )
    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=3.1,
        committed_clock_domain_id="capture",
    )

    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.samples[0].to_payload()["sample_index"] == 0
    assert loaded.records_by_stream["arm.primary.accepted"][0].values == (
        0.1,
        0.2,
        0.3,
    )


@pytest.mark.optional
def test_parquet_chunk_reader_rejects_physical_corruption(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    embodiment, native, _ = _profiles()
    identity = _identity()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        parquet_chunk_rows=2,
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )
    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=3.1,
        committed_clock_domain_id="capture",
    )
    sample_path = (
        tmp_path
        / "meta/v25p/native/episodes/native-episode/samples/chunk_000000.parquet"
    )
    payload = bytearray(sample_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    sample_path.write_bytes(payload)

    with pytest.raises(V25pNativeReadError, match="integrity mismatch"):
        load_v25p_native_episode(tmp_path, identity.episode_uuid)


@pytest.mark.integration
def test_native_store_rejects_missing_required_stream_and_future_data(
    tmp_path: Path,
) -> None:
    embodiment, native, _ = _profiles()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    with pytest.raises(ValueError, match="missing"):
        store.append_sample(
            identity,
            sample_index=0,
            cut_at=2.0,
            cut_clock_domain_id="capture",
            values_by_stream={
                "arm.primary.accepted": _sample_values(2.0)[
                    "arm.primary.accepted"
                ]
            },
        )
    future = _sample_values(2.0)
    future["arm.primary.measured"] = V25pComponentInput(
        values=(0.0, 0.1, 0.2),
        effective_at=1.9,
        available_at=2.1,
        effective_clock_domain_id="capture",
        available_clock_domain_id="capture",
    )
    with pytest.raises(ValueError, match="not available"):
        store.append_sample(
            identity,
            sample_index=0,
            cut_at=2.0,
            cut_clock_domain_id="capture",
            values_by_stream=future,
        )


@pytest.mark.integration
def test_complete_v23_receipt_rechecks_the_exact_legacy_episode_row(
    tmp_path: Path,
) -> None:
    embodiment, native, projection = _profiles()
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-with-v23-output",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        compatibility_projection_profiles=(projection,),
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )
    seal = store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    legacy_row = {
        "episode_index": identity.legacy_episode_index,
        "frame_count": 1,
        "parquet_path": "data/chunk-000/episode_000007.parquet",
    }
    episodes_path = tmp_path / "meta" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")
    receipt = V25pCompatibilityProjectionReceipt(
        projection_id="legacy-v23-output",
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=identity.legacy_episode_index,
        episode_identity_hash=identity.semantic_content_hash,
        native_capture_manifest_hash=seal.native_capture_manifest_hash,
        projection_profile_hash=projection.semantic_content_hash,
        target_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        status="complete",
        target_episode_record_hash=v25p_base_episode_manifest_hash(legacy_row),
        target_episode_locator="meta/episodes.jsonl#episode_index=7",
        lossy=False,
        omitted_stream_ids=(),
        completed_at=3.1,
        completed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=3.2,
        committed_clock_domain_id="capture",
        compatibility_projection_receipts=(receipt,),
    )

    assert load_v25p_native_episode(tmp_path, identity.episode_uuid).identity == identity
    legacy_row["frame_count"] = 2
    episodes_path.write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")
    with pytest.raises(V25pNativeReadError, match="episode-record hash mismatch"):
        load_v25p_native_episode(tmp_path, identity.episode_uuid)


def test_native_stream_profile_is_not_bound_to_v23_layout() -> None:
    embodiment, native, _ = _profiles()

    assert native.embodiment_profile_hash == embodiment.semantic_content_hash
    assert not hasattr(native, "base_dataset_schema")
    assert not hasattr(native, "policy_io_layout_hash")


def test_native_stream_id_cannot_escape_its_episode_directory() -> None:
    with pytest.raises(ValueError, match="dataset-relative"):
        V25pComponentStreamSpec(
            stream_id="../escape",
            component_id="arm.primary",
            dataset_role="action",
            value_stage="accepted_reference",
            payload_schema_hash=HASH_C,
            representation_family="joint_position",
            dtype="float64",
            shape=(1,),
            coordinate_names=("j1",),
            units=("rad",),
        )


def test_v23_projection_fails_instead_of_guessing_an_incompatible_stage() -> None:
    _, native, projection = _profiles()
    incompatible = CaptureProjectionProfile(
        projection_id="legacy-v23-next-observation",
        projection_version="1",
        temporal_alignment_contract_id=projection.temporal_alignment_contract_id,
        temporal_alignment_contract_hash=projection.temporal_alignment_contract_hash,
        row_encoder_contract_id=projection.row_encoder_contract_id,
        row_encoder_contract_hash=projection.row_encoder_contract_hash,
        policy_io_layout_hash=projection.policy_io_layout_hash,
        pose_frame_contract_hash=projection.pose_frame_contract_hash,
        causal_snapshot_contract_hash=projection.causal_snapshot_contract_hash,
        action_stage="next_accepted_observation",
        components=projection.components,
    )

    with pytest.raises(V25pV23ProjectionError, match="no native stream"):
        plan_v23_compatibility_projection(
            stream_profile=native,
            projection_profile=incompatible,
        )


@pytest.mark.integration
def test_native_external_observation_is_hash_bound_and_explicitly_lossy_for_v23(
    tmp_path: Path,
) -> None:
    base_embodiment, base_native, projection = _profiles()
    embodiment = EmbodimentCaptureProfile(
        profile_id="arm-with-camera",
        profile_version="1",
        topology_id="arm-gripper-camera",
        component_catalog_contract_id=(
            base_embodiment.component_catalog_contract_id
        ),
        component_catalog_contract_hash=(
            base_embodiment.component_catalog_contract_hash
        ),
        target_capability_contract_id=(
            base_embodiment.target_capability_contract_id
        ),
        target_capability_contract_hash=(
            base_embodiment.target_capability_contract_hash
        ),
        components=base_embodiment.components
        + (
            EmbodimentComponentCaptureSpec(
                component_id="camera.front",
                component_kind="camera",
                controlled_entity_id="sensor.front",
                measurement_payload_schema_hash=HASH_E,
                required_for_capture=False,
            ),
        ),
    )
    native = V25pComponentStreamProfile(
        profile_id="native-with-camera",
        profile_version="1",
        embodiment_profile_hash=embodiment.semantic_content_hash,
        streams=base_native.streams
        + (
            V25pComponentStreamSpec(
                stream_id="camera.front.rgb",
                component_id="camera.front",
                dataset_role="observation",
                value_stage="causal_observed",
                payload_schema_hash=HASH_E,
                representation_family="rgb_image",
                dtype="uint8",
                shape=(480, 640, 3),
                coordinate_names=(),
                units=("pixel",),
                required_for_sample=False,
                storage_kind="external_artifact",
            ),
        ),
    )
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-camera-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        compatibility_projection_profiles=(projection,),
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    values = _sample_values(2.0)
    reference = V25pExternalPayloadReference(
        artifact_locator="videos/front.mp4",
        artifact_content_hash=HASH_A,
        encoding="video/h264",
        sample_selector="frame:0",
    )
    values["camera.front.rgb"] = V25pComponentInput(
        values=None,
        payload_reference=reference,
        effective_at=1.99,
        available_at=1.995,
        effective_clock_domain_id="capture",
        available_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=values,
    )
    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=3.1,
        committed_clock_domain_id="capture",
    )

    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    camera_record = loaded.records_by_stream["camera.front.rgb"][0]
    assert camera_record.payload_reference == reference
    plan = plan_v23_compatibility_projection(
        stream_profile=loaded.component_stream_profile,
        projection_profile=projection,
    )
    assert plan.lossy is True
    assert plan.omitted_stream_ids == ("camera.front.rgb",)


@pytest.mark.integration
def test_native_task_timeline_preserves_failure_then_recovery_success(
    tmp_path: Path,
) -> None:
    embodiment, native, _ = _profiles()
    task = LongHorizonTaskProfile(
        profile_id="long-pick",
        profile_version="1",
        task_id="pick",
        subtasks=(LongHorizonSubtask("grasp", "Grasp"),),
    )
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id="native-task-dataset",
        embodiment_profile=embodiment,
        component_stream_profile=native,
        task_profile=task,
    )
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.0,
        bound_clock_domain_id="capture",
    )
    for ordinal, outcome in enumerate(("failure", "success"), start=1):
        attempt_id = f"grasp-{ordinal}"
        begin_at = 1.1 + ordinal * 0.1
        store.begin_subtask(
            identity,
            track_id="primary_subtask",
            attempt_id=attempt_id,
            subtask_id="grasp",
            effective_at=begin_at,
            available_at=begin_at,
            effective_clock_domain_id="capture",
            available_clock_domain_id="capture",
            actor_id="operator",
        )
        store.finish_subtask(
            identity,
            track_id="primary_subtask",
            attempt_id=attempt_id,
            subtask_id="grasp",
            outcome=outcome,
            effective_at=begin_at + 0.05,
            available_at=begin_at + 0.05,
            effective_clock_domain_id="capture",
            available_clock_domain_id="capture",
            actor_id="operator",
        )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=_sample_values(2.0),
    )
    store.seal_native_episode(
        identity,
        sealed_at=3.0,
        sealed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=3.1,
        committed_clock_domain_id="capture",
    )

    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.task_profile == task
    assert tuple(
        event.outcome for event in loaded.task_events if event.event_kind == "finish"
    ) == ("failure", "success")

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dataset_contracts.stream_catalog import (
    NativeStreamContract,
    NativeStreamExternalArtifactReceipt,
    NativeStreamExternalSampleIntent,
)
from dataset_interface.streams import (
    NativeStreamExternalPayloadReference,
    NativeStreamReadError,
    load_native_stream_episode,
)
from robot_motion.runtime.observation_cut import ObservationSampleRef
from collection_runtime.stream_sink import NativeStreamEpisodeWriter


pytestmark = pytest.mark.integration


def _contract(stream_id: str) -> NativeStreamContract:
    return NativeStreamContract(
        stream_id=stream_id,
        modality="human_joint_input",
        producer_id="virtual-q6",
        value_schema="float32[6]",
        shape=(6,),
        dtype="<f4",
        units="normalized_joint_position",
        frame=None,
        measured_clock_domain_id="q6.device",
        available_clock_domain_id="control.monotonic",
        nominal_rate_hz=30.0,
        min_expected_rate_hz=10.0,
        max_expected_rate_hz=60.0,
        access_role="actor_input",
    )


def _sample(sequence: int) -> ObservationSampleRef:
    return ObservationSampleRef(
        stream_id="human.left_hand_q6",
        sample_id=f"q6-{sequence}",
        sequence=sequence,
        measured_at=100.0 + sequence / 30.0,
        measured_clock_domain_id="q6.device",
        available_at=10.0 + sequence / 30.0,
        available_clock_domain_id="control.monotonic",
        payload_object_id=f"upstream-q6-{sequence}",
    )


def _external_contract(
    stream_id: str,
    *,
    modality: str,
    shape: tuple[int, ...],
    dtype: str,
    encoding: str,
    compression: str,
) -> NativeStreamContract:
    return NativeStreamContract(
        stream_id=stream_id,
        modality=modality,
        producer_id=f"fake-{modality}",
        value_schema=f"{dtype}{shape}",
        shape=shape,
        dtype=dtype,
        units="raw",
        frame=None,
        measured_clock_domain_id="sensor.device",
        available_clock_domain_id="control.monotonic",
        encoding=encoding,
        compression=compression,
        access_role="actor_input",
    )


def _external_intent(
    contract: NativeStreamContract,
    payload: np.ndarray,
    *,
    sample_id: str = "sample-0",
) -> NativeStreamExternalSampleIntent:
    return NativeStreamExternalSampleIntent.create(
        contract=contract,
        sample_id=sample_id,
        sequence=0,
        measured_at=1.0,
        measured_clock_domain_id="sensor.device",
        available_at=1.1,
        available_clock_domain_id="control.monotonic",
        source_payload_object_id=f"shared-slot:{sample_id}",
        payload=payload,
    )


def _external_receipt(
    *,
    episode_uuid: str,
    legacy_episode_index: int,
    contract: NativeStreamContract,
    intent: NativeStreamExternalSampleIntent,
    artifact_relative_path: str,
    artifact_bytes: bytes,
) -> NativeStreamExternalArtifactReceipt:
    return NativeStreamExternalArtifactReceipt(
        episode_uuid=episode_uuid,
        legacy_episode_index=legacy_episode_index,
        stream_id=contract.stream_id,
        sample_id=intent.sample_id,
        payload_content_hash=intent.payload_content_hash,
        stored_payload_object_id=(
            f"dataset:{artifact_relative_path}#sample={intent.sample_id}"
        ),
        artifact_relative_path=artifact_relative_path,
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        artifact_byte_count=len(artifact_bytes),
        encoding=contract.encoding,
        compression=contract.compression,
        durable_at=2.0,
        durable_clock_domain_id="local_wall_clock",
    )


def test_native_stream_episode_round_trip_and_idempotent_sample(tmp_path) -> None:
    contract = _contract("human.left_hand_q6")
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid="episode-native-0",
        legacy_episode_index=3,
        contracts={contract.stream_id: contract},
    )
    payload = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    first = writer.append_observation_sample(
        _sample(0),
        payload,
        written_at=10.01,
        written_clock_domain_id="control.monotonic",
    )
    repeated = writer.append_observation_sample(
        _sample(0),
        payload.copy(),
        written_at=10.02,
        written_clock_domain_id="control.monotonic",
    )
    assert first.accepted and first.durable and not first.idempotent
    assert repeated.accepted and repeated.idempotent

    writer.append_observation_sample(
        _sample(2),
        payload + 0.1,
        written_at=10.08,
        written_clock_domain_id="control.monotonic",
    )
    writer.note_drop(contract.stream_id, reason="source_queue_overflow")
    manifest = writer.seal(
        sealed_at=10.2,
        sealed_clock_domain_id="control.monotonic",
    )

    stream_range = manifest.stream_ranges[0]
    assert stream_range.sample_count == 2
    assert stream_range.dropped_count == 1
    assert set(stream_range.gap_reasons) == {
        "sequence_gap",
        "source_queue_overflow",
    }
    episode = load_native_stream_episode(tmp_path, "episode-native-0")
    assert episode.manifest.semantic_content_hash == (
        manifest.semantic_content_hash
    )
    records = episode.records_by_stream[contract.stream_id]
    assert tuple(record.sequence for record in records) == (0, 2)
    np.testing.assert_allclose(
        episode.payload_by_stream_and_sample[contract.stream_id]["q6-0"],
        payload,
    )
    with pytest.raises(RuntimeError, match="closed"):
        writer.append_observation_sample(
            _sample(3),
            payload,
            written_at=10.3,
            written_clock_domain_id="control.monotonic",
        )


def test_native_stream_reader_rejects_tampered_index(tmp_path) -> None:
    contract = _contract("human.left_hand_q6")
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid="episode-native-1",
        legacy_episode_index=4,
        contracts={contract.stream_id: contract},
    )
    receipt = writer.append_observation_sample(
        _sample(0),
        np.zeros(6, dtype=np.float32),
        written_at=10.0,
        written_clock_domain_id="control.monotonic",
    )
    writer.seal(
        sealed_at=10.1,
        sealed_clock_domain_id="control.monotonic",
    )
    index_path = tmp_path / receipt.index_relative_path
    index_path.write_bytes(index_path.read_bytes() + b"\n")

    with pytest.raises(NativeStreamReadError, match="index hash"):
        load_native_stream_episode(tmp_path, "episode-native-1")


def test_native_stream_writer_rejects_payload_shape_or_dtype_drift(tmp_path) -> None:
    contract = _contract("human.left_hand_q6")
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid="episode-native-2",
        legacy_episode_index=5,
        contracts={contract.stream_id: contract},
    )
    with pytest.raises(ValueError, match="shape"):
        writer.append_observation_sample(
            _sample(0),
            np.zeros(5, dtype=np.float32),
            written_at=10.0,
            written_clock_domain_id="control.monotonic",
        )
    with pytest.raises(ValueError, match="dtype"):
        writer.append_observation_sample(
            _sample(0),
            np.zeros(6, dtype=np.float64),
            written_at=10.0,
            written_clock_domain_id="control.monotonic",
        )
    writer.close_unsealed()


@pytest.mark.parametrize(
    ("contract", "payload", "artifact_bytes"),
    (
        (
            _external_contract(
                "camera.front.rgb",
                modality="rgb",
                shape=(2, 2, 3),
                dtype="|u1",
                encoding="fake_video_v1",
                compression="fake_h264",
            ),
            np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            b"fake-video-container-frame-0",
        ),
        (
            _external_contract(
                "tactile.left.raw",
                modality="tactile_raw",
                shape=(4,),
                dtype="<f4",
                encoding="fake_tactile_binary_v1",
                compression="fake_zstd",
            ),
            np.linspace(0.0, 1.0, 4, dtype=np.float32),
            b"fake-tactile-block-0",
        ),
    ),
)
def test_external_native_stream_round_trip_returns_validated_locator(
    tmp_path,
    contract,
    payload,
    artifact_bytes,
) -> None:
    episode_uuid = "episode-external-0"
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid=episode_uuid,
        legacy_episode_index=8,
        contracts={contract.stream_id: contract},
        external_payload_stream_ids=(contract.stream_id,),
    )
    intent = _external_intent(contract, payload)
    assert writer.stage_external_sample(intent) == intent
    assert writer.stage_external_sample(intent) == intent
    writer.freeze_ingestion()
    with pytest.raises(RuntimeError, match="frozen"):
        writer.stage_external_sample(intent)

    relative_path = f"external/{contract.stream_id}/episode.bin"
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact_bytes)
    receipt = _external_receipt(
        episode_uuid=episode_uuid,
        legacy_episode_index=8,
        contract=contract,
        intent=intent,
        artifact_relative_path=relative_path,
        artifact_bytes=artifact_bytes,
    )
    manifest = writer.seal_after_legacy_finalize(
        (receipt,),
        sealed_at=2.1,
        sealed_clock_domain_id="control.monotonic",
    )
    assert manifest.stream_ranges[0].sample_count == 1

    episode = load_native_stream_episode(tmp_path, episode_uuid)
    reference = episode.payload_by_stream_and_sample[
        contract.stream_id
    ][intent.sample_id]
    assert isinstance(reference, NativeStreamExternalPayloadReference)
    assert reference.artifact_relative_path == relative_path
    assert reference.stored_payload_object_id.endswith(
        f"#sample={intent.sample_id}"
    )
    assert reference.payload_content_hash == intent.payload_content_hash
    assert reference.artifact_sha256 == receipt.artifact_sha256


def test_external_native_stream_requires_exact_receipts_and_safe_paths(
    tmp_path,
) -> None:
    contract = _external_contract(
        "camera.front.rgb",
        modality="rgb",
        shape=(1, 1, 3),
        dtype="|u1",
        encoding="fake_video_v1",
        compression="fake_h264",
    )
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid="episode-external-coverage",
        legacy_episode_index=9,
        contracts={contract.stream_id: contract},
        external_payload_stream_ids=(contract.stream_id,),
    )
    intent = _external_intent(
        contract,
        np.zeros((1, 1, 3), dtype=np.uint8),
    )
    writer.stage_external_sample(intent)
    writer.freeze_ingestion()
    with pytest.raises(ValueError, match="coverage mismatch"):
        writer.seal_after_legacy_finalize(
            (),
            sealed_at=2.0,
            sealed_clock_domain_id="control.monotonic",
        )
    with pytest.raises(ValueError, match="dataset root"):
        _external_receipt(
            episode_uuid="episode-external-coverage",
            legacy_episode_index=9,
            contract=contract,
            intent=intent,
            artifact_relative_path="../escape.bin",
            artifact_bytes=b"bad",
        )
    writer.close_unsealed()


def test_external_native_stream_reader_rejects_tampered_artifact(
    tmp_path,
) -> None:
    contract = _external_contract(
        "tactile.left.raw",
        modality="tactile_raw",
        shape=(2,),
        dtype="<f4",
        encoding="fake_tactile_binary_v1",
        compression="fake_zstd",
    )
    episode_uuid = "episode-external-tamper"
    writer = NativeStreamEpisodeWriter(
        tmp_path,
        episode_uuid=episode_uuid,
        legacy_episode_index=10,
        contracts={contract.stream_id: contract},
        external_payload_stream_ids=(contract.stream_id,),
    )
    intent = _external_intent(
        contract,
        np.zeros(2, dtype=np.float32),
    )
    writer.stage_external_sample(intent)
    writer.freeze_ingestion()
    artifact_bytes = b"original-tactile-block"
    relative_path = "external/tactile/episode.bin"
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact_bytes)
    receipt = _external_receipt(
        episode_uuid=episode_uuid,
        legacy_episode_index=10,
        contract=contract,
        intent=intent,
        artifact_relative_path=relative_path,
        artifact_bytes=artifact_bytes,
    )
    writer.seal_after_legacy_finalize(
        (receipt,),
        sealed_at=2.0,
        sealed_clock_domain_id="control.monotonic",
    )
    artifact_path.write_bytes(b"tampered-tactile-block")

    with pytest.raises(NativeStreamReadError, match="artifact"):
        load_native_stream_episode(
            tmp_path,
            episode_uuid,
            load_payloads=False,
        )

from __future__ import annotations

import pytest

from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native import (
    V25pComponentRecord,
    V25pComponentStreamProfile,
    V25pComponentStreamSpec,
    V25pExternalPayloadReference,
)
from dataset_interface.v25p_parquet import (
    component_records_from_arrow,
    component_records_to_arrow,
)


HASH_A = "a" * 64


def _identity() -> EpisodeCaptureIdentity:
    return EpisodeCaptureIdentity(
        episode_uuid="parquet-adapter-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="test.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="clock",
    )


def _stream(*, dtype: str, storage_kind: str = "inline_tensor"):
    return V25pComponentStreamSpec(
        stream_id="component.state",
        component_id="component",
        dataset_role="state",
        value_stage="causal_measured",
        payload_schema_hash=HASH_A,
        representation_family="test_vector",
        dtype=dtype,
        shape=(2,),
        coordinate_names=("a", "b"),
        units=("1",),
        storage_kind=storage_kind,
    )


def _profile(stream: V25pComponentStreamSpec) -> V25pComponentStreamProfile:
    return V25pComponentStreamProfile(
        profile_id="parquet_adapter",
        profile_version="1",
        embodiment_profile_hash=HASH_A,
        streams=(stream,),
    )


@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        ("bool", (True, False)),
        ("int8", (-1, 2)),
        ("int16", (-1, 2)),
        ("int32", (-1, 2)),
        ("int64", (-1, 2)),
        ("uint8", (1, 2)),
        ("uint16", (1, 2)),
        ("uint32", (1, 2)),
        ("float16", (0.25, -0.5)),
        ("float32", (0.25, -0.5)),
        ("float64", (0.25, -0.5)),
    ),
)
def test_component_parquet_round_trip_preserves_every_native_dtype(
    dtype: str,
    values: tuple[object, object],
) -> None:
    identity = _identity()
    stream = _stream(dtype=dtype)
    profile = _profile(stream)
    record = V25pComponentRecord(
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=identity.legacy_episode_index,
        component_stream_profile_hash=profile.semantic_content_hash,
        stream_id=stream.stream_id,
        component_id=stream.component_id,
        dataset_role=stream.dataset_role,
        value_stage=stream.value_stage,
        payload_schema_hash=stream.payload_schema_hash,
        record_sequence=1,
        sample_index=0,
        effective_at=2.0,
        available_at=2.1,
        effective_clock_domain_id="clock",
        available_clock_domain_id="clock",
        values=stream.normalize_values(values),
    )

    rebuilt = component_records_from_arrow(
        component_records_to_arrow((record,), stream=stream),
        identity=identity,
        profile=profile,
        stream=stream,
    )

    assert rebuilt == (record,)
    assert rebuilt[0].semantic_content_hash == record.semantic_content_hash


def test_component_parquet_round_trip_preserves_external_and_invalid_records() -> None:
    identity = _identity()
    stream = _stream(dtype="uint8", storage_kind="external_artifact")
    profile = _profile(stream)
    common = {
        "episode_uuid": identity.episode_uuid,
        "legacy_episode_index": identity.legacy_episode_index,
        "component_stream_profile_hash": profile.semantic_content_hash,
        "stream_id": stream.stream_id,
        "component_id": stream.component_id,
        "dataset_role": stream.dataset_role,
        "value_stage": stream.value_stage,
        "payload_schema_hash": stream.payload_schema_hash,
        "effective_clock_domain_id": "clock",
        "available_clock_domain_id": "clock",
    }
    external = V25pComponentRecord(
        **common,
        record_sequence=1,
        sample_index=0,
        effective_at=2.0,
        available_at=2.1,
        payload_reference=V25pExternalPayloadReference(
            artifact_locator="media/chunk.bin",
            artifact_content_hash="b" * 64,
            encoding="application/octet-stream",
            sample_selector="frame=4",
            byte_offset=128,
            byte_length=64,
        ),
        source_record_id="sensor:4",
    )
    invalid = V25pComponentRecord(
        **common,
        record_sequence=2,
        sample_index=1,
        effective_at=3.0,
        available_at=3.1,
        valid=False,
        invalid_reason="sensor_stale",
    )

    rebuilt = component_records_from_arrow(
        component_records_to_arrow((external, invalid), stream=stream),
        identity=identity,
        profile=profile,
        stream=stream,
    )

    assert rebuilt == (external, invalid)


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


@pytest.mark.parametrize("validity", [(False, False), (True, False), (False, True)])
def test_missing_fixed_vectors_round_trip_through_actual_parquet(tmp_path, validity):
    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = _identity()
    stream = _stream(dtype="float64")
    profile = _profile(stream)
    records = tuple(V25pComponentRecord(
        episode_uuid=identity.episode_uuid, legacy_episode_index=0,
        component_stream_profile_hash=profile.semantic_content_hash,
        stream_id=stream.stream_id, component_id=stream.component_id,
        dataset_role=stream.dataset_role, value_stage=stream.value_stage,
        payload_schema_hash=stream.payload_schema_hash,
        record_sequence=i + 1, sample_index=i, effective_at=float(i), available_at=float(i),
        effective_clock_domain_id="clock", available_clock_domain_id="clock",
        values=(.1, -.2) if valid else None, valid=valid,
        invalid_reason="not_available" if not valid else "",
    ) for i, valid in enumerate(validity))
    table = component_records_to_arrow(records, stream=stream)
    path = tmp_path / "records.parquet"
    pq.write_table(table, path, compression="zstd")
    rebuilt = component_records_from_arrow(pq.read_table(path), identity=identity, profile=profile, stream=stream)
    assert rebuilt == records
    assert [r.semantic_content_hash for r in rebuilt] == [r.semantic_content_hash for r in records]
    legacy_rows = table.to_pylist()
    for row in legacy_rows:
        if not row["valid"]:
            row["values"] = None
    assert component_records_from_arrow(pa.Table.from_pylist(legacy_rows, schema=table.schema),
        identity=identity, profile=profile, stream=stream) == records


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



def _storage_record(stream, *, sequence=1, values=None, valid=False):
    identity = _identity()
    return V25pComponentRecord(
        episode_uuid=identity.episode_uuid,
        legacy_episode_index=identity.legacy_episode_index,
        component_stream_profile_hash=_profile(stream).semantic_content_hash,
        stream_id=stream.stream_id,
        component_id=stream.component_id,
        dataset_role=stream.dataset_role,
        value_stage=stream.value_stage,
        payload_schema_hash=stream.payload_schema_hash,
        record_sequence=sequence,
        sample_index=sequence - 1,
        effective_at=float(sequence),
        available_at=float(sequence),
        effective_clock_domain_id="clock",
        available_clock_domain_id="clock",
        values=values,
        valid=valid,
        invalid_reason="not_available" if not valid else "",
    )


@pytest.mark.parametrize("dtype", [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "float16", "float32", "float64",
])
@pytest.mark.parametrize("shape", [(1,), (2, 2)])
def test_missing_and_zero_vectors_keep_physical_width_and_logical_identity(tmp_path, dtype, shape):
    from dataclasses import replace
    import numpy as np
    import pyarrow.parquet as pq

    stream = replace(_stream(dtype=dtype), shape=shape, coordinate_names=())
    zero = stream.normalize_values(np.zeros(shape, dtype=dtype))
    one = stream.normalize_values(np.ones(shape, dtype=dtype))
    records = (
        _storage_record(stream, sequence=1),
        _storage_record(stream, sequence=2, values=zero, valid=True),
        _storage_record(stream, sequence=3, values=one, valid=True),
        _storage_record(stream, sequence=4),
    )
    table = component_records_to_arrow(records, stream=stream)
    assert table.column("values").null_count == 0
    assert table.column("values").to_pylist() == [
        [None] * stream.flattened_size, list(zero), list(one),
        [None] * stream.flattened_size,
    ]
    path = tmp_path / "mixed.parquet"
    pq.write_table(table, path, compression="zstd", row_group_size=2)
    restored_table = pq.read_table(path)
    assert restored_table.schema == table.schema
    assert restored_table.column("values").null_count == 0
    assert restored_table.column("values").to_pylist() == table.column("values").to_pylist()
    rebuilt = component_records_from_arrow(
        restored_table, identity=_identity(), profile=_profile(stream), stream=stream,
    )
    assert rebuilt == records
    assert tuple(record.semantic_content_hash for record in rebuilt) == tuple(
        record.semantic_content_hash for record in records
    )
    assert rebuilt[0].values is None and rebuilt[0].valid is False
    assert rebuilt[1].values == zero and rebuilt[1].valid is True
    assert rebuilt[3].values is None and rebuilt[3].valid is False


@pytest.mark.parametrize("dtype", [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "float16", "float32", "float64",
])
def test_partial_null_fixed_vectors_are_rejected(dtype):
    import pyarrow as pa

    stream = _stream(dtype=dtype)
    values = stream.normalize_values([0, 0])
    table = component_records_to_arrow((_storage_record(stream, values=values, valid=True),), stream=stream)
    rows = table.to_pylist()
    rows[0]["values"][0] = None
    malformed = pa.Table.from_pylist(rows, schema=table.schema)
    with pytest.raises(ValueError):
        component_records_from_arrow(malformed, identity=_identity(), profile=_profile(stream), stream=stream)


@pytest.mark.parametrize("dtype", ["bool", "int32", "float64"])
@pytest.mark.parametrize("representation", ["parent", "children"])
def test_valid_record_cannot_claim_an_all_null_fixed_vector(dtype, representation):
    import pyarrow as pa

    stream = _stream(dtype=dtype)
    values = stream.normalize_values([0, 0])
    table = component_records_to_arrow((_storage_record(stream, values=values, valid=True),), stream=stream)
    rows = table.to_pylist()
    rows[0]["values"] = None if representation == "parent" else [None, None]
    malformed = pa.Table.from_pylist(rows, schema=table.schema)
    with pytest.raises(ValueError, match="valid component record requires"):
        component_records_from_arrow(malformed, identity=_identity(), profile=_profile(stream), stream=stream)


@pytest.mark.parametrize("tamper", ["missing_to_zero", "zero_to_missing", "hash", "reason"])
def test_fixed_vector_missing_representation_does_not_bypass_integrity(tamper):
    import pyarrow as pa

    stream = _stream(dtype="float64")
    record = _storage_record(stream, values=(0.0, 0.0), valid=True) if tamper == "zero_to_missing" else _storage_record(stream)
    table = component_records_to_arrow((record,), stream=stream)
    rows = table.to_pylist()
    if tamper == "missing_to_zero":
        rows[0]["values"] = [0.0, 0.0]
    elif tamper == "zero_to_missing":
        rows[0]["values"] = [None, None]
    elif tamper == "hash":
        rows[0]["record_hash"] = bytes(32)
    else:
        rows[0]["invalid_reason"] = "different_reason"
    malformed = pa.Table.from_pylist(rows, schema=table.schema)
    with pytest.raises(ValueError):
        component_records_from_arrow(malformed, identity=_identity(), profile=_profile(stream), stream=stream)


def test_empty_fixed_vector_batch_round_trips_through_parquet(tmp_path):
    import pyarrow.parquet as pq

    stream = _stream(dtype="float64")
    table = component_records_to_arrow((), stream=stream)
    path = tmp_path / "empty.parquet"
    pq.write_table(table, path, compression="zstd")
    restored = pq.read_table(path)
    assert restored.schema == table.schema
    assert restored.num_rows == 0
    assert component_records_from_arrow(restored, identity=_identity(), profile=_profile(stream), stream=stream) == ()

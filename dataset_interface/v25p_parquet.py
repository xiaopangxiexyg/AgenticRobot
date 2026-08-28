"""Typed Parquet-table adaptation for native v2.5-p logical records.

This module owns no files, episode lifecycle, or commit authority.  It only
provides a lossless mapping between validated logical contracts and Arrow
tables whose schemas are derived from the bound component-stream profile.
"""

from __future__ import annotations

from typing import Any, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native import (
    V25pCausalSample,
    V25pComponentRecord,
    V25pComponentStreamProfileLike,
    V25pComponentStreamSpecLike,
    V25pExternalPayloadReference,
    normalize_component_stream_stored_values,
)


V25P_PARQUET_PHYSICAL_SCHEMA_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.parquet_physical_schema.semantic.v1"
)


def _arrow():
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Native v2.5-p Parquet chunks require pyarrow."
        ) from exc
    return pa


def _scalar_type(dtype: str):
    pa = _arrow()
    mapping = {
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
    }
    try:
        return mapping[str(dtype)]
    except KeyError as exc:  # pragma: no cover - stream contracts reject this first.
        raise ValueError(f"Unsupported native Parquet scalar dtype {dtype!r}.") from exc


def component_record_arrow_schema(stream: V25pComponentStreamSpecLike):
    pa = _arrow()
    fields = [
        pa.field("record_sequence", pa.int64(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("effective_at", pa.float64(), nullable=False),
        pa.field("available_at", pa.float64(), nullable=False),
        pa.field("effective_clock_domain_id", pa.string(), nullable=False),
        pa.field("available_clock_domain_id", pa.string(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("invalid_reason", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=True),
    ]
    if stream.storage_kind == "inline_tensor":
        fields.append(
            pa.field(
                "values",
                pa.list_(_scalar_type(stream.dtype), stream.flattened_size),
                nullable=True,
            )
        )
    else:
        fields.extend(
            (
                pa.field("artifact_locator", pa.string(), nullable=True),
                pa.field("artifact_content_hash", pa.binary(32), nullable=True),
                pa.field("encoding", pa.string(), nullable=True),
                pa.field("sample_selector", pa.string(), nullable=True),
                pa.field("byte_offset", pa.int64(), nullable=True),
                pa.field("byte_length", pa.int64(), nullable=True),
            )
        )
    fields.append(pa.field("record_hash", pa.binary(32), nullable=False))
    return pa.schema(fields)


def causal_sample_arrow_schema(profile: V25pComponentStreamProfileLike):
    pa = _arrow()
    return pa.schema(
        (
            pa.field("sample_index", pa.int64(), nullable=False),
            pa.field("cut_at", pa.float64(), nullable=False),
            pa.field("cut_clock_domain_id", pa.string(), nullable=False),
            pa.field(
                "record_hashes",
                pa.list_(pa.binary(32), len(profile.streams)),
                nullable=False,
            ),
            pa.field("sample_hash", pa.binary(32), nullable=False),
        )
    )


def component_record_physical_schema_hash(
    stream: V25pComponentStreamSpecLike,
) -> str:
    payload_columns: tuple[tuple[str, str], ...]
    if stream.storage_kind == "inline_tensor":
        payload_columns = (("values", f"fixed[{stream.flattened_size}]<{stream.dtype}>"),)
    else:
        payload_columns = (
            ("artifact_locator", "string?"),
            ("artifact_content_hash", "sha256?"),
            ("encoding", "string?"),
            ("sample_selector", "string?"),
            ("byte_offset", "int64?"),
            ("byte_length", "int64?"),
        )
    return canonical_content_hash(
        {
            "kind": "component_records",
            "stream_id": stream.stream_id,
            "storage_kind": stream.storage_kind,
            "payload_schema_hash": stream.payload_schema_hash,
            "columns": (
                ("record_sequence", "int64"),
                ("sample_index", "int64"),
                ("effective_at", "float64"),
                ("available_at", "float64"),
                ("effective_clock_domain_id", "string"),
                ("available_clock_domain_id", "string"),
                ("valid", "bool"),
                ("invalid_reason", "string"),
                ("source_record_id", "string?"),
                *payload_columns,
                ("record_hash", "sha256"),
            ),
        },
        domain=V25P_PARQUET_PHYSICAL_SCHEMA_HASH_DOMAIN_V1,
    )


def causal_sample_physical_schema_hash(
    profile: V25pComponentStreamProfileLike,
) -> str:
    return canonical_content_hash(
        {
            "kind": "causal_samples",
            "component_stream_profile_hash": profile.semantic_content_hash,
            "stream_order": tuple(item.stream_id for item in profile.streams),
            "columns": (
                ("sample_index", "int64"),
                ("cut_at", "float64"),
                ("cut_clock_domain_id", "string"),
                ("record_hashes", f"fixed[{len(profile.streams)}]<sha256?>"),
                ("sample_hash", "sha256"),
            ),
        },
        domain=V25P_PARQUET_PHYSICAL_SCHEMA_HASH_DOMAIN_V1,
    )


def component_records_to_arrow(
    records: Sequence[V25pComponentRecord],
    *,
    stream: V25pComponentStreamSpecLike,
):
    pa = _arrow()
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, V25pComponentRecord) or record.stream_id != stream.stream_id:
            raise ValueError("Component-record Arrow batch contains another stream.")
        row: dict[str, Any] = {
            "record_sequence": record.record_sequence,
            "sample_index": record.sample_index,
            "effective_at": record.effective_at,
            "available_at": record.available_at,
            "effective_clock_domain_id": record.effective_clock_domain_id,
            "available_clock_domain_id": record.available_clock_domain_id,
            "valid": record.valid,
            "invalid_reason": record.invalid_reason,
            "source_record_id": record.source_record_id,
            "record_hash": bytes.fromhex(record.semantic_content_hash),
        }
        if stream.storage_kind == "inline_tensor":
            row["values"] = None if record.values is None else list(record.values)
        else:
            reference = record.payload_reference
            row.update(
                {
                    "artifact_locator": None if reference is None else reference.artifact_locator,
                    "artifact_content_hash": (
                        None
                        if reference is None
                        else bytes.fromhex(reference.artifact_content_hash)
                    ),
                    "encoding": None if reference is None else reference.encoding,
                    "sample_selector": None if reference is None else reference.sample_selector,
                    "byte_offset": None if reference is None else reference.byte_offset,
                    "byte_length": None if reference is None else reference.byte_length,
                }
            )
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=component_record_arrow_schema(stream))


def component_records_from_arrow(
    table: Any,
    *,
    identity: EpisodeCaptureIdentity,
    profile: V25pComponentStreamProfileLike,
    stream: V25pComponentStreamSpecLike,
) -> tuple[V25pComponentRecord, ...]:
    expected_schema = component_record_arrow_schema(stream)
    if table.schema != expected_schema:
        raise ValueError(
            f"Parquet schema mismatch for native stream {stream.stream_id!r}."
        )
    records: list[V25pComponentRecord] = []
    for row in table.to_pylist():
        reference = None
        values = None
        if stream.storage_kind == "inline_tensor":
            raw_values = row["values"]
            values = (
                None
                if raw_values is None
                else normalize_component_stream_stored_values(stream, raw_values)
            )
        elif row["artifact_locator"] is not None:
            reference = V25pExternalPayloadReference(
                artifact_locator=row["artifact_locator"],
                artifact_content_hash=bytes(row["artifact_content_hash"]).hex(),
                encoding=row["encoding"],
                sample_selector=row["sample_selector"],
                byte_offset=row["byte_offset"],
                byte_length=row["byte_length"],
            )
        record = V25pComponentRecord(
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            component_stream_profile_hash=profile.semantic_content_hash,
            stream_id=stream.stream_id,
            component_id=stream.component_id,
            dataset_role=stream.dataset_role,
            value_stage=stream.value_stage,
            payload_schema_hash=stream.payload_schema_hash,
            record_sequence=row["record_sequence"],
            sample_index=row["sample_index"],
            effective_at=row["effective_at"],
            available_at=row["available_at"],
            effective_clock_domain_id=row["effective_clock_domain_id"],
            available_clock_domain_id=row["available_clock_domain_id"],
            values=values,
            payload_reference=reference,
            valid=row["valid"],
            invalid_reason=row["invalid_reason"],
            source_record_id=row["source_record_id"],
        )
        if record.semantic_content_hash != bytes(row["record_hash"]).hex():
            raise ValueError(
                f"Parquet record hash mismatch for native stream {stream.stream_id!r}."
            )
        records.append(record)
    return tuple(records)


def causal_samples_to_arrow(
    samples: Sequence[V25pCausalSample],
    *,
    profile: V25pComponentStreamProfileLike,
):
    pa = _arrow()
    stream_ids = tuple(item.stream_id for item in profile.streams)
    rows = [
        {
            "sample_index": sample.sample_index,
            "cut_at": sample.cut_at,
            "cut_clock_domain_id": sample.cut_clock_domain_id,
            "record_hashes": [
                (
                    None
                    if sample.record_hash_by_stream.get(stream_id) is None
                    else bytes.fromhex(sample.record_hash_by_stream[stream_id])
                )
                for stream_id in stream_ids
            ],
            "sample_hash": bytes.fromhex(sample.semantic_content_hash),
        }
        for sample in samples
    ]
    return pa.Table.from_pylist(rows, schema=causal_sample_arrow_schema(profile))


def causal_samples_from_arrow(
    table: Any,
    *,
    identity: EpisodeCaptureIdentity,
    profile: V25pComponentStreamProfileLike,
) -> tuple[V25pCausalSample, ...]:
    expected_schema = causal_sample_arrow_schema(profile)
    if table.schema != expected_schema:
        raise ValueError("Native causal-sample Parquet schema mismatch.")
    stream_ids = tuple(item.stream_id for item in profile.streams)
    samples: list[V25pCausalSample] = []
    for row in table.to_pylist():
        hashes = {
            stream_id: bytes(value).hex()
            for stream_id, value in zip(stream_ids, row["record_hashes"])
            if value is not None
        }
        sample = V25pCausalSample(
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            component_stream_profile_hash=profile.semantic_content_hash,
            sample_index=row["sample_index"],
            cut_at=row["cut_at"],
            cut_clock_domain_id=row["cut_clock_domain_id"],
            record_hash_by_stream=hashes,
        )
        if sample.semantic_content_hash != bytes(row["sample_hash"]).hex():
            raise ValueError("Native causal-sample Parquet hash mismatch.")
        samples.append(sample)
    return tuple(samples)


__all__ = [
    "V25P_PARQUET_PHYSICAL_SCHEMA_HASH_DOMAIN_V1",
    "causal_sample_arrow_schema",
    "causal_sample_physical_schema_hash",
    "causal_samples_from_arrow",
    "causal_samples_to_arrow",
    "component_record_arrow_schema",
    "component_record_physical_schema_hash",
    "component_records_from_arrow",
    "component_records_to_arrow",
]

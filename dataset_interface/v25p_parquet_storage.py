"""Read-only verification of native v2.5-p Parquet chunk storage."""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash, canonical_loads
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native import (
    V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
    V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
    V25pCausalSample,
    V25pComponentRecord,
    V25pComponentStreamProfileLike,
)
from dataset_contracts.v25p_native_storage import (
    V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1,
    V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1,
    V25pNativeStorageManifest,
    V25pParquetChunkManifest,
)

from .v25p_parquet import (
    causal_sample_physical_schema_hash,
    causal_samples_from_arrow,
    component_record_physical_schema_hash,
    component_records_from_arrow,
)


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Native v2.5-p Parquet chunks require pyarrow."
        ) from exc
    return pq


def _mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required native Parquet artifact is missing: {path}")
    payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Native Parquet manifest is not a mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_v25p_parquet_storage(
    *,
    episode_dir: Path,
    identity: EpisodeCaptureIdentity,
    profile: V25pComponentStreamProfileLike,
) -> tuple[
    V25pNativeStorageManifest,
    Mapping[str, tuple[V25pComponentRecord, ...]],
    tuple[V25pCausalSample, ...],
]:
    pq = _parquet()
    storage = V25pNativeStorageManifest.from_payload(
        _mapping(
            episode_dir / "native_storage_manifest.cjson",
            domain=V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    if (
        storage.episode_uuid,
        storage.episode_identity_hash,
        storage.component_stream_profile_hash,
    ) != (
        identity.episode_uuid,
        identity.semantic_content_hash,
        profile.semantic_content_hash,
    ):
        raise ValueError("Native Parquet storage identity/profile binding mismatch.")
    records_by_stream: dict[str, list[V25pComponentRecord]] = {
        stream.stream_id: [] for stream in profile.streams
    }
    samples: list[V25pCausalSample] = []
    expected_sample_begin = 0
    for chunk_index, expected_hash in enumerate(storage.chunk_manifest_hashes):
        chunk = V25pParquetChunkManifest.from_payload(
            _mapping(
                episode_dir / "chunks" / f"chunk_{chunk_index:06d}.cjson",
                domain=V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1,
            )
        )
        if chunk.semantic_content_hash != expected_hash:
            raise ValueError("Native Parquet chunk manifest hash mismatch.")
        if (
            chunk.episode_uuid,
            chunk.episode_identity_hash,
            chunk.component_stream_profile_hash,
            chunk.chunk_index,
            chunk.sample_index_begin,
        ) != (
            identity.episode_uuid,
            identity.semantic_content_hash,
            profile.semantic_content_hash,
            chunk_index,
            expected_sample_begin,
        ):
            raise ValueError("Native Parquet chunk ordering/binding mismatch.")
        if storage.chunk_sample_ranges[chunk_index] != (
            chunk.sample_index_begin,
            chunk.sample_index_end,
        ):
            raise ValueError("Native Parquet storage/chunk sample range mismatch.")
        sample_artifact = chunk.sample_artifact
        sample_path = episode_dir / sample_artifact.relative_path
        if (
            sample_path.stat().st_size != sample_artifact.byte_count
            or _file_sha256(sample_path) != sample_artifact.file_sha256
            or sample_artifact.physical_schema_hash
            != causal_sample_physical_schema_hash(profile)
        ):
            raise ValueError("Native causal-sample Parquet artifact integrity mismatch.")
        chunk_samples = causal_samples_from_arrow(
            pq.read_table(sample_path),
            identity=identity,
            profile=profile,
        )
        if (
            len(chunk_samples) != sample_artifact.row_count
            or not chunk_samples
            or chunk_samples[0].sample_index != sample_artifact.first_index
            or chunk_samples[-1].sample_index != sample_artifact.last_index
            or canonical_content_hash(
                tuple(item.semantic_content_hash for item in chunk_samples),
                domain=V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
            )
            != sample_artifact.logical_bundle_hash
        ):
            raise ValueError("Native causal-sample Parquet logical bundle mismatch.")
        artifacts_by_stream = {
            item.stream_id: item for item in chunk.stream_artifacts
        }
        if set(artifacts_by_stream) - set(profile.stream_by_id):
            raise ValueError("Native Parquet chunk contains an unknown stream.")
        for stream in profile.streams:
            artifact = artifacts_by_stream.get(stream.stream_id)
            if artifact is None:
                continue
            path = episode_dir / artifact.relative_path
            if (
                path.stat().st_size != artifact.byte_count
                or _file_sha256(path) != artifact.file_sha256
                or artifact.physical_schema_hash
                != component_record_physical_schema_hash(stream)
            ):
                raise ValueError(
                    f"Native stream {stream.stream_id!r} Parquet integrity mismatch."
                )
            chunk_records = component_records_from_arrow(
                pq.read_table(path),
                identity=identity,
                profile=profile,
                stream=stream,
            )
            expected_first = len(records_by_stream[stream.stream_id]) + 1
            if (
                len(chunk_records) != artifact.row_count
                or not chunk_records
                or artifact.first_index != expected_first
                or chunk_records[0].record_sequence != artifact.first_index
                or chunk_records[-1].record_sequence != artifact.last_index
                or canonical_content_hash(
                    tuple(item.semantic_content_hash for item in chunk_records),
                    domain=V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
                )
                != artifact.logical_bundle_hash
            ):
                raise ValueError(
                    f"Native stream {stream.stream_id!r} Parquet logical bundle mismatch."
                )
            records_by_stream[stream.stream_id].extend(chunk_records)
        samples.extend(chunk_samples)
        expected_sample_begin = chunk.sample_index_end + 1

    frozen_records = {
        stream_id: tuple(records)
        for stream_id, records in records_by_stream.items()
    }
    if len(samples) != storage.sample_count or {
        stream_id: len(records)
        for stream_id, records in frozen_records.items()
    } != dict(storage.record_count_by_stream):
        raise ValueError("Native Parquet storage counts disagree with its manifest.")
    return storage, frozen_records, tuple(samples)


def load_v25p_parquet_sample(
    *,
    episode_dir: Path,
    identity: EpisodeCaptureIdentity,
    profile: V25pComponentStreamProfileLike,
    sample_index: int,
    stream_ids: tuple[str, ...] | None = None,
) -> tuple[V25pCausalSample, Mapping[str, V25pComponentRecord]]:
    """Read one causal sample and only the requested component stream shards."""

    pq = _parquet()
    storage = V25pNativeStorageManifest.from_payload(
        _mapping(
            episode_dir / "native_storage_manifest.cjson",
            domain=V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    if (
        storage.episode_uuid,
        storage.episode_identity_hash,
        storage.component_stream_profile_hash,
    ) != (
        identity.episode_uuid,
        identity.semantic_content_hash,
        profile.semantic_content_hash,
    ):
        raise ValueError("Native random-read storage identity/profile mismatch.")
    if set(storage.record_count_by_stream) != set(profile.stream_by_id):
        raise ValueError("Native random-read storage stream coverage mismatch.")
    requested_index = int(sample_index)
    if requested_index < 0 or requested_index >= storage.sample_count:
        raise IndexError(f"Native sample index out of range: {requested_index}.")
    ends = tuple(item[1] for item in storage.chunk_sample_ranges)
    chunk_index = bisect_left(ends, requested_index)
    chunk = V25pParquetChunkManifest.from_payload(
        _mapping(
            episode_dir / "chunks" / f"chunk_{chunk_index:06d}.cjson",
            domain=V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1,
        )
    )
    if chunk.semantic_content_hash != storage.chunk_manifest_hashes[chunk_index]:
        raise ValueError("Native random-read chunk manifest hash mismatch.")
    if (
        chunk.episode_uuid,
        chunk.episode_identity_hash,
        chunk.component_stream_profile_hash,
        chunk.chunk_index,
        (chunk.sample_index_begin, chunk.sample_index_end),
    ) != (
        identity.episode_uuid,
        identity.semantic_content_hash,
        profile.semantic_content_hash,
        chunk_index,
        storage.chunk_sample_ranges[chunk_index],
    ):
        raise ValueError("Native random-read chunk ordering/binding mismatch.")
    sample_artifact = chunk.sample_artifact
    sample_path = episode_dir / sample_artifact.relative_path
    if (
        sample_path.stat().st_size != sample_artifact.byte_count
        or _file_sha256(sample_path) != sample_artifact.file_sha256
        or sample_artifact.physical_schema_hash
        != causal_sample_physical_schema_hash(profile)
    ):
        raise ValueError("Native random-read sample chunk integrity mismatch.")
    chunk_samples = causal_samples_from_arrow(
        pq.read_table(sample_path),
        identity=identity,
        profile=profile,
    )
    if (
        len(chunk_samples) != sample_artifact.row_count
        or canonical_content_hash(
            tuple(item.semantic_content_hash for item in chunk_samples),
            domain=V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
        )
        != sample_artifact.logical_bundle_hash
    ):
        raise ValueError("Native random-read sample logical bundle mismatch.")
    sample_offset = requested_index - chunk.sample_index_begin
    sample = chunk_samples[sample_offset]
    if sample.sample_index != requested_index:
        raise ValueError("Native random-read sample index lookup failed.")
    selected = (
        tuple(sample.record_hash_by_stream)
        if stream_ids is None
        else tuple(str(item) for item in stream_ids)
    )
    if len(selected) != len(set(selected)) or set(selected) - set(profile.stream_by_id):
        raise ValueError("Native random read requested unknown/duplicate streams.")
    artifacts_by_stream = {
        item.stream_id: item for item in chunk.stream_artifacts
    }
    records: dict[str, V25pComponentRecord] = {}
    for stream_id in selected:
        expected_hash = sample.record_hash_by_stream.get(stream_id)
        if expected_hash is None:
            continue
        artifact = artifacts_by_stream.get(stream_id)
        if artifact is None:
            raise ValueError("Native random-read sample references an absent stream shard.")
        path = episode_dir / artifact.relative_path
        stream = profile.stream_by_id[stream_id]
        if (
            path.stat().st_size != artifact.byte_count
            or _file_sha256(path) != artifact.file_sha256
            or artifact.physical_schema_hash
            != component_record_physical_schema_hash(stream)
        ):
            raise ValueError(
                f"Native random-read stream {stream_id!r} integrity mismatch."
            )
        candidates = component_records_from_arrow(
            pq.read_table(path),
            identity=identity,
            profile=profile,
            stream=stream,
        )
        matches = tuple(
            item for item in candidates if item.semantic_content_hash == expected_hash
        )
        if len(matches) != 1 or matches[0].sample_index != requested_index:
            raise ValueError("Native random-read causal record lookup failed.")
        records[stream_id] = matches[0]
    return sample, MappingProxyType(records)


__all__ = ["load_v25p_parquet_sample", "load_v25p_parquet_storage"]

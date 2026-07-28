"""Pure physical-storage contracts for provisional native v2.5-p episodes.

Logical component-record and causal-sample hashes remain owned by
``dataset_contracts.v25p_native``.  These contracts bind one immutable physical
representation without making Parquet bytes part of the logical data identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


V25P_RECORD_CJSON_LAYOUT_V1 = "record_cjson_v1"
V25P_PARQUET_CHUNK_LAYOUT_V1 = "parquet_stream_chunks_v1"
V25P_NATIVE_STORAGE_LAYOUTS_V1 = (
    V25P_RECORD_CJSON_LAYOUT_V1,
    V25P_PARQUET_CHUNK_LAYOUT_V1,
)
V25P_PARQUET_CHUNK_MANIFEST_SCHEMA_V1 = (
    "dexteleop_v25p_parquet_chunk_manifest_v1"
)
V25P_NATIVE_STORAGE_MANIFEST_SCHEMA_V1 = (
    "dexteleop_v25p_native_storage_manifest_v1"
)
V25P_PARQUET_CHUNK_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.parquet_chunk_manifest.semantic.v1"
)
V25P_NATIVE_STORAGE_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.v25p.native_storage_manifest.semantic.v1"
)
V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.v25p.parquet_chunk_manifest.file.v1"
)
V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.v25p.native_storage_manifest.file.v1"
)


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _hash(value: object, *, name: str) -> str:
    result = _required(value, name=name).removeprefix("sha256:").lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return result


def _index(value: object, *, name: str, minimum: int = 0) -> int:
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _relative_path(value: object) -> str:
    result = _required(value, name="relative_path").replace("\\", "/")
    parts = result.split("/")
    if result.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("relative_path must be a safe normalized relative path.")
    return result


@dataclass(frozen=True)
class V25pChunkArtifact:
    artifact_kind: str
    relative_path: str
    file_sha256: str
    byte_count: int
    row_count: int
    first_index: int
    last_index: int
    logical_bundle_hash: str
    physical_schema_hash: str
    stream_id: str | None = None

    def __post_init__(self) -> None:
        kind = _required(self.artifact_kind, name="artifact_kind")
        if kind not in {"component_records", "causal_samples"}:
            raise ValueError(f"Unsupported chunk artifact_kind {kind!r}.")
        stream_id = None if self.stream_id is None else _required(
            self.stream_id, name="stream_id"
        )
        if (kind == "component_records") != (stream_id is not None):
            raise ValueError(
                "Only component-record chunk artifacts bind a stream_id."
            )
        first = _index(self.first_index, name="first_index")
        last = _index(self.last_index, name="last_index")
        rows = _index(self.row_count, name="row_count", minimum=1)
        if last < first or last - first + 1 != rows:
            raise ValueError("Chunk artifact indices must be contiguous and match row_count.")
        object.__setattr__(self, "artifact_kind", kind)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "file_sha256", _hash(self.file_sha256, name="file_sha256"))
        object.__setattr__(
            self,
            "byte_count",
            _index(self.byte_count, name="byte_count", minimum=1),
        )
        object.__setattr__(self, "row_count", rows)
        object.__setattr__(self, "first_index", first)
        object.__setattr__(self, "last_index", last)
        object.__setattr__(
            self,
            "logical_bundle_hash",
            _hash(self.logical_bundle_hash, name="logical_bundle_hash"),
        )
        object.__setattr__(
            self,
            "physical_schema_hash",
            _hash(self.physical_schema_hash, name="physical_schema_hash"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "stream_id": self.stream_id,
            "relative_path": self.relative_path,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "row_count": self.row_count,
            "first_index": self.first_index,
            "last_index": self.last_index,
            "logical_bundle_hash": self.logical_bundle_hash,
            "physical_schema_hash": self.physical_schema_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pChunkArtifact":
        return cls(
            artifact_kind=payload.get("artifact_kind", ""),
            stream_id=payload.get("stream_id"),
            relative_path=payload.get("relative_path", ""),
            file_sha256=payload.get("file_sha256", ""),
            byte_count=payload.get("byte_count", 0),
            row_count=payload.get("row_count", 0),
            first_index=payload.get("first_index", -1),
            last_index=payload.get("last_index", -1),
            logical_bundle_hash=payload.get("logical_bundle_hash", ""),
            physical_schema_hash=payload.get("physical_schema_hash", ""),
        )


@dataclass(frozen=True)
class V25pParquetChunkManifest:
    episode_uuid: str
    episode_identity_hash: str
    component_stream_profile_hash: str
    chunk_index: int
    sample_index_begin: int
    sample_index_end: int
    sample_artifact: V25pChunkArtifact
    stream_artifacts: tuple[V25pChunkArtifact, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_PARQUET_CHUNK_MANIFEST_SCHEMA_V1, init=False)
    storage_layout: str = field(default=V25P_PARQUET_CHUNK_LAYOUT_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "episode_identity_hash",
            _hash(self.episode_identity_hash, name="episode_identity_hash"),
        )
        object.__setattr__(
            self,
            "component_stream_profile_hash",
            _hash(
                self.component_stream_profile_hash,
                name="component_stream_profile_hash",
            ),
        )
        chunk_index = _index(self.chunk_index, name="chunk_index")
        begin = _index(self.sample_index_begin, name="sample_index_begin")
        end = _index(self.sample_index_end, name="sample_index_end")
        if end < begin:
            raise ValueError("Chunk sample range is inverted.")
        if not isinstance(self.sample_artifact, V25pChunkArtifact):
            raise TypeError("sample_artifact must be V25pChunkArtifact.")
        if self.sample_artifact.artifact_kind != "causal_samples":
            raise ValueError("sample_artifact must contain causal samples.")
        if (
            self.sample_artifact.first_index,
            self.sample_artifact.last_index,
        ) != (begin, end):
            raise ValueError("Sample artifact range does not match its chunk.")
        artifacts = tuple(self.stream_artifacts)
        if any(
            not isinstance(item, V25pChunkArtifact)
            or item.artifact_kind != "component_records"
            for item in artifacts
        ):
            raise TypeError("stream_artifacts must contain component-record artifacts.")
        stream_ids = tuple(item.stream_id for item in artifacts)
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("Chunk stream artifacts must bind unique streams.")
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "sample_index_begin", begin)
        object.__setattr__(self, "sample_index_end", end)
        object.__setattr__(self, "stream_artifacts", artifacts)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_PARQUET_CHUNK_MANIFEST_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "storage_layout": self.storage_layout,
            "episode_uuid": self.episode_uuid,
            "episode_identity_hash": self.episode_identity_hash,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "chunk_index": self.chunk_index,
            "sample_index_begin": self.sample_index_begin,
            "sample_index_end": self.sample_index_end,
            "sample_artifact": self.sample_artifact.to_payload(),
            "stream_artifacts": [item.to_payload() for item in self.stream_artifacts],
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pParquetChunkManifest":
        if payload.get("schema") != V25P_PARQUET_CHUNK_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported v2.5-p Parquet chunk manifest schema.")
        if payload.get("storage_layout") != V25P_PARQUET_CHUNK_LAYOUT_V1:
            raise ValueError("Unsupported v2.5-p Parquet chunk layout.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            component_stream_profile_hash=payload.get(
                "component_stream_profile_hash", ""
            ),
            chunk_index=payload.get("chunk_index", -1),
            sample_index_begin=payload.get("sample_index_begin", -1),
            sample_index_end=payload.get("sample_index_end", -1),
            sample_artifact=V25pChunkArtifact.from_payload(
                dict(payload.get("sample_artifact", {}))
            ),
            stream_artifacts=tuple(
                V25pChunkArtifact.from_payload(item)
                for item in payload.get("stream_artifacts", ())
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared not in {None, result.semantic_content_hash}:
            raise ValueError("Parquet chunk manifest hash mismatch.")
        return result


@dataclass(frozen=True)
class V25pNativeStorageManifest:
    episode_uuid: str
    episode_identity_hash: str
    component_stream_profile_hash: str
    chunk_manifest_hashes: tuple[str, ...]
    chunk_sample_ranges: tuple[tuple[int, int], ...]
    sample_count: int
    record_count_by_stream: Mapping[str, int]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V25P_NATIVE_STORAGE_MANIFEST_SCHEMA_V1, init=False)
    storage_layout: str = field(default=V25P_PARQUET_CHUNK_LAYOUT_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_uuid", _required(self.episode_uuid, name="episode_uuid"))
        object.__setattr__(
            self,
            "episode_identity_hash",
            _hash(self.episode_identity_hash, name="episode_identity_hash"),
        )
        object.__setattr__(
            self,
            "component_stream_profile_hash",
            _hash(
                self.component_stream_profile_hash,
                name="component_stream_profile_hash",
            ),
        )
        chunk_hashes = tuple(
            _hash(item, name="chunk_manifest_hash")
            for item in self.chunk_manifest_hashes
        )
        if not chunk_hashes or len(chunk_hashes) != len(set(chunk_hashes)):
            raise ValueError("Storage manifest requires unique chunk manifest hashes.")
        ranges = tuple(
            (
                _index(item[0], name="chunk sample range begin"),
                _index(item[1], name="chunk sample range end"),
            )
            for item in self.chunk_sample_ranges
        )
        if len(ranges) != len(chunk_hashes):
            raise ValueError("Storage chunk ranges and hashes must have equal length.")
        expected_begin = 0
        for begin, end in ranges:
            if begin != expected_begin or end < begin:
                raise ValueError("Storage chunk sample ranges must be contiguous.")
            expected_begin = end + 1
        counts = {
            _required(key, name="stream_id"): _index(value, name="record_count")
            for key, value in dict(self.record_count_by_stream).items()
        }
        object.__setattr__(self, "chunk_manifest_hashes", chunk_hashes)
        object.__setattr__(self, "chunk_sample_ranges", ranges)
        object.__setattr__(
            self,
            "sample_count",
            _index(self.sample_count, name="sample_count", minimum=1),
        )
        if expected_begin != self.sample_count:
            raise ValueError("Storage chunk sample ranges do not cover sample_count.")
        object.__setattr__(
            self,
            "record_count_by_stream",
            MappingProxyType(dict(sorted(counts.items()))),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=V25P_NATIVE_STORAGE_MANIFEST_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "storage_layout": self.storage_layout,
            "episode_uuid": self.episode_uuid,
            "episode_identity_hash": self.episode_identity_hash,
            "component_stream_profile_hash": self.component_stream_profile_hash,
            "chunk_manifest_hashes": self.chunk_manifest_hashes,
            "chunk_sample_ranges": self.chunk_sample_ranges,
            "sample_count": self.sample_count,
            "record_count_by_stream": dict(self.record_count_by_stream),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "V25pNativeStorageManifest":
        if payload.get("schema") != V25P_NATIVE_STORAGE_MANIFEST_SCHEMA_V1:
            raise ValueError("Unsupported native v2.5-p storage manifest schema.")
        if payload.get("storage_layout") != V25P_PARQUET_CHUNK_LAYOUT_V1:
            raise ValueError("Unsupported native v2.5-p storage layout.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            episode_identity_hash=payload.get("episode_identity_hash", ""),
            component_stream_profile_hash=payload.get(
                "component_stream_profile_hash", ""
            ),
            chunk_manifest_hashes=tuple(payload.get("chunk_manifest_hashes", ())),
            chunk_sample_ranges=tuple(
                tuple(item) for item in payload.get("chunk_sample_ranges", ())
            ),
            sample_count=payload.get("sample_count", 0),
            record_count_by_stream=dict(payload.get("record_count_by_stream", {})),
        )
        declared = payload.get("semantic_content_hash")
        if declared not in {None, result.semantic_content_hash}:
            raise ValueError("Native storage manifest hash mismatch.")
        return result


__all__ = [
    "V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1",
    "V25P_NATIVE_STORAGE_MANIFEST_HASH_DOMAIN_V1",
    "V25P_NATIVE_STORAGE_MANIFEST_SCHEMA_V1",
    "V25P_PARQUET_CHUNK_LAYOUT_V1",
    "V25P_RECORD_CJSON_LAYOUT_V1",
    "V25P_NATIVE_STORAGE_LAYOUTS_V1",
    "V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1",
    "V25P_PARQUET_CHUNK_MANIFEST_HASH_DOMAIN_V1",
    "V25P_PARQUET_CHUNK_MANIFEST_SCHEMA_V1",
    "V25pChunkArtifact",
    "V25pNativeStorageManifest",
    "V25pParquetChunkManifest",
]

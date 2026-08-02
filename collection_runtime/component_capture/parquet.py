"""Bounded immutable Parquet-chunk persistence for component samples."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
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
    V25pChunkArtifact,
    V25pNativeStorageManifest,
    V25pParquetChunkManifest,
)
from dataset_interface.v25p_parquet import (
    causal_sample_physical_schema_hash,
    causal_samples_from_arrow,
    causal_samples_to_arrow,
    component_record_physical_schema_hash,
    component_records_from_arrow,
    component_records_to_arrow,
)

from ..immutable_canonical_store import fsync_directory, publish_immutable_canonical


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Native v2.5-p Parquet chunks require pyarrow."
        ) from exc
    return pq


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_parquet_table(path: Path, table, *, compression: str):
    pq = _parquet()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        if existing.schema != table.schema or not existing.equals(table):
            raise ValueError(f"Immutable Parquet artifact differs at {path}.")
        return existing
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        codec = None if compression == "none" else compression
        pq.write_table(
            table,
            temporary,
            compression=codec,
            row_group_size=max(1, int(table.num_rows)),
            write_statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        verified = pq.read_table(temporary)
        if verified.schema != table.schema or not verified.equals(table):
            raise ValueError(f"Parquet round-trip changed native data at {path}.")
        os.replace(temporary, path)
        fsync_directory(path.parent)
        return verified
    finally:
        temporary.unlink(missing_ok=True)


class ParquetChunkEpisodeWriter:
    """One externally-owned episode with bounded sample/record buffers."""

    def __init__(
        self,
        *,
        episode_dir: Path,
        identity: EpisodeCaptureIdentity,
        component_stream_profile: V25pComponentStreamProfileLike,
        chunk_rows: int = 128,
        compression: str = "none",
    ) -> None:
        rows = int(chunk_rows)
        if rows <= 0:
            raise ValueError("v2.5-p Parquet chunk_rows must be positive.")
        normalized_compression = str(compression).strip().lower()
        if normalized_compression not in {"none", "zstd"}:
            raise ValueError("v2.5-p Parquet compression must be none or zstd.")
        self.episode_dir = Path(episode_dir)
        self.identity = identity
        self.profile = component_stream_profile
        self.chunk_rows = rows
        self.compression = normalized_compression
        self._buffer_records = {
            stream.stream_id: [] for stream in self.profile.streams
        }
        self._buffer_samples: list[V25pCausalSample] = []
        self._record_counts = {
            stream.stream_id: 0 for stream in self.profile.streams
        }
        self._sample_count = 0
        self._chunk_manifests: list[V25pParquetChunkManifest] = []
        self._next_chunk_index = 0
        self._pending: list[tuple[int, Future[V25pParquetChunkManifest]]] = []
        self._max_pending_chunks = 2
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="v25p-parquet",
        )
        self._sealed = False
        self._closed = False

    @property
    def buffered_sample_count(self) -> int:
        return len(self._buffer_samples)

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_manifests)

    def append(
        self,
        *,
        records_by_stream: Mapping[str, V25pComponentRecord],
        sample: V25pCausalSample,
    ) -> None:
        if self._closed:
            raise RuntimeError("The v2.5-p Parquet chunk writer is closed.")
        self._collect_ready_chunks()
        expected_sample = self._sample_count
        if sample.sample_index != expected_sample:
            raise ValueError("Parquet chunk sample indices must be contiguous.")
        if set(records_by_stream) != set(sample.record_hash_by_stream):
            raise ValueError("Parquet chunk sample and record stream coverage differs.")
        known = set(self.profile.stream_by_id)
        if set(records_by_stream) - known:
            raise ValueError("Parquet chunk contains an unknown native stream.")
        for stream_id, record in records_by_stream.items():
            expected_sequence = self._record_counts[stream_id] + 1
            if (
                record.record_sequence != expected_sequence
                or record.sample_index != sample.sample_index
                or record.semantic_content_hash
                != sample.record_hash_by_stream[stream_id]
            ):
                raise ValueError("Parquet chunk component record ordering mismatch.")
        for stream_id, record in records_by_stream.items():
            self._buffer_records[stream_id].append(record)
            self._record_counts[stream_id] += 1
        self._buffer_samples.append(sample)
        self._sample_count += 1
        if len(self._buffer_samples) >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        """Schedule the current bounded chunk; file I/O stays off the caller."""

        if self._closed:
            raise RuntimeError("The v2.5-p Parquet chunk writer is closed.")
        if not self._buffer_samples:
            return
        chunk_index = self._next_chunk_index
        self._next_chunk_index += 1
        records_by_stream = {
            stream_id: tuple(records)
            for stream_id, records in self._buffer_records.items()
        }
        samples = tuple(self._buffer_samples)
        for records in self._buffer_records.values():
            records.clear()
        self._buffer_samples.clear()
        future = self._executor.submit(
            self._write_chunk,
            chunk_index=chunk_index,
            records_by_stream=records_by_stream,
            samples=samples,
        )
        self._pending.append((chunk_index, future))
        self._collect_ready_chunks()
        if len(self._pending) > self._max_pending_chunks:
            self._collect_oldest_chunk(block=True)

    def _write_chunk(
        self,
        *,
        chunk_index: int,
        records_by_stream: Mapping[str, tuple[V25pComponentRecord, ...]],
        samples: tuple[V25pCausalSample, ...],
    ) -> V25pParquetChunkManifest:
        sample_begin = samples[0].sample_index
        sample_end = samples[-1].sample_index
        stream_artifacts: list[V25pChunkArtifact] = []
        for stream_ordinal, stream in enumerate(self.profile.streams):
            records = records_by_stream[stream.stream_id]
            if not records:
                continue
            relative_path = (
                Path("streams")
                / f"s{stream_ordinal:04d}"
                / f"chunk_{chunk_index:06d}.parquet"
            )
            table = component_records_to_arrow(records, stream=stream)
            verified = _publish_parquet_table(
                self.episode_dir / relative_path,
                table,
                compression=self.compression,
            )
            rebuilt = component_records_from_arrow(
                verified,
                identity=self.identity,
                profile=self.profile,
                stream=stream,
            )
            if tuple(item.semantic_content_hash for item in rebuilt) != tuple(
                item.semantic_content_hash for item in records
            ):
                raise ValueError("Parquet component-record verification changed content.")
            path = self.episode_dir / relative_path
            stream_artifacts.append(
                V25pChunkArtifact(
                    artifact_kind="component_records",
                    stream_id=stream.stream_id,
                    relative_path=relative_path.as_posix(),
                    file_sha256=_file_sha256(path),
                    byte_count=path.stat().st_size,
                    row_count=len(records),
                    first_index=records[0].record_sequence,
                    last_index=records[-1].record_sequence,
                    logical_bundle_hash=canonical_content_hash(
                        tuple(item.semantic_content_hash for item in records),
                        domain=V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1,
                    ),
                    physical_schema_hash=component_record_physical_schema_hash(
                        stream
                    ),
                )
            )

        sample_relative_path = Path("samples") / f"chunk_{chunk_index:06d}.parquet"
        sample_table = causal_samples_to_arrow(
            samples,
            profile=self.profile,
        )
        verified_samples = _publish_parquet_table(
            self.episode_dir / sample_relative_path,
            sample_table,
            compression=self.compression,
        )
        rebuilt_samples = causal_samples_from_arrow(
            verified_samples,
            identity=self.identity,
            profile=self.profile,
        )
        if tuple(item.semantic_content_hash for item in rebuilt_samples) != tuple(
            item.semantic_content_hash for item in samples
        ):
            raise ValueError("Parquet causal-sample verification changed content.")
        sample_path = self.episode_dir / sample_relative_path
        sample_artifact = V25pChunkArtifact(
            artifact_kind="causal_samples",
            relative_path=sample_relative_path.as_posix(),
            file_sha256=_file_sha256(sample_path),
            byte_count=sample_path.stat().st_size,
            row_count=len(samples),
            first_index=sample_begin,
            last_index=sample_end,
            logical_bundle_hash=canonical_content_hash(
                tuple(item.semantic_content_hash for item in samples),
                domain=V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1,
            ),
            physical_schema_hash=causal_sample_physical_schema_hash(self.profile),
        )
        manifest = V25pParquetChunkManifest(
            episode_uuid=self.identity.episode_uuid,
            episode_identity_hash=self.identity.semantic_content_hash,
            component_stream_profile_hash=self.profile.semantic_content_hash,
            chunk_index=chunk_index,
            sample_index_begin=sample_begin,
            sample_index_end=sample_end,
            sample_artifact=sample_artifact,
            stream_artifacts=tuple(stream_artifacts),
        )
        publish_immutable_canonical(
            self.episode_dir / "chunks" / f"chunk_{chunk_index:06d}.cjson",
            manifest.to_payload(),
            domain=V25P_PARQUET_CHUNK_MANIFEST_FILE_DOMAIN_V1,
        )
        return manifest

    def _collect_ready_chunks(self) -> None:
        while self._pending and self._pending[0][1].done():
            self._collect_oldest_chunk(block=False)

    def _collect_oldest_chunk(self, *, block: bool) -> None:
        if not self._pending:
            return
        chunk_index, future = self._pending[0]
        if not block and not future.done():
            return
        manifest = future.result()
        if (
            manifest.chunk_index != chunk_index
            or chunk_index != len(self._chunk_manifests)
        ):
            raise RuntimeError("Parquet background chunks completed out of order.")
        self._pending.pop(0)
        self._chunk_manifests.append(manifest)

    def seal(self) -> V25pNativeStorageManifest:
        if self._closed:
            raise RuntimeError("The v2.5-p Parquet chunk writer is already sealed.")
        try:
            self.flush()
            while self._pending:
                self._collect_oldest_chunk(block=True)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True
        if (
            not self._chunk_manifests
            or self._sample_count <= 0
            or len(self._chunk_manifests) != self._next_chunk_index
        ):
            raise RuntimeError("A native Parquet episode cannot seal incomplete chunks.")
        manifest = V25pNativeStorageManifest(
            episode_uuid=self.identity.episode_uuid,
            episode_identity_hash=self.identity.semantic_content_hash,
            component_stream_profile_hash=self.profile.semantic_content_hash,
            chunk_manifest_hashes=tuple(
                item.semantic_content_hash for item in self._chunk_manifests
            ),
            chunk_sample_ranges=tuple(
                (item.sample_index_begin, item.sample_index_end)
                for item in self._chunk_manifests
            ),
            sample_count=self._sample_count,
            record_count_by_stream=self._record_counts,
        )
        publish_immutable_canonical(
            self.episode_dir / "native_storage_manifest.cjson",
            manifest.to_payload(),
            domain=V25P_NATIVE_STORAGE_MANIFEST_FILE_DOMAIN_V1,
        )
        self._sealed = True
        return manifest

    def close_unsealed(self) -> None:
        """Release a subordinate writer without publishing a storage seal."""

        if self._closed:
            return
        for records in self._buffer_records.values():
            records.clear()
        self._buffer_samples.clear()
        self._executor.shutdown(wait=True, cancel_futures=False)
        for _chunk_index, future in self._pending:
            try:
                future.result()
            except Exception:
                pass
        self._pending.clear()
        self._closed = True

    def logical_hash_sequences(
        self,
    ) -> tuple[Mapping[str, tuple[str, ...]], tuple[str, ...]]:
        """Rebuild exact logical hash order without retaining episode objects."""

        if not self._sealed:
            raise RuntimeError("Seal the v2.5-p Parquet writer before reading hashes.")
        pq = _parquet()
        record_hashes: dict[str, list[str]] = {
            stream.stream_id: [] for stream in self.profile.streams
        }
        sample_hashes: list[str] = []
        for chunk in self._chunk_manifests:
            sample_path = self.episode_dir / chunk.sample_artifact.relative_path
            sample_values = self._verified_hash_column(
                pq=pq,
                path=sample_path,
                artifact=chunk.sample_artifact,
                column="sample_hash",
            )
            sample_hashes.extend(sample_values)
            for artifact in chunk.stream_artifacts:
                assert artifact.stream_id is not None
                path = self.episode_dir / artifact.relative_path
                record_hashes[artifact.stream_id].extend(
                    self._verified_hash_column(
                        pq=pq,
                        path=path,
                        artifact=artifact,
                        column="record_hash",
                    )
                )
        if len(sample_hashes) != self._sample_count or {
            stream_id: len(values)
            for stream_id, values in record_hashes.items()
        } != self._record_counts:
            raise ValueError("Parquet logical hash scan disagrees with episode counts.")
        return (
            {stream_id: tuple(values) for stream_id, values in record_hashes.items()},
            tuple(sample_hashes),
        )

    @staticmethod
    def _verified_hash_column(*, pq, path: Path, artifact, column: str) -> tuple[str, ...]:
        if (
            not path.is_file()
            or path.stat().st_size != artifact.byte_count
            or _file_sha256(path) != artifact.file_sha256
        ):
            raise ValueError(f"Parquet hash scan integrity mismatch at {path}.")
        table = pq.read_table(path, columns=[column])
        values = tuple(bytes(item.as_py()).hex() for item in table.column(0))
        if len(values) != artifact.row_count:
            raise ValueError(f"Parquet hash scan row count mismatch at {path}.")
        domain = (
            V25P_NATIVE_SAMPLE_BUNDLE_HASH_DOMAIN_V1
            if artifact.artifact_kind == "causal_samples"
            else V25P_NATIVE_RECORD_BUNDLE_HASH_DOMAIN_V1
        )
        if canonical_content_hash(values, domain=domain) != artifact.logical_bundle_hash:
            raise ValueError(f"Parquet hash scan logical bundle mismatch at {path}.")
        return values


__all__ = ["ParquetChunkEpisodeWriter"]

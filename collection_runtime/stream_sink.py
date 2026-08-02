"""Append-only v2.4 native-stream episode storage.

This storage owner is deliberately separate from the decision-row writer.
Callers may feed it from the existing bounded collection writer runtime; this
module itself never polls a device, advances control state, or dispatches an
action.
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataset_contracts.canonical_encoding import canonical_bytes
from dataset_contracts.layout import (
    chunk_id_from_episode_index,
    episode_id_from_index,
)
from dataset_contracts.stream_catalog import (
    NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1,
    NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1,
    NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1,
    NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
    NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1,
    NativeStreamContract,
    NativeStreamEpisodeManifest,
    NativeStreamEpisodeRange,
    NativeStreamExternalArtifactReceipt,
    NativeStreamExternalSampleIntent,
    NativeStreamSampleRecord,
)
from robot_motion.runtime.observation_cut import ObservationSampleRef


@dataclass(frozen=True)
class NativeStreamAppendReceipt:
    stream_id: str
    sample_id: str
    sample_semantic_content_hash: str
    index_relative_path: str
    payload_relative_path: str
    accepted: bool
    durable: bool
    idempotent: bool


@dataclass
class _StreamState:
    contract: NativeStreamContract
    directory: Path
    index_path: Path
    index_fd: int
    records: list[NativeStreamSampleRecord] = field(default_factory=list)
    record_by_sample_id: dict[str, NativeStreamSampleRecord] = field(
        default_factory=dict
    )
    external_intents: list[NativeStreamExternalSampleIntent] = field(
        default_factory=list
    )
    external_intent_by_sample_id: dict[
        str, NativeStreamExternalSampleIntent
    ] = field(default_factory=dict)
    dropped_count: int = 0
    gap_reasons: set[str] = field(default_factory=set)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(fd, payload[offset:])
        if count <= 0:
            raise OSError("Native stream write made no progress.")
        offset += count


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _publish_immutable(path: Path, payload: bytes) -> bool:
    """Publish bytes atomically, returning whether an identical file existed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable native stream artifact conflicts at {path}.")
        return True
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(temporary, flags, 0o600)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            os.close(fd)
        raise
    os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return False


class NativeStreamEpisodeWriter:
    """One append-only native-stream session subordinate to an episode owner."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        episode_uuid: str,
        legacy_episode_index: int,
        contracts: Mapping[str, NativeStreamContract],
        external_payload_stream_ids: Iterable[str] = (),
        chunk_size: int = 1000,
        fsync_on_append: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.episode_uuid = str(episode_uuid).strip()
        if not self.episode_uuid or "/" in self.episode_uuid:
            raise ValueError("episode_uuid is not safe for native stream storage.")
        self.legacy_episode_index = int(legacy_episode_index)
        if self.legacy_episode_index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        normalized = dict(contracts)
        if not normalized:
            raise ValueError("At least one native stream contract is required.")
        for stream_id, contract in normalized.items():
            if not isinstance(contract, NativeStreamContract):
                raise TypeError("contracts must contain NativeStreamContract values.")
            if str(stream_id) != contract.stream_id:
                raise ValueError("Contract mapping key must match contract.stream_id.")
        self.contracts = dict(sorted(normalized.items()))
        external_stream_ids = frozenset(
            str(stream_id) for stream_id in external_payload_stream_ids
        )
        unknown_external = external_stream_ids - set(self.contracts)
        if unknown_external:
            raise ValueError(
                "External payload streams are not declared in contracts: "
                f"{sorted(unknown_external)}."
            )
        self.external_payload_stream_ids = external_stream_ids
        self.fsync_on_append = bool(fsync_on_append)
        self._lock = threading.RLock()
        self._frozen = False
        self._sealed = False
        self._closed = False
        self._verified_external_artifacts: set[tuple[str, str, int]] = set()
        self._states: dict[str, _StreamState] = {}

        chunk_id = chunk_id_from_episode_index(
            self.legacy_episode_index,
            chunk_size=self.chunk_size,
        )
        episode_id = episode_id_from_index(self.legacy_episode_index)
        try:
            for stream_id, contract in self.contracts.items():
                contract_path = (
                    self.dataset_root
                    / NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1
                    / f"{stream_id}.cjson"
                )
                _publish_immutable(
                    contract_path,
                    canonical_bytes(
                        contract.to_payload(),
                        domain=NATIVE_STREAM_CONTRACT_FILE_DOMAIN_V1,
                    ),
                )
                directory = (
                    self.dataset_root
                    / "streams"
                    / stream_id
                    / chunk_id
                    / episode_id
                )
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "payloads").mkdir(parents=True, exist_ok=True)
                index_path = directory / "samples.cjsonl"
                flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                fd = os.open(index_path, flags, 0o600)
                if os.fstat(fd).st_size != 0:
                    os.close(fd)
                    raise ValueError(
                        "NativeStreamEpisodeWriter does not append to an "
                        "existing unsealed episode index."
                    )
                self._states[stream_id] = _StreamState(
                    contract=contract,
                    directory=directory,
                    index_path=index_path,
                    index_fd=fd,
                )
        except BaseException:
            for state in self._states.values():
                os.close(state.index_fd)
            self._closed = True
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Native stream episode writer is closed.")
        if self._sealed:
            raise RuntimeError("Native stream episode writer is sealed.")

    def _ensure_ingestion_open(self) -> None:
        self._ensure_open()
        if self._frozen:
            raise RuntimeError("Native stream episode ingestion is frozen.")

    @property
    def requires_post_finalize_seal(self) -> bool:
        return bool(self.external_payload_stream_ids)

    def append(
        self,
        *,
        stream_id: str,
        sample_id: str,
        sequence: int,
        measured_at: float,
        measured_clock_domain_id: str,
        available_at: float,
        available_clock_domain_id: str,
        source_payload_object_id: str,
        payload: Any,
        written_at: float,
        written_clock_domain_id: str,
        valid: bool = True,
        stale: bool = False,
        gap_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NativeStreamAppendReceipt:
        with self._lock:
            self._ensure_ingestion_open()
            normalized_stream_id = str(stream_id)
            state = self._states.get(normalized_stream_id)
            if state is None:
                raise KeyError(f"Unknown native stream {normalized_stream_id!r}.")
            if normalized_stream_id in self.external_payload_stream_ids:
                raise ValueError(
                    "External payload streams must use stage_external_sample()."
                )
            record = NativeStreamSampleRecord.create(
                contract=state.contract,
                sample_id=sample_id,
                sequence=sequence,
                measured_at=measured_at,
                measured_clock_domain_id=measured_clock_domain_id,
                available_at=available_at,
                available_clock_domain_id=available_clock_domain_id,
                source_payload_object_id=source_payload_object_id,
                payload=payload,
                valid=valid,
                stale=stale,
                gap_reason=gap_reason,
                written_at=written_at,
                written_clock_domain_id=written_clock_domain_id,
                metadata=metadata,
            )
            existing = state.record_by_sample_id.get(record.sample_id)
            payload_path = (
                state.directory
                / "payloads"
                / f"{record.stored_payload_hash}.cjson"
            )
            if existing is not None:
                same_source_sample = (
                    existing.stream_contract_hash
                    == record.stream_contract_hash
                    and existing.sequence == record.sequence
                    and existing.measured_at == record.measured_at
                    and existing.measured_clock_domain_id
                    == record.measured_clock_domain_id
                    and existing.source_payload_object_id
                    == record.source_payload_object_id
                    and existing.stored_payload_hash
                    == record.stored_payload_hash
                    and existing.valid == record.valid
                    and existing.stale == record.stale
                    and existing.gap_reason == record.gap_reason
                )
                if not same_source_sample:
                    raise ValueError(
                        "Native stream sample_id was reused with different content."
                    )
                return NativeStreamAppendReceipt(
                    stream_id=record.stream_id,
                    sample_id=record.sample_id,
                    sample_semantic_content_hash=(
                        existing.semantic_content_hash
                    ),
                    index_relative_path=state.index_path.relative_to(
                        self.dataset_root
                    ).as_posix(),
                    payload_relative_path=payload_path.relative_to(
                        self.dataset_root
                    ).as_posix(),
                    accepted=True,
                    durable=True,
                    idempotent=True,
                )
            if state.records and record.sequence <= state.records[-1].sequence:
                raise ValueError(
                    "Native stream sequence must strictly increase per stream."
                )
            if state.records and record.sequence > state.records[-1].sequence + 1:
                state.gap_reasons.add("sequence_gap")
            if record.gap_reason is not None:
                state.gap_reasons.add(record.gap_reason)

            payload_encoded = canonical_bytes(
                payload,
                domain=NATIVE_STREAM_PAYLOAD_DOMAIN_V1,
            )
            _publish_immutable(payload_path, payload_encoded)
            record_encoded = canonical_bytes(
                record.to_payload(),
                domain=NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1,
            )
            _write_all(state.index_fd, record_encoded + b"\n")
            if self.fsync_on_append:
                os.fsync(state.index_fd)
            state.records.append(record)
            state.record_by_sample_id[record.sample_id] = record
            return NativeStreamAppendReceipt(
                stream_id=record.stream_id,
                sample_id=record.sample_id,
                sample_semantic_content_hash=record.semantic_content_hash,
                index_relative_path=state.index_path.relative_to(
                    self.dataset_root
                ).as_posix(),
                payload_relative_path=payload_path.relative_to(
                    self.dataset_root
                ).as_posix(),
                accepted=True,
                durable=self.fsync_on_append,
                idempotent=False,
            )

    def append_observation_sample(
        self,
        sample: ObservationSampleRef,
        payload: Any,
        *,
        written_at: float,
        written_clock_domain_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> NativeStreamAppendReceipt:
        if not isinstance(sample, ObservationSampleRef):
            raise TypeError("sample must be an ObservationSampleRef.")
        return self.append(
            stream_id=sample.stream_id,
            sample_id=sample.sample_id,
            sequence=sample.sequence,
            measured_at=sample.measured_at,
            measured_clock_domain_id=sample.measured_clock_domain_id,
            available_at=sample.available_at,
            available_clock_domain_id=sample.available_clock_domain_id,
            source_payload_object_id=sample.payload_object_id,
            payload=payload,
            valid=sample.valid,
            stale=sample.stale,
            gap_reason=None,
            written_at=written_at,
            written_clock_domain_id=written_clock_domain_id,
            metadata={
                **dict(sample.content_metadata),
                **dict(metadata or {}),
            },
        )

    def note_drop(self, stream_id: str, *, reason: str) -> None:
        with self._lock:
            self._ensure_ingestion_open()
            state = self._states.get(str(stream_id))
            if state is None:
                raise KeyError(f"Unknown native stream {stream_id!r}.")
            normalized_reason = str(reason).strip()
            if not normalized_reason:
                raise ValueError("Native stream drop reason must be non-empty.")
            state.dropped_count += 1
            state.gap_reasons.add(normalized_reason)

    def stage_external_sample(
        self,
        intent: NativeStreamExternalSampleIntent,
    ) -> NativeStreamExternalSampleIntent:
        """Register immutable capture-time identity before legacy finalization."""

        if not isinstance(intent, NativeStreamExternalSampleIntent):
            raise TypeError("intent must be NativeStreamExternalSampleIntent.")
        with self._lock:
            self._ensure_ingestion_open()
            if intent.stream_id not in self.external_payload_stream_ids:
                raise ValueError(
                    "External sample stream is not configured for external payloads."
                )
            state = self._states[intent.stream_id]
            if intent.stream_contract_hash != state.contract.semantic_content_hash:
                raise ValueError(
                    "External sample intent does not match its stream contract."
                )
            existing = state.external_intent_by_sample_id.get(intent.sample_id)
            if existing is not None:
                if existing.semantic_content_hash != intent.semantic_content_hash:
                    raise ValueError(
                        "External sample_id was reused with different content."
                    )
                return existing
            if (
                state.external_intents
                and intent.sequence <= state.external_intents[-1].sequence
            ):
                raise ValueError(
                    "External stream sequence must strictly increase per stream."
                )
            if (
                state.external_intents
                and intent.sequence > state.external_intents[-1].sequence + 1
            ):
                state.gap_reasons.add("sequence_gap")
            if intent.gap_reason is not None:
                state.gap_reasons.add(intent.gap_reason)
            state.external_intents.append(intent)
            state.external_intent_by_sample_id[intent.sample_id] = intent
            return intent

    def freeze_ingestion(self) -> None:
        """Stop new samples while keeping indexes open for artifact receipts."""

        with self._lock:
            self._ensure_open()
            if self._frozen:
                return
            for state in self._states.values():
                os.fsync(state.index_fd)
            self._frozen = True

    def bind_external_artifact(
        self,
        receipt: NativeStreamExternalArtifactReceipt,
    ) -> NativeStreamAppendReceipt:
        """Verify and append one durable artifact receipt after freeze."""

        if not isinstance(receipt, NativeStreamExternalArtifactReceipt):
            raise TypeError(
                "receipt must be NativeStreamExternalArtifactReceipt."
            )
        with self._lock:
            self._ensure_open()
            if not self._frozen:
                raise RuntimeError(
                    "External artifacts may only be bound after ingestion freeze."
                )
            if (
                receipt.episode_uuid != self.episode_uuid
                or receipt.legacy_episode_index != self.legacy_episode_index
            ):
                raise ValueError(
                    "External artifact receipt belongs to another episode."
                )
            if receipt.stream_id not in self.external_payload_stream_ids:
                raise ValueError(
                    "External artifact stream is not configured for external payloads."
                )
            state = self._states[receipt.stream_id]
            intent = state.external_intent_by_sample_id.get(receipt.sample_id)
            if intent is None:
                raise ValueError(
                    "External artifact receipt has no staged sample intent."
                )
            existing = state.record_by_sample_id.get(receipt.sample_id)
            if existing is not None:
                if (
                    existing.external_artifact_receipt_hash
                    != receipt.semantic_content_hash
                ):
                    raise ValueError(
                        "External sample receipt changed after admission."
                    )
                return NativeStreamAppendReceipt(
                    stream_id=existing.stream_id,
                    sample_id=existing.sample_id,
                    sample_semantic_content_hash=(
                        existing.semantic_content_hash
                    ),
                    index_relative_path=state.index_path.relative_to(
                        self.dataset_root
                    ).as_posix(),
                    payload_relative_path=receipt.artifact_relative_path,
                    accepted=True,
                    durable=True,
                    idempotent=True,
                )
            expected_index = len(state.records)
            if (
                expected_index >= len(state.external_intents)
                or state.external_intents[expected_index].sample_id
                != receipt.sample_id
            ):
                raise ValueError(
                    "External artifact receipts must follow staged stream order."
                )

            artifact_path = self.dataset_root / receipt.artifact_relative_path
            try:
                artifact_path.resolve().relative_to(
                    self.dataset_root.resolve()
                )
            except ValueError as exc:
                raise ValueError(
                    "External artifact escapes the dataset root."
                ) from exc
            if not artifact_path.is_file():
                raise ValueError(
                    f"External artifact does not exist: {artifact_path}"
                )
            artifact_stat = artifact_path.stat()
            if artifact_stat.st_size != receipt.artifact_byte_count:
                raise ValueError("External artifact byte count mismatch.")
            cache_key = (
                receipt.artifact_relative_path,
                receipt.artifact_sha256,
                receipt.artifact_byte_count,
            )
            if cache_key not in self._verified_external_artifacts:
                if _file_sha256(artifact_path) != receipt.artifact_sha256:
                    raise ValueError("External artifact SHA-256 mismatch.")
                self._verified_external_artifacts.add(cache_key)

            record = NativeStreamSampleRecord.from_external(
                contract=state.contract,
                intent=intent,
                receipt=receipt,
            )
            record_encoded = canonical_bytes(
                record.to_payload(),
                domain=NATIVE_STREAM_SAMPLE_FILE_DOMAIN_V1,
            )
            _write_all(state.index_fd, record_encoded + b"\n")
            os.fsync(state.index_fd)
            state.records.append(record)
            state.record_by_sample_id[record.sample_id] = record
            return NativeStreamAppendReceipt(
                stream_id=record.stream_id,
                sample_id=record.sample_id,
                sample_semantic_content_hash=record.semantic_content_hash,
                index_relative_path=state.index_path.relative_to(
                    self.dataset_root
                ).as_posix(),
                payload_relative_path=receipt.artifact_relative_path,
                accepted=True,
                durable=True,
                idempotent=False,
            )

    def seal_after_legacy_finalize(
        self,
        receipts: Iterable[NativeStreamExternalArtifactReceipt],
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest:
        """Bind complete external receipts, then publish the native manifest."""

        supplied = tuple(receipts)
        if any(
            not isinstance(receipt, NativeStreamExternalArtifactReceipt)
            for receipt in supplied
        ):
            raise TypeError(
                "receipts must contain NativeStreamExternalArtifactReceipt values."
            )
        with self._lock:
            self._ensure_open()
            if not self._frozen:
                raise RuntimeError(
                    "Native stream ingestion must be frozen before finalize."
                )
            expected = {
                (intent.stream_id, intent.sample_id)
                for state in self._states.values()
                for intent in state.external_intents
            }
            provided = {
                (receipt.stream_id, receipt.sample_id)
                for receipt in supplied
            }
            if len(provided) != len(supplied):
                raise ValueError("External artifact receipts contain duplicates.")
            if provided != expected:
                missing = sorted(expected - provided)
                extra = sorted(provided - expected)
                raise ValueError(
                    "External artifact receipt coverage mismatch: "
                    f"missing={missing}, extra={extra}."
                )
            receipt_by_identity = {
                (receipt.stream_id, receipt.sample_id): receipt
                for receipt in supplied
            }
            for state in self._states.values():
                for intent in state.external_intents:
                    self.bind_external_artifact(
                        receipt_by_identity[(intent.stream_id, intent.sample_id)]
                    )
            return self.seal(
                sealed_at=sealed_at,
                sealed_clock_domain_id=sealed_clock_domain_id,
            )

    def seal(
        self,
        *,
        sealed_at: float,
        sealed_clock_domain_id: str,
    ) -> NativeStreamEpisodeManifest:
        with self._lock:
            self._ensure_open()
            if self.external_payload_stream_ids and not self._frozen:
                raise RuntimeError(
                    "External native streams require freeze_ingestion() before seal."
                )
            if not self._frozen:
                self.freeze_ingestion()
            missing_external = [
                (intent.stream_id, intent.sample_id)
                for state in self._states.values()
                for intent in state.external_intents
                if intent.sample_id not in state.record_by_sample_id
            ]
            if missing_external:
                raise RuntimeError(
                    "External native stream samples lack durable artifact "
                    f"receipts: {missing_external}."
                )
            ranges: list[NativeStreamEpisodeRange] = []
            for state in self._states.values():
                os.fsync(state.index_fd)
                raw_index = state.index_path.read_bytes()
                records = state.records
                ranges.append(
                    NativeStreamEpisodeRange(
                        stream_id=state.contract.stream_id,
                        stream_contract_hash=(
                            state.contract.semantic_content_hash
                        ),
                        index_relative_path=state.index_path.relative_to(
                            self.dataset_root
                        ).as_posix(),
                        index_sha256=hashlib.sha256(raw_index).hexdigest(),
                        sample_count=len(records),
                        first_sequence=(
                            None if not records else records[0].sequence
                        ),
                        last_sequence=(
                            None if not records else records[-1].sequence
                        ),
                        first_measured_at=(
                            None if not records else records[0].measured_at
                        ),
                        last_measured_at=(
                            None if not records else records[-1].measured_at
                        ),
                        first_available_at=(
                            None if not records else records[0].available_at
                        ),
                        last_available_at=(
                            None if not records else records[-1].available_at
                        ),
                        dropped_count=state.dropped_count,
                        gap_reasons=tuple(state.gap_reasons),
                    )
                )
            manifest = NativeStreamEpisodeManifest(
                episode_uuid=self.episode_uuid,
                legacy_episode_index=self.legacy_episode_index,
                stream_ranges=tuple(ranges),
                sealed_at=sealed_at,
                sealed_clock_domain_id=sealed_clock_domain_id,
            )
            manifest_path = (
                self.dataset_root
                / NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1
                / self.episode_uuid
                / "manifest.cjson"
            )
            _publish_immutable(
                manifest_path,
                canonical_bytes(
                    manifest.to_payload(),
                    domain=NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
                ),
            )
            for state in self._states.values():
                os.close(state.index_fd)
            self._sealed = True
            self._closed = True
            return manifest

    def close_unsealed(self) -> None:
        """Close file descriptors without publishing a sealed manifest."""

        with self._lock:
            if self._closed:
                return
            for state in self._states.values():
                try:
                    os.close(state.index_fd)
                except OSError:
                    pass
            self._closed = True


__all__ = [
    "NATIVE_STREAM_CONTRACT_RELATIVE_ROOT_V1",
    "NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1",
    "NativeStreamAppendReceipt",
    "NativeStreamEpisodeWriter",
]

# Copyright (c) 2026 DexTeleop Contributors
"""Disk-primary, source-neutral history for low-rate planning observations.

The source owner supplies already-read immutable JSON projections.  This module
owns only append/query/retention mechanics; it never reads a robot or simulator,
advances a source, starts itself, or interprets payload semantics.  One optional
recorder can be started explicitly by an application that already owns the
source lifecycle.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from typing_extensions import Self

from dataset_contracts.canonical_encoding import canonical_content_hash

PLANNING_HISTORY_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.planning_history.source_binding.v1"
)
PLANNING_HISTORY_RECORD_HASH_DOMAIN_V1 = "dexteleop.planning_history.record.v1"
PLANNING_HISTORY_SLICE_HASH_DOMAIN_V1 = "dexteleop.planning_history.slice.v1"
PLANNING_HISTORY_STORE_SCHEMA_VERSION_V1 = 1
PLANNING_HISTORY_SAMPLING_POLICIES_V1 = (
    "latest",
    "uniform",
    "boundary_only",
    "boundary_plus_uniform",
)


class PlanningHistoryStoreError(RuntimeError):
    """A durable history operation failed or violated its stream contract."""


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _unique_texts(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    parsed = tuple(_required(item, field_name=f"{field_name} item") for item in values)
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    return tuple(sorted(parsed))


def _json_object(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} must be finite JSON data.") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{field_name} must encode a JSON object.")
    return _freeze_json(decoded)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class PlanningHistorySourceBinding:
    """Immutable identity of one lower-owned low-rate projection stream."""

    source_id: str
    runtime_session_id: str
    clock_domain_id: str
    channels: tuple[str, ...]
    capture_profile_hash: str
    source_contract: Mapping[str, Any]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "runtime_session_id",
            "clock_domain_id",
            "capture_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "channels",
            _unique_texts(self.channels, field_name="channels"),
        )
        if not self.channels:
            raise ValueError("A planning history source requires at least one channel.")
        object.__setattr__(
            self,
            "source_contract",
            _json_object(self.source_contract, field_name="source_contract"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(),
                domain=PLANNING_HISTORY_SOURCE_BINDING_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "runtime_session_id": self.runtime_session_id,
            "clock_domain_id": self.clock_domain_id,
            "channels": self.channels,
            "capture_profile_hash": self.capture_profile_hash,
            "source_contract": _thaw_json(self.source_contract),
            "store_schema_version": PLANNING_HISTORY_STORE_SCHEMA_VERSION_V1,
        }


@dataclass(frozen=True)
class PlanningHistoryRecord:
    """One already-produced causal projection accepted by the disk store."""

    record_id: str
    source_binding_hash: str
    runtime_session_id: str
    channel: str
    measured_at: float
    available_at: float
    cutoff_at: float
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    boundary_event: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "source_binding_hash",
            "runtime_session_id",
            "channel",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        measured_at = _finite(self.measured_at, field_name="measured_at")
        available_at = _finite(self.available_at, field_name="available_at")
        cutoff_at = _finite(self.cutoff_at, field_name="cutoff_at")
        if measured_at > available_at or available_at > cutoff_at:
            raise ValueError(
                "Planning history requires measured_at <= available_at <= cutoff_at."
            )
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "cutoff_at", cutoff_at)
        object.__setattr__(
            self,
            "payload",
            _json_object(self.payload, field_name="payload"),
        )
        refs = _unique_texts(self.evidence_refs, field_name="evidence_refs")
        if not refs:
            raise ValueError("A planning history record requires evidence refs.")
        object.__setattr__(self, "evidence_refs", refs)
        if self.boundary_event is not None:
            object.__setattr__(
                self,
                "boundary_event",
                _required(self.boundary_event, field_name="boundary_event"),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(),
                domain=PLANNING_HISTORY_RECORD_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_binding_hash": self.source_binding_hash,
            "runtime_session_id": self.runtime_session_id,
            "channel": self.channel,
            "measured_at": self.measured_at,
            "available_at": self.available_at,
            "cutoff_at": self.cutoff_at,
            "payload": _thaw_json(self.payload),
            "evidence_refs": self.evidence_refs,
            "boundary_event": self.boundary_event,
        }


@dataclass(frozen=True)
class StoredPlanningHistoryRecord:
    sequence: int
    record: PlanningHistoryRecord

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, field_name="sequence"),
        )
        if not isinstance(self.record, PlanningHistoryRecord):
            raise TypeError("record must be PlanningHistoryRecord.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "record": self.record.to_payload(),
            "record_hash": self.record.semantic_content_hash,
        }


@dataclass(frozen=True)
class PlanningHistorySlice:
    """One bounded immutable query result, independent of storage layout."""

    slice_id: str
    source_id: str
    source_binding_hash: str
    runtime_session_id: str
    clock_domain_id: str
    channels: tuple[str, ...]
    window_begin_at: float
    window_end_at: float
    sampling_policy: str
    records: tuple[StoredPlanningHistoryRecord, ...]
    truncated: bool
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "slice_id",
            "source_id",
            "source_binding_hash",
            "runtime_session_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "channels",
            _unique_texts(self.channels, field_name="channels"),
        )
        begin = _finite(self.window_begin_at, field_name="window_begin_at")
        end = _finite(self.window_end_at, field_name="window_end_at")
        if begin > end:
            raise ValueError("Planning history window cannot be negative.")
        object.__setattr__(self, "window_begin_at", begin)
        object.__setattr__(self, "window_end_at", end)
        policy = _required(self.sampling_policy, field_name="sampling_policy")
        if policy not in PLANNING_HISTORY_SAMPLING_POLICIES_V1:
            raise ValueError(f"Unsupported sampling_policy: {policy!r}.")
        records = tuple(self.records)
        if any(not isinstance(item, StoredPlanningHistoryRecord) for item in records):
            raise TypeError("records must contain StoredPlanningHistoryRecord values.")
        sequences = tuple(item.sequence for item in records)
        if tuple(sorted(sequences)) != sequences or len(set(sequences)) != len(
            sequences
        ):
            raise ValueError("Planning history slice sequences must strictly increase.")
        for item in records:
            record = item.record
            if (
                record.source_binding_hash != self.source_binding_hash
                or record.runtime_session_id != self.runtime_session_id
                or record.channel not in self.channels
                or record.measured_at < begin
                or record.measured_at > end
                or record.available_at > end
                or record.cutoff_at > end
            ):
                raise ValueError(
                    "Planning history slice contains an incompatible record."
                )
        object.__setattr__(self, "records", records)
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be boolean.")
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(),
                domain=PLANNING_HISTORY_SLICE_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "source_id": self.source_id,
            "source_binding_hash": self.source_binding_hash,
            "runtime_session_id": self.runtime_session_id,
            "clock_domain_id": self.clock_domain_id,
            "channels": self.channels,
            "window_begin_at": self.window_begin_at,
            "window_end_at": self.window_end_at,
            "sampling_policy": self.sampling_policy,
            "records": tuple(item.to_payload() for item in self.records),
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class PlanningHistoryStoreLimits:
    retention_sec: float
    max_records: int
    max_record_payload_bytes: int
    memory_buffer_max_records: int = 64

    def __post_init__(self) -> None:
        retention = _finite(self.retention_sec, field_name="retention_sec")
        if retention <= 0.0:
            raise ValueError("retention_sec must be positive.")
        object.__setattr__(self, "retention_sec", retention)
        for field_name in (
            "max_records",
            "max_record_payload_bytes",
            "memory_buffer_max_records",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True)
class PlanningHistoryAppendResult:
    sequence: int
    accepted: bool
    durable: bool
    idempotent: bool


@dataclass(frozen=True)
class PlanningHistoryGcResult:
    deleted_records: int
    retained_records: int


@dataclass(frozen=True)
class _StoredRow:
    sequence: int
    record: PlanningHistoryRecord
    payload_bytes: int


class SQLitePlanningHistoryStore:
    """One disk-primary history store for one immutable source binding."""

    def __init__(
        self,
        *,
        root: Path,
        binding: PlanningHistorySourceBinding,
        limits: PlanningHistoryStoreLimits,
        filename: str = "planning_history.sqlite3",
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be pathlib.Path.")
        if not root.is_absolute():
            raise ValueError("Planning history root must be absolute.")
        if not isinstance(binding, PlanningHistorySourceBinding):
            raise TypeError("binding must be PlanningHistorySourceBinding.")
        if not isinstance(limits, PlanningHistoryStoreLimits):
            raise TypeError("limits must be PlanningHistoryStoreLimits.")
        name = _required(filename, field_name="filename")
        if Path(name).name != name:
            raise ValueError("filename must be a basename.")
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / name
        self.binding = binding
        self.limits = limits
        self._lock = threading.RLock()
        self._recent: deque[int] = deque(maxlen=limits.memory_buffer_max_records)
        self._connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._closed = False
        self._create_schema()
        self._bind_source()

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def buffered_record_count(self) -> int:
        with self._lock:
            return len(self._recent)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def append(self, record: PlanningHistoryRecord) -> PlanningHistoryAppendResult:
        if not isinstance(record, PlanningHistoryRecord):
            raise TypeError("record must be PlanningHistoryRecord.")
        if (
            record.source_binding_hash != self.binding.semantic_content_hash
            or record.runtime_session_id != self.binding.runtime_session_id
            or record.channel not in self.binding.channels
        ):
            raise PlanningHistoryStoreError(
                "Record does not match the immutable history source binding."
            )
        payload_json = _compact_json(record.payload)
        evidence_json = _compact_json(record.evidence_refs)
        payload_bytes = len(payload_json.encode("utf-8")) + len(
            evidence_json.encode("utf-8")
        )
        if payload_bytes > self.limits.max_record_payload_bytes:
            raise PlanningHistoryStoreError("Record exceeds max_record_payload_bytes.")
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT sequence, record_hash
                FROM planning_history_records
                WHERE record_id = ?
                """,
                (record.record_id,),
            ).fetchone()
            if existing is not None:
                if existing[1] != record.semantic_content_hash:
                    raise PlanningHistoryStoreError(
                        f"History record ID conflict: {record.record_id!r}."
                    )
                return PlanningHistoryAppendResult(
                    sequence=int(existing[0]),
                    accepted=False,
                    durable=True,
                    idempotent=True,
                )
            previous = self._connection.execute(
                """
                SELECT measured_at, available_at, cutoff_at
                FROM planning_history_records
                ORDER BY sequence DESC
                LIMIT 1
                """
            ).fetchone()
            if previous is not None and (
                record.measured_at < float(previous[0])
                or record.available_at < float(previous[1])
                or record.cutoff_at < float(previous[2])
            ):
                raise PlanningHistoryStoreError(
                    "History record clocks must be monotonic within a source."
                )
            cursor = self._connection.execute(
                """
                INSERT INTO planning_history_records (
                    record_id, record_hash, source_binding_hash,
                    runtime_session_id, channel, measured_at, available_at,
                    cutoff_at, payload_json, evidence_refs_json,
                    boundary_event, payload_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.semantic_content_hash,
                    record.source_binding_hash,
                    record.runtime_session_id,
                    record.channel,
                    record.measured_at,
                    record.available_at,
                    record.cutoff_at,
                    payload_json,
                    evidence_json,
                    record.boundary_event,
                    payload_bytes,
                ),
            )
            sequence = int(cursor.lastrowid)
            self._recent.append(sequence)
            self._gc_locked(now=record.cutoff_at)
        return PlanningHistoryAppendResult(
            sequence=sequence,
            accepted=True,
            durable=True,
            idempotent=False,
        )

    def query(
        self,
        *,
        slice_id: str,
        channels: tuple[str, ...],
        cutoff_at: float,
        history_window_sec: float,
        max_samples: int,
        max_payload_bytes: int,
        sampling_policy: str,
    ) -> PlanningHistorySlice:
        query_channels = _unique_texts(channels, field_name="channels")
        if not query_channels or set(query_channels) - set(self.binding.channels):
            raise PlanningHistoryStoreError(
                "Query channels must be a non-empty source-binding subset."
            )
        cutoff = _finite(cutoff_at, field_name="cutoff_at")
        window = _finite(history_window_sec, field_name="history_window_sec")
        if window <= 0.0 or window > self.limits.retention_sec:
            raise PlanningHistoryStoreError(
                "history_window_sec must be within configured retention."
            )
        sample_limit = _positive_int(max_samples, field_name="max_samples")
        payload_limit = _positive_int(
            max_payload_bytes,
            field_name="max_payload_bytes",
        )
        policy = _required(sampling_policy, field_name="sampling_policy")
        if policy not in PLANNING_HISTORY_SAMPLING_POLICIES_V1:
            raise PlanningHistoryStoreError(f"Unsupported sampling_policy: {policy!r}.")
        begin = cutoff - window
        placeholders = ",".join("?" for _ in query_channels)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT sequence, record_id, record_hash, source_binding_hash,
                       runtime_session_id, channel, measured_at, available_at,
                       cutoff_at, payload_json, evidence_refs_json,
                       boundary_event, payload_bytes
                FROM planning_history_records
                WHERE channel IN ({placeholders})
                  AND measured_at >= ? AND measured_at <= ?
                  AND available_at <= ? AND cutoff_at <= ?
                ORDER BY sequence ASC
                """,
                (*query_channels, begin, cutoff, cutoff, cutoff),
            ).fetchall()
        candidates = tuple(self._row_from_sql(row) for row in rows)
        selected, truncated = _sample_rows(
            candidates,
            max_samples=sample_limit,
            sampling_policy=policy,
        )
        selected_list = list(selected)
        while selected_list and _rows_wire_bytes(selected_list) > payload_limit:
            selected_list.pop(0)
            truncated = True
        return PlanningHistorySlice(
            slice_id=_required(slice_id, field_name="slice_id"),
            source_id=self.binding.source_id,
            source_binding_hash=self.binding.semantic_content_hash,
            runtime_session_id=self.binding.runtime_session_id,
            clock_domain_id=self.binding.clock_domain_id,
            channels=query_channels,
            window_begin_at=begin,
            window_end_at=cutoff,
            sampling_policy=policy,
            records=tuple(
                StoredPlanningHistoryRecord(item.sequence, item.record)
                for item in selected_list
            ),
            truncated=truncated,
        )

    def gc(self, *, now: float) -> PlanningHistoryGcResult:
        timestamp = _finite(now, field_name="now")
        with self._lock, self._connection:
            return self._gc_locked(now=timestamp)

    def _gc_locked(self, *, now: float) -> PlanningHistoryGcResult:
        cutoff = now - self.limits.retention_sec
        before = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM planning_history_records"
            ).fetchone()[0]
        )
        self._connection.execute(
            "DELETE FROM planning_history_records WHERE measured_at < ?",
            (cutoff,),
        )
        after_time = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM planning_history_records"
            ).fetchone()[0]
        )
        overflow = max(0, after_time - self.limits.max_records)
        if overflow:
            self._connection.execute(
                """
                DELETE FROM planning_history_records
                WHERE sequence IN (
                    SELECT sequence FROM planning_history_records
                    ORDER BY sequence ASC LIMIT ?
                )
                """,
                (overflow,),
            )
        retained = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM planning_history_records"
            ).fetchone()[0]
        )
        return PlanningHistoryGcResult(
            deleted_records=before - retained,
            retained_records=retained,
        )

    def _row_from_sql(self, row: tuple[Any, ...]) -> _StoredRow:
        payload = json.loads(row[9])
        evidence_refs = tuple(json.loads(row[10]))
        record = PlanningHistoryRecord(
            record_id=row[1],
            source_binding_hash=row[3],
            runtime_session_id=row[4],
            channel=row[5],
            measured_at=row[6],
            available_at=row[7],
            cutoff_at=row[8],
            payload=payload,
            evidence_refs=evidence_refs,
            boundary_event=row[11],
        )
        if record.semantic_content_hash != row[2]:
            raise PlanningHistoryStoreError(
                f"Persisted history record hash mismatch: {record.record_id!r}."
            )
        return _StoredRow(int(row[0]), record, int(row[12]))

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS planning_history_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS planning_history_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    record_hash TEXT NOT NULL,
                    source_binding_hash TEXT NOT NULL,
                    runtime_session_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    measured_at REAL NOT NULL,
                    available_at REAL NOT NULL,
                    cutoff_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    boundary_event TEXT,
                    payload_bytes INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS planning_history_query_idx
                ON planning_history_records (
                    runtime_session_id, channel, measured_at, available_at
                );
                """
            )

    def _bind_source(self) -> None:
        expected = {
            "store_schema_version": str(PLANNING_HISTORY_STORE_SCHEMA_VERSION_V1),
            "source_binding_hash": self.binding.semantic_content_hash,
            "source_binding_payload": _compact_json(self.binding.to_payload()),
        }
        with self._connection:
            existing = dict(
                self._connection.execute(
                    "SELECT key, value FROM planning_history_metadata"
                ).fetchall()
            )
            if existing and existing != expected:
                raise PlanningHistoryStoreError(
                    "Existing planning history database uses another source binding."
                )
            for key, value in expected.items():
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO planning_history_metadata (key, value)
                    VALUES (?, ?)
                    """,
                    (key, value),
                )


def _uniform_indexes(length: int, count: int) -> tuple[int, ...]:
    if count >= length:
        return tuple(range(length))
    if count == 1:
        return (length - 1,)
    return tuple(
        sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})
    )


def _sample_rows(
    rows: tuple[_StoredRow, ...],
    *,
    max_samples: int,
    sampling_policy: str,
) -> tuple[tuple[_StoredRow, ...], bool]:
    if sampling_policy == "latest":
        selected = rows[-1:] if rows else ()
    elif sampling_policy == "boundary_only":
        boundaries = tuple(
            item for item in rows if item.record.boundary_event is not None
        )
        selected = boundaries[-max_samples:]
    elif sampling_policy == "uniform":
        selected = tuple(
            rows[index] for index in _uniform_indexes(len(rows), max_samples)
        )
    else:
        boundaries = tuple(
            item for item in rows if item.record.boundary_event is not None
        )
        if len(boundaries) >= max_samples:
            selected = boundaries[-max_samples:]
        else:
            boundary_sequences = {item.sequence for item in boundaries}
            ordinary = tuple(
                item for item in rows if item.sequence not in boundary_sequences
            )
            remaining = max_samples - len(boundaries)
            sampled = tuple(
                ordinary[index] for index in _uniform_indexes(len(ordinary), remaining)
            )
            selected = tuple(
                sorted((*boundaries, *sampled), key=lambda item: item.sequence)
            )
    return selected, len(selected) < len(rows)


def _rows_wire_bytes(rows: Sequence[_StoredRow]) -> int:
    return len(
        _compact_json(
            [
                StoredPlanningHistoryRecord(item.sequence, item.record).to_payload()
                for item in rows
            ]
        ).encode("utf-8")
    )


class PlanningHistoryRecorder:
    """Explicitly-started periodic caller around one application-owned capture."""

    def __init__(
        self,
        *,
        store: SQLitePlanningHistoryStore,
        capture: Callable[[], PlanningHistoryRecord | None],
        interval_sec: float,
    ) -> None:
        if not isinstance(store, SQLitePlanningHistoryStore):
            raise TypeError("store must be SQLitePlanningHistoryStore.")
        if not callable(capture):
            raise TypeError("capture must be callable.")
        interval = _finite(interval_sec, field_name="interval_sec")
        if interval <= 0.0:
            raise ValueError("interval_sec must be positive.")
        self.store = store
        self._capture = capture
        self.interval_sec = interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: Exception | None = None

    @property
    def started(self) -> bool:
        with self._lock:
            return self._thread is not None

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def capture_once(self) -> PlanningHistoryAppendResult | None:
        record = self._capture()
        if record is None:
            return None
        if not isinstance(record, PlanningHistoryRecord):
            raise TypeError("capture must return PlanningHistoryRecord or None.")
        return self.store.append(record)

    def start(self) -> None:
        """Start periodic capture; construction itself is always default-off."""

        with self._lock:
            if self._thread is not None:
                raise PlanningHistoryStoreError("History recorder is already started.")
            self._stop.clear()
            self._last_error = None
            thread = threading.Thread(
                target=self._run,
                name=f"planning-history:{self.store.binding.source_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def close(self, *, timeout_sec: float = 5.0) -> bool:
        timeout = _finite(timeout_sec, field_name="timeout_sec")
        if timeout < 0.0:
            raise ValueError("timeout_sec cannot be negative.")
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is None:
            return True
        thread.join(timeout=timeout)
        stopped = not thread.is_alive()
        if stopped:
            with self._lock:
                self._thread = None
        return stopped

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.capture_once()
                self._stop.wait(self.interval_sec)
        except Exception as error:  # noqa: BLE001 - fail closed and expose cause.
            with self._lock:
                self._last_error = error
            self._stop.set()


__all__ = [
    "PLANNING_HISTORY_RECORD_HASH_DOMAIN_V1",
    "PLANNING_HISTORY_SAMPLING_POLICIES_V1",
    "PLANNING_HISTORY_SLICE_HASH_DOMAIN_V1",
    "PLANNING_HISTORY_SOURCE_BINDING_HASH_DOMAIN_V1",
    "PLANNING_HISTORY_STORE_SCHEMA_VERSION_V1",
    "PlanningHistoryAppendResult",
    "PlanningHistoryGcResult",
    "PlanningHistoryRecord",
    "PlanningHistoryRecorder",
    "PlanningHistorySlice",
    "PlanningHistorySourceBinding",
    "PlanningHistoryStoreError",
    "PlanningHistoryStoreLimits",
    "SQLitePlanningHistoryStore",
    "StoredPlanningHistoryRecord",
]

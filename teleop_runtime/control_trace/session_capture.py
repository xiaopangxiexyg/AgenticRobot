"""Lower-owned durable ControlTrace session capture lifecycle.

This module composes the existing journal, spool, writer-health, and episode
seal/commit primitives.  Application layers may coordinate it through its
public methods, but do not own or reconstruct persistence state.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.completeness import TRACE_REQUIREMENT_PROFILES_V1
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)

from .episode import EpisodeTraceRecorderSink
from .episode_manifest import EpisodeCaptureManifestStore
from .health_ledger import ControlTraceWriterHealthLedger
from .lifecycle import RuntimeTraceSession
from .runtime import ControlTraceRuntime
from .sink import ControlTraceSpoolSink
from .spool import ControlTraceSpool
from .worker import ControlTraceSpoolWorker


CONTROL_TRACE_SESSION_CAPTURE_RUNTIME_CAPACITY_V1 = 524_288
# Preserve the established persisted hash domains while moving their owner.
CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.station_control_trace_terminal_receipt.v1"
)
CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.station_control_trace_finalization.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@runtime_checkable
class ControlTraceEpisodeCapturePort(Protocol):
    """Application-facing lifecycle without exposing persistence internals."""

    @property
    def runtime_session_id(self) -> str: ...

    @property
    def control_domain_id(self) -> str: ...

    @property
    def clock_domain_id(self) -> str: ...

    def prepare(self, *, now: float) -> dict[str, object]: ...

    def finalize(self, *, now: float) -> dict[str, object]: ...

    def abort(self, *, now: float) -> None: ...

    def rotate_runtime_session(self, *, now: float) -> dict[str, object]: ...

    def status(self) -> dict[str, object]: ...

    def close(self, *, now: float | None = None) -> None: ...


@dataclass(frozen=True)
class ControlTraceSessionFinalizationReceipt:
    """Durable lower-owned references after episode commit and runtime close.

    The runtime terminal must be admitted after the formal episode is unbound:
    the episode seal intentionally excludes ``runtime_trace_terminal`` from its
    range.  This receipt therefore joins, but never fabricates, the committed
    episode metadata and the later durable terminal record.
    """

    trace_session_id: str
    runtime_session_id: str
    control_domain_id: str
    episode_uuid: str
    terminal_fact_id: str
    terminal_fact_hash: str
    terminal_journal_sequence: int
    terminal_record_integrity_hash: str
    trace_seal_hash: str
    trace_commit_hash: str
    trace_commit_relative_path: str
    trace_terminal_receipt_ref: str = ""
    trace_commit_receipt_ref: str = ""
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "trace_session_id",
            "runtime_session_id",
            "control_domain_id",
            "episode_uuid",
            "terminal_fact_id",
            "terminal_fact_hash",
            "terminal_record_integrity_hash",
            "trace_seal_hash",
            "trace_commit_hash",
            "trace_commit_relative_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), name=field_name),
            )
        sequence = int(self.terminal_journal_sequence)
        if (
            isinstance(self.terminal_journal_sequence, bool)
            or sequence <= 0
            or sequence != self.terminal_journal_sequence
        ):
            raise ValueError("terminal_journal_sequence must be a positive integer.")
        object.__setattr__(self, "terminal_journal_sequence", sequence)
        terminal_payload = {
            "trace_session_id": self.trace_session_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "terminal_fact_id": self.terminal_fact_id,
            "terminal_fact_hash": self.terminal_fact_hash,
            "terminal_journal_sequence": sequence,
            "terminal_record_integrity_hash": (
                self.terminal_record_integrity_hash
            ),
        }
        terminal_ref = (
            "control-trace-terminal-durable:"
            + canonical_content_hash(
                terminal_payload,
                domain=(
                    CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1
                ),
            )
        )
        commit_ref = f"control-trace-capture-commit:{self.trace_commit_hash}"
        payload = {
            **terminal_payload,
            "episode_uuid": self.episode_uuid,
            "trace_seal_hash": self.trace_seal_hash,
            "trace_commit_hash": self.trace_commit_hash,
            "trace_commit_relative_path": self.trace_commit_relative_path,
            "trace_terminal_receipt_ref": terminal_ref,
            "trace_commit_receipt_ref": commit_ref,
        }
        object.__setattr__(self, "trace_terminal_receipt_ref", terminal_ref)
        object.__setattr__(self, "trace_commit_receipt_ref", commit_ref)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1,
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "trace_session_id": self.trace_session_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "episode_uuid": self.episode_uuid,
            "terminal_fact_id": self.terminal_fact_id,
            "terminal_fact_hash": self.terminal_fact_hash,
            "terminal_journal_sequence": self.terminal_journal_sequence,
            "terminal_record_integrity_hash": (
                self.terminal_record_integrity_hash
            ),
            "trace_seal_hash": self.trace_seal_hash,
            "trace_commit_hash": self.trace_commit_hash,
            "trace_commit_relative_path": self.trace_commit_relative_path,
            "trace_terminal_receipt_ref": self.trace_terminal_receipt_ref,
            "trace_commit_receipt_ref": self.trace_commit_receipt_ref,
            "semantic_content_hash": self.semantic_content_hash,
        }


class ControlTraceEpisodeSessionCapture:
    """Concrete binding to one lower-owned durable ControlTrace session."""

    def __init__(
        self,
        root: str | Path,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        trace_requirement_profile_id: str,
        clock_domain_id: str,
        capture_purpose: str,
        trace_session_id: str | None = None,
        max_spool_bytes: int = 64 << 20,
        runtime_capacity: int = (
            CONTROL_TRACE_SESSION_CAPTURE_RUNTIME_CAPACITY_V1
        ),
        legacy_episode_index: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._max_spool_bytes = int(max_spool_bytes)
        self._runtime_capacity = int(runtime_capacity)
        if self._runtime_capacity <= 0:
            raise ValueError("runtime_capacity must be positive.")
        self._runtime_session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        self._control_domain_id = _required(
            control_domain_id,
            name="control_domain_id",
        )
        self._trace_requirement_profile_id = _required(
            trace_requirement_profile_id,
            name="trace_requirement_profile_id",
        )
        self._clock_domain_id = _required(
            clock_domain_id,
            name="clock_domain_id",
        )
        self._capture_purpose = _required(
            capture_purpose,
            name="capture_purpose",
        )
        episode_index = int(legacy_episode_index)
        if (
            isinstance(legacy_episode_index, bool)
            or episode_index < 0
            or episode_index != legacy_episode_index
        ):
            raise ValueError("legacy_episode_index must be a non-negative integer.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._legacy_episode_index = episode_index
        self._clock = clock
        self._lock = threading.RLock()
        self._open_session(trace_session_id=trace_session_id)

    def _open_session(self, *, trace_session_id: str | None) -> None:
        session_id = (
            str(trace_session_id).strip()
            if trace_session_id is not None
            else self._capture_purpose.replace("_", "-") + "-" + uuid.uuid4().hex
        )
        if not session_id:
            raise ValueError("trace_session_id must be non-empty.")
        try:
            profile = TRACE_REQUIREMENT_PROFILES_V1[
                self._trace_requirement_profile_id
            ]
        except KeyError as error:
            raise ValueError(
                "Unknown ControlTrace requirement profile: "
                f"{self._trace_requirement_profile_id!r}."
            ) from error
        self.trace_runtime = ControlTraceRuntime(
            RuntimeTraceSession(
                trace_session_id=session_id,
                runtime_session_id=self._runtime_session_id,
                control_domain_id=self._control_domain_id,
                profile_id=profile.profile_id,
                started_at=float(self._clock()),
                clock_domain_id=self._clock_domain_id,
            ),
            profile=profile,
            capacity=self._runtime_capacity,
        )
        self.session_root = (
            self.root
            / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
            / session_id
        )
        self.session_root.mkdir(parents=True, exist_ok=False)
        self.spool = ControlTraceSpool(
            self.session_root / "control_trace.wal",
            trace_session_id=session_id,
            max_bytes=self._max_spool_bytes,
            physical_preallocation=False,
            allow_growth=True,
            growth_bytes=self._max_spool_bytes,
        )
        self.health = ControlTraceWriterHealthLedger(
            self.session_root / "writer_health.ledger",
            trace_session_id=session_id,
        )
        self.durable_sink = ControlTraceSpoolSink(
            sink_id=f"{session_id}:spool",
            runtime=self.trace_runtime,
            spool=self.spool,
            health_ledger=self.health,
        )
        self.spool_worker = ControlTraceSpoolWorker(
            self.durable_sink,
            poll_interval_sec=0.01,
            thread_name=f"control-trace-capture-{session_id}",
        )
        self.spool_worker.start()
        self._dependency_sequences: tuple[int, ...] = ()
        self.recorder = EpisodeTraceRecorderSink(
            self.trace_runtime,
            lifecycle_owner_id=f"{session_id}:episode-recorder",
            flush_to_durable=self.spool_worker.flush_once,
            dependency_sequences=(
                lambda _runtime: self._dependency_sequences
            ),
            manifest_store=EpisodeCaptureManifestStore(self.root),
        )
        self._prepared: Any | None = None
        self._final_report: Any | None = None
        self._runtime_finalization: (
            ControlTraceSessionFinalizationReceipt | None
        ) = None
        self._admission_failure: str | None = None
        self._closed = False

    @property
    def runtime_session_id(self) -> str:
        return self._runtime_session_id

    @property
    def control_domain_id(self) -> str:
        return self._control_domain_id

    @property
    def clock_domain_id(self) -> str:
        return self._clock_domain_id

    def rotate_runtime_session(self, *, now: float) -> dict[str, object]:
        """Start a fresh trace identity after one whole-runtime teardown."""

        with self._lock:
            if self._prepared is not None:
                self.abort(now=float(now))
            self.spool_worker.notify()
            self.spool_worker.stop(timeout_sec=5.0)
            self.health.close()
            self.spool.close()
            self._open_session(trace_session_id=None)
            return self.status()

    def admit_trace(self, fact: Any) -> None:
        receipt = self.trace_runtime.issue(fact)
        if not receipt.accepted:
            failure = (
                "ControlTrace session capture admission failed: "
                f"{receipt.rejection_code}:{receipt.message}"
            )
            with self._lock:
                self._admission_failure = failure
            raise RuntimeError(failure)
        # Persistence is deliberately subordinate to trace admission.  Keep
        # fsync work off the Station control thread and continuously drain the
        # journal so episode release/rebuild has only a small tail to flush.
        self.spool_worker.notify()

    def prepare(self, *, now: float) -> dict[str, object]:
        with self._lock:
            if self._prepared is not None:
                return self.status()
            if self._final_report is not None:
                raise RuntimeError("ControlTrace episode was already finalized.")
            prepared = self.recorder.prepare_episode(
                legacy_episode_index=self._legacy_episode_index,
                prepared_at=float(now),
            )
            # Publish the subordinate preparation before binding so every
            # failure path can find and release it.  Previously a bind error
            # happened before ``self._prepared`` was assigned, leaving the
            # recorder's private preparation active and making the next retry
            # fail with ``episode preparation is already active``.
            self._prepared = prepared
            try:
                self.recorder.bind_prepared_episode(
                    prepared,
                    bound_at=float(now),
                )
                # Runtime/schema/source activation facts are intentionally
                # published before native viewer startup is accepted.  Freeze
                # every admitted session-level fact after the atomic bind so
                # an episode can reference that immutable bootstrap lineage
                # without pretending those facts occurred inside the episode.
                self._dependency_sequences = tuple(
                    entry.journal_sequence
                    for entry in self.trace_runtime.journal_entries()
                    if entry.episode_membership is None
                )
            except BaseException as error:  # noqa: BLE001 - preserve bind cause.
                cleanup_error: BaseException | None = None
                try:
                    lifecycle = getattr(self.recorder, "lifecycle", None)
                    active_episode = getattr(
                        lifecycle,
                        "active_episode_uuid",
                        None,
                    )
                    if active_episode is not None:
                        self.recorder.release_episode(
                            prepared,
                            unbound_at=float(now),
                            reason=f"{self._capture_purpose}_bind_failed",
                        )
                    else:
                        cancel_prepared = getattr(
                            self.recorder,
                            "cancel_prepared_episode",
                            None,
                        )
                        if not callable(cancel_prepared):
                            raise RuntimeError(
                                "ControlTrace recorder cannot cancel an "
                                "unbound preparation."
                            )
                        cancel_prepared(prepared)
                    self.spool_worker.flush_once()
                except BaseException as reset_error:  # noqa: BLE001 - retain both causes.
                    cleanup_error = reset_error
                finally:
                    self._prepared = None
                    self._dependency_sequences = ()
                if cleanup_error is not None:
                    raise RuntimeError(
                        f"{type(error).__name__}:{error}; "
                        "bind_cleanup_failed="
                        f"{type(cleanup_error).__name__}:{cleanup_error}"
                    ) from error
                raise
            return self.status()

    def finalize(self, *, now: float) -> dict[str, object]:
        with self._lock:
            prepared = self._prepared
            if prepared is None:
                raise RuntimeError("ControlTrace episode is not prepared.")
            released = self.recorder.release_episode(
                prepared,
                unbound_at=float(now),
                reason=f"{self._capture_purpose}_finalize",
            )
            finalized = self.recorder.legacy_finalize_completed(
                released,
                completed_at=float(now),
            )
            self.spool_worker.flush_once()
            self._prepared = None
            self._final_report = finalized
            if finalized.capture_commit_status != "committed":
                raise RuntimeError(
                    "ControlTrace episode did not commit: "
                    f"{finalized.message}"
                )
            return self.status()

    def complete_runtime(self) -> ControlTraceSessionFinalizationReceipt:
        """Join one durable commit with the later durable terminal fact.

        The owning execution runtime must already be closed and have emitted
        exactly one ``runtime_trace_terminal`` outside the episode. A simulator
        application may retain passive scene resources until session close.
        """

        with self._lock:
            if self._runtime_finalization is not None:
                return self._runtime_finalization
            if self._prepared is not None:
                raise RuntimeError(
                    "Runtime finalization requires the episode to be unbound."
                )
            finalized = self._final_report
            if finalized is None or finalized.capture_commit_status != "committed":
                raise RuntimeError(
                    "Runtime finalization requires a durable episode commit."
                )
            commit = finalized.capture_commit_candidate
            publish_receipt = finalized.capture_commit_publish_receipt
            if commit is None or publish_receipt is None:
                raise RuntimeError("Committed trace is missing its commit receipt.")
            if (
                not bool(publish_receipt.durable)
                or publish_receipt.semantic_content_hash
                != commit.semantic_content_hash
                or publish_receipt.episode_uuid != finalized.episode_uuid
            ):
                raise RuntimeError("ControlTrace commit publication receipt mismatch.")
            flush_report = self.spool_worker.flush_once()
            if flush_report.failure_code is not None:
                raise RuntimeError(
                    "ControlTrace terminal persistence failed: "
                    f"{flush_report.failure_code}."
                )
            terminal_entries = tuple(
                entry
                for entry in self.trace_runtime.journal_entries()
                if entry.fact.event_type == "runtime_trace_terminal"
            )
            if len(terminal_entries) != 1:
                raise RuntimeError(
                    "Runtime finalization requires exactly one terminal fact."
                )
            terminal_entry = terminal_entries[0]
            if terminal_entry.episode_membership is not None:
                raise RuntimeError(
                    "Runtime terminal fact must be outside the formal episode."
                )
            snapshot = self.trace_runtime.snapshot()
            if terminal_entry.journal_sequence not in snapshot.durable_sequences:
                raise RuntimeError("Runtime terminal fact is not durable.")
            records = {
                record.journal_sequence: record
                for record in self.trace_runtime.records()
            }
            terminal_record = records.get(terminal_entry.journal_sequence)
            if terminal_record is None:
                raise RuntimeError("Runtime terminal record is missing.")
            receipt = ControlTraceSessionFinalizationReceipt(
                trace_session_id=self.trace_runtime.session.trace_session_id,
                runtime_session_id=self.trace_runtime.session.runtime_session_id,
                control_domain_id=self.trace_runtime.session.control_domain_id,
                episode_uuid=finalized.episode_uuid,
                terminal_fact_id=terminal_entry.fact.fact_id,
                terminal_fact_hash=terminal_entry.fact.semantic_content_hash,
                terminal_journal_sequence=terminal_entry.journal_sequence,
                terminal_record_integrity_hash=(
                    terminal_record.record_integrity_hash
                ),
                trace_seal_hash=commit.trace_seal_hash,
                trace_commit_hash=commit.semantic_content_hash,
                trace_commit_relative_path=publish_receipt.relative_path,
            )
            self._runtime_finalization = receipt
            return receipt

    def abort(self, *, now: float) -> None:
        with self._lock:
            prepared = self._prepared
            if prepared is None:
                # Recover a lower-owned preparation that was created before a
                # bind failure could publish this wrapper's receipt.  This is
                # intentionally read-only discovery; release/cancel remains
                # owned by EpisodeTraceRecorderSink.
                prepared = getattr(self.recorder, "prepared_episode", None)
            if prepared is None:
                return
            try:
                self.recorder.release_episode(
                    prepared,
                    unbound_at=float(now),
                    reason=f"{self._capture_purpose}_aborted",
                )
            except RuntimeError as error:
                # A bind can fail after ``prepare_episode`` but before the
                # lifecycle records an active membership.  Treat that state
                # as an unbound preparation and cancel it transactionally.
                if "No ControlTrace episode is currently bound" not in str(error):
                    raise
                cancel_prepared = getattr(
                    self.recorder,
                    "cancel_prepared_episode",
                    None,
                )
                if not callable(cancel_prepared):
                    raise
                cancel_prepared(prepared)
            finally:
                self.spool_worker.flush_once()
                self._prepared = None
                self._dependency_sequences = ()

    def status(self) -> dict[str, object]:
        with self._lock:
            prepared = self._prepared
            finalized = self._final_report
            writer = self.spool_worker.snapshot()
            writer_failure = (
                self._admission_failure
                or writer.fatal_error_type
                or (
                    None
                    if writer.last_report is None
                    else writer.last_report.failure_code
                )
            )
            write_offset = self.spool.write_offset
            allocated_bytes = self.spool.max_bytes
            if writer_failure is not None:
                capture_state = "faulted"
                message = (
                    f"ControlTrace capture fault: {writer_failure}"
                )
            elif (
                finalized is not None
                and finalized.capture_commit_status == "committed"
            ):
                capture_state = "committed"
                message = finalized.message
            elif prepared is not None:
                capture_state = "recording"
                message = ""
            else:
                capture_state = "ready"
                message = "" if finalized is None else finalized.message
            return {
                "enabled": True,
                "trace_session_id": (
                    self.trace_runtime.session.trace_session_id
                ),
                "runtime_session_id": (
                    self.trace_runtime.session.runtime_session_id
                ),
                "control_domain_id": (
                    self.trace_runtime.session.control_domain_id
                ),
                "trace_requirement_profile_id": (
                    self.trace_runtime.session.profile_id
                ),
                "session_root": str(self.session_root),
                "episode_uuid": (
                    prepared.episode_uuid
                    if prepared is not None
                    else None
                    if finalized is None
                    else finalized.episode_uuid
                ),
                "status": capture_state,
                "commit_status": (
                    None
                    if finalized is None
                    else finalized.capture_commit_status
                ),
                "message": message,
                "runtime_finalization": (
                    None
                    if self._runtime_finalization is None
                    else self._runtime_finalization.to_wire()
                ),
                "dependency_count": len(self._dependency_sequences),
                "writer": {
                    "thread_alive": writer.thread_alive,
                    "flush_count": writer.flush_count,
                    "fatal_error_type": writer.fatal_error_type,
                    "last_failure_code": (
                        self._admission_failure
                        or (
                            None
                            if writer.last_report is None
                            else writer.last_report.failure_code
                        )
                    ),
                    "write_offset_bytes": write_offset,
                    "allocated_bytes": allocated_bytes,
                    "runtime_entry_count": (
                        self.trace_runtime.resident_entry_count
                    ),
                    "runtime_capacity": self._runtime_capacity,
                    "issued_sequence": self.trace_runtime.issued_sequence,
                },
            }

    def close(self, *, now: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if self._prepared is not None:
                self.abort(
                    now=time.monotonic() if now is None else float(now)
                )
            self.spool_worker.notify()
            self.spool_worker.stop(timeout_sec=5.0)
            self.health.close()
            self.spool.close()
            self._closed = True




__all__ = [
    "CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1",
    "CONTROL_TRACE_SESSION_CAPTURE_RUNTIME_CAPACITY_V1",
    "CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1",
    "ControlTraceEpisodeCapturePort",
    "ControlTraceEpisodeSessionCapture",
    "ControlTraceSessionFinalizationReceipt",
]

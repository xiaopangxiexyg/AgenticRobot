"""Deterministic bounded in-memory ControlTrace journal for M4A."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Sequence

from dataset_contracts.completeness import (
    ControlTraceCompletenessReport,
    TraceRequirementProfile,
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from dataset_contracts.control_trace import ControlTraceFact, PersistedRecordEnvelope

from .lifecycle import (
    EpisodeTraceMembership,
    EpisodeTraceMembershipRelease,
    RuntimeTraceSession,
)


TRACE_ADMISSION_REJECTION_CODES_V1 = (
    "session_mismatch",
    "domain_mismatch",
    "event_not_applicable",
    "duplicate_fact_id",
    "journal_capacity_exhausted",
    "terminal_already_accepted",
)


@dataclass(frozen=True)
class TraceAdmissionReceipt:
    fact_id: str
    event_type: str
    journal_issued_sequence: int
    accepted: bool
    sink_accepted_sequence: int | None
    sink_accepted_high_watermark: int
    sink_durable_high_watermark: int
    rejection_code: str | None
    message: str
    record_integrity_hash: str | None


@dataclass(frozen=True)
class ControlTraceRuntimeSnapshot:
    trace_session_id: str
    profile_id: str
    journal_issued_sequence: int
    sink_accepted_high_watermark: int
    sink_durable_high_watermark: int
    accepted_sequences: tuple[int, ...]
    durable_sequences: tuple[int, ...]
    rejected_sequences: tuple[int, ...]
    accepted_fact_ids: tuple[str, ...]
    current_episode_uuid: str | None
    durability_supported: bool


@dataclass(frozen=True)
class ControlTraceJournalEntry:
    journal_sequence: int
    fact: ControlTraceFact
    episode_membership: EpisodeTraceMembership | None

    def materialize(self, *, trace_session_id: str) -> PersistedRecordEnvelope:
        return PersistedRecordEnvelope.wrap(
            trace_session_id=trace_session_id,
            journal_sequence=self.journal_sequence,
            fact=self.fact,
            episode_membership=self.episode_membership,
            schema_refs=("dexteleop_control_trace_v2.4",),
        )


@dataclass(frozen=True)
class ControlTraceAdmissionLatencyReport:
    sample_count: int
    p50_sec: float
    p95_sec: float
    p99_sec: float
    max_sec: float
    deadline_sec: float
    deadline_miss_count: int


def summarize_control_trace_admission_latency(
    durations_sec: Sequence[float],
    *,
    deadline_sec: float,
) -> ControlTraceAdmissionLatencyReport:
    values = sorted(float(value) for value in durations_sec)
    deadline = float(deadline_sec)
    if not values or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Latency samples/deadline must be non-empty, finite, and positive.")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite and non-negative.")

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(values)) - 1)
        return values[index]

    return ControlTraceAdmissionLatencyReport(
        sample_count=len(values),
        p50_sec=percentile(0.50),
        p95_sec=percentile(0.95),
        p99_sec=percentile(0.99),
        max_sec=values[-1],
        deadline_sec=deadline,
        deadline_miss_count=sum(value > deadline for value in values),
    )


def _gap_ranges(values: set[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    ordered = sorted(values)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return tuple(ranges)


def _contiguous_high_watermark(sequences: set[int], *, issued: int) -> int:
    high = 0
    while high < issued and high + 1 in sequences:
        high += 1
    return high


class ControlTraceRuntime:
    """M4A journal with explicit evidence of admission and durability gaps.

    ``issue`` never performs file I/O and never waits for storage capacity. A rejected
    fact still consumes a journal-issued sequence so loss remains visible.  M4A
    has no durable sink; consequently it can establish in-memory completeness
    but can never declare a capture commit eligible.
    """

    def __init__(
        self,
        session: RuntimeTraceSession,
        *,
        capacity: int = 4096,
        profile: TraceRequirementProfile | None = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("ControlTraceRuntime capacity must be positive.")
        resolved = profile or trace_requirement_profile(session.profile_id)
        if resolved.profile_id != session.profile_id:
            raise ValueError("Trace profile must match RuntimeTraceSession.profile_id.")
        self.session = session
        self.profile = resolved
        self.capacity = int(capacity)
        self._membership: EpisodeTraceMembership | None = None
        self._lifecycle_owner_id: str | None = None
        self._membership_epoch_high_watermark = -1
        self._issued = 0
        self._accepted_sequences: set[int] = set()
        self._durable_sequences: set[int] = set()
        self._accepted_high_watermark = 0
        self._durable_high_watermark = 0
        self._rejected_sequences: set[int] = set()
        self._entries: list[ControlTraceJournalEntry] = []
        self._receipts: list[TraceAdmissionReceipt] = []
        self._fact_ids: set[str] = set()
        self._terminal_accepted = False
        self._durable_sink_id: str | None = None
        self._lock = threading.RLock()

    @property
    def episode_membership(self) -> EpisodeTraceMembership | None:
        with self._lock:
            return self._membership

    def bind_episode(self, membership: EpisodeTraceMembership) -> TraceAdmissionReceipt:
        """Bind externally owned episode membership without owning its lifecycle."""

        with self._lock:
            current = self._membership
            owner = self._lifecycle_owner_id
            if owner is not None and membership.lifecycle_owner_id != owner:
                raise ValueError("A different episode lifecycle owner is already bound.")
            if membership.membership_epoch <= self._membership_epoch_high_watermark:
                raise ValueError("Episode membership epoch must strictly increase.")
            if current is not None:
                raise ValueError("Clear the current episode membership before binding another.")
            self._lifecycle_owner_id = membership.lifecycle_owner_id
            self._membership_epoch_high_watermark = membership.membership_epoch
            self._membership = membership
            fact = ControlTraceFact.create(
                fact_id=membership.binding_event_id,
                event_type="episode_membership_transition",
                runtime_session_id=self.session.runtime_session_id,
                control_domain_id=self.session.control_domain_id,
                occurred_at=membership.bound_at,
                clock_domain_id=self.session.clock_domain_id,
                payload={
                    "transition_id": membership.binding_event_id,
                    "transition": "bind",
                    "episode_uuid": membership.episode_uuid,
                    "lifecycle_owner_id": membership.lifecycle_owner_id,
                    "membership_epoch": membership.membership_epoch,
                    "occurred_at": membership.bound_at,
                    "previous_episode_uuid": None,
                    "reason": "external_episode_owner_bind",
                },
            )
            receipt = self.issue(fact)
            if not receipt.accepted:
                self._membership = None
            return receipt

    def clear_episode_membership(
        self,
        release: EpisodeTraceMembershipRelease,
    ) -> TraceAdmissionReceipt | None:
        """Detach membership after the real owner finalized or aborted it."""

        with self._lock:
            current = self._membership
            if current is None:
                return None
            if release.lifecycle_owner_id != current.lifecycle_owner_id:
                raise ValueError("Only the bound lifecycle owner may clear membership.")
            if release.episode_uuid != current.episode_uuid:
                raise ValueError("Membership release belongs to another episode.")
            if release.membership_epoch <= current.membership_epoch:
                raise ValueError("Membership clear epoch must advance.")
            fact = ControlTraceFact.create(
                fact_id=release.unbinding_event_id,
                event_type="episode_membership_transition",
                runtime_session_id=self.session.runtime_session_id,
                control_domain_id=self.session.control_domain_id,
                occurred_at=release.unbound_at,
                clock_domain_id=self.session.clock_domain_id,
                payload={
                    "transition_id": release.unbinding_event_id,
                    "transition": "unbind",
                    "episode_uuid": release.episode_uuid,
                    "lifecycle_owner_id": release.lifecycle_owner_id,
                    "membership_epoch": release.membership_epoch,
                    "occurred_at": release.unbound_at,
                    "previous_episode_uuid": current.episode_uuid,
                    "reason": release.reason,
                },
            )
            receipt = self.issue(fact)
            self._membership_epoch_high_watermark = release.membership_epoch
            self._membership = None
            return receipt

    def _reject(
        self,
        fact: ControlTraceFact,
        *,
        issued: int,
        code: str,
        message: str,
    ) -> TraceAdmissionReceipt:
        self._rejected_sequences.add(issued)
        receipt = TraceAdmissionReceipt(
            fact_id=fact.fact_id,
            event_type=fact.event_type,
            journal_issued_sequence=issued,
            accepted=False,
            sink_accepted_sequence=None,
            sink_accepted_high_watermark=self._accepted_high_watermark,
            sink_durable_high_watermark=self._durable_high_watermark,
            rejection_code=code,
            message=message,
            record_integrity_hash=None,
        )
        self._receipts.append(receipt)
        return receipt

    def issue(self, fact: ControlTraceFact) -> TraceAdmissionReceipt:
        """Attempt immediate in-memory admission and return without I/O."""

        if not isinstance(fact, ControlTraceFact):
            raise TypeError("ControlTraceRuntime.issue requires ControlTraceFact.")
        with self._lock:
            self._issued += 1
            issued = self._issued
            if fact.runtime_session_id != self.session.runtime_session_id:
                return self._reject(
                    fact,
                    issued=issued,
                    code="session_mismatch",
                    message="Fact belongs to another runtime session.",
                )
            if fact.control_domain_id != self.session.control_domain_id:
                return self._reject(
                    fact,
                    issued=issued,
                    code="domain_mismatch",
                    message="Fact belongs to another control domain.",
                )
            if fact.event_type in self.profile.not_applicable_event_types:
                return self._reject(
                    fact,
                    issued=issued,
                    code="event_not_applicable",
                    message="Event type is not applicable to this trace profile.",
                )
            if fact.fact_id in self._fact_ids:
                return self._reject(
                    fact,
                    issued=issued,
                    code="duplicate_fact_id",
                    message="Fact ID was already admitted.",
                )
            if self._terminal_accepted:
                return self._reject(
                    fact,
                    issued=issued,
                    code="terminal_already_accepted",
                    message="No facts may be admitted after the terminal fact.",
                )
            if len(self._entries) >= self.capacity:
                return self._reject(
                    fact,
                    issued=issued,
                    code="journal_capacity_exhausted",
                    message="Bounded in-memory journal is full.",
                )

            self._entries.append(
                ControlTraceJournalEntry(
                    journal_sequence=issued,
                    fact=fact,
                    episode_membership=self._membership,
                )
            )
            self._fact_ids.add(fact.fact_id)
            self._accepted_sequences.add(issued)
            while self._accepted_high_watermark + 1 in self._accepted_sequences:
                self._accepted_high_watermark += 1
            if fact.event_type == "runtime_trace_terminal":
                self._terminal_accepted = True
            receipt = TraceAdmissionReceipt(
                fact_id=fact.fact_id,
                event_type=fact.event_type,
                journal_issued_sequence=issued,
                accepted=True,
                sink_accepted_sequence=issued,
                sink_accepted_high_watermark=self._accepted_high_watermark,
                sink_durable_high_watermark=self._durable_high_watermark,
                rejection_code=None,
                message="accepted_in_memory_pending_storage_encoding",
                record_integrity_hash=None,
            )
            self._receipts.append(receipt)
            return receipt

    def records(self) -> tuple[PersistedRecordEnvelope, ...]:
        """Materialize storage envelopes outside the control-facing admission lock."""

        with self._lock:
            entries = tuple(self._entries)
        return tuple(
            entry.materialize(trace_session_id=self.session.trace_session_id)
            for entry in entries
        )

    def journal_entries(self) -> tuple[ControlTraceJournalEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def journal_entries_from_offset(
        self,
        accepted_entry_offset: int,
    ) -> tuple[ControlTraceJournalEntry, ...]:
        """Return accepted entries not yet scanned by one subordinate sink.

        The offset is an accepted-entry count, not a journal sequence. Rejected
        facts consume journal sequences but do not occupy ``_entries``.
        """

        offset = int(accepted_entry_offset)
        if isinstance(accepted_entry_offset, bool) or offset < 0:
            raise ValueError(
                "accepted_entry_offset must be a non-negative integer."
            )
        with self._lock:
            if offset > len(self._entries):
                raise ValueError(
                    "accepted_entry_offset exceeds admitted journal entries."
                )
            return tuple(self._entries[offset:])

    def register_durable_sink(self, sink_id: str) -> None:
        """Register the sole subordinate sink allowed to acknowledge durability."""

        normalized = str(sink_id).strip()
        if not normalized:
            raise ValueError("sink_id must be non-empty.")
        with self._lock:
            if self._durable_sink_id is not None and self._durable_sink_id != normalized:
                raise ValueError("A different durable ControlTrace sink is already registered.")
            self._durable_sink_id = normalized

    def acknowledge_durable(
        self,
        *,
        sink_id: str,
        journal_sequence: int,
        record_integrity_hash: str,
    ) -> None:
        """Accept a post-fsync acknowledgement for an admitted immutable record."""

        normalized_sink = str(sink_id)
        sequence = int(journal_sequence)
        with self._lock:
            if self._durable_sink_id is None or normalized_sink != self._durable_sink_id:
                raise ValueError("Durability acknowledgement comes from an unregistered sink.")
            entry = next(
                (
                    item
                    for item in self._entries
                    if item.journal_sequence == sequence
                ),
                None,
            )
            if entry is None:
                raise ValueError("Durability acknowledgement references an unaccepted sequence.")
        record = entry.materialize(trace_session_id=self.session.trace_session_id)
        if record.record_integrity_hash != str(record_integrity_hash):
            raise ValueError("Durability acknowledgement record hash mismatch.")
        with self._lock:
            if self._durable_sink_id != normalized_sink:
                raise ValueError("Durable sink changed during acknowledgement.")
            if sequence not in self._accepted_sequences:
                raise ValueError("Durability acknowledgement lost its admitted sequence.")
            if record.record_integrity_hash != str(record_integrity_hash):
                raise ValueError("Durability acknowledgement record hash mismatch.")
            self._durable_sequences.add(sequence)
            while self._durable_high_watermark + 1 in self._durable_sequences:
                self._durable_high_watermark += 1

    def receipts(self) -> tuple[TraceAdmissionReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def snapshot(self) -> ControlTraceRuntimeSnapshot:
        # Copy only mutable references while holding the admission lock.  Set
        # sorting and fact-ID materialization must not let a diagnostic reader
        # starve the control-facing issue() path.
        with self._lock:
            membership = self._membership
            issued = self._issued
            accepted_high_watermark = self._accepted_high_watermark
            durable_high_watermark = self._durable_high_watermark
            accepted_sequences = set(self._accepted_sequences)
            durable_sequences = set(self._durable_sequences)
            rejected_sequences = set(self._rejected_sequences)
            entries = tuple(self._entries)
            durability_supported = self._durable_sink_id is not None
        return ControlTraceRuntimeSnapshot(
            trace_session_id=self.session.trace_session_id,
            profile_id=self.profile.profile_id,
            journal_issued_sequence=issued,
            sink_accepted_high_watermark=accepted_high_watermark,
            sink_durable_high_watermark=durable_high_watermark,
            accepted_sequences=tuple(sorted(accepted_sequences)),
            durable_sequences=tuple(sorted(durable_sequences)),
            rejected_sequences=tuple(sorted(rejected_sequences)),
            accepted_fact_ids=tuple(entry.fact.fact_id for entry in entries),
            current_episode_uuid=None if membership is None else membership.episode_uuid,
            durability_supported=durability_supported,
        )

    def completeness(self) -> ControlTraceCompletenessReport:
        with self._lock:
            observed = frozenset(entry.fact.event_type for entry in self._entries)
            missing = self.profile.required_event_types - observed
            not_applicable = observed & self.profile.not_applicable_event_types
            all_sequences = set(range(1, self._issued + 1))
            admission_gaps = all_sequences - self._accepted_sequences
            durability_gaps = self._accepted_sequences - self._durable_sequences
            terminal = "runtime_trace_terminal" in observed
            lineage_issues = evaluate_control_trace_lineage(
                tuple(entry.fact for entry in self._entries),
                profile=self.profile,
            )
            lineage_complete = not lineage_issues
            durability_supported = self._durable_sink_id is not None
            durable_complete = bool(
                durability_supported
                and not durability_gaps
                and self._accepted_sequences == self._durable_sequences
            )
            in_memory_complete = bool(
                not missing
                and not not_applicable
                and not admission_gaps
                and terminal
                and lineage_complete
            )
            return ControlTraceCompletenessReport(
                profile_id=self.profile.profile_id,
                observed_event_types=observed,
                missing_required_event_types=frozenset(missing),
                observed_not_applicable_event_types=frozenset(not_applicable),
                journal_issued_sequence=self._issued,
                sink_accepted_high_watermark=_contiguous_high_watermark(
                    self._accepted_sequences, issued=self._issued
                ),
                sink_durable_high_watermark=0,
                admission_gap_ranges=_gap_ranges(admission_gaps),
                durability_gap_ranges=_gap_ranges(durability_gaps),
                terminal_observed=terminal,
                lineage_policy_id=self.profile.lineage_policy_id,
                lineage_issues=lineage_issues,
                lineage_complete=lineage_complete,
                in_memory_complete=in_memory_complete,
                durability_supported=durability_supported,
                durable_complete=durable_complete,
                capture_commit_eligible=False,
            )


__all__ = [
    "TRACE_ADMISSION_REJECTION_CODES_V1",
    "ControlTraceRuntime",
    "ControlTraceJournalEntry",
    "ControlTraceAdmissionLatencyReport",
    "ControlTraceRuntimeSnapshot",
    "TraceAdmissionReceipt",
    "summarize_control_trace_admission_latency",
]

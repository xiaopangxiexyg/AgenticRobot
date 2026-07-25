"""Immutable, representation-neutral context available to composition strategies.

Context is admitted data, not a live backend handle.  A strategy may consume
history, state, future actions, labels, vision features, or other modalities
only after they have been frozen behind an explicit payload contract and
receiver-side causal cut.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
    ProducerEpochSnapshot,
)

from control_composition.retention import (
    TimeWindowRetentionSpec,
    TimeWindowStoreSnapshot,
)


# These are common interoperable names, not an exhaustive registry.  Strategy
# specs declare the exact kinds they consume so new modalities do not require a
# framework change.
COMPOSITION_CONTEXT_KINDS_V1 = frozenset(
    {
        "history",
        "future_action",
        "state",
        "modality",
        "vision",
        "label",
        "policy_advisory",
    }
)

# Keep the original hash domain so existing pose-context hashes and downstream
# trace identities remain stable during the package generalization.
COMPOSITION_CONTEXT_HASH_DOMAIN_V1 = "dexteleop.pose_mixing_context.v1"
COMPOSITION_CONTEXT_RECORD_HASH_DOMAIN_V1 = (
    "dexteleop.composition_context.semantic.v1"
)
COMPOSITION_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_context_snapshot.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


@dataclass(frozen=True)
class CompositionContext:
    """One immutable causal fact available to a replaceable strategy.

    ``semantic_payload`` contains immutable semantic data or stable blob
    references.  It must never expose a mutable simulator, device, model, or
    sensor object that can be reread after the decision cut.
    """

    context_id: str
    context_kind: str
    source_fact_id: str
    source_fact_hash: str
    payload_contract_hash: str
    semantic_payload: Mapping[str, Any]
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float | None = None
    effective_until: float | None = None
    blob_refs: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        context_id = _required(self.context_id, name="context_id")
        kind = _required(self.context_kind, name="context_kind")
        source_fact_id = _required(self.source_fact_id, name="source_fact_id")
        source_fact_hash = _required(self.source_fact_hash, name="source_fact_hash")
        contract_hash = _required(
            self.payload_contract_hash,
            name="payload_contract_hash",
        )
        available_at = _finite(self.available_at, name="available_at")
        clock_domain = _required(
            self.available_clock_domain_id,
            name="available_clock_domain_id",
        )
        admission_sequence = _nonnegative(
            self.control_admission_sequence,
            name="control_admission_sequence",
        )
        effective_from = (
            None
            if self.effective_from is None
            else _finite(self.effective_from, name="effective_from")
        )
        effective_until = (
            None
            if self.effective_until is None
            else _finite(self.effective_until, name="effective_until")
        )
        if (effective_from is None) != (effective_until is None):
            raise ValueError(
                "Composition context effective bounds must be both set or absent."
            )
        if (
            effective_from is not None
            and effective_until is not None
            and effective_until <= effective_from
        ):
            raise ValueError("Composition context effective interval must be non-empty.")
        blob_refs = tuple(
            _required(value, name="blob_ref") for value in self.blob_refs
        )
        if len(blob_refs) != len(set(blob_refs)):
            raise ValueError("Composition context blob_refs must be unique.")
        payload = deep_freeze(dict(self.semantic_payload))
        semantic = {
            "context_id": context_id,
            "context_kind": kind,
            "source_fact_id": source_fact_id,
            "source_fact_hash": source_fact_hash,
            "payload_contract_hash": contract_hash,
            "semantic_payload": payload,
            "available_at": available_at,
            "available_clock_domain_id": clock_domain,
            "control_admission_sequence": admission_sequence,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "blob_refs": blob_refs,
        }
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "context_kind", kind)
        object.__setattr__(self, "source_fact_id", source_fact_id)
        object.__setattr__(self, "source_fact_hash", source_fact_hash)
        object.__setattr__(self, "payload_contract_hash", contract_hash)
        object.__setattr__(self, "semantic_payload", payload)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "available_clock_domain_id", clock_domain)
        object.__setattr__(self, "control_admission_sequence", admission_sequence)
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "blob_refs", blob_refs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=COMPOSITION_CONTEXT_HASH_DOMAIN_V1,
            ),
        )

    def validate_for_cut(
        self,
        *,
        cutoff_at: float,
        cutoff_admission_sequence: int,
        target_effective_at: float,
        clock_domain_id: str,
    ) -> None:
        """Reject future, stale, or cross-clock context at a decision cut."""

        cutoff = _finite(cutoff_at, name="cutoff_at")
        cutoff_sequence = _nonnegative(
            cutoff_admission_sequence,
            name="cutoff_admission_sequence",
        )
        target = _finite(target_effective_at, name="target_effective_at")
        clock = _required(clock_domain_id, name="clock_domain_id")
        if self.available_clock_domain_id != clock:
            raise ValueError("Composition context uses another control clock domain.")
        if (
            self.available_at > cutoff
            or self.control_admission_sequence > cutoff_sequence
        ):
            raise ValueError(
                "Composition context was not available at the decision cut."
            )
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and not self.effective_from <= target < self.effective_until
        ):
            raise ValueError(
                "Composition context is not effective at the decision target."
            )


class CompositionContextAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class CompositionContextDraft:
    """Producer-side generic context submission.

    Producer/model-specific identities are optional lineage. They never decide
    whether a non-policy context is admissible.
    """

    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    context_channel: str
    context_kind: str
    semantic_payload: Mapping[str, Any]
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    payload_contract_hash: str
    source_fact_ids: tuple[str, ...] = ()
    source_fact_hashes: tuple[str, ...] = ()
    parent_context_ids: tuple[str, ...] = ()
    blob_refs: tuple[str, ...] = ()
    availability: str = "available"
    fault_code: str | None = None
    producer_node_id: str | None = None
    observation_cut_id: str | None = None
    inference_id: str | None = None
    policy_output_id: str | None = None
    producer_artifact_hash: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionContextRecord:
    context_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    context_channel: str
    context_kind: str
    semantic_payload: Mapping[str, Any]
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    payload_contract_hash: str
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    parent_context_ids: tuple[str, ...]
    blob_refs: tuple[str, ...]
    availability: str
    fault_code: str | None
    producer_node_id: str | None
    observation_cut_id: str | None
    inference_id: str | None
    policy_output_id: str | None
    producer_artifact_hash: str | None
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]

    def to_strategy_view(self) -> CompositionContext:
        source_id = (
            self.source_fact_ids[0] if self.source_fact_ids else self.context_id
        )
        source_hash = (
            self.source_fact_hashes[0]
            if self.source_fact_hashes
            else self.semantic_content_hash
        )
        return CompositionContext(
            context_id=self.context_id,
            context_kind=self.context_kind,
            source_fact_id=source_id,
            source_fact_hash=source_hash,
            payload_contract_hash=self.payload_contract_hash,
            semantic_payload=self.semantic_payload,
            available_at=self.available_at,
            available_clock_domain_id=self.available_clock_domain_id,
            control_admission_sequence=self.control_admission_sequence,
            effective_from=self.effective_from,
            effective_until=self.effective_until,
            blob_refs=self.blob_refs,
        )


@dataclass(frozen=True)
class CompositionContextAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    context_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    context: CompositionContextRecord | None = None


@dataclass(frozen=True)
class CompositionContextProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    contexts: tuple[CompositionContextRecord, ...]


@dataclass(frozen=True)
class CompositionContextSnapshot:
    context_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    required_channels: tuple[str, ...]
    selected_context_by_channel: Mapping[str, CompositionContextRecord]
    selected_context_id_by_channel: Mapping[str, str]
    selected_context_hash_by_channel: Mapping[str, str]
    missing_channels: tuple[str, ...]
    unavailable_channels: tuple[str, ...]
    producer_epoch_snapshot_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_context_by_channel",
            MappingProxyType(dict(self.selected_context_by_channel)),
        )
        object.__setattr__(
            self,
            "selected_context_id_by_channel",
            MappingProxyType(dict(self.selected_context_id_by_channel)),
        )
        object.__setattr__(
            self,
            "selected_context_hash_by_channel",
            MappingProxyType(dict(self.selected_context_hash_by_channel)),
        )

    @property
    def strategy_contexts(self) -> tuple[CompositionContext, ...]:
        return tuple(
            self.selected_context_by_channel[channel].to_strategy_view()
            for channel in self.required_channels
            if (
                channel in self.selected_context_by_channel
                and self.selected_context_by_channel[channel].availability
                == "available"
            )
        )


@dataclass(frozen=True)
class _PreparedContext:
    identity: RuntimeFactIdentity
    draft: CompositionContextDraft
    context_channel: str
    context_kind: str
    semantic_payload: Mapping[str, Any]
    audit_metadata: Mapping[str, Any]
    produced_at: float
    effective_from: float
    effective_until: float
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    parent_context_ids: tuple[str, ...]
    blob_refs: tuple[str, ...]
    availability: str
    fault_code: str | None


def _optional(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


def _prepare_context(draft: CompositionContextDraft) -> _PreparedContext:
    session = _required(draft.runtime_session_id, name="runtime_session_id")
    domain = _required(draft.control_domain_id, name="control_domain_id")
    producer = _required(draft.producer_id, name="producer_id")
    channel = _required(draft.context_channel, name="context_channel")
    kind = _required(draft.context_kind, name="context_kind")
    _required(draft.payload_contract_hash, name="payload_contract_hash")
    _required(draft.produced_clock_domain_id, name="produced_clock_domain_id")
    _required(draft.effective_clock_domain_id, name="effective_clock_domain_id")
    producer_epoch = _nonnegative(draft.producer_epoch, name="producer_epoch")
    sequence = _nonnegative(draft.sequence, name="sequence")
    produced_at = _finite(draft.produced_at, name="produced_at")
    effective_from = _finite(draft.effective_from, name="effective_from")
    effective_until = _finite(draft.effective_until, name="effective_until")
    if effective_until <= effective_from:
        raise CompositionContextAdmissionError(
            "invalid_effective_interval",
            "Composition context interval must be [effective_from, effective_until).",
        )
    source_ids = tuple(
        _required(value, name="source_fact_id") for value in draft.source_fact_ids
    )
    source_hashes = tuple(
        _required(value, name="source_fact_hash")
        for value in draft.source_fact_hashes
    )
    if len(source_ids) != len(source_hashes):
        raise CompositionContextAdmissionError(
            "invalid_lineage",
            "source_fact_ids and source_fact_hashes must have equal length.",
        )
    if len(source_ids) != len(set(source_ids)):
        raise CompositionContextAdmissionError(
            "invalid_lineage", "source_fact_ids must be unique."
        )
    parent_ids = tuple(
        _required(value, name="parent_context_id")
        for value in draft.parent_context_ids
    )
    blob_refs = tuple(
        _required(value, name="blob_ref") for value in draft.blob_refs
    )
    if len(parent_ids) != len(set(parent_ids)) or len(blob_refs) != len(
        set(blob_refs)
    ):
        raise CompositionContextAdmissionError(
            "invalid_lineage",
            "Context dependency and blob references must be unique.",
        )
    availability = _required(draft.availability, name="availability")
    if availability not in {"available", "missing", "fault"}:
        raise CompositionContextAdmissionError(
            "invalid_availability",
            "availability must be available, missing, or fault.",
        )
    fault_code = _optional(draft.fault_code, name="fault_code")
    if (availability == "fault") != (fault_code is not None):
        raise CompositionContextAdmissionError(
            "invalid_availability",
            "fault availability requires fault_code and other states forbid it.",
        )
    optional_fields = (
        "producer_node_id",
        "observation_cut_id",
        "inference_id",
        "policy_output_id",
        "producer_artifact_hash",
    )
    for field_name in optional_fields:
        _optional(getattr(draft, field_name), name=field_name)
    if kind == "artifact_bound_latent" and draft.producer_artifact_hash is None:
        raise CompositionContextAdmissionError(
            "artifact_hash_required",
            "artifact_bound_latent requires producer_artifact_hash.",
        )
    try:
        semantic_payload = deep_freeze(dict(draft.semantic_payload))
        audit_metadata = deep_freeze(dict(draft.audit_metadata))
    except TypeError as exc:
        raise CompositionContextAdmissionError(
            "unsupported_payload", str(exc)
        ) from exc
    return _PreparedContext(
        identity=RuntimeFactIdentity(
            runtime_session_id=session,
            control_domain_id=domain,
            producer_id=producer,
            producer_epoch=producer_epoch,
            sequence=sequence,
            namespace="composition_context",
        ),
        draft=draft,
        context_channel=channel,
        context_kind=kind,
        semantic_payload=semantic_payload,
        audit_metadata=audit_metadata,
        produced_at=produced_at,
        effective_from=effective_from,
        effective_until=effective_until,
        source_fact_ids=source_ids,
        source_fact_hashes=source_hashes,
        parent_context_ids=parent_ids,
        blob_refs=blob_refs,
        availability=availability,
        fault_code=fault_code,
    )


def _context_payload(
    prepared: _PreparedContext,
    *,
    available_at: float,
    available_clock_domain_id: str,
    control_admission_sequence: int,
) -> dict[str, Any]:
    draft = prepared.draft
    return {
        "context_id": prepared.identity.fact_id,
        "runtime_session_id": prepared.identity.runtime_session_id,
        "control_domain_id": prepared.identity.control_domain_id,
        "producer_id": prepared.identity.producer_id,
        "producer_epoch": prepared.identity.producer_epoch,
        "sequence": prepared.identity.sequence,
        "context_channel": prepared.context_channel,
        "context_kind": prepared.context_kind,
        "semantic_payload": prepared.semantic_payload,
        "produced_at": prepared.produced_at,
        "produced_clock_domain_id": draft.produced_clock_domain_id,
        "available_at": available_at,
        "available_clock_domain_id": available_clock_domain_id,
        "control_admission_sequence": control_admission_sequence,
        "effective_from": prepared.effective_from,
        "effective_until": prepared.effective_until,
        "effective_clock_domain_id": draft.effective_clock_domain_id,
        "payload_contract_hash": draft.payload_contract_hash,
        "source_fact_ids": prepared.source_fact_ids,
        "source_fact_hashes": prepared.source_fact_hashes,
        "parent_context_ids": prepared.parent_context_ids,
        "blob_refs": prepared.blob_refs,
        "availability": prepared.availability,
        "fault_code": prepared.fault_code,
        "producer_node_id": draft.producer_node_id,
        "observation_cut_id": draft.observation_cut_id,
        "inference_id": draft.inference_id,
        "policy_output_id": draft.policy_output_id,
        "producer_artifact_hash": draft.producer_artifact_hash,
    }


def composition_context_semantic_payload(
    context: CompositionContextRecord,
) -> dict[str, Any]:
    return {
        key: getattr(context, key)
        for key in (
            "context_id",
            "runtime_session_id",
            "control_domain_id",
            "producer_id",
            "producer_epoch",
            "sequence",
            "context_channel",
            "context_kind",
            "semantic_payload",
            "produced_at",
            "produced_clock_domain_id",
            "available_at",
            "available_clock_domain_id",
            "control_admission_sequence",
            "effective_from",
            "effective_until",
            "effective_clock_domain_id",
            "payload_contract_hash",
            "source_fact_ids",
            "source_fact_hashes",
            "parent_context_ids",
            "blob_refs",
            "availability",
            "fault_code",
            "producer_node_id",
            "observation_cut_id",
            "inference_id",
            "policy_output_id",
            "producer_artifact_hash",
        )
    }


class CompositionContextStore:
    """Receiver-admitted generic context with time-window retention."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        retention_spec: TimeWindowRetentionSpec,
    ) -> None:
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Context store and epochs must share one coordinator.")
        self.runtime_session_id = _required(
            runtime_session_id, name="runtime_session_id"
        )
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Context store session must match the control clock.")
        if not isinstance(retention_spec, TimeWindowRetentionSpec):
            raise TypeError("retention_spec must be a TimeWindowRetentionSpec.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.retention_spec = retention_spec
        self._records: list[CompositionContextRecord] = []
        self._by_id: dict[str, CompositionContextRecord] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._closed_target_through: float | None = None
        self._pruned_record_count = 0
        self._lock = threading.RLock()

    def admit(
        self, draft: CompositionContextDraft
    ) -> CompositionContextAdmissionReceipt:
        context_id: str | None = None
        try:
            prepared = _prepare_context(draft)
            context_id = prepared.identity.fact_id
            if prepared.identity.runtime_session_id != self.runtime_session_id:
                raise CompositionContextAdmissionError(
                    "session_mismatch", "Draft belongs to another runtime session."
                )
            if prepared.identity.control_domain_id != self.control_domain_id:
                raise CompositionContextAdmissionError(
                    "domain_mismatch", "Draft belongs to another control domain."
                )
            if (
                draft.effective_clock_domain_id
                != self.coordinator.clock_domain.clock_domain_id
            ):
                raise CompositionContextAdmissionError(
                    "clock_domain_mismatch",
                    "Context validity must use the receiver control clock.",
                )

            def build(
                available_at: float, admission_sequence: int
            ) -> CompositionContextRecord:
                try:
                    self.retention_spec.validate_admitted_interval(
                        available_at=available_at,
                        effective_from=prepared.effective_from,
                        effective_until=prepared.effective_until,
                    )
                except ValueError as exc:
                    raise CompositionContextAdmissionError(
                        "retention_window_violation", str(exc)
                    ) from exc
                payload = _context_payload(
                    prepared,
                    available_at=available_at,
                    available_clock_domain_id=(
                        self.coordinator.clock_domain.clock_domain_id
                    ),
                    control_admission_sequence=admission_sequence,
                )
                return CompositionContextRecord(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        payload, domain=COMPOSITION_CONTEXT_RECORD_HASH_DOMAIN_V1
                    ),
                    audit_metadata=prepared.audit_metadata,
                )

            record = self.coordinator.commit(store=self, build=build)
            return CompositionContextAdmissionReceipt(
                accepted=True,
                context_id=record.context_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                context=record,
            )
        except (
            CompositionContextAdmissionError,
            ProducerEpochError,
            TypeError,
            ValueError,
        ) as exc:
            code = (
                exc.code
                if isinstance(exc, CompositionContextAdmissionError)
                else "validation_failed"
            )
            return CompositionContextAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                context_id=context_id,
            )

    def _insert_admitted(self, record: CompositionContextRecord) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=record.producer_id,
                    producer_epoch=record.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise CompositionContextAdmissionError(
                    "inactive_producer_epoch", str(exc)
                ) from exc
            if record.context_id in self._by_id:
                raise CompositionContextAdmissionError(
                    "duplicate", "Composition context is already admitted."
                )
            key = (record.producer_id, record.producer_epoch)
            previous = self._last_sequence.get(key)
            if previous is not None and record.sequence <= previous:
                code = "duplicate" if record.sequence == previous else "sequence_rollback"
                raise CompositionContextAdmissionError(
                    code, f"Producer sequence must increase beyond {previous}."
                )
            self._records.append(record)
            self._by_id[record.context_id] = record
            self._last_sequence[key] = record.sequence

    def records_at_or_before(
        self, cut: ControlCut
    ) -> tuple[CompositionContextRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(
                record
                for record in self._records
                if admitted_at_or_before(record, cut)
            )

    def capture_snapshot(
        self,
        cut: ControlCut,
        *,
        producer_epochs: ProducerEpochSnapshot,
        target_effective_at: float,
        required_channels: tuple[str, ...],
    ) -> CompositionContextSnapshot:
        with self._lock:
            snapshot = build_composition_context_snapshot(
                CompositionContextProjection(
                    control_domain_id=cut.control_domain_id,
                    cutoff_at=cut.cutoff_at,
                    cutoff_admission_sequence=cut.cutoff_admission_sequence,
                    contexts=tuple(
                        record
                        for record in self._records
                        if admitted_at_or_before(record, cut)
                    ),
                ),
                producer_epochs=producer_epochs,
                target_effective_at=target_effective_at,
                required_channels=required_channels,
            )
            self._advance_closed_target_locked(float(target_effective_at))
            return snapshot

    def _advance_closed_target_locked(self, target: float) -> None:
        self._validate_target_locked(target)
        self._closed_target_through = target
        prune_before = self.retention_spec.prune_before(
            closed_target_through=target
        )
        retained = [
            record
            for record in self._records
            if record.effective_until > prune_before
        ]
        self._pruned_record_count += len(self._records) - len(retained)
        self._records = retained
        self._by_id = {record.context_id: record for record in retained}

    def _validate_target_locked(self, target: float) -> None:
        if not math.isfinite(target):
            raise ValueError("Context target time must be finite.")
        if (
            self._closed_target_through is not None
            and target < self._closed_target_through
        ):
            raise ValueError("Context target time cannot move behind the closed window.")

    def validate_target(self, target_effective_at: float) -> None:
        with self._lock:
            self._validate_target_locked(float(target_effective_at))

    def advance_closed_target(self, target_effective_at: float) -> None:
        """Close an evaluation target even when no context channel was requested."""

        with self._lock:
            self._advance_closed_target_locked(float(target_effective_at))

    def retention_snapshot(self) -> TimeWindowStoreSnapshot:
        with self._lock:
            return TimeWindowStoreSnapshot(
                retention_spec_hash=self.retention_spec.semantic_content_hash,
                closed_target_through=self._closed_target_through,
                retained_record_count=len(self._records),
                pruned_record_count=self._pruned_record_count,
                producer_epoch_tombstone_count=len(self._last_sequence),
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def project_composition_contexts(
    store: CompositionContextStore, cut: ControlCut
) -> CompositionContextProjection:
    return CompositionContextProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        contexts=store.records_at_or_before(cut),
    )


def _producer_epoch_snapshot_hash(snapshot: ProducerEpochSnapshot) -> str:
    return canonical_content_hash(
        {
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "active_epoch_by_producer": snapshot.active_epoch_by_producer,
            "activation_id_by_producer": snapshot.activation_id_by_producer,
        },
        domain="dexteleop.composition_context_epoch_snapshot.v1",
    )


def build_composition_context_snapshot(
    projection: CompositionContextProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    target_effective_at: float,
    required_channels: tuple[str, ...],
) -> CompositionContextSnapshot:
    target = _finite(target_effective_at, name="target_effective_at")
    channels = tuple(
        _required(value, name="required_channel") for value in required_channels
    )
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("required_channels must be non-empty and unique.")
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError("Context projection and epochs belong to different domains.")
    if (
        producer_epochs.cutoff_admission_sequence
        != projection.cutoff_admission_sequence
    ):
        raise ValueError("Context projection and epochs must share one cut.")
    selected: dict[str, CompositionContextRecord] = {}
    for record in projection.contexts:
        if record.context_channel not in channels:
            continue
        if (
            producer_epochs.active_epoch_by_producer.get(record.producer_id)
            != record.producer_epoch
        ):
            continue
        if not record.effective_from <= target < record.effective_until:
            continue
        current = selected.get(record.context_channel)
        rank = (record.effective_from, record.sequence, record.context_id)
        if current is None or rank > (
            current.effective_from,
            current.sequence,
            current.context_id,
        ):
            selected[record.context_channel] = record
    selected_ids = {
        channel: selected[channel].context_id
        for channel in channels
        if channel in selected
    }
    selected_hashes = {
        channel: selected[channel].semantic_content_hash
        for channel in channels
        if channel in selected
    }
    unavailable = tuple(
        channel
        for channel in channels
        if channel in selected and selected[channel].availability != "available"
    )
    missing = tuple(
        channel
        for channel in channels
        if channel not in selected or channel in unavailable
    )
    epoch_hash = _producer_epoch_snapshot_hash(producer_epochs)
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_effective_at": target,
        "required_channels": channels,
        "selected_context_id_by_channel": selected_ids,
        "selected_context_hash_by_channel": selected_hashes,
        "missing_channels": missing,
        "unavailable_channels": unavailable,
        "producer_epoch_snapshot_hash": epoch_hash,
    }
    digest = canonical_content_hash(
        payload, domain=COMPOSITION_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1
    )
    return CompositionContextSnapshot(
        context_snapshot_id=f"composition-context-snapshot:{digest}",
        selected_context_by_channel=selected,
        semantic_content_hash=digest,
        **payload,
    )


# Historical public name retained as an identity alias.  New generic code
# should import CompositionContext.
PoseMixingContext = CompositionContext
POSE_MIXING_CONTEXT_KINDS_V1 = COMPOSITION_CONTEXT_KINDS_V1


__all__ = [
    "COMPOSITION_CONTEXT_HASH_DOMAIN_V1",
    "COMPOSITION_CONTEXT_KINDS_V1",
    "COMPOSITION_CONTEXT_RECORD_HASH_DOMAIN_V1",
    "COMPOSITION_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1",
    "POSE_MIXING_CONTEXT_KINDS_V1",
    "CompositionContext",
    "CompositionContextAdmissionError",
    "CompositionContextAdmissionReceipt",
    "CompositionContextDraft",
    "CompositionContextProjection",
    "CompositionContextRecord",
    "CompositionContextSnapshot",
    "CompositionContextStore",
    "PoseMixingContext",
    "build_composition_context_snapshot",
    "composition_context_semantic_payload",
    "project_composition_contexts",
]

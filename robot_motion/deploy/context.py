"""Receiver-admitted policy context records and deterministic snapshots.

Policy contexts carry slow-policy guidance to downstream policy nodes.  They
are not action proposals and can never enter authority selection directly.
Runtime payload handles remain outside this module; records here contain only
immutable semantic payloads and stable blob references.
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


POLICY_CONTEXT_KINDS_V1 = frozenset(
    {
        "semantic_subgoal",
        "skill_state",
        "trajectory_plan",
        "constraint",
        "artifact_bound_latent",
    }
)
POLICY_CONTEXT_HASH_DOMAIN_V1 = "dexteleop.policy_context.semantic.v1"
POLICY_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.policy_context_snapshot.v1"


class PolicyContextAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise PolicyContextAdmissionError(
            "invalid_field", f"{field_name} must be non-empty."
        )
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PolicyContextAdmissionError(
            "invalid_time", f"{field_name} must be finite."
        )
    return normalized


@dataclass(frozen=True)
class PolicyContextDraft:
    """Producer-side submission without authoritative ``available_at``."""

    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    policy_node_id: str
    context_channel: str
    context_kind: str
    semantic_payload: Mapping[str, Any]
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    context_contract_hash: str
    observation_cut_id: str
    inference_id: str | None = None
    policy_output_id: str | None = None
    parent_context_ids: tuple[str, ...] = ()
    blob_refs: tuple[str, ...] = ()
    producer_artifact_hash: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyContextRecord:
    context_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    policy_node_id: str
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
    context_contract_hash: str
    observation_cut_id: str
    inference_id: str | None
    policy_output_id: str | None
    parent_context_ids: tuple[str, ...]
    blob_refs: tuple[str, ...]
    producer_artifact_hash: str | None
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyContextAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    context_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    context: PolicyContextRecord | None = None


@dataclass(frozen=True)
class PolicyContextProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    contexts: tuple[PolicyContextRecord, ...]


@dataclass(frozen=True)
class PolicyContextSnapshot:
    context_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    required_channels: tuple[str, ...]
    selected_context_id_by_channel: Mapping[str, str]
    selected_context_hash_by_channel: Mapping[str, str]
    missing_channels: tuple[str, ...]
    producer_epoch_snapshot_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class _PreparedContext:
    identity: RuntimeFactIdentity
    draft: PolicyContextDraft
    context_kind: str
    semantic_payload: Mapping[str, Any]
    audit_metadata: Mapping[str, Any]
    produced_at: float
    effective_from: float
    effective_until: float
    parent_context_ids: tuple[str, ...]
    blob_refs: tuple[str, ...]


def _prepare(draft: PolicyContextDraft) -> _PreparedContext:
    session = _required(draft.runtime_session_id, field_name="runtime_session_id")
    domain = _required(draft.control_domain_id, field_name="control_domain_id")
    producer = _required(draft.producer_id, field_name="producer_id")
    _required(draft.policy_node_id, field_name="policy_node_id")
    _required(draft.context_channel, field_name="context_channel")
    kind = _required(draft.context_kind, field_name="context_kind")
    if kind not in POLICY_CONTEXT_KINDS_V1:
        raise PolicyContextAdmissionError(
            "unsupported_context_kind", f"Unsupported policy context kind {kind!r}."
        )
    _required(draft.context_contract_hash, field_name="context_contract_hash")
    _required(draft.observation_cut_id, field_name="observation_cut_id")
    _required(
        draft.produced_clock_domain_id, field_name="produced_clock_domain_id"
    )
    _required(
        draft.effective_clock_domain_id, field_name="effective_clock_domain_id"
    )
    if int(draft.producer_epoch) < 0 or int(draft.sequence) < 0:
        raise PolicyContextAdmissionError(
            "invalid_identity", "producer_epoch and sequence must be non-negative."
        )
    produced_at = _finite(draft.produced_at, field_name="produced_at")
    effective_from = _finite(draft.effective_from, field_name="effective_from")
    effective_until = _finite(draft.effective_until, field_name="effective_until")
    if effective_until <= effective_from:
        raise PolicyContextAdmissionError(
            "invalid_effective_interval",
            "Policy context interval must be [effective_from, effective_until).",
        )
    for field_name in ("inference_id", "policy_output_id", "producer_artifact_hash"):
        value = getattr(draft, field_name)
        if value is not None:
            _required(value, field_name=field_name)
    parent_ids = tuple(_required(value, field_name="parent_context_id") for value in draft.parent_context_ids)
    blob_refs = tuple(_required(value, field_name="blob_ref") for value in draft.blob_refs)
    if len(parent_ids) != len(set(parent_ids)) or len(blob_refs) != len(set(blob_refs)):
        raise PolicyContextAdmissionError(
            "invalid_field", "Context dependency and blob references must be unique."
        )
    if kind == "artifact_bound_latent" and draft.producer_artifact_hash is None:
        raise PolicyContextAdmissionError(
            "artifact_hash_required",
            "artifact_bound_latent requires producer_artifact_hash.",
        )
    try:
        semantic_payload = deep_freeze(dict(draft.semantic_payload))
        audit_metadata = deep_freeze(dict(draft.audit_metadata))
    except TypeError as exc:
        raise PolicyContextAdmissionError("unsupported_payload", str(exc)) from exc
    identity = RuntimeFactIdentity(
        runtime_session_id=session,
        control_domain_id=domain,
        producer_id=producer,
        producer_epoch=int(draft.producer_epoch),
        sequence=int(draft.sequence),
        namespace="policy_context",
    )
    return _PreparedContext(
        identity=identity,
        draft=draft,
        context_kind=kind,
        semantic_payload=semantic_payload,
        audit_metadata=audit_metadata,
        produced_at=produced_at,
        effective_from=effective_from,
        effective_until=effective_until,
        parent_context_ids=parent_ids,
        blob_refs=blob_refs,
    )


def _semantic_payload(
    prepared: _PreparedContext,
    *,
    available_at: float,
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
        "policy_node_id": draft.policy_node_id,
        "context_channel": draft.context_channel,
        "context_kind": prepared.context_kind,
        "semantic_payload": prepared.semantic_payload,
        "produced_at": prepared.produced_at,
        "produced_clock_domain_id": draft.produced_clock_domain_id,
        "available_at": available_at,
        "available_clock_domain_id": draft.effective_clock_domain_id,
        "control_admission_sequence": control_admission_sequence,
        "effective_from": prepared.effective_from,
        "effective_until": prepared.effective_until,
        "effective_clock_domain_id": draft.effective_clock_domain_id,
        "context_contract_hash": draft.context_contract_hash,
        "observation_cut_id": draft.observation_cut_id,
        "inference_id": draft.inference_id,
        "policy_output_id": draft.policy_output_id,
        "parent_context_ids": prepared.parent_context_ids,
        "blob_refs": prepared.blob_refs,
        "producer_artifact_hash": draft.producer_artifact_hash,
    }


def policy_context_semantic_payload(context: PolicyContextRecord) -> dict[str, Any]:
    return {
        "context_id": context.context_id,
        "runtime_session_id": context.runtime_session_id,
        "control_domain_id": context.control_domain_id,
        "producer_id": context.producer_id,
        "producer_epoch": context.producer_epoch,
        "sequence": context.sequence,
        "policy_node_id": context.policy_node_id,
        "context_channel": context.context_channel,
        "context_kind": context.context_kind,
        "semantic_payload": context.semantic_payload,
        "produced_at": context.produced_at,
        "produced_clock_domain_id": context.produced_clock_domain_id,
        "available_at": context.available_at,
        "available_clock_domain_id": context.available_clock_domain_id,
        "control_admission_sequence": context.control_admission_sequence,
        "effective_from": context.effective_from,
        "effective_until": context.effective_until,
        "effective_clock_domain_id": context.effective_clock_domain_id,
        "context_contract_hash": context.context_contract_hash,
        "observation_cut_id": context.observation_cut_id,
        "inference_id": context.inference_id,
        "policy_output_id": context.policy_output_id,
        "parent_context_ids": context.parent_context_ids,
        "blob_refs": context.blob_refs,
        "producer_artifact_hash": context.producer_artifact_hash,
    }


class PolicyContextStore:
    """Bounded context store sharing the control-domain admission prefix."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        capacity: int,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("PolicyContextStore capacity must be positive.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Context store and producer epochs must share a coordinator.")
        self.runtime_session_id = _required(
            runtime_session_id, field_name="runtime_session_id"
        )
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Context store session must match the control clock session.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.capacity = int(capacity)
        self._records: list[PolicyContextRecord] = []
        self._by_id: dict[str, PolicyContextRecord] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}
        self._lock = threading.RLock()

    def admit(self, draft: PolicyContextDraft) -> PolicyContextAdmissionReceipt:
        context_id: str | None = None
        try:
            prepared = _prepare(draft)
            context_id = prepared.identity.fact_id
            if prepared.identity.runtime_session_id != self.runtime_session_id:
                raise PolicyContextAdmissionError(
                    "session_mismatch", "Draft belongs to another runtime session."
                )
            if prepared.identity.control_domain_id != self.control_domain_id:
                raise PolicyContextAdmissionError(
                    "domain_mismatch", "Draft belongs to another control domain."
                )
            if draft.effective_clock_domain_id != self.coordinator.clock_domain.clock_domain_id:
                raise PolicyContextAdmissionError(
                    "clock_domain_mismatch",
                    "Context effective interval must use the receiver control clock.",
                )

            def build(available_at: float, admission_sequence: int) -> PolicyContextRecord:
                payload = _semantic_payload(
                    prepared,
                    available_at=available_at,
                    control_admission_sequence=admission_sequence,
                )
                return PolicyContextRecord(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        payload, domain=POLICY_CONTEXT_HASH_DOMAIN_V1
                    ),
                    audit_metadata=prepared.audit_metadata,
                )

            record = self.coordinator.commit(store=self, build=build)
            return PolicyContextAdmissionReceipt(
                accepted=True,
                context_id=record.context_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                context=record,
            )
        except (PolicyContextAdmissionError, ProducerEpochError, TypeError, ValueError) as exc:
            code = exc.code if isinstance(exc, PolicyContextAdmissionError) else "validation_failed"
            return PolicyContextAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                context_id=context_id,
            )

    def _insert_admitted(self, record: PolicyContextRecord) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=record.producer_id,
                    producer_epoch=record.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise PolicyContextAdmissionError(
                    "inactive_producer_epoch", str(exc)
                ) from exc
            if record.context_id in self._by_id:
                raise PolicyContextAdmissionError(
                    "duplicate", "Policy context ID is already admitted."
                )
            sequence_key = (record.producer_id, record.producer_epoch)
            previous = self._last_sequence.get(sequence_key)
            if previous is not None and record.sequence <= previous:
                raise PolicyContextAdmissionError(
                    "sequence_rollback",
                    f"Producer sequence must increase beyond {previous}.",
                )
            if len(self._records) >= self.capacity:
                raise PolicyContextAdmissionError(
                    "capacity_exhausted", "PolicyContextStore is full."
                )
            self._records.append(record)
            self._by_id[record.context_id] = record
            self._last_sequence[sequence_key] = record.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[PolicyContextRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(
                record for record in self._records if admitted_at_or_before(record, cut)
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def project_policy_contexts(
    store: PolicyContextStore, cut: ControlCut
) -> PolicyContextProjection:
    return PolicyContextProjection(
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
        domain="dexteleop.policy_context_epoch_snapshot.v1",
    )


def build_policy_context_snapshot(
    projection: PolicyContextProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    target_effective_at: float,
    required_channels: tuple[str, ...],
) -> PolicyContextSnapshot:
    target = _finite(target_effective_at, field_name="target_effective_at")
    channels = tuple(_required(value, field_name="required_channel") for value in required_channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("required_channels must be non-empty and unique.")
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError("Context projection and producer epochs belong to different domains.")
    if producer_epochs.cutoff_admission_sequence != projection.cutoff_admission_sequence:
        raise ValueError("Context projection and producer epochs must share one committed cut.")

    selected: dict[str, PolicyContextRecord] = {}
    for record in projection.contexts:
        if record.context_channel not in channels:
            continue
        if producer_epochs.active_epoch_by_producer.get(record.producer_id) != record.producer_epoch:
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

    selected_ids = {channel: selected[channel].context_id for channel in channels if channel in selected}
    selected_hashes = {
        channel: selected[channel].semantic_content_hash
        for channel in channels
        if channel in selected
    }
    missing = tuple(channel for channel in channels if channel not in selected)
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
        "producer_epoch_snapshot_hash": epoch_hash,
    }
    digest = canonical_content_hash(payload, domain=POLICY_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1)
    return PolicyContextSnapshot(
        context_snapshot_id=f"policy-context-snapshot:{digest}",
        semantic_content_hash=digest,
        **payload,
    )


__all__ = [
    "POLICY_CONTEXT_HASH_DOMAIN_V1",
    "POLICY_CONTEXT_KINDS_V1",
    "POLICY_CONTEXT_SNAPSHOT_HASH_DOMAIN_V1",
    "PolicyContextAdmissionError",
    "PolicyContextAdmissionReceipt",
    "PolicyContextDraft",
    "PolicyContextProjection",
    "PolicyContextRecord",
    "PolicyContextSnapshot",
    "PolicyContextStore",
    "build_policy_context_snapshot",
    "policy_context_semantic_payload",
    "project_policy_contexts",
]

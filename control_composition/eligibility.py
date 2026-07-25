"""Receiver-admitted source eligibility facts for control composition.

Action validity and source liveness are intentionally separate.  A source
state may invalidate an otherwise unexpired action, but it never grants motion
authority, executes HOLD, cancels work, or dispatches a command.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
    ProducerEpochSnapshot,
)

from .retention import TimeWindowRetentionSpec, TimeWindowStoreSnapshot

if TYPE_CHECKING:
    from .composition.core import CompositionSpec


COMPOSITION_SOURCE_ELIGIBILITY_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_spec.v1"
)
COMPOSITION_SOURCE_ELIGIBILITY_IDENTITY_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_identity.v1"
)
COMPOSITION_SOURCE_ELIGIBILITY_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_state.v1"
)
COMPOSITION_SOURCE_ELIGIBILITY_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_snapshot.v1"
)
COMPOSITION_SOURCE_STATUSES_V1 = frozenset(
    {"available", "paused", "fault", "disconnected", "rearming"}
)


class CompositionSourceEligibilityAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_field",
            f"{name} must be non-empty.",
        )
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_time",
            f"{name} must be finite.",
        )
    return normalized


@dataclass(frozen=True)
class CompositionSourceEligibilitySpec:
    """Opt-in binding from composition roles to stable source slots/producers."""

    spec_id: str
    composition_spec_hash: str
    source_slot_by_role: Mapping[str, str]
    producer_id_by_source_slot: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _required(self.spec_id, name="spec_id"))
        object.__setattr__(
            self,
            "composition_spec_hash",
            _required(
                self.composition_spec_hash,
                name="composition_spec_hash",
            ),
        )
        role_bindings = {
            _required(role, name="input_role"): _required(
                source_slot,
                name=f"source_slot_by_role[{role!r}]",
            )
            for role, source_slot in dict(self.source_slot_by_role).items()
        }
        producer_bindings = {
            _required(source_slot, name="source_slot_id"): _required(
                producer_id,
                name=f"producer_id_by_source_slot[{source_slot!r}]",
            )
            for source_slot, producer_id in dict(
                self.producer_id_by_source_slot
            ).items()
        }
        if not role_bindings:
            raise ValueError("source_slot_by_role must be non-empty.")
        referenced_slots = set(role_bindings.values())
        if set(producer_bindings) != referenced_slots:
            raise ValueError(
                "producer_id_by_source_slot must bind every referenced source "
                "slot exactly."
            )
        object.__setattr__(
            self,
            "source_slot_by_role",
            MappingProxyType(role_bindings),
        )
        object.__setattr__(
            self,
            "producer_id_by_source_slot",
            MappingProxyType(producer_bindings),
        )

    def validate_composition_spec(self, spec: CompositionSpec) -> None:
        if self.composition_spec_hash != spec.semantic_content_hash:
            raise ValueError(
                "Source eligibility spec is bound to another composition spec."
            )
        if set(self.source_slot_by_role) != set(spec.input_roles):
            raise ValueError(
                "Source eligibility spec must bind every composition role exactly."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "composition_spec_hash": self.composition_spec_hash,
                "source_slot_by_role": self.source_slot_by_role,
                "producer_id_by_source_slot": (
                    self.producer_id_by_source_slot
                ),
            },
            domain=COMPOSITION_SOURCE_ELIGIBILITY_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class CompositionSourceEligibilityDraft:
    runtime_session_id: str
    control_domain_id: str
    source_slot_id: str
    producer_id: str
    producer_epoch: int
    status_epoch: int
    sequence: int
    status: str
    observed_at: float
    observed_clock_domain_id: str
    effective_from: float
    lease_expires_at: float
    effective_clock_domain_id: str
    rearm_required: bool = False
    fault_code: str | None = None
    rearm_receipt_id: str | None = None
    rearm_receipt_hash: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionSourceEligibilityState:
    source_eligibility_state_id: str
    runtime_session_id: str
    control_domain_id: str
    source_slot_id: str
    producer_id: str
    producer_epoch: int
    status_epoch: int
    sequence: int
    status: str
    observed_at: float
    observed_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float
    lease_expires_at: float
    effective_clock_domain_id: str
    rearm_required: bool
    fault_code: str | None
    rearm_receipt_id: str | None
    rearm_receipt_hash: str | None
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionSourceEligibilityAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    source_eligibility_state_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    source_state: CompositionSourceEligibilityState | None = None


@dataclass(frozen=True)
class CompositionSourceEligibilityProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    records: tuple[CompositionSourceEligibilityState, ...]


@dataclass(frozen=True)
class CompositionSourceEligibilitySnapshot:
    source_eligibility_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    source_eligibility_spec_hash: str
    selected_state_by_role: Mapping[
        str,
        CompositionSourceEligibilityState,
    ]
    ineligible_reason_by_role: Mapping[str, str]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_state_by_role",
            MappingProxyType(dict(self.selected_state_by_role)),
        )
        object.__setattr__(
            self,
            "ineligible_reason_by_role",
            MappingProxyType(dict(self.ineligible_reason_by_role)),
        )

    def is_role_eligible(self, role: str) -> bool:
        return (
            role in self.selected_state_by_role
            and role not in self.ineligible_reason_by_role
        )


@dataclass(frozen=True)
class _PreparedState:
    state_id: str
    draft: CompositionSourceEligibilityDraft
    observed_at: float
    effective_from: float
    lease_expires_at: float
    audit_metadata: Mapping[str, Any]


def _prepare_state(
    draft: CompositionSourceEligibilityDraft,
) -> _PreparedState:
    session = _required(draft.runtime_session_id, name="runtime_session_id")
    domain = _required(draft.control_domain_id, name="control_domain_id")
    slot = _required(draft.source_slot_id, name="source_slot_id")
    producer = _required(draft.producer_id, name="producer_id")
    observed_clock = _required(
        draft.observed_clock_domain_id,
        name="observed_clock_domain_id",
    )
    effective_clock = _required(
        draft.effective_clock_domain_id,
        name="effective_clock_domain_id",
    )
    producer_epoch = int(draft.producer_epoch)
    status_epoch = int(draft.status_epoch)
    sequence = int(draft.sequence)
    if min(producer_epoch, status_epoch, sequence) < 0:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_identity",
            "producer_epoch, status_epoch, and sequence must be non-negative.",
        )
    status = _required(draft.status, name="status")
    if status not in COMPOSITION_SOURCE_STATUSES_V1:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_status",
            f"Unsupported source status {status!r}.",
        )
    observed_at = _finite(draft.observed_at, name="observed_at")
    effective_from = _finite(draft.effective_from, name="effective_from")
    lease_expires_at = _finite(
        draft.lease_expires_at,
        name="lease_expires_at",
    )
    if lease_expires_at <= effective_from:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_effective_interval",
            "Source lease interval must be [effective_from, lease_expires_at).",
        )
    fault_code = (
        None
        if draft.fault_code is None
        else _required(draft.fault_code, name="fault_code")
    )
    if status == "fault" and fault_code is None:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_status",
            "fault status requires fault_code.",
        )
    if status != "fault" and fault_code is not None:
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_status",
            "fault_code is only valid for fault status.",
        )
    receipt_fields = (draft.rearm_receipt_id, draft.rearm_receipt_hash)
    if any(value is None for value in receipt_fields) and any(
        value is not None for value in receipt_fields
    ):
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_rearm_receipt",
            "rearm receipt id and hash must be provided together.",
        )
    rearm_receipt_id = (
        None
        if draft.rearm_receipt_id is None
        else _required(draft.rearm_receipt_id, name="rearm_receipt_id")
    )
    rearm_receipt_hash = (
        None
        if draft.rearm_receipt_hash is None
        else _required(draft.rearm_receipt_hash, name="rearm_receipt_hash")
    )
    if status == "available" and bool(draft.rearm_required):
        raise CompositionSourceEligibilityAdmissionError(
            "invalid_status",
            "available source state cannot remain rearm_required.",
        )
    identity_payload = {
        "runtime_session_id": session,
        "control_domain_id": domain,
        "source_slot_id": slot,
        "producer_id": producer,
        "producer_epoch": producer_epoch,
        "status_epoch": status_epoch,
        "sequence": sequence,
    }
    identity_hash = canonical_content_hash(
        identity_payload,
        domain=COMPOSITION_SOURCE_ELIGIBILITY_IDENTITY_HASH_DOMAIN_V1,
    )
    normalized_draft = CompositionSourceEligibilityDraft(
        runtime_session_id=session,
        control_domain_id=domain,
        source_slot_id=slot,
        producer_id=producer,
        producer_epoch=producer_epoch,
        status_epoch=status_epoch,
        sequence=sequence,
        status=status,
        observed_at=observed_at,
        observed_clock_domain_id=observed_clock,
        effective_from=effective_from,
        lease_expires_at=lease_expires_at,
        effective_clock_domain_id=effective_clock,
        rearm_required=bool(draft.rearm_required),
        fault_code=fault_code,
        rearm_receipt_id=rearm_receipt_id,
        rearm_receipt_hash=rearm_receipt_hash,
        audit_metadata={},
    )
    return _PreparedState(
        state_id=f"composition-source-eligibility:{identity_hash}",
        draft=normalized_draft,
        observed_at=observed_at,
        effective_from=effective_from,
        lease_expires_at=lease_expires_at,
        audit_metadata=deep_freeze(dict(draft.audit_metadata)),
    )


def source_eligibility_state_semantic_payload(
    state: CompositionSourceEligibilityState,
) -> dict[str, Any]:
    return {
        "source_eligibility_state_id": state.source_eligibility_state_id,
        "runtime_session_id": state.runtime_session_id,
        "control_domain_id": state.control_domain_id,
        "source_slot_id": state.source_slot_id,
        "producer_id": state.producer_id,
        "producer_epoch": state.producer_epoch,
        "status_epoch": state.status_epoch,
        "sequence": state.sequence,
        "status": state.status,
        "observed_at": state.observed_at,
        "observed_clock_domain_id": state.observed_clock_domain_id,
        "available_at": state.available_at,
        "available_clock_domain_id": state.available_clock_domain_id,
        "control_admission_sequence": state.control_admission_sequence,
        "effective_from": state.effective_from,
        "lease_expires_at": state.lease_expires_at,
        "effective_clock_domain_id": state.effective_clock_domain_id,
        "rearm_required": state.rearm_required,
        "fault_code": state.fault_code,
        "rearm_receipt_id": state.rearm_receipt_id,
        "rearm_receipt_hash": state.rearm_receipt_hash,
    }


class CompositionSourceEligibilityStore:
    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        retention_spec: TimeWindowRetentionSpec,
    ) -> None:
        if producer_epochs.coordinator is not coordinator:
            raise ValueError(
                "Source eligibility store and epochs must share one coordinator."
            )
        if not isinstance(retention_spec, TimeWindowRetentionSpec):
            raise TypeError("retention_spec must be a TimeWindowRetentionSpec.")
        self.runtime_session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        if (
            self.runtime_session_id
            != coordinator.clock_domain.runtime_session_id
        ):
            raise ValueError(
                "Source eligibility store session must match the control clock."
            )
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.retention_spec = retention_spec
        self._records: list[CompositionSourceEligibilityState] = []
        self._by_id: dict[str, CompositionSourceEligibilityState] = {}
        self._last_state_by_key: dict[
            tuple[str, int, str],
            CompositionSourceEligibilityState,
        ] = {}
        self._rearm_latched_by_source: dict[tuple[str, str], bool] = {}
        self._closed_target_through: float | None = None
        self._pruned_record_count = 0
        self._lock = threading.RLock()

    def admit(
        self,
        draft: CompositionSourceEligibilityDraft,
    ) -> CompositionSourceEligibilityAdmissionReceipt:
        state_id: str | None = None
        try:
            prepared = _prepare_state(draft)
            state_id = prepared.state_id
            normalized = prepared.draft
            if normalized.runtime_session_id != self.runtime_session_id:
                raise CompositionSourceEligibilityAdmissionError(
                    "session_mismatch",
                    "Source state belongs to another session.",
                )
            if normalized.control_domain_id != self.control_domain_id:
                raise CompositionSourceEligibilityAdmissionError(
                    "domain_mismatch",
                    "Source state belongs to another control domain.",
                )
            if (
                normalized.effective_clock_domain_id
                != self.coordinator.clock_domain.clock_domain_id
            ):
                raise CompositionSourceEligibilityAdmissionError(
                    "clock_domain_mismatch",
                    "Source lease must use the receiver control clock.",
                )

            def build(
                available_at: float,
                admission_sequence: int,
            ) -> CompositionSourceEligibilityState:
                try:
                    self.retention_spec.validate_admitted_interval(
                        available_at=available_at,
                        effective_from=prepared.effective_from,
                        effective_until=prepared.lease_expires_at,
                    )
                except ValueError as exc:
                    raise CompositionSourceEligibilityAdmissionError(
                        "retention_window_violation",
                        str(exc),
                    ) from exc
                payload = {
                    "source_eligibility_state_id": prepared.state_id,
                    "runtime_session_id": normalized.runtime_session_id,
                    "control_domain_id": normalized.control_domain_id,
                    "source_slot_id": normalized.source_slot_id,
                    "producer_id": normalized.producer_id,
                    "producer_epoch": normalized.producer_epoch,
                    "status_epoch": normalized.status_epoch,
                    "sequence": normalized.sequence,
                    "status": normalized.status,
                    "observed_at": prepared.observed_at,
                    "observed_clock_domain_id": (
                        normalized.observed_clock_domain_id
                    ),
                    "available_at": available_at,
                    "available_clock_domain_id": (
                        self.coordinator.clock_domain.clock_domain_id
                    ),
                    "control_admission_sequence": admission_sequence,
                    "effective_from": prepared.effective_from,
                    "lease_expires_at": prepared.lease_expires_at,
                    "effective_clock_domain_id": (
                        normalized.effective_clock_domain_id
                    ),
                    "rearm_required": normalized.rearm_required,
                    "fault_code": normalized.fault_code,
                    "rearm_receipt_id": normalized.rearm_receipt_id,
                    "rearm_receipt_hash": normalized.rearm_receipt_hash,
                }
                return CompositionSourceEligibilityState(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        payload,
                        domain=(
                            COMPOSITION_SOURCE_ELIGIBILITY_STATE_HASH_DOMAIN_V1
                        ),
                    ),
                    audit_metadata=prepared.audit_metadata,
                )

            state = self.coordinator.commit(store=self, build=build)
            return CompositionSourceEligibilityAdmissionReceipt(
                accepted=True,
                source_eligibility_state_id=state.source_eligibility_state_id,
                available_at=state.available_at,
                available_clock_domain_id=state.available_clock_domain_id,
                control_admission_sequence=state.control_admission_sequence,
                semantic_content_hash=state.semantic_content_hash,
                source_state=state,
            )
        except (
            CompositionSourceEligibilityAdmissionError,
            ProducerEpochError,
            TypeError,
            ValueError,
        ) as exc:
            code = (
                exc.code
                if isinstance(
                    exc,
                    CompositionSourceEligibilityAdmissionError,
                )
                else "validation_failed"
            )
            return CompositionSourceEligibilityAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                source_eligibility_state_id=state_id,
            )

    def _insert_admitted(
        self,
        state: CompositionSourceEligibilityState,
    ) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=state.producer_id,
                    producer_epoch=state.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise CompositionSourceEligibilityAdmissionError(
                    "inactive_producer_epoch",
                    str(exc),
                ) from exc
            if state.source_eligibility_state_id in self._by_id:
                raise CompositionSourceEligibilityAdmissionError(
                    "duplicate",
                    "Source eligibility state is already admitted.",
                )
            key = (
                state.producer_id,
                state.producer_epoch,
                state.source_slot_id,
            )
            source_key = (state.producer_id, state.source_slot_id)
            previous = self._last_state_by_key.get(key)
            if previous is not None:
                if state.status_epoch < previous.status_epoch:
                    raise CompositionSourceEligibilityAdmissionError(
                        "status_epoch_rollback",
                        "Source status epoch cannot move backwards.",
                    )
                if state.status_epoch == previous.status_epoch:
                    if state.sequence <= previous.sequence:
                        code = (
                            "duplicate"
                            if state.sequence == previous.sequence
                            else "sequence_rollback"
                        )
                        raise CompositionSourceEligibilityAdmissionError(
                            code,
                            "Source status sequence must increase.",
                        )
                    stable_fields = (
                        "status",
                        "rearm_required",
                        "fault_code",
                        "rearm_receipt_id",
                        "rearm_receipt_hash",
                    )
                    if any(
                        getattr(state, name) != getattr(previous, name)
                        for name in stable_fields
                    ):
                        raise CompositionSourceEligibilityAdmissionError(
                            "status_epoch_transition_required",
                            "Source status transitions require a newer status epoch.",
                        )
            rearm_latched = self._rearm_latched_by_source.get(
                source_key,
                False,
            )
            if (
                rearm_latched
                and state.status != "available"
                and not state.rearm_required
            ):
                raise CompositionSourceEligibilityAdmissionError(
                    "rearm_latch_cannot_clear",
                    "A latched source must remain rearm_required until typed "
                    "recovery becomes available.",
                )
            if (
                rearm_latched
                and state.status == "available"
                and (
                    state.rearm_receipt_id is None
                    or state.rearm_receipt_hash is None
                )
            ):
                raise CompositionSourceEligibilityAdmissionError(
                    "rearm_receipt_required",
                    "A rearm-required source needs typed receipt lineage "
                    "before becoming available.",
                )
            self._records.append(state)
            self._by_id[state.source_eligibility_state_id] = state
            self._last_state_by_key[key] = state
            if state.status == "available":
                self._rearm_latched_by_source[source_key] = False
            elif state.rearm_required:
                self._rearm_latched_by_source[source_key] = True

    def records_at_or_before(
        self,
        cut: ControlCut,
    ) -> tuple[CompositionSourceEligibilityState, ...]:
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
        spec: CompositionSourceEligibilitySpec,
        composition_spec: CompositionSpec,
        target_effective_at: float,
    ) -> CompositionSourceEligibilitySnapshot:
        with self._lock:
            snapshot = build_composition_source_eligibility_snapshot(
                CompositionSourceEligibilityProjection(
                    control_domain_id=cut.control_domain_id,
                    cutoff_at=cut.cutoff_at,
                    cutoff_admission_sequence=cut.cutoff_admission_sequence,
                    records=tuple(
                        record
                        for record in self._records
                        if admitted_at_or_before(record, cut)
                    ),
                ),
                producer_epochs=producer_epochs,
                spec=spec,
                composition_spec=composition_spec,
                target_effective_at=target_effective_at,
            )
            self._advance_closed_target_locked(float(target_effective_at))
            return snapshot

    def _advance_closed_target_locked(self, target: float) -> None:
        self._validate_target_locked(target)
        self._closed_target_through = target
        prune_before = self.retention_spec.prune_before(
            closed_target_through=target
        )
        latest_ids = {
            state.source_eligibility_state_id
            for state in self._last_state_by_key.values()
        }
        retained = [
            state
            for state in self._records
            if (
                state.lease_expires_at > prune_before
                or state.source_eligibility_state_id in latest_ids
            )
        ]
        self._pruned_record_count += len(self._records) - len(retained)
        self._records = retained
        self._by_id = {
            state.source_eligibility_state_id: state for state in retained
        }

    def _validate_target_locked(self, target: float) -> None:
        if not math.isfinite(target):
            raise ValueError("Source eligibility target time must be finite.")
        if (
            self._closed_target_through is not None
            and target < self._closed_target_through
        ):
            raise ValueError(
                "Source eligibility target cannot move behind the closed window."
            )

    def validate_target(self, target_effective_at: float) -> None:
        with self._lock:
            self._validate_target_locked(float(target_effective_at))

    def retention_snapshot(self) -> TimeWindowStoreSnapshot:
        with self._lock:
            return TimeWindowStoreSnapshot(
                retention_spec_hash=self.retention_spec.semantic_content_hash,
                closed_target_through=self._closed_target_through,
                retained_record_count=len(self._records),
                pruned_record_count=self._pruned_record_count,
                producer_epoch_tombstone_count=len(self._last_state_by_key),
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def project_composition_source_eligibility(
    store: CompositionSourceEligibilityStore,
    cut: ControlCut,
) -> CompositionSourceEligibilityProjection:
    return CompositionSourceEligibilityProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        records=store.records_at_or_before(cut),
    )


def build_composition_source_eligibility_snapshot(
    projection: CompositionSourceEligibilityProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    spec: CompositionSourceEligibilitySpec,
    composition_spec: CompositionSpec,
    target_effective_at: float,
) -> CompositionSourceEligibilitySnapshot:
    target = _finite(target_effective_at, name="target_effective_at")
    spec.validate_composition_spec(composition_spec)
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError(
            "Source eligibility projection and epochs use different domains."
        )
    if (
        producer_epochs.cutoff_admission_sequence
        != projection.cutoff_admission_sequence
    ):
        raise ValueError(
            "Source eligibility projection and epochs must share one cut."
        )
    selected_by_role: dict[str, CompositionSourceEligibilityState] = {}
    reasons: dict[str, str] = {}
    for role in composition_spec.input_roles:
        source_slot = spec.source_slot_by_role[role]
        producer_id = spec.producer_id_by_source_slot[source_slot]
        candidates = tuple(
            state
            for state in projection.records
            if (
                state.source_slot_id == source_slot
                and state.producer_id == producer_id
                and producer_epochs.active_epoch_by_producer.get(producer_id)
                == state.producer_epoch
                and state.effective_from <= target
            )
        )
        if not candidates:
            reasons[role] = "source_state_missing"
            continue
        state = max(
            candidates,
            key=lambda item: (
                item.status_epoch,
                item.sequence,
                item.control_admission_sequence,
                item.source_eligibility_state_id,
            ),
        )
        selected_by_role[role] = state
        if state.status != "available":
            reasons[role] = f"source_status:{state.status}"
        elif state.rearm_required:
            reasons[role] = "source_rearm_required"
        elif not target < state.lease_expires_at:
            reasons[role] = "source_lease_expired"
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_effective_at": target,
        "source_eligibility_spec_hash": spec.semantic_content_hash,
        "selected_state_ids_by_role": {
            role: selected_by_role[role].source_eligibility_state_id
            for role in composition_spec.input_roles
            if role in selected_by_role
        },
        "selected_state_hashes_by_role": {
            role: selected_by_role[role].semantic_content_hash
            for role in composition_spec.input_roles
            if role in selected_by_role
        },
        "ineligible_reason_by_role": {
            role: reasons[role]
            for role in composition_spec.input_roles
            if role in reasons
        },
    }
    digest = canonical_content_hash(
        payload,
        domain=COMPOSITION_SOURCE_ELIGIBILITY_SNAPSHOT_HASH_DOMAIN_V1,
    )
    return CompositionSourceEligibilitySnapshot(
        source_eligibility_snapshot_id=(
            f"composition-source-eligibility-snapshot:{digest}"
        ),
        control_domain_id=projection.control_domain_id,
        cutoff_at=projection.cutoff_at,
        cutoff_admission_sequence=projection.cutoff_admission_sequence,
        target_effective_at=target,
        source_eligibility_spec_hash=spec.semantic_content_hash,
        selected_state_by_role=selected_by_role,
        ineligible_reason_by_role=reasons,
        semantic_content_hash=digest,
    )


__all__ = [
    "COMPOSITION_SOURCE_ELIGIBILITY_IDENTITY_HASH_DOMAIN_V1",
    "COMPOSITION_SOURCE_ELIGIBILITY_SNAPSHOT_HASH_DOMAIN_V1",
    "COMPOSITION_SOURCE_ELIGIBILITY_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SOURCE_ELIGIBILITY_STATE_HASH_DOMAIN_V1",
    "COMPOSITION_SOURCE_STATUSES_V1",
    "CompositionSourceEligibilityAdmissionError",
    "CompositionSourceEligibilityAdmissionReceipt",
    "CompositionSourceEligibilityDraft",
    "CompositionSourceEligibilityProjection",
    "CompositionSourceEligibilitySnapshot",
    "CompositionSourceEligibilitySpec",
    "CompositionSourceEligibilityState",
    "CompositionSourceEligibilityStore",
    "build_composition_source_eligibility_snapshot",
    "project_composition_source_eligibility",
    "source_eligibility_state_semantic_payload",
]

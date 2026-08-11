"""Immutable contracts for receipt-driven allocation publication."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlCut

from control_composition.activation import CompositionActivationScope
from control_composition.composition.context import (
    CompositionContextDraft,
    composition_context_submission_semantic_hash,
    composition_context_submission_semantic_payload,
)

from .contracts import (
    COMPOSITION_ALLOCATION_STATUS_READY,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
)
from .decision import (
    CompositionAllocationDecision,
    CompositionAllocationState,
)
from .override import CompositionAllocationOverrideState


ALLOCATION_PUBLICATION_RUNTIME_READY = "ready"
ALLOCATION_PUBLICATION_RUNTIME_PENDING = "admission_pending"
ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED = "invalidated"

ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY = "attempt_ready"
ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION = "no_decision"
ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED = "basis_coalesced"
ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED = "basis_ignored"
ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW = "accepted_new"
ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING = (
    "already_accepted_same_fact"
)
ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION = "terminal_rejection"
ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN = "outcome_unknown"
ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED = "invalidated"
ALLOCATION_PUBLICATION_OUTCOME_LATE = "late_outcome_ignored"
ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED = "anchor_generation_fenced"
ALLOCATION_PUBLICATION_RUNTIME_STATUSES_DEVELOPMENT = frozenset(
    {
        ALLOCATION_PUBLICATION_RUNTIME_READY,
        ALLOCATION_PUBLICATION_RUNTIME_PENDING,
        ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED,
    }
)
ALLOCATION_PUBLICATION_OUTCOMES_DEVELOPMENT = frozenset(
    {
        ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
        ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION,
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED,
        ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED,
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
        ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
        ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN,
        ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
        ALLOCATION_PUBLICATION_OUTCOME_LATE,
        ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
    }
)

ALLOCATION_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_publication_basis.development"
)
ALLOCATION_PUBLICATION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_publication_control_cut.development"
)
ALLOCATION_PUBLICATION_ATTEMPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_publication_attempt.development"
)
ALLOCATION_PUBLICATION_OUTCOME_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.allocation_publication_outcome.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def allocation_publication_control_cut_semantic_hash(cut: ControlCut) -> str:
    """Hash an allocation evidence cut without preview/closing semantics."""

    if not isinstance(cut, ControlCut):
        raise TypeError("cut must be a ControlCut.")
    return canonical_content_hash(
        {
            "runtime_session_id": cut.runtime_session_id,
            "control_domain_id": cut.control_domain_id,
            "clock_domain_id": cut.clock_domain_id,
            "clock_reset_epoch": cut.clock_reset_epoch,
            "cutoff_at": cut.cutoff_at,
            "cutoff_admission_sequence": cut.cutoff_admission_sequence,
        },
        domain=ALLOCATION_PUBLICATION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT,
    )


@dataclass(frozen=True)
class AllocationPublicationBasis:
    """Immutable accepted evidence selected at one causal cut."""

    allocation_scope_id: str
    control_cut: ControlCut
    activation_scope: CompositionActivationScope
    activity_by_component: Mapping[str, ComponentSourceActivityObservation]
    residual_magnitude_by_component: Mapping[
        str, ComponentResidualMagnitudeObservation
    ]
    requested_effective_from: float
    support_until: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scope_id = _required(self.allocation_scope_id, name="allocation_scope_id")
        if not isinstance(self.control_cut, ControlCut):
            raise TypeError("control_cut must be a ControlCut.")
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        cut = self.control_cut
        activation = self.activation_scope
        if cut.clock_domain_id != activation.clock_domain_id:
            raise ValueError("Allocation basis and activation use another clock.")
        if cut.cutoff_at < activation.activated_at:
            raise ValueError("Allocation basis cut predates its activation.")
        start = _finite(
            self.requested_effective_from,
            name="requested_effective_from",
        )
        support_until = _finite(self.support_until, name="support_until")
        if start < cut.cutoff_at:
            raise ValueError("requested_effective_from cannot predate the basis cut.")
        if support_until <= start:
            raise ValueError("Allocation publication support interval is empty.")
        activity = dict(self.activity_by_component)
        magnitude = dict(self.residual_magnitude_by_component)
        for component_id, observation in activity.items():
            _required(component_id, name="activity_component_id")
            if not isinstance(observation, ComponentSourceActivityObservation):
                raise TypeError("activity values must be activity observations.")
            if observation.component_id != component_id:
                raise ValueError("Activity observation component/key mismatch.")
            if observation.available_at > cut.cutoff_at:
                raise ValueError("Activity observation was unavailable at the cut.")
        for component_id, observation in magnitude.items():
            _required(component_id, name="magnitude_component_id")
            if not isinstance(observation, ComponentResidualMagnitudeObservation):
                raise TypeError(
                    "residual magnitude values must be magnitude observations."
                )
            if observation.component_id != component_id:
                raise ValueError("Magnitude observation component/key mismatch.")
            if observation.available_at > cut.cutoff_at:
                raise ValueError("Magnitude observation was unavailable at the cut.")
        anchor_identities = {
            (
                observation.anchor_receipt_id,
                observation.anchor_receipt_hash,
                observation.anchor_request_generation,
            )
            for observation in magnitude.values()
        }
        if len(anchor_identities) != 1:
            raise ValueError(
                "Allocation publication basis must bind one anchor identity."
            )
        control_cut_hash = allocation_publication_control_cut_semantic_hash(cut)
        payload = {
            "allocation_scope_id": scope_id,
            "control_cut_hash": control_cut_hash,
            "composition_activation_id": activation.activation_id,
            "composition_activation_hash": activation.semantic_content_hash,
            "activity_observation_hash_by_component": {
                component_id: observation.semantic_content_hash
                for component_id, observation in sorted(activity.items())
            },
            "residual_magnitude_hash_by_component": {
                component_id: observation.semantic_content_hash
                for component_id, observation in sorted(magnitude.items())
            },
            "requested_effective_from": start,
            "support_until": support_until,
        }
        object.__setattr__(self, "allocation_scope_id", scope_id)
        object.__setattr__(
            self,
            "activity_by_component",
            MappingProxyType(dict(sorted(activity.items()))),
        )
        object.__setattr__(
            self,
            "residual_magnitude_by_component",
            MappingProxyType(dict(sorted(magnitude.items()))),
        )
        object.__setattr__(self, "requested_effective_from", start)
        object.__setattr__(self, "support_until", support_until)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ALLOCATION_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def basis_id(self) -> str:
        return f"allocation-publication-basis:{self.semantic_content_hash}"

    @property
    def control_cut_hash(self) -> str:
        return allocation_publication_control_cut_semantic_hash(self.control_cut)

    @property
    def ordering_key(self) -> tuple[int, int, float, str]:
        cut = self.control_cut
        return (
            cut.clock_reset_epoch,
            cut.cutoff_admission_sequence,
            cut.cutoff_at,
            self.semantic_content_hash,
        )

    @property
    def residual_anchor_identity(self) -> tuple[str, str, int]:
        first = next(iter(self.residual_magnitude_by_component.values()))
        return (
            first.anchor_receipt_id,
            first.anchor_receipt_hash,
            first.anchor_request_generation,
        )

    @property
    def anchor_request_generation(self) -> int:
        return self.residual_anchor_identity[2]


@dataclass(frozen=True)
class PendingAllocationPublication:
    """One immutable candidate retained byte-for-byte across retries."""

    allocation_scope_id: str
    publication_sequence: int
    activity_basis: AllocationPublicationBasis
    previous_reducer_state_hash: str
    candidate_next_state: CompositionAllocationState
    allocation_decision: CompositionAllocationDecision
    context_draft: CompositionContextDraft
    created_at: float
    context_draft_id: str = field(init=False)
    context_draft_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scope_id = _required(self.allocation_scope_id, name="allocation_scope_id")
        sequence = _nonnegative_int(
            self.publication_sequence,
            name="publication_sequence",
        )
        if not isinstance(self.activity_basis, AllocationPublicationBasis):
            raise TypeError("activity_basis must be AllocationPublicationBasis.")
        if self.activity_basis.allocation_scope_id != scope_id:
            raise ValueError("Publication attempt belongs to another basis scope.")
        previous_hash = _required(
            self.previous_reducer_state_hash,
            name="previous_reducer_state_hash",
        )
        if not isinstance(self.candidate_next_state, CompositionAllocationState):
            raise TypeError("candidate_next_state must be allocation state.")
        if not isinstance(self.allocation_decision, CompositionAllocationDecision):
            raise TypeError("allocation_decision must be allocation decision.")
        decision = self.allocation_decision
        if decision.output_status != COMPOSITION_ALLOCATION_STATUS_READY:
            raise ValueError("Only a ready decision may form a publication attempt.")
        if decision.previous_state_hash != previous_hash:
            raise ValueError("Decision and attempt previous state hashes disagree.")
        if self.candidate_next_state.last_decision_hash != (
            decision.semantic_content_hash
        ):
            raise ValueError("Candidate state does not bind the exact decision.")
        if not isinstance(self.context_draft, CompositionContextDraft):
            raise TypeError("context_draft must be CompositionContextDraft.")
        submission_payload = composition_context_submission_semantic_payload(
            self.context_draft
        )
        if submission_payload["sequence"] != sequence:
            raise ValueError("Context draft uses another publication sequence.")
        context_id = str(submission_payload["context_id"])
        context_hash = composition_context_submission_semantic_hash(
            self.context_draft
        )
        created_at = _finite(self.created_at, name="created_at")
        payload = {
            "allocation_scope_id": scope_id,
            "publication_sequence": sequence,
            "activity_basis_id": self.activity_basis.basis_id,
            "activity_basis_hash": self.activity_basis.semantic_content_hash,
            "control_cut_hash": self.activity_basis.control_cut_hash,
            "composition_activation_id": (
                self.activity_basis.activation_scope.activation_id
            ),
            "composition_activation_hash": (
                self.activity_basis.activation_scope.semantic_content_hash
            ),
            "previous_reducer_state_hash": previous_hash,
            "candidate_next_state_hash": (
                self.candidate_next_state.semantic_content_hash
            ),
            "allocation_decision_id": decision.decision_id,
            "allocation_decision_hash": decision.semantic_content_hash,
            "context_draft_id": context_id,
            "context_draft_hash": context_hash,
            "created_at": created_at,
            "effective_from": self.context_draft.effective_from,
            "effective_until": self.context_draft.effective_until,
        }
        object.__setattr__(self, "allocation_scope_id", scope_id)
        object.__setattr__(self, "publication_sequence", sequence)
        object.__setattr__(self, "previous_reducer_state_hash", previous_hash)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "context_draft_id", context_id)
        object.__setattr__(self, "context_draft_hash", context_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ALLOCATION_PUBLICATION_ATTEMPT_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def attempt_id(self) -> str:
        return f"allocation-publication-attempt:{self.semantic_content_hash}"


@dataclass(frozen=True)
class AllocationPublicationOutcome:
    outcome_kind: str
    allocation_scope_id: str
    runtime_status: str
    occurred_at: float
    committed_reducer_state_hash: str
    reason: str
    decision_id: str | None = None
    decision_hash: str | None = None
    attempt_id: str | None = None
    attempt_hash: str | None = None
    publication_sequence: int | None = None
    receipt_context_id: str | None = None
    receipt_context_hash: str | None = None
    rejection_code: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        kind = _required(self.outcome_kind, name="outcome_kind")
        runtime_status = _required(self.runtime_status, name="runtime_status")
        if kind not in ALLOCATION_PUBLICATION_OUTCOMES_DEVELOPMENT:
            raise ValueError("Unsupported allocation publication outcome kind.")
        if runtime_status not in ALLOCATION_PUBLICATION_RUNTIME_STATUSES_DEVELOPMENT:
            raise ValueError("Unsupported allocation publication runtime status.")
        if (self.decision_id is None) != (self.decision_hash is None):
            raise ValueError("Allocation decision identity must be complete.")
        if (self.attempt_id is None) != (self.attempt_hash is None):
            raise ValueError("Allocation attempt identity must be complete.")
        if (self.receipt_context_id is None) != (
            self.receipt_context_hash is None
        ):
            raise ValueError("Receipt context identity must be complete.")
        if (self.publication_sequence is None) != (self.attempt_id is None):
            raise ValueError("Publication sequence must accompany an attempt.")
        payload = {
            "outcome_kind": kind,
            "allocation_scope_id": _required(
                self.allocation_scope_id,
                name="allocation_scope_id",
            ),
            "runtime_status": runtime_status,
            "occurred_at": _finite(self.occurred_at, name="occurred_at"),
            "committed_reducer_state_hash": _required(
                self.committed_reducer_state_hash,
                name="committed_reducer_state_hash",
            ),
            "reason": _required(self.reason, name="reason"),
            "decision_id": self.decision_id,
            "decision_hash": self.decision_hash,
            "attempt_id": self.attempt_id,
            "attempt_hash": self.attempt_hash,
            "publication_sequence": self.publication_sequence,
            "receipt_context_id": self.receipt_context_id,
            "receipt_context_hash": self.receipt_context_hash,
            "rejection_code": self.rejection_code,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ALLOCATION_PUBLICATION_OUTCOME_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def outcome_id(self) -> str:
        return f"allocation-publication-outcome:{self.semantic_content_hash}"


@dataclass(frozen=True)
class AllocationPublicationRuntimeSnapshot:
    runtime_status: str
    committed_reducer_state: CompositionAllocationState
    last_reserved_sequence: int
    last_terminal_sequence: int | None
    last_accepted_sequence: int | None
    pending_attempt: PendingAllocationPublication | None
    latest_coalesced_basis: AllocationPublicationBasis | None
    pending_retry_count: int
    last_attempt_at: float | None
    submission_in_progress: bool
    invalidation_reason: str | None
    active_anchor_request_generation: int | None = None
    allocation_override_state: CompositionAllocationOverrideState | None = None

__all__ = [
    "ALLOCATION_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_ATTEMPT_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_OUTCOME_HASH_DOMAIN_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_RUNTIME_READY",
    "ALLOCATION_PUBLICATION_RUNTIME_PENDING",
    "ALLOCATION_PUBLICATION_RUNTIME_INVALIDATED",
    "ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY",
    "ALLOCATION_PUBLICATION_OUTCOME_NO_DECISION",
    "ALLOCATION_PUBLICATION_OUTCOME_BASIS_COALESCED",
    "ALLOCATION_PUBLICATION_OUTCOME_BASIS_IGNORED",
    "ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW",
    "ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING",
    "ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION",
    "ALLOCATION_PUBLICATION_OUTCOME_UNKNOWN",
    "ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED",
    "ALLOCATION_PUBLICATION_OUTCOME_LATE",
    "ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED",
    "ALLOCATION_PUBLICATION_OUTCOMES_DEVELOPMENT",
    "ALLOCATION_PUBLICATION_RUNTIME_STATUSES_DEVELOPMENT",
    "AllocationPublicationBasis",
    "AllocationPublicationOutcome",
    "AllocationPublicationRuntimeSnapshot",
    "PendingAllocationPublication",
    "allocation_publication_control_cut_semantic_hash",
]

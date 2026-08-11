"""Operation rule, proposal, and reducer-state trigger contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from ._validation import finite, lineage, nonnegative_int, positive, required
from .evidence import (
    COMPOSITION_TRIGGER_ORIGIN_CONDITION,
    COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
    COMPOSITION_TRIGGER_ORIGINS_DEVELOPMENT,
    CompositionTriggerEvidence,
)


COMPOSITION_TRIGGER_EDGE_RISING = "rising_edge"
COMPOSITION_TRIGGER_EDGE_EACH_EVENT = "each_event"
COMPOSITION_TRIGGER_EDGE_POLICIES_DEVELOPMENT = frozenset(
    {
        COMPOSITION_TRIGGER_EDGE_RISING,
        COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
    }
)

COMPOSITION_TRIGGER_STATUS_TRIGGERED = "triggered"
COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE = "condition_inactive"
COMPOSITION_TRIGGER_STATUS_CONDITION_HELD = "condition_held"
COMPOSITION_TRIGGER_STATUS_DUPLICATE = "duplicate"
COMPOSITION_TRIGGER_STATUS_STALE = "stale"
COMPOSITION_TRIGGER_STATUSES_DEVELOPMENT = frozenset(
    {
        COMPOSITION_TRIGGER_STATUS_TRIGGERED,
        COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE,
        COMPOSITION_TRIGGER_STATUS_CONDITION_HELD,
        COMPOSITION_TRIGGER_STATUS_DUPLICATE,
        COMPOSITION_TRIGGER_STATUS_STALE,
    }
)

COMPOSITION_TRIGGER_RULE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_trigger_rule.development"
)
COMPOSITION_TRIGGER_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_trigger_state.development"
)
COMPOSITION_OPERATION_TRIGGER_PROPOSAL_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_operation_trigger_proposal.development"
)


@dataclass(frozen=True)
class CompositionTriggerRule:
    """Bind one declared signal source to one requested operation."""

    rule_id: str
    trigger_source_id: str
    trigger_source_spec_hash: str
    origin_kind: str
    signal_name: str
    edge_policy: str
    operation_kind: str
    subject_id: str
    component_ids: tuple[str, ...]
    operation_payload_contract_hash: str
    operation_payload: Mapping[str, Any]
    proposal_ttl_sec: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rule_id = required(self.rule_id, name="rule_id")
        source_id = required(self.trigger_source_id, name="trigger_source_id")
        source_hash = required(
            self.trigger_source_spec_hash,
            name="trigger_source_spec_hash",
        )
        origin = required(self.origin_kind, name="origin_kind")
        if origin not in COMPOSITION_TRIGGER_ORIGINS_DEVELOPMENT:
            raise ValueError("Unsupported composition trigger origin.")
        signal_name = required(self.signal_name, name="signal_name")
        edge_policy = required(self.edge_policy, name="edge_policy")
        if edge_policy not in COMPOSITION_TRIGGER_EDGE_POLICIES_DEVELOPMENT:
            raise ValueError("Unsupported composition trigger edge policy.")
        expected_edge = (
            COMPOSITION_TRIGGER_EDGE_RISING
            if origin == COMPOSITION_TRIGGER_ORIGIN_CONDITION
            else COMPOSITION_TRIGGER_EDGE_EACH_EVENT
        )
        if edge_policy != expected_edge:
            raise ValueError(
                "Condition triggers use rising_edge; direct events use each_event."
            )
        operation_kind = required(self.operation_kind, name="operation_kind")
        subject_id = required(self.subject_id, name="subject_id")
        component_ids = tuple(
            sorted(required(value, name="component_id") for value in self.component_ids)
        )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component_ids must be unique.")
        payload_contract_hash = required(
            self.operation_payload_contract_hash,
            name="operation_payload_contract_hash",
        )
        operation_payload = deep_freeze(dict(self.operation_payload))
        ttl = positive(self.proposal_ttl_sec, name="proposal_ttl_sec")
        semantic = {
            "rule_id": rule_id,
            "trigger_source_id": source_id,
            "trigger_source_spec_hash": source_hash,
            "origin_kind": origin,
            "signal_name": signal_name,
            "edge_policy": edge_policy,
            "operation_kind": operation_kind,
            "subject_id": subject_id,
            "component_ids": component_ids,
            "operation_payload_contract_hash": payload_contract_hash,
            "operation_payload": operation_payload,
            "proposal_ttl_sec": ttl,
        }
        for name, value in semantic.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=COMPOSITION_TRIGGER_RULE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class CompositionOperationTriggerProposal:
    """Non-authoritative request for an existing operation owner to admit."""

    runtime_session_id: str
    control_domain_id: str
    composition_activation_id: str
    composition_activation_hash: str
    rule_id: str
    rule_hash: str
    evidence_id: str
    evidence_hash: str
    control_cut_hash: str
    operation_kind: str
    subject_id: str
    component_ids: tuple[str, ...]
    operation_payload_contract_hash: str
    operation_payload: Mapping[str, Any]
    triggered_at: float
    available_at: float
    valid_until: float
    clock_domain_id: str
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        required_fields = (
            "runtime_session_id",
            "control_domain_id",
            "composition_activation_id",
            "composition_activation_hash",
            "rule_id",
            "rule_hash",
            "evidence_id",
            "evidence_hash",
            "control_cut_hash",
            "operation_kind",
            "subject_id",
            "operation_payload_contract_hash",
            "clock_domain_id",
        )
        normalized = {
            name: required(getattr(self, name), name=name) for name in required_fields
        }
        component_ids = tuple(
            sorted(required(value, name="component_id") for value in self.component_ids)
        )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component_ids must be unique.")
        triggered_at = finite(self.triggered_at, name="triggered_at")
        available_at = finite(self.available_at, name="available_at")
        valid_until = finite(self.valid_until, name="valid_until")
        if not triggered_at <= available_at < valid_until:
            raise ValueError(
                "Proposal timing must satisfy triggered_at <= available_at < "
                "valid_until."
            )
        payload = deep_freeze(dict(self.operation_payload))
        fact_ids, fact_hashes = lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        semantic = {
            **normalized,
            "component_ids": component_ids,
            "operation_payload": payload,
            "triggered_at": triggered_at,
            "available_at": available_at,
            "valid_until": valid_until,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "authoritative": False,
        }
        for name, value in semantic.items():
            if name != "authoritative":
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=(
                    COMPOSITION_OPERATION_TRIGGER_PROPOSAL_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def proposal_id(self) -> str:
        return f"composition-operation-trigger:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False


@dataclass(frozen=True)
class CompositionTriggerState:
    """Activation-local edge/deduplication state for exactly one rule."""

    rule_id: str
    rule_hash: str
    composition_activation_id: str
    composition_activation_hash: str
    revision: int
    condition_active: bool
    last_producer_epoch: int | None = None
    last_sequence: int | None = None
    last_evidence_id: str | None = None
    last_evidence_hash: str | None = None
    last_proposal_id: str | None = None
    last_proposal_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rule_id = required(self.rule_id, name="rule_id")
        rule_hash = required(self.rule_hash, name="rule_hash")
        activation_id = required(
            self.composition_activation_id,
            name="composition_activation_id",
        )
        activation_hash = required(
            self.composition_activation_hash,
            name="composition_activation_hash",
        )
        revision = nonnegative_int(self.revision, name="revision")
        if not isinstance(self.condition_active, bool):
            raise TypeError("condition_active must be a bool.")
        last_identity = (
            self.last_producer_epoch,
            self.last_sequence,
            self.last_evidence_id,
            self.last_evidence_hash,
        )
        if any(value is None for value in last_identity) and any(
            value is not None for value in last_identity
        ):
            raise ValueError("Last trigger evidence identity must be complete.")
        if self.last_producer_epoch is not None:
            nonnegative_int(self.last_producer_epoch, name="last_producer_epoch")
            nonnegative_int(self.last_sequence, name="last_sequence")
            required(self.last_evidence_id, name="last_evidence_id")
            required(self.last_evidence_hash, name="last_evidence_hash")
        proposal_identity = (self.last_proposal_id, self.last_proposal_hash)
        if (proposal_identity[0] is None) != (proposal_identity[1] is None):
            raise ValueError("Last trigger proposal identity must be complete.")
        if self.last_proposal_id is not None:
            required(self.last_proposal_id, name="last_proposal_id")
            required(self.last_proposal_hash, name="last_proposal_hash")
        semantic = {
            "rule_id": rule_id,
            "rule_hash": rule_hash,
            "composition_activation_id": activation_id,
            "composition_activation_hash": activation_hash,
            "revision": revision,
            "condition_active": self.condition_active,
            "last_producer_epoch": self.last_producer_epoch,
            "last_sequence": self.last_sequence,
            "last_evidence_id": self.last_evidence_id,
            "last_evidence_hash": self.last_evidence_hash,
            "last_proposal_id": self.last_proposal_id,
            "last_proposal_hash": self.last_proposal_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=COMPOSITION_TRIGGER_STATE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class CompositionTriggerEvaluation:
    status: str
    reason: str
    state: CompositionTriggerState
    evidence: CompositionTriggerEvidence
    proposal: CompositionOperationTriggerProposal | None = None

    def __post_init__(self) -> None:
        status = required(self.status, name="status")
        if status not in COMPOSITION_TRIGGER_STATUSES_DEVELOPMENT:
            raise ValueError("Unsupported composition trigger evaluation status.")
        reason = required(self.reason, name="reason")
        if not isinstance(self.state, CompositionTriggerState):
            raise TypeError("state must be CompositionTriggerState.")
        if not isinstance(self.evidence, CompositionTriggerEvidence):
            raise TypeError("evidence must be CompositionTriggerEvidence.")
        if status == COMPOSITION_TRIGGER_STATUS_TRIGGERED:
            if not isinstance(self.proposal, CompositionOperationTriggerProposal):
                raise TypeError("A triggered evaluation requires a proposal.")
        elif self.proposal is not None:
            raise ValueError("A non-triggered evaluation cannot carry a proposal.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)


__all__ = [
    "COMPOSITION_OPERATION_TRIGGER_PROPOSAL_HASH_DOMAIN_DEVELOPMENT",
    "COMPOSITION_TRIGGER_EDGE_EACH_EVENT",
    "COMPOSITION_TRIGGER_EDGE_POLICIES_DEVELOPMENT",
    "COMPOSITION_TRIGGER_EDGE_RISING",
    "COMPOSITION_TRIGGER_STATUS_CONDITION_HELD",
    "COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE",
    "COMPOSITION_TRIGGER_STATUS_DUPLICATE",
    "COMPOSITION_TRIGGER_STATUS_STALE",
    "COMPOSITION_TRIGGER_STATUS_TRIGGERED",
    "COMPOSITION_TRIGGER_STATUSES_DEVELOPMENT",
    "CompositionOperationTriggerProposal",
    "CompositionTriggerEvaluation",
    "CompositionTriggerRule",
    "CompositionTriggerState",
]

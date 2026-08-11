"""Pure edge and deduplication reducer for composition operation triggers."""

from __future__ import annotations

from control_composition.activation import CompositionActivationScope
from robot_motion.runtime.control_admission import ControlCut

from .contracts import (
    COMPOSITION_TRIGGER_STATUS_CONDITION_HELD,
    COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE,
    COMPOSITION_TRIGGER_STATUS_DUPLICATE,
    COMPOSITION_TRIGGER_STATUS_STALE,
    COMPOSITION_TRIGGER_STATUS_TRIGGERED,
    CompositionOperationTriggerProposal,
    CompositionTriggerEvaluation,
    CompositionTriggerRule,
    CompositionTriggerState,
)
from .evidence import (
    COMPOSITION_TRIGGER_ORIGIN_CONDITION,
    COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
    CompositionTriggerEvidence,
    composition_trigger_control_cut_hash,
)


def empty_composition_trigger_state(
    rule: CompositionTriggerRule,
    *,
    activation_scope: CompositionActivationScope,
) -> CompositionTriggerState:
    """Create one rule-local latch for one exact composition activation."""

    if not isinstance(rule, CompositionTriggerRule):
        raise TypeError("rule must be CompositionTriggerRule.")
    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    return CompositionTriggerState(
        rule_id=rule.rule_id,
        rule_hash=rule.semantic_content_hash,
        composition_activation_id=activation_scope.activation_id,
        composition_activation_hash=activation_scope.semantic_content_hash,
        revision=0,
        condition_active=False,
    )


def _validate_binding(
    state: CompositionTriggerState,
    *,
    rule: CompositionTriggerRule,
    evidence: CompositionTriggerEvidence,
    control_cut: ControlCut,
    activation_scope: CompositionActivationScope,
) -> None:
    if state.rule_id != rule.rule_id or state.rule_hash != rule.semantic_content_hash:
        raise ValueError("Trigger state belongs to another rule.")
    if (
        state.composition_activation_id != activation_scope.activation_id
        or state.composition_activation_hash
        != activation_scope.semantic_content_hash
    ):
        raise ValueError("Trigger state belongs to another activation.")
    if (
        evidence.composition_activation_id != activation_scope.activation_id
        or evidence.composition_activation_hash
        != activation_scope.semantic_content_hash
    ):
        raise ValueError("Trigger evidence belongs to another activation.")
    if control_cut.clock_domain_id != activation_scope.clock_domain_id:
        raise ValueError("Trigger cut and activation use another clock.")
    if control_cut.cutoff_at < activation_scope.activated_at:
        raise ValueError("Trigger cut predates its activation.")
    if evidence.runtime_session_id != control_cut.runtime_session_id:
        raise ValueError("Trigger evidence belongs to another runtime session.")
    if evidence.control_domain_id != control_cut.control_domain_id:
        raise ValueError("Trigger evidence belongs to another control domain.")
    if evidence.clock_domain_id != control_cut.clock_domain_id:
        raise ValueError("Trigger evidence uses another clock domain.")
    if evidence.clock_reset_epoch != control_cut.clock_reset_epoch:
        raise ValueError("Trigger evidence uses another clock reset epoch.")
    if evidence.available_at > control_cut.cutoff_at:
        raise ValueError("Trigger evidence was unavailable at the control cut.")
    if (
        evidence.control_admission_sequence
        > control_cut.cutoff_admission_sequence
    ):
        raise ValueError("Trigger evidence was admitted after the control cut.")
    if evidence.available_at < activation_scope.input_barrier_at:
        raise ValueError("Trigger evidence predates the activation barrier.")
    if evidence.trigger_source_id != rule.trigger_source_id:
        raise ValueError("Trigger evidence comes from another declared source.")
    if evidence.trigger_source_spec_hash != rule.trigger_source_spec_hash:
        raise ValueError("Trigger evidence source spec changed.")
    if evidence.origin_kind != rule.origin_kind:
        raise ValueError("Trigger evidence origin does not match its rule.")
    if evidence.signal_name != rule.signal_name:
        raise ValueError("Trigger evidence signal does not match its rule.")


def _is_duplicate_or_validate_order(
    state: CompositionTriggerState,
    evidence: CompositionTriggerEvidence,
) -> bool:
    if state.last_producer_epoch is None:
        return False
    ordering = (evidence.producer_epoch, evidence.sequence)
    previous = (state.last_producer_epoch, state.last_sequence)
    if ordering == previous:
        if (
            evidence.evidence_id == state.last_evidence_id
            and evidence.semantic_content_hash == state.last_evidence_hash
        ):
            return True
        raise ValueError("Trigger producer reused one sequence for another fact.")
    if ordering < previous:
        raise ValueError("Trigger producer ordering moved backwards.")
    return False


def _next_state(
    state: CompositionTriggerState,
    *,
    evidence: CompositionTriggerEvidence,
    condition_active: bool,
    proposal: CompositionOperationTriggerProposal | None,
) -> CompositionTriggerState:
    return CompositionTriggerState(
        rule_id=state.rule_id,
        rule_hash=state.rule_hash,
        composition_activation_id=state.composition_activation_id,
        composition_activation_hash=state.composition_activation_hash,
        revision=state.revision + 1,
        condition_active=condition_active,
        last_producer_epoch=evidence.producer_epoch,
        last_sequence=evidence.sequence,
        last_evidence_id=evidence.evidence_id,
        last_evidence_hash=evidence.semantic_content_hash,
        last_proposal_id=(
            state.last_proposal_id if proposal is None else proposal.proposal_id
        ),
        last_proposal_hash=(
            state.last_proposal_hash
            if proposal is None
            else proposal.semantic_content_hash
        ),
    )


def _proposal(
    rule: CompositionTriggerRule,
    *,
    evidence: CompositionTriggerEvidence,
    control_cut: ControlCut,
    activation_scope: CompositionActivationScope,
) -> CompositionOperationTriggerProposal:
    return CompositionOperationTriggerProposal(
        runtime_session_id=control_cut.runtime_session_id,
        control_domain_id=control_cut.control_domain_id,
        composition_activation_id=activation_scope.activation_id,
        composition_activation_hash=activation_scope.semantic_content_hash,
        rule_id=rule.rule_id,
        rule_hash=rule.semantic_content_hash,
        evidence_id=evidence.evidence_id,
        evidence_hash=evidence.semantic_content_hash,
        control_cut_hash=composition_trigger_control_cut_hash(control_cut),
        operation_kind=rule.operation_kind,
        subject_id=rule.subject_id,
        component_ids=rule.component_ids,
        operation_payload_contract_hash=rule.operation_payload_contract_hash,
        operation_payload=rule.operation_payload,
        triggered_at=evidence.occurred_at,
        available_at=control_cut.cutoff_at,
        valid_until=min(
            evidence.valid_until,
            control_cut.cutoff_at + rule.proposal_ttl_sec,
        ),
        clock_domain_id=control_cut.clock_domain_id,
        source_fact_ids=evidence.source_fact_ids,
        source_fact_hashes=evidence.source_fact_hashes,
    )


def evaluate_composition_operation_trigger(
    state: CompositionTriggerState,
    *,
    rule: CompositionTriggerRule,
    evidence: CompositionTriggerEvidence,
    control_cut: ControlCut,
    activation_scope: CompositionActivationScope,
) -> CompositionTriggerEvaluation:
    """Evaluate one signal without invoking the requested operation.

    Condition sources are rising-edge latched: repeated active samples do not
    emit repeatedly, and an inactive sample rearms the rule.  Every distinct
    direct event emits once.  Exact duplicates are idempotent; sequence reuse,
    ordering rollback, cross-activation evidence, and post-cut evidence are
    rejected.
    """

    if not isinstance(state, CompositionTriggerState):
        raise TypeError("state must be CompositionTriggerState.")
    if not isinstance(rule, CompositionTriggerRule):
        raise TypeError("rule must be CompositionTriggerRule.")
    if not isinstance(evidence, CompositionTriggerEvidence):
        raise TypeError("evidence must be CompositionTriggerEvidence.")
    if not isinstance(control_cut, ControlCut):
        raise TypeError("control_cut must be ControlCut.")
    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    _validate_binding(
        state,
        rule=rule,
        evidence=evidence,
        control_cut=control_cut,
        activation_scope=activation_scope,
    )
    if evidence.valid_until <= control_cut.cutoff_at:
        return CompositionTriggerEvaluation(
            status=COMPOSITION_TRIGGER_STATUS_STALE,
            reason="trigger_evidence_expired_before_cut",
            state=state,
            evidence=evidence,
        )
    if _is_duplicate_or_validate_order(state, evidence):
        return CompositionTriggerEvaluation(
            status=COMPOSITION_TRIGGER_STATUS_DUPLICATE,
            reason="same_trigger_evidence_already_evaluated",
            state=state,
            evidence=evidence,
        )

    if evidence.origin_kind == COMPOSITION_TRIGGER_ORIGIN_CONDITION:
        if not evidence.active:
            next_state = _next_state(
                state,
                evidence=evidence,
                condition_active=False,
                proposal=None,
            )
            return CompositionTriggerEvaluation(
                status=COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE,
                reason="condition_inactive_rule_rearmed",
                state=next_state,
                evidence=evidence,
            )
        if state.condition_active:
            next_state = _next_state(
                state,
                evidence=evidence,
                condition_active=True,
                proposal=None,
            )
            return CompositionTriggerEvaluation(
                status=COMPOSITION_TRIGGER_STATUS_CONDITION_HELD,
                reason="condition_remains_active_without_new_edge",
                state=next_state,
                evidence=evidence,
            )
    elif evidence.origin_kind != COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT:
        raise ValueError("Unsupported trigger evidence origin.")

    proposal = _proposal(
        rule,
        evidence=evidence,
        control_cut=control_cut,
        activation_scope=activation_scope,
    )
    next_state = _next_state(
        state,
        evidence=evidence,
        condition_active=(
            evidence.origin_kind == COMPOSITION_TRIGGER_ORIGIN_CONDITION
        ),
        proposal=proposal,
    )
    return CompositionTriggerEvaluation(
        status=COMPOSITION_TRIGGER_STATUS_TRIGGERED,
        reason=(
            "condition_rising_edge"
            if evidence.origin_kind == COMPOSITION_TRIGGER_ORIGIN_CONDITION
            else "direct_event_edge"
        ),
        state=next_state,
        evidence=evidence,
        proposal=proposal,
    )


__all__ = [
    "empty_composition_trigger_state",
    "evaluate_composition_operation_trigger",
]

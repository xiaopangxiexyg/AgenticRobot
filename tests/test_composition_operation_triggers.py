from __future__ import annotations

from dataclasses import replace

import pytest

from control_composition.activation import build_composition_activation_scope
from control_composition.triggers import (
    COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
    COMPOSITION_TRIGGER_EDGE_RISING,
    COMPOSITION_TRIGGER_ORIGIN_CONDITION,
    COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
    COMPOSITION_TRIGGER_STATUS_CONDITION_HELD,
    COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE,
    COMPOSITION_TRIGGER_STATUS_DUPLICATE,
    COMPOSITION_TRIGGER_STATUS_STALE,
    COMPOSITION_TRIGGER_STATUS_TRIGGERED,
    CompositionTriggerEvaluationBasis,
    CompositionTriggerEvidence,
    CompositionTriggerEvidenceProvider,
    CompositionTriggerRule,
    empty_composition_trigger_state,
    evaluate_composition_operation_trigger,
)
from robot_motion.runtime.control_admission import ControlCut


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _activation(*, session_epoch: int = 1):
    return build_composition_activation_scope(
        operation_spec_id="trigger-test",
        operation_spec_hash="trigger-test-operation-hash",
        session_epoch=session_epoch,
        activated_at=10.0 + session_epoch - 1,
        input_barrier_at=10.0 + session_epoch - 1,
        clock_domain_id="control.monotonic",
        required_roles=("base", "overlay"),
    )


def _cut(*, at: float, admission_sequence: int) -> ControlCut:
    return ControlCut(
        runtime_session_id="session",
        control_domain_id="robot",
        clock_domain_id="control.monotonic",
        clock_reset_epoch=0,
        cutoff_at=at,
        cutoff_admission_sequence=admission_sequence,
    )


def _rule(
    *,
    origin: str,
    source_id: str,
    signal_name: str,
    operation_kind: str = "residual_rebase_and_clear",
) -> CompositionTriggerRule:
    return CompositionTriggerRule(
        rule_id=f"{source_id}:{operation_kind}",
        trigger_source_id=source_id,
        trigger_source_spec_hash=f"{source_id}-spec-hash",
        origin_kind=origin,
        signal_name=signal_name,
        edge_policy=(
            COMPOSITION_TRIGGER_EDGE_RISING
            if origin == COMPOSITION_TRIGGER_ORIGIN_CONDITION
            else COMPOSITION_TRIGGER_EDGE_EACH_EVENT
        ),
        operation_kind=operation_kind,
        subject_id="composer-primary",
        component_ids=("right_arm",),
        operation_payload_contract_hash="residual-rebase-request-v1",
        operation_payload={"capture_mode": "first_fresh_after_barrier"},
        proposal_ttl_sec=0.25,
    )


def _evidence(
    rule: CompositionTriggerRule,
    activation,
    *,
    sequence: int,
    active: bool,
    at: float,
    admission_sequence: int | None = None,
    valid_until: float | None = None,
    payload: dict[str, object] | None = None,
) -> CompositionTriggerEvidence:
    return CompositionTriggerEvidence(
        runtime_session_id="session",
        control_domain_id="robot",
        clock_reset_epoch=0,
        composition_activation_id=activation.activation_id,
        composition_activation_hash=activation.semantic_content_hash,
        trigger_source_id=rule.trigger_source_id,
        trigger_source_spec_hash=rule.trigger_source_spec_hash,
        origin_kind=rule.origin_kind,
        signal_name=rule.signal_name,
        producer_epoch=1,
        sequence=sequence,
        active=active,
        occurred_at=at - 0.02,
        available_at=at - 0.01,
        valid_until=at + 0.5 if valid_until is None else valid_until,
        clock_domain_id="control.monotonic",
        control_admission_sequence=(
            sequence if admission_sequence is None else admission_sequence
        ),
        payload_contract_hash="trigger-signal-v1",
        semantic_payload=(
            {"signal_value": float(active)} if payload is None else payload
        ),
        source_fact_ids=(f"source-fact:{rule.trigger_source_id}:{sequence}",),
        source_fact_hashes=(
            f"source-fact-hash:{rule.trigger_source_id}:{sequence}",
        ),
    )


class LargeMotionConditionProvider:
    trigger_source_id = "human-motion-detector"
    trigger_source_spec_hash = "human-motion-detector-spec-hash"
    origin_kind = COMPOSITION_TRIGGER_ORIGIN_CONDITION

    def __init__(self, *, threshold: float) -> None:
        self.threshold = float(threshold)
        self.sequence = 0

    def evaluate(
        self,
        basis: CompositionTriggerEvaluationBasis,
    ) -> CompositionTriggerEvidence:
        self.sequence += 1
        level = float(basis.semantic_payload["motion_level"])
        cut = basis.control_cut
        activation = basis.activation_scope
        return CompositionTriggerEvidence(
            runtime_session_id=cut.runtime_session_id,
            control_domain_id=cut.control_domain_id,
            clock_reset_epoch=cut.clock_reset_epoch,
            composition_activation_id=activation.activation_id,
            composition_activation_hash=activation.semantic_content_hash,
            trigger_source_id=self.trigger_source_id,
            trigger_source_spec_hash=self.trigger_source_spec_hash,
            origin_kind=self.origin_kind,
            signal_name="human_motion_large",
            producer_epoch=1,
            sequence=self.sequence,
            active=level >= self.threshold,
            occurred_at=cut.cutoff_at,
            available_at=cut.cutoff_at,
            valid_until=basis.valid_until,
            clock_domain_id=cut.clock_domain_id,
            control_admission_sequence=cut.cutoff_admission_sequence,
            payload_contract_hash="human-motion-level-v1",
            semantic_payload={"motion_level": level, "threshold": self.threshold},
            source_fact_ids=basis.source_fact_ids,
            source_fact_hashes=basis.source_fact_hashes,
        )


class DirectButtonEventProvider:
    trigger_source_id = "operator-button"
    trigger_source_spec_hash = "operator-button-spec-hash"
    origin_kind = COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT

    def __init__(self) -> None:
        self.sequence = 0
        self.last_event_id: str | None = None

    def evaluate(
        self,
        basis: CompositionTriggerEvaluationBasis,
    ) -> CompositionTriggerEvidence | None:
        event_id = str(basis.semantic_payload.get("event_id", "")).strip()
        if (
            not bool(basis.semantic_payload.get("pressed", False))
            or not event_id
            or event_id == self.last_event_id
        ):
            return None
        self.last_event_id = event_id
        self.sequence += 1
        cut = basis.control_cut
        activation = basis.activation_scope
        return CompositionTriggerEvidence(
            runtime_session_id=cut.runtime_session_id,
            control_domain_id=cut.control_domain_id,
            clock_reset_epoch=cut.clock_reset_epoch,
            composition_activation_id=activation.activation_id,
            composition_activation_hash=activation.semantic_content_hash,
            trigger_source_id=self.trigger_source_id,
            trigger_source_spec_hash=self.trigger_source_spec_hash,
            origin_kind=self.origin_kind,
            signal_name="rebase_button_pressed",
            producer_epoch=1,
            sequence=self.sequence,
            active=True,
            occurred_at=cut.cutoff_at,
            available_at=cut.cutoff_at,
            valid_until=basis.valid_until,
            clock_domain_id=cut.clock_domain_id,
            control_admission_sequence=cut.cutoff_admission_sequence,
            payload_contract_hash="operator-button-event-v1",
            semantic_payload={"event_id": event_id, "pressed": True},
            source_fact_ids=basis.source_fact_ids,
            source_fact_hashes=basis.source_fact_hashes,
        )


def _basis(
    activation,
    *,
    at: float,
    admission_sequence: int,
    payload: dict[str, object],
) -> CompositionTriggerEvaluationBasis:
    return CompositionTriggerEvaluationBasis(
        control_cut=_cut(at=at, admission_sequence=admission_sequence),
        activation_scope=activation,
        payload_contract_hash="trigger-provider-input-v1",
        semantic_payload=payload,
        source_fact_ids=(f"basis-source:{admission_sequence}",),
        source_fact_hashes=(f"basis-source-hash:{admission_sequence}",),
        valid_until=at + 0.5,
    )


def test_condition_and_direct_event_providers_share_one_interface() -> None:
    condition = LargeMotionConditionProvider(threshold=2.0)
    button = DirectButtonEventProvider()

    assert isinstance(condition, CompositionTriggerEvidenceProvider)
    assert isinstance(button, CompositionTriggerEvidenceProvider)

    activation = _activation()
    condition_evidence = condition.evaluate(
        _basis(
            activation,
            at=10.1,
            admission_sequence=1,
            payload={"motion_level": 3.0},
        )
    )
    button_evidence = button.evaluate(
        _basis(
            activation,
            at=10.2,
            admission_sequence=2,
            payload={"event_id": "button-edge:1", "pressed": True},
        )
    )

    assert condition_evidence.active
    assert condition_evidence.origin_kind == COMPOSITION_TRIGGER_ORIGIN_CONDITION
    assert button_evidence is not None
    assert button_evidence.origin_kind == COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT
    assert button.evaluate(
        _basis(
            activation,
            at=10.3,
            admission_sequence=3,
            payload={"event_id": "button-edge:1", "pressed": True},
        )
    ) is None


def test_condition_emits_only_on_rising_edge_and_rearms_when_inactive() -> None:
    activation = _activation()
    rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_CONDITION,
        source_id="human-motion-detector",
        signal_name="human_motion_large",
    )
    state = empty_composition_trigger_state(rule, activation_scope=activation)

    inactive = evaluate_composition_operation_trigger(
        state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=1, active=False, at=10.1),
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )
    assert inactive.status == COMPOSITION_TRIGGER_STATUS_CONDITION_INACTIVE
    assert inactive.proposal is None

    rising = evaluate_composition_operation_trigger(
        inactive.state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=2, active=True, at=10.2),
        control_cut=_cut(at=10.2, admission_sequence=2),
        activation_scope=activation,
    )
    assert rising.status == COMPOSITION_TRIGGER_STATUS_TRIGGERED
    assert rising.proposal is not None
    assert not rising.proposal.authoritative

    held = evaluate_composition_operation_trigger(
        rising.state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=3, active=True, at=10.3),
        control_cut=_cut(at=10.3, admission_sequence=3),
        activation_scope=activation,
    )
    assert held.status == COMPOSITION_TRIGGER_STATUS_CONDITION_HELD
    assert held.proposal is None

    rearmed = evaluate_composition_operation_trigger(
        held.state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=4, active=False, at=10.4),
        control_cut=_cut(at=10.4, admission_sequence=4),
        activation_scope=activation,
    )
    second = evaluate_composition_operation_trigger(
        rearmed.state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=5, active=True, at=10.5),
        control_cut=_cut(at=10.5, admission_sequence=5),
        activation_scope=activation,
    )
    assert second.status == COMPOSITION_TRIGGER_STATUS_TRIGGERED
    assert second.proposal is not None
    assert second.proposal.proposal_id != rising.proposal.proposal_id


def test_each_distinct_direct_event_triggers_once_and_duplicate_is_idempotent() -> None:
    activation = _activation()
    rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        source_id="operator-button",
        signal_name="rebase_button_pressed",
    )
    state = empty_composition_trigger_state(rule, activation_scope=activation)
    first_evidence = _evidence(
        rule,
        activation,
        sequence=1,
        active=True,
        at=10.1,
    )
    first = evaluate_composition_operation_trigger(
        state,
        rule=rule,
        evidence=first_evidence,
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )
    duplicate = evaluate_composition_operation_trigger(
        first.state,
        rule=rule,
        evidence=first_evidence,
        control_cut=_cut(at=10.2, admission_sequence=2),
        activation_scope=activation,
    )
    second = evaluate_composition_operation_trigger(
        first.state,
        rule=rule,
        evidence=_evidence(rule, activation, sequence=2, active=True, at=10.2),
        control_cut=_cut(at=10.2, admission_sequence=2),
        activation_scope=activation,
    )

    assert first.status == COMPOSITION_TRIGGER_STATUS_TRIGGERED
    assert duplicate.status == COMPOSITION_TRIGGER_STATUS_DUPLICATE
    assert duplicate.state is first.state
    assert second.status == COMPOSITION_TRIGGER_STATUS_TRIGGERED
    assert second.proposal is not None
    assert first.proposal is not None
    assert second.proposal.proposal_id != first.proposal.proposal_id


def test_condition_and_button_can_request_the_same_operation_contract() -> None:
    activation = _activation()
    condition_rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_CONDITION,
        source_id="human-motion-detector",
        signal_name="human_motion_large",
    )
    button_rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        source_id="operator-button",
        signal_name="rebase_button_pressed",
    )
    condition = evaluate_composition_operation_trigger(
        empty_composition_trigger_state(
            condition_rule,
            activation_scope=activation,
        ),
        rule=condition_rule,
        evidence=_evidence(
            condition_rule,
            activation,
            sequence=1,
            active=True,
            at=10.1,
        ),
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )
    button = evaluate_composition_operation_trigger(
        empty_composition_trigger_state(button_rule, activation_scope=activation),
        rule=button_rule,
        evidence=_evidence(
            button_rule,
            activation,
            sequence=1,
            active=True,
            at=10.1,
        ),
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )

    assert condition.proposal is not None
    assert button.proposal is not None
    for field_name in (
        "operation_kind",
        "subject_id",
        "component_ids",
        "operation_payload_contract_hash",
        "operation_payload",
    ):
        assert getattr(condition.proposal, field_name) == getattr(
            button.proposal,
            field_name,
        )
    assert condition.proposal.evidence_id != button.proposal.evidence_id


def test_operation_kind_and_component_scope_are_profile_data() -> None:
    activation = _activation()
    rule = CompositionTriggerRule(
        rule_id="button-soft-mode-switch",
        trigger_source_id="operator-button",
        trigger_source_spec_hash="operator-button-spec-hash",
        origin_kind=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        signal_name="overlay_mode_pressed",
        edge_policy=COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
        operation_kind="allocation_mode_request",
        subject_id="allocation-scope:right-side",
        component_ids=("right_gripper", "right_arm"),
        operation_payload_contract_hash="allocation-mode-request-v1",
        operation_payload={"requested_mode": "relative_overlay"},
        proposal_ttl_sec=0.25,
    )
    result = evaluate_composition_operation_trigger(
        empty_composition_trigger_state(rule, activation_scope=activation),
        rule=rule,
        evidence=_evidence(
            rule,
            activation,
            sequence=1,
            active=True,
            at=10.1,
        ),
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )

    assert result.proposal is not None
    assert result.proposal.operation_kind == "allocation_mode_request"
    assert result.proposal.component_ids == ("right_arm", "right_gripper")
    assert result.proposal.operation_payload == {
        "requested_mode": "relative_overlay"
    }


def test_stale_evidence_cannot_trigger_or_consume_the_rule_latch() -> None:
    activation = _activation()
    rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        source_id="operator-button",
        signal_name="rebase_button_pressed",
    )
    state = empty_composition_trigger_state(rule, activation_scope=activation)
    stale_evidence = _evidence(
        rule,
        activation,
        sequence=1,
        active=True,
        at=10.1,
        valid_until=10.15,
    )

    stale = evaluate_composition_operation_trigger(
        state,
        rule=rule,
        evidence=stale_evidence,
        control_cut=_cut(at=10.15, admission_sequence=1),
        activation_scope=activation,
    )

    assert stale.status == COMPOSITION_TRIGGER_STATUS_STALE
    assert stale.state is state
    assert stale.proposal is None


def test_trigger_rejects_post_cut_and_previous_activation_evidence() -> None:
    activation = _activation()
    rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_CONDITION,
        source_id="human-motion-detector",
        signal_name="human_motion_large",
    )
    state = empty_composition_trigger_state(rule, activation_scope=activation)
    future = _evidence(
        rule,
        activation,
        sequence=1,
        active=True,
        at=10.2,
        admission_sequence=2,
    )
    with pytest.raises(ValueError, match="unavailable at the control cut"):
        evaluate_composition_operation_trigger(
            state,
            rule=rule,
            evidence=future,
            control_cut=_cut(at=10.1, admission_sequence=1),
            activation_scope=activation,
        )

    next_activation = _activation(session_epoch=2)
    next_state = empty_composition_trigger_state(
        rule,
        activation_scope=next_activation,
    )
    with pytest.raises(ValueError, match="another activation"):
        evaluate_composition_operation_trigger(
            next_state,
            rule=rule,
            evidence=future,
            control_cut=_cut(at=11.1, admission_sequence=2),
            activation_scope=next_activation,
        )


def test_sequence_reuse_with_changed_payload_and_ordering_rollback_are_rejected() -> None:
    activation = _activation()
    rule = _rule(
        origin=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        source_id="operator-button",
        signal_name="rebase_button_pressed",
    )
    state = empty_composition_trigger_state(rule, activation_scope=activation)
    first_evidence = _evidence(
        rule,
        activation,
        sequence=2,
        active=True,
        at=10.2,
    )
    first = evaluate_composition_operation_trigger(
        state,
        rule=rule,
        evidence=first_evidence,
        control_cut=_cut(at=10.2, admission_sequence=2),
        activation_scope=activation,
    )
    reused = replace(first_evidence, semantic_payload={"button": "another"})
    with pytest.raises(ValueError, match="reused one sequence"):
        evaluate_composition_operation_trigger(
            first.state,
            rule=rule,
            evidence=reused,
            control_cut=_cut(at=10.3, admission_sequence=3),
            activation_scope=activation,
        )
    with pytest.raises(ValueError, match="ordering moved backwards"):
        evaluate_composition_operation_trigger(
            first.state,
            rule=rule,
            evidence=_evidence(
                rule,
                activation,
                sequence=1,
                active=True,
                at=10.3,
            ),
            control_cut=_cut(at=10.3, admission_sequence=3),
            activation_scope=activation,
        )


def test_payloads_are_frozen_and_proposal_exposes_no_execution_api() -> None:
    activation = _activation()
    rule_payload = {"capture": {"mode": "fresh"}}
    rule = CompositionTriggerRule(
        rule_id="generic-rebase",
        trigger_source_id="external-event",
        trigger_source_spec_hash="external-event-spec-hash",
        origin_kind=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
        signal_name="external_rebase",
        edge_policy=COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
        operation_kind="residual_rebase_and_clear",
        subject_id="composer-primary",
        component_ids=(),
        operation_payload_contract_hash="generic-rebase-v1",
        operation_payload=rule_payload,
        proposal_ttl_sec=0.25,
    )
    rule_payload["capture"]["mode"] = "mutated"
    evidence_payload = {"button": {"pressed": True}}
    evidence = _evidence(
        rule,
        activation,
        sequence=1,
        active=True,
        at=10.1,
        payload=evidence_payload,
    )
    evidence_payload["button"]["pressed"] = False
    result = evaluate_composition_operation_trigger(
        empty_composition_trigger_state(rule, activation_scope=activation),
        rule=rule,
        evidence=evidence,
        control_cut=_cut(at=10.1, admission_sequence=1),
        activation_scope=activation,
    )

    assert rule.operation_payload["capture"]["mode"] == "fresh"
    assert evidence.semantic_payload["button"]["pressed"] is True
    assert result.proposal is not None
    forbidden = {"dispatch", "execute", "hold", "grant_authority", "run_safety"}
    assert not any(hasattr(result.proposal, name) for name in forbidden)


def test_origin_and_edge_policy_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="Condition triggers use rising_edge"):
        CompositionTriggerRule(
            rule_id="invalid",
            trigger_source_id="condition",
            trigger_source_spec_hash="condition-spec-hash",
            origin_kind=COMPOSITION_TRIGGER_ORIGIN_CONDITION,
            signal_name="active",
            edge_policy=COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
            operation_kind="mode_switch",
            subject_id="composer",
            component_ids=(),
            operation_payload_contract_hash="mode-switch-v1",
            operation_payload={},
            proposal_ttl_sec=0.1,
        )

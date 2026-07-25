"""Project control-composition domain objects into v2.4 ControlTrace facts.

This is a dataset integration adapter.  It owns no composition decision,
authority, persistence, WAL, or episode lifecycle.
"""

from __future__ import annotations

from typing import Callable

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.action.proposal import robot_action_semantic_payload
from control_composition.activation import CompositionActivationScope
from control_composition.events import (
    CompositionActivationBinding,
    CompositionDomainEvent,
    CompositionSourceEligibilityBinding,
)
from control_composition.information import CompositionInformationPlan
from control_composition.eligibility import (
    CompositionSourceEligibilityAdmissionReceipt,
    CompositionSourceEligibilitySnapshot,
    CompositionSourceEligibilitySpec,
    CompositionSourceEligibilityState,
    source_eligibility_state_semantic_payload,
)

from control_composition.composition import (
    ComponentMixingShadowDecision,
    ComponentCompositionRuleSet,
    CompositionAuthorityDisposition,
    CompositionInputAdmissionReceipt,
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionResult,
    CompositionSpec,
    FormalCompositionStrategyInput,
    FormalCompositionStrategyResultBinding,
    FormalCompositionStrategySpec,
    FormalComponentCompositionBinding,
    composition_input_semantic_payload,
    CompositionContextAdmissionReceipt,
    CompositionContextRecord,
    CompositionContextSnapshot,
    composition_context_semantic_payload,
)
from control_composition.observability import (
    GuidanceFeedbackCommand,
    GuidanceFeedbackReceipt,
    SourceDifferenceObservation,
    SourceDifferenceSpec,
    action_difference_semantic_payload,
)
from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionTransition,
)


def project_composition_session_spec(
    spec: CompositionSessionOperationSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(spec, CompositionSessionOperationSpec):
        raise TypeError("spec must be a CompositionSessionOperationSpec.")
    return ControlTraceFact.create(
        fact_id=spec.operation_spec_id,
        event_type="composition_session_spec",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "operation_spec_id": spec.operation_spec_id,
            "session_key": spec.session_key,
            "composer_source_slot_id": spec.composer_source_slot_id,
            "contributor_source_slot_ids": spec.contributor_source_slot_ids,
            "composition_spec_hash": spec.composition_spec_hash,
            "authority_gate_spec_hash": spec.authority_gate_spec_hash,
            "execution_binding_hash": spec.execution_binding_hash,
            "capture_boundary_id": spec.capture_boundary_id,
            "pause_behavior": spec.pause_behavior,
            "end_behavior": spec.end_behavior,
            "resume_behavior": spec.resume_behavior,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_composition_session_transition(
    transition: CompositionSessionTransition,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(transition, CompositionSessionTransition):
        raise TypeError(
            "transition must be a CompositionSessionTransition."
        )
    return ControlTraceFact.create(
        fact_id=transition.transition_id,
        event_type="composition_session_transition",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "transition_id": transition.transition_id,
            "operation_spec_id": transition.operation_spec_id,
            "operation_spec_hash": transition.operation_spec_hash,
            "session_epoch": transition.session_epoch,
            "transition": transition.transition,
            "reason": transition.reason,
            "source_fact_id": transition.source_fact_id,
            "source_fact_hash": transition.source_fact_hash,
            "state_before_hash": transition.state_before_hash,
            "state_after_hash": transition.state_after_hash,
            "semantic_content_hash": transition.semantic_content_hash,
        },
    )


def project_composition_session_rearm_spec(
    spec: CompositionSessionRearmSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    if not isinstance(spec, CompositionSessionRearmSpec):
        raise TypeError("spec must be a CompositionSessionRearmSpec.")
    return ControlTraceFact.create(
        fact_id=spec.rearm_spec_id,
        event_type="composition_session_rearm_spec",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "rearm_spec_id": spec.rearm_spec_id,
            "rearm_key": spec.rearm_key,
            "logical_operation_spec_id": spec.logical_operation_spec_id,
            "logical_operation_spec_hash": spec.logical_operation_spec_hash,
            "prior_session_epoch": spec.prior_session_epoch,
            "prior_state_hash": spec.prior_state_hash,
            "execution_binding_hash": spec.execution_binding_hash,
            "readiness_observer_binding_id": (
                spec.readiness_observer_binding_id
            ),
            "readiness_timeout_sec": spec.readiness_timeout_sec,
            "required_readiness_checks": spec.required_readiness_checks,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_composition_session_rearm_receipt(
    receipt: CompositionSessionRearmReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
) -> ControlTraceFact:
    if not isinstance(receipt, CompositionSessionRearmReceipt):
        raise TypeError("receipt must be a CompositionSessionRearmReceipt.")
    return ControlTraceFact.create(
        fact_id=receipt.receipt_id,
        event_type="composition_session_rearm_receipt",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=receipt.completed_at,
        clock_domain_id=receipt.clock_domain_id,
        payload={
            field_name: getattr(receipt, field_name)
            for field_name in (
                "receipt_id",
                "rearm_spec_id",
                "rearm_spec_hash",
                "logical_operation_spec_id",
                "logical_operation_spec_hash",
                "prior_session_epoch",
                "next_session_epoch",
                "prior_composer_producer_epoch",
                "composer_producer_epoch",
                "prior_authority_epoch",
                "authority_epoch",
                "prior_execution_epoch",
                "execution_epoch",
                "cancellation_receipt_id",
                "measurement_id",
                "measurement_hash",
                "producer_activation_id",
                "authority_event_ids",
                "ready_gate_event_id",
                "activation_input_barrier_time",
                "completed_at",
                "clock_domain_id",
                "satisfied_readiness_checks",
                "status",
                "semantic_content_hash",
            )
        },
    )


def project_composition_context_admission(
    context: AuthoritativeTraceContext,
    receipt: CompositionContextAdmissionReceipt,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    context_id = receipt.context_id or "composition-context:rejected-unidentified"
    return context.fact(
        fact_id=f"composition-context-admission:{context_id}",
        event_type="composition_context_admission",
        occurred_at=float(occurred_at),
        payload={
            "context_id": receipt.context_id,
            "accepted": receipt.accepted,
            "rejection_code": receipt.rejection_code,
            "message": receipt.message,
            "available_at": receipt.available_at,
            "available_clock_domain_id": receipt.available_clock_domain_id,
            "control_admission_sequence": receipt.control_admission_sequence,
            "context_semantic_content_hash": receipt.semantic_content_hash,
        },
    )


def project_composition_context(
    context: AuthoritativeTraceContext,
    record: CompositionContextRecord,
) -> ControlTraceFact:
    payload = composition_context_semantic_payload(record)
    payload["context_semantic_content_hash"] = record.semantic_content_hash
    return context.fact(
        fact_id=record.context_id,
        event_type="composition_context",
        occurred_at=record.available_at,
        payload=payload,
    )


def project_composition_context_snapshot(
    context: AuthoritativeTraceContext,
    snapshot: CompositionContextSnapshot,
) -> ControlTraceFact:
    return context.fact(
        fact_id=snapshot.context_snapshot_id,
        event_type="composition_context_snapshot",
        occurred_at=snapshot.cutoff_at,
        payload={
            "context_snapshot_id": snapshot.context_snapshot_id,
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "target_effective_at": snapshot.target_effective_at,
            "required_channels": snapshot.required_channels,
            "selected_context_id_by_channel": (
                snapshot.selected_context_id_by_channel
            ),
            "selected_context_hash_by_channel": (
                snapshot.selected_context_hash_by_channel
            ),
            "missing_channels": snapshot.missing_channels,
            "unavailable_channels": snapshot.unavailable_channels,
            "producer_epoch_snapshot_hash": (
                snapshot.producer_epoch_snapshot_hash
            ),
            "semantic_content_hash": snapshot.semantic_content_hash,
        },
    )


def project_composition_input_admission(
    context: AuthoritativeTraceContext,
    receipt: CompositionInputAdmissionReceipt,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    input_id = receipt.composition_input_id or "composition-input:rejected-unidentified"
    payload = {
        "composition_input_id": receipt.composition_input_id,
        "accepted": receipt.accepted,
        "rejection_code": receipt.rejection_code,
        "message": receipt.message,
        "available_at": receipt.available_at,
        "available_clock_domain_id": receipt.available_clock_domain_id,
        "control_admission_sequence": receipt.control_admission_sequence,
        "input_semantic_content_hash": receipt.semantic_content_hash,
    }
    return context.fact(
        fact_id=f"composition-input-admission:{input_id}",
        event_type="composition_input_admission",
        occurred_at=occurred_at,
        payload=payload,
    )


def project_composition_source_eligibility_spec(
    context: AuthoritativeTraceContext,
    spec: CompositionSourceEligibilitySpec,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=(
            "composition-source-eligibility-spec:"
            f"{spec.semantic_content_hash}"
        ),
        event_type="composition_source_eligibility_spec",
        occurred_at=float(occurred_at),
        payload={
            "spec_id": spec.spec_id,
            "composition_spec_hash": spec.composition_spec_hash,
            "source_slot_by_role": spec.source_slot_by_role,
            "producer_id_by_source_slot": spec.producer_id_by_source_slot,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_composition_source_eligibility_admission(
    context: AuthoritativeTraceContext,
    receipt: CompositionSourceEligibilityAdmissionReceipt,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    state_id = (
        receipt.source_eligibility_state_id
        or "composition-source-eligibility:rejected-unidentified"
    )
    return context.fact(
        fact_id=f"composition-source-eligibility-admission:{state_id}",
        event_type="composition_source_eligibility_admission",
        occurred_at=float(occurred_at),
        payload={
            "source_eligibility_state_id": (
                receipt.source_eligibility_state_id
            ),
            "accepted": receipt.accepted,
            "rejection_code": receipt.rejection_code,
            "message": receipt.message,
            "available_at": receipt.available_at,
            "available_clock_domain_id": receipt.available_clock_domain_id,
            "control_admission_sequence": (
                receipt.control_admission_sequence
            ),
            "state_semantic_content_hash": receipt.semantic_content_hash,
        },
    )


def project_composition_source_eligibility_state(
    context: AuthoritativeTraceContext,
    state: CompositionSourceEligibilityState,
) -> ControlTraceFact:
    payload = source_eligibility_state_semantic_payload(state)
    payload["state_semantic_content_hash"] = state.semantic_content_hash
    return context.fact(
        fact_id=state.source_eligibility_state_id,
        event_type="composition_source_eligibility_state",
        occurred_at=state.available_at,
        payload=payload,
    )


def project_composition_source_eligibility_snapshot(
    context: AuthoritativeTraceContext,
    snapshot: CompositionSourceEligibilitySnapshot,
) -> ControlTraceFact:
    return context.fact(
        fact_id=snapshot.source_eligibility_snapshot_id,
        event_type="composition_source_eligibility_snapshot",
        occurred_at=snapshot.cutoff_at,
        payload={
            "source_eligibility_snapshot_id": (
                snapshot.source_eligibility_snapshot_id
            ),
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": (
                snapshot.cutoff_admission_sequence
            ),
            "target_effective_at": snapshot.target_effective_at,
            "source_eligibility_spec_hash": (
                snapshot.source_eligibility_spec_hash
            ),
            "selected_state_ids_by_role": {
                role: state.source_eligibility_state_id
                for role, state in snapshot.selected_state_by_role.items()
            },
            "selected_state_hashes_by_role": {
                role: state.semantic_content_hash
                for role, state in snapshot.selected_state_by_role.items()
            },
            "ineligible_reason_by_role": (
                snapshot.ineligible_reason_by_role
            ),
            "semantic_content_hash": snapshot.semantic_content_hash,
        },
    )


def project_composition_source_eligibility_binding(
    context: AuthoritativeTraceContext,
    binding: CompositionSourceEligibilityBinding,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=binding.binding_id,
        event_type="composition_source_eligibility_binding",
        occurred_at=float(occurred_at),
        payload={
            "binding_id": binding.binding_id,
            "composition_input_snapshot_id": (
                binding.composition_input_snapshot_id
            ),
            "composition_input_snapshot_hash": (
                binding.composition_input_snapshot_hash
            ),
            "source_eligibility_snapshot_id": (
                binding.source_eligibility_snapshot_id
            ),
            "source_eligibility_snapshot_hash": (
                binding.source_eligibility_snapshot_hash
            ),
            "semantic_content_hash": binding.semantic_content_hash,
        },
    )


def project_composition_spec(
    context: AuthoritativeTraceContext,
    spec: CompositionSpec,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    """Project the full immutable numerical contract, not only its hash."""

    return context.fact(
        fact_id=f"composition-spec:{spec.semantic_content_hash}",
        event_type="composition_spec",
        occurred_at=float(occurred_at),
        payload={
            "spec_id": spec.spec_id,
            "algebra": spec.algebra,
            "input_roles": spec.input_roles,
            "required_roles": spec.required_roles,
            "missing_required_behavior": spec.missing_required_behavior,
            "missing_residual_behavior": spec.missing_residual_behavior,
            "residual_limit_by_component": spec.residual_limit_by_component,
            "weight_by_role": spec.weight_by_role,
            "max_translation_disagreement_m": (
                spec.max_translation_disagreement_m
            ),
            "max_rotation_disagreement_rad": (
                spec.max_rotation_disagreement_rad
            ),
            "pose_component_dependencies": spec.pose_component_dependencies,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_component_composition_rule_set(
    context: AuthoritativeTraceContext,
    rule_set: ComponentCompositionRuleSet,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(rule_set, ComponentCompositionRuleSet):
        raise TypeError("rule_set must be a ComponentCompositionRuleSet.")
    return context.fact(
        fact_id=(
            "component-composition-rule-set:"
            f"{rule_set.semantic_content_hash}"
        ),
        event_type="component_composition_rule_set",
        occurred_at=float(occurred_at),
        payload={
            **rule_set.semantic_payload(),
            "semantic_content_hash": rule_set.semantic_content_hash,
        },
    )


def project_formal_component_composition_binding(
    context: AuthoritativeTraceContext,
    binding: FormalComponentCompositionBinding,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(binding, FormalComponentCompositionBinding):
        raise TypeError(
            "binding must be a FormalComponentCompositionBinding."
        )
    return context.fact(
        fact_id=binding.binding_id,
        event_type="formal_component_composition_binding",
        occurred_at=float(occurred_at),
        payload={
            "binding_id": binding.binding_id,
            "composition_spec_hash": binding.composition_spec_hash,
            "component_rule_set_hash": binding.component_rule_set_hash,
            "formal_strategy_spec_hash": (
                binding.formal_strategy_spec_hash
            ),
            "semantic_content_hash": binding.semantic_content_hash,
        },
    )


def project_composition_input(
    context: AuthoritativeTraceContext,
    record: CompositionInputRecord,
) -> ControlTraceFact:
    payload = composition_input_semantic_payload(record)
    # Activation v2 lineage is emitted as a separate binding fact so the
    # established v1 input record remains byte/schema compatible.
    payload.pop("composition_activation_id", None)
    payload.pop("composition_activation_hash", None)
    payload["input_semantic_content_hash"] = record.semantic_content_hash
    return context.fact(
        fact_id=record.composition_input_id,
        event_type=(
            "composition_contributor_input"
            if record.source_fact_ids
            else "composition_input"
        ),
        occurred_at=record.available_at,
        payload=payload,
    )


def project_formal_composition_strategy_spec(
    context: AuthoritativeTraceContext,
    spec: FormalCompositionStrategySpec,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(spec, FormalCompositionStrategySpec):
        raise TypeError(
            "spec must be a FormalCompositionStrategySpec."
        )
    return context.fact(
        fact_id=f"formal-composition-strategy-spec:{spec.semantic_content_hash}",
        event_type="formal_composition_strategy_spec",
        occurred_at=float(occurred_at),
        payload={
            "strategy_id": spec.strategy_id,
            "strategy_version": spec.strategy_version,
            "execution_class": "inline_bounded",
            "composition_spec_hash": spec.composition_spec_hash,
            "required_context_channels": spec.required_context_channels,
            "parameter_payload": spec.parameter_payload,
            "maximum_decision_latency_sec": (
                spec.maximum_decision_latency_sec
            ),
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_formal_composition_strategy_input(
    context: AuthoritativeTraceContext,
    strategy_input: FormalCompositionStrategyInput,
) -> ControlTraceFact:
    if not isinstance(strategy_input, FormalCompositionStrategyInput):
        raise TypeError(
            "strategy_input must be a FormalCompositionStrategyInput."
        )
    context_snapshot = strategy_input.context_snapshot
    information_plan = strategy_input.information_plan
    return context.fact(
        fact_id=strategy_input.strategy_input_id,
        event_type="formal_composition_strategy_input",
        occurred_at=strategy_input.input_snapshot.cutoff_at,
        payload={
            "strategy_input_id": strategy_input.strategy_input_id,
            "runtime_session_id": strategy_input.runtime_session_id,
            "control_domain_id": strategy_input.control_domain_id,
            "clock_domain_id": strategy_input.clock_domain_id,
            "strategy_id": strategy_input.strategy_spec.strategy_id,
            "strategy_spec_hash": (
                strategy_input.strategy_spec.semantic_content_hash
            ),
            "composition_spec_hash": (
                strategy_input.composition_spec.semantic_content_hash
            ),
            "composition_input_snapshot_id": (
                strategy_input.input_snapshot.composition_input_snapshot_id
            ),
            "composition_input_snapshot_hash": (
                strategy_input.input_snapshot.semantic_content_hash
            ),
            "context_snapshot_id": (
                None
                if context_snapshot is None
                else context_snapshot.context_snapshot_id
            ),
            "context_snapshot_hash": (
                None
                if context_snapshot is None
                else context_snapshot.semantic_content_hash
            ),
            "information_plan_hash": (
                None
                if information_plan is None
                else information_plan.semantic_content_hash
            ),
            "semantic_content_hash": strategy_input.semantic_content_hash,
        },
    )


def project_formal_composition_strategy_result_binding(
    context: AuthoritativeTraceContext,
    binding: FormalCompositionStrategyResultBinding,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(binding, FormalCompositionStrategyResultBinding):
        raise TypeError(
            "binding must be a FormalCompositionStrategyResultBinding."
        )
    return context.fact(
        fact_id=binding.binding_id,
        event_type="formal_composition_strategy_result_binding",
        occurred_at=float(occurred_at),
        payload={
            "binding_id": binding.binding_id,
            "strategy_input_id": binding.strategy_input_id,
            "strategy_input_hash": binding.strategy_input_hash,
            "composition_result_id": binding.composition_result_id,
            "composition_result_hash": binding.composition_result_hash,
            "semantic_content_hash": binding.semantic_content_hash,
        },
    )


def project_composition_input_snapshot(
    context: AuthoritativeTraceContext,
    snapshot: CompositionInputSnapshot,
) -> ControlTraceFact:
    payload = {
        "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
        "control_domain_id": snapshot.control_domain_id,
        "cutoff_at": snapshot.cutoff_at,
        "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
        "target_effective_at": snapshot.target_effective_at,
        "composition_spec_hash": snapshot.composition_spec_hash,
        "selected_input_ids_by_role": {
            role: record.composition_input_id
            for role, record in snapshot.selected_input_by_role.items()
        },
        "selected_input_hashes_by_role": {
            role: record.semantic_content_hash
            for role, record in snapshot.selected_input_by_role.items()
        },
        "missing_roles": snapshot.missing_roles,
        "producer_epoch_snapshot_hash": snapshot.producer_epoch_snapshot_hash,
        "semantic_content_hash": snapshot.semantic_content_hash,
    }
    return context.fact(
        fact_id=snapshot.composition_input_snapshot_id,
        event_type="composition_input_snapshot",
        occurred_at=snapshot.cutoff_at,
        payload=payload,
    )


def project_composition_result(
    context: AuthoritativeTraceContext,
    result: CompositionResult,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    payload = {
        "composition_result_id": result.composition_result_id,
        "composition_input_snapshot_id": result.composition_input_snapshot_id,
        "composition_input_snapshot_hash": result.composition_input_snapshot_hash,
        "composition_spec_hash": result.composition_spec_hash,
        "output_status": result.output_status,
        "reason": result.reason,
        "composed_action": (
            None
            if result.composed_action is None
            else robot_action_semantic_payload(result.composed_action)
        ),
        "composed_action_hash": result.composed_action_hash,
        "input_ids": result.input_ids,
        "input_hashes": result.input_hashes,
        "context_ids": result.context_ids,
        "semantic_content_hash": result.semantic_content_hash,
    }
    return context.fact(
        fact_id=result.composition_result_id,
        event_type="composition_result",
        occurred_at=occurred_at,
        payload=payload,
    )


def project_composition_activation_scope(
    context: AuthoritativeTraceContext,
    scope: CompositionActivationScope,
) -> ControlTraceFact:
    if not isinstance(scope, CompositionActivationScope):
        raise TypeError("scope must be a CompositionActivationScope.")
    return context.fact(
        fact_id=scope.activation_id,
        event_type="composition_activation_scope",
        occurred_at=scope.activated_at,
        payload={
            "activation_id": scope.activation_id,
            "operation_spec_id": scope.operation_spec_id,
            "operation_spec_hash": scope.operation_spec_hash,
            "session_epoch": scope.session_epoch,
            "activated_at": scope.activated_at,
            "clock_domain_id": scope.clock_domain_id,
            "input_barrier_at": scope.input_barrier_at,
            "freshness_policy": scope.freshness_policy,
            "required_roles": scope.required_roles,
            "semantic_content_hash": scope.semantic_content_hash,
        },
    )


def project_composition_activation_binding(
    context: AuthoritativeTraceContext,
    *,
    subject_kind: str,
    subject_id: str,
    subject_hash: str,
    activation_id: str,
    activation_hash: str,
    occurred_at: float,
) -> ControlTraceFact:
    binding = CompositionActivationBinding(
        subject_kind=subject_kind,
        subject_id=subject_id,
        subject_hash=subject_hash,
        activation_id=activation_id,
        activation_hash=activation_hash,
    )
    return context.fact(
        fact_id=binding.binding_id,
        event_type="composition_activation_binding",
        occurred_at=float(occurred_at),
        payload={
            "binding_id": binding.binding_id,
            "subject_kind": binding.subject_kind,
            "subject_id": binding.subject_id,
            "subject_hash": binding.subject_hash,
            "activation_id": binding.activation_id,
            "activation_hash": binding.activation_hash,
            "semantic_content_hash": binding.semantic_content_hash,
        },
    )


def project_composition_information_plan(
    context: AuthoritativeTraceContext,
    plan: CompositionInformationPlan,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(plan, CompositionInformationPlan):
        raise TypeError("plan must be a CompositionInformationPlan.")
    return context.fact(
        fact_id=f"composition-information-plan:{plan.semantic_content_hash}",
        event_type="composition_information_plan",
        occurred_at=float(occurred_at),
        payload={
            "plan_id": plan.plan_id,
            "context_requirements": tuple(
                {
                    "channel": item.channel,
                    "accepted_payload_contract_hashes": (
                        item.accepted_payload_contract_hashes
                    ),
                    "payload_access": item.payload_access,
                    "maximum_age_sec": item.maximum_age_sec,
                }
                for item in plan.context_requirements
            ),
            "source_difference_spec_hash": plan.source_difference_spec_hash,
            "diagnostic_level": plan.diagnostic_level,
            "semantic_content_hash": plan.semantic_content_hash,
        },
    )


def project_composition_activation_binding_record(
    context: AuthoritativeTraceContext,
    binding: CompositionActivationBinding,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(binding, CompositionActivationBinding):
        raise TypeError("binding must be a CompositionActivationBinding.")
    return context.fact(
        fact_id=binding.binding_id,
        event_type="composition_activation_binding",
        occurred_at=float(occurred_at),
        payload={
            "binding_id": binding.binding_id,
            "subject_kind": binding.subject_kind,
            "subject_id": binding.subject_id,
            "subject_hash": binding.subject_hash,
            "activation_id": binding.activation_id,
            "activation_hash": binding.activation_hash,
            "semantic_content_hash": binding.semantic_content_hash,
        },
    )


def project_composition_authority_disposition(
    context: AuthoritativeTraceContext,
    disposition: CompositionAuthorityDisposition,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=disposition.disposition_id,
        event_type="composition_authority_disposition",
        occurred_at=float(occurred_at),
        payload={
            "disposition_id": disposition.disposition_id,
            "gate_spec_hash": disposition.gate_spec_hash,
            "composition_result_id": disposition.composition_result_id,
            "composition_result_hash": disposition.composition_result_hash,
            "action": disposition.action,
            "reason": disposition.reason,
            "missing_roles": disposition.missing_roles,
            "semantic_content_hash": disposition.semantic_content_hash,
        },
    )


def project_source_difference_observation(
    context: AuthoritativeTraceContext,
    observation: SourceDifferenceObservation,
) -> ControlTraceFact:
    return context.fact(
        fact_id=observation.observation_id,
        event_type="source_difference_observation",
        occurred_at=observation.evaluated_at,
        payload={
            "observation_id": observation.observation_id,
            "composition_input_snapshot_id": observation.composition_input_snapshot_id,
            "composition_input_snapshot_hash": observation.composition_input_snapshot_hash,
            "composition_result_id": observation.composition_result_id,
            "composition_result_hash": observation.composition_result_hash,
            "composition_spec_hash": observation.composition_spec_hash,
            "source_difference_spec_hash": observation.source_difference_spec_hash,
            "target_effective_at": observation.target_effective_at,
            "evaluated_at": observation.evaluated_at,
            "clock_domain_id": observation.clock_domain_id,
            "valid_until": observation.valid_until,
            "pairwise_differences": tuple(
                action_difference_semantic_payload(item)
                for item in observation.pairwise_differences
            ),
            "output_residuals": tuple(
                action_difference_semantic_payload(item)
                for item in observation.output_residuals
            ),
            "semantic_content_hash": observation.semantic_content_hash,
        },
    )


def project_source_difference_spec(
    context: AuthoritativeTraceContext,
    spec: SourceDifferenceSpec,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=f"source-difference-spec:{spec.semantic_content_hash}",
        event_type="source_difference_spec",
        occurred_at=occurred_at,
        payload={
            "spec_id": spec.spec_id,
            "composition_spec_hash": spec.composition_spec_hash,
            "compared_reference_role_pairs": spec.compared_reference_role_pairs,
            "output_residual_roles": spec.output_residual_roles,
            "semantic_content_hash": spec.semantic_content_hash,
        },
    )


def project_guidance_feedback_command(
    context: AuthoritativeTraceContext,
    command: GuidanceFeedbackCommand,
) -> ControlTraceFact:
    return context.fact(
        fact_id=command.command_id,
        event_type="guidance_feedback_command",
        occurred_at=command.produced_at,
        payload={
            "command_id": command.command_id,
            "spec_hash": command.spec_hash,
            "source_observation_id": command.source_observation_id,
            "source_observation_hash": command.source_observation_hash,
            "sequence": command.sequence,
            "status": command.status,
            "reason": command.reason,
            "produced_at": command.produced_at,
            "valid_until": command.valid_until,
            "clock_domain_id": command.clock_domain_id,
            "cues": tuple(
                {
                    "component_id": cue.component_id,
                    "translation_direction_xyz": cue.translation_direction_xyz,
                    "translation_intensity": cue.translation_intensity,
                    "rotation_axis_xyz": cue.rotation_axis_xyz,
                    "rotation_intensity": cue.rotation_intensity,
                }
                for cue in command.cues
            ),
            "feedback_kind": command.feedback_kind,
            "authoritative": command.authoritative,
            "semantic_content_hash": command.semantic_content_hash,
        },
    )


def project_guidance_feedback_receipt(
    context: AuthoritativeTraceContext,
    receipt: GuidanceFeedbackReceipt,
) -> ControlTraceFact:
    return context.fact(
        fact_id=receipt.receipt_id,
        event_type="guidance_feedback_receipt",
        occurred_at=receipt.observed_at,
        payload={
            "receipt_id": receipt.receipt_id,
            "command_id": receipt.command_id,
            "command_hash": receipt.command_hash,
            "adapter_id": receipt.adapter_id,
            "accepted": receipt.accepted,
            "status": receipt.status,
            "observed_at": receipt.observed_at,
            "clock_domain_id": receipt.clock_domain_id,
            "semantic_content_hash": receipt.semantic_content_hash,
        },
    )


def project_component_mixing_shadow(
    context: AuthoritativeTraceContext,
    decision: ComponentMixingShadowDecision,
    *,
    evaluated_at: float,
) -> ControlTraceFact:
    """Project a non-authoritative arm/hand composition preview."""

    if not isinstance(decision, ComponentMixingShadowDecision):
        raise TypeError("decision must be a ComponentMixingShadowDecision.")
    hand_decisions = decision.hand_decisions_by_side
    shadow_id = f"component-mixing-shadow:{decision.semantic_content_hash}"
    return context.fact(
        fact_id=shadow_id,
        event_type="component_mixing_shadow",
        occurred_at=evaluated_at,
        payload={
            "shadow_id": shadow_id,
            "status": decision.status,
            "reason": decision.reason,
            "non_authoritative": True,
            "arm_decision_id": decision.arm_decision.decision_id,
            "arm_decision_hash": decision.arm_decision.semantic_content_hash,
            "arm_strategy_id": decision.arm_decision.strategy_id,
            "arm_strategy_spec_hash": decision.arm_decision.strategy_spec_hash,
            "arm_input_snapshot_id": decision.arm_decision.input_snapshot_id,
            "arm_input_snapshot_hash": decision.arm_decision.input_snapshot_hash,
            "hand_decision_id_by_side": {
                side: item.decision_id
                for side, item in sorted(hand_decisions.items())
            },
            "hand_decision_hash_by_side": {
                side: item.semantic_content_hash
                for side, item in sorted(hand_decisions.items())
            },
            "hand_strategy_id_by_side": {
                side: item.strategy_id
                for side, item in sorted(hand_decisions.items())
            },
            "hand_strategy_spec_hash_by_side": {
                side: item.strategy_spec_hash
                for side, item in sorted(hand_decisions.items())
            },
            "hand_input_snapshot_id_by_side": {
                side: item.input_snapshot_id
                for side, item in sorted(hand_decisions.items())
            },
            "hand_input_snapshot_hash_by_side": {
                side: item.input_snapshot_hash
                for side, item in sorted(hand_decisions.items())
            },
            "component_decision_hash_by_id": decision.component_decision_hash_by_id,
            "output_action": (
                None
                if decision.output_action is None
                else robot_action_semantic_payload(decision.output_action)
            ),
            "output_action_hash": decision.output_action_hash,
            "evaluated_at": float(evaluated_at),
            "semantic_content_hash": decision.semantic_content_hash,
        },
    )


def project_composition_domain_event(
    context: AuthoritativeTraceContext,
    event: CompositionDomainEvent,
) -> ControlTraceFact:
    """Project one protocol-neutral event without changing its domain identity."""

    if not isinstance(event, CompositionDomainEvent):
        raise TypeError("event must be a CompositionDomainEvent.")
    kind = event.event_kind
    subject = event.subject
    if kind == "composition_information_plan":
        return project_composition_information_plan(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_context_admission":
        return project_composition_context_admission(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_context":
        return project_composition_context(context, subject)
    if kind == "composition_context_snapshot":
        return project_composition_context_snapshot(context, subject)
    if kind == "composition_input_admission":
        return project_composition_input_admission(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_source_eligibility_spec":
        return project_composition_source_eligibility_spec(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_source_eligibility_admission":
        return project_composition_source_eligibility_admission(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_source_eligibility_state":
        return project_composition_source_eligibility_state(context, subject)
    if kind == "composition_source_eligibility_snapshot":
        return project_composition_source_eligibility_snapshot(
            context,
            subject,
        )
    if kind == "composition_source_eligibility_binding":
        return project_composition_source_eligibility_binding(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind in {"composition_input", "composition_contributor_input"}:
        return project_composition_input(context, subject)
    if kind == "composition_spec":
        return project_composition_spec(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "component_composition_rule_set":
        return project_component_composition_rule_set(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "formal_component_composition_binding":
        return project_formal_component_composition_binding(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_input_snapshot":
        return project_composition_input_snapshot(context, subject)
    if kind == "formal_composition_strategy_spec":
        return project_formal_composition_strategy_spec(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "formal_composition_strategy_input":
        return project_formal_composition_strategy_input(
            context,
            subject,
        )
    if kind == "composition_result":
        return project_composition_result(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "formal_composition_strategy_result_binding":
        return project_formal_composition_strategy_result_binding(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "composition_activation_binding":
        return project_composition_activation_binding_record(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "source_difference_spec":
        return project_source_difference_spec(
            context,
            subject,
            occurred_at=event.occurred_at,
        )
    if kind == "source_difference_observation":
        return project_source_difference_observation(context, subject)
    raise ValueError(f"Unsupported composition domain event kind: {kind}")


class CompositionControlTraceEmitter:
    """Mechanical domain-event to v2.4 adapter with an injected fact sink."""

    def __init__(
        self,
        *,
        trace_context: AuthoritativeTraceContext,
        trace_sink: Callable[[tuple[ControlTraceFact, ...]], object],
    ) -> None:
        if not isinstance(trace_context, AuthoritativeTraceContext):
            raise TypeError("trace_context must be an AuthoritativeTraceContext.")
        if not callable(trace_sink):
            raise TypeError("trace_sink must be callable.")
        self.trace_context = trace_context
        self.trace_sink = trace_sink

    def __call__(self, events: tuple[CompositionDomainEvent, ...]) -> None:
        batch = tuple(events)
        if any(not isinstance(item, CompositionDomainEvent) for item in batch):
            raise TypeError("Composition event batch contains an invalid value.")
        self.trace_sink(
            tuple(
                project_composition_domain_event(self.trace_context, event)
                for event in batch
            )
        )


__all__ = [
    "CompositionControlTraceEmitter",
    "project_composition_activation_binding",
    "project_composition_activation_binding_record",
    "project_composition_activation_scope",
    "project_composition_context",
    "project_composition_context_admission",
    "project_composition_context_snapshot",
    "project_component_composition_rule_set",
    "project_formal_component_composition_binding",
    "project_composition_domain_event",
    "project_composition_information_plan",
    "project_formal_composition_strategy_input",
    "project_formal_composition_strategy_result_binding",
    "project_formal_composition_strategy_spec",
    "project_composition_source_eligibility_admission",
    "project_composition_source_eligibility_binding",
    "project_composition_source_eligibility_snapshot",
    "project_composition_source_eligibility_spec",
    "project_composition_source_eligibility_state",
    "project_composition_session_rearm_receipt",
    "project_composition_session_rearm_spec",
    "project_composition_session_spec",
    "project_composition_session_transition",
    "project_composition_spec",
    "project_component_mixing_shadow",
    "project_composition_authority_disposition",
    "project_composition_input",
    "project_composition_input_admission",
    "project_composition_input_snapshot",
    "project_composition_result",
    "project_guidance_feedback_command",
    "project_guidance_feedback_receipt",
    "project_source_difference_observation",
    "project_source_difference_spec",
]

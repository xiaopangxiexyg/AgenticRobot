"""Canonical trace projections for backend-neutral control composition."""

from __future__ import annotations

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.action.proposal import robot_action_semantic_payload

from .composition import (
    ComponentMixingShadowDecision,
    CompositionAuthorityDisposition,
    CompositionInputAdmissionReceipt,
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionResult,
    CompositionSpec,
    composition_input_semantic_payload,
    CompositionContextAdmissionReceipt,
    CompositionContextRecord,
    CompositionContextSnapshot,
    composition_context_semantic_payload,
)
from .observability import (
    GuidanceFeedbackCommand,
    GuidanceFeedbackReceipt,
    SourceDifferenceObservation,
    SourceDifferenceSpec,
    action_difference_semantic_payload,
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
    return context.fact(
        fact_id=f"composition-input-admission:{input_id}",
        event_type="composition_input_admission",
        occurred_at=occurred_at,
        payload={
            "composition_input_id": receipt.composition_input_id,
            "accepted": receipt.accepted,
            "rejection_code": receipt.rejection_code,
            "message": receipt.message,
            "available_at": receipt.available_at,
            "available_clock_domain_id": receipt.available_clock_domain_id,
            "control_admission_sequence": receipt.control_admission_sequence,
            "input_semantic_content_hash": receipt.semantic_content_hash,
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


def project_composition_input(
    context: AuthoritativeTraceContext,
    record: CompositionInputRecord,
) -> ControlTraceFact:
    payload = composition_input_semantic_payload(record)
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


def project_composition_input_snapshot(
    context: AuthoritativeTraceContext,
    snapshot: CompositionInputSnapshot,
) -> ControlTraceFact:
    return context.fact(
        fact_id=snapshot.composition_input_snapshot_id,
        event_type="composition_input_snapshot",
        occurred_at=snapshot.cutoff_at,
        payload={
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
        },
    )


def project_composition_result(
    context: AuthoritativeTraceContext,
    result: CompositionResult,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=result.composition_result_id,
        event_type="composition_result",
        occurred_at=occurred_at,
        payload={
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


__all__ = [
    "project_composition_context",
    "project_composition_context_admission",
    "project_composition_context_snapshot",
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

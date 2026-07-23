"""ControlTrace projections for the scoped scheduled-qpos sim branch."""

from __future__ import annotations

from robot_motion.action import (
    GroupGenerationActivation,
    ExecutionTargetAdmissionReceipt,
    PendingGroupGenerationReservation,
    ReferenceSegment,
    ReferenceSegmentAdmissionReceipt,
    ReferenceSegmentTerminal,
    ResolvedExecutionTarget,
    SafetyAnchorCommitEvidence,
    SafetyAnchorCommitCandidate,
)
from robot_motion.action.dispatch_context import (
    AuthoritativeReceiverValidationReceipt,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.action.proposal import robot_action_semantic_payload

from ..adapters import (
    MujocoPlannerCommandFrameObservation,
    MujocoReferenceTargetActivationObservation,
)


def _qpos_payload(values):
    return {side: value.tolist() for side, value in values.items()}


def project_reference_receiver_validation(
    context: AuthoritativeTraceContext,
    validation: AuthoritativeReceiverValidationReceipt,
):
    payload = {
        name: getattr(validation, name)
        for name in validation.__dataclass_fields__
    }
    return context.fact(
        fact_id=validation.validation_id,
        event_type="reference_receiver_validation",
        occurred_at=validation.checked_at,
        payload=payload,
    )


def project_reference_segment(context: AuthoritativeTraceContext, segment: ReferenceSegment):
    payload = {
        "reference_segment_id": segment.reference_segment_id,
        "control_domain_id": segment.control_domain_id,
        "selection_group_id": segment.selection_group_id,
        "execution_epoch": segment.execution_epoch,
        "group_segment_generation": segment.group_segment_generation,
        "authority_kind": segment.authority_kind,
        "endpoint_parents": tuple(
            {
                "endpoint_parent_id": parent.endpoint_parent_id,
                "role": parent.role,
                "source_kind": parent.source_kind,
                "source_fact_id": parent.source_fact_id,
                "projected_group_action_hash": parent.projected_group_action_hash,
                "action_schema_hash": parent.action_schema_hash,
                "frame_schema_hash": parent.frame_schema_hash,
                "target_effective_at": parent.target_effective_at,
                "semantic_content_hash": parent.semantic_content_hash,
            }
            for parent in segment.endpoint_parents
        ),
        "component_dependencies": segment.component_dependencies,
        "created_at": segment.created_at,
        "scheduled_from": segment.scheduled_from,
        "scheduled_until": segment.scheduled_until,
        "clock_domain_id": segment.clock_domain_id,
        "interpolation_contract_hash": segment.interpolation_contract_hash,
        "resolver_contract_hash": segment.resolver_contract_hash,
        "planner_contract_hash": segment.planner_contract_hash,
        "input_action_schema_hash": segment.input_action_schema_hash,
        "output_reference_schema_hash": segment.output_reference_schema_hash,
        "required_admission_guarantee": segment.required_admission_guarantee,
        "segment_safety_evidence_level": segment.segment_safety_evidence_level,
        "semantic_content_hash": segment.semantic_content_hash,
    }
    return context.fact(
        fact_id=segment.reference_segment_id,
        event_type="reference_segment",
        occurred_at=segment.created_at,
        payload=payload,
    )


def project_reference_segment_admission(
    context: AuthoritativeTraceContext,
    admission: ReferenceSegmentAdmissionReceipt,
    *,
    receiver_validation: AuthoritativeReceiverValidationReceipt,
):
    payload = {
        name: getattr(admission, name)
        for name in admission.__dataclass_fields__
        if name != "semantic_content_hash"
    }
    payload["semantic_content_hash"] = admission.semantic_content_hash
    payload["receiver_validation_id"] = receiver_validation.validation_id
    payload["receiver_validation_hash"] = receiver_validation.semantic_content_hash
    return context.fact(
        fact_id=admission.admission_id,
        event_type="reference_segment_admission",
        occurred_at=admission.admitted_at,
        payload=payload,
    )


def project_safety_anchor_commit_candidate(
    context: AuthoritativeTraceContext,
    candidate: SafetyAnchorCommitCandidate,
):
    payload = {
        "candidate_id": candidate.candidate_id,
        "reference_segment_id": candidate.reference_segment_id,
        "reference_segment_hash": candidate.reference_segment_hash,
        "admission_id": candidate.admission_id,
        "admission_hash": candidate.admission_hash,
        "admission_sequence": candidate.admission_sequence,
        "control_domain_id": candidate.control_domain_id,
        "selection_group_id": candidate.selection_group_id,
        "execution_epoch": candidate.execution_epoch,
        "group_segment_generation": candidate.group_segment_generation,
        "authority_kind": candidate.authority_kind,
        "safety_result_id": candidate.safety_result_id,
        "safe_action": robot_action_semantic_payload(candidate.safe_action),
        "safe_action_hash": candidate.safe_action_hash,
        "evidence_created_at": candidate.evidence_created_at,
        "effective_from": candidate.effective_from,
        "required_admission_guarantee": candidate.required_admission_guarantee,
        "achieved_admission_guarantee": candidate.achieved_admission_guarantee,
        "semantic_content_hash": candidate.semantic_content_hash,
    }
    return context.fact(
        fact_id=candidate.candidate_id,
        event_type="safety_anchor_commit_candidate",
        occurred_at=candidate.evidence_created_at,
        payload=payload,
    )


def project_resolved_execution_target(
    context: AuthoritativeTraceContext,
    target: ResolvedExecutionTarget,
):
    payload = {
        name: getattr(target, name)
        for name in target.__dataclass_fields__
        if name not in {"group_safe_action", "payload"}
    }
    payload["group_safe_action"] = robot_action_semantic_payload(
        target.group_safe_action
    )
    payload["payload"] = dict(target.payload)
    return context.fact(
        fact_id=target.target_id,
        event_type="resolved_execution_target",
        occurred_at=target.resolved_at,
        payload=payload,
    )


def project_execution_target_admission(
    context: AuthoritativeTraceContext,
    admission: ExecutionTargetAdmissionReceipt,
):
    payload = {
        name: getattr(admission, name)
        for name in admission.__dataclass_fields__
    }
    return context.fact(
        fact_id=admission.admission_id,
        event_type="execution_target_admission",
        occurred_at=admission.admitted_at,
        payload=payload,
    )


def project_safety_anchor_commit_evidence(
    context: AuthoritativeTraceContext,
    evidence: SafetyAnchorCommitEvidence,
):
    payload = {
        name: getattr(evidence, name)
        for name in evidence.__dataclass_fields__
    }
    payload["semantic_content_hash"] = evidence.semantic_content_hash
    return context.fact(
        fact_id=f"safety-anchor-commit-evidence:{evidence.semantic_content_hash}",
        event_type="safety_anchor_commit_evidence",
        occurred_at=evidence.occurred_at,
        payload=payload,
    )


def project_group_generation_reservation(
    context: AuthoritativeTraceContext,
    reservation: PendingGroupGenerationReservation,
):
    payload = {
        name: getattr(reservation, name)
        for name in reservation.__dataclass_fields__
    }
    return context.fact(
        fact_id=reservation.reservation_id,
        event_type="group_generation_reservation",
        occurred_at=reservation.admitted_at,
        payload=payload,
    )


def project_group_generation_activation(
    context: AuthoritativeTraceContext,
    activation: GroupGenerationActivation,
):
    payload = {
        name: getattr(activation, name)
        for name in activation.__dataclass_fields__
    }
    return context.fact(
        fact_id=activation.activation_id,
        event_type="group_generation_activation",
        occurred_at=activation.activated_at,
        payload=payload,
    )


def project_reference_target_activation(
    context: AuthoritativeTraceContext,
    observation: MujocoReferenceTargetActivationObservation,
):
    payload = {
        "observation_id": observation.observation_id,
        "reference_segment_id": observation.reference_segment_id,
        "generation_activation_id": observation.generation_activation_id,
        "reference_sequence": observation.reference_sequence,
        "execution_epoch": observation.execution_epoch,
        "group_segment_generation": observation.group_segment_generation,
        "scheduled_for": observation.scheduled_for,
        "activated_at_worker_time": observation.activated_at_worker_time,
        "observed_at": observation.observed_at,
        "qref_by_side": _qpos_payload(observation.qref_by_side),
        "activation_status_by_side": dict(observation.activation_status_by_side),
        "semantic_content_hash": observation.semantic_content_hash,
    }
    return context.fact(
        fact_id=observation.observation_id,
        event_type="reference_target_activation",
        occurred_at=observation.observed_at,
        payload=payload,
    )


def project_planner_command_frame(
    context: AuthoritativeTraceContext,
    observation: MujocoPlannerCommandFrameObservation,
):
    payload = {
        "observation_id": observation.observation_id,
        "reference_segment_id": observation.reference_segment_id,
        "reference_sequence": observation.reference_sequence,
        "command_sequence": observation.command_sequence,
        "execution_epoch": observation.execution_epoch,
        "group_segment_generation": observation.group_segment_generation,
        "scheduled_for": observation.scheduled_for,
        "observed_at": observation.observed_at,
        "planner_proposal_token_by_side": dict(
            observation.planner_proposal_token_by_side
        ),
        "planner_generation_by_side": dict(observation.planner_generation_by_side),
        "qref_by_side": _qpos_payload(observation.qref_by_side),
        "planner_qcmd_by_side": _qpos_payload(observation.planner_qcmd_by_side),
        "follow_qcmd_by_side": _qpos_payload(observation.follow_qcmd_by_side),
        "accepted_qcmd_by_side": _qpos_payload(observation.accepted_qcmd_by_side),
        "limited_by_side": dict(observation.limited_by_side),
        "semantic_content_hash": observation.semantic_content_hash,
    }
    return context.fact(
        fact_id=observation.observation_id,
        event_type="planner_command_frame",
        occurred_at=observation.observed_at,
        payload=payload,
    )


def project_reference_segment_terminal(
    context: AuthoritativeTraceContext,
    terminal: ReferenceSegmentTerminal,
):
    payload = {
        name: getattr(terminal, name)
        for name in terminal.__dataclass_fields__
        if name != "semantic_content_hash"
    }
    payload["semantic_content_hash"] = terminal.semantic_content_hash
    return context.fact(
        fact_id=terminal.terminal_id,
        event_type="reference_segment_terminal",
        occurred_at=terminal.available_at,
        payload=payload,
    )


__all__ = [
    "project_group_generation_activation",
    "project_group_generation_reservation",
    "project_execution_target_admission",
    "project_planner_command_frame",
    "project_reference_segment",
    "project_reference_segment_admission",
    "project_reference_segment_terminal",
    "project_reference_target_activation",
    "project_reference_receiver_validation",
    "project_resolved_execution_target",
    "project_safety_anchor_commit_evidence",
    "project_safety_anchor_commit_candidate",
]

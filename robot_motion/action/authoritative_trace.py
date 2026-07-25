"""Shared projections for authoritative control facts.

This module is intentionally backend-neutral.  Real and simulated frontends
may emit the same explicit v2.4 facts without importing one another or passing
runtime handles into the dataset contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.replay import ReplayExecutionAuthorization
from robot_motion.runtime import CommandDispatchReceipt, TeleopCommand
from robot_motion.runtime.dispatch_owner import (
    DispatchPermit,
    DispatchOwnerToken,
    DispatchPermitValidationReceipt,
    MotionDispatchPermit,
)
from robot_motion.runtime.cancellation import (
    CancellationAcknowledgement,
    ExecutionCancellation,
)
from robot_motion.runtime.producer_epoch import ProducerEpochActivation
from robot_motion.runtime.observation_cut import ObservationCutRecord
from robot_motion.runtime.execution_gate import ExecutionGateEvent

from .arbitration_input import ArbitrationInputSnapshot
from .candidate_authorization import (
    CandidateAuthorizationAdmissionReceipt,
    CandidateAuthorizationDraft,
)
from .compound_authority import ConservativeCompoundAuthorityProfile
from .arbitration_runtime import (
    ArbitrationComputationMetrics,
    ArbitrationPublicationAttempt,
    AuthoritativeArbitrationDecision,
)
from .arbiter import ArbitrationResult
from .authority import AuthorityEvent
from .executor import teleop_command_semantic_payload
from .execution_shadow import SharedExecutionShadowEvaluation
from .execution_manifest import ControlExecutionManifest
from .persistence import ExecutionDirective
from .proposal import (
    ProposalAdmissionReceipt,
    action_proposal_semantic_payload,
    robot_action_semantic_payload,
)
from .safety import SharedCanonicalSafetyResult

AUTHORITATIVE_COMMAND_HASH_DOMAIN_V1 = "dexteleop.authoritative_command.v1"
AUTHORITATIVE_SAFETY_HASH_DOMAIN_V1 = "dexteleop.authoritative_safety_result.v1"
AUTHORITATIVE_DISPATCH_HASH_DOMAIN_V1 = "dexteleop.authoritative_dispatch_result.v1"
MEASUREMENT_ALIGNMENT_HASH_DOMAIN_V1 = "dexteleop.measurement_alignment.v1"
PUBLICATION_REJECTION_HASH_DOMAIN_V1 = "dexteleop.publication_rejection.v1"
COMPOUND_AUTHORITY_FAILURE_HASH_DOMAIN_V1 = (
    "dexteleop.compound_authority_failure.v1"
)


@dataclass(frozen=True)
class AuthoritativeTraceContext:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    clock_domain_id: str

    def fact(
        self,
        *,
        fact_id: str,
        event_type: str,
        occurred_at: float,
        payload: Mapping[str, Any],
    ) -> ControlTraceFact:
        return ControlTraceFact.create(
            fact_id=fact_id,
            event_type=event_type,
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            embodiment_instance_id=self.embodiment_instance_id,
            occurred_at=float(occurred_at),
            clock_domain_id=self.clock_domain_id,
            payload=payload,
        )


def project_control_execution_manifest(
    context: AuthoritativeTraceContext,
    manifest: ControlExecutionManifest,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    """Project immutable runtime configuration as an episode dependency fact."""

    if not isinstance(manifest, ControlExecutionManifest):
        raise TypeError("manifest must be a ControlExecutionManifest.")
    return context.fact(
        fact_id=f"execution-manifest:{manifest.semantic_content_hash}",
        event_type="execution_manifest",
        occurred_at=occurred_at,
        payload={
            "manifest_id": manifest.manifest_id,
            "manifest_version": manifest.manifest_version,
            "action_space": manifest.action_space,
            "component_catalog_hash": manifest.component_catalog_hash,
            "controlled_components": manifest.controlled_components,
            "selection_groups": manifest.selection_groups,
            "motion_authority_sources": manifest.motion_authority_sources,
            "human_motion_proposal_status": manifest.human_motion_proposal_status,
            "safety_control_sources": manifest.safety_control_sources,
            "safety_control_requests": manifest.safety_control_requests,
            "lifecycle_control_sources": manifest.lifecycle_control_sources,
            "lifecycle_control_requests": manifest.lifecycle_control_requests,
            "authoritative_decision_hz": manifest.authoritative_decision_hz,
            "reference_activation_spec_id": manifest.reference_activation_spec_id,
            "reference_activation_spec_hash": manifest.reference_activation_spec_hash,
            "execution_capability_profile_hash": (
                manifest.execution_capability_profile_hash
            ),
            "reference_timing_semantics_hz": (
                manifest.reference_timing_semantics_hz
            ),
            "actual_reference_frame_producer": (
                manifest.actual_reference_frame_producer
            ),
            "planner_command_hz": manifest.planner_command_hz,
            "exclusions": tuple(
                {
                    "capability_id": exclusion.capability_id,
                    "reason": exclusion.reason,
                    "evidence_ids": exclusion.evidence_ids,
                    "semantic_content_hash": exclusion.semantic_content_hash,
                }
                for exclusion in manifest.exclusions
            ),
            "semantic_content_hash": manifest.semantic_content_hash,
        },
    )


def project_conservative_compound_authority_profile(
    context: AuthoritativeTraceContext,
    profile: ConservativeCompoundAuthorityProfile,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    """Project the exact fail-closed compound authority contract."""

    if not isinstance(profile, ConservativeCompoundAuthorityProfile):
        raise TypeError(
            "profile must be a ConservativeCompoundAuthorityProfile."
        )
    return context.fact(
        fact_id=profile.profile_fact_id,
        event_type="conservative_compound_authority_profile",
        occurred_at=occurred_at,
        payload={
            **profile.semantic_payload(),
            "semantic_content_hash": profile.semantic_content_hash,
        },
    )


def project_compound_authority_failure(
    context: AuthoritativeTraceContext,
    profile: ConservativeCompoundAuthorityProfile,
    *,
    stage: str,
    source_event_ids: tuple[str, ...],
    occurred_at: float,
) -> ControlTraceFact:
    """Record why an authoritative compound attempt entered domain HOLD."""

    normalized_stage = str(stage).strip()
    if not normalized_stage:
        raise ValueError("Compound authority failure stage must be non-empty.")
    sources = tuple(dict.fromkeys(str(value) for value in source_event_ids))
    if not sources or any(not value for value in sources):
        raise ValueError(
            "Compound authority failure requires non-empty source_event_ids."
        )
    payload = {
        "profile_hash": profile.semantic_content_hash,
        "producer_id": profile.producer_id,
        "stage": normalized_stage,
        "failure_behavior": "hold_control_domain",
        "source_event_ids": sources,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=COMPOUND_AUTHORITY_FAILURE_HASH_DOMAIN_V1,
    )
    failure_id = f"compound-authority-failure:{semantic_hash}"
    return context.fact(
        fact_id=failure_id,
        event_type="compound_authority_failure",
        occurred_at=occurred_at,
        payload={
            "failure_id": failure_id,
            **payload,
            "semantic_content_hash": semantic_hash,
        },
    )


def _proposal_snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "proposal_snapshot_id": snapshot.proposal_snapshot_id,
        "control_domain_id": snapshot.control_domain_id,
        "cutoff_at": snapshot.cutoff_at,
        "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
        "included_admission_sequence_min": snapshot.included_admission_sequence_min,
        "included_admission_sequence_max": snapshot.included_admission_sequence_max,
        "target_effective_at": snapshot.target_effective_at,
        "component_schema_hash": snapshot.component_schema_hash,
        "producer_epoch_snapshot_hash": snapshot.producer_epoch_snapshot_hash,
        "reduction_policy_id": snapshot.reduction_policy_id,
        "candidates": tuple(
            {
                "selection_group_id": candidate.selection_group_id,
                "proposal_id": candidate.proposal.proposal_id,
                "proposal_content_hash": candidate.proposal.semantic_content_hash,
            }
            for candidate in snapshot.candidates
        ),
        "dispositions": tuple(
            {
                "proposal_id": item.proposal_id,
                "selection_group_id": item.selection_group_id,
                "disposition": item.disposition,
                "detail": item.detail,
            }
            for item in snapshot.dispositions
        ),
        "semantic_content_hash": snapshot.semantic_content_hash,
    }


def _tick_payload(tick: Any) -> dict[str, Any]:
    return {
        "schedule_tick_id": tick.schedule_tick_id,
        "tick_index": tick.tick_index,
        "scheduled_decision_at": tick.scheduled_decision_at,
        "decision_cutoff_at": tick.decision_cutoff_at,
        "cutoff_admission_sequence": tick.cutoff_admission_sequence,
        "target_effective_at": tick.target_effective_at,
        "publication_deadline_at": tick.publication_deadline_at,
        "trigger_ids": tick.trigger_ids,
        "trigger_reasons": tick.trigger_reasons,
        "coalesced_trigger_ids": tick.coalesced_trigger_ids,
        "coalesced_trigger_count": tick.coalesced_trigger_count,
        "trigger_evidence_truncated": tick.trigger_evidence_truncated,
        "schedule_spec_hash": tick.schedule_spec_hash,
        "semantic_content_hash": tick.semantic_content_hash,
    }


def project_producer_epoch_activation(
    context: AuthoritativeTraceContext,
    activation: ProducerEpochActivation,
) -> ControlTraceFact:
    payload = {
        "activation_id": activation.activation_id,
        "runtime_session_id": activation.runtime_session_id,
        "control_domain_id": activation.control_domain_id,
        "producer_id": activation.producer_id,
        "producer_epoch": activation.producer_epoch,
        "previous_epoch": activation.previous_epoch,
        "activation_sequence": activation.activation_sequence,
        "produced_at": activation.produced_at,
        "produced_clock_domain_id": activation.produced_clock_domain_id,
        "available_at": activation.available_at,
        "available_clock_domain_id": activation.available_clock_domain_id,
        "activated_at": activation.activated_at,
        "control_admission_sequence": activation.control_admission_sequence,
        "reason": activation.reason,
        "semantic_content_hash": activation.semantic_content_hash,
    }
    return context.fact(
        fact_id=activation.activation_id,
        event_type="producer_epoch_activation",
        occurred_at=activation.available_at,
        payload=payload,
    )


def project_authority_event(
    context: AuthoritativeTraceContext,
    event: AuthorityEvent,
) -> ControlTraceFact:
    """Project one receiver-admitted authority transition."""

    if not isinstance(event, AuthorityEvent):
        raise TypeError("event must be an AuthorityEvent.")
    return context.fact(
        fact_id=event.event_id,
        event_type="authority_event",
        occurred_at=event.available_at,
        payload={
            "event_id": event.event_id,
            "runtime_session_id": event.runtime_session_id,
            "control_domain_id": event.control_domain_id,
            "sequence": event.sequence,
            "authority_epoch": event.authority_epoch,
            "authority_event_type": event.event_type,
            "selection_group_id": event.selection_group_id,
            "owner_producer_id": event.owner_producer_id,
            "control_mode": event.control_mode,
            "expires_at": event.expires_at,
            "expires_clock_domain_id": event.expires_clock_domain_id,
            "produced_at": event.produced_at,
            "produced_clock_domain_id": event.produced_clock_domain_id,
            "available_at": event.available_at,
            "available_clock_domain_id": event.available_clock_domain_id,
            "control_admission_sequence": event.control_admission_sequence,
            "reason": event.reason,
            "semantic_content_hash": event.semantic_content_hash,
        },
    )


def project_observation_cut(
    context: AuthoritativeTraceContext,
    record: ObservationCutRecord,
) -> ControlTraceFact:
    return context.fact(
        fact_id=record.observation_cut_id,
        event_type="observation_cut",
        occurred_at=record.built_at,
        payload={
            "observation_cut_id": record.observation_cut_id,
            "runtime_session_id": record.runtime_session_id,
            "control_domain_id": record.control_domain_id,
            "policy_node_id": record.policy_node_id,
            "policy_node_epoch": record.policy_node_epoch,
            "sequence": record.sequence,
            "cutoff_at": record.cutoff_at,
            "cutoff_clock_domain_id": record.cutoff_clock_domain_id,
            "cutoff_clock_reset_epoch": record.cutoff_clock_reset_epoch,
            "cutoff_admission_sequence": record.cutoff_admission_sequence,
            "built_at": record.built_at,
            "built_clock_domain_id": record.built_clock_domain_id,
            "selected_sample_by_stream": {
                stream_id: sample.semantic_payload(
                    cutoff_at=record.cutoff_at,
                    cutoff_clock_domain_id=record.cutoff_clock_domain_id,
                )
                for stream_id, sample in record.selected_sample_by_stream.items()
            },
            "context_dependencies": record.context_dependencies,
            "observation_contract_hash": record.observation_contract_hash,
            "policy_io_layout_hash": record.policy_io_layout_hash,
            "trigger_reason": record.trigger_reason,
            "semantic_content_hash": record.semantic_content_hash,
        },
    )


def project_execution_gate_event(
    context: AuthoritativeTraceContext,
    event: ExecutionGateEvent,
) -> ControlTraceFact:
    return context.fact(
        fact_id=event.event_id,
        event_type="execution_gate_event",
        occurred_at=event.available_at,
        payload={
            "event_id": event.event_id,
            "runtime_session_id": event.runtime_session_id,
            "control_domain_id": event.control_domain_id,
            "sequence": event.sequence,
            "gate_epoch": event.gate_epoch,
            "status": event.status,
            "reason": event.reason,
            "source_event_ids": event.source_event_ids,
            "produced_at": event.produced_at,
            "produced_clock_domain_id": event.produced_clock_domain_id,
            "available_at": event.available_at,
            "available_clock_domain_id": event.available_clock_domain_id,
            "control_admission_sequence": event.control_admission_sequence,
            "semantic_content_hash": event.semantic_content_hash,
        },
    )


def project_proposal_admission(
    context: AuthoritativeTraceContext,
    receipt: ProposalAdmissionReceipt,
    *,
    source_event_id: str,
    occurred_at: float,
) -> tuple[ControlTraceFact, ...]:
    admission_id = f"proposal-admission:{source_event_id}"
    facts = [
        context.fact(
            fact_id=admission_id,
            event_type="proposal_admission",
            occurred_at=(
                float(receipt.available_at)
                if receipt.available_at is not None
                else float(occurred_at)
            ),
            payload={
                "emission_id": source_event_id,
                "accepted": receipt.accepted,
                "rejection_code": receipt.rejection_code,
                "message": receipt.message,
                "proposal_id": receipt.proposal_id,
                "available_at": receipt.available_at,
                "available_clock_domain_id": receipt.available_clock_domain_id,
                "control_admission_sequence": receipt.control_admission_sequence,
                "proposal_semantic_content_hash": receipt.semantic_content_hash,
            },
        )
    ]
    if receipt.proposal is not None:
        proposal = receipt.proposal
        facts.append(
            context.fact(
                fact_id=proposal.proposal_id,
                event_type="action_proposal",
                occurred_at=proposal.available_at,
                payload={
                    **action_proposal_semantic_payload(proposal),
                    "composition_input_ids": proposal.composition_input_ids,
                    "composition_input_hashes": proposal.composition_input_hashes,
                    "composition_spec_hash": proposal.composition_spec_hash,
                    "proposal_semantic_hash_domain": proposal.semantic_hash_domain,
                    "proposal_semantic_content_hash": proposal.semantic_content_hash,
                },
            )
        )
    return tuple(facts)


def project_candidate_authorization_admission(
    context: AuthoritativeTraceContext,
    draft: CandidateAuthorizationDraft,
    receipt: CandidateAuthorizationAdmissionReceipt,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    record = receipt.record
    authorization_id = (
        receipt.authorization_id
        or f"candidate-authorization-rejected:{draft.proposal_id}:{draft.evidence_id}"
    )
    return context.fact(
        fact_id=f"candidate-authorization-admission:{authorization_id}",
        event_type="candidate_authorization_admission",
        occurred_at=(
            float(receipt.available_at)
            if receipt.available_at is not None
            else float(occurred_at)
        ),
        payload={
            "authorization_id": receipt.authorization_id,
            "accepted": receipt.accepted,
            "rejection_code": receipt.rejection_code,
            "message": receipt.message,
            "proposal_id": draft.proposal_id,
            "proposal_content_hash": draft.proposal_content_hash,
            "producer_kind": draft.producer_kind,
            "evidence_id": draft.evidence_id,
            "evidence_hash": draft.evidence_hash,
            "binding_eligibility_id": draft.binding_eligibility_id,
            "binding_eligibility_hash": draft.binding_eligibility_hash,
            "target_runtime_contract_hash": draft.target_runtime_contract_hash,
            "authoritative_allowed": draft.authoritative_allowed,
            "reasons": draft.reasons,
            "available_at": receipt.available_at,
            "available_clock_domain_id": receipt.available_clock_domain_id,
            "control_admission_sequence": receipt.control_admission_sequence,
            "semantic_content_hash": (
                None if record is None else record.semantic_content_hash
            ),
        },
    )


def project_replay_execution_authorization(
    context: AuthoritativeTraceContext,
    authorization: ReplayExecutionAuthorization,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=authorization.authorization_id,
        event_type="replay_execution_authorization",
        occurred_at=occurred_at,
        payload={
            "authorization_id": authorization.authorization_id,
            "requested_mode": authorization.requested_mode,
            "assessment_id": authorization.assessment_id,
            "assessment_hash": authorization.assessment_hash,
            "safety_preflight_id": authorization.safety_preflight_id,
            "execution_capability_evidence_id": (
                authorization.execution_capability_evidence_id
            ),
            "dispatch_owner_state_id": authorization.dispatch_owner_state_id,
            "operator_authorization_id": authorization.operator_authorization_id,
            "authorized": authorization.authorized,
            "reasons": authorization.reasons,
            "semantic_content_hash": authorization.semantic_content_hash,
        },
    )
def project_arbitration_input(
    context: AuthoritativeTraceContext,
    snapshot: ArbitrationInputSnapshot,
) -> tuple[ControlTraceFact, ...]:
    input_id = snapshot.arbitration_input_snapshot_id
    occurred_at = snapshot.decision_schedule_tick.scheduled_decision_at
    authority_id = f"authority-state-snapshot:{input_id}"
    gate_id = f"execution-gate-snapshot:{input_id}"
    epoch_id = f"producer-epoch-snapshot:{input_id}"
    candidate_authorization_id = (
        f"candidate-authorization-snapshot:{input_id}"
    )
    authority = snapshot.authority_state_snapshot
    gate = snapshot.execution_gate_snapshot
    epochs = snapshot.producer_epoch_snapshot
    authority_payload = {
        "snapshot_fact_id": authority_id,
        "arbitration_input_snapshot_id": input_id,
        "control_domain_id": authority.control_domain_id,
        "cutoff_at": authority.cutoff_at,
        "cutoff_admission_sequence": authority.cutoff_admission_sequence,
        "included_admission_sequence_min": authority.included_admission_sequence_min,
        "included_admission_sequence_max": authority.included_admission_sequence_max,
        "authority_policy_hash": authority.authority_policy_hash,
        "control_mode": authority.control_mode,
        "authority_epoch": authority.authority_epoch,
        "active_leases": {
            group: {
                "selection_group_id": lease.selection_group_id,
                "owner_producer_id": lease.owner_producer_id,
                "acquired_event_id": lease.acquired_event_id,
                "acquired_at": lease.acquired_at,
                "expires_at": lease.expires_at,
            }
            for group, lease in authority.active_leases.items()
        },
        "causal_event_ids": authority.causal_event_ids,
        "semantic_content_hash": authority.semantic_content_hash,
    }
    gate_payload = {
        "snapshot_fact_id": gate_id,
        "arbitration_input_snapshot_id": input_id,
        "control_domain_id": gate.control_domain_id,
        "cutoff_at": gate.cutoff_at,
        "cutoff_admission_sequence": gate.cutoff_admission_sequence,
        "included_admission_sequence_min": gate.included_admission_sequence_min,
        "included_admission_sequence_max": gate.included_admission_sequence_max,
        "status": gate.status,
        "gate_epoch": gate.gate_epoch,
        "reason": gate.reason,
        "source_event_ids": gate.source_event_ids,
        "semantic_content_hash": gate.semantic_content_hash,
    }
    epoch_hash = snapshot.proposal_snapshot.producer_epoch_snapshot_hash
    epoch_payload = {
        "snapshot_fact_id": epoch_id,
        "arbitration_input_snapshot_id": input_id,
        "control_domain_id": epochs.control_domain_id,
        "cutoff_at": epochs.cutoff_at,
        "cutoff_admission_sequence": epochs.cutoff_admission_sequence,
        "included_admission_sequence_min": epochs.included_admission_sequence_min,
        "included_admission_sequence_max": epochs.included_admission_sequence_max,
        "active_epoch_by_producer": epochs.active_epoch_by_producer,
        "activation_id_by_producer": epochs.activation_id_by_producer,
        "semantic_content_hash": epoch_hash,
    }
    candidate_authorization = snapshot.candidate_authorization_snapshot
    candidate_authorization_payload = {
        "snapshot_id": candidate_authorization.snapshot_id,
        "arbitration_input_snapshot_id": input_id,
        "control_domain_id": candidate_authorization.control_domain_id,
        "cutoff_at": candidate_authorization.cutoff_at,
        "cutoff_admission_sequence": (
            candidate_authorization.cutoff_admission_sequence
        ),
        "target_runtime_contract_hash": (
            candidate_authorization.target_runtime_contract_hash
        ),
        "required_producer_kinds": candidate_authorization.required_producer_kinds,
        "authorization_by_proposal_id": {
            proposal_id: {
                "authorization_id": record.authorization_id,
                "authorization_hash": record.semantic_content_hash,
                "proposal_content_hash": record.proposal_content_hash,
                "evidence_id": record.evidence_id,
                "evidence_hash": record.evidence_hash,
                "binding_eligibility_id": record.binding_eligibility_id,
                "binding_eligibility_hash": record.binding_eligibility_hash,
                "authoritative_allowed": record.authoritative_allowed,
            }
            for proposal_id, record in sorted(
                candidate_authorization.authorization_by_proposal_id.items()
            )
        },
        "dispositions_by_proposal_id": (
            candidate_authorization.dispositions_by_proposal_id
        ),
        "semantic_content_hash": candidate_authorization.semantic_content_hash,
    }
    input_payload = {
        "arbitration_input_snapshot_id": input_id,
        "runtime_session_id": snapshot.runtime_session_id,
        "control_domain_id": snapshot.control_domain_id,
        "embodiment_instance_id": snapshot.embodiment_instance_id,
        "proposal_snapshot_id": snapshot.proposal_snapshot.proposal_snapshot_id,
        "proposal_snapshot_hash": snapshot.proposal_snapshot.semantic_content_hash,
        "authority_state_snapshot_fact_id": authority_id,
        "authority_state_snapshot_hash": authority.semantic_content_hash,
        "execution_gate_snapshot_fact_id": gate_id,
        "execution_gate_snapshot_hash": gate.semantic_content_hash,
        "producer_epoch_snapshot_fact_id": epoch_id,
        "producer_epoch_snapshot_hash": epoch_hash,
        "candidate_authorization_snapshot_fact_id": candidate_authorization_id,
        "candidate_authorization_snapshot_hash": (
            candidate_authorization.semantic_content_hash
        ),
        "authority_policy_hash": snapshot.authority_policy.semantic_content_hash,
        "previous_authoritative_decision_id": snapshot.previous_authoritative_decision_id,
        "previous_selection_by_group": snapshot.previous_selection_by_group,
        "active_selection_groups": snapshot.active_selection_groups,
        "component_catalog_hash": snapshot.component_catalog.semantic_content_hash,
        "decision_schedule_tick_id": snapshot.decision_schedule_tick.schedule_tick_id,
        "decision_schedule_tick_hash": snapshot.decision_schedule_tick.semantic_content_hash,
        "semantic_content_hash": snapshot.semantic_content_hash,
    }
    items = (
        (snapshot.proposal_snapshot.proposal_snapshot_id, "proposal_snapshot", _proposal_snapshot_payload(snapshot.proposal_snapshot)),
        (snapshot.decision_schedule_tick.schedule_tick_id, "decision_schedule_tick", _tick_payload(snapshot.decision_schedule_tick)),
        (authority_id, "authority_state_snapshot", authority_payload),
        (gate_id, "execution_gate_snapshot", gate_payload),
        (epoch_id, "producer_epoch_snapshot", epoch_payload),
        (
            candidate_authorization_id,
            "candidate_authorization_snapshot",
            candidate_authorization_payload,
        ),
        (input_id, "arbitration_input_snapshot", input_payload),
    )
    return tuple(
        context.fact(
            fact_id=fact_id,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
        )
        for fact_id, event_type, payload in items
    )


def project_arbitration_computation(
    context: AuthoritativeTraceContext,
    result: ArbitrationResult,
) -> ControlTraceFact:
    payload = {
        "result_id": result.result_id,
        "arbitration_input_snapshot_id": result.arbitration_input_snapshot_id,
        "arbitration_input_hash": result.arbitration_input_hash,
        "proposal_snapshot_id": result.proposal_snapshot_id,
        "arbiter_contract_id": result.arbiter_contract_id,
        "arbiter_contract_hash": result.arbiter_contract_hash,
        "arbiter_implementation_hash": result.arbiter_implementation_hash,
        "authority_policy_hash": result.authority_policy_hash,
        "component_selection_policy_hash": result.component_selection_policy_hash,
        "decision_schedule_spec_hash": result.decision_schedule_spec_hash,
        "canonical_encoding_version": result.canonical_encoding_version,
        "schedule_tick_id": result.schedule_tick_id,
        "decision_cutoff_at": result.decision_cutoff_at,
        "decided_at": result.decided_at,
        "target_effective_at": result.target_effective_at,
        "publication_deadline_at": result.publication_deadline_at,
        "authority_epoch": result.authority_epoch,
        "control_mode": result.control_mode,
        "execution_gate_status": result.execution_gate_status,
        "gate_epoch": result.gate_epoch,
        "selected_by_group": result.selected_by_group,
        "selected_by_component": result.selected_by_component,
        "owner_by_component": result.owner_by_component,
        "status_by_group": result.status_by_group,
        "composed_action": (
            None
            if result.composed_action is None
            else robot_action_semantic_payload(result.composed_action)
        ),
        "selected_action_hash": result.selected_action_hash,
        "output_status": result.output_status,
        "rejections": tuple(
            {
                "proposal_id": item.proposal_id,
                "selection_group_id": item.selection_group_id,
                "reason": item.reason,
            }
            for item in result.rejections
        ),
        "attribution": result.attribution,
        "semantic_content_hash": result.semantic_content_hash,
    }
    return context.fact(
        fact_id=result.result_id,
        event_type="arbitration_computation",
        occurred_at=result.decided_at,
        payload=payload,
    )


def project_publication(
    context: AuthoritativeTraceContext,
    attempt: ArbitrationPublicationAttempt,
    decision: AuthoritativeArbitrationDecision | None,
    *,
    selected_action_hash: str | None,
    computation_metrics: ArbitrationComputationMetrics,
) -> tuple[ControlTraceFact, ...]:
    attempt_payload = {
        "publication_attempt_id": attempt.publication_attempt_id,
        "result_id": attempt.result_id,
        "attempted_at": attempt.attempted_at,
        "accepted": attempt.accepted,
        "rejection_reason": attempt.rejection_reason,
        "required_fallback": attempt.required_fallback,
        "coalesced_before_publication": attempt.coalesced_before_publication,
        "computation_started_at": computation_metrics.started_at,
        "computation_completed_at": computation_metrics.completed_at,
        "computation_runtime_sec": computation_metrics.runtime_sec,
        "runtime_budget_sec": computation_metrics.runtime_budget_sec,
        "runtime_budget_missed": computation_metrics.runtime_budget_missed,
        "publication_deadline_missed_at_completion": (
            computation_metrics.publication_deadline_missed_at_completion
        ),
        "semantic_content_hash": attempt.semantic_content_hash,
    }
    facts = [
        context.fact(
            fact_id=attempt.publication_attempt_id,
            event_type="arbitration_publication_attempt",
            occurred_at=attempt.attempted_at,
            payload=attempt_payload,
        )
    ]
    if decision is None:
        rejection_payload = {
            "publication_attempt_id": attempt.publication_attempt_id,
            "result_id": attempt.result_id,
            "rejected_at": attempt.attempted_at,
            "rejection_reason": attempt.rejection_reason,
            "required_fallback": attempt.required_fallback,
        }
        rejection_hash = canonical_content_hash(
            rejection_payload,
            domain=PUBLICATION_REJECTION_HASH_DOMAIN_V1,
        )
        rejection_id = f"arbitration-rejection:{rejection_hash}"
        facts.append(
            context.fact(
                fact_id=rejection_id,
                event_type="arbitration_publication_rejected",
                occurred_at=attempt.attempted_at,
                payload={
                    "rejection_event_id": rejection_id,
                    **rejection_payload,
                    "semantic_content_hash": rejection_hash,
                },
            )
        )
        return tuple(facts)
    facts.append(
        context.fact(
            fact_id=decision.decision_id,
            event_type="arbitration_decision",
            occurred_at=decision.published_at,
            payload={
                "decision_id": decision.decision_id,
                "result_id": decision.result_id,
                "arbitration_input_snapshot_id": decision.arbitration_input_snapshot_id,
                "selected_action": robot_action_semantic_payload(decision.selected_action),
                "selected_action_hash": selected_action_hash,
                "selected_by_group": decision.selected_by_group,
                "published_at": decision.published_at,
                "authority_epoch": decision.authority_epoch,
                "gate_epoch": decision.gate_epoch,
                "semantic_content_hash": decision.semantic_content_hash,
            },
        )
    )
    return tuple(facts)


def project_execution_directive(
    context: AuthoritativeTraceContext,
    directive: ExecutionDirective,
    *,
    decision_id: str | None,
    publication_attempt_id: str,
    execution_epoch: int,
    occurred_at: float,
) -> ControlTraceFact:
    return context.fact(
        fact_id=directive.directive_id,
        event_type="execution_directive",
        occurred_at=occurred_at,
        payload={
            "directive_id": directive.directive_id,
            "decision_id": decision_id,
            "publication_attempt_id": publication_attempt_id,
            "arbitration_result_id": directive.arbitration_result_id,
            "global_status": directive.global_status,
            "directive_by_selection_group": {
                group: {
                    "selection_group_id": item.selection_group_id,
                    "directive": item.directive,
                    "selected_action_ref": item.selected_action_ref,
                    "existing_segment_id": item.existing_segment_id,
                    "segment_expiry": item.segment_expiry,
                    "group_segment_generation": item.group_segment_generation,
                }
                for group, item in directive.directive_by_selection_group.items()
            },
            "persistence_policy_hash": directive.persistence_policy_hash,
            "publication_fallback": directive.publication_fallback,
            "requires_execution_epoch_increment": directive.requires_execution_epoch_increment,
            "requires_backend_hold": directive.requires_backend_hold,
            "fault_latched": directive.fault_latched,
            "execution_epoch": int(execution_epoch),
            "semantic_content_hash": directive.semantic_content_hash,
        },
    )


def project_shared_execution_shadow_evaluation(
    context: AuthoritativeTraceContext,
    evaluation: SharedExecutionShadowEvaluation,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    """Project proposed execution without creating authoritative stage facts."""

    directive = evaluation.proposed_directive
    safety = evaluation.safety_result
    proposed_directive = {
        "directive_id": directive.directive_id,
        "arbitration_result_id": directive.arbitration_result_id,
        "global_status": directive.global_status,
        "directive_by_selection_group": {
            group: {
                "selection_group_id": item.selection_group_id,
                "directive": item.directive,
                "selected_action_ref": item.selected_action_ref,
                "existing_segment_id": item.existing_segment_id,
                "segment_expiry": item.segment_expiry,
                "group_segment_generation": item.group_segment_generation,
            }
            for group, item in directive.directive_by_selection_group.items()
        },
        "persistence_policy_hash": directive.persistence_policy_hash,
        "publication_fallback": directive.publication_fallback,
        "requires_execution_epoch_increment": (
            directive.requires_execution_epoch_increment
        ),
        "requires_backend_hold": directive.requires_backend_hold,
        "fault_latched": directive.fault_latched,
        "semantic_content_hash": directive.semantic_content_hash,
    }
    safety_payload = None
    if safety is not None:
        safety_payload = {
            "input_action_id": safety.input_action_id,
            "safe_action": (
                None
                if safety.safe_action is None
                else robot_action_semantic_payload(safety.safe_action)
            ),
            "modified_components": safety.modified_components,
            "reasons": safety.reasons,
            "status": safety.status,
            "safety_contract_id": safety.safety_contract_id,
            "safety_contract_hash": safety.safety_contract_hash,
            "anchor_id": safety.anchor_id,
            "anchor_kind": safety.anchor_kind,
            "anchor_hash": safety.anchor_hash,
            "details": safety.details,
        }
    return context.fact(
        fact_id=evaluation.evaluation_id,
        event_type="shared_execution_evaluation",
        occurred_at=occurred_at,
        payload={
            "evaluation_id": evaluation.evaluation_id,
            "source_arbitration_evaluation_id": (
                evaluation.source_arbitration_evaluation_id
            ),
            "arbitration_result_id": evaluation.arbitration_result_id,
            "output_status": evaluation.output_status,
            "non_authoritative": evaluation.non_authoritative,
            "proposed_directive": proposed_directive,
            "proposed_directive_hash": directive.semantic_content_hash,
            "safety_result": safety_payload,
            "safety_result_hash": evaluation.safety_result_hash,
            "command_intent": evaluation.command_intent,
            "command_intent_hash": evaluation.command_intent_hash,
            "semantic_content_hash": evaluation.semantic_content_hash,
        },
    )


def project_execution_cancellation(
    context: AuthoritativeTraceContext,
    cancellation: ExecutionCancellation,
) -> ControlTraceFact:
    return context.fact(
        fact_id=cancellation.cancellation_id,
        event_type="execution_cancellation",
        occurred_at=cancellation.effective_at,
        payload={
            "cancellation_id": cancellation.cancellation_id,
            "reason": cancellation.reason,
            "requested_scope": cancellation.requested_scope,
            "effective_scope": cancellation.effective_scope,
            "selection_group_ids": cancellation.selection_group_ids,
            "scope_escalation_reason": cancellation.scope_escalation_reason,
            "previous_execution_epoch": cancellation.previous_execution_epoch,
            "execution_epoch": cancellation.execution_epoch,
            "authority_epoch_evidence": cancellation.authority_epoch_evidence,
            "segment_generation_before": cancellation.segment_generation_before,
            "segment_generation_after": cancellation.segment_generation_after,
            "invalidated_decision_ids": cancellation.invalidated_decision_ids,
            "invalidated_reference_segment_ids": (
                cancellation.invalidated_reference_segment_ids
            ),
            "invalidated_queued_command_ids": (
                cancellation.invalidated_queued_command_ids
            ),
            "source_event_ids": cancellation.source_event_ids,
            "effective_at": cancellation.effective_at,
            "effective_clock_domain_id": cancellation.effective_clock_domain_id,
            "acknowledgement_deadline_at": (
                cancellation.acknowledgement_deadline_at
            ),
            "required_ack_levels": cancellation.required_ack_levels,
            "requests_backend_hold": cancellation.requests_backend_hold,
            "fault_latched": cancellation.fault_latched,
            "affects_decision_rate": cancellation.affects_decision_rate,
            "semantic_content_hash": cancellation.semantic_content_hash,
        },
    )


def project_execution_cancellation_ack(
    context: AuthoritativeTraceContext,
    acknowledgement: CancellationAcknowledgement,
) -> ControlTraceFact:
    acknowledgement_id = (
        "execution-cancellation-ack:"
        f"{acknowledgement.semantic_content_hash}"
    )
    return context.fact(
        fact_id=acknowledgement_id,
        event_type="execution_cancellation_ack",
        occurred_at=acknowledgement.acknowledged_at,
        payload={
            "acknowledgement_id": acknowledgement_id,
            "cancellation_id": acknowledgement.cancellation_id,
            "acknowledgement_level": acknowledgement.acknowledgement_level,
            "acknowledged_at": acknowledgement.acknowledged_at,
            "clock_domain_id": acknowledgement.clock_domain_id,
            "source_id": acknowledgement.source_id,
            "accepted": acknowledgement.accepted,
            "status": acknowledgement.status,
            "semantic_content_hash": acknowledgement.semantic_content_hash,
        },
    )


def project_shared_safety_result(
    context: AuthoritativeTraceContext,
    result: SharedCanonicalSafetyResult,
    *,
    decision_id: str,
    directive_id: str,
    input_action_hash: str,
    occurred_at: float,
) -> tuple[str, ControlTraceFact]:
    safe_payload = (
        None if result.safe_action is None else robot_action_semantic_payload(result.safe_action)
    )
    safe_hash = (
        None
        if safe_payload is None
        else canonical_content_hash(safe_payload, domain="dexteleop.robot_action_record.v1")
    )
    semantic_payload = {
        "decision_id": decision_id,
        "directive_id": directive_id,
        "input_action_id": result.input_action_id,
        "input_action_hash": input_action_hash,
        "safe_action": safe_payload,
        "safe_action_hash": safe_hash,
        "modified_components": result.modified_components,
        "reasons": result.reasons,
        "status": result.status,
        "safety_contract_id": result.safety_contract_id,
        "safety_contract_hash": result.safety_contract_hash,
        "anchor_id": result.anchor_id,
        "anchor_kind": result.anchor_kind,
        "anchor_hash": result.anchor_hash,
        "details": result.details,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=AUTHORITATIVE_SAFETY_HASH_DOMAIN_V1,
    )
    result_id = f"shared-safety-result:{semantic_hash}"
    return result_id, context.fact(
        fact_id=result_id,
        event_type="shared_safety_result",
        occurred_at=occurred_at,
        payload={
            "safety_result_id": result_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


def project_command(
    context: AuthoritativeTraceContext,
    command: TeleopCommand,
    *,
    decision_id: str,
    directive_id: str,
    safety_result_id: str,
    scheduled_for: float,
    created_at: float,
    authority_epoch: int,
    execution_epoch: int,
) -> tuple[str, str, ControlTraceFact]:
    command_payload = teleop_command_semantic_payload(command)
    command_hash = canonical_content_hash(
        command_payload,
        domain=AUTHORITATIVE_COMMAND_HASH_DOMAIN_V1,
    )
    identity_payload = {
        "decision_id": decision_id,
        "directive_id": directive_id,
        "safety_result_id": safety_result_id,
        "command_hash": command_hash,
        "authority_epoch": int(authority_epoch),
        "execution_epoch": int(execution_epoch),
    }
    command_identity_hash = canonical_content_hash(
        identity_payload,
        domain="dexteleop.command_identity.v1",
    )
    command_id = f"command:{command_identity_hash}"
    semantic_payload = {
        "command_id": command_id,
        **identity_payload,
        "command_schema_id": "dexteleop.teleop_command.v1",
        "commands": command_payload,
        "scheduled_for": float(scheduled_for),
        "created_at": float(created_at),
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain="dexteleop.command_record.v1",
    )
    return command_id, command_hash, context.fact(
        fact_id=command_id,
        event_type="command",
        occurred_at=created_at,
        payload={**semantic_payload, "semantic_content_hash": semantic_hash},
    )


def project_dispatch_owner_validation(
    context: AuthoritativeTraceContext,
    *,
    token: DispatchOwnerToken,
    permit: MotionDispatchPermit,
    validation: DispatchPermitValidationReceipt,
) -> ControlTraceFact:
    semantic_payload = {
        "event_kind": "motion_permit_validation",
        "permit_kind": permit.permit_kind,
        "operation": None,
        "evidence_event_id": None,
        "token_id": token.token_id,
        "owner_runtime_id": token.owner_runtime_id,
        "owner_kind": token.owner_kind,
        "owner_generation": token.owner_generation,
        "permit_id": permit.permit_id,
        "permit_hash": permit.semantic_content_hash,
        "validation_receipt_id": validation.receipt_id,
        "validation_accepted": validation.accepted,
        "validation_status": validation.status,
        "authority_epoch": validation.current_authority_epoch,
        "execution_epoch": validation.current_execution_epoch,
        "checked_at": validation.checked_at,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain="dexteleop.dispatch_owner_event.v1",
    )
    event_id = f"dispatch-owner-event:{semantic_hash}"
    return context.fact(
        fact_id=event_id,
        event_type="dispatch_owner",
        occurred_at=validation.checked_at,
        payload={
            "dispatch_event_id": event_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


def project_dispatch_permit_validation(
    context: AuthoritativeTraceContext,
    *,
    permit: DispatchPermit,
    validation: DispatchPermitValidationReceipt,
    token: DispatchOwnerToken | None = None,
) -> ControlTraceFact:
    if isinstance(permit, MotionDispatchPermit):
        if token is None:
            raise ValueError("Motion dispatch validation requires its owner token.")
        return project_dispatch_owner_validation(
            context,
            token=token,
            permit=permit,
            validation=validation,
        )
    semantic_payload = {
        "event_kind": f"{permit.permit_kind}_permit_validation",
        "permit_kind": permit.permit_kind,
        "operation": getattr(permit, "operation", None),
        "evidence_event_id": getattr(permit, "evidence_event_id", None),
        "token_id": None,
        "owner_runtime_id": None,
        "owner_kind": None,
        "owner_generation": None,
        "permit_id": permit.permit_id,
        "permit_hash": permit.semantic_content_hash,
        "validation_receipt_id": validation.receipt_id,
        "validation_accepted": validation.accepted,
        "validation_status": validation.status,
        "authority_epoch": validation.current_authority_epoch,
        "execution_epoch": validation.current_execution_epoch,
        "checked_at": validation.checked_at,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain="dexteleop.dispatch_owner_event.v1",
    )
    event_id = f"dispatch-owner-event:{semantic_hash}"
    return context.fact(
        fact_id=event_id,
        event_type="dispatch_owner",
        occurred_at=validation.checked_at,
        payload={
            "dispatch_event_id": event_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


def _dispatch_receipt_payload(receipt: CommandDispatchReceipt) -> dict[str, Any]:
    return {
        "effector_name": receipt.effector_name,
        "accepted": receipt.accepted,
        "operation": receipt.operation,
        "status": receipt.status,
        "timestamp": receipt.timestamp,
        "result_code": receipt.result_code,
        "result_text": receipt.result_text,
        "metadata": receipt.metadata,
    }


def project_dispatch_receipts(
    context: AuthoritativeTraceContext,
    receipts: Mapping[str, CommandDispatchReceipt],
    *,
    command_id: str,
    dispatch_event_id: str,
    backend_name: str,
    dispatched_at: float,
) -> tuple[str, ControlTraceFact]:
    receipt_payload = {
        name: _dispatch_receipt_payload(receipt)
        for name, receipt in sorted(receipts.items())
    }
    semantic_payload = {
        "command_id": command_id,
        "dispatch_event_id": dispatch_event_id,
        "backend_name": backend_name,
        "receipts_by_effector": receipt_payload,
        "all_accepted": bool(receipts) and all(item.accepted for item in receipts.values()),
        "dispatched_at": float(dispatched_at),
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=AUTHORITATIVE_DISPATCH_HASH_DOMAIN_V1,
    )
    receipt_id = f"dispatch-receipt:{semantic_hash}"
    return receipt_id, context.fact(
        fact_id=receipt_id,
        event_type="dispatch_receipt",
        occurred_at=dispatched_at,
        payload={
            "dispatch_receipt_id": receipt_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


def project_measurement_alignment(
    context: AuthoritativeTraceContext,
    *,
    decision_id: str,
    command_id: str,
    dispatch_receipt_id: str,
    measured_at: float,
    measurement: Mapping[str, Any],
    measurement_schema_id: str = "dexteleop.joint_measurement.v1",
) -> ControlTraceFact:
    semantic_payload = {
        "decision_id": decision_id,
        "command_id": command_id,
        "dispatch_receipt_id": dispatch_receipt_id,
        "measurement_schema_id": measurement_schema_id,
        "measured_at": float(measured_at),
        "measurement": dict(measurement),
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=MEASUREMENT_ALIGNMENT_HASH_DOMAIN_V1,
    )
    measurement_id = f"measurement-alignment:{semantic_hash}"
    return context.fact(
        fact_id=measurement_id,
        event_type="measurement_alignment",
        occurred_at=measured_at,
        payload={
            "measurement_alignment_id": measurement_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


__all__ = [
    "AuthoritativeTraceContext",
    "project_arbitration_computation",
    "project_arbitration_input",
    "project_authority_event",
    "project_candidate_authorization_admission",
    "project_command",
    "project_control_execution_manifest",
    "project_dispatch_owner_validation",
    "project_dispatch_permit_validation",
    "project_dispatch_receipts",
    "project_execution_cancellation",
    "project_execution_cancellation_ack",
    "project_execution_directive",
    "project_execution_gate_event",
    "project_measurement_alignment",
    "project_observation_cut",
    "project_producer_epoch_activation",
    "project_proposal_admission",
    "project_publication",
    "project_replay_execution_authorization",
    "project_shared_safety_result",
    "project_shared_execution_shadow_evaluation",
    "teleop_command_semantic_payload",
]

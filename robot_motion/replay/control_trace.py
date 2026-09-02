"""Backend-neutral ControlTrace projection for canonical replay evidence."""

from __future__ import annotations

from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.replay import (
    CandidateBindingEligibility,
    LegacyReplayContractDerivationReceipt,
    ReplayActionRecord,
    ReplayBindingAssessment,
    ReplaySourceActionContract,
    TargetRuntimeActionContract,
    replay_source_action_contract_payload,
    target_runtime_action_contract_payload,
)
from robot_motion.action.proposal import robot_action_semantic_payload

from .proposal_source import (
    ReplayProposalAdapterResult,
    ReplaySourceAdapterReceipt,
    ReplaySourceFrameEmission,
)


def project_replay_source_frame_emission(
    emission: ReplaySourceFrameEmission,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=emission.emission_id,
        event_type="replay_source_frame_emission",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=emission.observed_at,
        clock_domain_id=emission.clock_domain_id,
        payload={
            "emission_id": emission.emission_id,
            "source_path": emission.source_path,
            "source_episode_index": emission.source_episode_index,
            "group_key": emission.group_key,
            "local_frame_index": emission.local_frame_index,
            "source_action_index": emission.source_action_index,
            "source_relative_time": emission.source_relative_time,
            "action": robot_action_semantic_payload(emission.action),
            "observed_at": emission.observed_at,
            "target_effective_at": emission.target_effective_at,
            "effective_from": emission.effective_from,
            "effective_until": emission.effective_until,
            "clock_domain_id": emission.clock_domain_id,
            "semantic_content_hash": emission.semantic_content_hash,
        },
    )


def _fact(
    *,
    fact_id: str,
    event_type: str,
    payload: dict[str, object],
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=fact_id,
        event_type=event_type,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
        clock_domain_id=clock_domain_id,
        payload=payload,
    )


def project_legacy_replay_contract_derivation(
    receipt: LegacyReplayContractDerivationReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=receipt.derivation_receipt_id,
        event_type="legacy_replay_contract_derivation",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "derivation_receipt_id": receipt.derivation_receipt_id,
            "derivation_contract_id": receipt.derivation_contract_id,
            "derivation_contract_hash": receipt.derivation_contract_hash,
            "source_dataset_schema": receipt.source_dataset_schema,
            "consumed_metadata_fields": receipt.consumed_metadata_fields,
            "derived_action_stage": receipt.derived_action_stage,
            "derived_pose_frame_contract_hash": receipt.derived_pose_frame_contract_hash,
            "derived_component_schema_hash": receipt.derived_component_schema_hash,
            "unresolved_dimensions": receipt.unresolved_dimensions,
            "derivation_status": receipt.derivation_status,
            "semantic_content_hash": receipt.semantic_content_hash,
        },
    )


def project_replay_source_action_contract(
    contract: ReplaySourceActionContract,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=contract.contract_id,
        event_type="replay_source_action_contract",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            **replay_source_action_contract_payload(contract),
            "semantic_content_hash": contract.semantic_content_hash,
        },
    )


def project_target_runtime_action_contract(
    contract: TargetRuntimeActionContract,
    *,
    runtime_session_id: str,
    occurred_at: float,
    clock_domain_id: str,
    control_domain_id: str | None = None,
    embodiment_instance_id: str | None = None,
) -> ControlTraceFact:
    return _fact(
        fact_id=contract.target_contract_id,
        event_type="target_runtime_action_contract",
        runtime_session_id=runtime_session_id,
        control_domain_id=(
            contract.control_domain_id
            if control_domain_id is None
            else control_domain_id
        ),
        embodiment_instance_id=(
            contract.embodiment_instance_id
            if embodiment_instance_id is None
            else embodiment_instance_id
        ),
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            **target_runtime_action_contract_payload(contract),
            "semantic_content_hash": contract.semantic_content_hash,
        },
    )


def project_replay_binding_assessment(
    assessment: ReplayBindingAssessment,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=assessment.assessment_id,
        event_type="replay_binding_assessment",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "assessment_id": assessment.assessment_id,
            "binding_spec_id": assessment.binding_spec_id,
            "binding_spec_hash": assessment.binding_spec_hash,
            "source_action_contract_id": assessment.source_action_contract_id,
            "source_action_contract_hash": assessment.source_action_contract_hash,
            "target_runtime_contract_id": assessment.target_runtime_contract_id,
            "target_runtime_contract_hash": assessment.target_runtime_contract_hash,
            "verdict": assessment.verdict,
            "checks": tuple(
                {
                    "dimension": check.dimension,
                    "status": check.status,
                    "reason": check.reason,
                    "evidence_refs": check.evidence_refs,
                }
                for check in assessment.checks
            ),
            "permitted_uses": assessment.permitted_uses,
            "blocking_reasons": assessment.blocking_reasons,
            "verifier_contract_id": assessment.verifier_contract_id,
            "verifier_contract_hash": assessment.verifier_contract_hash,
            "semantic_content_hash": assessment.semantic_content_hash,
        },
    )


def project_candidate_binding_eligibility(
    eligibility: CandidateBindingEligibility,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=eligibility.eligibility_id,
        event_type="candidate_binding_eligibility",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "eligibility_id": eligibility.eligibility_id,
            "assessment_id": eligibility.assessment_id,
            "assessment_hash": eligibility.assessment_hash,
            "source_emission_id": eligibility.source_emission_id,
            "requested_use": eligibility.requested_use,
            "evaluation_allowed": eligibility.evaluation_allowed,
            "authoritative_allowed": eligibility.authoritative_allowed,
            "conditional_evaluation": eligibility.conditional_evaluation,
            "same_legacy_emission": eligibility.same_legacy_emission,
            "reasons": eligibility.reasons,
            "target_runtime_contract_hash": eligibility.target_runtime_contract_hash,
            "semantic_content_hash": eligibility.semantic_content_hash,
        },
    )


def project_replay_action_record(
    record: ReplayActionRecord,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=record.replay_action_record_id,
        event_type="replay_action_record",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "replay_action_record_id": record.replay_action_record_id,
            "source_contract_id": record.source_contract_id,
            "source_contract_hash": record.source_contract_hash,
            "episode_id": record.episode_id,
            "source_event_id": record.source_event_id,
            "source_action_index": record.source_action_index,
            "source_action_stage": record.source_action_stage,
            "source_timestamp": record.source_timestamp,
            "source_timestamp_clock_domain_id": record.source_timestamp_clock_domain_id,
            "canonical_action_payload": record.canonical_action_payload,
            "canonical_action_hash": record.canonical_action_hash,
            "schedule_id": record.schedule_id,
            "schedule_hash": record.schedule_hash,
            "old_safety_profile_hash": record.old_safety_profile_hash,
            "semantic_content_hash": record.semantic_content_hash,
        },
    )


def project_replay_source_adapter_receipt(
    receipt: ReplaySourceAdapterReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _fact(
        fact_id=receipt.representation_trace_id,
        event_type="replay_source_adapter_receipt",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        payload={
            "representation_trace_id": receipt.representation_trace_id,
            "adapter_contract_id": receipt.adapter_contract_id,
            "source_contract_id": receipt.source_contract_id,
            "source_contract_hash": receipt.source_contract_hash,
            "replay_action_record_id": receipt.replay_action_record_id,
            "replay_action_record_hash": receipt.replay_action_record_hash,
            "source_event_id": receipt.source_event_id,
            "source_emission_id": receipt.source_emission_id,
            "source_action_index": receipt.source_action_index,
            "source_action_stage": receipt.source_action_stage,
            "target_binding_spec_id": receipt.target_binding_spec_id,
            "target_binding_spec_hash": receipt.target_binding_spec_hash,
            "binding_assessment_id": receipt.binding_assessment_id,
            "binding_assessment_hash": receipt.binding_assessment_hash,
            "binding_verdict": receipt.binding_verdict,
            "binding_eligibility_id": receipt.binding_eligibility_id,
            "binding_eligibility_hash": receipt.binding_eligibility_hash,
            "accepted": receipt.accepted,
            "evaluation_allowed": receipt.evaluation_allowed,
            "authoritative_eligible": receipt.authoritative_eligible,
            "conditional_evaluation": receipt.conditional_evaluation,
            "old_safe_does_not_bypass_current_safety": (
                receipt.old_safe_does_not_bypass_current_safety
            ),
            "rejection_reasons": receipt.rejection_reasons,
            "output_action_semantic_hash": receipt.output_action_semantic_hash,
            "semantic_content_hash": receipt.semantic_content_hash,
        },
    )


def project_replay_binding_evidence(
    result: ReplayProposalAdapterResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
    target_control_domain_id: str | None = None,
    target_embodiment_instance_id: str | None = None,
) -> tuple[ControlTraceFact, ...]:
    common = {
        "runtime_session_id": runtime_session_id,
        "control_domain_id": control_domain_id,
        "embodiment_instance_id": embodiment_instance_id,
        "occurred_at": occurred_at,
        "clock_domain_id": clock_domain_id,
    }
    facts: list[ControlTraceFact] = []
    if result.derivation_receipt is not None:
        facts.append(
            project_legacy_replay_contract_derivation(
                result.derivation_receipt,
                **common,
            )
        )
    facts.extend(
        (
            project_replay_source_action_contract(result.source_action_contract, **common),
            project_target_runtime_action_contract(
                result.target_runtime_contract,
                runtime_session_id=runtime_session_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
                control_domain_id=target_control_domain_id,
                embodiment_instance_id=target_embodiment_instance_id,
            ),
            project_replay_binding_assessment(result.binding_assessment, **common),
        )
    )
    return tuple(facts)


def project_replay_proposal_adapter_result(
    result: ReplayProposalAdapterResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ...]:
    common = {
        "runtime_session_id": runtime_session_id,
        "control_domain_id": control_domain_id,
        "embodiment_instance_id": embodiment_instance_id,
        "occurred_at": occurred_at,
        "clock_domain_id": clock_domain_id,
    }
    facts = [
        project_candidate_binding_eligibility(
            result.binding_eligibility,
            **common,
        )
    ]
    if result.replay_action_record is not None:
        facts.append(project_replay_action_record(result.replay_action_record, **common))
    facts.append(
        project_replay_source_adapter_receipt(
            result.source_adapter_receipt,
            **common,
        )
    )
    return tuple(facts)


def project_replay_adapter_result(
    result: ReplayProposalAdapterResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ...]:
    """Project all immutable source/binding/eligibility facts for one emission."""

    common = {
        "runtime_session_id": runtime_session_id,
        "control_domain_id": control_domain_id,
        "embodiment_instance_id": embodiment_instance_id,
        "occurred_at": occurred_at,
        "clock_domain_id": clock_domain_id,
    }
    return (
        *project_replay_binding_evidence(
            result,
            target_control_domain_id=control_domain_id,
            target_embodiment_instance_id=embodiment_instance_id,
            **common,
        ),
        *project_replay_proposal_adapter_result(result, **common),
    )


__all__ = [
    "project_candidate_binding_eligibility",
    "project_legacy_replay_contract_derivation",
    "project_replay_adapter_result",
    "project_replay_action_record",
    "project_replay_binding_assessment",
    "project_replay_binding_evidence",
    "project_replay_proposal_adapter_result",
    "project_replay_source_action_contract",
    "project_replay_source_adapter_receipt",
    "project_replay_source_frame_emission",
    "project_target_runtime_action_contract",
]

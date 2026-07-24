"""No-I/O projection of M3 shadow facts into v2.4 ControlTrace facts."""

from __future__ import annotations

from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
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
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionResult,
    CompositionSpec,
)
from control_composition.trace import (
    project_composition_input as _project_composition_input,
    project_composition_input_admission as _project_composition_input_admission,
    project_composition_input_snapshot as _project_composition_input_snapshot,
    project_composition_result as _project_composition_result,
    project_composition_spec as _project_composition_spec,
)
from robot_motion.action.arbiter import ArbitrationResult
from robot_motion.action.arbitration_input import ArbitrationInputSnapshot
from robot_motion.action.authority import AuthorityEvent, AuthorityStateSnapshot
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.action.proposal import (
    ActionProposal,
    ProposalAdmissionReceipt,
    action_proposal_semantic_payload,
    robot_action_semantic_payload,
)
from robot_motion.action.snapshot import ProposalSnapshot
from robot_motion.deploy.control_trace import project_policy_action_emission
from robot_motion.runtime.execution_gate import ExecutionGateEvent, ExecutionGateSnapshot
from robot_motion.runtime.producer_epoch import ProducerEpochActivation, ProducerEpochSnapshot
from robot_motion.replay.proposal_source import (
    ReplayProposalAdapterResult,
    ReplaySourceAdapterReceipt,
    ReplaySourceFrameEmission,
)
from teleop_runtime.deploy.arbiter_shadow import (
    ArbiterShadowStep,
    LegacyExecutionObservationEvent,
    PassivePolicyArbiterShadow,
)
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanEffectorCommandSnapshot,
    HumanProposalAdapterResult,
    SourceAdapterReceipt,
)


TRACE_SOURCE_ID_HASH_DOMAIN_V1 = "dexteleop.control_trace_source_id.v1"


def project_replay_action_record(
    record: ReplayActionRecord,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=record.replay_action_record_id,
        event_type="replay_action_record",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
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
            "source_timestamp_clock_domain_id": (
                record.source_timestamp_clock_domain_id
            ),
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
    return ControlTraceFact.create(
        fact_id=receipt.representation_trace_id,
        event_type="replay_source_adapter_receipt",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
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


def project_legacy_replay_contract_derivation(
    receipt: LegacyReplayContractDerivationReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return ControlTraceFact.create(
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
    return ControlTraceFact.create(
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
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=contract.target_contract_id,
        event_type="target_runtime_action_contract",
        runtime_session_id=runtime_session_id,
        control_domain_id=contract.control_domain_id,
        embodiment_instance_id=contract.embodiment_instance_id,
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
    return ControlTraceFact.create(
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
    return ControlTraceFact.create(
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


def project_replay_binding_evidence(
    result: ReplayProposalAdapterResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ...]:
    facts: list[ControlTraceFact] = []
    if result.derivation_receipt is not None:
        facts.append(
            project_legacy_replay_contract_derivation(
                result.derivation_receipt,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            )
        )
    facts.extend(
        (
            project_replay_source_action_contract(
                result.source_action_contract,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            ),
            project_target_runtime_action_contract(
                result.target_runtime_contract,
                runtime_session_id=runtime_session_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            ),
            project_replay_binding_assessment(
                result.binding_assessment,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            ),
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
    facts: list[ControlTraceFact] = [
        project_candidate_binding_eligibility(
            result.binding_eligibility,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            occurred_at=occurred_at,
            clock_domain_id=clock_domain_id,
        )
    ]
    if result.replay_action_record is not None:
        facts.append(
            project_replay_action_record(
                result.replay_action_record,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                embodiment_instance_id=embodiment_instance_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            )
        )
    facts.append(
        project_replay_source_adapter_receipt(
            result.source_adapter_receipt,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            occurred_at=occurred_at,
            clock_domain_id=clock_domain_id,
        )
    )
    return tuple(facts)


def project_composition_input_admission(
    receipt: CompositionInputAdmissionReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_input_admission(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        receipt,
        occurred_at=occurred_at,
    )


def project_composition_spec(
    spec: CompositionSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    """Compatibility projection; canonical implementation lives in the package."""

    return _project_composition_spec(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        spec,
        occurred_at=occurred_at,
    )


def project_composition_input(
    record: CompositionInputRecord,
    *,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    return _project_composition_input(
        AuthoritativeTraceContext(
            runtime_session_id=record.runtime_session_id,
            control_domain_id=record.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=record.available_clock_domain_id,
        ),
        record,
    )


def project_composition_input_snapshot(
    snapshot: CompositionInputSnapshot,
    *,
    runtime_session_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_input_snapshot(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=snapshot.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        snapshot,
    )


def project_composition_result(
    result: CompositionResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_result(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        result,
        occurred_at=occurred_at,
    )


def _human_component_mapping_payload(mapping: Any) -> dict[str, Any]:
    return {
        "component_id": mapping.component_id,
        "selection_group_id": mapping.selection_group_id,
        "effector_name": mapping.effector_name,
        "contribution_index": mapping.contribution_index,
        "retargeter_name": mapping.retargeter_name,
        "input_command_type": mapping.input_command_type,
        "input_frame": mapping.input_frame,
        "output_representation": mapping.output_representation,
        "output_reference_frame": mapping.output_reference_frame,
        "output_controlled_frame": mapping.output_controlled_frame,
        "lossiness": mapping.lossiness,
        "reasons": mapping.reasons,
        "overwritten_contribution_indices": mapping.overwritten_contribution_indices,
    }


def _human_effector_command_payload(
    command: HumanEffectorCommandSnapshot,
) -> dict[str, Any]:
    return {
        "command_name": command.command_name,
        "effector_name": command.effector_name,
        "side": command.side,
        "command_type": command.command_type,
        "value": command.value,
        "frame_name": command.frame_name,
        "metadata": command.metadata,
        "unfreezable_metadata_keys": command.unfreezable_metadata_keys,
    }


def project_human_source_adapter_receipt(
    receipt: SourceAdapterReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ControlTraceFact]:
    """Project the typed human mapping receipt and its validation separately."""

    receipt_fact = ControlTraceFact.create(
        fact_id=receipt.representation_trace_id,
        event_type="source_adapter_receipt",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
        clock_domain_id=clock_domain_id,
        payload={
            "representation_trace_id": receipt.representation_trace_id,
            "adapter_contract_id": receipt.adapter_contract_id,
            "adapter_contract_hash": receipt.adapter_contract_hash,
            "source_emission_id": receipt.source_emission_id,
            "source_emission_sequence": receipt.source_emission_sequence,
            "source_emission_semantic_hash": receipt.source_emission_semantic_hash,
            "component_mappings": tuple(
                _human_component_mapping_payload(mapping)
                for mapping in receipt.component_mappings
            ),
            "lossiness": receipt.lossiness,
            "source_validation_accepted": receipt.source_validation_accepted,
            "authoritative_eligible": receipt.authoritative_eligible,
            "source_validation_reasons": receipt.source_validation_reasons,
            "output_action_space": receipt.output_action_space,
            "output_action_semantic_hash": receipt.output_action_semantic_hash,
            "semantic_content_hash": receipt.semantic_content_hash,
        },
    )
    validation_id = f"source-validation:{receipt.source_emission_id}"
    validation_fact = ControlTraceFact.create(
        fact_id=validation_id,
        event_type="source_validation",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
        clock_domain_id=clock_domain_id,
        payload={
            "validation_id": validation_id,
            "source_event_id": receipt.source_emission_id,
            "accepted": receipt.source_validation_accepted,
            "reasons": receipt.source_validation_reasons,
        },
    )
    return receipt_fact, validation_fact


def project_human_proposal_adapter_result(
    emission: HumanCommandEmission,
    result: HumanProposalAdapterResult,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ControlTraceFact, ControlTraceFact]:
    """Project the immutable source emission, mapping receipt, and validation."""

    receipt = result.source_adapter_receipt
    if receipt.source_emission_sequence != emission.emission_sequence:
        raise ValueError("Human source receipt and emission sequence disagree.")
    if receipt.source_emission_semantic_hash != emission.semantic_content_hash:
        raise ValueError("Human source receipt and emission content hash disagree.")
    emission_fact = ControlTraceFact.create(
        fact_id=receipt.source_emission_id,
        event_type="human_command_emission",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=float(occurred_at),
        clock_domain_id=clock_domain_id,
        payload={
            "source_emission_id": receipt.source_emission_id,
            "source_emission_sequence": emission.emission_sequence,
            "source_emission_semantic_hash": emission.semantic_content_hash,
            "emitted_at": emission.emitted_at,
            "command_timestamp": emission.command_timestamp,
            "contributions": tuple(
                {
                    "contribution_index": contribution.contribution_index,
                    "retargeter_name": contribution.retargeter_name,
                    "command_timestamp": contribution.command_timestamp,
                    "commands": tuple(
                        _human_effector_command_payload(command)
                        for command in contribution.commands
                    ),
                    "command_metadata": contribution.command_metadata,
                    "unfreezable_metadata_keys": (
                        contribution.unfreezable_metadata_keys
                    ),
                }
                for contribution in emission.contributions
            ),
            "merged_commands": tuple(
                _human_effector_command_payload(command)
                for command in emission.merged_commands
            ),
            "winning_contribution_by_effector": (
                emission.winning_contribution_by_effector
            ),
            "overwritten_contributions_by_effector": (
                emission.overwritten_contributions_by_effector
            ),
            "merged_command_metadata": emission.merged_command_metadata,
            "unfreezable_merged_metadata_keys": (
                emission.unfreezable_merged_metadata_keys
            ),
            "observation_sample_refs": {
                stream_id: {
                    "stream_id": sample.stream_id,
                    "sample_id": sample.sample_id,
                    "sequence": int(sample.sequence),
                    "measured_at": float(sample.measured_at),
                    "measured_clock_domain_id": (
                        sample.measured_clock_domain_id
                    ),
                    "available_at": float(sample.available_at),
                    "available_clock_domain_id": (
                        sample.available_clock_domain_id
                    ),
                    "payload_object_id": sample.payload_object_id,
                    "valid": bool(sample.valid),
                    "stale": bool(sample.stale),
                    "control_admission_sequence": (
                        sample.control_admission_sequence
                    ),
                    "content_metadata": sample.content_metadata,
                }
                for stream_id, sample in sorted(
                    emission.observation_sample_refs.items()
                )
            },
            "observation_sample_set_hash": (
                emission.observation_sample_set_hash
            ),
        },
    )
    receipt_fact, validation_fact = project_human_source_adapter_receipt(
        receipt,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
    )
    return emission_fact, receipt_fact, validation_fact


def _fact(
    coordinator: PassivePolicyArbiterShadow,
    *,
    event_type: str,
    payload: Mapping[str, Any],
    occurred_at: float,
    fact_id: str,
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=fact_id,
        event_type=event_type,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
        occurred_at=float(occurred_at),
        clock_domain_id=coordinator.config.clock_domain_id,
        payload=payload,
    )


def _producer_epoch_activation_payload(
    activation: ProducerEpochActivation,
) -> dict[str, Any]:
    return {
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


def _proposal_admission_payload(
    receipt: ProposalAdmissionReceipt,
    *,
    emission_id: str,
) -> dict[str, Any]:
    return {
        "emission_id": emission_id,
        "accepted": receipt.accepted,
        "rejection_code": receipt.rejection_code,
        "message": receipt.message,
        "proposal_id": receipt.proposal_id,
        "available_at": receipt.available_at,
        "available_clock_domain_id": receipt.available_clock_domain_id,
        "control_admission_sequence": receipt.control_admission_sequence,
        "proposal_semantic_content_hash": receipt.semantic_content_hash,
    }


def _proposal_snapshot_payload(snapshot: ProposalSnapshot) -> dict[str, Any]:
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
                "proposal_id": disposition.proposal_id,
                "selection_group_id": disposition.selection_group_id,
                "disposition": disposition.disposition,
                "detail": disposition.detail,
            }
            for disposition in snapshot.dispositions
        ),
        "semantic_content_hash": snapshot.semantic_content_hash,
    }


def _authority_snapshot_payload(
    snapshot: AuthorityStateSnapshot,
    *,
    arbitration_input_snapshot_id: str,
) -> dict[str, Any]:
    fact_id = f"authority-state-snapshot:{arbitration_input_snapshot_id}"
    return {
        "snapshot_fact_id": fact_id,
        "arbitration_input_snapshot_id": arbitration_input_snapshot_id,
        "control_domain_id": snapshot.control_domain_id,
        "cutoff_at": snapshot.cutoff_at,
        "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
        "included_admission_sequence_min": snapshot.included_admission_sequence_min,
        "included_admission_sequence_max": snapshot.included_admission_sequence_max,
        "authority_policy_hash": snapshot.authority_policy_hash,
        "control_mode": snapshot.control_mode,
        "authority_epoch": snapshot.authority_epoch,
        "active_leases": {
            group: {
                "selection_group_id": lease.selection_group_id,
                "owner_producer_id": lease.owner_producer_id,
                "acquired_event_id": lease.acquired_event_id,
                "acquired_at": lease.acquired_at,
                "expires_at": lease.expires_at,
            }
            for group, lease in snapshot.active_leases.items()
        },
        "causal_event_ids": snapshot.causal_event_ids,
        "semantic_content_hash": snapshot.semantic_content_hash,
    }


def _gate_snapshot_payload(
    snapshot: ExecutionGateSnapshot,
    *,
    arbitration_input_snapshot_id: str,
) -> dict[str, Any]:
    fact_id = f"execution-gate-snapshot:{arbitration_input_snapshot_id}"
    return {
        "snapshot_fact_id": fact_id,
        "arbitration_input_snapshot_id": arbitration_input_snapshot_id,
        "control_domain_id": snapshot.control_domain_id,
        "cutoff_at": snapshot.cutoff_at,
        "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
        "included_admission_sequence_min": snapshot.included_admission_sequence_min,
        "included_admission_sequence_max": snapshot.included_admission_sequence_max,
        "status": snapshot.status,
        "gate_epoch": snapshot.gate_epoch,
        "reason": snapshot.reason,
        "source_event_ids": snapshot.source_event_ids,
        "semantic_content_hash": snapshot.semantic_content_hash,
    }


def _producer_epoch_snapshot_payload(
    snapshot: ProducerEpochSnapshot,
    *,
    arbitration_input_snapshot_id: str,
    semantic_content_hash: str,
) -> dict[str, Any]:
    fact_id = f"producer-epoch-snapshot:{arbitration_input_snapshot_id}"
    return {
        "snapshot_fact_id": fact_id,
        "arbitration_input_snapshot_id": arbitration_input_snapshot_id,
        "control_domain_id": snapshot.control_domain_id,
        "cutoff_at": snapshot.cutoff_at,
        "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
        "included_admission_sequence_min": snapshot.included_admission_sequence_min,
        "included_admission_sequence_max": snapshot.included_admission_sequence_max,
        "active_epoch_by_producer": snapshot.active_epoch_by_producer,
        "activation_id_by_producer": snapshot.activation_id_by_producer,
        "semantic_content_hash": semantic_content_hash,
    }


def _arbitration_input_payload(snapshot: ArbitrationInputSnapshot) -> dict[str, Any]:
    snapshot_id = snapshot.arbitration_input_snapshot_id
    return {
        "arbitration_input_snapshot_id": snapshot_id,
        "runtime_session_id": snapshot.runtime_session_id,
        "control_domain_id": snapshot.control_domain_id,
        "embodiment_instance_id": snapshot.embodiment_instance_id,
        "proposal_snapshot_id": snapshot.proposal_snapshot.proposal_snapshot_id,
        "proposal_snapshot_hash": snapshot.proposal_snapshot.semantic_content_hash,
        "authority_state_snapshot_fact_id": f"authority-state-snapshot:{snapshot_id}",
        "authority_state_snapshot_hash": snapshot.authority_state_snapshot.semantic_content_hash,
        "execution_gate_snapshot_fact_id": f"execution-gate-snapshot:{snapshot_id}",
        "execution_gate_snapshot_hash": snapshot.execution_gate_snapshot.semantic_content_hash,
        "producer_epoch_snapshot_fact_id": f"producer-epoch-snapshot:{snapshot_id}",
        "producer_epoch_snapshot_hash": snapshot.proposal_snapshot.producer_epoch_snapshot_hash,
        "candidate_authorization_snapshot_fact_id": (
            f"candidate-authorization-snapshot:{snapshot_id}"
        ),
        "candidate_authorization_snapshot_hash": (
            snapshot.candidate_authorization_snapshot.semantic_content_hash
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


def _arbitration_computation_payload(result: ArbitrationResult) -> dict[str, Any]:
    return {
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
                "proposal_id": rejection.proposal_id,
                "selection_group_id": rejection.selection_group_id,
                "reason": rejection.reason,
            }
            for rejection in result.rejections
        ),
        "attribution": result.attribution,
        "semantic_content_hash": result.semantic_content_hash,
    }


def _authority_event_payload(event: AuthorityEvent) -> dict[str, Any]:
    return {
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
    }


def _execution_gate_event_payload(event: ExecutionGateEvent) -> dict[str, Any]:
    return {
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
    }


def project_shadow_startup(
    coordinator: PassivePolicyArbiterShadow,
) -> tuple[ControlTraceFact, ...]:
    activation = coordinator.producer_epoch_activation
    return (
        _fact(
            coordinator,
            event_type="producer_epoch_activation",
            payload=_producer_epoch_activation_payload(activation),
            occurred_at=activation.available_at,
            fact_id=activation.activation_id,
        ),
    )


def project_authority_event(
    coordinator: PassivePolicyArbiterShadow,
    event: AuthorityEvent,
) -> ControlTraceFact:
    return _fact(
        coordinator,
        event_type="authority_event",
        payload=_authority_event_payload(event),
        occurred_at=event.available_at,
        fact_id=event.event_id,
    )


def project_execution_gate_event(
    coordinator: PassivePolicyArbiterShadow,
    event: ExecutionGateEvent,
) -> ControlTraceFact:
    return _fact(
        coordinator,
        event_type="execution_gate_event",
        payload=_execution_gate_event_payload(event),
        occurred_at=event.available_at,
        fact_id=event.event_id,
    )


def project_shadow_step(
    coordinator: PassivePolicyArbiterShadow,
    step: ArbiterShadowStep,
) -> tuple[ControlTraceFact, ...]:
    """Project one passive step without reading or mutating policy/control state."""

    facts: list[ControlTraceFact] = []
    emission = step.emission
    if emission is not None:
        facts.extend(
            project_policy_action_emission(
                emission,
                runtime_session_id=coordinator.config.runtime_session_id,
                control_domain_id=coordinator.config.control_domain_id,
                embodiment_instance_id=coordinator.config.embodiment_instance_id,
                clock_domain_id=coordinator.config.clock_domain_id,
            )
        )
    receipt = step.admission_receipt
    if receipt is not None:
        receipt_time = (
            float(receipt.available_at)
            if receipt.available_at is not None
            else (
                float(step.status_event.occurred_at)
                if step.status_event is not None
                else coordinator.now()
            )
        )
        if step.emission_id is None:
            raise ValueError("Proposal admission projection requires emission_id.")
        receipt_fact_id = f"proposal-admission:{step.emission_id}"
        facts.append(
            _fact(
                coordinator,
                event_type="proposal_admission",
                payload=_proposal_admission_payload(
                    receipt,
                    emission_id=step.emission_id,
                ),
                occurred_at=receipt_time,
                fact_id=receipt_fact_id,
            )
        )
        if receipt.proposal is not None:
            facts.append(
                _fact(
                    coordinator,
                    event_type="action_proposal",
                    payload={
                        **action_proposal_semantic_payload(receipt.proposal),
                        "composition_input_ids": receipt.proposal.composition_input_ids,
                        "composition_input_hashes": receipt.proposal.composition_input_hashes,
                        "composition_spec_hash": receipt.proposal.composition_spec_hash,
                        "proposal_semantic_hash_domain": (
                            receipt.proposal.semantic_hash_domain
                        ),
                        "proposal_semantic_content_hash": (
                            receipt.proposal.semantic_content_hash
                        ),
                    },
                    occurred_at=receipt.proposal.available_at,
                    fact_id=receipt.proposal.proposal_id,
                )
            )

    snapshot = step.arbitration_input_snapshot
    if snapshot is not None:
        tick = snapshot.decision_schedule_tick
        occurred_at = tick.scheduled_decision_at
        input_id = snapshot.arbitration_input_snapshot_id
        authority_payload = _authority_snapshot_payload(
            snapshot.authority_state_snapshot,
            arbitration_input_snapshot_id=input_id,
        )
        gate_payload = _gate_snapshot_payload(
            snapshot.execution_gate_snapshot,
            arbitration_input_snapshot_id=input_id,
        )
        producer_epoch_payload = _producer_epoch_snapshot_payload(
            snapshot.producer_epoch_snapshot,
            arbitration_input_snapshot_id=input_id,
            semantic_content_hash=snapshot.proposal_snapshot.producer_epoch_snapshot_hash,
        )
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
            "required_producer_kinds": (
                candidate_authorization.required_producer_kinds
            ),
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
        facts.extend(
            (
                _fact(
                    coordinator,
                    event_type="proposal_snapshot",
                    payload=_proposal_snapshot_payload(snapshot.proposal_snapshot),
                    occurred_at=occurred_at,
                    fact_id=snapshot.proposal_snapshot.proposal_snapshot_id,
                ),
                _fact(
                    coordinator,
                    event_type="decision_schedule_tick",
                    payload={
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
                    },
                    occurred_at=occurred_at,
                    fact_id=tick.schedule_tick_id,
                ),
                _fact(
                    coordinator,
                    event_type="authority_state_snapshot",
                    payload=authority_payload,
                    occurred_at=occurred_at,
                    fact_id=authority_payload["snapshot_fact_id"],
                ),
                _fact(
                    coordinator,
                    event_type="execution_gate_snapshot",
                    payload=gate_payload,
                    occurred_at=occurred_at,
                    fact_id=gate_payload["snapshot_fact_id"],
                ),
                _fact(
                    coordinator,
                    event_type="producer_epoch_snapshot",
                    payload=producer_epoch_payload,
                    occurred_at=occurred_at,
                    fact_id=producer_epoch_payload["snapshot_fact_id"],
                ),
                _fact(
                    coordinator,
                    event_type="candidate_authorization_snapshot",
                    payload=candidate_authorization_payload,
                    occurred_at=occurred_at,
                    fact_id=f"candidate-authorization-snapshot:{input_id}",
                ),
                _fact(
                    coordinator,
                    event_type="arbitration_input_snapshot",
                    payload=_arbitration_input_payload(snapshot),
                    occurred_at=occurred_at,
                    fact_id=input_id,
                ),
            )
        )

    if step.computation is not None:
        occurred_at = (
            step.evaluation.evaluated_at
            if step.evaluation is not None
            else coordinator.now()
        )
        facts.append(
            _fact(
                coordinator,
                event_type="arbitration_computation",
                payload=_arbitration_computation_payload(step.computation),
                occurred_at=occurred_at,
                fact_id=step.computation.result_id,
            )
        )
    if step.evaluation is not None:
        evaluation = step.evaluation
        facts.append(
            _fact(
                coordinator,
                event_type="arbitration_evaluation",
                payload={
                    "evaluation_id": evaluation.evaluation_id,
                    "mode": evaluation.mode,
                    "non_authoritative": evaluation.non_authoritative,
                    "arbitration_input_snapshot_id": evaluation.arbitration_input_snapshot_id,
                    "arbitration_input_hash": evaluation.arbitration_input_hash,
                    "proposal_snapshot_id": evaluation.proposal_snapshot_id,
                    "computation_result_id": evaluation.computation_result_id,
                    "computation_result_hash": evaluation.computation_result_hash,
                    "output_status": evaluation.output_status,
                    "would_select_action": (
                        None
                        if evaluation.would_select_action is None
                        else robot_action_semantic_payload(
                            evaluation.would_select_action
                        )
                    ),
                    "would_select_action_hash": evaluation.would_select_action_hash,
                    "would_select_by_group": evaluation.would_select_by_group,
                    "would_select_by_component": evaluation.would_select_by_component,
                    "evaluated_at": evaluation.evaluated_at,
                    "semantic_content_hash": evaluation.semantic_content_hash,
                },
                occurred_at=evaluation.evaluated_at,
                fact_id=evaluation.evaluation_id,
            )
        )
    if step.status_event is not None:
        status = step.status_event
        facts.append(
            _fact(
                coordinator,
                event_type="arbiter_shadow_status",
                payload={
                    "status_event_id": status.status_event_id,
                    "status": status.status,
                    "reason": status.reason,
                    "emission_id": status.emission_id,
                    "occurred_at": status.occurred_at,
                    "non_authoritative": status.non_authoritative,
                    "semantic_content_hash": status.semantic_content_hash,
                },
                occurred_at=status.occurred_at,
                fact_id=status.status_event_id,
            )
        )
    return tuple(facts)


def project_legacy_execution(
    coordinator: PassivePolicyArbiterShadow,
    event: LegacyExecutionObservationEvent,
) -> ControlTraceFact:
    parity = event.component_parity
    return _fact(
        coordinator,
        event_type="legacy_execution_observation",
        payload={
            "event_id": event.event_id,
            "owner_kind": event.owner_kind,
            "non_authoritative_shadow_reference": event.non_authoritative_shadow_reference,
            "emission_id": event.emission_id,
            "evaluation_id": event.evaluation_id,
            "legacy_safe_representation_trace_id": event.legacy_safe_representation_trace_id,
            "command_present": event.command_present,
            "execution_accepted": event.execution_accepted,
            "execution_mode": event.execution_mode,
            "dispatch_attempted": event.dispatch_attempted,
            "dispatch_completed": event.dispatch_completed,
            "dispatch_error_type": event.dispatch_error_type,
            "component_parity": (
                None
                if parity is None
                else {
                    "matched": parity.matched,
                    "compared_components": parity.compared_components,
                    "mismatched_components": parity.mismatched_components,
                    "max_abs_error": parity.max_abs_error,
                    "absolute_tolerance": parity.absolute_tolerance,
                    "semantic_content_hash": parity.semantic_content_hash,
                }
            ),
            "observed_at": event.observed_at,
            "semantic_content_hash": event.semantic_content_hash,
        },
        occurred_at=event.observed_at,
        fact_id=event.event_id,
    )


def make_runtime_trace_terminal(
    coordinator: PassivePolicyArbiterShadow,
    *,
    terminal_status: str,
    reason: str,
    occurred_at: float | None = None,
) -> ControlTraceFact:
    status = str(terminal_status).strip().lower()
    if status not in {"completed", "truncated", "aborted"}:
        raise ValueError("terminal_status must be completed, truncated, or aborted.")
    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise ValueError("Terminal reason must be non-empty.")
    timestamp = coordinator.now() if occurred_at is None else float(occurred_at)
    payload = {
        "terminal_status": status,
        "reason": normalized_reason,
        "occurred_at": timestamp,
    }
    digest = canonical_content_hash(payload, domain=TRACE_SOURCE_ID_HASH_DOMAIN_V1)
    return ControlTraceFact.create(
        fact_id=f"runtime-trace-terminal:{digest}",
        event_type="runtime_trace_terminal",
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
        occurred_at=timestamp,
        clock_domain_id=coordinator.config.clock_domain_id,
        payload=payload,
    )


__all__ = [
    "TRACE_SOURCE_ID_HASH_DOMAIN_V1",
    "make_runtime_trace_terminal",
    "project_authority_event",
    "project_execution_gate_event",
    "project_human_proposal_adapter_result",
    "project_human_source_adapter_receipt",
    "project_legacy_execution",
    "project_composition_input",
    "project_composition_input_admission",
    "project_composition_input_snapshot",
    "project_composition_result",
    "project_composition_spec",
    "project_candidate_binding_eligibility",
    "project_legacy_replay_contract_derivation",
    "project_replay_binding_assessment",
    "project_replay_binding_evidence",
    "project_replay_action_record",
    "project_replay_proposal_adapter_result",
    "project_replay_source_action_contract",
    "project_replay_source_adapter_receipt",
    "project_replay_source_frame_emission",
    "project_target_runtime_action_contract",
    "project_shadow_startup",
    "project_shadow_step",
]

"""Pure v2.4 ControlTrace Core event contracts.

The contracts in this module describe logical facts and deterministic in-memory
record envelopes.  They deliberately do not choose a WAL, Parquet layout, or
dataset episode lifecycle owner.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .canonical_encoding import canonical_content_hash


CONTROL_TRACE_SCHEMA_V24 = "dexteleop_control_trace_v2.4"
CONTROL_TRACE_FACT_HASH_DOMAIN_V1 = "dexteleop.control_trace_fact.v1"
PERSISTED_RECORD_ENVELOPE_HASH_DOMAIN_V1 = (
    "dexteleop.control_trace_persisted_record_envelope.v1"
)


@dataclass(frozen=True)
class ControlTraceEventSpec:
    event_type: str
    event_family: str
    authority_kind: str = "neutral"


def _spec(event_type: str, event_family: str, authority_kind: str = "neutral") -> ControlTraceEventSpec:
    return ControlTraceEventSpec(event_type, event_family, authority_kind)


CONTROL_TRACE_EVENT_SPECS_V24 = (
    _spec("execution_manifest", "lifecycle"),
    _spec("conservative_compound_authority_profile", "arbitration_input"),
    _spec("compound_authority_failure", "execution", "authoritative"),
    _spec("station_run_manifest", "lifecycle"),
    _spec("station_intent_transition", "lifecycle"),
    _spec("station_operation_transition", "lifecycle"),
    _spec("station_publication_barrier_transition", "lifecycle"),
    _spec("exclusive_hitl_switch", "lifecycle"),
    _spec("composition_session_spec", "lifecycle"),
    _spec("composition_session_rearm_spec", "lifecycle"),
    _spec("composition_session_rearm_receipt", "lifecycle"),
    _spec("composition_session_transition", "lifecycle"),
    _spec("composition_activation_scope", "lifecycle"),
    _spec("composition_activation_binding", "lineage"),
    _spec("composition_input_semantic_binding", "lineage"),
    _spec("observation_cut", "observation"),
    _spec("policy_inference", "policy"),
    _spec("policy_action_emission", "policy"),
    _spec("policy_action_projection", "policy"),
    _spec("policy_execution_feedback", "policy"),
    _spec("component_action_trajectory", "source"),
    _spec("policy_context_admission", "admission"),
    _spec("policy_context", "policy"),
    _spec("policy_context_snapshot", "arbitration_input"),
    _spec("composition_context_admission", "admission"),
    _spec("allocation_publication_outcome", "admission"),
    _spec("composition_context", "context"),
    _spec("composition_context_snapshot", "arbitration_input"),
    _spec("composition_information_plan", "arbitration_input"),
    _spec("composition_temporal_selection_spec", "arbitration_input"),
    _spec("composition_source_eligibility_spec", "arbitration_input"),
    _spec("composition_source_eligibility_admission", "admission"),
    _spec("composition_source_eligibility_state", "arbitration_input"),
    _spec("composition_source_eligibility_snapshot", "arbitration_input"),
    _spec("composition_source_eligibility_binding", "arbitration_input"),
    _spec("composition_input_batch_admission", "admission"),
    _spec("composition_input_admission", "admission"),
    _spec("residual_anchor_lifecycle_spec", "arbitration_input"),
    _spec("residual_anchor_lifecycle_admission", "admission"),
    _spec("residual_anchor_lifecycle", "arbitration_input"),
    _spec("residual_anchor_lifecycle_snapshot", "arbitration_input"),
    _spec("action_payload_schema", "schema"),
    _spec("component_action_schema", "schema"),
    _spec("composition_spec", "arbitration_input"),
    _spec("component_composition_rule_set", "arbitration_input"),
    _spec("formal_component_composition_binding", "arbitration_input"),
    _spec("composition_input", "policy"),
    _spec("composition_contributor_input", "arbitration_input"),
    _spec("composition_input_snapshot", "arbitration_input"),
    _spec("formal_composition_strategy_spec", "arbitration_input"),
    _spec("formal_composition_strategy_input", "arbitration_input"),
    _spec("composition_dynamic_parameter_contract", "arbitration_input"),
    _spec("composition_strategy_revision", "arbitration_input"),
    _spec("composition_dynamic_parameter", "arbitration_input"),
    _spec("composition_dynamic_parameter_snapshot", "arbitration_input"),
    _spec("composition_strategy_control_binding", "arbitration_input"),
    _spec("hot_composition_strategy_input", "arbitration_input"),
    _spec("composition_evaluation", "proposal"),
    _spec("formal_composition_strategy_result_binding", "proposal"),
    _spec("composition_authority_disposition", "admission"),
    _spec("human_command_emission", "source"),
    _spec("replay_action_record", "source"),
    _spec("replay_source_frame_emission", "source"),
    _spec("legacy_replay_contract_derivation", "source"),
    _spec("replay_source_action_contract", "source"),
    _spec("target_runtime_action_contract", "source"),
    _spec("replay_binding_assessment", "source"),
    _spec("candidate_binding_eligibility", "arbitration_input"),
    _spec("replay_execution_authorization", "admission"),
    _spec("candidate_authorization_admission", "admission"),
    _spec("candidate_authorization_snapshot", "arbitration_input"),
    _spec("replay_source_adapter_receipt", "source"),
    _spec("producer_epoch_activation", "admission"),
    _spec("authority_event", "admission"),
    _spec("execution_gate_event", "admission"),
    _spec("producer_epoch_snapshot", "arbitration_input"),
    _spec("proposal_admission", "admission"),
    _spec("action_proposal", "proposal"),
    _spec("proposal_snapshot", "arbitration_input"),
    _spec("decision_schedule_tick", "arbitration_input"),
    _spec("authority_state_snapshot", "arbitration_input"),
    _spec("execution_gate_snapshot", "arbitration_input"),
    _spec("arbitration_input_snapshot", "arbitration_input"),
    _spec("arbitration_computation", "arbitration_output"),
    _spec("arbitration_evaluation", "arbitration_output", "shadow"),
    _spec("arbiter_shadow_status", "arbitration_output", "shadow"),
    _spec("component_mixing_shadow", "arbitration_output", "shadow"),
    _spec(
        "trajectory_composition_shadow_result",
        "arbitration_output",
        "shadow",
    ),
    _spec("source_difference_spec", "arbitration_input", "shadow"),
    _spec("source_difference_observation", "arbitration_output", "shadow"),
    _spec("guidance_feedback_command", "feedback", "shadow"),
    _spec("guidance_feedback_receipt", "feedback", "shadow"),
    _spec("arbitration_publication_attempt", "arbitration_output", "authoritative"),
    _spec("arbitration_decision", "arbitration_output", "authoritative"),
    _spec("arbitration_publication_rejected", "arbitration_output", "authoritative"),
    _spec("action_adaptation_request", "execution", "authoritative"),
    _spec("target_capability_assessment", "execution", "authoritative"),
    _spec("adapted_action_candidate", "execution", "authoritative"),
    _spec("action_adaptation_receipt", "execution", "authoritative"),
    _spec("target_execution_request", "execution", "authoritative"),
    _spec("target_execution_submission", "execution", "authoritative"),
    _spec("target_execution_completion", "execution", "authoritative"),
    _spec("execution_directive", "execution", "authoritative"),
    _spec("shared_execution_evaluation", "execution", "shadow"),
    _spec("legacy_execution_observation", "execution", "legacy"),
    _spec("source_adapter_receipt", "source"),
    _spec("source_validation", "safety"),
    _spec("shared_safety_result", "safety"),
    _spec("backend_guard_result", "safety"),
    _spec("resolved_execution_target", "reference", "authoritative"),
    _spec("execution_target_admission", "reference", "authoritative"),
    _spec("component_execution_target", "reference", "authoritative"),
    _spec("component_execution_target_admission", "reference", "authoritative"),
    _spec("component_execution_target_activation", "reference", "authoritative"),
    _spec("safety_anchor_commit_evidence", "reference", "authoritative"),
    _spec("reference_segment", "reference"),
    _spec("reference_receiver_validation", "reference", "authoritative"),
    _spec("reference_segment_admission", "reference", "authoritative"),
    _spec("group_generation_reservation", "reference", "authoritative"),
    _spec("group_generation_activation", "reference", "authoritative"),
    _spec("safety_anchor_commit_candidate", "reference", "authoritative"),
    _spec("reference_target_activation", "reference", "authoritative"),
    _spec("planner_command_frame", "command", "authoritative"),
    _spec("reference_segment_terminal", "reference", "authoritative"),
    _spec("command", "command"),
    _spec("dispatch_owner", "dispatch"),
    _spec("dispatch_receipt", "dispatch"),
    _spec("measurement_alignment", "measurement"),
    _spec("hardware_preflight_authorization", "execution"),
    _spec("execution_cancellation", "execution"),
    _spec("execution_cancellation_ack", "execution"),
    _spec("safety_control_operation", "execution"),
    _spec("safety_control_tick", "execution"),
    _spec("execution_worker_drain", "execution"),
    _spec("execution_worker_drain_aggregate", "execution"),
    _spec("execution_fault_escalation", "execution"),
    _spec("writer_health", "writer_health"),
    _spec("episode_membership_transition", "lifecycle"),
    _spec("runtime_trace_terminal", "lifecycle"),
)

CONTROL_TRACE_EVENT_SPEC_BY_TYPE_V24 = MappingProxyType(
    {spec.event_type: spec for spec in CONTROL_TRACE_EVENT_SPECS_V24}
)
CONTROL_TRACE_EVENT_TYPES_V24 = frozenset(CONTROL_TRACE_EVENT_SPEC_BY_TYPE_V24)
CONTROL_TRACE_EVENT_FAMILIES_V24 = frozenset(
    spec.event_family for spec in CONTROL_TRACE_EVENT_SPECS_V24
)


@dataclass(frozen=True)
class ControlTraceRecordSpec:
    """Explicit top-level payload contract for one persisted fact kind.

    Only event kinds listed here may currently be materialized as a
    ``ControlTraceFact``.  This keeps the broader event registry extensible
    without pretending that every planned event already has a stable record
    schema.
    """

    event_type: str
    schema_id: str
    schema_version: int
    payload_fields: frozenset[str]

    def __post_init__(self) -> None:
        if self.event_type not in CONTROL_TRACE_EVENT_TYPES_V24:
            raise ValueError(f"Unknown ControlTrace record event: {self.event_type!r}.")
        if not str(self.schema_id).strip() or int(self.schema_version) <= 0:
            raise ValueError("ControlTrace record schema identity is invalid.")
        fields = frozenset(str(value) for value in self.payload_fields)
        if not fields or any(not value for value in fields):
            raise ValueError("ControlTrace record payload fields must be non-empty.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "payload_fields", fields)

    def validate(self, payload: Mapping[str, Any]) -> None:
        actual = frozenset(str(key) for key in payload)
        if actual != self.payload_fields:
            missing = sorted(self.payload_fields - actual)
            extra = sorted(actual - self.payload_fields)
            raise ValueError(
                f"ControlTrace {self.event_type!r} payload does not match "
                f"{self.schema_id}: missing={missing!r}, extra={extra!r}."
            )


def _record_spec(
    event_type: str,
    *payload_fields: str,
    schema_version: int = 1,
) -> ControlTraceRecordSpec:
    return ControlTraceRecordSpec(
        event_type=event_type,
        schema_id=f"dexteleop.control_trace.{event_type}.v{schema_version}",
        schema_version=schema_version,
        payload_fields=frozenset(payload_fields),
    )


# These are the explicit schemas implemented by the current M3/M4A seam plus
# lifecycle/source-admission records needed before M4B. The component shadow
# schema is a non-authoritative CS-6b2 fixture record. Planned event kinds in
# CONTROL_TRACE_EVENT_SPECS_V24 intentionally remain absent until their producer
# and projection contract exist.
CONTROL_TRACE_RECORD_SPECS_V1 = (
    _record_spec(
        "execution_manifest",
        "manifest_id", "manifest_version", "action_space",
        "action_schema_hash", "component_target_capability_hash",
        "component_catalog_hash", "controlled_components",
        "selection_groups", "motion_authority_sources",
        "human_motion_proposal_status", "safety_control_sources",
        "safety_control_requests", "lifecycle_control_sources",
        "lifecycle_control_requests", "authoritative_decision_hz",
        "reference_activation_spec_id", "reference_activation_spec_hash",
        "execution_capability_profile_hash",
        "reference_timing_semantics_hz",
        "actual_reference_frame_producer", "planner_command_hz",
        "exclusions", "semantic_content_hash",
        schema_version=2,
    ),
    _record_spec(
        "conservative_compound_authority_profile",
        "profile_id", "producer_id", "component_coverage",
        "selection_groups", "composition_spec_hash",
        "component_rule_set_hash", "formal_strategy_spec_hash",
        "atomicity_mode", "missing_or_stale_behavior",
        "safety_failure_behavior", "downstream_failure_behavior",
        "activation_commit_mode", "semantic_content_hash",
    ),
    _record_spec(
        "compound_authority_failure",
        "failure_id", "profile_hash", "producer_id", "stage",
        "failure_behavior", "source_event_ids", "semantic_content_hash",
    ),
    _record_spec(
        "station_run_manifest",
        "manifest_id", "manifest_version", "execution_binding",
        "source_slots", "authority_policy_hash", "decision_schedule_hash",
        "trace_requirement_profile_id", "capture_policy_id",
        "allowed_intent_kinds", "operator_control_lease_sec",
        "explicit_exclusions", "semantic_content_hash",
    ),
    _record_spec(
        "station_intent_transition",
        "intent_id", "intent_kind", "client_id", "client_sequence",
        "subject_id", "resource_binding_id", "operation_spec_id",
        "produced_at", "produced_clock_domain_id", "available_at",
        "available_clock_domain_id", "admission_sequence",
        "expected_state_revision", "manifest_hash", "reason", "accepted",
        "status", "state_before", "state_after", "intent_semantic_hash",
        "receipt_semantic_hash", "transition_semantic_hash",
    ),
    _record_spec(
        "station_operation_transition",
        "operation_result_id", "source_intent_id", "operation_kind",
        "subject_id", "succeeded", "completed_at",
        "completed_clock_domain_id", "available_at",
        "available_clock_domain_id", "admission_sequence", "result_status",
        "evidence_refs", "transition_accepted", "transition_status",
        "state_before", "state_after", "result_semantic_hash",
        "transition_semantic_hash",
    ),
    _record_spec(
        "station_publication_barrier_transition",
        "source_slot_id", "generation", "enabled",
        "authority_operation_result_id", "authority_evidence_refs",
        "transition_reason", "causal_station_fact_id",
        "snapshot_semantic_hash",
    ),
    _record_spec(
        "exclusive_hitl_switch",
        "switch_id", "trigger_intent_id", "trigger_intent_hash",
        "direction", "previous_source", "target_source", "requested_at",
        "completed_at", "clock_domain_id", "succeeded", "status",
        "hold_evidence_refs", "home_capture_id", "home_capture_hash",
        "home_capture", "source_selection_record_id",
        "source_selection_record_hash", "activation_id", "activation_hash",
        "target_emission_id", "target_emission_hash",
        "source_reset_performed", "evidence_refs", "error",
        "authoritative", "semantic_content_hash",
    ),
    _record_spec(
        "composition_session_spec",
        "operation_spec_id", "session_key", "composer_source_slot_id",
        "contributor_source_slot_ids", "composition_spec_hash",
        "authority_gate_spec_hash", "execution_binding_hash",
        "capture_boundary_id", "pause_behavior", "end_behavior",
        "resume_behavior", "semantic_content_hash",
    ),
    _record_spec(
        "composition_session_transition",
        "transition_id", "operation_spec_id", "operation_spec_hash",
        "session_epoch", "transition", "reason", "source_fact_id",
        "source_fact_hash", "state_before_hash", "state_after_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_session_rearm_spec",
        "rearm_spec_id", "rearm_key", "logical_operation_spec_id",
        "logical_operation_spec_hash", "prior_session_epoch",
        "prior_state_hash", "execution_binding_hash",
        "readiness_observer_binding_id", "readiness_timeout_sec",
        "required_readiness_checks", "semantic_content_hash",
    ),
    _record_spec(
        "composition_session_rearm_receipt",
        "receipt_id", "rearm_spec_id", "rearm_spec_hash",
        "logical_operation_spec_id", "logical_operation_spec_hash",
        "prior_session_epoch", "next_session_epoch",
        "prior_composer_producer_epoch", "composer_producer_epoch",
        "prior_authority_epoch", "authority_epoch",
        "prior_execution_epoch", "execution_epoch",
        "cancellation_receipt_id", "measurement_id", "measurement_hash",
        "producer_activation_id", "authority_event_ids",
        "ready_gate_event_id", "activation_input_barrier_time",
        "completed_at", "clock_domain_id", "satisfied_readiness_checks",
        "status", "semantic_content_hash",
    ),
    _record_spec(
        "composition_activation_scope",
        "activation_id", "operation_spec_id", "operation_spec_hash",
        "session_epoch", "activated_at", "clock_domain_id",
        "input_barrier_at", "freshness_policy", "required_roles",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_activation_binding",
        "binding_id", "subject_kind", "subject_id", "subject_hash",
        "activation_id", "activation_hash", "semantic_content_hash",
    ),
    _record_spec(
        "composition_input_semantic_binding",
        "binding_id", "composition_input_id", "composition_input_hash",
        "binding_id_by_kind", "binding_hash_by_kind",
        "semantic_content_hash",
    ),
    _record_spec(
        "observation_cut",
        "observation_cut_id", "runtime_session_id", "control_domain_id",
        "policy_node_id", "policy_node_epoch", "sequence", "cutoff_at",
        "cutoff_clock_domain_id", "cutoff_clock_reset_epoch",
        "cutoff_admission_sequence", "built_at", "built_clock_domain_id",
        "selected_sample_by_stream", "context_dependencies",
        "observation_contract_hash", "policy_io_layout_hash", "trigger_reason",
        "semantic_content_hash",
    ),
    _record_spec(
        "replay_action_record",
        "replay_action_record_id", "source_contract_id", "source_contract_hash",
        "episode_id", "source_event_id", "source_action_index",
        "source_action_stage", "source_timestamp",
        "source_timestamp_clock_domain_id", "canonical_action_payload",
        "canonical_action_hash", "schedule_id", "schedule_hash",
        "old_safety_profile_hash", "semantic_content_hash",
    ),
    _record_spec(
        "legacy_replay_contract_derivation",
        "derivation_receipt_id", "derivation_contract_id",
        "derivation_contract_hash", "source_dataset_schema",
        "consumed_metadata_fields", "derived_action_stage",
        "derived_pose_frame_contract_hash", "derived_component_schema_hash",
        "unresolved_dimensions", "derivation_status", "semantic_content_hash",
    ),
    _record_spec(
        "replay_source_action_contract",
        "contract_id", "interpretation_status", "action_stage", "action_space",
        "component_ids", "selection_group_by_component",
        "coupled_selection_groups", "pose_frame_contract_hash",
        "component_schema_hash", "action_spec_hash_by_component",
        "units_convention_id", "derivation_receipt_id",
        "derivation_receipt_hash", "unresolved_dimensions",
        "semantic_content_hash",
    ),
    _record_spec(
        "target_runtime_action_contract",
        "target_contract_id", "control_domain_id", "embodiment_instance_id",
        "component_schema_hash", "component_ids",
        "selection_group_by_component", "accepted_action_spaces_by_component",
        "coupled_selection_groups", "pose_frame_contract_hash",
        "action_spec_hash_by_component", "units_convention_id",
        "supported_transform_hash_by_source_frame",
        "execution_capability_profile_hash", "unresolved_dimensions",
        "semantic_content_hash",
    ),
    _record_spec(
        "replay_binding_assessment",
        "assessment_id", "binding_spec_id", "binding_spec_hash",
        "source_action_contract_id", "source_action_contract_hash",
        "target_runtime_contract_id", "target_runtime_contract_hash", "verdict",
        "checks", "permitted_uses", "blocking_reasons", "verifier_contract_id",
        "verifier_contract_hash", "semantic_content_hash",
    ),
    _record_spec(
        "candidate_binding_eligibility",
        "eligibility_id", "assessment_id", "assessment_hash",
        "source_emission_id", "requested_use", "evaluation_allowed",
        "authoritative_allowed", "conditional_evaluation",
        "same_legacy_emission", "reasons", "target_runtime_contract_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "replay_execution_authorization",
        "authorization_id", "requested_mode", "assessment_id",
        "assessment_hash", "safety_preflight_id",
        "execution_capability_evidence_id", "dispatch_owner_state_id",
        "operator_authorization_id", "authorized", "reasons",
        "semantic_content_hash",
    ),
    _record_spec(
        "candidate_authorization_admission",
        "authorization_id", "accepted", "rejection_code", "message",
        "proposal_id", "proposal_content_hash", "producer_kind",
        "evidence_id", "evidence_hash", "binding_eligibility_id",
        "binding_eligibility_hash", "target_runtime_contract_hash",
        "authoritative_allowed", "reasons", "available_at",
        "available_clock_domain_id", "control_admission_sequence",
        "semantic_content_hash",
    ),
    _record_spec(
        "replay_source_adapter_receipt",
        "representation_trace_id", "adapter_contract_id", "source_contract_id",
        "source_contract_hash", "replay_action_record_id",
        "replay_action_record_hash", "source_event_id", "source_emission_id",
        "source_action_index", "source_action_stage", "target_binding_spec_id",
        "target_binding_spec_hash", "binding_assessment_id",
        "binding_assessment_hash", "binding_verdict", "binding_eligibility_id",
        "binding_eligibility_hash", "accepted", "evaluation_allowed",
        "authoritative_eligible", "conditional_evaluation",
        "old_safe_does_not_bypass_current_safety", "rejection_reasons",
        "output_action_semantic_hash", "semantic_content_hash",
    ),
    _record_spec(
        "replay_source_frame_emission",
        "emission_id", "source_path", "source_episode_index", "group_key",
        "local_frame_index", "source_action_index", "source_relative_time",
        "action", "observed_at", "target_effective_at", "effective_from",
        "effective_until", "clock_domain_id", "semantic_content_hash",
    ),
    _record_spec(
        "human_command_emission",
        "source_emission_id", "source_emission_sequence",
        "source_emission_semantic_hash", "emitted_at", "command_timestamp",
        "contributions", "merged_commands", "winning_contribution_by_effector",
        "overwritten_contributions_by_effector", "merged_command_metadata",
        "unfreezable_merged_metadata_keys", "observation_sample_refs",
        "observation_sample_set_hash",
    ),
    _record_spec(
        "policy_action_emission",
        "emission_id", "admission_mode", "policy_node_id", "inference_id",
        "policy_output_id", "chunk_id", "chunk_keyframe_index", "chunk_size",
        "pre_safety_action", "pre_safety_action_hash", "representation_trace_id",
        "pre_safety_faults", "produced_at", "produced_clock_domain_id",
        "emitted_at", "target_effective_at", "effective_from", "effective_until",
        "effective_clock_domain_id", "schedule_spec_id", "schedule_spec_hash",
        "queue_len_before", "queue_len_after", "queue_pop_count",
        "semantic_content_hash",
    ),
    _record_spec(
        "policy_action_projection",
        "emission_id", "emission_hash", "policy_node_id", "inference_id",
        "policy_output_id", "projection",
    ),
    _record_spec(
        "policy_execution_feedback",
        "feedback", "application_receipt_id", "application_receipt_hash",
        "application_decision", "application_reason",
        "target_generation_before", "target_generation_after",
    ),
    _record_spec(
        "component_action_trajectory",
        "trajectory_id", "revision", "source_kind", "source_instance_id",
        "source_clock_domain_id", "created_at", "action_schema_hash",
        "component_coverage", "effective_from", "effective_until",
        "source_fact_ids", "source_fact_hashes", "source_activation_id",
        "source_activation_hash", "source_immutable_through",
        "coordination_group_id", "keyframes", "semantic_content_hash",
    ),
    _record_spec(
        "policy_context_admission",
        "context_id", "accepted", "rejection_code", "message",
        "available_at", "available_clock_domain_id",
        "control_admission_sequence", "context_semantic_content_hash",
    ),
    _record_spec(
        "policy_context",
        "context_id", "runtime_session_id", "control_domain_id", "producer_id",
        "producer_epoch", "sequence", "policy_node_id", "context_channel",
        "context_kind", "semantic_payload", "produced_at",
        "produced_clock_domain_id", "available_at", "available_clock_domain_id",
        "control_admission_sequence", "effective_from", "effective_until",
        "effective_clock_domain_id", "context_contract_hash", "observation_cut_id",
        "inference_id", "policy_output_id", "parent_context_ids", "blob_refs",
        "producer_artifact_hash", "context_semantic_content_hash",
    ),
    _record_spec(
        "policy_context_snapshot",
        "context_snapshot_id", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "target_effective_at", "required_channels",
        "selected_context_id_by_channel", "selected_context_hash_by_channel",
        "missing_channels", "producer_epoch_snapshot_hash", "semantic_content_hash",
    ),
    _record_spec(
        "composition_context_admission",
        "context_id", "accepted", "rejection_code", "message",
        "available_at", "available_clock_domain_id",
        "control_admission_sequence", "context_semantic_content_hash",
    ),
    _record_spec(
        "allocation_publication_outcome",
        "outcome_kind", "allocation_scope_id", "runtime_status",
        "committed_reducer_state_hash", "reason", "decision_id",
        "decision_hash", "attempt_id", "attempt_hash",
        "publication_sequence", "receipt_context_id",
        "receipt_context_hash", "rejection_code", "semantic_content_hash",
    ),
    _record_spec(
        "composition_context",
        "context_id", "runtime_session_id", "control_domain_id", "producer_id",
        "producer_epoch", "sequence", "context_channel", "context_kind",
        "semantic_payload", "produced_at", "produced_clock_domain_id",
        "available_at", "available_clock_domain_id",
        "control_admission_sequence", "effective_from", "effective_until",
        "effective_clock_domain_id", "payload_contract_hash",
        "source_fact_ids", "source_fact_hashes", "parent_context_ids",
        "blob_refs", "availability", "fault_code", "producer_node_id",
        "observation_cut_id", "inference_id", "policy_output_id",
        "producer_artifact_hash", "context_semantic_content_hash",
    ),
    _record_spec(
        "composition_context_snapshot",
        "context_snapshot_id", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "target_effective_at", "required_channels",
        "selected_context_id_by_channel", "selected_context_hash_by_channel",
        "missing_channels", "unavailable_channels",
        "producer_epoch_snapshot_hash", "semantic_content_hash",
    ),
    _record_spec(
        "composition_information_plan",
        "plan_id", "context_requirements", "source_difference_spec_hash",
        "diagnostic_level", "semantic_content_hash",
    ),
    _record_spec(
        "composition_temporal_selection_spec",
        "spec_id", "selection", "overlap_behavior",
        "maximum_target_lead_sec", "maximum_target_lag_sec",
        "action_timestamp_semantics", "semantic_content_hash",
    ),
    _record_spec(
        "composition_source_eligibility_spec",
        "spec_id", "composition_spec_hash", "source_slot_by_role",
        "producer_id_by_source_slot", "semantic_content_hash",
    ),
    _record_spec(
        "composition_source_eligibility_admission",
        "source_eligibility_state_id", "accepted", "rejection_code",
        "message", "available_at", "available_clock_domain_id",
        "control_admission_sequence", "state_semantic_content_hash",
    ),
    _record_spec(
        "composition_source_eligibility_state",
        "source_eligibility_state_id", "runtime_session_id",
        "control_domain_id", "source_slot_id", "producer_id",
        "producer_epoch", "status_epoch", "sequence", "status",
        "observed_at", "observed_clock_domain_id", "available_at",
        "available_clock_domain_id", "control_admission_sequence",
        "effective_from", "lease_expires_at", "effective_clock_domain_id",
        "rearm_required", "fault_code", "rearm_receipt_id",
        "rearm_receipt_hash", "state_semantic_content_hash",
    ),
    _record_spec(
        "composition_source_eligibility_snapshot",
        "source_eligibility_snapshot_id", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "target_effective_at",
        "source_eligibility_spec_hash", "selected_state_ids_by_role",
        "selected_state_hashes_by_role", "ineligible_reason_by_role",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_source_eligibility_binding",
        "binding_id", "composition_input_snapshot_id",
        "composition_input_snapshot_hash", "source_eligibility_snapshot_id",
        "source_eligibility_snapshot_hash", "semantic_content_hash",
    ),
    _record_spec(
        "action_payload_schema",
        "schema_id", "schema_version", "representation_family", "dtype",
        "shape", "coordinate_names", "units", "value_semantics",
        "reference_frame", "controlled_frame", "expressed_in_frame",
        "semantic_parameters", "semantic_content_hash",
    ),
    _record_spec(
        "component_action_schema",
        "schema_id", "schema_version", "component_catalog_hash",
        "payload_schema_hash_by_component", "semantic_parameters",
        "semantic_content_hash", "component_catalog",
    ),
    _record_spec(
        "composition_spec",
        "spec_id", "algebra", "input_roles", "required_roles",
        "participation_requirement_by_role", "depends_on_roles_by_role",
        "missing_required_behavior", "missing_residual_behavior",
        "residual_limit_by_component", "weight_by_role",
        "max_translation_disagreement_m", "max_rotation_disagreement_rad",
        "pose_component_dependencies", "component_rule_set_hash",
        "action_schema_hash", "input_action_schema_hash_by_role",
        "semantic_content_hash",
    ),
    _record_spec(
        "component_composition_rule_set",
        "rule_set_id", "action_space", "rules", "semantic_content_hash",
    ),
    _record_spec(
        "formal_component_composition_binding",
        "binding_id", "composition_spec_hash", "component_rule_set_hash",
        "formal_strategy_spec_hash", "semantic_content_hash",
    ),
    _record_spec(
        "composition_input_batch_admission",
        "batch_id", "batch_semantic_content_hash", "accepted",
        "rejection_code", "message", "available_at",
        "available_clock_domain_id", "first_control_admission_sequence",
        "last_control_admission_sequence", "input_ids", "input_hashes",
    ),
    _record_spec(
        "composition_input_admission",
        "composition_input_id", "accepted", "rejection_code", "message",
        "available_at", "available_clock_domain_id",
        "control_admission_sequence", "input_semantic_content_hash",
    ),
    _record_spec(
        "residual_anchor_lifecycle_spec",
        "spec_id", "source_role", "component_ids",
        "absolute_action_schema_hash", "relative_action_schema_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "residual_anchor_lifecycle_admission",
        "lifecycle_record_id", "accepted", "accepted_existing",
        "rejection_code", "message", "available_at",
        "available_clock_domain_id", "control_admission_sequence",
        "lifecycle_record_hash",
    ),
    _record_spec(
        "residual_anchor_lifecycle",
        "lifecycle_record_id", "producer_id", "producer_epoch", "sequence",
        "lifecycle_scope_id", "lifecycle_spec_hash", "phase",
        "effective_for_target_at", "observed_at", "observed_clock_domain_id",
        "available_at", "available_clock_domain_id",
        "control_admission_sequence", "composition_activation_id",
        "composition_activation_hash", "anchor_request", "anchor_receipt",
        "semantic_content_hash",
    ),
    _record_spec(
        "residual_anchor_lifecycle_snapshot",
        "lifecycle_snapshot_id", "cutoff_at", "cutoff_admission_sequence",
        "target_effective_at", "lifecycle_scope_id", "lifecycle_spec_hash",
        "status", "reason", "selected_record_id", "selected_record_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_input",
        "composition_input_id", "runtime_session_id", "control_domain_id",
        "producer_id", "producer_epoch", "sequence", "input_role", "action",
        "produced_at", "produced_clock_domain_id", "available_at",
        "available_clock_domain_id", "control_admission_sequence", "effective_from",
        "effective_until", "effective_clock_domain_id", "context_ids",
        "observation_cut_id", "policy_output_id", "authority_candidate",
        "input_semantic_content_hash",
    ),
    _record_spec(
        "composition_contributor_input",
        "composition_input_id", "runtime_session_id", "control_domain_id",
        "producer_id", "producer_epoch", "sequence", "input_role", "action",
        "produced_at", "produced_clock_domain_id", "available_at",
        "available_clock_domain_id", "control_admission_sequence", "effective_from",
        "effective_until", "effective_clock_domain_id", "context_ids",
        "observation_cut_id", "policy_output_id", "authority_candidate",
        "source_fact_ids", "source_fact_hashes",
        "input_semantic_content_hash",
    ),
    _record_spec(
        "composition_input_snapshot",
        "composition_input_snapshot_id", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "target_effective_at", "composition_spec_hash",
        "action_schema_hash", "target_clock_domain_id",
        "input_action_schema_hash_by_role",
        "temporal_selection_spec_hash",
        "selected_input_ids_by_role", "selected_input_hashes_by_role",
        "selected_input_ids_before_dependency_by_role",
        "selected_input_hashes_before_dependency_by_role",
        "dependency_ineligibility_reason_by_role", "required_roles",
        "missing_roles", "producer_epoch_snapshot_hash", "semantic_content_hash",
    ),
    _record_spec(
        "formal_composition_strategy_spec",
        "strategy_id", "strategy_version", "execution_class",
        "composition_spec_hash", "required_context_channels",
        "required_residual_anchor_lifecycle_spec_hash",
        "parameter_payload", "maximum_decision_latency_sec",
        "semantic_content_hash",
        schema_version=2,
    ),
    _record_spec(
        "formal_composition_strategy_input",
        "strategy_input_id", "runtime_session_id", "control_domain_id",
        "clock_domain_id", "strategy_id", "strategy_spec_hash",
        "composition_spec_hash", "composition_input_snapshot_id",
        "composition_input_snapshot_hash", "context_snapshot_id",
        "context_snapshot_hash", "information_plan_hash",
        "residual_anchor_lifecycle_snapshot_id",
        "residual_anchor_lifecycle_snapshot_hash",
        "semantic_content_hash",
        schema_version=2,
    ),
    _record_spec(
        "composition_dynamic_parameter_contract",
        "parameter_set_id", "parameter_semantics", "base_role",
        "overlay_role", "component_ids", "default_alpha_by_component",
        "lifecycle", "semantic_content_hash",
    ),
    _record_spec(
        "composition_strategy_revision",
        "revision_id", "runtime_session_id", "control_domain_id",
        "strategy_instance_id", "revision_sequence",
        "previous_revision_hash", "request_id", "strategy_mode",
        "composition_spec_hash", "strategy_spec_hash",
        "component_rule_set_hash", "dynamic_parameter_contract_hash",
        "initial_alpha_by_component", "available_at",
        "available_clock_domain_id", "control_admission_sequence",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_dynamic_parameter",
        "parameter_record_id", "runtime_session_id", "control_domain_id",
        "producer_id", "producer_generation", "sequence",
        "strategy_revision_hash", "parameter_contract_hash",
        "alpha_by_component", "produced_at", "produced_clock_domain_id",
        "available_at", "available_clock_domain_id", "effective_from",
        "effective_until", "effective_clock_domain_id",
        "control_admission_sequence", "semantic_content_hash",
    ),
    _record_spec(
        "composition_dynamic_parameter_snapshot",
        "parameter_snapshot_id", "strategy_revision_hash",
        "parameter_contract_hash", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "target_effective_at",
        "selected_parameter_record_id", "selected_parameter_record_hash",
        "alpha_by_component", "semantic_content_hash",
    ),
    _record_spec(
        "composition_strategy_control_binding",
        "binding_id", "strategy_revision_id", "strategy_revision_hash",
        "parameter_snapshot_id", "parameter_snapshot_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "hot_composition_strategy_input",
        "strategy_input_id", "runtime_session_id", "control_domain_id",
        "clock_domain_id", "strategy_id", "strategy_spec_hash",
        "composition_spec_hash", "composition_input_snapshot_id",
        "composition_input_snapshot_hash", "context_snapshot_id",
        "context_snapshot_hash", "information_plan_hash",
        "residual_anchor_lifecycle_snapshot_id",
        "residual_anchor_lifecycle_snapshot_hash",
        "strategy_control_binding_id", "strategy_control_binding_hash",
        "semantic_content_hash",
        schema_version=2,
    ),
    _record_spec(
        "composition_evaluation",
        "composition_evaluation_id", "composition_input_snapshot_id",
        "composition_input_snapshot_hash", "composition_spec_hash", "output_status",
        "action_schema_hash", "target_effective_at", "target_clock_domain_id",
        "reason", "composed_candidate", "composed_candidate_hash", "input_ids",
        "input_hashes", "context_ids", "composition_supported_until",
        "strategy_result_data",
        "semantic_content_hash",
        schema_version=2,
    ),
    _record_spec(
        "formal_composition_strategy_result_binding",
        "binding_id", "strategy_input_id", "strategy_input_hash",
        "composition_evaluation_id", "composition_evaluation_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "composition_authority_disposition",
        "disposition_id", "gate_spec_hash", "composition_evaluation_id",
        "composition_evaluation_hash", "action", "reason", "missing_roles",
        "semantic_content_hash",
    ),
    _record_spec(
        "source_difference_spec",
        "spec_id", "composition_spec_hash",
        "compared_reference_role_pairs", "output_residual_roles",
        "semantic_content_hash",
    ),
    _record_spec(
        "source_difference_observation",
        "observation_id", "composition_input_snapshot_id",
        "composition_input_snapshot_hash", "composition_evaluation_id",
        "composition_evaluation_hash", "composition_spec_hash",
        "source_difference_spec_hash", "target_effective_at",
        "evaluated_at", "clock_domain_id", "valid_until",
        "pairwise_differences", "output_residuals", "semantic_content_hash",
    ),
    _record_spec(
        "guidance_feedback_command",
        "command_id", "spec_hash", "source_observation_id",
        "source_observation_hash", "sequence", "status", "reason",
        "produced_at", "valid_until", "clock_domain_id", "cues",
        "feedback_kind", "authoritative", "semantic_content_hash",
    ),
    _record_spec(
        "guidance_feedback_receipt",
        "receipt_id", "command_id", "command_hash", "adapter_id",
        "accepted", "status", "observed_at", "clock_domain_id",
        "semantic_content_hash",
    ),
    _record_spec(
        "producer_epoch_activation",
        "activation_id", "runtime_session_id", "control_domain_id", "producer_id",
        "producer_epoch", "previous_epoch", "activation_sequence", "produced_at",
        "produced_clock_domain_id", "available_at", "available_clock_domain_id",
        "activated_at", "control_admission_sequence", "reason",
        "semantic_content_hash",
    ),
    _record_spec(
        "authority_event",
        "event_id", "runtime_session_id", "control_domain_id", "sequence",
        "authority_epoch", "authority_event_type", "selection_group_id",
        "owner_producer_id", "control_mode", "expires_at",
        "expires_clock_domain_id", "produced_at", "produced_clock_domain_id",
        "available_at", "available_clock_domain_id", "control_admission_sequence",
        "reason", "semantic_content_hash",
    ),
    _record_spec(
        "execution_gate_event",
        "event_id", "runtime_session_id", "control_domain_id", "sequence",
        "gate_epoch", "status", "reason", "source_event_ids", "produced_at",
        "produced_clock_domain_id", "available_at", "available_clock_domain_id",
        "control_admission_sequence", "semantic_content_hash",
    ),
    _record_spec(
        "source_adapter_receipt",
        "representation_trace_id", "adapter_contract_id", "adapter_contract_hash",
        "source_emission_id", "source_emission_sequence",
        "source_emission_semantic_hash", "component_mappings", "lossiness",
        "source_validation_accepted", "authoritative_eligible",
        "source_validation_reasons", "output_action_space",
        "output_action_semantic_hash", "semantic_content_hash",
    ),
    _record_spec(
        "source_validation",
        "validation_id", "source_event_id", "accepted", "reasons",
    ),
    _record_spec(
        "proposal_admission",
        "emission_id", "accepted", "rejection_code", "message", "proposal_id",
        "available_at", "available_clock_domain_id", "control_admission_sequence",
        "proposal_semantic_content_hash",
    ),
    _record_spec(
        "action_proposal",
        "proposal_id", "runtime_session_id", "control_domain_id",
        "embodiment_instance_id", "producer_kind", "producer_id", "producer_epoch",
        "sequence", "action", "produced_at", "produced_clock_domain_id",
        "available_at", "available_clock_domain_id", "control_admission_sequence",
        "effective_from", "effective_until", "target_effective_at",
        "effective_clock_domain_id", "action_schema_hash", "component_coverage",
        "selection_groups", "authority_priority", "authority_request", "validity",
        "status", "fallback_policy", "confidence", "policy_node_id",
        "observation_cut_id", "context_ids", "inference_id", "policy_output_id",
        "chunk_id", "chunk_keyframe_index", "representation_trace_id",
        "composition_input_ids", "composition_input_hashes",
        "composition_spec_hash", "composition_evaluation_id",
        "composition_evaluation_hash",
        "proposal_semantic_hash_domain",
        "proposal_semantic_content_hash",
    ),
    _record_spec(
        "proposal_snapshot",
        "proposal_snapshot_id", "control_domain_id", "cutoff_at",
        "cutoff_admission_sequence", "included_admission_sequence_min",
        "included_admission_sequence_max", "target_effective_at",
        "action_schema_hash", "producer_epoch_snapshot_hash",
        "reduction_policy_id", "candidates", "dispositions",
        "semantic_content_hash",
    ),
    _record_spec(
        "decision_schedule_tick",
        "schedule_tick_id", "tick_index", "scheduled_decision_at",
        "decision_cutoff_at", "cutoff_admission_sequence", "target_effective_at",
        "publication_deadline_at", "trigger_ids", "trigger_reasons",
        "coalesced_trigger_ids", "coalesced_trigger_count",
        "trigger_evidence_truncated", "schedule_spec_hash", "semantic_content_hash",
    ),
    _record_spec(
        "authority_state_snapshot",
        "snapshot_fact_id", "arbitration_input_snapshot_id", "control_domain_id",
        "cutoff_at", "cutoff_admission_sequence", "included_admission_sequence_min",
        "included_admission_sequence_max", "authority_policy_hash", "control_mode",
        "authority_epoch", "active_leases", "causal_event_ids",
        "semantic_content_hash",
    ),
    _record_spec(
        "execution_gate_snapshot",
        "snapshot_fact_id", "arbitration_input_snapshot_id", "control_domain_id",
        "cutoff_at", "cutoff_admission_sequence", "included_admission_sequence_min",
        "included_admission_sequence_max", "status", "gate_epoch", "reason",
        "source_event_ids", "semantic_content_hash",
    ),
    _record_spec(
        "producer_epoch_snapshot",
        "snapshot_fact_id", "arbitration_input_snapshot_id", "control_domain_id",
        "cutoff_at", "cutoff_admission_sequence", "included_admission_sequence_min",
        "included_admission_sequence_max", "active_epoch_by_producer",
        "activation_id_by_producer", "semantic_content_hash",
    ),
    _record_spec(
        "candidate_authorization_snapshot",
        "snapshot_id", "arbitration_input_snapshot_id", "control_domain_id",
        "cutoff_at", "cutoff_admission_sequence",
        "target_runtime_contract_hash", "required_producer_kinds",
        "authorization_by_proposal_id", "dispositions_by_proposal_id",
        "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_input_snapshot",
        "arbitration_input_snapshot_id", "runtime_session_id", "control_domain_id",
        "embodiment_instance_id", "proposal_snapshot_id", "proposal_snapshot_hash",
        "authority_state_snapshot_fact_id", "authority_state_snapshot_hash",
        "execution_gate_snapshot_fact_id", "execution_gate_snapshot_hash",
        "producer_epoch_snapshot_fact_id", "producer_epoch_snapshot_hash",
        "candidate_authorization_snapshot_fact_id",
        "candidate_authorization_snapshot_hash",
        "authority_policy_hash", "previous_authoritative_decision_id",
        "previous_selection_by_group", "active_selection_groups",
        "action_schema_hash", "component_catalog_hash",
        "decision_schedule_tick_id",
        "decision_schedule_tick_hash", "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_computation",
        "result_id", "arbitration_input_snapshot_id", "arbitration_input_hash",
        "proposal_snapshot_id", "arbiter_contract_id", "arbiter_contract_hash",
        "arbiter_implementation_hash", "authority_policy_hash",
        "component_selection_policy_hash", "decision_schedule_spec_hash",
        "canonical_encoding_version", "schedule_tick_id", "decision_cutoff_at",
        "decided_at", "target_effective_at", "target_clock_domain_id",
        "publication_deadline_at",
        "authority_epoch", "control_mode", "execution_gate_status", "gate_epoch",
        "selected_by_group", "selected_by_component", "owner_by_component",
        "status_by_group", "composed_candidate", "selected_action_hash",
        "supported_until", "output_status", "rejections", "attribution",
        "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_evaluation",
        "evaluation_id", "mode", "non_authoritative",
        "arbitration_input_snapshot_id", "arbitration_input_hash",
        "proposal_snapshot_id", "computation_result_id", "computation_result_hash",
        "output_status", "would_select_action", "would_select_action_hash",
        "would_select_by_group", "would_select_by_component", "evaluated_at",
        "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_publication_attempt",
        "publication_attempt_id", "result_id", "attempted_at", "accepted",
        "rejection_reason", "required_fallback", "coalesced_before_publication",
        "computation_started_at", "computation_completed_at",
        "computation_runtime_sec", "runtime_budget_sec", "runtime_budget_missed",
        "publication_deadline_missed_at_completion",
        "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_decision",
        "decision_id", "result_id", "arbitration_input_snapshot_id",
        "arbitration_result_hash", "selected_action", "selected_action_hash",
        "action_schema_hash", "selected_by_group", "target_effective_at",
        "supported_until", "target_clock_domain_id",
        "published_at", "authority_epoch", "gate_epoch", "semantic_content_hash",
    ),
    _record_spec(
        "arbitration_publication_rejected",
        "rejection_event_id", "publication_attempt_id", "result_id",
        "rejected_at", "rejection_reason", "required_fallback",
        "semantic_content_hash",
    ),
    _record_spec(
        "action_adaptation_request",
        "request_id", "authoritative_decision_id",
        "authoritative_decision_hash", "arbitration_result_id",
        "candidate_action_hash", "action_schema_hash",
        "component_target_capability_id",
        "component_target_capability_hash", "component_coverage",
        "target_effective_at", "supported_until", "authority_sequence",
        "execution_gate_sequence", "execution_fence_sequence",
        "requested_at", "clock_domain_id", "semantic_content_hash",
    ),
    _record_spec(
        "target_capability_assessment",
        "assessment_id", "capability_id", "capability_hash",
        "candidate_action_hash", "candidate_action_schema_hash",
        "component_coverage", "active_components_by_resource",
        "accepted", "rejection_reasons", "semantic_content_hash",
    ),
    _record_spec(
        "adapted_action_candidate",
        "adapted_candidate_id", "request_id", "request_hash",
        "source_candidate_action_hash", "payload_kind",
        "output_action_contract_hash", "adapter_contract_id",
        "adapter_contract_hash", "payload_semantic", "created_at",
        "clock_domain_id", "semantic_content_hash",
    ),
    _record_spec(
        "action_adaptation_receipt",
        "receipt_id", "request_id", "request_hash",
        "capability_assessment_id", "capability_assessment_hash",
        "receiver_id", "processing_started_at",
        "processing_completed_at", "clock_domain_id", "accepted",
        "rejection_code", "adapted_candidate_id",
        "adapted_candidate_hash", "semantic_content_hash",
    ),
    _record_spec(
        "target_execution_request",
        "request_id", "control_domain_id", "target_handler_id",
        "authoritative_decision_id", "authoritative_decision_hash",
        "adaptation_request_id", "adaptation_request_hash",
        "adaptation_receipt_id", "adaptation_receipt_hash",
        "adapted_candidate_id", "adapted_candidate_hash",
        "component_target_capability_id",
        "component_target_capability_hash", "component_coverage",
        "target_effective_at", "supported_until", "authority_sequence",
        "execution_gate_sequence", "execution_fence_sequence",
        "submitted_at", "clock_domain_id", "semantic_content_hash",
    ),
    _record_spec(
        "target_execution_submission",
        "receipt_id", "target_execution_request_id",
        "target_execution_request_hash", "control_domain_id",
        "target_handler_id", "received_at", "clock_domain_id",
        "accepted_for_processing", "rejection_code", "processing_token",
        "semantic_content_hash",
    ),
    _record_spec(
        "target_execution_completion",
        "receipt_id", "target_execution_request_id",
        "target_execution_request_hash", "submission_receipt_id",
        "submission_receipt_hash", "processing_token",
        "control_domain_id", "target_handler_id", "status",
        "completed_at", "available_at", "clock_domain_id",
        "target_id", "target_hash", "outcome_code",
        "semantic_content_hash",
    ),
    _record_spec(
        "execution_directive",
        "directive_id", "decision_id", "publication_attempt_id",
        "arbitration_result_id", "global_status",
        "directive_by_selection_group", "persistence_policy_hash",
        "publication_fallback",
        "action_adaptation_receipt_id", "action_adaptation_receipt_hash",
        "action_adaptation_fallback",
        "requires_execution_epoch_increment", "requires_backend_hold",
        "fault_latched", "execution_epoch", "semantic_content_hash",
    ),
    _record_spec(
        "shared_execution_evaluation",
        "evaluation_id", "source_arbitration_evaluation_id",
        "arbitration_result_id", "output_status", "non_authoritative",
        "proposed_directive", "proposed_directive_hash", "safety_result",
        "safety_result_hash", "command_intent", "command_intent_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "hardware_preflight_authorization",
        "authorization_id", "operator_authorization_id", "profile_id",
        "execution_manifest_id", "execution_manifest_hash",
        "dispatch_inventory_id", "dispatch_inventory_hash",
        "worker_drain_plan_id", "worker_drain_plan_hash",
        "control_domain_id", "embodiment_instance_id", "allowed_sides",
        "required_participant_ids", "allowed_stop_seam_id",
        "allowed_stop_sdk_method", "issued_at", "expires_at",
        "motion_authorized", "lifecycle_authorized",
        "semantic_content_hash",
    ),
    _record_spec(
        "execution_cancellation",
        "cancellation_id", "reason", "requested_scope", "effective_scope",
        "selection_group_ids", "scope_escalation_reason",
        "previous_execution_epoch", "execution_epoch",
        "authority_epoch_evidence", "segment_generation_before",
        "segment_generation_after", "invalidated_decision_ids",
        "invalidated_reference_segment_ids", "invalidated_queued_command_ids",
        "source_event_ids", "effective_at", "effective_clock_domain_id",
        "acknowledgement_deadline_at", "required_ack_levels",
        "requests_backend_hold", "fault_latched", "affects_decision_rate",
        "semantic_content_hash",
    ),
    _record_spec(
        "execution_cancellation_ack",
        "acknowledgement_id", "cancellation_id", "acknowledgement_level",
        "acknowledged_at", "clock_domain_id", "source_id", "accepted",
        "status", "semantic_content_hash",
    ),
    _record_spec(
        "safety_control_operation",
        "operation_id", "operation_kind", "runtime_session_id",
        "control_domain_id", "safety_source", "source_event_id",
        "target_execution_epoch", "affected_selection_groups",
        "frozen_stop_target_hash", "allowed_dispatch_operation",
        "maximum_duration_sec", "maximum_tick_count", "authorized_at",
        "semantic_content_hash",
    ),
    _record_spec(
        "safety_control_tick",
        "tick_evidence_id", "operation_id", "operation_hash",
        "tick_sequence", "target_execution_epoch", "command_lineage_id",
        "admitted_at", "control_admission_sequence", "work_item_id",
        "work_kind", "authority_epoch", "segment_generation_by_group",
        "source_safety_operation_id", "source_safety_tick_evidence_id",
        "semantic_content_hash",
    ),
    _record_spec(
        "execution_worker_drain",
        "receipt_id", "cancellation_id", "participant_id",
        "participant_spec_hash", "invalidated_execution_epoch",
        "observed_execution_epoch", "affected_selection_group_ids",
        "discarded_work_ids", "last_old_epoch_invocation_started_at",
        "last_old_epoch_invocation_completed_at",
        "in_flight_old_epoch_count", "no_future_old_epoch_calls",
        "acknowledged_at", "accepted", "status", "semantic_content_hash",
    ),
    _record_spec(
        "execution_worker_drain_aggregate",
        "aggregate_id", "cancellation_id", "required_participant_ids",
        "accepted_participant_ids", "missing_participant_ids",
        "rejected_participant_ids", "receipt_refs", "aggregated_at",
        "complete", "status", "semantic_content_hash",
    ),
    _record_spec(
        "execution_fault_escalation",
        "escalation_id", "cancellation_id", "runtime_session_id",
        "control_domain_id", "execution_epoch", "authority_epoch_evidence",
        "reason", "requested_gate_status", "missing_ack_levels",
        "rejected_ack_levels", "observed_at", "semantic_content_hash",
    ),
    _record_spec(
        "shared_safety_result",
        "safety_result_id", "decision_id", "directive_id", "input_action_id",
        "input_action_hash", "safe_action", "safe_action_hash",
        "modified_components", "reasons", "status", "safety_contract_id",
        "safety_contract_hash", "anchor_id", "anchor_kind", "anchor_hash",
        "details", "semantic_content_hash",
    ),
    _record_spec(
        "resolved_execution_target",
        "target_id", "control_domain_id", "selection_group_id",
        "execution_epoch", "group_segment_generation",
        "source_safety_result_id", "source_safe_action_hash",
        "group_safe_action", "group_safe_action_hash", "target_kind",
        "anchor_commit_policy", "component_dependencies", "resolved_at",
        "scheduled_from", "scheduled_until", "clock_domain_id",
        "input_action_schema_hash", "output_target_schema_hash",
        "resolver_contract_id", "resolver_contract_hash",
        "transform_contract_hash", "planner_contract_hash", "payload",
        "payload_hash", "semantic_content_hash",
    ),
    _record_spec(
        "execution_target_admission",
        "admission_id", "target_id", "target_hash", "control_domain_id",
        "selection_group_id", "execution_epoch", "group_segment_generation",
        "receiver_id", "admitted_at", "clock_domain_id", "accepted",
        "rejection_reason", "semantic_content_hash",
    ),
    _record_spec(
        "component_execution_target",
        "target_id", "target_execution_request_id",
        "target_execution_request_hash", "submission_receipt_id",
        "submission_receipt_hash", "authoritative_decision_id",
        "authoritative_decision_hash", "adaptation_request_id",
        "adaptation_request_hash", "adaptation_receipt_id",
        "adaptation_receipt_hash", "adapted_candidate_id",
        "adapted_candidate_hash", "capability_assessment_id",
        "capability_assessment_hash", "component_target_capability_id",
        "component_target_capability_hash", "control_domain_id",
        "target_handler_id", "component_dependencies",
        "input_action_schema_hash", "output_target_schema_hash",
        "target_kind", "target_effective_at", "supported_until",
        "authority_sequence",
        "execution_gate_sequence", "execution_fence_sequence",
        "resolved_at", "temporal_semantics", "scheduled_for",
        "scheduled_until", "clock_domain_id", "payload",
        "processing_fact_references", "payload_hash",
        "semantic_content_hash",
    ),
    _record_spec(
        "component_execution_target_admission",
        "receipt_id", "target_id", "target_hash",
        "target_execution_request_id", "target_execution_request_hash",
        "control_domain_id", "target_handler_id", "receiver_id",
        "execution_fence_sequence", "admission_sequence",
        "admitted_at", "available_at", "clock_domain_id", "accepted",
        "rejection_code", "admission_token", "semantic_content_hash",
    ),
    _record_spec(
        "component_execution_target_activation",
        "activation_id", "target_id", "target_hash",
        "target_execution_request_id", "target_execution_request_hash",
        "target_admission_receipt_id", "target_admission_receipt_hash",
        "control_domain_id", "target_handler_id", "receiver_id",
        "activation_sequence", "activation_status", "activation_at",
        "available_at", "clock_domain_id", "effective_from",
        "effective_until", "rejection_code", "semantic_content_hash",
    ),
    _record_spec(
        "safety_anchor_commit_evidence",
        "evidence_id", "evidence_kind", "target_id", "safety_result_id",
        "safe_action_hash", "control_domain_id", "selection_group_id",
        "execution_epoch", "group_segment_generation", "occurred_at",
        "accepted", "effective_from", "rejection_reason",
        "semantic_content_hash",
    ),
    _record_spec(
        "reference_receiver_validation",
        "validation_id", "dispatch_context_hash", "receiver_state_hash",
        "accepted", "status", "checked_at", "semantic_content_hash",
    ),
    _record_spec(
        "reference_segment",
        "reference_segment_id", "control_domain_id", "selection_group_id",
        "execution_epoch", "group_segment_generation", "authority_kind",
        "endpoint_parents", "component_dependencies", "created_at",
        "scheduled_from", "scheduled_until", "clock_domain_id",
        "interpolation_contract_hash", "resolver_contract_hash",
        "planner_contract_hash", "input_action_schema_hash",
        "output_reference_schema_hash", "required_admission_guarantee",
        "segment_safety_evidence_level", "semantic_content_hash",
    ),
    _record_spec(
        "reference_segment_admission",
        "admission_id", "reference_segment_id", "reference_segment_hash",
        "control_domain_id", "selection_group_id", "execution_epoch",
        "group_segment_generation", "receiver_id", "admitted_at",
        "admission_sequence", "clock_domain_id",
        "achieved_admission_guarantee", "supersedes_segment_ids",
        "accepted", "rejection_reason", "receiver_validation_id",
        "receiver_validation_hash", "semantic_content_hash",
    ),
    _record_spec(
        "group_generation_reservation",
        "reservation_id", "control_domain_id", "selection_group_id",
        "execution_epoch", "active_generation_at_reservation",
        "reserved_generation", "source_segment_id", "source_segment_hash",
        "admission_id", "admission_hash", "relation_kind",
        "predecessor_segment_id", "supersedes_reservation_id",
        "effective_from", "admitted_at", "admission_sequence",
        "clock_domain_id", "semantic_content_hash",
    ),
    _record_spec(
        "group_generation_activation",
        "activation_id", "reservation_id", "source_segment_id",
        "control_domain_id", "selection_group_id", "execution_epoch",
        "previous_active_generation", "active_generation", "effective_at",
        "activated_at", "activation_sequence", "clock_domain_id",
        "semantic_content_hash",
    ),
    _record_spec(
        "safety_anchor_commit_candidate",
        "candidate_id", "reference_segment_id", "reference_segment_hash",
        "admission_id", "admission_hash", "admission_sequence",
        "control_domain_id", "selection_group_id", "execution_epoch",
        "group_segment_generation", "authority_kind", "safety_result_id",
        "safe_action", "safe_action_hash", "evidence_created_at",
        "effective_from", "required_admission_guarantee",
        "achieved_admission_guarantee", "semantic_content_hash",
    ),
    _record_spec(
        "reference_target_activation",
        "observation_id", "reference_segment_id", "generation_activation_id",
        "reference_sequence",
        "execution_epoch", "group_segment_generation", "scheduled_for",
        "activated_at_worker_time", "observed_at", "qref_by_side",
        "activation_status_by_side", "semantic_content_hash",
    ),
    _record_spec(
        "planner_command_frame",
        "observation_id", "reference_segment_id", "reference_sequence",
        "command_sequence", "execution_epoch", "group_segment_generation",
        "scheduled_for", "observed_at", "planner_proposal_token_by_side",
        "planner_generation_by_side", "qref_by_side",
        "planner_qcmd_by_side", "follow_qcmd_by_side",
        "accepted_qcmd_by_side", "limited_by_side", "semantic_content_hash",
    ),
    _record_spec(
        "reference_segment_terminal",
        "terminal_id", "reference_segment_id", "reference_segment_hash",
        "control_domain_id", "selection_group_id", "execution_epoch",
        "group_segment_generation", "status", "available_at",
        "terminal_admission_sequence", "effective_at", "clock_domain_id",
        "last_reference_frame_id", "replacement_segment_id", "evidence_ids",
        "semantic_content_hash",
    ),
    _record_spec(
        "command",
        "command_id", "decision_id", "directive_id", "safety_result_id",
        "command_schema_id", "commands", "command_hash", "scheduled_for",
        "created_at", "authority_epoch", "execution_epoch",
        "semantic_content_hash",
    ),
    _record_spec(
        "dispatch_owner",
        "dispatch_event_id", "event_kind", "permit_kind", "operation",
        "evidence_event_id", "token_id", "owner_runtime_id",
        "owner_kind", "owner_generation", "permit_id", "permit_hash",
        "validation_receipt_id", "validation_accepted", "validation_status",
        "authority_epoch", "execution_epoch", "checked_at",
        "semantic_content_hash",
    ),
    _record_spec(
        "dispatch_receipt",
        "dispatch_receipt_id", "command_id", "dispatch_event_id",
        "backend_name", "receipts_by_effector", "all_accepted", "dispatched_at",
        "semantic_content_hash",
    ),
    _record_spec(
        "measurement_alignment",
        "measurement_alignment_id", "decision_id", "command_id",
        "dispatch_receipt_id", "measurement_schema_id", "measured_at",
        "measurement", "semantic_content_hash",
    ),
    _record_spec(
        "arbiter_shadow_status",
        "status_event_id", "status", "reason", "emission_id", "occurred_at",
        "non_authoritative", "semantic_content_hash",
    ),
    _record_spec(
        "component_mixing_shadow",
        "shadow_id", "status", "reason", "non_authoritative",
        "arm_decision_id", "arm_decision_hash", "arm_strategy_id",
        "arm_strategy_spec_hash", "arm_input_snapshot_id",
        "arm_input_snapshot_hash", "hand_decision_id_by_side",
        "hand_decision_hash_by_side", "hand_strategy_id_by_side",
        "hand_strategy_spec_hash_by_side", "hand_input_snapshot_id_by_side",
        "hand_input_snapshot_hash_by_side", "component_decision_hash_by_id",
        "output_action", "output_action_hash", "evaluated_at",
        "semantic_content_hash",
    ),
    _record_spec(
        "trajectory_composition_shadow_result",
        "result_id", "editing_state_hash", "interaction_mode",
        "base_lifecycle_status", "preparation_hash",
        "target_effective_at", "publication_basis_hash", "output_status",
        "reason", "nominal_action_hash", "corrected_action_hash",
        "residual_sample_hash", "residual_weight",
        "active_base_binding_id", "active_base_binding_hash",
        "shadow_trajectory_id", "shadow_trajectory_hash",
        "shadow_keyframe_hash", "shadow_action_hash",
        "composition_evaluation_id", "composition_evaluation_hash",
        "source_difference_observation_id",
        "source_difference_observation_hash", "proposal_admitted",
        "authoritative", "executable_preview_status",
        "semantic_content_hash",
    ),
    _record_spec(
        "legacy_execution_observation",
        "event_id", "owner_kind", "non_authoritative_shadow_reference",
        "emission_id", "evaluation_id", "legacy_safe_representation_trace_id",
        "command_present", "execution_accepted", "execution_mode",
        "dispatch_attempted", "dispatch_completed", "dispatch_error_type",
        "component_parity", "observed_at", "semantic_content_hash",
    ),
    _record_spec(
        "episode_membership_transition",
        "transition_id", "transition", "episode_uuid", "lifecycle_owner_id",
        "membership_epoch", "occurred_at", "previous_episode_uuid", "reason",
    ),
    _record_spec(
        "runtime_trace_terminal",
        "terminal_status", "reason", "occurred_at",
    ),
)

CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1 = MappingProxyType(
    {spec.event_type: spec for spec in CONTROL_TRACE_RECORD_SPECS_V1}
)


def _required_text(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("ControlTrace payload rejects object arrays.")
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("ControlTrace payload mappings require string keys.")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    raise TypeError(
        "ControlTrace payload accepts only canonical primitive values; "
        f"got {type(value).__name__}."
    )


def _validate_sha256(value: object, *, field_name: str) -> str:
    normalized = _required_text(value, field_name=field_name)
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")
    return normalized


@dataclass(frozen=True)
class ControlTraceFact:
    """A transport-neutral immutable fact admitted to a trace journal."""

    fact_id: str
    event_type: str
    event_family: str
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str | None
    occurred_at: float
    clock_domain_id: str
    record_schema_id: str
    record_schema_version: int
    payload: Mapping[str, Any]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "fact_id",
            "event_type",
            "event_family",
            "runtime_session_id",
            "control_domain_id",
            "clock_domain_id",
            "record_schema_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), field_name=name))
        spec = CONTROL_TRACE_EVENT_SPEC_BY_TYPE_V24.get(self.event_type)
        if spec is None:
            raise ValueError(f"Unknown ControlTrace event type: {self.event_type!r}.")
        if self.event_family != spec.event_family:
            raise ValueError(
                f"Event {self.event_type!r} belongs to family {spec.event_family!r}, "
                f"not {self.event_family!r}."
            )
        record_spec = CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1.get(self.event_type)
        if record_spec is None:
            raise ValueError(
                f"ControlTrace event {self.event_type!r} has no implemented explicit record schema."
            )
        if (
            self.record_schema_id != record_spec.schema_id
            or int(self.record_schema_version) != record_spec.schema_version
        ):
            raise ValueError(f"ControlTrace record schema mismatch for {self.event_type!r}.")
        object.__setattr__(self, "record_schema_version", int(self.record_schema_version))
        if self.embodiment_instance_id is not None:
            object.__setattr__(
                self,
                "embodiment_instance_id",
                _required_text(self.embodiment_instance_id, field_name="embodiment_instance_id"),
            )
        object.__setattr__(self, "occurred_at", _finite(self.occurred_at, field_name="occurred_at"))
        raw_payload = dict(self.payload)
        record_spec.validate(raw_payload)
        object.__setattr__(self, "payload", _freeze(raw_payload))
        object.__setattr__(
            self,
            "semantic_content_hash",
            _validate_sha256(self.semantic_content_hash, field_name="semantic_content_hash"),
        )

    @classmethod
    def create(
        cls,
        *,
        fact_id: str,
        event_type: str,
        runtime_session_id: str,
        control_domain_id: str,
        occurred_at: float,
        clock_domain_id: str,
        payload: Mapping[str, Any],
        embodiment_instance_id: str | None = None,
    ) -> "ControlTraceFact":
        spec = CONTROL_TRACE_EVENT_SPEC_BY_TYPE_V24.get(str(event_type))
        if spec is None:
            raise ValueError(f"Unknown ControlTrace event type: {event_type!r}.")
        record_spec = CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1.get(spec.event_type)
        if record_spec is None:
            raise ValueError(
                f"ControlTrace event {spec.event_type!r} has no implemented explicit record schema."
            )
        record_spec.validate(payload)
        # Hash the exact immutable representation that the fact stores and the
        # WAL later serializes.  Hashing a mutable list and freezing it to a
        # tuple afterwards makes an otherwise valid record fail round-trip.
        frozen_payload = _freeze(dict(payload))
        semantic_payload = {
            "fact_id": str(fact_id),
            "event_type": spec.event_type,
            "event_family": spec.event_family,
            "runtime_session_id": str(runtime_session_id),
            "control_domain_id": str(control_domain_id),
            "embodiment_instance_id": embodiment_instance_id,
            "occurred_at": float(occurred_at),
            "clock_domain_id": str(clock_domain_id),
            "record_schema_id": record_spec.schema_id,
            "record_schema_version": record_spec.schema_version,
            "payload": frozen_payload,
        }
        return cls(
            **semantic_payload,
            semantic_content_hash=canonical_content_hash(
                semantic_payload,
                domain=CONTROL_TRACE_FACT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class EpisodeMembershipRef:
    episode_uuid: str
    lifecycle_owner_id: str
    binding_event_id: str
    membership_epoch: int
    bound_at: float

    def __post_init__(self) -> None:
        for name in ("episode_uuid", "lifecycle_owner_id", "binding_event_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), field_name=name))
        if int(self.membership_epoch) < 0:
            raise ValueError("membership_epoch must be non-negative.")
        object.__setattr__(self, "membership_epoch", int(self.membership_epoch))
        object.__setattr__(self, "bound_at", _finite(self.bound_at, field_name="bound_at"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_uuid": self.episode_uuid,
            "lifecycle_owner_id": self.lifecycle_owner_id,
            "binding_event_id": self.binding_event_id,
            "membership_epoch": self.membership_epoch,
            "bound_at": self.bound_at,
        }


@dataclass(frozen=True)
class PersistedRecordEnvelope:
    """Logical persistence envelope; M4A keeps it in memory only."""

    schema: str
    trace_session_id: str
    journal_sequence: int
    episode_uuid: str | None
    episode_membership: EpisodeMembershipRef | None
    fact: ControlTraceFact
    encoded_record: Mapping[str, Any]
    schema_refs: tuple[str, ...]
    registry_refs: tuple[str, ...]
    record_integrity_hash: str

    def __post_init__(self) -> None:
        if self.schema != CONTROL_TRACE_SCHEMA_V24:
            raise ValueError(f"Unsupported ControlTrace schema: {self.schema!r}.")
        object.__setattr__(self, "trace_session_id", _required_text(self.trace_session_id, field_name="trace_session_id"))
        if int(self.journal_sequence) <= 0:
            raise ValueError("journal_sequence must be positive.")
        object.__setattr__(self, "journal_sequence", int(self.journal_sequence))
        if self.episode_uuid is not None:
            object.__setattr__(self, "episode_uuid", _required_text(self.episode_uuid, field_name="episode_uuid"))
        if self.episode_membership is not None:
            if not isinstance(self.episode_membership, EpisodeMembershipRef):
                raise TypeError("episode_membership must be an EpisodeMembershipRef.")
            if self.episode_uuid != self.episode_membership.episode_uuid:
                raise ValueError("episode_uuid and episode_membership disagree.")
        object.__setattr__(self, "encoded_record", _freeze(dict(self.encoded_record)))
        object.__setattr__(self, "schema_refs", tuple(str(value) for value in self.schema_refs))
        object.__setattr__(self, "registry_refs", tuple(str(value) for value in self.registry_refs))
        object.__setattr__(
            self,
            "record_integrity_hash",
            _validate_sha256(self.record_integrity_hash, field_name="record_integrity_hash"),
        )

    @classmethod
    def wrap(
        cls,
        *,
        trace_session_id: str,
        journal_sequence: int,
        fact: ControlTraceFact,
        episode_membership: EpisodeMembershipRef | None,
        schema_refs: tuple[str, ...] = (),
        registry_refs: tuple[str, ...] = (),
    ) -> "PersistedRecordEnvelope":
        episode_uuid = (
            None if episode_membership is None else episode_membership.episode_uuid
        )
        encoded_record = {
            "fact_id": fact.fact_id,
            "event_type": fact.event_type,
            "event_family": fact.event_family,
            "runtime_session_id": fact.runtime_session_id,
            "control_domain_id": fact.control_domain_id,
            "embodiment_instance_id": fact.embodiment_instance_id,
            "occurred_at": fact.occurred_at,
            "clock_domain_id": fact.clock_domain_id,
            "record_schema_id": fact.record_schema_id,
            "record_schema_version": fact.record_schema_version,
            "payload": fact.payload,
            "semantic_content_hash": fact.semantic_content_hash,
        }
        payload = {
            "schema": CONTROL_TRACE_SCHEMA_V24,
            "trace_session_id": str(trace_session_id),
            "journal_sequence": int(journal_sequence),
            "episode_uuid": episode_uuid,
            "episode_membership": (
                None if episode_membership is None else episode_membership.to_payload()
            ),
            "encoded_record": encoded_record,
            "schema_refs": tuple(schema_refs),
            "registry_refs": tuple(registry_refs),
        }
        return cls(
            schema=CONTROL_TRACE_SCHEMA_V24,
            trace_session_id=str(trace_session_id),
            journal_sequence=int(journal_sequence),
            episode_uuid=episode_uuid,
            episode_membership=episode_membership,
            fact=fact,
            encoded_record=encoded_record,
            schema_refs=tuple(schema_refs),
            registry_refs=tuple(registry_refs),
            record_integrity_hash=canonical_content_hash(
                payload,
                domain=PERSISTED_RECORD_ENVELOPE_HASH_DOMAIN_V1,
            ),
        )

    def to_storage_payload(self) -> dict[str, Any]:
        """Return the versioned logical record consumed by the M4B WAL codec."""

        return {
            "schema": self.schema,
            "trace_session_id": self.trace_session_id,
            "journal_sequence": self.journal_sequence,
            "episode_uuid": self.episode_uuid,
            "episode_membership": (
                None
                if self.episode_membership is None
                else self.episode_membership.to_payload()
            ),
            "encoded_record": self.encoded_record,
            "schema_refs": self.schema_refs,
            "registry_refs": self.registry_refs,
            "record_integrity_hash": self.record_integrity_hash,
        }

    @classmethod
    def from_storage_payload(cls, payload: Mapping[str, Any]) -> "PersistedRecordEnvelope":
        expected_fields = {
            "schema",
            "trace_session_id",
            "journal_sequence",
            "episode_uuid",
            "episode_membership",
            "encoded_record",
            "schema_refs",
            "registry_refs",
            "record_integrity_hash",
        }
        if set(payload) != expected_fields:
            raise ValueError("ControlTrace storage envelope fields are invalid.")
        if payload["schema"] != CONTROL_TRACE_SCHEMA_V24:
            raise ValueError("ControlTrace storage envelope schema is unsupported.")
        encoded = dict(payload["encoded_record"])
        expected_record_fields = {
            "fact_id",
            "event_type",
            "event_family",
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "occurred_at",
            "clock_domain_id",
            "record_schema_id",
            "record_schema_version",
            "payload",
            "semantic_content_hash",
        }
        if set(encoded) != expected_record_fields:
            raise ValueError("ControlTrace encoded record fields are invalid.")
        fact = ControlTraceFact.create(
            fact_id=encoded["fact_id"],
            event_type=encoded["event_type"],
            runtime_session_id=encoded["runtime_session_id"],
            control_domain_id=encoded["control_domain_id"],
            embodiment_instance_id=encoded["embodiment_instance_id"],
            occurred_at=encoded["occurred_at"],
            clock_domain_id=encoded["clock_domain_id"],
            payload=encoded["payload"],
        )
        if encoded["event_family"] != fact.event_family:
            raise ValueError("ControlTrace encoded event family is inconsistent.")
        if (
            encoded["record_schema_id"] != fact.record_schema_id
            or int(encoded["record_schema_version"]) != fact.record_schema_version
        ):
            raise ValueError("ControlTrace encoded record schema is inconsistent.")
        if encoded["semantic_content_hash"] != fact.semantic_content_hash:
            raise ValueError("ControlTrace fact semantic hash mismatch.")
        membership_payload = payload["episode_membership"]
        membership = (
            None
            if membership_payload is None
            else EpisodeMembershipRef(**dict(membership_payload))
        )
        rebuilt = cls.wrap(
            trace_session_id=payload["trace_session_id"],
            journal_sequence=int(payload["journal_sequence"]),
            fact=fact,
            episode_membership=membership,
            schema_refs=tuple(payload["schema_refs"]),
            registry_refs=tuple(payload["registry_refs"]),
        )
        if payload["episode_uuid"] != rebuilt.episode_uuid:
            raise ValueError("ControlTrace episode membership payload is inconsistent.")
        if payload["record_integrity_hash"] != rebuilt.record_integrity_hash:
            raise ValueError("ControlTrace record integrity hash mismatch.")
        return rebuilt


__all__ = [
    "CONTROL_TRACE_EVENT_FAMILIES_V24",
    "CONTROL_TRACE_EVENT_SPECS_V24",
    "CONTROL_TRACE_EVENT_SPEC_BY_TYPE_V24",
    "CONTROL_TRACE_EVENT_TYPES_V24",
    "CONTROL_TRACE_RECORD_SPECS_V1",
    "CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1",
    "CONTROL_TRACE_FACT_HASH_DOMAIN_V1",
    "CONTROL_TRACE_SCHEMA_V24",
    "PERSISTED_RECORD_ENVELOPE_HASH_DOMAIN_V1",
    "ControlTraceEventSpec",
    "ControlTraceFact",
    "ControlTraceRecordSpec",
    "EpisodeMembershipRef",
    "PersistedRecordEnvelope",
]

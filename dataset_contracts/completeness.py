"""M4A ControlTrace requirement profiles and in-memory completeness reports."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .control_trace import CONTROL_TRACE_EVENT_TYPES_V24, ControlTraceFact
from .control_trace_validation.arbitration_shadow import (
    ArbitrationShadowValidationIndex,
    validate_arbitration_shadow_lineage,
)
from .control_trace_validation.action_adaptation import (
    ActionAdaptationValidationIndex,
    validate_action_adaptation_lineage,
)
from .control_trace_validation.component_authority import (
    ComponentAuthorityValidationIndex,
    validate_component_authority_lineage,
)
from .control_trace_validation.composition_session import (
    CompositionSessionValidationIndex,
    validate_composition_session_lineage,
)
from .control_trace_validation.composition_evaluation import (
    CompositionEvaluationValidationIndex,
    validate_composition_evaluation_lineage,
)
from .control_trace_validation.context import ControlTraceValidationContext
from .control_trace_validation.execution_directive import (
    ExecutionDirectiveValidationIndex,
    validate_execution_directive_lineage,
)
from .control_trace_validation.execution_cancellation import (
    ExecutionCancellationValidationIndex,
    validate_execution_cancellation_lineage,
)
from .control_trace_validation.issue_sink import (
    ControlTraceLineageIssueSink,
)
from .control_trace_validation.reference_lifecycle import (
    ReferenceLifecycleValidationIndex,
    validate_reference_lifecycle_lineage,
)
from .control_trace_validation.replay_lineage import (
    ReplayLineageValidationIndex,
    validate_replay_lineage,
)
from .control_trace_validation.runner import run_control_trace_foundation
from .control_trace_validation.source_proposal import (
    SourceProposalValidationIndex,
    validate_source_proposal_lineage,
)
from .control_trace_validation.target_execution import (
    TargetExecutionValidationIndex,
    validate_target_execution_lineage,
)


WRITER_FAILURE_BEHAVIORS_V1 = (
    "mark_uncommitted_continue",
    "request_runtime_hold",
)


@dataclass(frozen=True)
class TraceRequirementProfile:
    profile_id: str
    runtime_mode: str
    required_event_types: frozenset[str]
    optional_event_types: frozenset[str]
    not_applicable_event_types: frozenset[str]
    required_streams: tuple[str, ...] = ()
    receipt_policy: str = "optional"
    measurement_policy: str = "optional"
    writer_failure_behavior: str = "mark_uncommitted_continue"
    durable_before_capture_commit: bool = True
    lineage_policy_id: str = "generic_reference_integrity_v1"
    profile_scope: str = "runtime_diagnostic"
    implementation_status: str = "fixture_only"

    def __post_init__(self) -> None:
        for name in ("profile_id", "runtime_mode", "lineage_policy_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        required = frozenset(self.required_event_types)
        optional = frozenset(self.optional_event_types)
        not_applicable = frozenset(self.not_applicable_event_types)
        if required & optional or required & not_applicable or optional & not_applicable:
            raise ValueError("Trace profile event sets must be pairwise disjoint.")
        covered = required | optional | not_applicable
        unknown = covered - CONTROL_TRACE_EVENT_TYPES_V24
        if unknown:
            raise ValueError(f"Trace profile contains unknown event types: {sorted(unknown)!r}.")
        missing = CONTROL_TRACE_EVENT_TYPES_V24 - covered
        if missing:
            raise ValueError(f"Trace profile must classify every event type: {sorted(missing)!r}.")
        if self.receipt_policy not in {
            "required",
            "optional",
            "conditional_on_dispatch",
            "not_applicable",
        }:
            raise ValueError("Unsupported receipt_policy.")
        if self.measurement_policy not in {
            "required",
            "optional",
            "conditional_on_execution",
            "not_applicable",
        }:
            raise ValueError("Unsupported measurement_policy.")
        if self.writer_failure_behavior not in WRITER_FAILURE_BEHAVIORS_V1:
            raise ValueError("Unsupported writer_failure_behavior.")
        if self.profile_scope not in {"runtime_diagnostic", "formal_capture"}:
            raise ValueError("Unsupported trace profile_scope.")
        if self.implementation_status not in {"fixture_only", "runtime_integrated", "planned"}:
            raise ValueError("Unsupported trace implementation_status.")
        object.__setattr__(self, "required_event_types", required)
        object.__setattr__(self, "optional_event_types", optional)
        object.__setattr__(self, "not_applicable_event_types", not_applicable)
        object.__setattr__(self, "required_streams", tuple(str(value) for value in self.required_streams))
        receipt_bucket = (
            "required"
            if "dispatch_receipt" in required
            else "not_applicable"
            if "dispatch_receipt" in not_applicable
            else "optional"
        )
        measurement_bucket = (
            "required"
            if "measurement_alignment" in required
            else "not_applicable"
            if "measurement_alignment" in not_applicable
            else "optional"
        )
        if self.receipt_policy not in {receipt_bucket, "conditional_on_dispatch"}:
            raise ValueError("receipt_policy must match dispatch_receipt classification.")
        if self.measurement_policy not in {
            measurement_bucket,
            "conditional_on_execution",
        }:
            raise ValueError(
                "measurement_policy must match measurement_alignment classification."
            )


_POLICY_SHADOW_REQUIRED = frozenset(
    {"producer_epoch_activation", "runtime_trace_terminal"}
)
_COMPONENT_MIXING_SHADOW_REQUIRED = frozenset(
    {"component_mixing_shadow", "runtime_trace_terminal"}
)
_REPLAY_SHADOW_REQUIRED = _POLICY_SHADOW_REQUIRED | {
    "replay_action_record",
    "replay_source_action_contract",
    "target_runtime_action_contract",
    "replay_binding_assessment",
    "candidate_binding_eligibility",
    "replay_source_adapter_receipt",
}
_FORMAL_HUMAN_SHADOW_REQUIRED = _POLICY_SHADOW_REQUIRED | {
    "human_command_emission",
    "source_adapter_receipt",
    "source_validation",
    "legacy_execution_observation",
}
_FORMAL_POLICY_SHADOW_REQUIRED = _POLICY_SHADOW_REQUIRED | {
    "policy_action_emission",
    "source_validation",
    "legacy_execution_observation",
}
_FORMAL_REPLAY_SHADOW_REQUIRED = _REPLAY_SHADOW_REQUIRED | {
    "legacy_execution_observation",
}
_SHADOW_NOT_APPLICABLE = frozenset(
    {
        "arbitration_publication_attempt",
        "arbitration_decision",
        "arbitration_publication_rejected",
        "action_adaptation_request",
        "target_capability_assessment",
        "adapted_action_candidate",
        "action_adaptation_receipt",
        "execution_directive",
        "dispatch_owner",
    }
)
_DRY_RUN_NOT_APPLICABLE = _SHADOW_NOT_APPLICABLE | {
    "dispatch_receipt",
    "measurement_alignment",
}

_AUTHORITATIVE_REQUIRED = frozenset({"runtime_trace_terminal"})
_AUTHORITATIVE_SIM_CORE_REQUIRED = frozenset(
    {
        "producer_epoch_activation",
        "proposal_admission",
        "action_proposal",
        "proposal_snapshot",
        "decision_schedule_tick",
        "authority_state_snapshot",
        "execution_gate_snapshot",
        "arbitration_input_snapshot",
        "arbitration_computation",
        "arbitration_publication_attempt",
        "arbitration_decision",
        "action_adaptation_request",
        "target_capability_assessment",
        "adapted_action_candidate",
        "action_adaptation_receipt",
        "execution_directive",
        "shared_safety_result",
        "command",
        "dispatch_owner",
        "dispatch_receipt",
        "measurement_alignment",
        "runtime_trace_terminal",
    }
)
_EXECUTION_MANIFEST_REQUIRED = frozenset({"execution_manifest"})
_STATION_LIFECYCLE_REQUIRED = frozenset(
    {
        "station_run_manifest",
        "station_intent_transition",
        "station_operation_transition",
        "station_publication_barrier_transition",
    }
)
_COMPOSITION_SESSION_REQUIRED = frozenset(
    {
        "composition_session_spec",
        "composition_session_transition",
        "composition_spec",
        "composition_input_admission",
        "composition_contributor_input",
        "composition_input_snapshot",
        "composition_evaluation",
        "composition_authority_disposition",
    }
)
_AUTHORITATIVE_REPLAY_SOURCE_REQUIRED = frozenset(
    {
        "replay_source_frame_emission",
        "replay_action_record",
        "replay_source_action_contract",
        "target_runtime_action_contract",
        "replay_binding_assessment",
        "candidate_binding_eligibility",
        "replay_source_adapter_receipt",
        "replay_execution_authorization",
        "candidate_authorization_admission",
        "candidate_authorization_snapshot",
    }
)
_AUTHORITATIVE_HUMAN_SOURCE_REQUIRED = frozenset(
    {
        "human_command_emission",
        "source_adapter_receipt",
        "source_validation",
    }
)
_AUTHORITATIVE_POSE_HUMAN_REPLAY_SOURCE_REQUIRED = (
    _AUTHORITATIVE_HUMAN_SOURCE_REQUIRED
    | _AUTHORITATIVE_REPLAY_SOURCE_REQUIRED
)
_AUTHORITATIVE_POLICY_SOURCE_REQUIRED = frozenset(
    {
        "policy_action_emission",
        "source_validation",
    }
)
_DIRECT_COMPONENT_COMPOSITION_REQUIRED = frozenset(
    {
        "composition_activation_scope",
        "composition_activation_binding",
        "composition_input_admission",
        "action_payload_schema",
        "component_action_schema",
        "composition_spec",
        "composition_temporal_selection_spec",
        "component_composition_rule_set",
        "formal_component_composition_binding",
        "composition_contributor_input",
        "composition_input_snapshot",
        "formal_composition_strategy_spec",
        "formal_composition_strategy_input",
        "composition_evaluation",
        "formal_composition_strategy_result_binding",
        "composition_authority_disposition",
        "conservative_compound_authority_profile",
        "source_difference_spec",
        "source_difference_observation",
    }
)
_DIRECT_COMPONENT_HOT_COMPOSITION_REQUIRED = frozenset(
    (
        _DIRECT_COMPONENT_COMPOSITION_REQUIRED
        - {"formal_composition_strategy_input"}
    )
    | {
        "composition_dynamic_parameter_contract",
        "composition_strategy_revision",
        "composition_dynamic_parameter_snapshot",
        "composition_strategy_control_binding",
        "hot_composition_strategy_input",
    }
)
_IMMEDIATE_COMPONENT_EXECUTION_EVENT_TYPES = frozenset(
    {
        "target_execution_request",
        "target_execution_submission",
        "target_execution_completion",
        "component_execution_target",
        "component_execution_target_admission",
    }
)
_SCHEDULED_COMPONENT_EXECUTION_EVENT_TYPES = (
    _IMMEDIATE_COMPONENT_EXECUTION_EVENT_TYPES
    | {
        "component_execution_target_activation",
        "reference_receiver_validation",
        "reference_segment",
        "reference_segment_admission",
        "group_generation_reservation",
        "reference_segment_terminal",
    }
)
_DIRECT_COMPONENT_COMPOSITION_LEGACY_NOT_APPLICABLE = frozenset(
    {
        "station_run_manifest",
        "station_intent_transition",
        "station_operation_transition",
        "station_publication_barrier_transition",
        "composition_session_spec",
        "composition_session_rearm_spec",
        "composition_session_rearm_receipt",
        "composition_session_transition",
        "replay_action_record",
        "replay_source_frame_emission",
        "legacy_replay_contract_derivation",
        "replay_source_action_contract",
        "replay_binding_assessment",
        "replay_source_adapter_receipt",
        "replay_execution_authorization",
    }
)
_DIRECT_COMPONENT_COMPOSITION_LINEAGE_POLICIES = frozenset(
    {
        "authoritative_immediate_component_policy_human_"
        "composition_lineage_v1",
        "authoritative_scheduled_component_policy_human_"
        "composition_lineage_v1",
    }
)
_SCHEDULED_REFERENCE_EVENT_TYPES = frozenset(
    {
        "reference_receiver_validation",
        "reference_segment",
        "reference_segment_admission",
        "group_generation_reservation",
        "group_generation_activation",
        "safety_anchor_commit_candidate",
        "reference_target_activation",
        "planner_command_frame",
        "reference_segment_terminal",
    }
)
_SCHEDULED_POSE_TO_QPOS_EVENT_TYPES = (
    _SCHEDULED_REFERENCE_EVENT_TYPES - {"safety_anchor_commit_candidate"}
) | {
    "resolved_execution_target",
    "execution_target_admission",
    "safety_anchor_commit_evidence",
}
_AUTHORITATIVE_NOT_APPLICABLE = frozenset(
    {
        "arbitration_evaluation",
        "arbiter_shadow_status",
        "shared_execution_evaluation",
        "legacy_execution_observation",
    }
)

_RM65_IDLE_WORKER_PREFLIGHT_REQUIRED = frozenset(
    {
        "hardware_preflight_authorization",
        "execution_cancellation",
        "execution_cancellation_ack",
        "execution_worker_drain",
        "execution_worker_drain_aggregate",
        "runtime_trace_terminal",
    }
)


def _profile(
    profile_id: str,
    runtime_mode: str,
    *,
    required: frozenset[str],
    not_applicable: frozenset[str],
    receipt_policy: str,
    measurement_policy: str,
    writer_failure_behavior: str,
    lineage_policy_id: str,
    profile_scope: str,
    implementation_status: str,
) -> TraceRequirementProfile:
    return TraceRequirementProfile(
        profile_id=profile_id,
        runtime_mode=runtime_mode,
        required_event_types=required,
        optional_event_types=CONTROL_TRACE_EVENT_TYPES_V24 - required - not_applicable,
        not_applicable_event_types=not_applicable,
        receipt_policy=receipt_policy,
        measurement_policy=measurement_policy,
        writer_failure_behavior=writer_failure_behavior,
        lineage_policy_id=lineage_policy_id,
        profile_scope=profile_scope,
        implementation_status=implementation_status,
    )


TRACE_REQUIREMENT_PROFILES_V1: Mapping[str, TraceRequirementProfile] = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (
            _profile(
                "plain_dry_run_v1",
                "plain_dry_run",
                required=frozenset({"runtime_trace_terminal"}),
                not_applicable=_DRY_RUN_NOT_APPLICABLE,
                receipt_policy="not_applicable",
                measurement_policy="not_applicable",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="generic_reference_integrity_v1",
                profile_scope="runtime_diagnostic",
                implementation_status="fixture_only",
            ),
            _profile(
                "m5_8b1_real_a_idle_worker_drain_stop_only_v1",
                "rm65_hardware_idle_worker_drain_preflight",
                required=_RM65_IDLE_WORKER_PREFLIGHT_REQUIRED,
                not_applicable=frozenset(),
                receipt_policy="optional",
                measurement_policy="optional",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="rm65_idle_worker_drain_preflight_lineage_v1",
                profile_scope="runtime_diagnostic",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "policy_shadow_dry_run_v1",
                "policy_shadow_dry_run",
                required=_POLICY_SHADOW_REQUIRED,
                not_applicable=_DRY_RUN_NOT_APPLICABLE,
                receipt_policy="not_applicable",
                measurement_policy="not_applicable",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="policy_shadow_lineage_v1",
                profile_scope="runtime_diagnostic",
                implementation_status="fixture_only",
            ),
            _profile(
                "deployable_sim_component_mixing_shadow_v1",
                "deployable_sim_component_mixing_shadow",
                required=_COMPONENT_MIXING_SHADOW_REQUIRED,
                not_applicable=_DRY_RUN_NOT_APPLICABLE,
                receipt_policy="not_applicable",
                measurement_policy="not_applicable",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="component_mixing_shadow_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="fixture_only",
            ),
            _profile(
                "human_shadow_dry_run_v1",
                "human_shadow_dry_run",
                required=_POLICY_SHADOW_REQUIRED,
                not_applicable=_DRY_RUN_NOT_APPLICABLE,
                receipt_policy="not_applicable",
                measurement_policy="not_applicable",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="human_shadow_lineage_v1",
                profile_scope="runtime_diagnostic",
                implementation_status="fixture_only",
            ),
            _profile(
                "replay_shadow_dry_run_v1",
                "replay_shadow_dry_run",
                required=_REPLAY_SHADOW_REQUIRED,
                not_applicable=_DRY_RUN_NOT_APPLICABLE,
                receipt_policy="not_applicable",
                measurement_policy="not_applicable",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="replay_shadow_lineage_v1",
                profile_scope="runtime_diagnostic",
                implementation_status="fixture_only",
            ),
            _profile(
                "formal_real_shadow_v1",
                "formal_real_shadow",
                required=_POLICY_SHADOW_REQUIRED,
                not_applicable=_SHADOW_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="optional",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="policy_shadow_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="planned",
            ),
            _profile(
                "formal_real_human_shadow_v1",
                "formal_real_human_shadow",
                required=_FORMAL_HUMAN_SHADOW_REQUIRED,
                not_applicable=_SHADOW_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="optional",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="human_shadow_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "formal_real_policy_shadow_v1",
                "formal_real_policy_shadow",
                required=_FORMAL_POLICY_SHADOW_REQUIRED,
                not_applicable=_SHADOW_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="optional",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="policy_shadow_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="planned",
            ),
            _profile(
                "formal_real_replay_shadow_v1",
                "formal_real_replay_shadow",
                required=_FORMAL_REPLAY_SHADOW_REQUIRED,
                not_applicable=_SHADOW_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="optional",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="replay_shadow_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="planned",
            ),
            _profile(
                "authoritative_real_v1",
                "authoritative_real",
                required=_AUTHORITATIVE_REQUIRED,
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="conditional_on_execution",
                writer_failure_behavior="request_runtime_hold",
                lineage_policy_id="authoritative_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="planned",
            ),
            _profile(
                "deployable_sim_v1",
                "deployable_sim",
                required=_AUTHORITATIVE_REQUIRED,
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="conditional_on_dispatch",
                measurement_policy="conditional_on_execution",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="authoritative_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="planned",
            ),
            _profile(
                "deployable_sim_immediate_qref_v1",
                "deployable_sim_immediate_qref",
                required=_AUTHORITATIVE_SIM_CORE_REQUIRED,
                not_applicable=(
                    _AUTHORITATIVE_NOT_APPLICABLE
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                ),
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="authoritative_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="fixture_only",
            ),
            _profile(
                "deployable_sim_scheduled_qpos_v1",
                "deployable_sim_scheduled_qpos",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="authoritative_scheduled_qpos_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_qpos_human_replay_handoff_v1",
                "deployable_sim_scheduled_qpos_human_replay_handoff",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                    | _AUTHORITATIVE_HUMAN_SOURCE_REQUIRED
                    | _AUTHORITATIVE_REPLAY_SOURCE_REQUIRED
                    | {
                        "authority_event",
                        "execution_gate_event",
                        "execution_cancellation",
                        "execution_cancellation_ack",
                    }
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_immediate_qref_station_v1",
                "deployable_sim_immediate_qref_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _STATION_LIFECYCLE_REQUIRED
                ),
                not_applicable=(
                    _AUTHORITATIVE_NOT_APPLICABLE
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                ),
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="authoritative_station_immediate_qref_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="fixture_only",
            ),
            _profile(
                "deployable_sim_immediate_qref_rtc_policy_station_v1",
                "deployable_sim_immediate_qref_rtc_policy_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _STATION_LIFECYCLE_REQUIRED
                    | _AUTHORITATIVE_POLICY_SOURCE_REQUIRED
                ),
                not_applicable=(
                    _AUTHORITATIVE_NOT_APPLICABLE
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                ),
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_station_immediate_qref_rtc_policy_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_qpos_station_v1",
                "deployable_sim_scheduled_qpos_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                    | _STATION_LIFECYCLE_REQUIRED
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id="authoritative_station_scheduled_qpos_lineage_v1",
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_qpos_policy_station_v1",
                "deployable_sim_scheduled_qpos_policy_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                    | _STATION_LIFECYCLE_REQUIRED
                    | _AUTHORITATIVE_POLICY_SOURCE_REQUIRED
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_station_scheduled_qpos_policy_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_pose_to_qpos_station_v1",
                "deployable_sim_scheduled_pose_to_qpos_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_POSE_TO_QPOS_EVENT_TYPES
                    | _STATION_LIFECYCLE_REQUIRED
                    | _AUTHORITATIVE_REPLAY_SOURCE_REQUIRED
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_station_scheduled_pose_to_qpos_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_pose_human_replay_station_v1",
                "deployable_sim_scheduled_pose_human_replay_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_POSE_TO_QPOS_EVENT_TYPES
                    | _STATION_LIFECYCLE_REQUIRED
                    | _AUTHORITATIVE_POSE_HUMAN_REPLAY_SOURCE_REQUIRED
                    | {
                        "authority_event",
                        "execution_gate_event",
                        "execution_cancellation",
                        "execution_cancellation_ack",
                    }
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_station_scheduled_pose_human_replay_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_pose_composer_station_v1",
                "deployable_sim_scheduled_pose_composer_station",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _EXECUTION_MANIFEST_REQUIRED
                    | _SCHEDULED_POSE_TO_QPOS_EVENT_TYPES
                    | _STATION_LIFECYCLE_REQUIRED
                    | _COMPOSITION_SESSION_REQUIRED
                    | _AUTHORITATIVE_HUMAN_SOURCE_REQUIRED
                    | {
                        "replay_source_frame_emission",
                        "replay_source_adapter_receipt",
                        "execution_gate_event",
                        "execution_cancellation",
                        "execution_cancellation_ack",
                    }
                ),
                not_applicable=_AUTHORITATIVE_NOT_APPLICABLE,
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_station_pose_composition_session_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_immediate_component_policy_human_composer_v1",
                "deployable_sim_immediate_component_policy_human_composer",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _IMMEDIATE_COMPONENT_EXECUTION_EVENT_TYPES
                    | _DIRECT_COMPONENT_HOT_COMPOSITION_REQUIRED
                    | _AUTHORITATIVE_POLICY_SOURCE_REQUIRED
                    | _AUTHORITATIVE_HUMAN_SOURCE_REQUIRED
                    | {
                        "execution_gate_event",
                        "execution_cancellation",
                        "execution_cancellation_ack",
                    }
                ),
                not_applicable=(
                    _AUTHORITATIVE_NOT_APPLICABLE
                    | _DIRECT_COMPONENT_COMPOSITION_LEGACY_NOT_APPLICABLE
                    | _SCHEDULED_REFERENCE_EVENT_TYPES
                    | {"component_execution_target_activation"}
                ),
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_immediate_component_policy_human_"
                    "composition_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
            _profile(
                "deployable_sim_scheduled_component_policy_human_composer_v1",
                "deployable_sim_scheduled_component_policy_human_composer",
                required=(
                    _AUTHORITATIVE_SIM_CORE_REQUIRED
                    | _SCHEDULED_COMPONENT_EXECUTION_EVENT_TYPES
                    | _DIRECT_COMPONENT_HOT_COMPOSITION_REQUIRED
                    | _AUTHORITATIVE_POLICY_SOURCE_REQUIRED
                    | _AUTHORITATIVE_HUMAN_SOURCE_REQUIRED
                    | {
                        "execution_gate_event",
                        "execution_cancellation",
                        "execution_cancellation_ack",
                    }
                ),
                not_applicable=(
                    _AUTHORITATIVE_NOT_APPLICABLE
                    | _DIRECT_COMPONENT_COMPOSITION_LEGACY_NOT_APPLICABLE
                ),
                receipt_policy="required",
                measurement_policy="required",
                writer_failure_behavior="mark_uncommitted_continue",
                lineage_policy_id=(
                    "authoritative_scheduled_component_policy_human_"
                    "composition_lineage_v1"
                ),
                profile_scope="formal_capture",
                implementation_status="runtime_integrated",
            ),
        )
    }
)


def trace_requirement_profile(profile_id: str) -> TraceRequirementProfile:
    try:
        return TRACE_REQUIREMENT_PROFILES_V1[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown TraceRequirementProfile: {profile_id!r}.") from exc


@dataclass(frozen=True)
class ControlTraceLineageIssue:
    code: str
    root_fact_id: str
    detail: str
    related_fact_ids: tuple[str, ...] = ()


def evaluate_control_trace_lineage(
    facts: Sequence[ControlTraceFact],
    *,
    profile: TraceRequirementProfile,
) -> tuple[ControlTraceLineageIssue, ...]:
    """Validate implemented branch-aware references without inferring facts."""

    context = ControlTraceValidationContext.build(facts)
    issue_sink = ControlTraceLineageIssueSink(ControlTraceLineageIssue)
    manifests = run_control_trace_foundation(context, profile, issue_sink)
    execution_manifest = manifests.execution_manifest
    action_adaptation_indexes = ActionAdaptationValidationIndex.build(
        context
    )
    validate_action_adaptation_lineage(
        context,
        action_adaptation_indexes,
        execution_manifest,
        issue_sink,
    )
    execution_directive_indexes = ExecutionDirectiveValidationIndex.build(
        context
    )
    validate_execution_directive_lineage(
        context,
        execution_directive_indexes,
        action_adaptation_indexes,
        issue_sink,
    )
    target_execution_indexes = TargetExecutionValidationIndex.build(context)
    execution_cancellation_indexes = (
        ExecutionCancellationValidationIndex.build(context)
    )
    component_authority_indexes = ComponentAuthorityValidationIndex.build(
        context
    )
    composition_spec_by_hash = (
        component_authority_indexes.composition_spec_by_hash
    )
    component_action_schema_by_hash = (
        component_authority_indexes.component_action_schema_by_hash
    )
    validate_component_authority_lineage(
        context,
        component_authority_indexes,
        issue_sink,
    )
    composition_session_indexes = CompositionSessionValidationIndex.build(
        context,
        numerical_spec_by_hash=composition_spec_by_hash,
    )
    composition_session_spec_by_id = (
        composition_session_indexes.session_spec_by_id
    )
    validate_composition_session_lineage(
        context,
        composition_session_indexes,
        issue_sink,
    )
    composition_evaluation_indexes = (
        CompositionEvaluationValidationIndex.build(context)
    )
    validate_composition_evaluation_lineage(
        context,
        composition_evaluation_indexes,
        composition_spec_by_hash=composition_spec_by_hash,
        component_action_schema_by_hash=component_action_schema_by_hash,
        composition_session_spec_by_id=composition_session_spec_by_id,
        direct_component_activation=(
            profile.lineage_policy_id
            in _DIRECT_COMPONENT_COMPOSITION_LINEAGE_POLICIES
        ),
        issues=issue_sink,
    )
    source_proposal_indexes = SourceProposalValidationIndex.build(context)
    validate_source_proposal_lineage(
        context,
        source_proposal_indexes,
        composition_evaluation_indexes,
        lineage_policy_id=profile.lineage_policy_id,
        issues=issue_sink,
    )
    replay_lineage_indexes = ReplayLineageValidationIndex.build(context)
    validate_replay_lineage(
        context,
        replay_lineage_indexes,
        source_proposal_indexes,
        lineage_policy_id=profile.lineage_policy_id,
        issues=issue_sink,
    )
    arbitration_shadow_indexes = ArbitrationShadowValidationIndex.build(
        context
    )
    validate_arbitration_shadow_lineage(
        context,
        arbitration_shadow_indexes,
        source_proposal_indexes,
        execution_manifest=execution_manifest,
        lineage_policy_id=profile.lineage_policy_id,
        issues=issue_sink,
    )
    reference_lifecycle_indexes = ReferenceLifecycleValidationIndex.build(
        context
    )
    validate_reference_lifecycle_lineage(
        context,
        reference_lifecycle_indexes,
        execution_manifest=execution_manifest,
        lineage_policy_id=profile.lineage_policy_id,
        issues=issue_sink,
    )
    validate_execution_cancellation_lineage(
        context,
        execution_cancellation_indexes,
        issue_sink,
    )

    validate_target_execution_lineage(
        target_execution_indexes,
        enabled=(
            profile.lineage_policy_id
            in _DIRECT_COMPONENT_COMPOSITION_LINEAGE_POLICIES
        ),
        activation_required=profile.lineage_policy_id.startswith(
            "authoritative_scheduled_"
        ),
        issues=issue_sink,
    )

    return issue_sink.sorted_issues()


@dataclass(frozen=True)
class ControlTraceCompletenessReport:
    profile_id: str
    observed_event_types: frozenset[str]
    missing_required_event_types: frozenset[str]
    observed_not_applicable_event_types: frozenset[str]
    journal_issued_sequence: int
    sink_accepted_high_watermark: int
    sink_durable_high_watermark: int
    admission_gap_ranges: tuple[tuple[int, int], ...]
    durability_gap_ranges: tuple[tuple[int, int], ...]
    terminal_observed: bool
    lineage_policy_id: str
    lineage_issues: tuple[ControlTraceLineageIssue, ...]
    lineage_complete: bool
    in_memory_complete: bool
    durability_supported: bool
    durable_complete: bool
    capture_commit_eligible: bool


__all__ = [
    "TRACE_REQUIREMENT_PROFILES_V1",
    "WRITER_FAILURE_BEHAVIORS_V1",
    "ControlTraceCompletenessReport",
    "ControlTraceLineageIssue",
    "TraceRequirementProfile",
    "evaluate_control_trace_lineage",
    "trace_requirement_profile",
]

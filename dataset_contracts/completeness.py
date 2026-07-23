"""M4A ControlTrace requirement profiles and in-memory completeness reports."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .control_trace import CONTROL_TRACE_EVENT_TYPES_V24, ControlTraceFact


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


def _payload_index(
    facts: Sequence[ControlTraceFact],
    *,
    event_type: str,
    payload_key: str,
) -> dict[str, list[ControlTraceFact]]:
    result: dict[str, list[ControlTraceFact]] = {}
    for fact in facts:
        if fact.event_type != event_type:
            continue
        value = fact.payload.get(payload_key)
        if value is not None:
            result.setdefault(str(value), []).append(fact)
    return result


def evaluate_control_trace_lineage(
    facts: Sequence[ControlTraceFact],
    *,
    profile: TraceRequirementProfile,
) -> tuple[ControlTraceLineageIssue, ...]:
    """Validate implemented branch-aware references without inferring facts."""

    ordered = tuple(facts)
    by_id = {fact.fact_id: fact for fact in ordered}
    issues: list[ControlTraceLineageIssue] = []

    def issue(
        code: str,
        root_fact_id: str,
        detail: str,
        *related_fact_ids: str,
    ) -> None:
        issues.append(
            ControlTraceLineageIssue(
                code=code,
                root_fact_id=root_fact_id,
                detail=detail,
                related_fact_ids=tuple(str(value) for value in related_fact_ids if value),
            )
        )

    if profile.implementation_status == "planned":
        issue(
            "profile_not_implemented",
            profile.profile_id,
            "The profile is classified but has no accepted runtime fixture yet.",
        )

    execution_manifests = tuple(
        fact for fact in ordered if fact.event_type == "execution_manifest"
    )
    if len(execution_manifests) > 1:
        for duplicate in execution_manifests[1:]:
            issue(
                "multiple_execution_manifests",
                duplicate.fact_id,
                "One assessed control domain may bind only one execution manifest.",
                execution_manifests[0].fact_id,
            )
    execution_manifest = execution_manifests[0] if execution_manifests else None

    station_manifests = tuple(
        fact for fact in ordered if fact.event_type == "station_run_manifest"
    )
    if len(station_manifests) > 1:
        for duplicate in station_manifests[1:]:
            issue(
                "multiple_station_run_manifests",
                duplicate.fact_id,
                "One assessed Station runtime may bind only one run manifest.",
                station_manifests[0].fact_id,
            )
    station_manifest = station_manifests[0] if station_manifests else None
    station_intent_by_id = _payload_index(
        ordered,
        event_type="station_intent_transition",
        payload_key="intent_id",
    )
    station_operation_by_id = _payload_index(
        ordered,
        event_type="station_operation_transition",
        payload_key="operation_result_id",
    )
    enabled_barrier_authority_operations = {"request_motion_authority"}
    if profile.lineage_policy_id == (
        "authoritative_station_scheduled_pose_human_replay_lineage_v1"
    ):
        enabled_barrier_authority_operations.add("request_human_takeover")

    station_state_transitions = sorted(
        (
            fact
            for fact in ordered
            if fact.event_type
            in {"station_intent_transition", "station_operation_transition"}
        ),
        key=lambda fact: int(fact.payload["admission_sequence"]),
    )
    previous_station_fact: ControlTraceFact | None = None
    seen_station_admission_sequences: set[int] = set()
    for station_fact in station_state_transitions:
        admission_sequence = int(station_fact.payload["admission_sequence"])
        if admission_sequence in seen_station_admission_sequences:
            issue(
                "station_admission_sequence_duplicate",
                station_fact.fact_id,
                "Station intent/result admission sequence must be unique.",
            )
        seen_station_admission_sequences.add(admission_sequence)
        if previous_station_fact is not None and (
            station_fact.payload["state_before"]["semantic_content_hash"]
            != previous_station_fact.payload["state_after"]["semantic_content_hash"]
        ):
            issue(
                "station_state_lineage_discontinuous",
                station_fact.fact_id,
                "Station transition state_before does not match the prior state_after.",
                previous_station_fact.fact_id,
            )
        previous_station_fact = station_fact

    if station_manifest is not None:
        station_manifest_hash = str(
            station_manifest.payload["semantic_content_hash"]
        )
        for fact in station_state_transitions:
            if fact.event_type == "station_intent_transition" and str(
                fact.payload["manifest_hash"]
            ) != station_manifest_hash:
                issue(
                    "station_intent_manifest_mismatch",
                    fact.fact_id,
                    "Station intent references another run manifest.",
                    station_manifest.fact_id,
                )

    for fact in ordered:
        if fact.event_type == "station_operation_transition":
            source_intent_id = str(fact.payload["source_intent_id"])
            source_intents = station_intent_by_id.get(source_intent_id, ())
            if len(source_intents) != 1:
                issue(
                    "station_operation_missing_intent",
                    fact.fact_id,
                    "Station operation result must resolve one source intent.",
                    source_intent_id,
                )
            elif (
                source_intents[0].payload["intent_kind"]
                != fact.payload["operation_kind"]
                or source_intents[0].payload["subject_id"]
                != fact.payload["subject_id"]
            ):
                issue(
                    "station_operation_intent_binding_mismatch",
                    fact.fact_id,
                    "Station operation kind/subject differs from its source intent.",
                    source_intent_id,
                )
        elif fact.event_type == "station_publication_barrier_transition":
            causal_id = str(fact.payload["causal_station_fact_id"])
            if causal_id not in by_id:
                issue(
                    "station_barrier_missing_cause",
                    fact.fact_id,
                    "Publication barrier transition must resolve its Station cause.",
                    causal_id,
                )
            if bool(fact.payload["enabled"]):
                operation_result_id = str(
                    fact.payload["authority_operation_result_id"]
                )
                results = station_operation_by_id.get(operation_result_id, ())
                if (
                    len(results) != 1
                    or not bool(results[0].payload["succeeded"])
                    or results[0].payload["operation_kind"]
                    not in enabled_barrier_authority_operations
                    or (
                        results[0].payload["operation_kind"]
                        == "request_human_takeover"
                        and results[0].payload.get("result_status")
                        != "human_takeover_ready"
                    )
                    or not tuple(fact.payload["authority_evidence_refs"])
                ):
                    issue(
                        "station_barrier_missing_authority_result",
                        fact.fact_id,
                        "Enabled publication must resolve one evidenced authority result.",
                        operation_result_id,
                    )

    # Membership events are an ordered assertion stream independent of the
    # source/control lineage policy.
    active_episode: str | None = None
    membership_epoch = -1
    for fact in ordered:
        if fact.event_type != "episode_membership_transition":
            continue
        payload = fact.payload
        epoch = int(payload["membership_epoch"])
        episode_uuid = str(payload["episode_uuid"])
        transition = str(payload["transition"])
        if epoch <= membership_epoch:
            issue(
                "membership_epoch_not_increasing",
                fact.fact_id,
                "Episode membership epochs must strictly increase.",
            )
        membership_epoch = max(membership_epoch, epoch)
        if transition == "bind":
            if active_episode is not None:
                issue(
                    "membership_bind_while_active",
                    fact.fact_id,
                    "A second episode was bound before the current membership was released.",
                )
            active_episode = episode_uuid
        elif transition == "unbind":
            if active_episode != episode_uuid:
                issue(
                    "membership_unbind_mismatch",
                    fact.fact_id,
                    "Episode unbind does not match the active membership.",
                )
            active_episode = None
        else:
            issue(
                "membership_transition_invalid",
                fact.fact_id,
                f"Unsupported membership transition {transition!r}.",
            )

    proposal_by_id = _payload_index(
        ordered, event_type="action_proposal", payload_key="proposal_id"
    )
    activation_by_id = _payload_index(
        ordered, event_type="producer_epoch_activation", payload_key="activation_id"
    )
    input_by_id = _payload_index(
        ordered,
        event_type="arbitration_input_snapshot",
        payload_key="arbitration_input_snapshot_id",
    )
    computation_by_id = _payload_index(
        ordered, event_type="arbitration_computation", payload_key="result_id"
    )
    evaluation_by_computation = _payload_index(
        ordered,
        event_type="arbitration_evaluation",
        payload_key="computation_result_id",
    )
    admission_by_emission = _payload_index(
        ordered, event_type="proposal_admission", payload_key="emission_id"
    )
    validation_by_emission = _payload_index(
        ordered, event_type="source_validation", payload_key="source_event_id"
    )
    human_emission_by_id = _payload_index(
        ordered, event_type="human_command_emission", payload_key="source_emission_id"
    )
    source_receipt_by_id = _payload_index(
        ordered,
        event_type="source_adapter_receipt",
        payload_key="representation_trace_id",
    )
    replay_receipt_by_id = _payload_index(
        ordered,
        event_type="replay_source_adapter_receipt",
        payload_key="representation_trace_id",
    )
    replay_record_by_id = _payload_index(
        ordered,
        event_type="replay_action_record",
        payload_key="replay_action_record_id",
    )
    replay_source_action_contract_by_id = _payload_index(
        ordered,
        event_type="replay_source_action_contract",
        payload_key="contract_id",
    )
    target_runtime_contract_by_id = _payload_index(
        ordered,
        event_type="target_runtime_action_contract",
        payload_key="target_contract_id",
    )
    replay_assessment_by_id = _payload_index(
        ordered,
        event_type="replay_binding_assessment",
        payload_key="assessment_id",
    )
    replay_eligibility_by_id = _payload_index(
        ordered,
        event_type="candidate_binding_eligibility",
        payload_key="eligibility_id",
    )
    replay_execution_authorization_by_id = _payload_index(
        ordered,
        event_type="replay_execution_authorization",
        payload_key="authorization_id",
    )
    candidate_authorization_admission_by_id = _payload_index(
        ordered,
        event_type="candidate_authorization_admission",
        payload_key="authorization_id",
    )
    status_by_emission = _payload_index(
        ordered, event_type="arbiter_shadow_status", payload_key="emission_id"
    )
    input_by_schedule_tick = _payload_index(
        ordered,
        event_type="arbitration_input_snapshot",
        payload_key="decision_schedule_tick_id",
    )
    publication_attempt_by_id = _payload_index(
        ordered,
        event_type="arbitration_publication_attempt",
        payload_key="publication_attempt_id",
    )
    decision_by_id = _payload_index(
        ordered,
        event_type="arbitration_decision",
        payload_key="decision_id",
    )
    cancellation_by_id = _payload_index(
        ordered,
        event_type="execution_cancellation",
        payload_key="cancellation_id",
    )
    cancellation_ack_by_id = _payload_index(
        ordered,
        event_type="execution_cancellation_ack",
        payload_key="cancellation_id",
    )
    dispatch_by_evidence_id = _payload_index(
        ordered,
        event_type="dispatch_owner",
        payload_key="evidence_event_id",
    )
    policy_context_by_id = _payload_index(
        ordered,
        event_type="policy_context",
        payload_key="context_id",
    )
    composition_input_by_id = _payload_index(
        ordered,
        event_type="composition_input",
        payload_key="composition_input_id",
    )
    composition_snapshot_by_id = _payload_index(
        ordered,
        event_type="composition_input_snapshot",
        payload_key="composition_input_snapshot_id",
    )
    composition_result_by_id = _payload_index(
        ordered,
        event_type="composition_result",
        payload_key="composition_result_id",
    )
    reference_segment_by_id = _payload_index(
        ordered,
        event_type="reference_segment",
        payload_key="reference_segment_id",
    )
    resolved_execution_target_by_id = _payload_index(
        ordered,
        event_type="resolved_execution_target",
        payload_key="target_id",
    )
    execution_target_admission_by_target = _payload_index(
        ordered,
        event_type="execution_target_admission",
        payload_key="target_id",
    )
    anchor_commit_evidence_by_target = _payload_index(
        ordered,
        event_type="safety_anchor_commit_evidence",
        payload_key="target_id",
    )
    if execution_manifest is not None:
        manifest_catalog_hash = str(
            execution_manifest.payload["component_catalog_hash"]
        )
        for fact in ordered:
            if fact.event_type == "arbitration_input_snapshot" and str(
                fact.payload["component_catalog_hash"]
            ) != manifest_catalog_hash:
                issue(
                    "execution_manifest_component_catalog_mismatch",
                    fact.fact_id,
                    "Arbitration input uses another component catalog.",
                    execution_manifest.fact_id,
                )
            if fact.event_type == "reference_segment" and str(
                fact.payload["input_action_schema_hash"]
            ) != manifest_catalog_hash:
                end_parents = tuple(
                    parent
                    for parent in fact.payload["endpoint_parents"]
                    if str(parent["role"]) == "end"
                )
                resolved_bridge_ok = False
                if (
                    len(end_parents) == 1
                    and str(end_parents[0]["source_kind"])
                    == "resolved_execution_target"
                ):
                    targets = resolved_execution_target_by_id.get(
                        str(end_parents[0]["source_fact_id"]), ()
                    )
                    resolved_bridge_ok = bool(
                        len(targets) == 1
                        and str(targets[0].payload["input_action_schema_hash"])
                        == manifest_catalog_hash
                        and str(targets[0].payload["output_target_schema_hash"])
                        == str(fact.payload["input_action_schema_hash"])
                    )
                if not resolved_bridge_ok:
                    issue(
                        "execution_manifest_reference_schema_mismatch",
                        fact.fact_id,
                        "Reference segment input schema differs from the execution manifest without an explicit resolved-target bridge.",
                        execution_manifest.fact_id,
                    )
    reference_validation_by_id = _payload_index(
        ordered,
        event_type="reference_receiver_validation",
        payload_key="validation_id",
    )
    reference_admission_by_id = _payload_index(
        ordered,
        event_type="reference_segment_admission",
        payload_key="admission_id",
    )
    generation_reservation_by_id = _payload_index(
        ordered,
        event_type="group_generation_reservation",
        payload_key="reservation_id",
    )
    generation_activation_by_id = _payload_index(
        ordered,
        event_type="group_generation_activation",
        payload_key="activation_id",
    )
    reference_activation_by_segment = _payload_index(
        ordered,
        event_type="reference_target_activation",
        payload_key="reference_segment_id",
    )
    anchor_candidate_by_segment = _payload_index(
        ordered,
        event_type="safety_anchor_commit_candidate",
        payload_key="reference_segment_id",
    )
    generation_reservation_by_segment = _payload_index(
        ordered,
        event_type="group_generation_reservation",
        payload_key="source_segment_id",
    )
    generation_activation_by_segment = _payload_index(
        ordered,
        event_type="group_generation_activation",
        payload_key="source_segment_id",
    )
    reference_terminal_by_segment = _payload_index(
        ordered,
        event_type="reference_segment_terminal",
        payload_key="reference_segment_id",
    )

    for fact in ordered:
        payload = fact.payload
        if fact.event_type == "policy_context_admission" and bool(payload["accepted"]):
            context_id = str(payload["context_id"])
            contexts = policy_context_by_id.get(context_id, ())
            if len(contexts) != 1 or contexts[0].payload[
                "context_semantic_content_hash"
            ] != payload["context_semantic_content_hash"]:
                issue(
                    "policy_context_admission_missing_context",
                    fact.fact_id,
                    "Accepted policy context admission must resolve its exact context.",
                    context_id,
                )
        elif fact.event_type == "policy_context":
            observation_cut_id = str(payload["observation_cut_id"])
            observation = by_id.get(observation_cut_id)
            if observation is None or observation.event_type != "observation_cut":
                issue(
                    "policy_context_missing_observation_cut",
                    fact.fact_id,
                    "Policy context must resolve its producer observation cut.",
                    observation_cut_id,
                )
            for parent_id in payload["parent_context_ids"]:
                if len(policy_context_by_id.get(str(parent_id), ())) != 1:
                    issue(
                        "policy_context_missing_parent",
                        fact.fact_id,
                        "Policy context parent dependency is absent.",
                        str(parent_id),
                    )
        elif fact.event_type == "policy_context_snapshot":
            selected_ids = payload["selected_context_id_by_channel"]
            selected_hashes = payload["selected_context_hash_by_channel"]
            for channel, context_id in selected_ids.items():
                contexts = policy_context_by_id.get(str(context_id), ())
                if len(contexts) != 1 or contexts[0].payload[
                    "context_semantic_content_hash"
                ] != selected_hashes[channel]:
                    issue(
                        "policy_context_snapshot_missing_context",
                        fact.fact_id,
                        "Context snapshot must resolve each selected context and hash.",
                        str(context_id),
                    )
        elif fact.event_type == "composition_input_admission" and bool(
            payload["accepted"]
        ):
            input_id = str(payload["composition_input_id"])
            inputs = composition_input_by_id.get(input_id, ())
            if len(inputs) != 1 or inputs[0].payload[
                "input_semantic_content_hash"
            ] != payload["input_semantic_content_hash"]:
                issue(
                    "composition_input_admission_missing_input",
                    fact.fact_id,
                    "Accepted composition input admission must resolve its exact input.",
                    input_id,
                )
        elif fact.event_type == "composition_input":
            for context_id in payload["context_ids"]:
                if len(policy_context_by_id.get(str(context_id), ())) != 1:
                    issue(
                        "composition_input_missing_context",
                        fact.fact_id,
                        "Composition input context dependency is absent.",
                        str(context_id),
                    )
            observation_cut_id = payload["observation_cut_id"]
            if observation_cut_id is not None:
                observation = by_id.get(str(observation_cut_id))
                if observation is None or observation.event_type != "observation_cut":
                    issue(
                        "composition_input_missing_observation_cut",
                        fact.fact_id,
                        "Composition input must resolve its producer observation cut.",
                        str(observation_cut_id),
                    )
        elif fact.event_type == "composition_input_snapshot":
            selected_ids = payload["selected_input_ids_by_role"]
            selected_hashes = payload["selected_input_hashes_by_role"]
            for role, input_id in selected_ids.items():
                inputs = composition_input_by_id.get(str(input_id), ())
                if len(inputs) != 1 or inputs[0].payload[
                    "input_semantic_content_hash"
                ] != selected_hashes[role]:
                    issue(
                        "composition_snapshot_missing_input",
                        fact.fact_id,
                        "Composition snapshot must resolve each selected input and hash.",
                        str(input_id),
                    )
        elif fact.event_type == "composition_result":
            snapshot_id = str(payload["composition_input_snapshot_id"])
            snapshots = composition_snapshot_by_id.get(snapshot_id, ())
            if len(snapshots) != 1 or snapshots[0].payload[
                "semantic_content_hash"
            ] != payload["composition_input_snapshot_hash"]:
                issue(
                    "composition_result_missing_snapshot",
                    fact.fact_id,
                    "Composition result must resolve its exact input snapshot.",
                    snapshot_id,
                )
            if len(payload["input_ids"]) != len(payload["input_hashes"]):
                issue(
                    "composition_result_input_lineage_length_mismatch",
                    fact.fact_id,
                    "Composition result input IDs and hashes must have equal length.",
                )
            for input_id, input_hash in zip(
                payload["input_ids"], payload["input_hashes"]
            ):
                inputs = composition_input_by_id.get(str(input_id), ())
                if len(inputs) != 1 or inputs[0].payload[
                    "input_semantic_content_hash"
                ] != input_hash:
                    issue(
                        "composition_result_missing_input",
                        fact.fact_id,
                        "Composition result input lineage is absent or hash-mismatched.",
                        str(input_id),
                    )
        elif fact.event_type == "reference_segment":
            parents = tuple(payload["endpoint_parents"])
            if len(parents) != 2 or tuple(item["role"] for item in parents) != (
                "start",
                "end",
            ):
                issue(
                    "reference_segment_parent_order_invalid",
                    fact.fact_id,
                    "Reference segment must preserve ordered START/END parents.",
                )
            elif str(parents[1]["source_fact_id"]) not in by_id:
                issue(
                    "reference_segment_missing_safe_result",
                    fact.fact_id,
                    "Reference END parent must resolve its shared-safety fact.",
                    str(parents[1]["source_fact_id"]),
                )
        elif fact.event_type == "reference_segment_admission":
            segment_id = str(payload["reference_segment_id"])
            segments = reference_segment_by_id.get(segment_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issue(
                    "reference_admission_missing_segment",
                    fact.fact_id,
                    "Reference admission must resolve its exact segment.",
                    segment_id,
                )
            validation_id = str(payload["receiver_validation_id"])
            validations = reference_validation_by_id.get(validation_id, ())
            if len(validations) != 1 or validations[0].payload[
                "semantic_content_hash"
            ] != payload["receiver_validation_hash"]:
                issue(
                    "reference_admission_missing_receiver_validation",
                    fact.fact_id,
                    "Reference admission must resolve its receiver-side validation.",
                    validation_id,
                )
            elif bool(payload["accepted"]) and not bool(
                validations[0].payload["accepted"]
            ):
                issue(
                    "reference_admission_validation_rejected",
                    fact.fact_id,
                    "Accepted segment cannot reference rejected receiver validation.",
                    validation_id,
                )
        elif fact.event_type == "safety_anchor_commit_candidate":
            segment_id = str(payload["reference_segment_id"])
            admission_id = str(payload["admission_id"])
            segments = reference_segment_by_id.get(segment_id, ())
            admissions = reference_admission_by_id.get(admission_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issue(
                    "anchor_candidate_missing_segment",
                    fact.fact_id,
                    "Anchor candidate must resolve its exact segment.",
                    segment_id,
                )
            if len(admissions) != 1 or admissions[0].payload[
                "semantic_content_hash"
            ] != payload["admission_hash"]:
                issue(
                    "anchor_candidate_missing_admission",
                    fact.fact_id,
                    "Anchor candidate must resolve its exact admission.",
                    admission_id,
                )
            elif not bool(admissions[0].payload["accepted"]):
                issue(
                    "anchor_candidate_from_rejected_admission",
                    fact.fact_id,
                    "Rejected admission cannot produce an anchor candidate.",
                    admission_id,
                )
            safety_id = str(payload["safety_result_id"])
            safety = by_id.get(safety_id)
            if safety is None or safety.event_type != "shared_safety_result":
                issue(
                    "anchor_candidate_missing_safety_result",
                    fact.fact_id,
                    "Anchor candidate must resolve its shared-safety result.",
                    safety_id,
                )
        elif fact.event_type == "group_generation_reservation":
            segment_id = str(payload["source_segment_id"])
            admission_id = str(payload["admission_id"])
            segments = reference_segment_by_id.get(segment_id, ())
            admissions = reference_admission_by_id.get(admission_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["source_segment_hash"]:
                issue(
                    "generation_reservation_missing_segment",
                    fact.fact_id,
                    "Generation reservation must resolve its exact segment.",
                    segment_id,
                )
            if len(admissions) != 1 or admissions[0].payload[
                "semantic_content_hash"
            ] != payload["admission_hash"]:
                issue(
                    "generation_reservation_missing_admission",
                    fact.fact_id,
                    "Generation reservation must resolve its exact admission.",
                    admission_id,
                )
            elif not bool(admissions[0].payload["accepted"]):
                issue(
                    "generation_reservation_from_rejected_admission",
                    fact.fact_id,
                    "Rejected segment admission cannot reserve a generation.",
                    admission_id,
                )
            if int(payload["reserved_generation"]) != int(
                payload["active_generation_at_reservation"]
            ) + 1:
                issue(
                    "generation_reservation_not_next",
                    fact.fact_id,
                    "Generation reservation must reserve active generation + 1.",
                )
            relation = str(payload["relation_kind"])
            predecessor_id = payload["predecessor_segment_id"]
            supersedes_id = payload["supersedes_reservation_id"]
            if relation == "successor" and len(
                reference_segment_by_id.get(str(predecessor_id), ())
            ) != 1:
                issue(
                    "successor_reservation_missing_predecessor",
                    fact.fact_id,
                    "Successor reservation must resolve its predecessor segment.",
                    str(predecessor_id),
                )
            if relation == "supersedes_pending" and len(
                generation_reservation_by_id.get(str(supersedes_id), ())
            ) != 1:
                issue(
                    "superseding_reservation_missing_old_reservation",
                    fact.fact_id,
                    "Superseding reservation must resolve the old reservation.",
                    str(supersedes_id),
                )
        elif fact.event_type == "group_generation_activation":
            reservation_id = str(payload["reservation_id"])
            reservations = generation_reservation_by_id.get(reservation_id, ())
            if len(reservations) != 1:
                issue(
                    "generation_activation_missing_reservation",
                    fact.fact_id,
                    "Generation activation must resolve one reservation.",
                    reservation_id,
                )
            else:
                reservation = reservations[0].payload
                if (
                    str(payload["source_segment_id"])
                    != str(reservation["source_segment_id"])
                    or int(payload["active_generation"])
                    != int(reservation["reserved_generation"])
                    or int(payload["execution_epoch"])
                    != int(reservation["execution_epoch"])
                ):
                    issue(
                        "generation_activation_reservation_mismatch",
                        fact.fact_id,
                        "Activation and reservation lineage disagree.",
                        reservation_id,
                    )
        elif fact.event_type == "reference_target_activation":
            segment_id = str(payload["reference_segment_id"])
            if len(reference_segment_by_id.get(segment_id, ())) != 1:
                issue(
                    "reference_activation_missing_segment",
                    fact.fact_id,
                    "Reference target activation must resolve one segment.",
                    segment_id,
                )
            generation_activation_id = str(payload["generation_activation_id"])
            generation_activations = generation_activation_by_id.get(
                generation_activation_id,
                (),
            )
            if len(generation_activations) != 1:
                issue(
                    "reference_activation_missing_generation_activation",
                    fact.fact_id,
                    "Reference activation must resolve its generation handoff.",
                    generation_activation_id,
                )
            else:
                generation_activation = generation_activations[0].payload
                if (
                    str(generation_activation["source_segment_id"]) != segment_id
                    or int(generation_activation["active_generation"])
                    != int(payload["group_segment_generation"])
                ):
                    issue(
                        "reference_generation_activation_mismatch",
                        fact.fact_id,
                        "Reference activation and generation handoff disagree.",
                        generation_activation_id,
                    )
        elif fact.event_type == "planner_command_frame":
            segment_id = str(payload["reference_segment_id"])
            if len(reference_segment_by_id.get(segment_id, ())) != 1:
                issue(
                    "planner_frame_missing_segment",
                    fact.fact_id,
                    "Planner command frame must resolve one segment.",
                    segment_id,
                )
            activations = reference_activation_by_segment.get(segment_id, ())
            if len(activations) != 1:
                issue(
                    "planner_frame_missing_activation",
                    fact.fact_id,
                    "Planner command frame must resolve one target activation.",
                    segment_id,
                )
            elif int(activations[0].payload["reference_sequence"]) != int(
                payload["reference_sequence"]
            ):
                issue(
                    "planner_frame_reference_sequence_mismatch",
                    fact.fact_id,
                    "Planner frame and target activation reference sequences differ.",
                    activations[0].fact_id,
                )
        elif fact.event_type == "reference_segment_terminal":
            segment_id = str(payload["reference_segment_id"])
            segments = reference_segment_by_id.get(segment_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issue(
                    "reference_terminal_missing_segment",
                    fact.fact_id,
                    "Reference terminal must resolve its exact segment.",
                    segment_id,
                )
            last_frame_id = payload["last_reference_frame_id"]
            if last_frame_id is not None:
                frame = by_id.get(str(last_frame_id))
                if frame is None or frame.event_type != "planner_command_frame":
                    issue(
                        "reference_terminal_missing_last_frame",
                        fact.fact_id,
                        "Reference terminal last frame must resolve a planner frame.",
                        str(last_frame_id),
                    )
                elif str(frame.payload["reference_segment_id"]) != segment_id:
                    issue(
                        "reference_terminal_last_frame_segment_mismatch",
                        fact.fact_id,
                        "Reference terminal last frame belongs to another segment.",
                        str(last_frame_id),
                    )
        elif fact.event_type == "execution_directive":
            attempt_id = str(payload["publication_attempt_id"])
            attempts = publication_attempt_by_id.get(attempt_id, ())
            if len(attempts) != 1:
                issue(
                    "execution_directive_missing_publication_attempt",
                    fact.fact_id,
                    "ExecutionDirective must resolve one publication attempt.",
                    attempt_id,
                )
            decision_id = payload["decision_id"]
            if decision_id is not None:
                decisions = decision_by_id.get(str(decision_id), ())
                if len(decisions) != 1:
                    issue(
                        "execution_directive_missing_decision",
                        fact.fact_id,
                        "Selected ExecutionDirective must resolve one authoritative decision.",
                        str(decision_id),
                    )
            elif len(attempts) == 1 and bool(attempts[0].payload["accepted"]):
                issue(
                    "nonselected_directive_has_accepted_publication",
                    fact.fact_id,
                    "A directive without decision_id cannot reference accepted publication.",
                    attempt_id,
                )
        elif fact.event_type == "execution_cancellation":
            cancellation_id = str(payload["cancellation_id"])
            for source_event_id in payload["source_event_ids"]:
                if str(source_event_id) not in by_id:
                    issue(
                        "execution_cancellation_missing_source_event",
                        fact.fact_id,
                        "Execution cancellation source lineage is absent.",
                        str(source_event_id),
                    )
            if payload["effective_scope"] == "control_domain" and int(
                payload["execution_epoch"]
            ) != int(payload["previous_execution_epoch"]) + 1:
                issue(
                    "execution_cancellation_epoch_not_advanced",
                    fact.fact_id,
                    "Domain cancellation must advance execution epoch exactly once.",
                )
            acknowledgements = cancellation_ack_by_id.get(cancellation_id, ())
            for level in payload["required_ack_levels"]:
                matches = tuple(
                    item
                    for item in acknowledgements
                    if item.payload["acknowledgement_level"] == level
                )
                if len(matches) != 1:
                    issue(
                        "execution_cancellation_missing_required_ack",
                        fact.fact_id,
                        "Cancellation must resolve exactly one required acknowledgement.",
                        *(item.fact_id for item in matches),
                    )
                elif not bool(matches[0].payload["accepted"]):
                    issue(
                        "execution_cancellation_ack_rejected",
                        fact.fact_id,
                        "A required cancellation acknowledgement was rejected.",
                        matches[0].fact_id,
                    )
            if bool(payload["requests_backend_hold"]):
                permits = tuple(
                    item
                    for item in dispatch_by_evidence_id.get(cancellation_id, ())
                    if item.payload["permit_kind"] == "safety_control"
                    and item.payload["operation"] == "hold"
                )
                # A control-domain HOLD may fan out to multiple physical final
                # SDK seams (for example one one-shot permit per coupled arm).
                # Require at least one accepted permit and reject any failed
                # member; do not collapse the fanout into a reusable permit.
                if not permits or any(
                    not bool(item.payload["validation_accepted"])
                    for item in permits
                ):
                    issue(
                        "execution_cancellation_missing_backend_hold_permit",
                        fact.fact_id,
                        "Domain hold requires accepted one-shot safety-control permits.",
                        *(item.fact_id for item in permits),
                    )
        elif fact.event_type == "execution_cancellation_ack":
            cancellation_id = str(payload["cancellation_id"])
            if len(cancellation_by_id.get(cancellation_id, ())) != 1:
                issue(
                    "execution_cancellation_ack_missing_request",
                    fact.fact_id,
                    "Cancellation acknowledgement must resolve one cancellation.",
                    cancellation_id,
                )
        elif fact.event_type == "producer_epoch_snapshot":
            for activation_id in payload["activation_id_by_producer"].values():
                if str(activation_id) not in activation_by_id:
                    issue(
                        "producer_snapshot_missing_activation",
                        fact.fact_id,
                        "Producer epoch snapshot references an absent activation.",
                        str(activation_id),
                    )
        elif fact.event_type == "proposal_admission":
            proposal_id = payload["proposal_id"]
            accepted = bool(payload["accepted"])
            matches = () if proposal_id is None else proposal_by_id.get(str(proposal_id), ())
            if accepted and len(matches) != 1:
                issue(
                    "accepted_admission_missing_proposal",
                    fact.fact_id,
                    "An accepted admission must resolve to exactly one action proposal.",
                    "" if proposal_id is None else str(proposal_id),
                )
            if not accepted and matches:
                issue(
                    "rejected_admission_has_proposal",
                    fact.fact_id,
                    "A rejected admission must not materialize an action proposal.",
                    str(proposal_id),
                )
        elif fact.event_type == "candidate_authorization_admission" and bool(
            payload["accepted"]
        ):
            proposal_id = str(payload["proposal_id"])
            if len(proposal_by_id.get(proposal_id, ())) != 1:
                issue(
                    "candidate_authorization_missing_proposal",
                    fact.fact_id,
                    "Accepted candidate authorization must resolve its proposal.",
                    proposal_id,
                )
            eligibility_id = str(payload["binding_eligibility_id"])
            if len(replay_eligibility_by_id.get(eligibility_id, ())) != 1:
                issue(
                    "candidate_authorization_missing_binding_eligibility",
                    fact.fact_id,
                    "Replay candidate authorization must resolve binding eligibility.",
                    eligibility_id,
                )
            evidence_id = str(payload["evidence_id"])
            if len(replay_execution_authorization_by_id.get(evidence_id, ())) != 1:
                issue(
                    "candidate_authorization_missing_execution_authorization",
                    fact.fact_id,
                    "Replay candidate authorization must resolve execution authorization.",
                    evidence_id,
                )
        elif fact.event_type == "candidate_authorization_snapshot":
            for relation in payload["authorization_by_proposal_id"].values():
                authorization_id = str(relation["authorization_id"])
                admissions = candidate_authorization_admission_by_id.get(
                    authorization_id,
                    (),
                )
                if len(admissions) != 1:
                    issue(
                        "candidate_authorization_snapshot_missing_admission",
                        fact.fact_id,
                        "Frozen candidate authorization must resolve one admission.",
                        authorization_id,
                    )
        elif fact.event_type == "source_adapter_receipt":
            source_emission_id = str(payload["source_emission_id"])
            emissions = human_emission_by_id.get(source_emission_id, ())
            if len(emissions) != 1:
                issue(
                    "source_receipt_missing_human_emission",
                    fact.fact_id,
                    "Human source receipt must resolve to one command emission.",
                    source_emission_id,
                )
            elif payload["source_emission_semantic_hash"] != emissions[0].payload[
                "source_emission_semantic_hash"
            ]:
                issue(
                    "source_receipt_emission_hash_mismatch",
                    fact.fact_id,
                    "Human source receipt and command emission hashes disagree.",
                    source_emission_id,
                )
            validations = validation_by_emission.get(source_emission_id, ())
            if len(validations) != 1:
                issue(
                    "source_receipt_missing_validation",
                    fact.fact_id,
                    "Human source receipt must resolve to one source validation.",
                    source_emission_id,
                )
            elif bool(payload["source_validation_accepted"]) != bool(
                validations[0].payload["accepted"]
            ):
                issue(
                    "source_receipt_validation_mismatch",
                    fact.fact_id,
                    "Human source receipt and source-validation outcomes disagree.",
                    validations[0].fact_id,
                )
        elif fact.event_type == "action_proposal" and payload["producer_kind"] == "human":
            representation_trace_id = payload["representation_trace_id"]
            receipts = (
                ()
                if representation_trace_id is None
                else source_receipt_by_id.get(str(representation_trace_id), ())
            )
            if len(receipts) != 1:
                issue(
                    "human_proposal_missing_source_receipt",
                    fact.fact_id,
                    "Human proposal must resolve to one typed source-adapter receipt.",
                    "" if representation_trace_id is None else str(representation_trace_id),
                )
        elif fact.event_type == "replay_binding_assessment":
            source_contracts = replay_source_action_contract_by_id.get(
                str(payload["source_action_contract_id"]), ()
            )
            target_contracts = target_runtime_contract_by_id.get(
                str(payload["target_runtime_contract_id"]), ()
            )
            if len(source_contracts) != 1 or (
                source_contracts[0].payload["semantic_content_hash"]
                != payload["source_action_contract_hash"]
            ):
                issue(
                    "replay_assessment_missing_source_contract",
                    fact.fact_id,
                    "Replay assessment must resolve its exact source action contract.",
                )
            if len(target_contracts) != 1 or (
                target_contracts[0].payload["semantic_content_hash"]
                != payload["target_runtime_contract_hash"]
            ):
                issue(
                    "replay_assessment_missing_target_contract",
                    fact.fact_id,
                    "Replay assessment must resolve its exact target runtime contract.",
                )
        elif fact.event_type == "candidate_binding_eligibility":
            assessments = replay_assessment_by_id.get(str(payload["assessment_id"]), ())
            if len(assessments) != 1 or (
                assessments[0].payload["semantic_content_hash"]
                != payload["assessment_hash"]
            ):
                issue(
                    "replay_eligibility_missing_assessment",
                    fact.fact_id,
                    "Replay candidate eligibility must resolve its exact assessment.",
                )
            if bool(payload["authoritative_allowed"]) and payload["requested_use"] != (
                "execution_candidate"
            ):
                issue(
                    "replay_eligibility_invalid_authoritative_scope",
                    fact.fact_id,
                    "Only execution-candidate eligibility may be authoritative-allowed.",
                )
        elif fact.event_type == "replay_source_adapter_receipt":
            record_id = payload["replay_action_record_id"]
            records = () if record_id is None else replay_record_by_id.get(str(record_id), ())
            if bool(payload["accepted"]):
                if len(records) != 1:
                    issue(
                        "replay_receipt_missing_record",
                        fact.fact_id,
                        "Accepted replay adapter receipt must resolve to one replay action record.",
                        "" if record_id is None else str(record_id),
                    )
                elif payload["replay_action_record_hash"] != records[0].payload[
                    "semantic_content_hash"
                ]:
                    issue(
                        "replay_receipt_record_hash_mismatch",
                        fact.fact_id,
                        "Replay adapter receipt and source record hashes disagree.",
                        records[0].fact_id,
                    )
            assessments = replay_assessment_by_id.get(
                str(payload["binding_assessment_id"]), ()
            )
            eligibility = replay_eligibility_by_id.get(
                str(payload["binding_eligibility_id"]), ()
            )
            if len(assessments) != 1 or (
                assessments[0].payload["semantic_content_hash"]
                != payload["binding_assessment_hash"]
            ):
                issue(
                    "replay_receipt_missing_assessment",
                    fact.fact_id,
                    "Replay receipt must resolve its exact binding assessment.",
                )
            if len(eligibility) != 1 or (
                eligibility[0].payload["semantic_content_hash"]
                != payload["binding_eligibility_hash"]
            ):
                issue(
                    "replay_receipt_missing_eligibility",
                    fact.fact_id,
                    "Replay receipt must resolve its exact candidate eligibility.",
                )
            source_emission_id = str(payload["source_emission_id"])
            source_emission = by_id.get(source_emission_id)
            if source_emission is None or source_emission.event_type not in {
                "policy_action_emission",
                "replay_source_frame_emission",
            }:
                issue(
                    "replay_receipt_missing_passive_emission",
                    fact.fact_id,
                    "Replay receipt must reference its unique passive source emission.",
                    source_emission_id,
                )
        elif fact.event_type == "action_proposal" and payload["producer_kind"] == "replay":
            representation_trace_id = payload["representation_trace_id"]
            receipts = (
                ()
                if representation_trace_id is None
                else replay_receipt_by_id.get(str(representation_trace_id), ())
            )
            if len(receipts) != 1:
                issue(
                    "replay_proposal_missing_source_receipt",
                    fact.fact_id,
                    "Replay proposal must resolve to one typed replay adapter receipt.",
                    "" if representation_trace_id is None else str(representation_trace_id),
                )
        elif fact.event_type == "action_proposal" and payload["producer_kind"] == "composer":
            result_id = str(payload["policy_output_id"])
            results = composition_result_by_id.get(result_id, ())
            if len(results) != 1:
                issue(
                    "composite_proposal_missing_result",
                    fact.fact_id,
                    "Composite proposal must resolve one composition result.",
                    result_id,
                )
            else:
                result_payload = results[0].payload
                if tuple(payload["composition_input_ids"]) != tuple(
                    result_payload["input_ids"]
                ) or tuple(payload["composition_input_hashes"]) != tuple(
                    result_payload["input_hashes"]
                ):
                    issue(
                        "composite_proposal_input_lineage_mismatch",
                        fact.fact_id,
                        "Composite proposal and composition result inputs disagree.",
                        result_id,
                    )
                if payload["composition_spec_hash"] != result_payload[
                    "composition_spec_hash"
                ]:
                    issue(
                        "composite_proposal_spec_mismatch",
                        fact.fact_id,
                        "Composite proposal and composition result specs disagree.",
                        result_id,
                    )
        elif fact.event_type == "proposal_snapshot":
            for candidate in payload["candidates"]:
                proposal_id = str(candidate["proposal_id"])
                matches = proposal_by_id.get(proposal_id, ())
                if len(matches) != 1:
                    issue(
                        "snapshot_candidate_missing_proposal",
                        fact.fact_id,
                        "ProposalSnapshot candidate does not resolve to exactly one proposal.",
                        proposal_id,
                    )
                elif (
                    matches[0].payload.get("component_schema_hash")
                    != payload["component_schema_hash"]
                ):
                    issue(
                        "snapshot_candidate_schema_mismatch",
                        fact.fact_id,
                        "Candidate and ProposalSnapshot component schemas disagree.",
                        proposal_id,
                    )
                elif candidate["proposal_content_hash"] != matches[0].payload.get(
                    "proposal_semantic_content_hash"
                ):
                    issue(
                        "snapshot_candidate_hash_mismatch",
                        fact.fact_id,
                        "Candidate content hash does not match its proposal fact.",
                        proposal_id,
                    )
        elif fact.event_type == "decision_schedule_tick":
            tick_id = str(payload["schedule_tick_id"])
            if len(input_by_schedule_tick.get(tick_id, ())) != 1:
                issue(
                    "decision_tick_missing_input",
                    fact.fact_id,
                    "Each materialized decision tick must resolve to one arbitration input.",
                )
        elif fact.event_type == "arbitration_input_snapshot":
            child_fields = {
                "proposal_snapshot_id": "proposal_snapshot",
                "authority_state_snapshot_fact_id": "authority_state_snapshot",
                "execution_gate_snapshot_fact_id": "execution_gate_snapshot",
                "producer_epoch_snapshot_fact_id": "producer_epoch_snapshot",
                "candidate_authorization_snapshot_fact_id": (
                    "candidate_authorization_snapshot"
                ),
                "decision_schedule_tick_id": "decision_schedule_tick",
            }
            for field_name, expected_event_type in child_fields.items():
                child_id = str(payload[field_name])
                child = by_id.get(child_id)
                if child is None or child.event_type != expected_event_type:
                    issue(
                        "arbitration_input_missing_child",
                        fact.fact_id,
                        (
                            f"Arbitration input child {field_name} is absent or has "
                            "the wrong event type."
                        ),
                        child_id,
                    )
            child_hash_fields = {
                "proposal_snapshot_id": "proposal_snapshot_hash",
                "authority_state_snapshot_fact_id": "authority_state_snapshot_hash",
                "execution_gate_snapshot_fact_id": "execution_gate_snapshot_hash",
                "producer_epoch_snapshot_fact_id": "producer_epoch_snapshot_hash",
                "candidate_authorization_snapshot_fact_id": (
                    "candidate_authorization_snapshot_hash"
                ),
                "decision_schedule_tick_id": "decision_schedule_tick_hash",
            }
            for child_id_field, child_hash_field in child_hash_fields.items():
                child = by_id.get(str(payload[child_id_field]))
                if child is None:
                    continue
                source_hash = child.payload.get("semantic_content_hash")
                if source_hash is not None and payload[child_hash_field] != source_hash:
                    issue(
                        "arbitration_input_child_hash_mismatch",
                        fact.fact_id,
                        f"Arbitration input {child_hash_field} disagrees with child fact.",
                        child.fact_id,
                    )
        elif fact.event_type == "arbitration_computation":
            input_id = str(payload["arbitration_input_snapshot_id"])
            if len(input_by_id.get(input_id, ())) != 1:
                issue(
                    "computation_missing_input",
                    fact.fact_id,
                    "Arbitration computation must resolve to one input snapshot.",
                    input_id,
                )
            elif payload["arbitration_input_hash"] != input_by_id[input_id][0].payload[
                "semantic_content_hash"
            ]:
                issue(
                    "computation_input_hash_mismatch",
                    fact.fact_id,
                    "Computation input hash disagrees with its input snapshot.",
                    input_id,
                )
        elif fact.event_type == "arbitration_evaluation":
            computation_id = str(payload["computation_result_id"])
            if len(computation_by_id.get(computation_id, ())) != 1:
                issue(
                    "evaluation_missing_computation",
                    fact.fact_id,
                    "Shadow evaluation must resolve to one computation.",
                    computation_id,
                )
            elif payload["computation_result_hash"] != computation_by_id[
                computation_id
            ][0].payload["semantic_content_hash"]:
                issue(
                    "evaluation_computation_hash_mismatch",
                    fact.fact_id,
                    "Evaluation computation hash disagrees with its computation fact.",
                    computation_id,
                )
            input_id = str(payload["arbitration_input_snapshot_id"])
            if len(input_by_id.get(input_id, ())) != 1:
                issue(
                    "evaluation_missing_input",
                    fact.fact_id,
                    "Shadow evaluation must resolve to one input snapshot.",
                    input_id,
                )
        elif fact.event_type == "shared_execution_evaluation":
            source_evaluation_id = payload["source_arbitration_evaluation_id"]
            if source_evaluation_id is None or (
                str(source_evaluation_id) not in by_id
                or by_id[str(source_evaluation_id)].event_type
                != "arbitration_evaluation"
            ):
                issue(
                    "shared_execution_shadow_missing_evaluation",
                    fact.fact_id,
                    "Shared execution shadow must reference one arbitration evaluation.",
                    str(source_evaluation_id),
                )
            elif str(
                by_id[str(source_evaluation_id)].payload["computation_result_id"]
            ) != str(payload["arbitration_result_id"]):
                issue(
                    "shared_execution_shadow_result_mismatch",
                    fact.fact_id,
                    "Shared execution shadow and arbitration evaluation disagree.",
                    str(source_evaluation_id),
                )
            if not bool(payload["non_authoritative"]):
                issue(
                    "shared_execution_shadow_marked_authoritative",
                    fact.fact_id,
                    "Shared execution evaluation must remain non-authoritative.",
                )
            directive = payload["proposed_directive"]
            if str(directive["arbitration_result_id"]) != str(
                payload["arbitration_result_id"]
            ):
                issue(
                    "shared_execution_shadow_directive_mismatch",
                    fact.fact_id,
                    "Proposed directive references another arbitration result.",
                )
        elif fact.event_type == "legacy_execution_observation":
            emission_id = str(payload["emission_id"])
            if emission_id not in by_id or by_id[emission_id].event_type not in {
                "policy_action_emission",
                "human_command_emission",
            }:
                issue(
                    "legacy_execution_missing_emission",
                    fact.fact_id,
                    "Legacy execution must reference its passive source emission.",
                    emission_id,
                )
            evaluation_id = payload["evaluation_id"]
            if evaluation_id is not None and (
                str(evaluation_id) not in by_id
                or by_id[str(evaluation_id)].event_type != "arbitration_evaluation"
            ):
                issue(
                    "legacy_execution_missing_evaluation",
                    fact.fact_id,
                    "Legacy execution references an absent shadow evaluation.",
                    str(evaluation_id),
                )

    if profile.lineage_policy_id in {
        "policy_shadow_lineage_v1",
        "authoritative_station_immediate_qref_rtc_policy_lineage_v1",
        "authoritative_station_scheduled_qpos_policy_lineage_v1",
    }:
        for fact in ordered:
            if fact.event_type != "policy_action_emission":
                continue
            emission_id = str(fact.payload["emission_id"])
            admissions = admission_by_emission.get(emission_id, ())
            statuses = status_by_emission.get(emission_id, ())
            validations = validation_by_emission.get(emission_id, ())
            if len(validations) != 1:
                issue(
                    "emission_missing_source_validation",
                    fact.fact_id,
                    "Each policy emission must have exactly one source-validation fact.",
                )
            if len(admissions) > 1:
                issue(
                    "emission_multiple_admissions",
                    fact.fact_id,
                    "A queue-pop emission may be admitted at most once.",
                    *(item.fact_id for item in admissions),
                )
            if not admissions and not statuses:
                issue(
                    "emission_missing_outcome",
                    fact.fact_id,
                    "An emission needs an admission receipt or typed shadow status.",
                )

    if profile.lineage_policy_id in {
        "human_shadow_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }:
        receipt_by_emission = _payload_index(
            ordered,
            event_type="source_adapter_receipt",
            payload_key="source_emission_id",
        )
        for fact in ordered:
            if fact.event_type != "human_command_emission":
                continue
            emission_id = str(fact.payload["source_emission_id"])
            if len(receipt_by_emission.get(emission_id, ())) != 1:
                issue(
                    "human_emission_missing_source_receipt",
                    fact.fact_id,
                    "Each human command emission needs one typed source-adapter receipt.",
                )
            if len(validation_by_emission.get(emission_id, ())) != 1:
                issue(
                    "human_emission_missing_source_validation",
                    fact.fact_id,
                    "Each human command emission needs one source-validation fact.",
                )
            admissions = admission_by_emission.get(emission_id, ())
            statuses = status_by_emission.get(emission_id, ())
            if len(admissions) > 1:
                issue(
                    "human_emission_multiple_admissions",
                    fact.fact_id,
                    "A human emission may be admitted at most once.",
                    *(item.fact_id for item in admissions),
                )
            if not admissions and not statuses:
                issue(
                    "human_emission_missing_outcome",
                    fact.fact_id,
                    "A human emission needs an admission receipt or typed shadow status.",
                )

    if profile.lineage_policy_id in {
        "replay_shadow_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }:
        receipt_by_source_emission = _payload_index(
            ordered,
            event_type="replay_source_adapter_receipt",
            payload_key="source_emission_id",
        )
        replay_emission_event_type = (
            "replay_source_frame_emission"
            if profile.lineage_policy_id
            in {
                "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
                "authoritative_station_scheduled_pose_human_replay_lineage_v1",
            }
            else "policy_action_emission"
        )
        for fact in ordered:
            if fact.event_type != replay_emission_event_type:
                continue
            emission_id = str(fact.payload["emission_id"])
            if len(receipt_by_source_emission.get(emission_id, ())) != 1:
                issue(
                    "replay_emission_missing_source_receipt",
                    fact.fact_id,
                    "Each replay queue-pop emission needs one typed replay adapter receipt.",
                )
            if (
                replay_emission_event_type == "policy_action_emission"
                and len(validation_by_emission.get(emission_id, ())) != 1
            ):
                issue(
                    "replay_emission_missing_source_validation",
                    fact.fact_id,
                    "Each replay queue-pop emission needs one source-validation fact.",
                )
        for fact in ordered:
            if fact.event_type != "replay_source_adapter_receipt" or not bool(
                fact.payload["accepted"]
            ):
                continue
            admission_key = str(
                fact.payload[
                    "source_emission_id"
                    if profile.lineage_policy_id
                    in {
                        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
                        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
                    }
                    else "representation_trace_id"
                ]
            )
            admissions = admission_by_emission.get(admission_key, ())
            if len(admissions) != 1:
                issue(
                    "replay_receipt_missing_admission",
                    fact.fact_id,
                    "Accepted replay adapter receipt needs exactly one proposal admission.",
                )

    if profile.lineage_policy_id in {
        "policy_shadow_lineage_v1",
        "human_shadow_lineage_v1",
        "replay_shadow_lineage_v1",
    }:
        for fact in ordered:
            if fact.event_type != "arbitration_input_snapshot":
                continue
            input_id = str(fact.payload["arbitration_input_snapshot_id"])
            computations = [
                item
                for item in ordered
                if item.event_type == "arbitration_computation"
                and str(item.payload["arbitration_input_snapshot_id"]) == input_id
            ]
            if len(computations) != 1:
                issue(
                    "arbitration_input_missing_computation",
                    fact.fact_id,
                    "Each shadow arbitration input needs exactly one computation.",
                )
            elif len(evaluation_by_computation.get(computations[0].fact_id, ())) != 1:
                issue(
                    "computation_missing_evaluation",
                    computations[0].fact_id,
                    "Each shadow computation needs exactly one evaluation.",
                )

    if profile.lineage_policy_id in {
        "authoritative_scheduled_qpos_lineage_v1",
        "authoritative_station_scheduled_qpos_lineage_v1",
        "authoritative_station_scheduled_pose_to_qpos_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }:
        pose_to_qpos = profile.lineage_policy_id in {
            "authoritative_station_scheduled_pose_to_qpos_lineage_v1",
            "authoritative_station_scheduled_pose_human_replay_lineage_v1",
        }
        # Scheduled-reference event families are branch-conditional.  A
        # segment canceled before START must not invent an activation or
        # planner frame, while every accepted segment still needs a complete
        # admission/candidate/reservation/terminal closure.
        for safety in ordered:
            if (
                safety.event_type != "shared_safety_result"
                or str(safety.payload["status"]) != "safe"
                or safety.payload["safe_action"] is None
            ):
                continue
            if pose_to_qpos:
                matching_targets = tuple(
                    target
                    for target in ordered
                    if target.event_type == "resolved_execution_target"
                    and str(target.payload["source_safety_result_id"])
                    == safety.fact_id
                )
                if len(matching_targets) != 1:
                    issue(
                        "scheduled_safe_pose_missing_resolved_target",
                        safety.fact_id,
                        "Each safe scheduled pose result needs exactly one resolved execution target.",
                        *(target.fact_id for target in matching_targets),
                    )
            else:
                matching_segments = tuple(
                    segment
                    for segment in ordered
                    if segment.event_type == "reference_segment"
                    and len(tuple(segment.payload["endpoint_parents"])) == 2
                    and str(segment.payload["endpoint_parents"][1]["source_fact_id"])
                    == safety.fact_id
                )
                if len(matching_segments) != 1:
                    issue(
                        "scheduled_safe_result_missing_segment_attempt",
                        safety.fact_id,
                        "Each safe scheduled-qpos result needs exactly one segment attempt.",
                        *(segment.fact_id for segment in matching_segments),
                    )

        if pose_to_qpos:
            for target_id, targets in resolved_execution_target_by_id.items():
                if len(targets) != 1:
                    continue
                target = targets[0]
                admissions = tuple(
                    execution_target_admission_by_target.get(target_id, ())
                )
                if len(admissions) != 1:
                    issue(
                        "resolved_target_missing_single_admission",
                        target.fact_id,
                        "Each resolved execution target needs one receiver admission.",
                        *(item.fact_id for item in admissions),
                    )
                    continue
                if not bool(admissions[0].payload["accepted"]):
                    continue
                segments = tuple(
                    segment
                    for segment in ordered
                    if segment.event_type == "reference_segment"
                    and any(
                        str(parent["role"]) == "end"
                        and str(parent["source_kind"])
                        == "resolved_execution_target"
                        and str(parent["source_fact_id"]) == target_id
                        for parent in segment.payload["endpoint_parents"]
                    )
                )
                if len(segments) != 1:
                    issue(
                        "accepted_resolved_target_missing_segment",
                        target.fact_id,
                        "An accepted resolved target needs exactly one reference segment.",
                        *(item.fact_id for item in segments),
                    )
                evidence = tuple(anchor_commit_evidence_by_target.get(target_id, ()))
                if len(evidence) != 1 or not bool(evidence[0].payload["accepted"]):
                    issue(
                        "accepted_resolved_target_missing_anchor_evidence",
                        target.fact_id,
                        "An accepted resolved target needs one accepted pose-anchor commit evidence fact.",
                        *(item.fact_id for item in evidence),
                    )

        for segment_id, segments in reference_segment_by_id.items():
            if len(segments) != 1:
                continue
            segment = segments[0]
            admissions = tuple(
                item
                for item in ordered
                if item.event_type == "reference_segment_admission"
                and str(item.payload["reference_segment_id"]) == segment_id
            )
            if len(admissions) != 1:
                issue(
                    "scheduled_segment_missing_single_admission",
                    segment.fact_id,
                    "Each scheduled segment attempt needs exactly one receiver admission.",
                    *(item.fact_id for item in admissions),
                )
                continue
            if not bool(admissions[0].payload["accepted"]):
                continue

            candidates = tuple(anchor_candidate_by_segment.get(segment_id, ()))
            reservations = tuple(generation_reservation_by_segment.get(segment_id, ()))
            terminals = tuple(reference_terminal_by_segment.get(segment_id, ()))
            if not pose_to_qpos and len(candidates) != 1:
                issue(
                    "accepted_segment_missing_anchor_candidate",
                    segment.fact_id,
                    "An accepted segment needs exactly one pending anchor candidate.",
                    *(item.fact_id for item in candidates),
                )
            if pose_to_qpos and candidates:
                issue(
                    "resolved_segment_has_qpos_anchor_candidate",
                    segment.fact_id,
                    "A pose-to-qpos segment must use resolved-target anchor evidence, not a qpos safety-anchor candidate.",
                    *(item.fact_id for item in candidates),
                )
            if len(reservations) != 1:
                issue(
                    "accepted_segment_missing_generation_reservation",
                    segment.fact_id,
                    "An accepted segment needs exactly one generation reservation.",
                    *(item.fact_id for item in reservations),
                )
            if len(terminals) != 1:
                issue(
                    "accepted_segment_missing_terminal",
                    segment.fact_id,
                    "An accepted segment must be terminal before its episode is sealed.",
                    *(item.fact_id for item in terminals),
                )

            generation_activations = tuple(
                generation_activation_by_segment.get(segment_id, ())
            )
            target_activations = tuple(reference_activation_by_segment.get(segment_id, ()))
            if len(generation_activations) != len(target_activations):
                issue(
                    "scheduled_segment_activation_pair_incomplete",
                    segment.fact_id,
                    "Generation and reference-target activation must appear as one pair.",
                    *(
                        item.fact_id
                        for item in generation_activations + target_activations
                    ),
                )
            if len(generation_activations) > 1 or len(target_activations) > 1:
                issue(
                    "scheduled_segment_activated_multiple_times",
                    segment.fact_id,
                    "A scheduled segment may activate at most once.",
                    *(
                        item.fact_id
                        for item in generation_activations + target_activations
                    ),
                )

    issues.sort(key=lambda item: (item.root_fact_id, item.code, item.related_fact_ids))
    return tuple(issues)


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

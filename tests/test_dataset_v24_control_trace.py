from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.completeness import (
    TRACE_REQUIREMENT_PROFILES_V1,
    TraceRequirementProfile,
    evaluate_control_trace_lineage,
)
from dataset_contracts.control_trace import (
    CONTROL_TRACE_EVENT_TYPES_V24,
    ControlTraceFact,
    EpisodeMembershipRef,
    PersistedRecordEnvelope,
)


def fact(
    event_type: str = "policy_action_emission",
    *,
    values: np.ndarray | None = None,
) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=f"fact:{event_type}",
        event_type=event_type,
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="local_monotonic",
        payload={
            "emission_id": "emission-1",
            "admission_mode": "on_emission_pop_v1",
            "policy_node_id": "policy-1",
            "inference_id": "inference-1",
            "policy_output_id": "output-1",
            "chunk_id": 1,
            "chunk_keyframe_index": 0,
            "chunk_size": 1,
            "pre_safety_action": {
                "values": (
                    np.asarray([1.0, 2.0], dtype=np.float32)
                    if values is None
                    else values
                )
            },
            "pre_safety_action_hash": "a" * 64,
            "representation_trace_id": None,
            "pre_safety_faults": (),
            "produced_at": 9.0,
            "produced_clock_domain_id": "local_monotonic",
            "emitted_at": 10.0,
            "target_effective_at": 10.0,
            "effective_from": 10.0,
            "effective_until": 11.0,
            "effective_clock_domain_id": "local_monotonic",
            "schedule_spec_id": "schedule-1",
            "schedule_spec_hash": "b" * 64,
            "queue_len_before": 1,
            "queue_len_after": 0,
            "queue_pop_count": 1,
            "semantic_content_hash": "c" * 64,
        },
    )


def test_initial_profiles_classify_every_event_type_once() -> None:
    assert set(TRACE_REQUIREMENT_PROFILES_V1) == {
        "plain_dry_run_v1",
        "m5_8b1_real_a_idle_worker_drain_stop_only_v1",
        "policy_shadow_dry_run_v1",
        "deployable_sim_component_mixing_shadow_v1",
        "human_shadow_dry_run_v1",
        "replay_shadow_dry_run_v1",
        "formal_real_shadow_v1",
        "formal_real_human_shadow_v1",
        "formal_real_policy_shadow_v1",
        "formal_real_replay_shadow_v1",
        "authoritative_real_v1",
        "deployable_sim_v1",
        "deployable_sim_immediate_qref_v1",
        "deployable_sim_scheduled_qpos_v1",
        "deployable_sim_scheduled_qpos_human_replay_handoff_v1",
        "deployable_sim_immediate_qref_station_v1",
        "deployable_sim_immediate_qref_rtc_policy_station_v1",
        "deployable_sim_scheduled_qpos_station_v1",
        "deployable_sim_scheduled_qpos_policy_station_v1",
        "deployable_sim_scheduled_pose_to_qpos_station_v1",
        "deployable_sim_scheduled_pose_human_replay_station_v1",
        "deployable_sim_scheduled_pose_composer_station_v1",
        "deployable_sim_immediate_component_human_composer_v1",
        "deployable_sim_immediate_component_policy_human_composer_v1",
        "deployable_sim_scheduled_component_policy_human_composer_v1",
    }
    for profile in TRACE_REQUIREMENT_PROFILES_V1.values():
        classified = (
            profile.required_event_types
            | profile.optional_event_types
            | profile.not_applicable_event_types
        )
        assert classified == CONTROL_TRACE_EVENT_TYPES_V24
        assert not profile.required_event_types & profile.optional_event_types
        assert not profile.required_event_types & profile.not_applicable_event_types
        assert not profile.optional_event_types & profile.not_applicable_event_types

    shadow = TRACE_REQUIREMENT_PROFILES_V1["policy_shadow_dry_run_v1"]
    assert shadow.lineage_policy_id == "policy_shadow_lineage_v1"
    assert "arbitration_evaluation" in shadow.optional_event_types
    assert "legacy_execution_observation" in shadow.optional_event_types
    assert "arbitration_decision" in shadow.not_applicable_event_types
    authoritative = TRACE_REQUIREMENT_PROFILES_V1["authoritative_real_v1"]
    assert authoritative.implementation_status == "planned"
    assert "arbitration_decision" in authoritative.optional_event_types
    assert "arbitration_evaluation" in authoritative.not_applicable_event_types
    assert "shared_execution_evaluation" in (
        authoritative.not_applicable_event_types
    )
    formal_human = TRACE_REQUIREMENT_PROFILES_V1["formal_real_human_shadow_v1"]
    assert formal_human.profile_scope == "formal_capture"
    assert formal_human.implementation_status == "runtime_integrated"
    assert {
        "human_command_emission",
        "source_adapter_receipt",
        "source_validation",
        "legacy_execution_observation",
    } <= formal_human.required_event_types
    assert "policy_action_emission" not in formal_human.required_event_types
    station_sim = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_qpos_station_v1"
    ]
    assert station_sim.implementation_status == "runtime_integrated"
    assert {
        "station_run_manifest",
        "station_intent_transition",
        "station_operation_transition",
        "station_publication_barrier_transition",
        "reference_segment_admission",
        "planner_command_frame",
    } <= station_sim.required_event_types
    policy_station_sim = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_qpos_policy_station_v1"
    ]
    assert policy_station_sim.lineage_policy_id == (
        "authoritative_station_scheduled_qpos_policy_lineage_v1"
    )
    assert {
        "policy_action_emission",
        "source_validation",
    } <= policy_station_sim.required_event_types
    rtc_policy_station_sim = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_immediate_qref_rtc_policy_station_v1"
    ]
    assert rtc_policy_station_sim.lineage_policy_id == (
        "authoritative_station_immediate_qref_rtc_policy_lineage_v1"
    )
    assert {
        "policy_action_emission",
        "source_validation",
    } <= rtc_policy_station_sim.required_event_types
    assert "reference_segment" in rtc_policy_station_sim.not_applicable_event_types
    policy_human_component_sim = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_component_policy_human_composer_v1"
    ]
    assert policy_human_component_sim.lineage_policy_id == (
        "authoritative_scheduled_component_policy_human_composition_lineage_v1"
    )
    assert policy_human_component_sim.profile_scope == "formal_capture"
    assert (
        policy_human_component_sim.implementation_status
        == "runtime_integrated"
    )
    assert {
        "policy_action_emission",
        "human_command_emission",
        "source_adapter_receipt",
        "source_validation",
        "composition_activation_scope",
        "composition_activation_binding",
        "composition_contributor_input",
        "composition_evaluation",
        "composition_authority_disposition",
        "source_difference_observation",
    } <= policy_human_component_sim.required_event_types
    assert {
        "replay_source_frame_emission",
        "replay_source_adapter_receipt",
        "station_run_manifest",
        "station_intent_transition",
        "station_operation_transition",
        "station_publication_barrier_transition",
        "composition_session_spec",
        "composition_session_transition",
    } <= policy_human_component_sim.not_applicable_event_types


def test_direct_component_activation_does_not_invent_legacy_session_spec() -> None:
    activation = ControlTraceFact.create(
        fact_id="composition-activation:direct",
        event_type="composition_activation_scope",
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="local_monotonic",
        payload={
            "activation_id": "composition-activation:direct",
            "operation_spec_id": "sim-hil-policy-human",
            "operation_spec_hash": "a" * 64,
            "session_epoch": 1,
            "activated_at": 10.0,
            "clock_domain_id": "local_monotonic",
            "input_barrier_at": 10.0,
            "freshness_policy": "strictly_after_input_barrier",
            "required_roles": ("policy", "human"),
            "semantic_content_hash": "b" * 64,
        },
    )
    direct_profile = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_component_policy_human_composer_v1"
    ]
    direct_codes = {
        issue.code
        for issue in evaluate_control_trace_lineage(
            (activation,),
            profile=direct_profile,
        )
    }
    assert "composition_activation_missing_operation_spec" not in direct_codes

    legacy_profile = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_pose_composer_station_v1"
    ]
    legacy_codes = {
        issue.code
        for issue in evaluate_control_trace_lineage(
            (activation,),
            profile=legacy_profile,
        )
    }
    assert "composition_activation_missing_operation_spec" in legacy_codes


def test_policy_component_lineage_requires_real_composition_input() -> None:
    profile = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_scheduled_component_policy_human_composer_v1"
    ]
    codes = {
        issue.code
        for issue in evaluate_control_trace_lineage(
            (fact(),),
            profile=profile,
        )
    }
    assert "policy_component_emission_missing_source_validation" in codes
    assert "policy_component_emission_missing_input" in codes


def test_profile_rejects_unclassified_or_overlapping_events() -> None:
    with pytest.raises(ValueError, match="classify every event"):
        TraceRequirementProfile(
            profile_id="invalid",
            runtime_mode="test",
            required_event_types=frozenset({"action_proposal"}),
            optional_event_types=frozenset(),
            not_applicable_event_types=frozenset(),
        )
    with pytest.raises(ValueError, match="pairwise disjoint"):
        TraceRequirementProfile(
            profile_id="invalid",
            runtime_mode="test",
            required_event_types=frozenset(CONTROL_TRACE_EVENT_TYPES_V24),
            optional_event_types=frozenset({"action_proposal"}),
            not_applicable_event_types=frozenset(),
        )


def test_fact_payload_is_detached_and_semantic_hash_is_deterministic() -> None:
    source = np.asarray([1.0, 2.0], dtype=np.float32)
    first = fact(values=source)
    second = fact()
    source[0] = 9.0

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.payload["pre_safety_action"]["values"].flags.writeable is False
    with pytest.raises(ValueError):
        first.payload["pre_safety_action"]["values"][0] = 3.0
    with pytest.raises(TypeError):
        first.payload["new"] = "forbidden"


def test_record_integrity_hash_covers_journal_and_episode_membership() -> None:
    source = fact()
    first = PersistedRecordEnvelope.wrap(
        trace_session_id="trace-1",
        journal_sequence=1,
        fact=source,
        episode_membership=None,
    )
    repeated = PersistedRecordEnvelope.wrap(
        trace_session_id="trace-1",
        journal_sequence=1,
        fact=source,
        episode_membership=None,
    )
    episode_bound = PersistedRecordEnvelope.wrap(
        trace_session_id="trace-1",
        journal_sequence=1,
        fact=source,
        episode_membership=EpisodeMembershipRef(
            episode_uuid="episode-1",
            lifecycle_owner_id="owner-1",
            binding_event_id="bind-1",
            membership_epoch=1,
            bound_at=9.5,
        ),
    )

    assert first.record_integrity_hash == repeated.record_integrity_hash
    assert first.record_integrity_hash != source.semantic_content_hash
    assert first.record_integrity_hash != episode_bound.record_integrity_hash
    assert first.encoded_record["fact_id"] == source.fact_id
    assert first.encoded_record["payload"]["pre_safety_action"]["values"].flags.writeable is False


def test_explicit_record_schema_rejects_runtime_dataclass_field_drift() -> None:
    payload = dict(fact().payload)
    payload["unexpected_runtime_field"] = "must-not-leak"
    with pytest.raises(ValueError, match="extra=.*unexpected_runtime_field"):
        ControlTraceFact.create(
            fact_id="fact:drift",
            event_type="policy_action_emission",
            runtime_session_id="runtime-1",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            occurred_at=10.0,
            clock_domain_id="local_monotonic",
            payload=payload,
        )

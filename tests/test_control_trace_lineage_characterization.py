from __future__ import annotations

from dataset_contracts.completeness import (
    ControlTraceLineageIssue,
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from dataset_contracts.control_trace import (
    CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1,
    ControlTraceFact,
)


def _manifest_fact(event_type: str, fact_id: str) -> ControlTraceFact:
    record_spec = CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1[event_type]
    payload = {
        field: f"{fact_id}:{field}"
        for field in record_spec.payload_fields
    }
    payload["semantic_content_hash"] = "a" * 64
    return ControlTraceFact.create(
        fact_id=fact_id,
        event_type=event_type,
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=1.0,
        clock_domain_id="local_monotonic",
        payload=payload,
    )


def _fact(
    event_type: str,
    fact_id: str,
    **payload_overrides: object,
) -> ControlTraceFact:
    record_spec = CONTROL_TRACE_RECORD_SPEC_BY_EVENT_V1[event_type]
    payload: dict[str, object] = {
        field: f"{fact_id}:{field}"
        for field in record_spec.payload_fields
    }
    payload.update(payload_overrides)
    return ControlTraceFact.create(
        fact_id=fact_id,
        event_type=event_type,
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=1.0,
        clock_domain_id="local_monotonic",
        payload=payload,
    )


def test_planned_profile_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("authoritative_real_v1")

    assert ControlTraceLineageIssue.__module__ == "dataset_contracts.completeness"
    assert evaluate_control_trace_lineage((), profile=profile) == (
        ControlTraceLineageIssue(
            code="profile_not_implemented",
            root_fact_id="authoritative_real_v1",
            detail=(
                "The profile is classified but has no accepted runtime fixture yet."
            ),
        ),
    )


def test_profile_and_manifest_preflight_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("authoritative_real_v1")
    facts = (
        _manifest_fact("execution_manifest", "execution:1"),
        _manifest_fact("station_run_manifest", "station:1"),
        _manifest_fact("execution_manifest", "execution:2"),
        _manifest_fact("station_run_manifest", "station:2"),
        _manifest_fact("execution_manifest", "execution:3"),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="profile_not_implemented",
            root_fact_id="authoritative_real_v1",
            detail=(
                "The profile is classified but has no accepted runtime fixture yet."
            ),
        ),
        ControlTraceLineageIssue(
            code="multiple_execution_manifests",
            root_fact_id="execution:2",
            detail=(
                "One assessed control domain may bind only one execution manifest."
            ),
            related_fact_ids=("execution:1",),
        ),
        ControlTraceLineageIssue(
            code="multiple_execution_manifests",
            root_fact_id="execution:3",
            detail=(
                "One assessed control domain may bind only one execution manifest."
            ),
            related_fact_ids=("execution:1",),
        ),
        ControlTraceLineageIssue(
            code="multiple_station_run_manifests",
            root_fact_id="station:2",
            detail="One assessed Station runtime may bind only one run manifest.",
            related_fact_ids=("station:1",),
        ),
    )

def test_station_transition_and_barrier_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_station_v1")
    state_0 = {"semantic_content_hash": "0" * 64}
    state_1 = {"semantic_content_hash": "1" * 64}
    state_2 = {"semantic_content_hash": "2" * 64}
    state_3 = {"semantic_content_hash": "3" * 64}
    state_4 = {"semantic_content_hash": "4" * 64}
    unrelated_state = {"semantic_content_hash": "f" * 64}
    facts = (
        _fact(
            "station_run_manifest",
            "station:manifest",
            semantic_content_hash="a" * 64,
        ),
        _fact(
            "station_intent_transition",
            "station:intent:1",
            intent_id="intent:1",
            intent_kind="request_motion_authority",
            subject_id="subject:1",
            admission_sequence=1,
            state_before=state_0,
            state_after=state_1,
            manifest_hash="b" * 64,
        ),
        _fact(
            "station_intent_transition",
            "station:intent:2",
            intent_id="intent:2",
            intent_kind="request_motion_authority",
            subject_id="subject:2",
            admission_sequence=1,
            state_before=unrelated_state,
            state_after=state_2,
            manifest_hash="a" * 64,
        ),
        _fact(
            "station_operation_transition",
            "station:operation:missing",
            operation_result_id="operation:missing",
            source_intent_id="intent:missing",
            operation_kind="request_motion_authority",
            subject_id="subject:missing",
            admission_sequence=2,
            state_before=state_2,
            state_after=state_3,
            succeeded=True,
            result_status="motion_authority_ready",
        ),
        _fact(
            "station_operation_transition",
            "station:operation:mismatch",
            operation_result_id="operation:mismatch",
            source_intent_id="intent:1",
            operation_kind="stop",
            subject_id="subject:mismatch",
            admission_sequence=3,
            state_before=state_3,
            state_after=state_4,
            succeeded=True,
            result_status="stopped",
        ),
        _fact(
            "station_publication_barrier_transition",
            "station:barrier",
            enabled=True,
            causal_station_fact_id="station:missing-cause",
            authority_operation_result_id="operation:mismatch",
            authority_evidence_refs=(),
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="station_barrier_missing_authority_result",
            root_fact_id="station:barrier",
            detail=(
                "Enabled publication must resolve one evidenced authority result."
            ),
            related_fact_ids=("operation:mismatch",),
        ),
        ControlTraceLineageIssue(
            code="station_barrier_missing_cause",
            root_fact_id="station:barrier",
            detail=(
                "Publication barrier transition must resolve its Station cause."
            ),
            related_fact_ids=("station:missing-cause",),
        ),
        ControlTraceLineageIssue(
            code="station_intent_manifest_mismatch",
            root_fact_id="station:intent:1",
            detail="Station intent references another run manifest.",
            related_fact_ids=("station:manifest",),
        ),
        ControlTraceLineageIssue(
            code="station_admission_sequence_duplicate",
            root_fact_id="station:intent:2",
            detail="Station intent/result admission sequence must be unique.",
        ),
        ControlTraceLineageIssue(
            code="station_state_lineage_discontinuous",
            root_fact_id="station:intent:2",
            detail=(
                "Station transition state_before does not match the prior state_after."
            ),
            related_fact_ids=("station:intent:1",),
        ),
        ControlTraceLineageIssue(
            code="station_operation_intent_binding_mismatch",
            root_fact_id="station:operation:mismatch",
            detail=(
                "Station operation kind/subject differs from its source intent."
            ),
            related_fact_ids=("intent:1",),
        ),
        ControlTraceLineageIssue(
            code="station_operation_missing_intent",
            root_fact_id="station:operation:missing",
            detail="Station operation result must resolve one source intent.",
            related_fact_ids=("intent:missing",),
        ),
    )


def test_episode_membership_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_station_v1")
    facts = (
        _fact(
            "episode_membership_transition",
            "membership:1",
            membership_epoch=1,
            episode_uuid="episode:1",
            transition="bind",
        ),
        _fact(
            "episode_membership_transition",
            "membership:2",
            membership_epoch=1,
            episode_uuid="episode:2",
            transition="bind",
        ),
        _fact(
            "episode_membership_transition",
            "membership:3",
            membership_epoch=0,
            episode_uuid="episode:other",
            transition="unbind",
        ),
        _fact(
            "episode_membership_transition",
            "membership:4",
            membership_epoch=2,
            episode_uuid="episode:4",
            transition="replace",
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="membership_bind_while_active",
            root_fact_id="membership:2",
            detail=(
                "A second episode was bound before the current membership was released."
            ),
        ),
        ControlTraceLineageIssue(
            code="membership_epoch_not_increasing",
            root_fact_id="membership:2",
            detail="Episode membership epochs must strictly increase.",
        ),
        ControlTraceLineageIssue(
            code="membership_epoch_not_increasing",
            root_fact_id="membership:3",
            detail="Episode membership epochs must strictly increase.",
        ),
        ControlTraceLineageIssue(
            code="membership_unbind_mismatch",
            root_fact_id="membership:3",
            detail="Episode unbind does not match the active membership.",
        ),
        ControlTraceLineageIssue(
            code="membership_transition_invalid",
            root_fact_id="membership:4",
            detail="Unsupported membership transition 'replace'.",
        ),
    )


def test_context_lineage_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_station_v1")
    facts = (
        _fact(
            "policy_context_admission",
            "context:admission:policy",
            accepted=True,
            context_id="context:missing-policy",
            context_semantic_content_hash="1" * 64,
        ),
        _fact(
            "composition_context_admission",
            "context:admission:composition",
            accepted=True,
            context_id="context:missing-composition",
            context_semantic_content_hash="2" * 64,
        ),
        _fact(
            "policy_context",
            "context:policy",
            context_id="context:policy-present",
            context_semantic_content_hash="3" * 64,
            observation_cut_id="observation:missing",
            parent_context_ids=("parent:2", "parent:1"),
        ),
        _fact(
            "composition_context",
            "context:composition",
            context_id="context:composition-present",
            context_semantic_content_hash="4" * 64,
            observation_cut_id=None,
            parent_context_ids=("parent:composition",),
        ),
        _fact(
            "policy_context_snapshot",
            "context:snapshot:policy",
            selected_context_id_by_channel={
                "instruction": "context:snapshot-missing-policy"
            },
            selected_context_hash_by_channel={"instruction": "5" * 64},
        ),
        _fact(
            "composition_context_snapshot",
            "context:snapshot:composition",
            selected_context_id_by_channel={
                "guidance": "context:snapshot-missing-composition"
            },
            selected_context_hash_by_channel={"guidance": "6" * 64},
        ),
    )
    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="composition_context_admission_missing_context",
            root_fact_id="context:admission:composition",
            detail="Accepted context admission must resolve its exact context.",
            related_fact_ids=("context:missing-composition",),
        ),
        ControlTraceLineageIssue(
            code="policy_context_admission_missing_context",
            root_fact_id="context:admission:policy",
            detail="Accepted context admission must resolve its exact context.",
            related_fact_ids=("context:missing-policy",),
        ),
        ControlTraceLineageIssue(
            code="composition_context_missing_parent",
            root_fact_id="context:composition",
            detail="Context parent dependency is absent.",
            related_fact_ids=("parent:composition",),
        ),
        ControlTraceLineageIssue(
            code="policy_context_missing_observation_cut",
            root_fact_id="context:policy",
            detail="Context observation-cut lineage must resolve when declared.",
            related_fact_ids=("observation:missing",),
        ),
        ControlTraceLineageIssue(
            code="policy_context_missing_parent",
            root_fact_id="context:policy",
            detail="Context parent dependency is absent.",
            related_fact_ids=("parent:1",),
        ),
        ControlTraceLineageIssue(
            code="policy_context_missing_parent",
            root_fact_id="context:policy",
            detail="Context parent dependency is absent.",
            related_fact_ids=("parent:2",),
        ),
        ControlTraceLineageIssue(
            code="composition_context_snapshot_missing_context",
            root_fact_id="context:snapshot:composition",
            detail="Context snapshot must resolve each selected context and hash.",
            related_fact_ids=("context:snapshot-missing-composition",),
        ),
        ControlTraceLineageIssue(
            code="policy_context_snapshot_missing_context",
            root_fact_id="context:snapshot:policy",
            detail="Context snapshot must resolve each selected context and hash.",
            related_fact_ids=("context:snapshot-missing-policy",),
        ),
    )


def test_composition_source_eligibility_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_station_v1")
    facts = (
        _fact(
            "composition_source_eligibility_admission",
            "eligibility:admission",
            accepted=True,
            source_eligibility_state_id="eligibility:state-missing",
            state_semantic_content_hash="1" * 64,
        ),
        _fact(
            "composition_source_eligibility_snapshot",
            "eligibility:snapshot",
            source_eligibility_snapshot_id="eligibility:snapshot-id",
            source_eligibility_spec_hash="2" * 64,
            selected_state_ids_by_role={"human": "eligibility:selected-missing"},
            selected_state_hashes_by_role={"human": "3" * 64},
        ),
        _fact(
            "composition_source_eligibility_binding",
            "eligibility:binding",
            composition_input_snapshot_id="composition:input-missing",
            composition_input_snapshot_hash="4" * 64,
            source_eligibility_snapshot_id="eligibility:snapshot-missing",
            source_eligibility_snapshot_hash="5" * 64,
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="composition_source_eligibility_admission_missing_state",
            root_fact_id="eligibility:admission",
            detail="Accepted eligibility admission must resolve its exact state.",
            related_fact_ids=("eligibility:state-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_source_eligibility_binding_missing_input_snapshot",
            root_fact_id="eligibility:binding",
            detail="Eligibility binding must resolve the input snapshot.",
            related_fact_ids=("composition:input-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_source_eligibility_binding_missing_snapshot",
            root_fact_id="eligibility:binding",
            detail="Eligibility binding must resolve the eligibility snapshot.",
            related_fact_ids=("eligibility:snapshot-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_source_eligibility_snapshot_missing_spec",
            root_fact_id="eligibility:snapshot",
            detail="Eligibility snapshot must resolve its exact spec.",
            related_fact_ids=("2" * 64,),
        ),
        ControlTraceLineageIssue(
            code="composition_source_eligibility_snapshot_missing_state",
            root_fact_id="eligibility:snapshot",
            detail="Eligibility snapshot must resolve each selected state.",
            related_fact_ids=("eligibility:selected-missing",),
        ),
    )


def test_component_schema_and_compound_authority_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "component_action_schema",
            "schema:component",
            payload_schema_hash_by_component={"arm": "payload:missing"},
            component_catalog={
                "components": (
                    {
                        "component_id": "arm",
                        "accepted_payload_schema_hashes": (
                            "payload:other",
                        ),
                    },
                ),
            },
        ),
        _fact(
            "composition_spec",
            "spec:formal",
            algebra="formal_component_strategy_v2",
            action_schema_hash="schema:output",
            input_action_schema_hash_by_role={"human": "schema:input"},
            semantic_content_hash="spec:formal:hash",
        ),
        _fact(
            "formal_component_composition_binding",
            "binding:formal",
            composition_spec_hash="spec:missing",
            component_rule_set_hash="rules:missing",
            formal_strategy_spec_hash="strategy:missing",
        ),
        _fact(
            "conservative_compound_authority_profile",
            "profile:compound",
            composition_spec_hash="spec:missing",
            component_rule_set_hash="rules:missing",
            formal_strategy_spec_hash="strategy:missing",
            semantic_content_hash="profile:compound:hash",
        ),
        _fact(
            "compound_authority_failure",
            "failure:compound",
            profile_hash="profile:missing",
            source_event_ids=("source:missing:a", "source:missing:b"),
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="formal_component_binding_missing_rule_set",
            root_fact_id="binding:formal",
            detail=(
                "Formal component binding must resolve one typed rule set."
            ),
            related_fact_ids=("rules:missing",),
        ),
        ControlTraceLineageIssue(
            code="formal_component_binding_missing_spec",
            root_fact_id="binding:formal",
            detail=(
                "Formal component binding must resolve one v2 CompositionSpec."
            ),
            related_fact_ids=("spec:missing",),
        ),
        ControlTraceLineageIssue(
            code="formal_component_binding_missing_strategy_spec",
            root_fact_id="binding:formal",
            detail=(
                "Formal component binding must resolve one strategy bound "
                "to the same CompositionSpec."
            ),
            related_fact_ids=("strategy:missing",),
        ),
        ControlTraceLineageIssue(
            code="compound_authority_failure_missing_profile",
            root_fact_id="failure:compound",
            detail=(
                "Compound authority failure must resolve its exact profile."
            ),
            related_fact_ids=("profile:missing",),
        ),
        ControlTraceLineageIssue(
            code="compound_authority_failure_missing_source",
            root_fact_id="failure:compound",
            detail=(
                "Compound authority failure sources must resolve in this trace."
            ),
            related_fact_ids=("source:missing:a", "source:missing:b"),
        ),
        ControlTraceLineageIssue(
            code="compound_authority_profile_missing_composition_spec",
            root_fact_id="profile:compound",
            detail=(
                "Compound authority profile must resolve one CompositionSpec."
            ),
            related_fact_ids=("spec:missing",),
        ),
        ControlTraceLineageIssue(
            code="compound_authority_profile_missing_rule_set",
            root_fact_id="profile:compound",
            detail=(
                "Compound authority profile must resolve one component rule set."
            ),
            related_fact_ids=("rules:missing",),
        ),
        ControlTraceLineageIssue(
            code="compound_authority_profile_missing_strategy_spec",
            root_fact_id="profile:compound",
            detail=(
                "Compound authority profile must resolve one formal strategy spec."
            ),
            related_fact_ids=("strategy:missing",),
        ),
        ControlTraceLineageIssue(
            code="component_action_schema_missing_payload_schema",
            root_fact_id="schema:component",
            detail=(
                "Component action schema must resolve each payload schema "
                "definition."
            ),
            related_fact_ids=("payload:missing",),
        ),
        ControlTraceLineageIssue(
            code="component_action_schema_payload_not_accepted",
            root_fact_id="schema:component",
            detail=(
                "Embedded component catalog does not accept the bound payload "
                "schema."
            ),
            related_fact_ids=("arm", "payload:missing"),
        ),
        ControlTraceLineageIssue(
            code="formal_component_spec_missing_action_schema",
            root_fact_id="spec:formal",
            detail=(
                "Role-specific formal composition must resolve every input "
                "and output action schema definition."
            ),
            related_fact_ids=("schema:input",),
        ),
        ControlTraceLineageIssue(
            code="formal_component_spec_missing_action_schema",
            root_fact_id="spec:formal",
            detail=(
                "Role-specific formal composition must resolve every input "
                "and output action schema definition."
            ),
            related_fact_ids=("schema:output",),
        ),
        ControlTraceLineageIssue(
            code="formal_component_spec_missing_binding",
            root_fact_id="spec:formal",
            detail=(
                "Formal component CompositionSpec must resolve exactly one "
                "rule-set/strategy binding."
            ),
            related_fact_ids=("spec:formal:hash",),
        ),
    )


def test_action_adaptation_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "action_adaptation_request",
            "adapt:request",
            authoritative_decision_id="decision:missing",
        ),
        _fact(
            "adapted_action_candidate",
            "adapt:candidate",
            request_id="request:missing",
        ),
        _fact(
            "action_adaptation_receipt",
            "adapt:receipt",
            request_id="request:missing:receipt",
            capability_assessment_id="assessment:missing",
            accepted=True,
            adapted_candidate_id="candidate:missing",
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="adapted_candidate_missing_request",
            root_fact_id="adapt:candidate",
            detail="Adapted candidate must resolve its exact request.",
            related_fact_ids=("request:missing",),
        ),
        ControlTraceLineageIssue(
            code="action_adaptation_receipt_missing_assessment",
            root_fact_id="adapt:receipt",
            detail="Adaptation receipt must resolve its exact assessment.",
            related_fact_ids=("assessment:missing",),
        ),
        ControlTraceLineageIssue(
            code="action_adaptation_receipt_missing_candidate",
            root_fact_id="adapt:receipt",
            detail="Accepted adaptation must resolve one candidate.",
            related_fact_ids=("candidate:missing",),
        ),
        ControlTraceLineageIssue(
            code="action_adaptation_receipt_missing_request",
            root_fact_id="adapt:receipt",
            detail="Adaptation receipt must resolve its exact request.",
            related_fact_ids=("request:missing:receipt",),
        ),
        ControlTraceLineageIssue(
            code="action_adaptation_missing_decision",
            root_fact_id="adapt:request",
            detail="Action adaptation must resolve one authoritative decision.",
            related_fact_ids=("decision:missing",),
        ),
    )


def test_execution_directive_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "arbitration_publication_attempt",
            "publication:accepted",
            publication_attempt_id="publication:accepted",
            accepted=True,
        ),
        _fact(
            "execution_directive",
            "directive:missing",
            publication_attempt_id="publication:missing",
            decision_id="decision:missing",
        ),
        _fact(
            "execution_directive",
            "directive:nonselected",
            publication_attempt_id="publication:accepted",
            decision_id=None,
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="execution_directive_missing_decision",
            root_fact_id="directive:missing",
            detail=(
                "Selected ExecutionDirective must resolve one authoritative "
                "decision."
            ),
            related_fact_ids=("decision:missing",),
        ),
        ControlTraceLineageIssue(
            code="execution_directive_missing_publication_attempt",
            root_fact_id="directive:missing",
            detail="ExecutionDirective must resolve one publication attempt.",
            related_fact_ids=("publication:missing",),
        ),
        ControlTraceLineageIssue(
            code="nonselected_directive_has_accepted_publication",
            root_fact_id="directive:nonselected",
            detail=(
                "A directive without decision_id cannot reference accepted "
                "publication."
            ),
            related_fact_ids=("publication:accepted",),
        ),
    )


def test_adapted_directive_and_shared_safety_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    decision_hash = "d" * 64
    request_hash = "r" * 64
    assessment_hash = "a" * 64
    receipt_hash = "e" * 64
    facts = (
        _fact(
            "arbitration_decision",
            "decision:1",
            decision_id="decision:1",
            semantic_content_hash=decision_hash,
            result_id="result:1",
            selected_action_hash="action:1",
            selected_action={"action_schema_hash": "schema:1"},
        ),
        _fact(
            "action_adaptation_request",
            "request:1",
            request_id="request:1",
            semantic_content_hash=request_hash,
            authoritative_decision_id="decision:1",
            authoritative_decision_hash=decision_hash,
            arbitration_result_id="result:1",
            candidate_action_hash="action:1",
            action_schema_hash="schema:1",
        ),
        _fact(
            "target_capability_assessment",
            "assessment:1",
            assessment_id="assessment:1",
            semantic_content_hash=assessment_hash,
        ),
        _fact(
            "action_adaptation_receipt",
            "receipt:rejected",
            receipt_id="receipt:rejected",
            semantic_content_hash=receipt_hash,
            request_id="request:1",
            request_hash=request_hash,
            capability_assessment_id="assessment:1",
            capability_assessment_hash=assessment_hash,
            accepted=False,
            adapted_candidate_id=None,
        ),
        _fact(
            "arbitration_publication_attempt",
            "publication:1",
            publication_attempt_id="publication:1",
            accepted=True,
        ),
        _fact(
            "execution_directive",
            "directive:missing-receipt",
            publication_attempt_id="publication:1",
            decision_id="decision:1",
            action_adaptation_receipt_id="receipt:missing",
        ),
        _fact(
            "execution_directive",
            "directive:fallback-mismatch",
            publication_attempt_id="publication:1",
            decision_id="decision:1",
            action_adaptation_receipt_id="receipt:rejected",
            action_adaptation_receipt_hash=receipt_hash,
            action_adaptation_fallback=None,
        ),
        _fact(
            "shared_safety_result",
            "safety:missing-candidate",
            input_action_id="candidate:missing",
            input_action_hash="candidate:hash",
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="execution_directive_adaptation_fallback_mismatch",
            root_fact_id="directive:fallback-mismatch",
            detail="Adaptation fallback must exist exactly for rejection.",
            related_fact_ids=("receipt:rejected",),
        ),
        ControlTraceLineageIssue(
            code="execution_directive_missing_adaptation_receipt",
            root_fact_id="directive:missing-receipt",
            detail="Selected directive must resolve adaptation receipt.",
            related_fact_ids=("receipt:missing",),
        ),
        ControlTraceLineageIssue(
            code="shared_safety_missing_adapted_candidate",
            root_fact_id="safety:missing-candidate",
            detail="Shared safety must resolve its exact adapted input.",
            related_fact_ids=("candidate:missing",),
        ),
    )


def test_target_execution_missing_submission_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile(
        "deployable_sim_immediate_component_policy_human_composer_v1"
    )
    facts = (
        _fact(
            "target_execution_request",
            "handoff:request:missing-submission",
            request_id="request:missing-submission",
            semantic_content_hash="r" * 64,
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="component_execution_request_missing_submission",
            root_fact_id="handoff:request:missing-submission",
            detail=(
                "Each component execution request needs exactly one target "
                "submission receipt."
            ),
        ),
    )


def test_scheduled_target_execution_binding_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_component_policy_human_composer_v1"
    )
    request_hash = "r" * 64
    submission_hash = "s" * 64
    target_hash = "t" * 64
    admission_hash = "a" * 64
    facts = (
        _fact(
            "target_execution_request",
            "handoff:request",
            request_id="request:1",
            semantic_content_hash=request_hash,
        ),
        _fact(
            "target_execution_submission",
            "handoff:submission",
            target_execution_request_id="request:1",
            target_execution_request_hash="wrong-request-hash",
            accepted_for_processing=True,
            semantic_content_hash=submission_hash,
        ),
        _fact(
            "target_execution_completion",
            "handoff:completion",
            target_execution_request_id="request:1",
            target_execution_request_hash="wrong-request-hash",
            submission_receipt_id="submission:wrong",
            submission_receipt_hash="wrong-submission-hash",
            status="resolved",
            target_id="target:1",
            target_hash="wrong-target-hash",
        ),
        _fact(
            "component_execution_target",
            "handoff:target",
            target_execution_request_id="request:1",
            target_execution_request_hash="wrong-request-hash",
            submission_receipt_id="submission:wrong",
            submission_receipt_hash="wrong-submission-hash",
            target_id="target:1",
            semantic_content_hash=target_hash,
        ),
        _fact(
            "component_execution_target_admission",
            "handoff:admission",
            target_id="target:1",
            target_hash="wrong-target-hash",
            target_execution_request_id="request:wrong",
            target_execution_request_hash="wrong-request-hash",
            accepted=True,
            semantic_content_hash=admission_hash,
        ),
        _fact(
            "component_execution_target_activation",
            "handoff:activation",
            target_id="target:1",
            target_hash="wrong-target-hash",
            target_execution_request_id="request:wrong",
            target_execution_request_hash="wrong-request-hash",
            target_admission_receipt_id="admission:wrong",
            target_admission_receipt_hash="wrong-admission-hash",
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="component_execution_target_activation_binding_mismatch",
            root_fact_id="handoff:activation",
            detail=(
                "Target activation must bind the exact request, target, and "
                "admission receipt."
            ),
            related_fact_ids=(
                "handoff:request",
                "handoff:target",
                "handoff:admission",
            ),
        ),
        ControlTraceLineageIssue(
            code="component_execution_target_admission_binding_mismatch",
            root_fact_id="handoff:admission",
            detail=(
                "Target admission must bind the exact component target and "
                "execution request."
            ),
            related_fact_ids=("handoff:target", "handoff:request"),
        ),
        ControlTraceLineageIssue(
            code="component_execution_completion_binding_mismatch",
            root_fact_id="handoff:completion",
            detail=(
                "Execution completion must bind the exact request, submission, "
                "and component target."
            ),
            related_fact_ids=(
                "handoff:request",
                "handoff:submission",
                "handoff:target",
            ),
        ),
        ControlTraceLineageIssue(
            code="component_execution_submission_request_hash_mismatch",
            root_fact_id="handoff:submission",
            detail=(
                "Target submission receipt must bind the exact execution "
                "request hash."
            ),
            related_fact_ids=("handoff:request",),
        ),
        ControlTraceLineageIssue(
            code="component_execution_target_request_binding_mismatch",
            root_fact_id="handoff:target",
            detail=(
                "Component target must bind the exact request and submission "
                "receipt."
            ),
            related_fact_ids=("handoff:request", "handoff:submission"),
        ),
    )


def test_execution_cancellation_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "execution_cancellation",
            "cancellation:request",
            cancellation_id="cancellation:1",
            source_event_ids=("source:missing",),
            effective_scope="control_domain",
            previous_execution_epoch=3,
            execution_epoch=3,
            required_ack_levels=("software_queue", "sdk_request"),
            requests_backend_hold=True,
        ),
        _fact(
            "execution_cancellation_ack",
            "cancellation:ack:rejected",
            cancellation_id="cancellation:1",
            acknowledgement_level="software_queue",
            accepted=False,
        ),
        _fact(
            "execution_cancellation_ack",
            "cancellation:ack:orphan",
            cancellation_id="cancellation:missing",
        ),
    )

    assert evaluate_control_trace_lineage(facts, profile=profile) == (
        ControlTraceLineageIssue(
            code="execution_cancellation_ack_missing_request",
            root_fact_id="cancellation:ack:orphan",
            detail=(
                "Cancellation acknowledgement must resolve one cancellation."
            ),
            related_fact_ids=("cancellation:missing",),
        ),
        ControlTraceLineageIssue(
            code="execution_cancellation_ack_rejected",
            root_fact_id="cancellation:request",
            detail="A required cancellation acknowledgement was rejected.",
            related_fact_ids=("cancellation:ack:rejected",),
        ),
        ControlTraceLineageIssue(
            code="execution_cancellation_epoch_not_advanced",
            root_fact_id="cancellation:request",
            detail=(
                "Domain cancellation must advance execution epoch exactly once."
            ),
        ),
        ControlTraceLineageIssue(
            code="execution_cancellation_missing_backend_hold_permit",
            root_fact_id="cancellation:request",
            detail=(
                "Domain hold requires accepted one-shot safety-control permits."
            ),
        ),
        ControlTraceLineageIssue(
            code="execution_cancellation_missing_required_ack",
            root_fact_id="cancellation:request",
            detail=(
                "Cancellation must resolve exactly one required acknowledgement."
            ),
        ),
        ControlTraceLineageIssue(
            code="execution_cancellation_missing_source_event",
            root_fact_id="cancellation:request",
            detail="Execution cancellation source lineage is absent.",
            related_fact_ids=("source:missing",),
        ),
    )


def test_composition_session_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    station_rearm = _fact(
        "station_operation_transition",
        "station:rearm",
        operation_result_id="station:rearm",
        source_intent_id="intent:missing",
        operation_kind="rearm_composition_session",
        subject_id="session:rearm-transition",
        succeeded=True,
        admission_sequence=1,
        evidence_refs=(),
        state_before={"semantic_content_hash": "0" * 64},
        state_after={"semantic_content_hash": "1" * 64},
    )
    facts = (
        _fact(
            "composition_session_spec",
            "session:spec",
            operation_spec_id="session:missing-numerical",
            composition_spec_hash="numerical:missing",
        ),
        _fact(
            "composition_session_rearm_spec",
            "rearm:spec",
            rearm_spec_id="rearm:spec-id",
            logical_operation_spec_id="session:missing-logical",
            logical_operation_spec_hash="logical:missing-hash",
        ),
        _fact(
            "composition_session_rearm_receipt",
            "rearm:receipt",
            rearm_spec_id="rearm:missing-spec",
            rearm_spec_hash="rearm:missing-hash",
            prior_session_epoch=3,
            next_session_epoch=3,
            prior_composer_producer_epoch=5,
            composer_producer_epoch=5,
            prior_authority_epoch=5,
            authority_epoch=5,
            prior_execution_epoch=5,
            execution_epoch=5,
        ),
        station_rearm,
        _fact(
            "composition_session_transition",
            "transition:bad:1",
            operation_spec_id="session:bad",
            operation_spec_hash="session:bad-hash",
            session_epoch=2,
            transition="hold",
            source_fact_id="source:missing:1",
            source_fact_hash="source:missing-hash:1",
            state_before_hash="state:a",
            state_after_hash="state:b",
        ),
        _fact(
            "composition_session_transition",
            "transition:bad:2",
            operation_spec_id="session:bad",
            operation_spec_hash="session:bad-hash",
            session_epoch=2,
            transition="fault",
            source_fact_id="source:missing:2",
            source_fact_hash="source:missing-hash:2",
            state_before_hash="state:wrong",
            state_after_hash="state:c",
        ),
        _fact(
            "composition_session_transition",
            "transition:bad:3",
            operation_spec_id="session:bad",
            operation_spec_hash="session:bad-hash",
            session_epoch=2,
            transition="stop",
            source_fact_id="source:missing:3",
            source_fact_hash="source:missing-hash:3",
            state_before_hash="state:c",
            state_after_hash="state:d",
        ),
        _fact(
            "composition_session_transition",
            "transition:bad:4",
            operation_spec_id="session:bad",
            operation_spec_hash="session:bad-hash",
            session_epoch=4,
            transition="start",
            source_fact_id="source:missing:4",
            source_fact_hash="source:missing-hash:4",
            state_before_hash="state:e",
            state_after_hash="state:f",
        ),
        _fact(
            "composition_session_transition",
            "transition:rearm",
            operation_spec_id="session:rearm-transition",
            operation_spec_hash="session:rearm-transition-hash",
            session_epoch=1,
            transition="supersede",
            source_fact_id=station_rearm.fact_id,
            source_fact_hash=station_rearm.semantic_content_hash,
            state_before_hash="state:r0",
            state_after_hash="state:r1",
        ),
    )

    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.code.startswith("composition_session_")
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="composition_session_rearm_epoch_not_advanced",
            root_fact_id="rearm:receipt",
            detail=(
                "Composition rearm must advance activation, producer, "
                "authority, and execution epochs."
            ),
        ),
        ControlTraceLineageIssue(
            code="composition_session_rearm_receipt_missing_spec",
            root_fact_id="rearm:receipt",
            detail="Composition rearm receipt must resolve its exact rearm spec.",
            related_fact_ids=("rearm:missing-spec",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_rearm_missing_logical_spec",
            root_fact_id="rearm:spec",
            detail="Composition rearm spec must resolve its exact logical session.",
            related_fact_ids=("session:missing-logical",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_missing_numerical_spec",
            root_fact_id="session:spec",
            detail="Composition session must resolve its exact numerical spec.",
            related_fact_ids=("numerical:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_epoch_sequence_invalid",
            root_fact_id="transition:bad:1",
            detail="Composition activation epochs must be ordered and contiguous.",
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_missing_start",
            root_fact_id="transition:bad:1",
            detail=(
                "Each composition activation must begin with start or start_failed."
            ),
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_source_fact",
            root_fact_id="transition:bad:1",
            detail=(
                "Composition session transition must resolve its exact source fact."
            ),
            related_fact_ids=("source:missing:1",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_spec",
            root_fact_id="transition:bad:1",
            detail="Composition session transition must resolve its exact spec.",
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_state_chain_mismatch",
            root_fact_id="transition:bad:2",
            detail=(
                "State hashes must form one exact chain within an activation epoch."
            ),
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_source_fact",
            root_fact_id="transition:bad:2",
            detail=(
                "Composition session transition must resolve its exact source fact."
            ),
            related_fact_ids=("source:missing:2",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_spec",
            root_fact_id="transition:bad:2",
            detail="Composition session transition must resolve its exact spec.",
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_epoch_not_superseded",
            root_fact_id="transition:bad:3",
            detail=(
                "A later activation requires the prior held activation to be "
                "superseded."
            ),
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_terminal_source_invalid",
            root_fact_id="transition:bad:3",
            detail=(
                "Session activation transitions require the matching successful "
                "Station operation."
            ),
            related_fact_ids=("source:missing:3",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_after_terminal",
            root_fact_id="transition:bad:3",
            detail=(
                "A composition activation cannot transition after its terminal event."
            ),
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_source_fact",
            root_fact_id="transition:bad:3",
            detail=(
                "Composition session transition must resolve its exact source fact."
            ),
            related_fact_ids=("source:missing:3",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_spec",
            root_fact_id="transition:bad:3",
            detail="Composition session transition must resolve its exact spec.",
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_terminal_source_invalid",
            root_fact_id="transition:bad:4",
            detail=(
                "Session activation transitions require the matching successful "
                "Station operation."
            ),
            related_fact_ids=("source:missing:4",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_source_fact",
            root_fact_id="transition:bad:4",
            detail=(
                "Composition session transition must resolve its exact source fact."
            ),
            related_fact_ids=("source:missing:4",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_spec",
            root_fact_id="transition:bad:4",
            detail="Composition session transition must resolve its exact spec.",
            related_fact_ids=("session:bad",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_missing_start",
            root_fact_id="transition:rearm",
            detail=(
                "Each composition activation must begin with start or start_failed."
            ),
            related_fact_ids=("session:rearm-transition",),
        ),
        ControlTraceLineageIssue(
            code="composition_session_rearm_transition_missing_receipt",
            root_fact_id="transition:rearm",
            detail=(
                "Rearm transitions must resolve one receipt for the exact "
                "prior/next activation epochs."
            ),
            related_fact_ids=(station_rearm.fact_id,),
        ),
        ControlTraceLineageIssue(
            code="composition_session_transition_missing_spec",
            root_fact_id="transition:rearm",
            detail="Composition session transition must resolve its exact spec.",
            related_fact_ids=("session:rearm-transition",),
        ),
    )

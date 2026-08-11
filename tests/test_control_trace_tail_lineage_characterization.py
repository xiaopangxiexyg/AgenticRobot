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


def test_composition_evaluation_and_activation_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "composition_input_admission",
            "composition:admission",
            accepted=True,
            composition_input_id="composition:input-missing",
            input_semantic_content_hash="1" * 64,
        ),
        _fact(
            "composition_input",
            "composition:input",
            composition_input_id="composition:input-id",
            input_semantic_content_hash="2" * 64,
            action={"action_schema_hash": "schema:free"},
            context_ids=("context:missing",),
            observation_cut_id="observation:missing",
        ),
        _fact(
            "composition_contributor_input",
            "composition:contributor",
            composition_input_id="composition:contributor-id",
            input_semantic_content_hash="3" * 64,
            action={"action_schema_hash": "schema:free"},
            context_ids=(),
            observation_cut_id=None,
            source_fact_ids=("source:missing", "source:extra"),
            source_fact_hashes=("4" * 64,),
        ),
        _fact(
            "composition_input_snapshot",
            "composition:snapshot",
            composition_input_snapshot_id="composition:snapshot-id",
            composition_spec_hash="spec:missing",
            action_schema_hash="schema:free",
            input_action_schema_hash_by_role={},
            selected_input_ids_by_role={"human": "composition:selected-missing"},
            selected_input_hashes_by_role={"human": "5" * 64},
        ),
        _fact(
            "composition_evaluation",
            "composition:evaluation",
            composition_evaluation_id="composition:evaluation-id",
            composition_spec_hash="spec:missing",
            composition_input_snapshot_id="composition:snapshot-missing",
            composition_input_snapshot_hash="6" * 64,
            input_ids=("composition:evaluation-input", "composition:extra"),
            input_hashes=("7" * 64,),
        ),
        _fact(
            "composition_activation_scope",
            "composition:activation",
            activation_id="composition:activation-id",
            operation_spec_id="operation:missing",
            operation_spec_hash="8" * 64,
        ),
        _fact(
            "composition_activation_binding",
            "composition:binding",
            activation_id="activation:missing",
            activation_hash="9" * 64,
            subject_kind="input",
            subject_id="subject:missing",
            subject_hash="a" * 64,
        ),
        _fact(
            "composition_authority_disposition",
            "composition:disposition",
            composition_evaluation_id="evaluation:missing",
            composition_evaluation_hash="b" * 64,
        ),
    )
    relevant_codes = {
        "composition_input_admission_missing_input",
        "composition_input_missing_context",
        "composition_input_missing_observation_cut",
        "composition_contributor_lineage_length_mismatch",
        "composition_contributor_missing_source_fact",
        "composition_snapshot_missing_spec",
        "composition_snapshot_missing_input",
        "composition_evaluation_missing_spec",
        "composition_evaluation_missing_snapshot",
        "composition_evaluation_input_lineage_length_mismatch",
        "composition_evaluation_missing_input",
        "composition_activation_missing_operation_spec",
        "composition_activation_binding_missing_scope",
        "composition_activation_binding_missing_subject",
        "composition_disposition_missing_result",
    }
    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.code in relevant_codes
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="composition_activation_missing_operation_spec",
            root_fact_id="composition:activation",
            detail="Composition activation must resolve its exact operation spec.",
            related_fact_ids=("operation:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_input_admission_missing_input",
            root_fact_id="composition:admission",
            detail="Accepted composition input admission must resolve its exact input.",
            related_fact_ids=("composition:input-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_activation_binding_missing_scope",
            root_fact_id="composition:binding",
            detail="Activation binding must resolve its exact scope.",
            related_fact_ids=("activation:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_activation_binding_missing_subject",
            root_fact_id="composition:binding",
            detail="Activation binding must resolve its exact typed subject.",
            related_fact_ids=("subject:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_contributor_lineage_length_mismatch",
            root_fact_id="composition:contributor",
            detail="Contributor source IDs and hashes must have equal length.",
        ),
        ControlTraceLineageIssue(
            code="composition_contributor_missing_source_fact",
            root_fact_id="composition:contributor",
            detail="Contributor input must resolve each exact source fact.",
            related_fact_ids=("source:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_disposition_missing_result",
            root_fact_id="composition:disposition",
            detail="Composition authority disposition must resolve its exact result.",
            related_fact_ids=("evaluation:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_evaluation_input_lineage_length_mismatch",
            root_fact_id="composition:evaluation",
            detail="Composition result input IDs and hashes must have equal length.",
        ),
        ControlTraceLineageIssue(
            code="composition_evaluation_missing_input",
            root_fact_id="composition:evaluation",
            detail="Composition result input lineage is absent or hash-mismatched.",
            related_fact_ids=("composition:evaluation-input",),
        ),
        ControlTraceLineageIssue(
            code="composition_evaluation_missing_snapshot",
            root_fact_id="composition:evaluation",
            detail="Composition result must resolve its exact input snapshot.",
            related_fact_ids=("composition:snapshot-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_evaluation_missing_spec",
            root_fact_id="composition:evaluation",
            detail="Composition result must resolve its exact numerical spec.",
            related_fact_ids=("spec:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_input_missing_context",
            root_fact_id="composition:input",
            detail="Composition input context dependency is absent.",
            related_fact_ids=("context:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_input_missing_observation_cut",
            root_fact_id="composition:input",
            detail="Composition input must resolve its producer observation cut.",
            related_fact_ids=("observation:missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_snapshot_missing_input",
            root_fact_id="composition:snapshot",
            detail="Composition snapshot must resolve each selected input and hash.",
            related_fact_ids=("composition:selected-missing",),
        ),
        ControlTraceLineageIssue(
            code="composition_snapshot_missing_spec",
            root_fact_id="composition:snapshot",
            detail="Composition snapshot must resolve its exact numerical spec.",
            related_fact_ids=("spec:missing",),
        ),
    )


def test_reference_lifecycle_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    reference_fact_ids = {
        "reference:segment",
        "reference:admission",
        "reference:anchor",
        "reference:reservation",
        "reference:generation",
        "reference:activation",
        "reference:frame",
        "reference:terminal",
    }
    facts = (
        _fact(
            "reference_segment",
            "reference:segment",
            reference_segment_id="segment:1",
            input_action_schema_hash="schema:other",
            endpoint_parents=(
                {"role": "end", "source_fact_id": "source:missing"},
                {"role": "start", "source_fact_id": "source:start"},
            ),
            semantic_content_hash="1" * 64,
        ),
        _fact(
            "reference_segment_admission",
            "reference:admission",
            admission_id="admission:1",
            reference_segment_id="segment:missing",
            reference_segment_hash="segment:wrong-hash",
            receiver_validation_id="validation:missing",
            receiver_validation_hash="validation:wrong-hash",
            accepted=True,
        ),
        _fact(
            "safety_anchor_commit_candidate",
            "reference:anchor",
            reference_segment_id="segment:missing",
            reference_segment_hash="segment:wrong-hash",
            admission_id="admission:missing",
            admission_hash="admission:wrong-hash",
            safety_result_id="safety:missing",
        ),
        _fact(
            "group_generation_reservation",
            "reference:reservation",
            reservation_id="reservation:1",
            source_segment_id="segment:missing",
            source_segment_hash="segment:wrong-hash",
            admission_id="admission:missing",
            admission_hash="admission:wrong-hash",
            active_generation_at_reservation=4,
            reserved_generation=4,
            relation_kind="successor",
            predecessor_segment_id="segment:predecessor:missing",
            supersedes_reservation_id=None,
        ),
        _fact(
            "group_generation_activation",
            "reference:generation",
            activation_id="generation:1",
            reservation_id="reservation:missing",
            source_segment_id="segment:missing",
            active_generation=4,
            execution_epoch=2,
        ),
        _fact(
            "reference_target_activation",
            "reference:activation",
            reference_segment_id="segment:missing",
            generation_activation_id="generation:missing",
            group_segment_generation=4,
            reference_sequence=8,
        ),
        _fact(
            "planner_command_frame",
            "reference:frame",
            reference_segment_id="segment:missing",
            reference_sequence=9,
        ),
        _fact(
            "reference_segment_terminal",
            "reference:terminal",
            reference_segment_id="segment:missing",
            reference_segment_hash="segment:wrong-hash",
            last_reference_frame_id="frame:missing",
        ),
    )

    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.root_fact_id in reference_fact_ids
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="reference_activation_missing_generation_activation",
            root_fact_id="reference:activation",
            detail="Reference activation must resolve its generation handoff.",
            related_fact_ids=("generation:missing",),
        ),
        ControlTraceLineageIssue(
            code="reference_activation_missing_segment",
            root_fact_id="reference:activation",
            detail="Reference target activation must resolve one segment.",
            related_fact_ids=("segment:missing",),
        ),
        ControlTraceLineageIssue(
            code="reference_admission_missing_receiver_validation",
            root_fact_id="reference:admission",
            detail=(
                "Reference admission must resolve its receiver-side validation."
            ),
            related_fact_ids=("validation:missing",),
        ),
        ControlTraceLineageIssue(
            code="reference_admission_missing_segment",
            root_fact_id="reference:admission",
            detail="Reference admission must resolve its exact segment.",
            related_fact_ids=("segment:missing",),
        ),
        ControlTraceLineageIssue(
            code="anchor_candidate_missing_admission",
            root_fact_id="reference:anchor",
            detail="Anchor candidate must resolve its exact admission.",
            related_fact_ids=("admission:missing",),
        ),
        ControlTraceLineageIssue(
            code="anchor_candidate_missing_safety_result",
            root_fact_id="reference:anchor",
            detail="Anchor candidate must resolve its shared-safety result.",
            related_fact_ids=("safety:missing",),
        ),
        ControlTraceLineageIssue(
            code="anchor_candidate_missing_segment",
            root_fact_id="reference:anchor",
            detail="Anchor candidate must resolve its exact segment.",
            related_fact_ids=("segment:missing",),
        ),
        ControlTraceLineageIssue(
            code="planner_frame_missing_segment",
            root_fact_id="reference:frame",
            detail="Planner command frame must resolve one segment.",
            related_fact_ids=("segment:missing",),
        ),
        ControlTraceLineageIssue(
            code="planner_frame_reference_sequence_mismatch",
            root_fact_id="reference:frame",
            detail=(
                "Planner frame and target activation reference sequences differ."
            ),
            related_fact_ids=("reference:activation",),
        ),
        ControlTraceLineageIssue(
            code="generation_activation_missing_reservation",
            root_fact_id="reference:generation",
            detail="Generation activation must resolve one reservation.",
            related_fact_ids=("reservation:missing",),
        ),
        ControlTraceLineageIssue(
            code="generation_reservation_missing_admission",
            root_fact_id="reference:reservation",
            detail="Generation reservation must resolve its exact admission.",
            related_fact_ids=("admission:missing",),
        ),
        ControlTraceLineageIssue(
            code="generation_reservation_missing_segment",
            root_fact_id="reference:reservation",
            detail="Generation reservation must resolve its exact segment.",
            related_fact_ids=("segment:missing",),
        ),
        ControlTraceLineageIssue(
            code="generation_reservation_not_next",
            root_fact_id="reference:reservation",
            detail="Generation reservation must reserve active generation + 1.",
        ),
        ControlTraceLineageIssue(
            code="successor_reservation_missing_predecessor",
            root_fact_id="reference:reservation",
            detail="Successor reservation must resolve its predecessor segment.",
            related_fact_ids=("segment:predecessor:missing",),
        ),
        ControlTraceLineageIssue(
            code="reference_segment_parent_order_invalid",
            root_fact_id="reference:segment",
            detail="Reference segment must preserve ordered START/END parents.",
        ),
        ControlTraceLineageIssue(
            code="reference_terminal_missing_last_frame",
            root_fact_id="reference:terminal",
            detail="Reference terminal last frame must resolve a planner frame.",
            related_fact_ids=("frame:missing",),
        ),
        ControlTraceLineageIssue(
            code="reference_terminal_missing_segment",
            root_fact_id="reference:terminal",
            detail="Reference terminal must resolve its exact segment.",
            related_fact_ids=("segment:missing",),
        ),
    )


def test_scheduled_reference_completeness_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_v1")
    segment = _fact(
        "reference_segment",
        "reference:scheduled:segment",
        reference_segment_id="segment:scheduled",
        endpoint_parents=(
            {"role": "start", "source_fact_id": "source:start"},
            {"role": "end", "source_fact_id": "safety:scheduled"},
        ),
        semantic_content_hash="2" * 64,
    )
    safety = _fact(
        "shared_safety_result",
        "safety:scheduled",
        status="safe",
        safe_action={"arm": "qpos"},
    )

    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(
            (safety, segment),
            profile=profile,
        )
        if issue.root_fact_id in {safety.fact_id, segment.fact_id}
        and (
            issue.code.startswith("scheduled_")
            or issue.code.startswith("accepted_segment_")
        )
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="scheduled_segment_missing_single_admission",
            root_fact_id="reference:scheduled:segment",
            detail=(
                "Each scheduled segment attempt needs exactly one receiver admission."
            ),
        ),
    )


def test_source_proposal_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "proposal_admission",
            "source:admission-accepted",
            proposal_id="proposal:missing",
            accepted=True,
        ),
        _fact(
            "proposal_admission",
            "source:admission-rejected",
            proposal_id="proposal:rejected",
            accepted=False,
        ),
        _fact(
            "action_proposal",
            "source:composer",
            proposal_id="proposal:composer",
            producer_kind="composer",
            composition_evaluation_id="evaluation:missing",
        ),
        _fact(
            "producer_epoch_snapshot",
            "source:epoch",
            activation_id_by_producer={"policy": "activation:missing"},
        ),
        _fact(
            "action_proposal",
            "source:human-proposal",
            proposal_id="proposal:human",
            producer_kind="human",
            representation_trace_id=None,
        ),
        _fact(
            "source_adapter_receipt",
            "source:receipt",
            source_emission_id="human:missing",
            source_validation_accepted=False,
        ),
        _fact(
            "action_proposal",
            "source:rejected-proposal",
            proposal_id="proposal:rejected",
            producer_kind="human",
            representation_trace_id=None,
        ),
    )
    relevant_codes = {
        "accepted_admission_missing_proposal",
        "rejected_admission_has_proposal",
        "composite_proposal_missing_result",
        "producer_snapshot_missing_activation",
        "human_proposal_missing_source_receipt",
        "source_receipt_missing_human_emission",
        "source_receipt_missing_validation",
    }
    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.code in relevant_codes
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="accepted_admission_missing_proposal",
            root_fact_id="source:admission-accepted",
            detail="An accepted admission must resolve to exactly one action proposal.",
            related_fact_ids=("proposal:missing",),
        ),
        ControlTraceLineageIssue(
            code="rejected_admission_has_proposal",
            root_fact_id="source:admission-rejected",
            detail="A rejected admission must not materialize an action proposal.",
            related_fact_ids=("proposal:rejected",),
        ),
        ControlTraceLineageIssue(
            code="composite_proposal_missing_result",
            root_fact_id="source:composer",
            detail="Composite proposal must resolve one composition evaluation.",
            related_fact_ids=("evaluation:missing",),
        ),
        ControlTraceLineageIssue(
            code="producer_snapshot_missing_activation",
            root_fact_id="source:epoch",
            detail="Producer epoch snapshot references an absent activation.",
            related_fact_ids=("activation:missing",),
        ),
        ControlTraceLineageIssue(
            code="human_proposal_missing_source_receipt",
            root_fact_id="source:human-proposal",
            detail="Human proposal must resolve to one typed source-adapter receipt.",
        ),
        ControlTraceLineageIssue(
            code="source_receipt_missing_human_emission",
            root_fact_id="source:receipt",
            detail="Human source receipt must resolve to one command emission.",
            related_fact_ids=("human:missing",),
        ),
        ControlTraceLineageIssue(
            code="source_receipt_missing_validation",
            root_fact_id="source:receipt",
            detail="Human source receipt must resolve to one source validation.",
            related_fact_ids=("human:missing",),
        ),
        ControlTraceLineageIssue(
            code="human_proposal_missing_source_receipt",
            root_fact_id="source:rejected-proposal",
            detail="Human proposal must resolve to one typed source-adapter receipt.",
        ),
    )


def test_replay_source_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "replay_binding_assessment",
            "replay:assessment",
            source_action_contract_id="source-contract:missing",
            source_action_contract_hash="1" * 64,
            target_runtime_contract_id="target-contract:missing",
            target_runtime_contract_hash="2" * 64,
        ),
        _fact(
            "candidate_authorization_admission",
            "replay:authorization",
            accepted=True,
            proposal_id="proposal:missing",
            binding_eligibility_id="eligibility:missing",
            evidence_id="authorization:missing",
        ),
        _fact(
            "candidate_binding_eligibility",
            "replay:eligibility",
            assessment_id="assessment:missing",
            assessment_hash="3" * 64,
            authoritative_allowed=True,
            requested_use="preview",
        ),
        _fact(
            "action_proposal",
            "replay:proposal",
            proposal_id="proposal:replay",
            producer_kind="replay",
            representation_trace_id=None,
        ),
        _fact(
            "replay_source_adapter_receipt",
            "replay:receipt",
            accepted=True,
            representation_trace_id="replay:receipt-trace",
            replay_action_record_id=None,
            binding_assessment_id="assessment:missing",
            binding_assessment_hash="4" * 64,
            binding_eligibility_id="eligibility:missing",
            binding_eligibility_hash="5" * 64,
            source_emission_id="emission:missing",
        ),
        _fact(
            "candidate_authorization_snapshot",
            "replay:snapshot",
            authorization_by_proposal_id={
                "proposal:missing": {"authorization_id": "admission:missing"}
            },
        ),
    )
    relevant_codes = {
        "candidate_authorization_missing_proposal",
        "candidate_authorization_missing_binding_eligibility",
        "candidate_authorization_missing_execution_authorization",
        "candidate_authorization_snapshot_missing_admission",
        "replay_assessment_missing_source_contract",
        "replay_assessment_missing_target_contract",
        "replay_eligibility_missing_assessment",
        "replay_eligibility_invalid_authoritative_scope",
        "replay_receipt_missing_record",
        "replay_receipt_missing_assessment",
        "replay_receipt_missing_eligibility",
        "replay_receipt_missing_passive_emission",
        "replay_proposal_missing_source_receipt",
    }
    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.code in relevant_codes
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="replay_assessment_missing_source_contract",
            root_fact_id="replay:assessment",
            detail="Replay assessment must resolve its exact source action contract.",
        ),
        ControlTraceLineageIssue(
            code="replay_assessment_missing_target_contract",
            root_fact_id="replay:assessment",
            detail="Replay assessment must resolve its exact target runtime contract.",
        ),
        ControlTraceLineageIssue(
            code="candidate_authorization_missing_binding_eligibility",
            root_fact_id="replay:authorization",
            detail="Replay candidate authorization must resolve binding eligibility.",
            related_fact_ids=("eligibility:missing",),
        ),
        ControlTraceLineageIssue(
            code="candidate_authorization_missing_execution_authorization",
            root_fact_id="replay:authorization",
            detail="Replay candidate authorization must resolve execution authorization.",
            related_fact_ids=("authorization:missing",),
        ),
        ControlTraceLineageIssue(
            code="candidate_authorization_missing_proposal",
            root_fact_id="replay:authorization",
            detail="Accepted candidate authorization must resolve its proposal.",
            related_fact_ids=("proposal:missing",),
        ),
        ControlTraceLineageIssue(
            code="replay_eligibility_invalid_authoritative_scope",
            root_fact_id="replay:eligibility",
            detail="Only execution-candidate eligibility may be authoritative-allowed.",
        ),
        ControlTraceLineageIssue(
            code="replay_eligibility_missing_assessment",
            root_fact_id="replay:eligibility",
            detail="Replay candidate eligibility must resolve its exact assessment.",
        ),
        ControlTraceLineageIssue(
            code="replay_proposal_missing_source_receipt",
            root_fact_id="replay:proposal",
            detail="Replay proposal must resolve to one typed replay adapter receipt.",
        ),
        ControlTraceLineageIssue(
            code="replay_receipt_missing_assessment",
            root_fact_id="replay:receipt",
            detail="Replay receipt must resolve its exact binding assessment.",
        ),
        ControlTraceLineageIssue(
            code="replay_receipt_missing_eligibility",
            root_fact_id="replay:receipt",
            detail="Replay receipt must resolve its exact candidate eligibility.",
        ),
        ControlTraceLineageIssue(
            code="replay_receipt_missing_passive_emission",
            root_fact_id="replay:receipt",
            detail="Replay receipt must reference its unique passive source emission.",
            related_fact_ids=("emission:missing",),
        ),
        ControlTraceLineageIssue(
            code="replay_receipt_missing_record",
            root_fact_id="replay:receipt",
            detail="Accepted replay adapter receipt must resolve to one replay action record.",
        ),
        ControlTraceLineageIssue(
            code="candidate_authorization_snapshot_missing_admission",
            root_fact_id="replay:snapshot",
            detail="Frozen candidate authorization must resolve one admission.",
            related_fact_ids=("admission:missing",),
        ),
    )


def test_arbitration_shadow_issue_tuple_is_stable() -> None:
    profile = trace_requirement_profile("plain_dry_run_v1")
    facts = (
        _fact(
            "arbitration_computation",
            "arbitration:computation",
            arbitration_input_snapshot_id="input:missing",
        ),
        _fact(
            "arbitration_evaluation",
            "arbitration:evaluation",
            computation_result_id="computation:missing",
            arbitration_input_snapshot_id="input:missing",
        ),
        _fact(
            "legacy_execution_observation",
            "arbitration:legacy",
            emission_id="emission:missing",
            evaluation_id="evaluation:missing",
        ),
        _fact(
            "shared_execution_evaluation",
            "arbitration:shared",
            source_arbitration_evaluation_id="evaluation:missing",
            arbitration_result_id="result:expected",
            non_authoritative=False,
            proposed_directive={"arbitration_result_id": "result:other"},
        ),
        _fact(
            "proposal_snapshot",
            "arbitration:snapshot",
            candidates=(
                {
                    "proposal_id": "proposal:missing",
                    "proposal_content_hash": "1" * 64,
                },
            ),
        ),
        _fact(
            "decision_schedule_tick",
            "arbitration:tick",
            schedule_tick_id="tick:missing",
        ),
    )
    relevant_codes = {
        "snapshot_candidate_missing_proposal",
        "decision_tick_missing_input",
        "computation_missing_input",
        "evaluation_missing_computation",
        "evaluation_missing_input",
        "shared_execution_shadow_missing_evaluation",
        "shared_execution_shadow_marked_authoritative",
        "shared_execution_shadow_directive_mismatch",
        "legacy_execution_missing_emission",
        "legacy_execution_missing_evaluation",
    }
    issues = tuple(
        issue
        for issue in evaluate_control_trace_lineage(facts, profile=profile)
        if issue.code in relevant_codes
    )

    assert issues == (
        ControlTraceLineageIssue(
            code="computation_missing_input",
            root_fact_id="arbitration:computation",
            detail="Arbitration computation must resolve to one input snapshot.",
            related_fact_ids=("input:missing",),
        ),
        ControlTraceLineageIssue(
            code="evaluation_missing_computation",
            root_fact_id="arbitration:evaluation",
            detail="Shadow evaluation must resolve to one computation.",
            related_fact_ids=("computation:missing",),
        ),
        ControlTraceLineageIssue(
            code="evaluation_missing_input",
            root_fact_id="arbitration:evaluation",
            detail="Shadow evaluation must resolve to one input snapshot.",
            related_fact_ids=("input:missing",),
        ),
        ControlTraceLineageIssue(
            code="legacy_execution_missing_emission",
            root_fact_id="arbitration:legacy",
            detail="Legacy execution must reference its passive source emission.",
            related_fact_ids=("emission:missing",),
        ),
        ControlTraceLineageIssue(
            code="legacy_execution_missing_evaluation",
            root_fact_id="arbitration:legacy",
            detail="Legacy execution references an absent shadow evaluation.",
            related_fact_ids=("evaluation:missing",),
        ),
        ControlTraceLineageIssue(
            code="shared_execution_shadow_directive_mismatch",
            root_fact_id="arbitration:shared",
            detail="Proposed directive references another arbitration result.",
        ),
        ControlTraceLineageIssue(
            code="shared_execution_shadow_marked_authoritative",
            root_fact_id="arbitration:shared",
            detail="Shared execution evaluation must remain non-authoritative.",
        ),
        ControlTraceLineageIssue(
            code="shared_execution_shadow_missing_evaluation",
            root_fact_id="arbitration:shared",
            detail="Shared execution shadow must reference one arbitration evaluation.",
            related_fact_ids=("evaluation:missing",),
        ),
        ControlTraceLineageIssue(
            code="snapshot_candidate_missing_proposal",
            root_fact_id="arbitration:snapshot",
            detail="ProposalSnapshot candidate does not resolve to exactly one proposal.",
            related_fact_ids=("proposal:missing",),
        ),
        ControlTraceLineageIssue(
            code="decision_tick_missing_input",
            root_fact_id="arbitration:tick",
            detail="Each materialized decision tick must resolve to one arbitration input.",
        ),
    )

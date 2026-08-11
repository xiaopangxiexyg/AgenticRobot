"""Replay contract, eligibility, receipt, and proposal lineage validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .source_proposal import SourceProposalValidationIndex


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ReplayLineageValidationIndex:
    receipt_by_representation_id: FactIndex
    receipt_by_source_emission: FactIndex
    record_by_id: FactIndex
    source_action_contract_by_id: FactIndex
    target_runtime_contract_by_id: FactIndex
    assessment_by_id: FactIndex
    eligibility_by_id: FactIndex
    execution_authorization_by_id: FactIndex
    candidate_authorization_admission_by_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ReplayLineageValidationIndex":
        index = context.payload_index
        return cls(
            receipt_by_representation_id=index(
                event_type="replay_source_adapter_receipt",
                payload_key="representation_trace_id",
            ),
            receipt_by_source_emission=index(
                event_type="replay_source_adapter_receipt",
                payload_key="source_emission_id",
            ),
            record_by_id=index(
                event_type="replay_action_record",
                payload_key="replay_action_record_id",
            ),
            source_action_contract_by_id=index(
                event_type="replay_source_action_contract",
                payload_key="contract_id",
            ),
            target_runtime_contract_by_id=index(
                event_type="target_runtime_action_contract",
                payload_key="target_contract_id",
            ),
            assessment_by_id=index(
                event_type="replay_binding_assessment",
                payload_key="assessment_id",
            ),
            eligibility_by_id=index(
                event_type="candidate_binding_eligibility",
                payload_key="eligibility_id",
            ),
            execution_authorization_by_id=index(
                event_type="replay_execution_authorization",
                payload_key="authorization_id",
            ),
            candidate_authorization_admission_by_id=index(
                event_type="candidate_authorization_admission",
                payload_key="authorization_id",
            ),
        )


def _validate_replay_records(
    context: ControlTraceValidationContext,
    indexes: ReplayLineageValidationIndex,
    source_indexes: SourceProposalValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    by_id = context.by_id
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "candidate_authorization_admission" and bool(
            payload["accepted"]
        ):
            proposal_id = str(payload["proposal_id"])
            if len(source_indexes.proposal_by_id.get(proposal_id, ())) != 1:
                issues.add(
                    "candidate_authorization_missing_proposal",
                    fact.fact_id,
                    "Accepted candidate authorization must resolve its proposal.",
                    proposal_id,
                )
            eligibility_id = str(payload["binding_eligibility_id"])
            if len(indexes.eligibility_by_id.get(eligibility_id, ())) != 1:
                issues.add(
                    "candidate_authorization_missing_binding_eligibility",
                    fact.fact_id,
                    "Replay candidate authorization must resolve binding eligibility.",
                    eligibility_id,
                )
            evidence_id = str(payload["evidence_id"])
            if len(
                indexes.execution_authorization_by_id.get(evidence_id, ())
            ) != 1:
                issues.add(
                    "candidate_authorization_missing_execution_authorization",
                    fact.fact_id,
                    "Replay candidate authorization must resolve execution authorization.",
                    evidence_id,
                )
        elif fact.event_type == "candidate_authorization_snapshot":
            for relation in payload["authorization_by_proposal_id"].values():
                authorization_id = str(relation["authorization_id"])
                admissions = (
                    indexes.candidate_authorization_admission_by_id.get(
                        authorization_id, ()
                    )
                )
                if len(admissions) != 1:
                    issues.add(
                        "candidate_authorization_snapshot_missing_admission",
                        fact.fact_id,
                        "Frozen candidate authorization must resolve one admission.",
                        authorization_id,
                    )
        elif fact.event_type == "replay_binding_assessment":
            source_contracts = indexes.source_action_contract_by_id.get(
                str(payload["source_action_contract_id"]), ()
            )
            target_contracts = indexes.target_runtime_contract_by_id.get(
                str(payload["target_runtime_contract_id"]), ()
            )
            if len(source_contracts) != 1 or source_contracts[0].payload[
                "semantic_content_hash"
            ] != payload["source_action_contract_hash"]:
                issues.add(
                    "replay_assessment_missing_source_contract",
                    fact.fact_id,
                    "Replay assessment must resolve its exact source action contract.",
                )
            if len(target_contracts) != 1 or target_contracts[0].payload[
                "semantic_content_hash"
            ] != payload["target_runtime_contract_hash"]:
                issues.add(
                    "replay_assessment_missing_target_contract",
                    fact.fact_id,
                    "Replay assessment must resolve its exact target runtime contract.",
                )
        elif fact.event_type == "candidate_binding_eligibility":
            assessments = indexes.assessment_by_id.get(
                str(payload["assessment_id"]), ()
            )
            if len(assessments) != 1 or assessments[0].payload[
                "semantic_content_hash"
            ] != payload["assessment_hash"]:
                issues.add(
                    "replay_eligibility_missing_assessment",
                    fact.fact_id,
                    "Replay candidate eligibility must resolve its exact assessment.",
                )
            if bool(payload["authoritative_allowed"]) and payload[
                "requested_use"
            ] != "execution_candidate":
                issues.add(
                    "replay_eligibility_invalid_authoritative_scope",
                    fact.fact_id,
                    "Only execution-candidate eligibility may be authoritative-allowed.",
                )
        elif fact.event_type == "replay_source_adapter_receipt":
            record_id = payload["replay_action_record_id"]
            records = (
                ()
                if record_id is None
                else indexes.record_by_id.get(str(record_id), ())
            )
            if bool(payload["accepted"]):
                if len(records) != 1:
                    issues.add(
                        "replay_receipt_missing_record",
                        fact.fact_id,
                        "Accepted replay adapter receipt must resolve to one replay action record.",
                        "" if record_id is None else str(record_id),
                    )
                elif payload["replay_action_record_hash"] != records[0].payload[
                    "semantic_content_hash"
                ]:
                    issues.add(
                        "replay_receipt_record_hash_mismatch",
                        fact.fact_id,
                        "Replay adapter receipt and source record hashes disagree.",
                        records[0].fact_id,
                    )
            assessments = indexes.assessment_by_id.get(
                str(payload["binding_assessment_id"]), ()
            )
            eligibility = indexes.eligibility_by_id.get(
                str(payload["binding_eligibility_id"]), ()
            )
            if len(assessments) != 1 or assessments[0].payload[
                "semantic_content_hash"
            ] != payload["binding_assessment_hash"]:
                issues.add(
                    "replay_receipt_missing_assessment",
                    fact.fact_id,
                    "Replay receipt must resolve its exact binding assessment.",
                )
            if len(eligibility) != 1 or eligibility[0].payload[
                "semantic_content_hash"
            ] != payload["binding_eligibility_hash"]:
                issues.add(
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
                issues.add(
                    "replay_receipt_missing_passive_emission",
                    fact.fact_id,
                    "Replay receipt must reference its unique passive source emission.",
                    source_emission_id,
                )
        elif (
            fact.event_type == "action_proposal"
            and payload["producer_kind"] == "replay"
        ):
            representation_trace_id = payload["representation_trace_id"]
            receipts = (
                ()
                if representation_trace_id is None
                else indexes.receipt_by_representation_id.get(
                    str(representation_trace_id), ()
                )
            )
            if len(receipts) != 1:
                issues.add(
                    "replay_proposal_missing_source_receipt",
                    fact.fact_id,
                    "Replay proposal must resolve to one typed replay adapter receipt.",
                    "" if representation_trace_id is None else str(representation_trace_id),
                )


def _validate_replay_closure(
    context: ControlTraceValidationContext,
    indexes: ReplayLineageValidationIndex,
    source_indexes: SourceProposalValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    replay_policies = {
        "replay_shadow_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }
    if lineage_policy_id not in replay_policies:
        return
    explicit_replay_emission = lineage_policy_id in {
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }
    replay_emission_event_type = (
        "replay_source_frame_emission"
        if explicit_replay_emission
        else "policy_action_emission"
    )
    for fact in context.ordered:
        if fact.event_type != replay_emission_event_type:
            continue
        emission_id = str(fact.payload["emission_id"])
        if len(indexes.receipt_by_source_emission.get(emission_id, ())) != 1:
            issues.add(
                "replay_emission_missing_source_receipt",
                fact.fact_id,
                "Each replay queue-pop emission needs one typed replay adapter receipt.",
            )
        if replay_emission_event_type == "policy_action_emission" and len(
            source_indexes.validation_by_emission.get(emission_id, ())
        ) != 1:
            issues.add(
                "replay_emission_missing_source_validation",
                fact.fact_id,
                "Each replay queue-pop emission needs one source-validation fact.",
            )
    for fact in context.ordered:
        if fact.event_type != "replay_source_adapter_receipt" or not bool(
            fact.payload["accepted"]
        ):
            continue
        admission_key = str(
            fact.payload[
                "source_emission_id"
                if explicit_replay_emission
                else "representation_trace_id"
            ]
        )
        admissions = source_indexes.admission_by_emission.get(admission_key, ())
        if len(admissions) != 1:
            issues.add(
                "replay_receipt_missing_admission",
                fact.fact_id,
                "Accepted replay adapter receipt needs exactly one proposal admission.",
            )


def validate_replay_lineage(
    context: ControlTraceValidationContext,
    indexes: ReplayLineageValidationIndex,
    source_indexes: SourceProposalValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    _validate_replay_records(context, indexes, source_indexes, issues)
    _validate_replay_closure(
        context,
        indexes,
        source_indexes,
        lineage_policy_id=lineage_policy_id,
        issues=issues,
    )


__all__ = [
    "ReplayLineageValidationIndex",
    "validate_replay_lineage",
]

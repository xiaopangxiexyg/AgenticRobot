"""Policy, human, and composer source/proposal lineage validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .composition_evaluation import CompositionEvaluationValidationIndex
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class SourceProposalValidationIndex:
    """Shared source indexes also consumed by replay and arbitration rules."""

    activation_by_id: FactIndex
    proposal_by_id: FactIndex
    admission_by_emission: FactIndex
    validation_by_emission: FactIndex
    human_emission_by_id: FactIndex
    source_receipt_by_representation_id: FactIndex
    source_receipt_by_emission: FactIndex
    status_by_emission: FactIndex
    composition_input_admission_by_input_id: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "SourceProposalValidationIndex":
        index = context.payload_index
        return cls(
            activation_by_id=index(
                event_type="producer_epoch_activation",
                payload_key="activation_id",
            ),
            proposal_by_id=index(
                event_type="action_proposal",
                payload_key="proposal_id",
            ),
            admission_by_emission=index(
                event_type="proposal_admission",
                payload_key="emission_id",
            ),
            validation_by_emission=index(
                event_type="source_validation",
                payload_key="source_event_id",
            ),
            human_emission_by_id=index(
                event_type="human_command_emission",
                payload_key="source_emission_id",
            ),
            source_receipt_by_representation_id=index(
                event_type="source_adapter_receipt",
                payload_key="representation_trace_id",
            ),
            source_receipt_by_emission=index(
                event_type="source_adapter_receipt",
                payload_key="source_emission_id",
            ),
            status_by_emission=index(
                event_type="arbiter_shadow_status",
                payload_key="emission_id",
            ),
            composition_input_admission_by_input_id=index(
                event_type="composition_input_admission",
                payload_key="composition_input_id",
            ),
        )


def _validate_source_records(
    context: ControlTraceValidationContext,
    indexes: SourceProposalValidationIndex,
    composition_indexes: CompositionEvaluationValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "producer_epoch_snapshot":
            for activation_id in payload["activation_id_by_producer"].values():
                if str(activation_id) not in indexes.activation_by_id:
                    issues.add(
                        "producer_snapshot_missing_activation",
                        fact.fact_id,
                        "Producer epoch snapshot references an absent activation.",
                        str(activation_id),
                    )
        elif fact.event_type == "proposal_admission":
            proposal_id = payload["proposal_id"]
            accepted = bool(payload["accepted"])
            matches = (
                ()
                if proposal_id is None
                else indexes.proposal_by_id.get(str(proposal_id), ())
            )
            if accepted and len(matches) != 1:
                issues.add(
                    "accepted_admission_missing_proposal",
                    fact.fact_id,
                    "An accepted admission must resolve to exactly one action proposal.",
                    "" if proposal_id is None else str(proposal_id),
                )
            if not accepted and matches:
                issues.add(
                    "rejected_admission_has_proposal",
                    fact.fact_id,
                    "A rejected admission must not materialize an action proposal.",
                    str(proposal_id),
                )
        elif fact.event_type == "source_adapter_receipt":
            source_emission_id = str(payload["source_emission_id"])
            emissions = indexes.human_emission_by_id.get(
                source_emission_id, ()
            )
            if len(emissions) != 1:
                issues.add(
                    "source_receipt_missing_human_emission",
                    fact.fact_id,
                    "Human source receipt must resolve to one command emission.",
                    source_emission_id,
                )
            elif payload["source_emission_semantic_hash"] != emissions[0].payload[
                "source_emission_semantic_hash"
            ]:
                issues.add(
                    "source_receipt_emission_hash_mismatch",
                    fact.fact_id,
                    "Human source receipt and command emission hashes disagree.",
                    source_emission_id,
                )
            validations = indexes.validation_by_emission.get(
                source_emission_id, ()
            )
            if len(validations) != 1:
                issues.add(
                    "source_receipt_missing_validation",
                    fact.fact_id,
                    "Human source receipt must resolve to one source validation.",
                    source_emission_id,
                )
            elif bool(payload["source_validation_accepted"]) != bool(
                validations[0].payload["accepted"]
            ):
                issues.add(
                    "source_receipt_validation_mismatch",
                    fact.fact_id,
                    "Human source receipt and source-validation outcomes disagree.",
                    validations[0].fact_id,
                )
        elif (
            fact.event_type == "action_proposal"
            and payload["producer_kind"] == "human"
        ):
            representation_trace_id = payload["representation_trace_id"]
            receipts = (
                ()
                if representation_trace_id is None
                else indexes.source_receipt_by_representation_id.get(
                    str(representation_trace_id), ()
                )
            )
            if len(receipts) != 1:
                issues.add(
                    "human_proposal_missing_source_receipt",
                    fact.fact_id,
                    "Human proposal must resolve to one typed source-adapter receipt.",
                    "" if representation_trace_id is None else str(representation_trace_id),
                )
        elif (
            fact.event_type == "action_proposal"
            and payload["producer_kind"] == "composer"
        ):
            result_id = str(payload["composition_evaluation_id"])
            results = composition_indexes.evaluation_by_id.get(result_id, ())
            if len(results) != 1:
                issues.add(
                    "composite_proposal_missing_result",
                    fact.fact_id,
                    "Composite proposal must resolve one composition evaluation.",
                    result_id,
                )
            else:
                result_payload = results[0].payload
                if payload["composition_evaluation_hash"] != result_payload[
                    "semantic_content_hash"
                ]:
                    issues.add(
                        "composite_proposal_evaluation_hash_mismatch",
                        fact.fact_id,
                        "Composite proposal and composition evaluation hashes disagree.",
                        result_id,
                    )
                if tuple(payload["composition_input_ids"]) != tuple(
                    result_payload["input_ids"]
                ) or tuple(payload["composition_input_hashes"]) != tuple(
                    result_payload["input_hashes"]
                ):
                    issues.add(
                        "composite_proposal_input_lineage_mismatch",
                        fact.fact_id,
                        "Composite proposal and composition evaluation inputs disagree.",
                        result_id,
                    )
                if payload["composition_spec_hash"] != result_payload[
                    "composition_spec_hash"
                ]:
                    issues.add(
                        "composite_proposal_spec_mismatch",
                        fact.fact_id,
                        "Composite proposal and composition evaluation specs disagree.",
                        result_id,
                    )
            dispositions = composition_indexes.disposition_by_result_id.get(
                result_id, ()
            )
            if dispositions and (
                len(dispositions) != 1
                or dispositions[0].payload["action"] != "publish_proposal"
            ):
                issues.add(
                    "composite_proposal_disposition_invalid",
                    fact.fact_id,
                    "Formal composite proposal requires one publish disposition.",
                    result_id,
                )


def _validate_policy_closure(
    context: ControlTraceValidationContext,
    indexes: SourceProposalValidationIndex,
    composition_indexes: CompositionEvaluationValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    if lineage_policy_id in {
        "policy_shadow_lineage_v1",
        "authoritative_station_immediate_qref_rtc_policy_lineage_v1",
        "authoritative_station_scheduled_qpos_policy_lineage_v1",
    }:
        for fact in context.ordered:
            if fact.event_type != "policy_action_emission":
                continue
            emission_id = str(fact.payload["emission_id"])
            admissions = indexes.admission_by_emission.get(emission_id, ())
            statuses = indexes.status_by_emission.get(emission_id, ())
            validations = indexes.validation_by_emission.get(emission_id, ())
            if len(validations) != 1:
                issues.add(
                    "emission_missing_source_validation",
                    fact.fact_id,
                    "Each policy emission must have exactly one source-validation fact.",
                )
            if len(admissions) > 1:
                issues.add(
                    "emission_multiple_admissions",
                    fact.fact_id,
                    "A queue-pop emission may be admitted at most once.",
                    *(item.fact_id for item in admissions),
                )
            if not admissions and not statuses:
                issues.add(
                    "emission_missing_outcome",
                    fact.fact_id,
                    "An emission needs an admission receipt or typed shadow status.",
                )

    if lineage_policy_id != (
        "authoritative_scheduled_component_policy_human_"
        "composition_lineage_v1"
    ):
        return
    for fact in context.ordered:
        if fact.event_type != "policy_action_emission":
            continue
        emission_id = str(fact.payload["emission_id"])
        validations = indexes.validation_by_emission.get(emission_id, ())
        if len(validations) != 1:
            issues.add(
                "policy_component_emission_missing_source_validation",
                fact.fact_id,
                "Each policy contributor emission must have exactly one "
                "source-validation fact.",
            )
        contributor_inputs = tuple(
            composition_indexes.contributor_input_by_source_fact_id.get(
                emission_id, ()
            )
        )
        if len(contributor_inputs) != 1:
            issues.add(
                "policy_component_emission_missing_input",
                fact.fact_id,
                "Each accepted policy contributor emission must resolve "
                "to exactly one composition contributor input.",
                *(item.fact_id for item in contributor_inputs),
            )
            continue
        input_id = str(contributor_inputs[0].payload["composition_input_id"])
        admissions = tuple(
            indexes.composition_input_admission_by_input_id.get(input_id, ())
        )
        if len(admissions) != 1 or not bool(admissions[0].payload["accepted"]):
            issues.add(
                "policy_component_input_missing_admission",
                contributor_inputs[0].fact_id,
                "A policy contributor input needs one accepted "
                "composition admission receipt.",
                *(item.fact_id for item in admissions),
            )


def _validate_human_closure(
    context: ControlTraceValidationContext,
    indexes: SourceProposalValidationIndex,
    composition_indexes: CompositionEvaluationValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    if lineage_policy_id in {
        "human_shadow_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }:
        for fact in context.ordered:
            if fact.event_type != "human_command_emission":
                continue
            emission_id = str(fact.payload["source_emission_id"])
            if len(indexes.source_receipt_by_emission.get(emission_id, ())) != 1:
                issues.add(
                    "human_emission_missing_source_receipt",
                    fact.fact_id,
                    "Each human command emission needs one typed source-adapter receipt.",
                )
            if len(indexes.validation_by_emission.get(emission_id, ())) != 1:
                issues.add(
                    "human_emission_missing_source_validation",
                    fact.fact_id,
                    "Each human command emission needs one source-validation fact.",
                )
            admissions = indexes.admission_by_emission.get(emission_id, ())
            statuses = indexes.status_by_emission.get(emission_id, ())
            if len(admissions) > 1:
                issues.add(
                    "human_emission_multiple_admissions",
                    fact.fact_id,
                    "A human emission may be admitted at most once.",
                    *(item.fact_id for item in admissions),
                )
            if not admissions and not statuses:
                issues.add(
                    "human_emission_missing_outcome",
                    fact.fact_id,
                    "A human emission needs an admission receipt or typed shadow status.",
                )

    if lineage_policy_id != (
        "authoritative_scheduled_component_policy_human_"
        "composition_lineage_v1"
    ):
        return
    for fact in context.ordered:
        if fact.event_type != "human_command_emission":
            continue
        emission_id = str(fact.payload["source_emission_id"])
        receipts = tuple(
            indexes.source_receipt_by_emission.get(emission_id, ())
        )
        if len(receipts) != 1:
            issues.add(
                "human_component_emission_missing_source_receipt",
                fact.fact_id,
                "Each human contributor emission must resolve exactly one "
                "source-adapter receipt.",
                *(item.fact_id for item in receipts),
            )
        validations = tuple(
            indexes.validation_by_emission.get(emission_id, ())
        )
        if len(validations) != 1:
            issues.add(
                "human_component_emission_missing_source_validation",
                fact.fact_id,
                "Each human contributor emission must have exactly one "
                "source-validation fact.",
                *(item.fact_id for item in validations),
            )
        output_action_present = bool(
            len(receipts) == 1
            and receipts[0].payload["output_action_semantic_hash"] is not None
        )
        contributor_inputs = tuple(
            composition_indexes.contributor_input_by_source_fact_id.get(
                emission_id, ()
            )
        )
        expected_input_count = 1 if output_action_present else 0
        if len(contributor_inputs) != expected_input_count:
            issues.add(
                "human_component_emission_input_count_mismatch",
                fact.fact_id,
                "A human emission contributes exactly when its adapter "
                "receipt contains an output action.",
                *(item.fact_id for item in contributor_inputs),
            )
            continue
        if not contributor_inputs:
            continue
        input_id = str(contributor_inputs[0].payload["composition_input_id"])
        admissions = tuple(
            indexes.composition_input_admission_by_input_id.get(input_id, ())
        )
        if len(admissions) != 1 or not bool(admissions[0].payload["accepted"]):
            issues.add(
                "human_component_input_missing_admission",
                contributor_inputs[0].fact_id,
                "A human contributor input needs one accepted "
                "composition admission receipt.",
                *(item.fact_id for item in admissions),
            )


def validate_source_proposal_lineage(
    context: ControlTraceValidationContext,
    indexes: SourceProposalValidationIndex,
    composition_indexes: CompositionEvaluationValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate source products without owning production or admission."""

    _validate_source_records(context, indexes, composition_indexes, issues)
    _validate_policy_closure(
        context,
        indexes,
        composition_indexes,
        lineage_policy_id=lineage_policy_id,
        issues=issues,
    )
    _validate_human_closure(
        context,
        indexes,
        composition_indexes,
        lineage_policy_id=lineage_policy_id,
        issues=issues,
    )


__all__ = [
    "SourceProposalValidationIndex",
    "validate_source_proposal_lineage",
]

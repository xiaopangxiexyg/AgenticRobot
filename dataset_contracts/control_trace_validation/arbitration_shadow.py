"""Arbitration snapshot, computation, and passive shadow validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .source_proposal import SourceProposalValidationIndex


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ArbitrationShadowValidationIndex:
    input_by_id: FactIndex
    computation_by_id: FactIndex
    evaluation_by_computation: FactIndex
    input_by_schedule_tick: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ArbitrationShadowValidationIndex":
        index = context.payload_index
        return cls(
            input_by_id=index(
                event_type="arbitration_input_snapshot",
                payload_key="arbitration_input_snapshot_id",
            ),
            computation_by_id=index(
                event_type="arbitration_computation",
                payload_key="result_id",
            ),
            evaluation_by_computation=index(
                event_type="arbitration_evaluation",
                payload_key="computation_result_id",
            ),
            input_by_schedule_tick=index(
                event_type="arbitration_input_snapshot",
                payload_key="decision_schedule_tick_id",
            ),
        )


def _validate_manifest_binding(
    context: ControlTraceValidationContext,
    execution_manifest: ControlTraceFact | None,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    if execution_manifest is None:
        return
    manifest_action_schema_hash = str(
        execution_manifest.payload["action_schema_hash"]
    )
    manifest_catalog_hash = str(
        execution_manifest.payload["component_catalog_hash"]
    )
    for fact in context.ordered:
        if fact.event_type != "arbitration_input_snapshot":
            continue
        if str(fact.payload["action_schema_hash"]) != manifest_action_schema_hash:
            issues.add(
                "execution_manifest_action_schema_mismatch",
                fact.fact_id,
                "Arbitration input uses another complete action schema.",
                execution_manifest.fact_id,
            )
        if str(fact.payload["component_catalog_hash"]) != manifest_catalog_hash:
            issues.add(
                "execution_manifest_component_catalog_mismatch",
                fact.fact_id,
                "Arbitration input uses another component catalog.",
                execution_manifest.fact_id,
            )


def _validate_arbitration_records(
    context: ControlTraceValidationContext,
    indexes: ArbitrationShadowValidationIndex,
    source_indexes: SourceProposalValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    by_id = context.by_id
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "proposal_snapshot":
            for candidate in payload["candidates"]:
                proposal_id = str(candidate["proposal_id"])
                matches = source_indexes.proposal_by_id.get(proposal_id, ())
                if len(matches) != 1:
                    issues.add(
                        "snapshot_candidate_missing_proposal",
                        fact.fact_id,
                        "ProposalSnapshot candidate does not resolve to exactly one proposal.",
                        proposal_id,
                    )
                elif matches[0].payload.get("action_schema_hash") != payload[
                    "action_schema_hash"
                ]:
                    issues.add(
                        "snapshot_candidate_schema_mismatch",
                        fact.fact_id,
                        "Candidate and ProposalSnapshot component schemas disagree.",
                        proposal_id,
                    )
                elif candidate["proposal_content_hash"] != matches[0].payload.get(
                    "proposal_semantic_content_hash"
                ):
                    issues.add(
                        "snapshot_candidate_hash_mismatch",
                        fact.fact_id,
                        "Candidate content hash does not match its proposal fact.",
                        proposal_id,
                    )
        elif fact.event_type == "decision_schedule_tick":
            tick_id = str(payload["schedule_tick_id"])
            if len(indexes.input_by_schedule_tick.get(tick_id, ())) != 1:
                issues.add(
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
                    issues.add(
                        "arbitration_input_missing_child",
                        fact.fact_id,
                        f"Arbitration input child {field_name} is absent or has "
                        "the wrong event type.",
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
                if source_hash is not None and payload[
                    child_hash_field
                ] != source_hash:
                    issues.add(
                        "arbitration_input_child_hash_mismatch",
                        fact.fact_id,
                        f"Arbitration input {child_hash_field} disagrees with child fact.",
                        child.fact_id,
                    )
        elif fact.event_type == "arbitration_computation":
            input_id = str(payload["arbitration_input_snapshot_id"])
            inputs = indexes.input_by_id.get(input_id, ())
            if len(inputs) != 1:
                issues.add(
                    "computation_missing_input",
                    fact.fact_id,
                    "Arbitration computation must resolve to one input snapshot.",
                    input_id,
                )
            elif payload["arbitration_input_hash"] != inputs[0].payload[
                "semantic_content_hash"
            ]:
                issues.add(
                    "computation_input_hash_mismatch",
                    fact.fact_id,
                    "Computation input hash disagrees with its input snapshot.",
                    input_id,
                )
        elif fact.event_type == "arbitration_evaluation":
            computation_id = str(payload["computation_result_id"])
            computations = indexes.computation_by_id.get(computation_id, ())
            if len(computations) != 1:
                issues.add(
                    "evaluation_missing_computation",
                    fact.fact_id,
                    "Shadow evaluation must resolve to one computation.",
                    computation_id,
                )
            elif payload["computation_result_hash"] != computations[0].payload[
                "semantic_content_hash"
            ]:
                issues.add(
                    "evaluation_computation_hash_mismatch",
                    fact.fact_id,
                    "Evaluation computation hash disagrees with its computation fact.",
                    computation_id,
                )
            input_id = str(payload["arbitration_input_snapshot_id"])
            if len(indexes.input_by_id.get(input_id, ())) != 1:
                issues.add(
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
                issues.add(
                    "shared_execution_shadow_missing_evaluation",
                    fact.fact_id,
                    "Shared execution shadow must reference one arbitration evaluation.",
                    str(source_evaluation_id),
                )
            elif str(
                by_id[str(source_evaluation_id)].payload["computation_result_id"]
            ) != str(payload["arbitration_result_id"]):
                issues.add(
                    "shared_execution_shadow_result_mismatch",
                    fact.fact_id,
                    "Shared execution shadow and arbitration evaluation disagree.",
                    str(source_evaluation_id),
                )
            if not bool(payload["non_authoritative"]):
                issues.add(
                    "shared_execution_shadow_marked_authoritative",
                    fact.fact_id,
                    "Shared execution evaluation must remain non-authoritative.",
                )
            directive = payload["proposed_directive"]
            if str(directive["arbitration_result_id"]) != str(
                payload["arbitration_result_id"]
            ):
                issues.add(
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
                issues.add(
                    "legacy_execution_missing_emission",
                    fact.fact_id,
                    "Legacy execution must reference its passive source emission.",
                    emission_id,
                )
            evaluation_id = payload["evaluation_id"]
            if evaluation_id is not None and (
                str(evaluation_id) not in by_id
                or by_id[str(evaluation_id)].event_type
                != "arbitration_evaluation"
            ):
                issues.add(
                    "legacy_execution_missing_evaluation",
                    fact.fact_id,
                    "Legacy execution references an absent shadow evaluation.",
                    str(evaluation_id),
                )


def _validate_shadow_closure(
    context: ControlTraceValidationContext,
    indexes: ArbitrationShadowValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    if lineage_policy_id not in {
        "policy_shadow_lineage_v1",
        "human_shadow_lineage_v1",
        "replay_shadow_lineage_v1",
    }:
        return
    for fact in context.ordered:
        if fact.event_type != "arbitration_input_snapshot":
            continue
        input_id = str(fact.payload["arbitration_input_snapshot_id"])
        computations = tuple(
            item
            for item in context.ordered
            if item.event_type == "arbitration_computation"
            and str(item.payload["arbitration_input_snapshot_id"]) == input_id
        )
        if len(computations) != 1:
            issues.add(
                "arbitration_input_missing_computation",
                fact.fact_id,
                "Each shadow arbitration input needs exactly one computation.",
            )
        elif len(
            indexes.evaluation_by_computation.get(computations[0].fact_id, ())
        ) != 1:
            issues.add(
                "computation_missing_evaluation",
                computations[0].fact_id,
                "Each shadow computation needs exactly one evaluation.",
            )


def validate_arbitration_shadow_lineage(
    context: ControlTraceValidationContext,
    indexes: ArbitrationShadowValidationIndex,
    source_indexes: SourceProposalValidationIndex,
    *,
    execution_manifest: ControlTraceFact | None,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    _validate_manifest_binding(context, execution_manifest, issues)
    _validate_arbitration_records(context, indexes, source_indexes, issues)
    _validate_shadow_closure(
        context,
        indexes,
        lineage_policy_id=lineage_policy_id,
        issues=issues,
    )


__all__ = [
    "ArbitrationShadowValidationIndex",
    "validate_arbitration_shadow_lineage",
]

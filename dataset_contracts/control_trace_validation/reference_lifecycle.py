"""Scheduled-reference lineage and terminal lifecycle validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


FactIndex = Mapping[str, Sequence[ControlTraceFact]]


@dataclass(frozen=True, slots=True)
class ReferenceLifecycleValidationIndex:
    """Shared indexes for one complete scheduled-reference lifecycle."""

    segment_by_id: FactIndex
    resolved_target_by_id: FactIndex
    target_admission_by_target: FactIndex
    anchor_evidence_by_target: FactIndex
    validation_by_id: FactIndex
    admission_by_id: FactIndex
    reservation_by_id: FactIndex
    generation_activation_by_id: FactIndex
    target_activation_by_segment: FactIndex
    anchor_candidate_by_segment: FactIndex
    reservation_by_segment: FactIndex
    generation_activation_by_segment: FactIndex
    terminal_by_segment: FactIndex

    @classmethod
    def build(
        cls,
        context: ControlTraceValidationContext,
    ) -> "ReferenceLifecycleValidationIndex":
        index = context.payload_index
        return cls(
            segment_by_id=index(
                event_type="reference_segment",
                payload_key="reference_segment_id",
            ),
            resolved_target_by_id=index(
                event_type="resolved_execution_target",
                payload_key="target_id",
            ),
            target_admission_by_target=index(
                event_type="execution_target_admission",
                payload_key="target_id",
            ),
            anchor_evidence_by_target=index(
                event_type="safety_anchor_commit_evidence",
                payload_key="target_id",
            ),
            validation_by_id=index(
                event_type="reference_receiver_validation",
                payload_key="validation_id",
            ),
            admission_by_id=index(
                event_type="reference_segment_admission",
                payload_key="admission_id",
            ),
            reservation_by_id=index(
                event_type="group_generation_reservation",
                payload_key="reservation_id",
            ),
            generation_activation_by_id=index(
                event_type="group_generation_activation",
                payload_key="activation_id",
            ),
            target_activation_by_segment=index(
                event_type="reference_target_activation",
                payload_key="reference_segment_id",
            ),
            anchor_candidate_by_segment=index(
                event_type="safety_anchor_commit_candidate",
                payload_key="reference_segment_id",
            ),
            reservation_by_segment=index(
                event_type="group_generation_reservation",
                payload_key="source_segment_id",
            ),
            generation_activation_by_segment=index(
                event_type="group_generation_activation",
                payload_key="source_segment_id",
            ),
            terminal_by_segment=index(
                event_type="reference_segment_terminal",
                payload_key="reference_segment_id",
            ),
        )


def _validate_manifest_reference_schema(
    context: ControlTraceValidationContext,
    indexes: ReferenceLifecycleValidationIndex,
    execution_manifest: ControlTraceFact | None,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    if execution_manifest is None:
        return
    manifest_catalog_hash = str(
        execution_manifest.payload["component_catalog_hash"]
    )
    for fact in context.ordered:
        if fact.event_type != "reference_segment" or str(
            fact.payload["input_action_schema_hash"]
        ) == manifest_catalog_hash:
            continue
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
            targets = indexes.resolved_target_by_id.get(
                str(end_parents[0]["source_fact_id"]),
                (),
            )
            resolved_bridge_ok = bool(
                len(targets) == 1
                and str(targets[0].payload["input_action_schema_hash"])
                == manifest_catalog_hash
                and str(targets[0].payload["output_target_schema_hash"])
                == str(fact.payload["input_action_schema_hash"])
            )
        if not resolved_bridge_ok:
            issues.add(
                "execution_manifest_reference_schema_mismatch",
                fact.fact_id,
                "Reference segment input schema differs from the execution manifest without an explicit resolved-target bridge.",
                execution_manifest.fact_id,
            )


def _validate_reference_records(
    context: ControlTraceValidationContext,
    indexes: ReferenceLifecycleValidationIndex,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    by_id = context.by_id
    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type == "reference_segment":
            parents = tuple(payload["endpoint_parents"])
            if len(parents) != 2 or tuple(item["role"] for item in parents) != (
                "start",
                "end",
            ):
                issues.add(
                    "reference_segment_parent_order_invalid",
                    fact.fact_id,
                    "Reference segment must preserve ordered START/END parents.",
                )
            elif str(parents[1]["source_fact_id"]) not in by_id:
                issues.add(
                    "reference_segment_missing_safe_result",
                    fact.fact_id,
                    "Reference END parent must resolve its shared-safety fact.",
                    str(parents[1]["source_fact_id"]),
                )
        elif fact.event_type == "reference_segment_admission":
            segment_id = str(payload["reference_segment_id"])
            segments = indexes.segment_by_id.get(segment_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issues.add(
                    "reference_admission_missing_segment",
                    fact.fact_id,
                    "Reference admission must resolve its exact segment.",
                    segment_id,
                )
            validation_id = str(payload["receiver_validation_id"])
            validations = indexes.validation_by_id.get(validation_id, ())
            if len(validations) != 1 or validations[0].payload[
                "semantic_content_hash"
            ] != payload["receiver_validation_hash"]:
                issues.add(
                    "reference_admission_missing_receiver_validation",
                    fact.fact_id,
                    "Reference admission must resolve its receiver-side validation.",
                    validation_id,
                )
            elif bool(payload["accepted"]) and not bool(
                validations[0].payload["accepted"]
            ):
                issues.add(
                    "reference_admission_validation_rejected",
                    fact.fact_id,
                    "Accepted segment cannot reference rejected receiver validation.",
                    validation_id,
                )
        elif fact.event_type == "safety_anchor_commit_candidate":
            segment_id = str(payload["reference_segment_id"])
            admission_id = str(payload["admission_id"])
            segments = indexes.segment_by_id.get(segment_id, ())
            admissions = indexes.admission_by_id.get(admission_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issues.add(
                    "anchor_candidate_missing_segment",
                    fact.fact_id,
                    "Anchor candidate must resolve its exact segment.",
                    segment_id,
                )
            if len(admissions) != 1 or admissions[0].payload[
                "semantic_content_hash"
            ] != payload["admission_hash"]:
                issues.add(
                    "anchor_candidate_missing_admission",
                    fact.fact_id,
                    "Anchor candidate must resolve its exact admission.",
                    admission_id,
                )
            elif not bool(admissions[0].payload["accepted"]):
                issues.add(
                    "anchor_candidate_from_rejected_admission",
                    fact.fact_id,
                    "Rejected admission cannot produce an anchor candidate.",
                    admission_id,
                )
            safety_id = str(payload["safety_result_id"])
            safety = by_id.get(safety_id)
            if safety is None or safety.event_type != "shared_safety_result":
                issues.add(
                    "anchor_candidate_missing_safety_result",
                    fact.fact_id,
                    "Anchor candidate must resolve its shared-safety result.",
                    safety_id,
                )
        elif fact.event_type == "group_generation_reservation":
            segment_id = str(payload["source_segment_id"])
            admission_id = str(payload["admission_id"])
            segments = indexes.segment_by_id.get(segment_id, ())
            admissions = indexes.admission_by_id.get(admission_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["source_segment_hash"]:
                issues.add(
                    "generation_reservation_missing_segment",
                    fact.fact_id,
                    "Generation reservation must resolve its exact segment.",
                    segment_id,
                )
            if len(admissions) != 1 or admissions[0].payload[
                "semantic_content_hash"
            ] != payload["admission_hash"]:
                issues.add(
                    "generation_reservation_missing_admission",
                    fact.fact_id,
                    "Generation reservation must resolve its exact admission.",
                    admission_id,
                )
            elif not bool(admissions[0].payload["accepted"]):
                issues.add(
                    "generation_reservation_from_rejected_admission",
                    fact.fact_id,
                    "Rejected segment admission cannot reserve a generation.",
                    admission_id,
                )
            if int(payload["reserved_generation"]) != int(
                payload["active_generation_at_reservation"]
            ) + 1:
                issues.add(
                    "generation_reservation_not_next",
                    fact.fact_id,
                    "Generation reservation must reserve active generation + 1.",
                )
            relation = str(payload["relation_kind"])
            predecessor_id = payload["predecessor_segment_id"]
            supersedes_id = payload["supersedes_reservation_id"]
            if relation == "successor" and len(
                indexes.segment_by_id.get(str(predecessor_id), ())
            ) != 1:
                issues.add(
                    "successor_reservation_missing_predecessor",
                    fact.fact_id,
                    "Successor reservation must resolve its predecessor segment.",
                    str(predecessor_id),
                )
            if relation == "supersedes_pending" and len(
                indexes.reservation_by_id.get(str(supersedes_id), ())
            ) != 1:
                issues.add(
                    "superseding_reservation_missing_old_reservation",
                    fact.fact_id,
                    "Superseding reservation must resolve the old reservation.",
                    str(supersedes_id),
                )
        elif fact.event_type == "group_generation_activation":
            reservation_id = str(payload["reservation_id"])
            reservations = indexes.reservation_by_id.get(reservation_id, ())
            if len(reservations) != 1:
                issues.add(
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
                    issues.add(
                        "generation_activation_reservation_mismatch",
                        fact.fact_id,
                        "Activation and reservation lineage disagree.",
                        reservation_id,
                    )
        elif fact.event_type == "reference_target_activation":
            segment_id = str(payload["reference_segment_id"])
            if len(indexes.segment_by_id.get(segment_id, ())) != 1:
                issues.add(
                    "reference_activation_missing_segment",
                    fact.fact_id,
                    "Reference target activation must resolve one segment.",
                    segment_id,
                )
            activation_id = str(payload["generation_activation_id"])
            activations = indexes.generation_activation_by_id.get(
                activation_id,
                (),
            )
            if len(activations) != 1:
                issues.add(
                    "reference_activation_missing_generation_activation",
                    fact.fact_id,
                    "Reference activation must resolve its generation handoff.",
                    activation_id,
                )
            else:
                activation = activations[0].payload
                if (
                    str(activation["source_segment_id"]) != segment_id
                    or int(activation["active_generation"])
                    != int(payload["group_segment_generation"])
                ):
                    issues.add(
                        "reference_generation_activation_mismatch",
                        fact.fact_id,
                        "Reference activation and generation handoff disagree.",
                        activation_id,
                    )
        elif fact.event_type == "planner_command_frame":
            segment_id = str(payload["reference_segment_id"])
            if len(indexes.segment_by_id.get(segment_id, ())) != 1:
                issues.add(
                    "planner_frame_missing_segment",
                    fact.fact_id,
                    "Planner command frame must resolve one segment.",
                    segment_id,
                )
            activations = indexes.target_activation_by_segment.get(segment_id, ())
            if len(activations) != 1:
                issues.add(
                    "planner_frame_missing_activation",
                    fact.fact_id,
                    "Planner command frame must resolve one target activation.",
                    segment_id,
                )
            elif int(activations[0].payload["reference_sequence"]) != int(
                payload["reference_sequence"]
            ):
                issues.add(
                    "planner_frame_reference_sequence_mismatch",
                    fact.fact_id,
                    "Planner frame and target activation reference sequences differ.",
                    activations[0].fact_id,
                )
        elif fact.event_type == "reference_segment_terminal":
            segment_id = str(payload["reference_segment_id"])
            segments = indexes.segment_by_id.get(segment_id, ())
            if len(segments) != 1 or segments[0].payload[
                "semantic_content_hash"
            ] != payload["reference_segment_hash"]:
                issues.add(
                    "reference_terminal_missing_segment",
                    fact.fact_id,
                    "Reference terminal must resolve its exact segment.",
                    segment_id,
                )
            last_frame_id = payload["last_reference_frame_id"]
            if last_frame_id is not None:
                frame = by_id.get(str(last_frame_id))
                if frame is None or frame.event_type != "planner_command_frame":
                    issues.add(
                        "reference_terminal_missing_last_frame",
                        fact.fact_id,
                        "Reference terminal last frame must resolve a planner frame.",
                        str(last_frame_id),
                    )
                elif str(frame.payload["reference_segment_id"]) != segment_id:
                    issues.add(
                        "reference_terminal_last_frame_segment_mismatch",
                        fact.fact_id,
                        "Reference terminal last frame belongs to another segment.",
                        str(last_frame_id),
                    )


def _validate_scheduled_closure(
    context: ControlTraceValidationContext,
    indexes: ReferenceLifecycleValidationIndex,
    *,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    scheduled_policies = {
        "authoritative_scheduled_qpos_lineage_v1",
        "authoritative_station_scheduled_qpos_lineage_v1",
        "authoritative_station_scheduled_pose_to_qpos_lineage_v1",
        "authoritative_scheduled_qpos_human_replay_handoff_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }
    if lineage_policy_id not in scheduled_policies:
        return
    pose_to_qpos = lineage_policy_id in {
        "authoritative_station_scheduled_pose_to_qpos_lineage_v1",
        "authoritative_station_scheduled_pose_human_replay_lineage_v1",
    }
    ordered = context.ordered
    for safety in ordered:
        if (
            safety.event_type != "shared_safety_result"
            or str(safety.payload["status"]) != "safe"
            or safety.payload["safe_action"] is None
        ):
            continue
        if pose_to_qpos:
            matches = tuple(
                target
                for target in ordered
                if target.event_type == "resolved_execution_target"
                and str(target.payload["source_safety_result_id"])
                == safety.fact_id
            )
            if len(matches) != 1:
                issues.add(
                    "scheduled_safe_pose_missing_resolved_target",
                    safety.fact_id,
                    "Each safe scheduled pose result needs exactly one resolved execution target.",
                    *(target.fact_id for target in matches),
                )
        else:
            matches = tuple(
                segment
                for segment in ordered
                if segment.event_type == "reference_segment"
                and len(tuple(segment.payload["endpoint_parents"])) == 2
                and str(segment.payload["endpoint_parents"][1]["source_fact_id"])
                == safety.fact_id
            )
            if len(matches) != 1:
                issues.add(
                    "scheduled_safe_result_missing_segment_attempt",
                    safety.fact_id,
                    "Each safe scheduled-qpos result needs exactly one segment attempt.",
                    *(segment.fact_id for segment in matches),
                )

    if pose_to_qpos:
        for target_id, targets in indexes.resolved_target_by_id.items():
            if len(targets) != 1:
                continue
            target = targets[0]
            admissions = tuple(
                indexes.target_admission_by_target.get(target_id, ())
            )
            if len(admissions) != 1:
                issues.add(
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
                issues.add(
                    "accepted_resolved_target_missing_segment",
                    target.fact_id,
                    "An accepted resolved target needs exactly one reference segment.",
                    *(item.fact_id for item in segments),
                )
            evidence = tuple(indexes.anchor_evidence_by_target.get(target_id, ()))
            if len(evidence) != 1 or not bool(evidence[0].payload["accepted"]):
                issues.add(
                    "accepted_resolved_target_missing_anchor_evidence",
                    target.fact_id,
                    "An accepted resolved target needs one accepted pose-anchor commit evidence fact.",
                    *(item.fact_id for item in evidence),
                )

    for segment_id, segments in indexes.segment_by_id.items():
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
            issues.add(
                "scheduled_segment_missing_single_admission",
                segment.fact_id,
                "Each scheduled segment attempt needs exactly one receiver admission.",
                *(item.fact_id for item in admissions),
            )
            continue
        if not bool(admissions[0].payload["accepted"]):
            continue
        candidates = tuple(indexes.anchor_candidate_by_segment.get(segment_id, ()))
        reservations = tuple(indexes.reservation_by_segment.get(segment_id, ()))
        terminals = tuple(indexes.terminal_by_segment.get(segment_id, ()))
        if not pose_to_qpos and len(candidates) != 1:
            issues.add(
                "accepted_segment_missing_anchor_candidate",
                segment.fact_id,
                "An accepted segment needs exactly one pending anchor candidate.",
                *(item.fact_id for item in candidates),
            )
        if pose_to_qpos and candidates:
            issues.add(
                "resolved_segment_has_qpos_anchor_candidate",
                segment.fact_id,
                "A pose-to-qpos segment must use resolved-target anchor evidence, not a qpos safety-anchor candidate.",
                *(item.fact_id for item in candidates),
            )
        if len(reservations) != 1:
            issues.add(
                "accepted_segment_missing_generation_reservation",
                segment.fact_id,
                "An accepted segment needs exactly one generation reservation.",
                *(item.fact_id for item in reservations),
            )
        if len(terminals) != 1:
            issues.add(
                "accepted_segment_missing_terminal",
                segment.fact_id,
                "An accepted segment must be terminal before its episode is sealed.",
                *(item.fact_id for item in terminals),
            )
        generation_activations = tuple(
            indexes.generation_activation_by_segment.get(segment_id, ())
        )
        target_activations = tuple(
            indexes.target_activation_by_segment.get(segment_id, ())
        )
        if len(generation_activations) != len(target_activations):
            issues.add(
                "scheduled_segment_activation_pair_incomplete",
                segment.fact_id,
                "Generation and reference-target activation must appear as one pair.",
                *(item.fact_id for item in generation_activations + target_activations),
            )
        if len(generation_activations) > 1 or len(target_activations) > 1:
            issues.add(
                "scheduled_segment_activated_multiple_times",
                segment.fact_id,
                "A scheduled segment may activate at most once.",
                *(item.fact_id for item in generation_activations + target_activations),
            )


def validate_reference_lifecycle_lineage(
    context: ControlTraceValidationContext,
    indexes: ReferenceLifecycleValidationIndex,
    *,
    execution_manifest: ControlTraceFact | None,
    lineage_policy_id: str,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    """Validate one reference lifecycle without creating runtime facts."""

    _validate_manifest_reference_schema(context, indexes, execution_manifest, issues)
    _validate_reference_records(context, indexes, issues)
    _validate_scheduled_closure(
        context,
        indexes,
        lineage_policy_id=lineage_policy_id,
        issues=issues,
    )


__all__ = [
    "ReferenceLifecycleValidationIndex",
    "validate_reference_lifecycle_lineage",
]

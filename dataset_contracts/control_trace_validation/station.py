"""Station transition, intent binding, and publication-barrier lineage."""

from __future__ import annotations

from typing import Protocol

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .preflight import ControlTraceManifestSelection


class StationValidationProfileView(Protocol):
    lineage_policy_id: str


def _enabled_barrier_authority_operations(
    profile: StationValidationProfileView,
) -> set[str]:
    enabled = {"request_motion_authority"}
    if profile.lineage_policy_id == (
        "authoritative_station_scheduled_pose_human_replay_lineage_v1"
    ):
        enabled.add("request_human_takeover")
    if profile.lineage_policy_id == (
        "authoritative_station_pose_composition_session_lineage_v1"
    ):
        enabled.update(
            {
                "rearm_composition_session",
                "start_composition_session",
            }
        )
    return enabled


def _ordered_station_transitions(
    context: ControlTraceValidationContext,
) -> tuple[ControlTraceFact, ...]:
    return tuple(
        sorted(
            (
                fact
                for fact in context.ordered
                if fact.event_type
                in {
                    "station_intent_transition",
                    "station_operation_transition",
                }
            ),
            key=lambda fact: int(fact.payload["admission_sequence"]),
        )
    )


def validate_station_lineage(
    context: ControlTraceValidationContext,
    profile: StationValidationProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    station_intent_by_id = context.payload_index(
        event_type="station_intent_transition",
        payload_key="intent_id",
    )
    station_operation_by_id = context.payload_index(
        event_type="station_operation_transition",
        payload_key="operation_result_id",
    )
    enabled_barrier_authority_operations = (
        _enabled_barrier_authority_operations(profile)
    )
    station_state_transitions = _ordered_station_transitions(context)

    previous_station_fact: ControlTraceFact | None = None
    seen_station_admission_sequences: set[int] = set()
    for station_fact in station_state_transitions:
        admission_sequence = int(station_fact.payload["admission_sequence"])
        if admission_sequence in seen_station_admission_sequences:
            issues.add(
                "station_admission_sequence_duplicate",
                station_fact.fact_id,
                "Station intent/result admission sequence must be unique.",
            )
        seen_station_admission_sequences.add(admission_sequence)
        if previous_station_fact is not None and (
            station_fact.payload["state_before"]["semantic_content_hash"]
            != previous_station_fact.payload["state_after"]["semantic_content_hash"]
        ):
            issues.add(
                "station_state_lineage_discontinuous",
                station_fact.fact_id,
                "Station transition state_before does not match the prior state_after.",
                previous_station_fact.fact_id,
            )
        previous_station_fact = station_fact

    station_manifest = manifests.station_manifest
    if station_manifest is not None:
        station_manifest_hash = str(
            station_manifest.payload["semantic_content_hash"]
        )
        for fact in station_state_transitions:
            if fact.event_type == "station_intent_transition" and str(
                fact.payload["manifest_hash"]
            ) != station_manifest_hash:
                issues.add(
                    "station_intent_manifest_mismatch",
                    fact.fact_id,
                    "Station intent references another run manifest.",
                    station_manifest.fact_id,
                )

    for fact in context.ordered:
        if fact.event_type == "station_operation_transition":
            source_intent_id = str(fact.payload["source_intent_id"])
            source_intents = station_intent_by_id.get(source_intent_id, ())
            if len(source_intents) != 1:
                issues.add(
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
                issues.add(
                    "station_operation_intent_binding_mismatch",
                    fact.fact_id,
                    "Station operation kind/subject differs from its source intent.",
                    source_intent_id,
                )
        elif fact.event_type == "station_publication_barrier_transition":
            causal_id = str(fact.payload["causal_station_fact_id"])
            if causal_id not in context.by_id:
                issues.add(
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
                    or (
                        results[0].payload["operation_kind"]
                        == "start_composition_session"
                        and results[0].payload.get("result_status")
                        != "composition_session_ready"
                    )
                    or (
                        results[0].payload["operation_kind"]
                        == "rearm_composition_session"
                        and results[0].payload.get("result_status")
                        != "composition_session_rearmed"
                    )
                    or not tuple(fact.payload["authority_evidence_refs"])
                ):
                    issues.add(
                        "station_barrier_missing_authority_result",
                        fact.fact_id,
                        "Enabled publication must resolve one evidenced authority result.",
                        operation_result_id,
                    )


__all__ = ["StationValidationProfileView", "validate_station_lineage"]

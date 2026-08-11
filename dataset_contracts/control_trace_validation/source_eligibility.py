"""Composition-source eligibility admission, snapshot, and binding lineage."""

from __future__ import annotations

from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .preflight import ControlTraceManifestSelection, TraceRequirementProfileView


def validate_composition_source_eligibility(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    del profile, manifests
    spec_by_hash = context.payload_index(
        event_type="composition_source_eligibility_spec",
        payload_key="semantic_content_hash",
    )
    state_by_id = context.payload_index(
        event_type="composition_source_eligibility_state",
        payload_key="source_eligibility_state_id",
    )
    eligibility_snapshot_by_id = context.payload_index(
        event_type="composition_source_eligibility_snapshot",
        payload_key="source_eligibility_snapshot_id",
    )
    composition_input_snapshot_by_id = context.payload_index(
        event_type="composition_input_snapshot",
        payload_key="composition_input_snapshot_id",
    )

    for fact in context.ordered:
        payload = fact.payload
        if (
            fact.event_type == "composition_source_eligibility_admission"
            and bool(payload["accepted"])
        ):
            state_id = str(payload["source_eligibility_state_id"])
            states = state_by_id.get(state_id, ())
            if len(states) != 1 or states[0].payload[
                "state_semantic_content_hash"
            ] != payload["state_semantic_content_hash"]:
                issues.add(
                    "composition_source_eligibility_admission_missing_state",
                    fact.fact_id,
                    "Accepted eligibility admission must resolve its exact state.",
                    state_id,
                )
        elif fact.event_type == "composition_source_eligibility_snapshot":
            spec_hash = str(payload["source_eligibility_spec_hash"])
            if len(spec_by_hash.get(spec_hash, ())) != 1:
                issues.add(
                    "composition_source_eligibility_snapshot_missing_spec",
                    fact.fact_id,
                    "Eligibility snapshot must resolve its exact spec.",
                    spec_hash,
                )
            state_ids = payload["selected_state_ids_by_role"]
            state_hashes = payload["selected_state_hashes_by_role"]
            for role, state_id in state_ids.items():
                states = state_by_id.get(str(state_id), ())
                if len(states) != 1 or states[0].payload[
                    "state_semantic_content_hash"
                ] != state_hashes[role]:
                    issues.add(
                        "composition_source_eligibility_snapshot_missing_state",
                        fact.fact_id,
                        "Eligibility snapshot must resolve each selected state.",
                        str(state_id),
                    )
        elif fact.event_type == "composition_source_eligibility_binding":
            input_snapshot_id = str(
                payload["composition_input_snapshot_id"]
            )
            input_snapshots = composition_input_snapshot_by_id.get(
                input_snapshot_id,
                (),
            )
            if len(input_snapshots) != 1 or input_snapshots[0].payload[
                "semantic_content_hash"
            ] != payload["composition_input_snapshot_hash"]:
                issues.add(
                    "composition_source_eligibility_binding_missing_input_snapshot",
                    fact.fact_id,
                    "Eligibility binding must resolve the input snapshot.",
                    input_snapshot_id,
                )
            eligibility_snapshot_id = str(
                payload["source_eligibility_snapshot_id"]
            )
            eligibility_snapshots = eligibility_snapshot_by_id.get(
                eligibility_snapshot_id,
                (),
            )
            if (
                len(eligibility_snapshots) != 1
                or eligibility_snapshots[0].payload[
                    "semantic_content_hash"
                ]
                != payload["source_eligibility_snapshot_hash"]
            ):
                issues.add(
                    "composition_source_eligibility_binding_missing_snapshot",
                    fact.fact_id,
                    "Eligibility binding must resolve the eligibility snapshot.",
                    eligibility_snapshot_id,
                )


__all__ = ["validate_composition_source_eligibility"]

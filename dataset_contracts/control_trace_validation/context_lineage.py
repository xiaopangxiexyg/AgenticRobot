"""Policy and composition context dependency lineage validation."""

from __future__ import annotations

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .preflight import ControlTraceManifestSelection, TraceRequirementProfileView


def _context_facts_by_id(
    context: ControlTraceValidationContext,
) -> dict[str, list[ControlTraceFact]]:
    policy_context_by_id = context.payload_index(
        event_type="policy_context",
        payload_key="context_id",
    )
    composition_context_by_id = context.payload_index(
        event_type="composition_context",
        payload_key="context_id",
    )
    context_by_id: dict[str, list[ControlTraceFact]] = dict(
        policy_context_by_id
    )
    for context_id, facts in composition_context_by_id.items():
        context_by_id[context_id] = [
            *context_by_id.get(context_id, ()),
            *facts,
        ]
    return context_by_id


def validate_context_lineage(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    del profile, manifests
    context_by_id = _context_facts_by_id(context)

    for fact in context.ordered:
        payload = fact.payload
        if fact.event_type in {
            "policy_context_admission",
            "composition_context_admission",
        } and bool(payload["accepted"]):
            context_id = str(payload["context_id"])
            contexts = context_by_id.get(context_id, ())
            if len(contexts) != 1 or contexts[0].payload[
                "context_semantic_content_hash"
            ] != payload["context_semantic_content_hash"]:
                prefix = (
                    "policy_context"
                    if fact.event_type == "policy_context_admission"
                    else "composition_context"
                )
                issues.add(
                    f"{prefix}_admission_missing_context",
                    fact.fact_id,
                    "Accepted context admission must resolve its exact context.",
                    context_id,
                )
        elif fact.event_type in {"policy_context", "composition_context"}:
            observation_cut_id = payload.get("observation_cut_id")
            if observation_cut_id is not None:
                observation = context.by_id.get(str(observation_cut_id))
            else:
                observation = None
            if (
                fact.event_type == "policy_context"
                or observation_cut_id is not None
            ) and (
                observation is None
                or observation.event_type != "observation_cut"
            ):
                issues.add(
                    f"{fact.event_type}_missing_observation_cut",
                    fact.fact_id,
                    "Context observation-cut lineage must resolve when declared.",
                    None if observation_cut_id is None else str(observation_cut_id),
                )
            for parent_id in payload["parent_context_ids"]:
                if len(context_by_id.get(str(parent_id), ())) != 1:
                    issues.add(
                        f"{fact.event_type}_missing_parent",
                        fact.fact_id,
                        "Context parent dependency is absent.",
                        str(parent_id),
                    )
        elif fact.event_type in {
            "policy_context_snapshot",
            "composition_context_snapshot",
        }:
            selected_ids = payload["selected_context_id_by_channel"]
            selected_hashes = payload["selected_context_hash_by_channel"]
            for channel, context_id in selected_ids.items():
                contexts = context_by_id.get(str(context_id), ())
                if len(contexts) != 1 or contexts[0].payload[
                    "context_semantic_content_hash"
                ] != selected_hashes[channel]:
                    prefix = (
                        "policy_context"
                        if fact.event_type == "policy_context_snapshot"
                        else "composition_context"
                    )
                    issues.add(
                        f"{prefix}_snapshot_missing_context",
                        fact.fact_id,
                        "Context snapshot must resolve each selected context and hash.",
                        str(context_id),
                    )


__all__ = ["validate_context_lineage"]

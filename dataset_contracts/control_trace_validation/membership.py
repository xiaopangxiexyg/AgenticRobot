"""Ordered episode-membership assertion validation."""

from __future__ import annotations

from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector
from .preflight import ControlTraceManifestSelection, TraceRequirementProfileView


def validate_episode_membership(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    del profile, manifests
    active_episode: str | None = None
    membership_epoch = -1
    for fact in context.facts_of_type("episode_membership_transition"):
        payload = fact.payload
        epoch = int(payload["membership_epoch"])
        episode_uuid = str(payload["episode_uuid"])
        transition = str(payload["transition"])
        if epoch <= membership_epoch:
            issues.add(
                "membership_epoch_not_increasing",
                fact.fact_id,
                "Episode membership epochs must strictly increase.",
            )
        membership_epoch = max(membership_epoch, epoch)
        if transition == "bind":
            if active_episode is not None:
                issues.add(
                    "membership_bind_while_active",
                    fact.fact_id,
                    "A second episode was bound before the current membership was released.",
                )
            active_episode = episode_uuid
        elif transition == "unbind":
            if active_episode != episode_uuid:
                issues.add(
                    "membership_unbind_mismatch",
                    fact.fact_id,
                    "Episode unbind does not match the active membership.",
                )
            active_episode = None
        else:
            issues.add(
                "membership_transition_invalid",
                fact.fact_id,
                f"Unsupported membership transition {transition!r}.",
            )


__all__ = ["validate_episode_membership"]

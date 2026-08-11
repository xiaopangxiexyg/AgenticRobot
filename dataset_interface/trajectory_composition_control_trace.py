"""ControlTrace projection for trajectory-composition shadow summaries.

This adapter records compact lineage only.  Composition input/evaluation facts
remain the owners of full action payloads, so the summary does not duplicate
high-rate vectors.
"""

from __future__ import annotations

from control_composition.events import CompositionDomainEvent
from control_composition.trajectory import (
    TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND,
    TrajectoryCompositionShadowResult,
    trajectory_composition_shadow_semantic_payload,
)
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext


def project_trajectory_composition_shadow_result(
    context: AuthoritativeTraceContext,
    result: TrajectoryCompositionShadowResult,
    *,
    occurred_at: float,
) -> ControlTraceFact:
    if not isinstance(context, AuthoritativeTraceContext):
        raise TypeError("context must be an AuthoritativeTraceContext.")
    if not isinstance(result, TrajectoryCompositionShadowResult):
        raise TypeError("result must be a TrajectoryCompositionShadowResult.")
    return context.fact(
        fact_id=result.result_id,
        event_type=TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND,
        occurred_at=float(occurred_at),
        payload=trajectory_composition_shadow_semantic_payload(result),
    )


def project_trajectory_composition_shadow_domain_event(
    context: AuthoritativeTraceContext,
    event: CompositionDomainEvent,
) -> ControlTraceFact:
    if not isinstance(event, CompositionDomainEvent):
        raise TypeError("event must be a CompositionDomainEvent.")
    if event.event_kind != TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND:
        raise ValueError("Unsupported trajectory composition event kind.")
    return project_trajectory_composition_shadow_result(
        context,
        event.subject,
        occurred_at=event.occurred_at,
    )


__all__ = [
    "project_trajectory_composition_shadow_domain_event",
    "project_trajectory_composition_shadow_result",
]

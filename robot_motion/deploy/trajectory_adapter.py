"""Pure projection from deploy-owned timed plans to generic trajectories."""

from __future__ import annotations

from collections.abc import Callable

from robot_motion.action import (
    ComponentAction,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
)

from .chunk_plan import TimedPolicyActionKeyframe, TimedPolicyActionPlanSnapshot


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def project_timed_policy_plan_to_component_trajectory(
    plan: TimedPolicyActionPlanSnapshot,
    *,
    action_projector: Callable[[TimedPolicyActionKeyframe], ComponentAction],
    trajectory_id: str | None = None,
    source_kind: str = "policy",
    source_activation_id: str | None = None,
    source_activation_hash: str | None = None,
    coordination_group_id: str | None = None,
) -> ComponentActionTrajectory:
    """Project one immutable plan snapshot without reading or advancing it.

    Deploy representations deliberately remain outside the generic carrier.
    The caller supplies the exact representation projector already bound to
    its artifact/target profile.  This function never peeks a queue, mutates a
    cursor, commits a prefix, or emits an action.
    """

    if not isinstance(plan, TimedPolicyActionPlanSnapshot):
        raise TypeError("plan must be a TimedPolicyActionPlanSnapshot.")
    if not callable(action_projector):
        raise TypeError("action_projector must be callable.")
    if not plan.keyframes:
        raise ValueError("Cannot project an empty timed policy plan.")
    if (source_activation_id is None) != (source_activation_hash is None):
        raise ValueError(
            "source_activation_id/hash must be both present or both absent."
        )

    plan_hash = plan.semantic_content_hash
    keyframes: list[ComponentActionTrajectoryKeyframe] = []
    for trajectory_index, timed_keyframe in enumerate(plan.keyframes):
        action = action_projector(timed_keyframe)
        if not isinstance(action, ComponentAction):
            raise TypeError("action_projector must return ComponentAction.")
        keyframe_hash = timed_keyframe.semantic_content_hash
        keyframes.append(
            ComponentActionTrajectoryKeyframe(
                target_effective_at=timed_keyframe.target_effective_at,
                effective_until=timed_keyframe.effective_until,
                action=action,
                source_index=trajectory_index,
                source_fact_ids=(
                    "timed-policy-action-keyframe:"
                    f"{timed_keyframe.chunk_id}:"
                    f"{timed_keyframe.chunk_keyframe_index}:"
                    f"{keyframe_hash}",
                ),
                source_fact_hashes=(keyframe_hash,),
            )
        )

    resolved_trajectory_id = (
        f"component-action-trajectory:{plan.plan_id}"
        if trajectory_id is None
        else _required(trajectory_id, name="trajectory_id")
    )
    return ComponentActionTrajectory(
        trajectory_id=resolved_trajectory_id,
        revision=plan.generation,
        source_kind=_required(source_kind, name="source_kind"),
        source_instance_id=plan.producer_id,
        clock_domain_id=plan.clock_domain_id,
        created_at=plan.created_at,
        keyframes=tuple(keyframes),
        source_fact_ids=(f"timed-policy-action-plan:{plan_hash}",),
        source_fact_hashes=(plan_hash,),
        source_activation_id=source_activation_id,
        source_activation_hash=source_activation_hash,
        source_immutable_through=plan.producer_committed_through,
        coordination_group_id=coordination_group_id,
    )


__all__ = ["project_timed_policy_plan_to_component_trajectory"]

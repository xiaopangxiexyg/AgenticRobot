"""ControlTrace projection for immutable component-action trajectories.

The source owner emits this actual-data fact when it publishes a trajectory.
The projector does not infer a plan, advance a cursor, compose an action, or
claim proposal/target/reference/dispatch admission.
"""

from __future__ import annotations

from dataset_contracts.control_trace import ControlTraceFact

from .trajectory import ComponentActionTrajectory


def component_action_trajectory_payload(
    trajectory: ComponentActionTrajectory,
) -> dict[str, object]:
    """Return the lossless v2.4 payload for one immutable source plan."""

    if not isinstance(trajectory, ComponentActionTrajectory):
        raise TypeError("trajectory must be a ComponentActionTrajectory.")
    return {
        "trajectory_id": trajectory.trajectory_id,
        "revision": trajectory.revision,
        "source_kind": trajectory.source_kind,
        "source_instance_id": trajectory.source_instance_id,
        "source_clock_domain_id": trajectory.clock_domain_id,
        "created_at": trajectory.created_at,
        "action_schema_hash": trajectory.action_schema_hash,
        "component_coverage": trajectory.component_coverage,
        "effective_from": trajectory.effective_from,
        "effective_until": trajectory.effective_until,
        "source_fact_ids": trajectory.source_fact_ids,
        "source_fact_hashes": trajectory.source_fact_hashes,
        "source_activation_id": trajectory.source_activation_id,
        "source_activation_hash": trajectory.source_activation_hash,
        "source_immutable_through": trajectory.source_immutable_through,
        "coordination_group_id": trajectory.coordination_group_id,
        "keyframes": tuple(
            {
                "keyframe_id": keyframe.keyframe_id,
                "target_effective_at": keyframe.target_effective_at,
                "effective_until": keyframe.effective_until,
                "source_index": keyframe.source_index,
                "source_fact_ids": keyframe.source_fact_ids,
                "source_fact_hashes": keyframe.source_fact_hashes,
                "action": keyframe.action.semantic_payload(),
                "semantic_content_hash": keyframe.semantic_content_hash,
            }
            for keyframe in trajectory.keyframes
        ),
        "semantic_content_hash": trajectory.semantic_content_hash,
    }


def project_component_action_trajectory(
    trajectory: ComponentActionTrajectory,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str | None,
) -> ControlTraceFact:
    """Project a source-owned trajectory without changing its time domain."""

    payload = component_action_trajectory_payload(trajectory)
    return ControlTraceFact.create(
        fact_id=trajectory.fact_id,
        event_type="component_action_trajectory",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=trajectory.created_at,
        clock_domain_id=trajectory.clock_domain_id,
        payload=payload,
    )


__all__ = [
    "component_action_trajectory_payload",
    "project_component_action_trajectory",
]

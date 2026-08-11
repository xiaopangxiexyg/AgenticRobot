"""Pure ControlTrace projection for lower-owned policy emissions.

The projector lives beside ``PolicyActionEmission`` so simulated and real
application compositions can record the exact queue-pop fact without importing
one another or the Control Station.  It performs no queue access, inference,
admission, safety, or dispatch work.
"""

from __future__ import annotations

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.proposal import (
    robot_action_semantic_payload,
)

from .runtime import PolicyActionEmission
from .source_projection import policy_source_projection_from_metadata


ROBOT_ACTION_RECORD_HASH_DOMAIN_V1 = "dexteleop.robot_action_record.v1"


def policy_action_emission_payload(
    emission: PolicyActionEmission,
) -> dict[str, object]:
    """Return the registered v2.4 payload for one immutable queue-pop fact."""

    if not isinstance(emission, PolicyActionEmission):
        raise TypeError("emission must be a PolicyActionEmission.")
    action = emission.pre_safety_robot_action
    action_payload = None if action is None else robot_action_semantic_payload(action)
    payload = {
        "emission_id": emission.emission_id,
        "admission_mode": emission.admission_mode,
        "policy_node_id": emission.policy_node_id,
        "inference_id": emission.inference_id,
        "policy_output_id": emission.policy_output_id,
        "chunk_id": emission.chunk_id,
        "chunk_keyframe_index": emission.chunk_keyframe_index,
        "chunk_size": emission.chunk_size,
        "pre_safety_action": action_payload,
        "pre_safety_action_hash": (
            None
            if action_payload is None
            else canonical_content_hash(
                action_payload,
                domain=ROBOT_ACTION_RECORD_HASH_DOMAIN_V1,
            )
        ),
        "representation_trace_id": (
            emission.representation_trace_id
        ),
        "pre_safety_faults": emission.pre_safety_faults,
        "produced_at": emission.produced_at,
        "produced_clock_domain_id": emission.produced_clock_domain_id,
        "emitted_at": emission.emitted_at,
        "target_effective_at": emission.target_effective_at,
        "effective_from": emission.effective_from,
        "effective_until": emission.effective_until,
        "effective_clock_domain_id": emission.effective_clock_domain_id,
        "schedule_spec_id": emission.schedule_spec_id,
        "schedule_spec_hash": emission.schedule_spec_hash,
        "queue_len_before": emission.queue_len_before,
        "queue_len_after": emission.queue_len_after,
        "queue_pop_count": emission.queue_pop_count,
        "semantic_content_hash": emission.semantic_content_hash,
    }
    if emission.canonical_action is not None:
        payload["action_payload_kind"] = emission.action_payload_kind
        payload["canonical_policy_action_hash"] = (
            emission.canonical_action.semantic_content_hash
        )
        payload["canonicalization_receipt_hash"] = (
            emission.canonical_action.canonicalization_receipt.semantic_content_hash
        )
    if emission.arm_hold_resolution is not None:
        payload["arm_hold_resolution"] = (
            emission.arm_hold_resolution.to_dict(include_hash=True)
        )
        model_action = emission.model_decoded_absolute_action
        payload["model_decoded_action_hash"] = (
            None
            if model_action is None
            else canonical_content_hash(
                {
                    "vector": model_action.vector,
                    "arm_joint_targets": model_action.arm_joint_targets,
                    "arm_hold_prediction": (
                        None
                        if model_action.arm_hold_prediction is None
                        else model_action.arm_hold_prediction.to_dict()
                    ),
                },
                domain="dexteleop.model_decoded_action.v1",
            )
        )
    return payload


def project_policy_action_emission(
    emission: PolicyActionEmission,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> tuple[ControlTraceFact, ...]:
    """Project an emission and its independent source-validation fact."""

    payload = policy_action_emission_payload(emission)
    common = {
        "runtime_session_id": str(runtime_session_id),
        "control_domain_id": str(control_domain_id),
        "embodiment_instance_id": str(embodiment_instance_id),
        "occurred_at": float(emission.emitted_at),
        "clock_domain_id": str(clock_domain_id),
    }
    emission_fact = ControlTraceFact.create(
        fact_id=emission.emission_id,
        event_type="policy_action_emission",
        payload=payload,
        **common,
    )
    validation = emission.source_validation
    validation_id = f"source-validation:{emission.emission_id}"
    accepted = False if validation is None else bool(validation.accepted)
    reasons = (
        ("source_validation_unavailable", *emission.pre_safety_faults)
        if validation is None
        else validation.reasons
    )
    validation_fact = ControlTraceFact.create(
        fact_id=validation_id,
        event_type="source_validation",
        payload={
            "validation_id": validation_id,
            "source_event_id": emission.emission_id,
            "accepted": accepted,
            "reasons": reasons,
        },
        **common,
    )
    projection_fact = project_policy_action_projection(
        emission,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        clock_domain_id=clock_domain_id,
    )
    return (
        (emission_fact, validation_fact)
        if projection_fact is None
        else (emission_fact, validation_fact, projection_fact)
    )


def project_policy_action_projection(
    emission: PolicyActionEmission,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact | None:
    """Record source values and anchor semantics without copying model internals."""

    if not isinstance(emission, PolicyActionEmission):
        raise TypeError("emission must be a PolicyActionEmission.")
    decoded = emission.decoded_absolute_action
    if decoded is None:
        return None
    projection = policy_source_projection_from_metadata(decoded.metadata)
    if projection is None:
        return None
    fact_id = f"policy-action-projection:{emission.emission_id}"
    return ControlTraceFact.create(
        fact_id=fact_id,
        event_type="policy_action_projection",
        runtime_session_id=str(runtime_session_id),
        control_domain_id=str(control_domain_id),
        embodiment_instance_id=str(embodiment_instance_id),
        occurred_at=float(emission.emitted_at),
        clock_domain_id=str(clock_domain_id),
        payload={
            "emission_id": emission.emission_id,
            "emission_hash": emission.semantic_content_hash,
            "policy_node_id": emission.policy_node_id,
            "inference_id": emission.inference_id,
            "policy_output_id": emission.policy_output_id,
            "projection": projection.to_dict(),
        },
    )


__all__ = [
    "policy_action_emission_payload",
    "project_policy_action_emission",
    "project_policy_action_projection",
]

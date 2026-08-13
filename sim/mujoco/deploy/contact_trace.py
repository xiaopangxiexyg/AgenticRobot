"""Pure ControlTrace projection for MuJoCo active-contact guard evidence."""

from __future__ import annotations

from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact


MUJOCO_CONTACT_GUARD_ID_V1 = "mujoco_forbidden_contact_hard_penetration_v1"
MUJOCO_CONTACT_GUARD_RESULT_HASH_DOMAIN_V1 = (
    "dexteleop.mujoco.contact_guard_result.v1"
)


def project_mujoco_forbidden_contact_guard_result(
    observation: Mapping[str, object],
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    """Persist one terminal, lower-owned forbidden-contact observation."""

    evidence = dict(observation)
    if not bool(evidence.get("available")) or bool(evidence.get("safe", True)):
        raise ValueError(
            "MuJoCo contact guard fact requires an available unsafe observation."
        )
    observed_at = float(evidence["observed_at"])
    physics_tick = int(evidence["physics_tick"])
    diagnostics = {
        name: evidence.get(name)
        for name in (
            "minimum_distance_m",
            "penetration_depth_m",
            "hard_penetration_limit_m",
            "geom1",
            "geom2",
            "body1",
            "body2",
            "pair_class",
            "collision_category",
        )
    }
    semantic_payload = {
        "guard_id": MUJOCO_CONTACT_GUARD_ID_V1,
        "observed_at": observed_at,
        "physics_tick": physics_tick,
        "status": "unsafe_terminal",
        "reason_code": "active_forbidden_contact_hard_penetration",
        "diagnostics": diagnostics,
    }
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=MUJOCO_CONTACT_GUARD_RESULT_HASH_DOMAIN_V1,
    )
    guard_result_id = f"backend-guard-result:{semantic_hash}"
    return ControlTraceFact.create(
        fact_id=guard_result_id,
        event_type="backend_guard_result",
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=observed_at,
        clock_domain_id=clock_domain_id,
        payload={
            "guard_result_id": guard_result_id,
            **semantic_payload,
            "semantic_content_hash": semantic_hash,
        },
    )


__all__ = [
    "MUJOCO_CONTACT_GUARD_ID_V1",
    "project_mujoco_forbidden_contact_guard_result",
]

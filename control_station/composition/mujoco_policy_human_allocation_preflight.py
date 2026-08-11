"""Read-only lower-capability preflight for allocation-aware Sim-HIL."""

from __future__ import annotations

from typing import Any, Mapping

from .mujoco_policy_human_allocation import (
    MujocoPolicyHumanAllocationProfileBundle,
)
from .mujoco_sim_hil import (
    MUJOCO_SIM_HIL_READINESS_FIELDS_V1,
    MujocoSimHILPreflightReport,
    MujocoSimHILReadinessEvidence,
)


_POLICY_ONLY_READINESS_FIELDS = (
    "policy_source_ready",
    "policy_chunk_ttl_ready",
    "measured_state_ready",
)


def allocation_required_readiness_fields(
    bundle: MujocoPolicyHumanAllocationProfileBundle,
) -> tuple[str, ...]:
    """Return the exact source readiness required for profile activation."""

    if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
        raise TypeError("bundle must be typed.")
    return (
        MUJOCO_SIM_HIL_READINESS_FIELDS_V1
        if bundle.config.require_human_roles_at_start
        else _POLICY_ONLY_READINESS_FIELDS
    )


def allocation_readiness_snapshot(
    readiness_provider: Any,
    *,
    now: float,
) -> MujocoSimHILReadinessEvidence:
    """Turn provider failures into explicit failed evidence without mutation."""

    try:
        readiness = readiness_provider.snapshot(now=float(now))
    except Exception as error:  # noqa: BLE001 - preflight is fail-closed.
        detail = f"provider_error:{type(error).__name__}:{error}"
        return MujocoSimHILReadinessEvidence(
            **{name: False for name in MUJOCO_SIM_HIL_READINESS_FIELDS_V1},
            detail_by_check={
                name: detail for name in MUJOCO_SIM_HIL_READINESS_FIELDS_V1
            },
        )
    if isinstance(readiness, MujocoSimHILReadinessEvidence):
        return readiness
    return MujocoSimHILReadinessEvidence(
        **{name: False for name in MUJOCO_SIM_HIL_READINESS_FIELDS_V1},
        detail_by_check={
            name: "provider_returned_invalid_evidence"
            for name in MUJOCO_SIM_HIL_READINESS_FIELDS_V1
        },
    )


def build_mujoco_policy_human_allocation_preflight(
    *,
    runtime: Any,
    bundle: MujocoPolicyHumanAllocationProfileBundle,
    binding_factory: Any,
    readiness_provider: Any,
    now: float,
) -> MujocoSimHILPreflightReport:
    """Verify lower execution capability and all-required source readiness."""

    readiness = allocation_readiness_snapshot(readiness_provider, now=now)
    contracts = bundle.composition_contracts
    strategy = runtime.composition_strategy
    catalog = runtime.catalog
    profile = bundle.authority_profile
    human_binding = getattr(binding_factory.human_source, "binding_receipt", None)
    policy_binding = binding_factory.policy_input_adapter.binding
    required_readiness_fields = allocation_required_readiness_fields(bundle)
    required_readiness_failures = tuple(
        name
        for name in required_readiness_fields
        if not bool(getattr(readiness, name))
    )
    checks: Mapping[str, bool] = {
        "profile_enabled": bool(bundle.config.enabled),
        "pose_execution": bool(runtime.pose_execution_enabled),
        # This is a lower-owned capability check, never pre-mix safety.
        "lower_shared_safety": bool(runtime.explicit_shared_safety),
        "formal_strategy": bool(
            strategy is not None
            and strategy.spec.semantic_content_hash
            == contracts.strategy_spec.semantic_content_hash
        ),
        "compound_scope": bool(
            tuple(item.component_id for item in catalog.components)
            == profile.component_coverage
            and catalog.selection_group_ids == profile.selection_groups
        ),
        "single_execution_owner": bool(
            runtime.config.enable_composer_authority
            and not runtime.config.enable_primary_policy_authority
            and not runtime.config.enable_replay_authority
        ),
        "human_passive": bool(
            getattr(human_binding, "rig_absent", False)
            and not getattr(human_binding, "authoritative", True)
        ),
        "policy_contributor_only": bool(
            not runtime.config.enable_primary_policy_authority
        ),
        "policy_receiver_binding": bool(
            policy_binding.runtime_session_id == runtime.config.runtime_session_id
            and policy_binding.control_domain_id == runtime.config.control_domain_id
            and policy_binding.producer_id == runtime.config.producer_id
            and policy_binding.producer_epoch == runtime.config.producer_epoch
        ),
        "composer_receiver_binding": bool(
            profile.producer_id == runtime.config.composer_producer_id
        ),
        "required_readiness": not required_readiness_failures,
    }
    reasons = [
        f"check_failed:{name}" for name, passed in checks.items() if not passed
    ]
    reasons.extend(
        f"readiness_failed:{name}" for name in required_readiness_failures
    )
    for name in required_readiness_failures:
        detail = readiness.detail_by_check.get(name)
        if detail:
            reasons.append(f"readiness_detail:{name}:{detail}")
    return MujocoSimHILPreflightReport(
        readiness=readiness,
        checks=checks,
        reasons=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "allocation_required_readiness_fields",
    "allocation_readiness_snapshot",
    "build_mujoco_policy_human_allocation_preflight",
]

from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ComponentAction,
    ConservativeCompoundAuthorityProfile,
    IMMEDIATE_QREF_SIM_V1,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    build_robot_action_v1_runtime_schema,
)
from sim.mujoco.deploy.composition_promotion import (
    MujocoCompositionPromotionPolicy,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _catalog(*, catalog_id: str = "fixture-mujoco-qpos") -> ActionComponentCatalog:
    return ActionComponentCatalog(
        catalog_id=catalog_id,
        action_space=BIMANUAL_QPOS_V1,
        components=(
            ActionComponentSpec(
                "left_arm",
                "arms",
                "left_arm",
                coupled=True,
            ),
            ActionComponentSpec(
                "right_arm",
                "arms",
                "right_arm",
                coupled=True,
            ),
        ),
    )


def _action(schema, *component_ids: str) -> ComponentAction:
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            component_id: np.zeros(6, dtype=np.float32)
            for component_id in component_ids
        },
    )


def _compound_profile() -> ConservativeCompoundAuthorityProfile:
    return ConservativeCompoundAuthorityProfile(
        profile_id="fixture-compound-authority",
        producer_id="policy-composer",
        component_coverage=("left_arm", "right_arm"),
        selection_groups=("arms",),
        composition_spec_hash="fixture-composition-spec-hash",
        component_rule_set_hash="fixture-component-rule-set-hash",
        formal_strategy_spec_hash="fixture-formal-strategy-spec-hash",
    )


def _policy(
    *,
    compound_profile: ConservativeCompoundAuthorityProfile | None = None,
    reference_activation_profile_id: str = IMMEDIATE_QREF_SIM_V1,
):
    catalog = _catalog()
    schema = build_robot_action_v1_runtime_schema(component_catalog=catalog)
    return (
        MujocoCompositionPromotionPolicy(
            catalog=catalog,
            action_schema=schema,
            compound_authority_profile=compound_profile,
            decision_hz=20.0,
            proposal_horizon_sec=0.2,
            reference_activation_profile_id=reference_activation_profile_id,
        ),
        schema,
    )


def test_ordinary_promotion_scope_follows_exact_candidate_and_target_catalog() -> None:
    policy, schema = _policy()
    action = _action(schema, "left_arm", "right_arm")

    assert policy.validate_scope(
        component_coverage=None,
        selection_groups=None,
        action=action,
    ) == (("left_arm", "right_arm"), ("arms",))

    with pytest.raises(ValueError, match="partially cover coupled"):
        policy.validate_scope(
            component_coverage=None,
            selection_groups=None,
            action=_action(schema, "left_arm"),
        )
    with pytest.raises(ValueError, match="follow catalog order"):
        policy.validate_scope(
            component_coverage=("right_arm", "left_arm"),
            selection_groups=("arms",),
            action=action,
        )
    with pytest.raises(ValueError, match="must be provided together"):
        policy.validate_scope(
            component_coverage=("left_arm", "right_arm"),
            selection_groups=None,
            action=action,
        )


def test_promotion_policy_rejects_another_runtime_action_schema() -> None:
    policy, _schema = _policy()
    other_catalog = _catalog(catalog_id="another-mujoco-qpos")
    other_schema = build_robot_action_v1_runtime_schema(
        component_catalog=other_catalog
    )

    with pytest.raises(ValueError, match="another runtime action schema"):
        policy.validate_scope(
            component_coverage=None,
            selection_groups=None,
            action=_action(other_schema, "left_arm", "right_arm"),
        )


def test_compound_promotion_requires_the_profile_scope() -> None:
    policy, schema = _policy(compound_profile=_compound_profile())

    with pytest.raises(
        ValueError,
        match="exactly match the composed action components",
    ):
        policy.validate_scope(
            component_coverage=None,
            selection_groups=None,
            action=_action(schema, "left_arm"),
        )


def test_promotion_deadline_preserves_scheduled_compound_semantics() -> None:
    ordinary_scheduled, _schema = _policy(
        reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    )
    compound_scheduled, _schema = _policy(
        compound_profile=_compound_profile(),
        reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    )
    compound_immediate, _schema = _policy(
        compound_profile=_compound_profile(),
        reference_activation_profile_id=IMMEDIATE_QREF_SIM_V1,
    )

    assert ordinary_scheduled.requested_effective_until(
        target_effective_at=10.0
    ) == pytest.approx(10.2)
    assert compound_scheduled.requested_effective_until(
        target_effective_at=10.0
    ) == pytest.approx(10.05)
    assert compound_immediate.requested_effective_until(
        target_effective_at=10.0
    ) == pytest.approx(10.2)

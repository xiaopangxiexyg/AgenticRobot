from __future__ import annotations

import math

import numpy as np
import pytest

from control_composition import (
    GuaranteedNextComponentAction,
    build_temporally_sampled_composition_input,
    sample_component_action_source,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)


def _payload(
    *,
    schema_id: str,
    family: str,
    dtype: str,
    coordinates: tuple[str, ...],
    units: tuple[str, ...],
    semantics: str,
    reference_frame: str | None = None,
    controlled_frame: str | None = None,
) -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id=schema_id,
        schema_version="development",
        representation_family=family,
        dtype=dtype,
        shape=(len(coordinates),),
        coordinate_names=coordinates,
        units=units,
        value_semantics=semantics,
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
    )


def _action_schema() -> ComponentActionSchema:
    pose = _payload(
        schema_id="arm-pose",
        family="cartesian_pose",
        dtype="float64",
        coordinates=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        semantics="absolute_target",
        reference_frame="world",
        controlled_frame="tool",
    )
    hand = _payload(
        schema_id="hand-qpos",
        family="joint_position",
        dtype="float64",
        coordinates=("finger_0", "finger_1"),
        units=("rad", "rad"),
        semantics="absolute_target",
    )
    mode = _payload(
        schema_id="mode",
        family="discrete_mode",
        dtype="int64",
        coordinates=("mode",),
        units=("1",),
        semantics="mode",
    )
    payloads = {"arm": pose, "hand": hand, "mode": mode}
    catalog = ActionComponentCatalogV2(
        catalog_id="source-temporal-fixture",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            )
            for component_id, payload in payloads.items()
        ),
    )
    return ComponentActionSchema(
        schema_id="source-temporal-fixture",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _actions() -> tuple[ComponentAction, ComponentAction]:
    schema = _action_schema()
    current = ComponentAction(
        action_schema=schema,
        values_by_component={
            "arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.0, 2.0],
            "mode": [0],
        },
    )
    next_action = ComponentAction(
        action_schema=schema,
        values_by_component={
            "arm": [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "hand": [2.0, 4.0],
            "mode": [1],
        },
    )
    return current, next_action


def _guarantee(next_action: ComponentAction) -> GuaranteedNextComponentAction:
    return GuaranteedNextComponentAction(
        guarantee_id="chunk:4:step:1:guaranteed",
        source_fact_id="chunk:4:step:1",
        source_fact_hash="chunk-step-1-hash",
        action=next_action,
        target_effective_at=12.0,
        clock_domain_id="control.monotonic",
    )


def test_missing_guaranteed_next_preserves_current_step() -> None:
    current, _ = _actions()
    sample = sample_component_action_source(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="chunk:4:step:0",
        current_source_fact_hash="chunk-step-0-hash",
        target_effective_at=11.0,
        clock_domain_id="control.monotonic",
        guaranteed_next_step=None,
    )

    assert sample.action is current
    assert sample.derivation_kind == "current_step_fallback"
    assert sample.interpolation_alpha == 0.0
    assert sample.source_fact_ids == (
        sample.sample_id,
        "chunk:4:step:0",
    )
    assert sample.next_step_guarantee_id is None
    assert sample.interpolation_contract_hash is None


def test_guaranteed_next_interpolates_pose_qpos_and_holds_discrete_mode() -> None:
    current, next_action = _actions()
    sample = sample_component_action_source(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="chunk:4:step:0",
        current_source_fact_hash="chunk-step-0-hash",
        target_effective_at=11.0,
        clock_domain_id="control.monotonic",
        guaranteed_next_step=_guarantee(next_action),
    )

    assert sample.derivation_kind == "interpolated_guaranteed_next"
    assert sample.interpolation_alpha == 0.5
    np.testing.assert_allclose(
        sample.action.value("arm").values[:3],
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        sample.action.value("arm").values[3:],
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        atol=1e-7,
    )
    np.testing.assert_allclose(sample.action.value("hand").values, [1.0, 3.0])
    np.testing.assert_array_equal(sample.action.value("mode").values, [0])
    assert sample.source_fact_ids == (
        sample.sample_id,
        "chunk:4:step:0",
        "chunk:4:step:1",
    )
    assert sample.interpolation_contract_hash is not None


def test_guaranteed_next_endpoint_yields_exact_next_values() -> None:
    current, next_action = _actions()
    sample = sample_component_action_source(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="chunk:4:step:0",
        current_source_fact_hash="chunk-step-0-hash",
        target_effective_at=12.0,
        clock_domain_id="control.monotonic",
        guaranteed_next_step=_guarantee(next_action),
    )

    assert sample.derivation_kind == "guaranteed_next_endpoint"
    assert sample.interpolation_alpha == 1.0
    assert sample.action.semantic_content_hash == next_action.semantic_content_hash
    np.testing.assert_array_equal(sample.action.value("mode").values, [1])


def test_claimed_guaranteed_next_fails_on_invalid_time_clock_or_schema() -> None:
    current, next_action = _actions()
    base = _guarantee(next_action)
    with pytest.raises(ValueError, match="another clock"):
        sample_component_action_source(
            current_action=current,
            current_target_effective_at=10.0,
            current_source_fact_id="current",
            current_source_fact_hash="current-hash",
            target_effective_at=11.0,
            clock_domain_id="other-clock",
            guaranteed_next_step=base,
        )
    with pytest.raises(ValueError, match="after the guaranteed-next"):
        sample_component_action_source(
            current_action=current,
            current_target_effective_at=10.0,
            current_source_fact_id="current",
            current_source_fact_hash="current-hash",
            target_effective_at=12.1,
            clock_domain_id="control.monotonic",
            guaranteed_next_step=base,
        )

    other_current, _ = _actions()
    other_schema = ComponentActionSchema(
        schema_id="other-schema",
        schema_version="development",
        component_catalog=other_current.action_schema.component_catalog,
        payload_schema_by_component=(
            other_current.action_schema.payload_schema_by_component
        ),
    )
    mismatched = GuaranteedNextComponentAction(
        guarantee_id="mismatched",
        source_fact_id="mismatched",
        source_fact_hash="mismatched-hash",
        action=ComponentAction(
            action_schema=other_schema,
            values_by_component={
                component_id: value.values
                for component_id, value in next_action.values_by_component.items()
            },
        ),
        target_effective_at=12.0,
        clock_domain_id="control.monotonic",
    )
    with pytest.raises(ValueError, match="another action schema"):
        sample_component_action_source(
            current_action=current,
            current_target_effective_at=10.0,
            current_source_fact_id="current",
            current_source_fact_hash="current-hash",
            target_effective_at=11.0,
            clock_domain_id="control.monotonic",
            guaranteed_next_step=mismatched,
        )


def test_temporal_sample_builds_narrow_composition_input_with_lineage() -> None:
    current, next_action = _actions()
    result = build_temporally_sampled_composition_input(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="chunk:4:step:0",
        current_source_fact_hash="chunk-step-0-hash",
        target_effective_at=10.5,
        effective_until=10.6,
        guaranteed_next_step=_guarantee(next_action),
        runtime_session_id="session",
        control_domain_id="domain",
        producer_id="policy",
        producer_epoch=3,
        sequence=8,
        input_role="policy",
        produced_at=10.49,
        clock_domain_id="control.monotonic",
        policy_output_id="policy-output:4",
    )

    assert result.draft.action.semantic_content_hash == result.sample.action_hash
    assert result.draft.effective_from == 10.5
    assert result.draft.effective_until == 10.6
    assert result.draft.authority_candidate is False
    assert result.draft.source_fact_ids == (
        result.sample.sample_id,
        "chunk:4:step:0",
        "chunk:4:step:1",
    )
    assert result.draft.audit_metadata["source_temporal_sample_hash"] == (
        result.sample.semantic_content_hash
    )
    assert result.draft.audit_metadata["source_interpolation_alpha"] == 0.25

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition.trajectory import (
    ResidualTemporalWarpSpec,
    activate_residual_session,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
    prepare_trajectory_target,
    sample_residual_for_trajectory_target,
)
from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
)
from robot_motion.runtime.control_admission import ControlCut


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schemas() -> tuple[ComponentActionSchema, ComponentActionSchema]:
    absolute_payloads = {
        "arm": ActionPayloadSchema(
            schema_id="warp-arm-absolute",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "hand": ActionPayloadSchema(
            schema_id="warp-hand-absolute",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(3,),
            coordinate_names=("finger_0", "finger_1", "finger_2"),
            units=("rad", "rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    residual_payloads = {
        "arm": ActionPayloadSchema(
            schema_id="warp-arm-relative",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="world",
            controlled_frame="tool",
            semantic_parameters={
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            },
        ),
        "hand": ActionPayloadSchema(
            schema_id="warp-hand-relative",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(3,),
            coordinate_names=("finger_0", "finger_1", "finger_2"),
            units=("rad", "rad", "rad"),
            value_semantics="relative_delta",
            semantic_parameters={"application": "elementwise_additive"},
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="warp-fixture",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    absolute_payloads[component_id].semantic_content_hash,
                    residual_payloads[component_id].semantic_content_hash,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="warp-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=absolute_payloads,
        ),
        ComponentActionSchema(
            schema_id="warp-relative",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=residual_payloads,
        ),
    )


def _state():
    absolute_schema, _ = _schemas()
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=10.0 + index * 0.1,
            effective_until=10.1 + index * 0.1,
            action=ComponentAction(
                action_schema=absolute_schema,
                values_by_component={
                    "arm": (
                        float(index),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    "hand": (0.0, 0.0, 0.0),
                },
            ),
            source_index=index,
            source_fact_ids=(f"nominal:{index}",),
            source_fact_hashes=(f"nominal-hash:{index}",),
        )
        for index in range(3)
    )
    trajectory = ComponentActionTrajectory(
        trajectory_id="nominal",
        revision=1,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy",
        clock_domain_id="control.monotonic",
        created_at=9.9,
        keyframes=keyframes,
        source_fact_ids=("nominal-plan",),
        source_fact_hashes=("nominal-plan-hash",),
        source_activation_id="virtual-policy-activation:1",
        source_activation_hash="virtual-policy-activation-hash:1",
    )
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="editor"),
        trajectory,
        offer_sequence=1,
    )
    scope = build_composition_activation_scope(
        operation_spec_id="trajectory-warp-test",
        operation_spec_hash="trajectory-warp-test-hash",
        session_epoch=1,
        activated_at=9.7,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.6,
        required_roles=("baseline", "residual"),
    )
    request = ResidualAnchorRequest.for_activation(
        scope,
        request_generation=0,
        source_role="residual",
        component_ids=("arm", "hand"),
        source_binding_id="virtual-human:1",
        source_binding_hash="virtual-human-hash:1",
        absolute_action_schema_hash=trajectory.action_schema_hash,
        relative_action_schema_hash=_schemas()[1].semantic_content_hash,
        requested_at=9.75,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="virtual-human-action:1",
        source_action_hash="virtual-human-action-hash:1",
        source_available_at=9.8,
        captured_at=9.81,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"virtual-human-{component_id}:1",
                source_sample_hash=f"virtual-human-{component_id}-hash:1",
                absolute_payload_schema_hash=(
                    trajectory.keyframes[0]
                    .action.value(component_id)
                    .payload_schema.semantic_content_hash
                ),
                anchor_id=f"virtual-human-{component_id}-anchor:1",
                anchor_hash=f"virtual-human-{component_id}-anchor-hash:1",
            )
            for component_id in ("arm", "hand")
        ),
    )
    return activate_residual_session(
        state,
        residual_session_id="human:1",
        activation_scope=scope,
        control_cut=ControlCut(
            runtime_session_id="test-runtime",
            control_domain_id="test-control",
            clock_domain_id="control.monotonic",
            clock_reset_epoch=0,
            cutoff_at=9.9,
            cutoff_admission_sequence=20,
        ),
        residual_anchor_receipt=receipt,
        first_editable_target_at=10.0,
    )


def _residual(*, rotation_degrees: float = 0.0) -> ComponentAction:
    _, residual_schema = _schemas()
    return ComponentAction(
        action_schema=residual_schema,
        values_by_component={
            "arm": (
                0.2,
                -0.1,
                0.0,
                *Rotation.from_euler(
                    "z",
                    rotation_degrees,
                    degrees=True,
                ).as_quat(),
            ),
            "hand": (0.2, -0.2, 0.4),
        },
    )


def _sample(state, target: float, residual: ComponentAction):
    preparation = prepare_trajectory_target(
        state,
        target_effective_at=target,
    )
    anchor_receipt = state.active_residual_anchor_receipt
    assert anchor_receipt is not None
    return sample_residual_for_trajectory_target(
        state,
        preparation,
        residual_action=residual,
        residual_fact_id="human-residual:1",
        residual_fact_hash="human-residual-hash:1",
        residual_anchor_receipt_id=anchor_receipt.receipt_id,
        residual_anchor_receipt_hash=anchor_receipt.semantic_content_hash,
        warp_spec=ResidualTemporalWarpSpec(
            spec_id="test-ramp",
            ramp_duration_sec=0.2,
        ),
    )


def test_ramp_and_hold_scales_pose_translation_and_arbitrary_qpos() -> None:
    state = _state()
    residual = _residual()
    start = _sample(state, 10.0, residual)
    middle = _sample(state, 10.1, residual)
    end = _sample(state, 10.2, residual)

    assert start.weight == 0.0
    assert middle.weight == pytest.approx(0.5)
    assert end.weight == 1.0
    np.testing.assert_allclose(
        start.action.value("arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        middle.action.value("arm").values[:3],
        (0.1, -0.05, 0.0),
    )
    np.testing.assert_allclose(
        middle.action.value("hand").values,
        (0.1, -0.1, 0.2),
    )
    np.testing.assert_allclose(
        end.action.value("hand").values,
        residual.value("hand").values,
    )


def test_temporal_sampling_is_state_valued_and_replay_deterministic() -> None:
    state = _state()
    residual = _residual()
    first = _sample(state, 10.1, residual)
    second = _sample(state, 10.1, residual)

    assert first.action.semantic_content_hash == second.action.semantic_content_hash
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.residual_action_hash == residual.semantic_content_hash
    assert first.active_trajectory_hash == (
        state.active_trajectory.semantic_content_hash
    )
    assert first.editing_basis_hash == state.publication_basis_hash


def test_non_identity_pose_rotation_is_rejected_until_full_pose_gate() -> None:
    state = _state()
    with pytest.raises(ValueError, match="rotation temporal warp"):
        _sample(state, 10.1, _residual(rotation_degrees=15.0))


def test_partial_qpos_residual_does_not_require_unrelated_pose_values() -> None:
    state = _state()
    _, residual_schema = _schemas()
    qpos_only = ComponentAction(
        action_schema=residual_schema,
        values_by_component={"hand": (0.3, 0.0, -0.3)},
    )
    sample = _sample(state, 10.1, qpos_only)

    assert sample.action.component_coverage == ("hand",)
    np.testing.assert_allclose(
        sample.action.value("hand").values,
        (0.15, 0.0, -0.15),
    )


def test_shadow_offer_does_not_invalidate_residual_sample_for_latched_base() -> None:
    state = _state()
    preparation = prepare_trajectory_target(
        state,
        target_effective_at=10.1,
    )
    shadow = _state().active_trajectory
    with_shadow = offer_nominal_trajectory(
        state,
        shadow,
        offer_sequence=2,
    )
    anchor_receipt = with_shadow.active_residual_anchor_receipt
    assert anchor_receipt is not None
    sample = sample_residual_for_trajectory_target(
        with_shadow,
        preparation,
        residual_action=_residual(),
        residual_fact_id="human-residual:1",
        residual_fact_hash="human-residual-hash:1",
        residual_anchor_receipt_id=anchor_receipt.receipt_id,
        residual_anchor_receipt_hash=anchor_receipt.semantic_content_hash,
        warp_spec=ResidualTemporalWarpSpec(
            spec_id="test-ramp",
            ramp_duration_sec=0.2,
        ),
    )

    assert sample.weight == pytest.approx(0.5)
    assert sample.active_trajectory_hash == (
        state.active_trajectory.semantic_content_hash
    )

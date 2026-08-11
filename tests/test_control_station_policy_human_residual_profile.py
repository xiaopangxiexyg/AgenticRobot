from __future__ import annotations

import numpy as np
import pytest

from control_composition import build_composition_activation_scope
from control_station.composition.mujoco_policy_human_residual import (
    MujocoPolicyHumanResidualProfileConfig,
    build_mujoco_policy_human_residual_profile,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)


def _schema() -> ComponentActionSchema:
    payloads = {
        "arm": ActionPayloadSchema(
            schema_id="virtual-arm",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="arm_base",
            controlled_frame="tool_tcp",
        ),
        "hand": ActionPayloadSchema(
            schema_id="virtual-hand",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="virtual-policy-human-residual",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    payloads[component_id].semantic_content_hash,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )
    return ComponentActionSchema(
        schema_id="virtual-policy-human-residual",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def _action(x: float, hand: tuple[float, float]) -> ComponentAction:
    return ComponentAction(
        action_schema=_schema(),
        values_by_component={
            "arm": (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": hand,
        },
    )


def _build(*, enabled: bool):
    return build_mujoco_policy_human_residual_profile(
        absolute_action_schema=_schema(),
        frame_graph=CoordinateFrameGraph(
            graph_id="virtual-policy-human-residual",
            frames=(
                CoordinateFrameSpec("arm/base"),
                CoordinateFrameSpec("arm/tool"),
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={"arm": "arm/base"},
        concrete_controlled_frame_by_component={"arm": "arm/tool"},
        source_binding_id="virtual-human-source:1",
        source_binding_hash="virtual-human-source-hash:1",
        composer_producer_id="virtual-composer",
        proposal_ttl_sec=0.1,
        config=MujocoPolicyHumanResidualProfileConfig(enabled=enabled),
    )


def _activation(bundle, *, epoch: int, at: float):
    return build_composition_activation_scope(
        operation_spec_id=bundle.config.profile_id,
        operation_spec_hash=bundle.sim_hil_contracts.semantic_content_hash,
        session_epoch=epoch,
        activated_at=at,
        clock_domain_id="control.monotonic",
        input_barrier_at=at,
        required_roles=(
            bundle.relative_contracts.composition_spec.required_roles
        ),
    )


def test_profile_is_default_off_and_does_not_capture_implicitly() -> None:
    bundle = _build(enabled=False)

    assert bundle.config.enabled is False
    with pytest.raises(RuntimeError, match="default-off"):
        bundle.input_adapter.request_anchor(
            _activation(bundle, epoch=1, at=10.0),
            requested_at=10.0,
        )
    assert bundle.input_adapter.anchor is None


def test_first_post_barrier_action_captures_anchor_and_emits_identity_residual() -> None:
    bundle = _build(enabled=True)
    request = bundle.input_adapter.request_anchor(
        _activation(bundle, epoch=1, at=10.0),
        requested_at=10.0,
    )
    first_action = _action(0.2, (0.4, 0.6))
    first = bundle.input_adapter.process_absolute_action(
        first_action,
        source_action_id="human:0",
        source_action_hash=first_action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        observed_at=10.02,
        observed_clock_domain_id="control.monotonic",
    )

    assert first.status == "anchor_captured_relative_action_ready"
    np.testing.assert_allclose(
        first.relative_action.value("arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        first.relative_action.value("hand").values,
        (0.0, 0.0),
    )
    assert bundle.input_adapter.ready
    assert first.anchor_request_id == request.request_id

    second_action = _action(0.3, (0.5, 0.2))
    second = bundle.input_adapter.process_absolute_action(
        second_action,
        source_action_id="human:1",
        source_action_hash=second_action.semantic_content_hash,
        source_available_at=10.1,
        measured_at=20.1,
        measured_clock_domain_id="human.source",
        observed_at=10.11,
        observed_clock_domain_id="control.monotonic",
    )

    assert second.status == "relative_action_ready"
    np.testing.assert_allclose(
        second.relative_action.value("arm").values[:3],
        (0.1, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        second.relative_action.value("hand").values,
        (0.1, -0.4),
    )


def test_recenter_clears_ready_state_and_next_action_only_reanchors() -> None:
    bundle = _build(enabled=True)
    bundle.input_adapter.request_anchor(
        _activation(bundle, epoch=1, at=10.0),
        requested_at=10.0,
    )
    first_action = _action(0.2, (0.4, 0.6))
    bundle.input_adapter.process_absolute_action(
        first_action,
        source_action_id="human:0",
        source_action_hash=first_action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        observed_at=10.02,
        observed_clock_domain_id="control.monotonic",
    )
    assert bundle.input_adapter.ready

    recenter = bundle.input_adapter.request_anchor(
        _activation(bundle, epoch=2, at=11.0),
        requested_at=11.0,
    )
    assert recenter.request_generation == 0
    assert not bundle.input_adapter.ready
    second_action = _action(0.8, (0.9, 0.1))
    step = bundle.input_adapter.process_absolute_action(
        second_action,
        source_action_id="human:recenter",
        source_action_hash=second_action.semantic_content_hash,
        source_available_at=11.01,
        measured_at=21.0,
        measured_clock_domain_id="human.source",
        observed_at=11.02,
        observed_clock_domain_id="control.monotonic",
    )

    assert step.status == "anchor_captured_relative_action_ready"
    np.testing.assert_allclose(
        step.relative_action.value("arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )


def test_allocation_derivation_keeps_fixed_absolute_and_residual_roles() -> None:
    bundle = _build(enabled=True)
    bundle.input_adapter.request_anchor(
        _activation(bundle, epoch=1, at=10.0),
        requested_at=10.0,
    )
    anchor_action = _action(0.2, (0.4, 0.6))
    anchor = bundle.input_adapter.derive_allocation_actions(
        anchor_action,
        source_action_id="human:0",
        source_action_hash=anchor_action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        observed_at=10.02,
        observed_clock_domain_id="control.monotonic",
    )
    assert anchor.ready
    assert anchor.captured_anchor
    assert set(anchor.action_by_role) == {
        "human_absolute",
        "human_residual",
    }
    np.testing.assert_allclose(
        anchor.action_by_role["human_residual"].value("hand").values,
        (0.0, 0.0),
    )

    moved_action = _action(0.3, (0.5, 0.2))
    derived = bundle.input_adapter.derive_allocation_actions(
        moved_action,
        source_action_id="human:1",
        source_action_hash=moved_action.semantic_content_hash,
        source_available_at=10.1,
        measured_at=20.1,
        measured_clock_domain_id="human.source",
        observed_at=10.11,
        observed_clock_domain_id="control.monotonic",
    )

    assert derived.ready
    assert set(derived.action_by_role) == {
        "human_absolute",
        "human_residual",
    }
    assert derived.action_by_role["human_absolute"] is moved_action
    np.testing.assert_allclose(
        derived.action_by_role["human_residual"].value("arm").values[:3],
        (0.1, 0.0, 0.0),
    )
    assert derived.profile_step.source_action_id == "human:1"


def test_profile_contract_names_are_concrete_but_generic_contract_is_role_keyed() -> None:
    bundle = _build(enabled=True)
    contracts = bundle.relative_contracts

    assert contracts.baseline_role == "policy"
    assert contracts.residual_role == "human_residual"
    assert contracts.composition_spec.depends_on_roles_by_role == {
        "human_residual": ("policy",),
    }
    assert contracts.composition_spec.input_action_schema_hash_by_role == {
        "policy": _schema().semantic_content_hash,
        "human_residual": (
            contracts.residual_action_schema.semantic_content_hash
        ),
    }

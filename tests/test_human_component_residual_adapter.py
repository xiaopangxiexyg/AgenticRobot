from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition import (
    ResidualAnchorRequest,
    build_composition_activation_scope,
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
from teleop_runtime.action import HumanComponentResidualAdapter


def _schemas() -> tuple[ComponentActionSchema, ComponentActionSchema]:
    absolute = {
        "arm": ActionPayloadSchema(
            schema_id="arm-absolute",
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
            schema_id="hand-absolute",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    relative = {
        "arm": ActionPayloadSchema(
            schema_id="arm-relative",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="arm_base",
            controlled_frame="tool_tcp",
            semantic_parameters={
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            },
        ),
        "hand": ActionPayloadSchema(
            schema_id="hand-relative",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="relative_delta",
            semantic_parameters={"application": "elementwise_additive"},
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="canonical-human-residual",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    absolute[component_id].semantic_content_hash,
                    relative[component_id].semantic_content_hash,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="canonical-human-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=absolute,
        ),
        ComponentActionSchema(
            schema_id="canonical-human-relative",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=relative,
        ),
    )


def _adapter() -> HumanComponentResidualAdapter:
    absolute, relative = _schemas()
    return HumanComponentResidualAdapter(
        absolute_action_schema=absolute,
        relative_action_schema=relative,
        source_binding_id="canonical-human-source:1",
        source_binding_hash="canonical-human-source-hash:1",
        frame_graph=CoordinateFrameGraph(
            graph_id="canonical-human-source",
            frames=(
                CoordinateFrameSpec("left_arm/base"),
                CoordinateFrameSpec("left_arm/tool_tcp"),
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={
            "arm": "left_arm/base",
        },
        concrete_controlled_frame_by_component={
            "arm": "left_arm/tool_tcp",
        },
        pose_translation_gain_by_component={"arm": 2.0},
        joint_gain_by_component={"hand": (1.0, 0.5)},
    )


def _action(
    *,
    x: float,
    rotation_deg: float,
    hand: tuple[float, float],
) -> ComponentAction:
    absolute, _ = _schemas()
    quaternion = Rotation.from_euler(
        "z",
        rotation_deg,
        degrees=True,
    ).as_quat()
    return ComponentAction(
        action_schema=absolute,
        values_by_component={
            "arm": [x, 0.0, 0.0, *quaternion],
            "hand": hand,
        },
    )


def _request(
    adapter: HumanComponentResidualAdapter,
    *,
    generation: int = 0,
    requested_at: float = 10.0,
    previous: ResidualAnchorRequest | None = None,
) -> ResidualAnchorRequest:
    activation = build_composition_activation_scope(
        operation_spec_id="example-relative-profile",
        operation_spec_hash="example-relative-profile-hash",
        session_epoch=generation + 1,
        activated_at=requested_at,
        clock_domain_id="receiver.monotonic",
        input_barrier_at=requested_at,
        required_roles=("baseline", "residual"),
    )
    return ResidualAnchorRequest.for_activation(
        activation,
        request_generation=generation,
        source_role="residual",
        component_ids=adapter.component_ids,
        source_binding_id=adapter.source_binding_id,
        source_binding_hash=adapter.source_binding_hash,
        absolute_action_schema_hash=(
            adapter.absolute_action_schema.semantic_content_hash
        ),
        relative_action_schema_hash=(
            adapter.relative_action_schema.semantic_content_hash
        ),
        requested_at=requested_at,
        previous_request=previous,
    )


def test_atomic_pose_and_joint_anchor_produces_one_relative_action() -> None:
    adapter = _adapter()
    request = _request(adapter)
    anchor_action = _action(x=0.2, rotation_deg=10.0, hand=(0.4, 0.6))
    anchor = adapter.capture_anchor(
        anchor_action,
        request=request,
        source_action_id="human-action:anchor",
        source_action_hash=anchor_action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        captured_at=10.02,
    )
    current = _action(x=0.3, rotation_deg=40.0, hand=(0.5, 0.2))

    result = adapter.adapt(
        current,
        anchor=anchor,
        source_action_id="human-action:current",
        source_action_hash=current.semantic_content_hash,
        measured_at=20.1,
        measured_clock_domain_id="human.source",
        adapted_at=10.1,
        adapted_clock_domain_id="receiver.monotonic",
    )

    assert set(anchor.receipt.component_binding_by_id) == {"arm", "hand"}
    assert result.relative_action.component_coverage == ("arm", "hand")
    np.testing.assert_allclose(
        result.relative_action.value("arm").values[:3],
        (0.2, 0.0, 0.0),
        atol=1e-12,
    )
    rotation = Rotation.from_quat(
        result.relative_action.value("arm").values[3:7]
    )
    assert abs(rotation.magnitude() - np.deg2rad(30.0)) < 1e-12
    np.testing.assert_allclose(
        result.relative_action.value("hand").values,
        (0.1, -0.2),
        atol=1e-12,
    )


def test_anchor_sample_returns_identity_and_does_not_accumulate() -> None:
    adapter = _adapter()
    request = _request(adapter)
    action = _action(x=0.2, rotation_deg=10.0, hand=(0.4, 0.6))
    anchor = adapter.capture_anchor(
        action,
        request=request,
        source_action_id="human-action:anchor",
        source_action_hash=action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        captured_at=10.02,
    )

    first = adapter.adapt(
        action,
        anchor=anchor,
        source_action_id="human-action:same",
        source_action_hash=action.semantic_content_hash,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        adapted_at=10.1,
        adapted_clock_domain_id="receiver.monotonic",
    )
    second = adapter.adapt(
        action,
        anchor=anchor,
        source_action_id="human-action:same",
        source_action_hash=action.semantic_content_hash,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        adapted_at=10.2,
        adapted_clock_domain_id="receiver.monotonic",
    )

    np.testing.assert_allclose(
        first.relative_action.value("arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first.relative_action.value("hand").values,
        (0.0, 0.0),
        atol=1e-12,
    )
    assert (
        first.relative_action.semantic_content_hash
        == second.relative_action.semantic_content_hash
    )


def test_request_binding_or_source_action_hash_mismatch_is_rejected() -> None:
    adapter = _adapter()
    request = _request(adapter)
    action = _action(x=0.2, rotation_deg=10.0, hand=(0.4, 0.6))
    wrong_request = ResidualAnchorRequest(
        request_generation=0,
        source_role=request.source_role,
        component_ids=request.component_ids,
        composition_activation_id=request.composition_activation_id,
        composition_activation_hash=request.composition_activation_hash,
        source_binding_id=request.source_binding_id,
        source_binding_hash="another-source-binding",
        absolute_action_schema_hash=request.absolute_action_schema_hash,
        relative_action_schema_hash=request.relative_action_schema_hash,
        requested_at=request.requested_at,
        clock_domain_id=request.clock_domain_id,
        input_barrier_at=request.input_barrier_at,
    )

    with pytest.raises(ValueError, match="source_binding_hash"):
        adapter.capture_anchor(
            action,
            request=wrong_request,
            source_action_id="human-action:anchor",
            source_action_hash=action.semantic_content_hash,
            source_available_at=10.01,
            measured_at=20.0,
            measured_clock_domain_id="human.source",
            captured_at=10.02,
        )
    with pytest.raises(ValueError, match="Source action hash"):
        adapter.capture_anchor(
            action,
            request=request,
            source_action_id="human-action:anchor",
            source_action_hash="wrong-action-hash",
            source_available_at=10.01,
            measured_at=20.0,
            measured_clock_domain_id="human.source",
            captured_at=10.02,
        )


def test_recenter_generation_reanchors_without_changing_adapter() -> None:
    adapter = _adapter()
    first_request = _request(adapter)
    first_action = _action(x=0.2, rotation_deg=10.0, hand=(0.4, 0.6))
    first_anchor = adapter.capture_anchor(
        first_action,
        request=first_request,
        source_action_id="human-action:first-anchor",
        source_action_hash=first_action.semantic_content_hash,
        source_available_at=10.01,
        measured_at=20.0,
        measured_clock_domain_id="human.source",
        captured_at=10.02,
    )
    second_request = _request(
        adapter,
        generation=1,
        requested_at=11.0,
        previous=first_request,
    )
    second_action = _action(x=0.8, rotation_deg=70.0, hand=(0.9, 0.1))
    second_anchor = adapter.capture_anchor(
        second_action,
        request=second_request,
        source_action_id="human-action:second-anchor",
        source_action_hash=second_action.semantic_content_hash,
        source_available_at=11.01,
        measured_at=21.0,
        measured_clock_domain_id="human.source",
        captured_at=11.02,
    )

    assert first_anchor.receipt.request_generation == 0
    assert second_anchor.receipt.request_generation == 1
    result = adapter.adapt(
        second_action,
        anchor=second_anchor,
        source_action_id="human-action:second-same",
        source_action_hash=second_action.semantic_content_hash,
        measured_at=21.0,
        measured_clock_domain_id="human.source",
        adapted_at=11.1,
        adapted_clock_domain_id="receiver.monotonic",
    )
    np.testing.assert_allclose(
        result.relative_action.value("hand").values,
        (0.0, 0.0),
        atol=1e-12,
    )

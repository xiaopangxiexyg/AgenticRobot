from __future__ import annotations

import pytest

from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentActionSchema,
    ControlComponentSpecV2,
    build_relative_delta_action_schema,
)


def _absolute_schema() -> ComponentActionSchema:
    payloads = {
        "tcp": ActionPayloadSchema(
            schema_id="tcp-absolute",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="base",
            controlled_frame="tool",
        ),
        "gripper": ActionPayloadSchema(
            schema_id="gripper-absolute",
            schema_version="development",
            representation_family="joint_position",
            dtype="float32",
            shape=(1,),
            coordinate_names=("opening",),
            units=("1",),
            value_semantics="absolute_target",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="one-arm-gripper",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="tcp",
                controlled_entity_id="arm",
                command_channel_id="tcp",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payloads["tcp"].semantic_content_hash,
                ),
            ),
            ControlComponentSpecV2(
                component_id="gripper",
                controlled_entity_id="gripper",
                command_channel_id="gripper",
                selection_group_id="gripper",
                resource_claims=("resource:gripper",),
                accepted_payload_schema_hashes=(
                    payloads["gripper"].semantic_content_hash,
                ),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="one-arm-gripper-absolute",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component=payloads,
    )


def test_relative_schema_preserves_arbitrary_topology_and_numeric_layout() -> None:
    absolute = _absolute_schema()
    relative = build_relative_delta_action_schema(
        absolute,
        schema_id="one-arm-gripper-relative",
    )

    assert relative.component_ids == ("tcp", "gripper")
    for component_id in relative.component_ids:
        before = absolute.payload_schema_by_component[component_id]
        after = relative.payload_schema_by_component[component_id]
        assert after.value_semantics == "relative_delta"
        assert after.representation_family == before.representation_family
        assert after.shape == before.shape
        assert after.coordinate_names == before.coordinate_names
        assert after.units == before.units
        assert after.reference_frame == before.reference_frame
        assert after.controlled_frame == before.controlled_frame
    assert (
        relative.payload_schema_by_component["tcp"].semantic_parameters[
            "rotation_application"
        ]
        == "baseline_local_right_multiply"
    )
    assert (
        relative.payload_schema_by_component["gripper"].semantic_parameters[
            "application"
        ]
        == "elementwise_additive"
    )


def test_relative_schema_can_select_one_component_without_side_or_dof_rules() -> None:
    relative = build_relative_delta_action_schema(
        _absolute_schema(),
        component_ids=("gripper",),
        schema_id="gripper-relative",
    )

    assert relative.component_ids == ("gripper",)
    assert relative.payload_schema_by_component["gripper"].shape == (1,)


def test_non_pose_qpos_absolute_component_is_not_silently_reinterpreted() -> None:
    absolute = _absolute_schema()
    payload = ActionPayloadSchema(
        schema_id="velocity",
        schema_version="development",
        representation_family="joint_velocity",
        dtype="float32",
        shape=(1,),
        coordinate_names=("joint",),
        units=("rad/s",),
        value_semantics="rate_target",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="velocity",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="velocity",
                controlled_entity_id="arm",
                command_channel_id="velocity",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            ),
        ),
    )
    velocity_schema = ComponentActionSchema(
        schema_id="velocity",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"velocity": payload},
    )

    with pytest.raises(ValueError, match="not an absolute pose/qpos"):
        build_relative_delta_action_schema(
            velocity_schema,
            schema_id="velocity-relative",
        )
    assert absolute.semantic_content_hash != velocity_schema.semantic_content_hash

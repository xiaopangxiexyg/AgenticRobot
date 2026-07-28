from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
    EndEffectorTarget,
    HandActionSpec,
    RobotAction,
    materialize_robot_action_v1,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.proposal import robot_action_semantic_payload


def _robot_action_hash(action: RobotAction) -> str:
    return canonical_content_hash(
        robot_action_semantic_payload(action),
        domain="test.robot_action_v1.semantic",
    )


def _pose_schema(
    schema_id: str,
    *,
    reference_frame: str,
    controlled_frame: str = "tool_tcp",
) -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id=schema_id,
        schema_version="1",
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
    )


def _joint_schema(
    schema_id: str,
    *,
    dof: int,
    family: str = "joint_position",
) -> ActionPayloadSchema:
    semantics = {
        "joint_position": "absolute_target",
        "joint_velocity": "rate_target",
        "joint_effort": "effort_target",
    }[family]
    units = "rad" if family == "joint_position" else (
        "rad/s" if family == "joint_velocity" else "N*m"
    )
    return ActionPayloadSchema(
        schema_id=schema_id,
        schema_version="1",
        representation_family=family,
        dtype="float32",
        shape=(dof,),
        coordinate_names=tuple(f"joint_{index}" for index in range(dof)),
        units=tuple(units for _ in range(dof)),
        value_semantics=semantics,
    )


def _component(
    component_id: str,
    payload_schema: ActionPayloadSchema,
    *,
    entity: str,
    group: str,
    atomic_group: str | None = None,
    dependencies: tuple[str, ...] = (),
) -> ControlComponentSpecV2:
    return ControlComponentSpecV2(
        component_id=component_id,
        controlled_entity_id=entity,
        command_channel_id=component_id,
        selection_group_id=group,
        resource_claims=(f"resource:{component_id}",),
        accepted_payload_schema_hashes=(
            payload_schema.semantic_content_hash,
        ),
        atomic_group_id=atomic_group,
        dependencies=dependencies,
    )


def _action_schema(
    *items: tuple[ControlComponentSpecV2, ActionPayloadSchema],
    schema_id: str = "fixture-action",
) -> ComponentActionSchema:
    catalog = ActionComponentCatalogV2(
        catalog_id=f"{schema_id}:catalog",
        catalog_version="1",
        components=tuple(item[0] for item in items),
    )
    return ComponentActionSchema(
        schema_id=schema_id,
        schema_version="1",
        component_catalog=catalog,
        payload_schema_by_component={
            component.component_id: payload
            for component, payload in items
        },
    )


def test_component_action_supports_arbitrary_names_and_multiple_end_effectors() -> None:
    arm = _pose_schema("pose:arm_7d", reference_frame="arm_0/base")
    gripper = _joint_schema("qpos:gripper", dof=1)
    tool = _joint_schema("qpos:tool_fingers", dof=3)
    schema = _action_schema(
        (
            _component(
                "manipulator/arm_0/tcp",
                arm,
                entity="manipulator/arm_0",
                group="arm_0_motion",
            ),
            arm,
        ),
        (
            _component(
                "manipulator/arm_0/gripper",
                gripper,
                entity="manipulator/arm_0/gripper",
                group="gripper",
            ),
            gripper,
        ),
        (
            _component(
                "manipulator/arm_0/tool_fingers",
                tool,
                entity="manipulator/arm_0/tool_fingers",
                group="tool_fingers",
            ),
            tool,
        ),
    )
    action = ComponentAction(
        action_schema=schema,
        values_by_component={
            "manipulator/arm_0/tcp": [0, 0, 0, 0, 0, 0, 1],
            "manipulator/arm_0/gripper": [0.4],
            "manipulator/arm_0/tool_fingers": [0.1, 0.2, 0.3],
        },
    )

    assert action.component_coverage == (
        "manipulator/arm_0/tcp",
        "manipulator/arm_0/gripper",
        "manipulator/arm_0/tool_fingers",
    )
    assert action.value("manipulator/arm_0/gripper").values.shape == (1,)


def test_component_action_supports_mixed_six_and_seven_dof_joint_schemas() -> None:
    left = _joint_schema("qpos:arm_a", dof=6)
    right = _joint_schema("qpos:arm_b", dof=7)
    schema = _action_schema(
        (
            _component(
                "arm_a",
                left,
                entity="arm_a",
                group="arms",
                atomic_group="coupled_arms",
            ),
            left,
        ),
        (
            _component(
                "arm_b",
                right,
                entity="arm_b",
                group="arms",
                atomic_group="coupled_arms",
            ),
            right,
        ),
    )

    action = ComponentAction(
        action_schema=schema,
        values_by_component={
            "arm_a": np.zeros(6, dtype=np.float32),
            "arm_b": np.zeros(7, dtype=np.float32),
        },
    )

    assert action.value("arm_a").values.shape == (6,)
    assert action.value("arm_b").values.shape == (7,)
    partial = ComponentAction(
        action_schema=schema,
        values_by_component={"arm_a": np.zeros(6, dtype=np.float32)},
    )
    assert partial.component_coverage == ("arm_a",)


def test_twist_and_wrench_are_distinct_frame_bound_payload_schemas() -> None:
    twist = ActionPayloadSchema(
        schema_id="mobile-base-twist",
        schema_version="1",
        representation_family="spatial_twist",
        dtype="float32",
        shape=(6,),
        coordinate_names=("vx", "vy", "vz", "wx", "wy", "wz"),
        units=("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"),
        value_semantics="rate_target",
        controlled_frame="base_link",
        expressed_in_frame="world",
    )
    wrench = ActionPayloadSchema(
        schema_id="tool-wrench",
        schema_version="1",
        representation_family="spatial_wrench",
        dtype="float32",
        shape=(6,),
        coordinate_names=("fx", "fy", "fz", "tx", "ty", "tz"),
        units=("N", "N", "N", "N*m", "N*m", "N*m"),
        value_semantics="effort_target",
        controlled_frame="tool_tcp",
        expressed_in_frame="tool_tcp",
    )
    schema = _action_schema(
        (
            _component(
                "mobile_base",
                twist,
                entity="mobile_base",
                group="base",
            ),
            twist,
        ),
        (
            _component(
                "arm_0/tool_wrench",
                wrench,
                entity="arm_0/tool",
                group="arm_force",
            ),
            wrench,
        ),
    )

    action = ComponentAction(
        action_schema=schema,
        values_by_component={
            "mobile_base": np.zeros(6, dtype=np.float32),
            "arm_0/tool_wrench": [0, 0, 5, 0, 0, 0],
        },
    )

    assert action.value("mobile_base").payload_schema.representation_family == (
        "spatial_twist"
    )
    assert action.value(
        "arm_0/tool_wrench"
    ).payload_schema.representation_family == "spatial_wrench"
    assert (
        action.value("mobile_base").payload_schema.semantic_content_hash
        != action.value(
            "arm_0/tool_wrench"
        ).payload_schema.semantic_content_hash
    )


@pytest.mark.parametrize(
    ("family", "semantics", "unit"),
    (
        ("joint_velocity", "rate_target", "rad/s"),
        ("joint_effort", "effort_target", "N*m"),
    ),
)
def test_joint_rate_and_effort_payloads_are_schema_visible(
    family: str,
    semantics: str,
    unit: str,
) -> None:
    schema = ActionPayloadSchema(
        schema_id=f"arm:{family}",
        schema_version="1",
        representation_family=family,
        dtype="float32",
        shape=(7,),
        coordinate_names=tuple(f"joint_{index}" for index in range(7)),
        units=tuple(unit for _ in range(7)),
        value_semantics=semantics,
    )

    assert schema.representation_family == family
    assert schema.value_semantics == semantics
    assert schema.shape == (7,)


def test_namespaced_representation_extension_does_not_change_carrier() -> None:
    payload = ActionPayloadSchema(
        schema_id="example-vendor-command",
        schema_version="1",
        representation_family="example.vendor_command",
        dtype="float32",
        shape=(2,),
        coordinate_names=("axis_a", "axis_b"),
        units=("1", "1"),
        value_semantics="example.vendor_target",
    )
    schema = _action_schema(
        (
            _component(
                "custom_device/channel",
                payload,
                entity="custom_device",
                group="custom_device",
            ),
            payload,
        )
    )

    action = ComponentAction(
        action_schema=schema,
        values_by_component={"custom_device/channel": [0.1, 0.2]},
    )

    assert action.value(
        "custom_device/channel"
    ).payload_schema.representation_family == "example.vendor_command"


def test_component_action_rejects_nonfinite_and_unnormalized_pose_values() -> None:
    pose = _pose_schema("pose", reference_frame="base")
    schema = _action_schema(
        (
            _component(
                "arm/tcp",
                pose,
                entity="arm",
                group="arm",
            ),
            pose,
        )
    )
    with pytest.raises(ValueError, match="fully finite"):
        ComponentAction(
            action_schema=schema,
            values_by_component={
                "arm/tcp": [0, 0, 0, 0, 0, 0, np.nan]
            },
        )
    with pytest.raises(ValueError, match="normalized"):
        ComponentAction(
            action_schema=schema,
            values_by_component={"arm/tcp": [0, 0, 0, 0, 0, 0, 2]},
        )


def test_component_action_hash_uses_schema_order_not_mapping_order() -> None:
    a = _joint_schema("a", dof=1)
    b = _joint_schema("b", dof=1)
    schema = _action_schema(
        (_component("a", a, entity="a", group="a"), a),
        (_component("b", b, entity="b", group="b"), b),
    )
    first = ComponentAction(
        action_schema=schema,
        values_by_component={"a": [1], "b": [2]},
    )
    second = ComponentAction(
        action_schema=schema,
        values_by_component={"b": [2], "a": [1]},
    )

    assert first.component_coverage == ("a", "b")
    assert first.semantic_content_hash == second.semantic_content_hash


def test_component_source_value_may_arrive_before_its_dependency() -> None:
    arm = _pose_schema("arm-pose", reference_frame="arm/base")
    gripper = _joint_schema("gripper-qpos", dof=1)
    schema = _action_schema(
        (_component("arm", arm, entity="arm", group="arm"), arm),
        (
            _component(
                "gripper",
                gripper,
                entity="gripper",
                group="gripper",
                dependencies=("arm",),
            ),
            gripper,
        ),
    )

    action = ComponentAction(
        action_schema=schema,
        values_by_component={"gripper": [0.5]},
    )

    assert action.component_coverage == ("gripper",)


def _pose(side: str) -> np.ndarray:
    x = -0.1 if side == "left" else 0.1
    return np.asarray([x, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32)


def test_robot_action_v1_pose_q6_bridge_round_trip_is_semantically_exact() -> None:
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    original = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        timestamp=12.5,
        arm_pose_by_side={
            side: _pose(side) for side in ("left", "right")
        },
        hand_qpos_by_side={
            "left": np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32),
            "right": np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32),
        },
        active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        source="fixture",
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata()
                for side, spec in hand_specs.items()
            }
        },
    )

    component_action = project_robot_action_v1_to_component_action(original)
    restored = materialize_robot_action_v1(
        component_action,
        timestamp=original.timestamp,
        source=original.source,
    )

    assert component_action.value("left_hand").values.shape == (1,)
    assert component_action.value("right_hand").values.shape == (1,)
    assert _robot_action_hash(restored) == _robot_action_hash(original)


def test_robot_action_v1_bridge_strips_and_restores_q16_padding() -> None:
    spec = HandActionSpec.q16(
        "left",
        active_dof=3,
        joint_names=("finger_a", "finger_b", "finger_c"),
    )
    full = np.full(16, np.nan, dtype=np.float32)
    full[:3] = (0.1, 0.2, 0.3)
    original = RobotAction(
        action_space=BIMANUAL_EE_POSE_HAND16_V1,
        hand_qpos_by_side={"left": full},
        active_effectors=("left_hand",),
        metadata={"hand_action_specs_by_side": {"left": spec.to_metadata()}},
    )

    component_action = project_robot_action_v1_to_component_action(original)
    restored = materialize_robot_action_v1(component_action)

    assert component_action.value("left_hand").values.shape == (3,)
    np.testing.assert_allclose(
        restored.hand_qpos_by_side["left"][:3],
        full[:3],
    )
    assert np.all(np.isnan(restored.hand_qpos_by_side["left"][3:]))


def test_robot_action_v1_native_discrete_target_round_trip() -> None:
    original = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        end_effector_targets_by_side={
            "left": EndEffectorTarget(
                value_type="discrete",
                values=np.asarray([2], dtype=np.int64),
                adapter="fixture-mode-adapter",
                names=("mode",),
                metadata={"token_contract": "fixture-v1"},
            )
        },
        active_effectors=("left_hand",),
    )

    component_action = project_robot_action_v1_to_component_action(original)
    restored = materialize_robot_action_v1(component_action)

    assert (
        component_action.value(
            "left_hand"
        ).payload_schema.representation_family
        == "discrete_mode"
    )
    assert _robot_action_hash(restored) == _robot_action_hash(original)


def test_v2_action_hash_excludes_v1_timestamp_and_source() -> None:
    first = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        timestamp=1.0,
        arm_pose_by_side={"left": _pose("left")},
        source="policy",
    )
    second = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        timestamp=99.0,
        arm_pose_by_side={"left": _pose("left")},
        source="human",
    )

    first_v2 = project_robot_action_v1_to_component_action(first)
    second_v2 = project_robot_action_v1_to_component_action(second)

    assert first_v2.action_schema.semantic_content_hash == (
        second_v2.action_schema.semantic_content_hash
    )
    assert first_v2.semantic_content_hash == second_v2.semantic_content_hash


def test_v1_bridge_projects_partial_values_into_complete_runtime_schema() -> None:
    complete = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "left": _pose("left"),
            "right": _pose("right"),
        },
        active_effectors=("left_arm", "right_arm"),
    )
    runtime_schema = project_robot_action_v1_to_component_action(
        complete
    ).action_schema
    partial = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"left": _pose("left")},
        active_effectors=("left_arm",),
    )

    projected = project_robot_action_v1_to_component_action(
        partial,
        action_schema=runtime_schema,
    )

    assert projected.action_schema is runtime_schema
    assert projected.component_coverage == ("left_arm",)
    assert runtime_schema.component_ids == ("left_arm", "right_arm")
    restored = materialize_robot_action_v1(projected)
    assert restored.active_effectors == ("left_arm",)
    assert set(restored.arm_pose_by_side) == {"left"}


def test_v1_bridge_rejects_partial_payload_schema_drift() -> None:
    complete = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"left": _pose("left")},
        active_effectors=("left_arm",),
    )
    runtime_schema = project_robot_action_v1_to_component_action(
        complete
    ).action_schema
    drifted = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"left": _pose("left")},
        active_effectors=("left_arm",),
        frame_by_effector={"left_arm": "another_world"},
    )

    with pytest.raises(ValueError, match="runtime payload schema"):
        project_robot_action_v1_to_component_action(
            drifted,
            action_schema=runtime_schema,
        )

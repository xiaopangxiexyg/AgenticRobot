from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ActionAdaptationResult,
    ComponentAction,
    ComponentActionSchema,
    ComponentTargetCapabilitySpec,
    ControlComponentSpecV2,
    TargetComponentRouteSpec,
    TargetResourceCombinationSpec,
    RobotActionV1RepresentationAdapter,
    RobotAction,
    assess_component_target_capability,
    build_action_adaptation_request,
    build_robot_action_v1_target_capability,
    require_robot_action_v1,
    project_robot_action_v1_to_component_action,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
)


def _pose_schema() -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id="tcp-pose-v1",
        schema_version="1",
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame="robot_base",
        controlled_frame="tool_tcp",
    )


def _wrench_schema() -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id="tcp-wrench-v1",
        schema_version="1",
        representation_family="spatial_wrench",
        dtype="float32",
        shape=(6,),
        coordinate_names=("fx", "fy", "fz", "tx", "ty", "tz"),
        units=("N", "N", "N", "N*m", "N*m", "N*m"),
        value_semantics="effort_target",
        expressed_in_frame="tool_tcp",
        controlled_frame="tool_tcp",
    )


def _shared_tcp_schema() -> ComponentActionSchema:
    pose = _pose_schema()
    wrench = _wrench_schema()
    return ComponentActionSchema(
        schema_id="pose-wrench-action-v1",
        schema_version="1",
        component_catalog=ActionComponentCatalogV2(
            catalog_id="pose-wrench-catalog-v1",
            catalog_version="1",
            components=(
                ControlComponentSpecV2(
                    component_id="arm/tcp_pose",
                    controlled_entity_id="arm/tool_tcp",
                    command_channel_id="arm/tcp_pose",
                    selection_group_id="arm_motion",
                    resource_claims=("arm/tool_tcp",),
                    accepted_payload_schema_hashes=(
                        pose.semantic_content_hash,
                    ),
                ),
                ControlComponentSpecV2(
                    component_id="arm/tcp_wrench",
                    controlled_entity_id="arm/tool_tcp",
                    command_channel_id="arm/tcp_wrench",
                    selection_group_id="arm_force",
                    resource_claims=("arm/tool_tcp",),
                    accepted_payload_schema_hashes=(
                        wrench.semantic_content_hash,
                    ),
                ),
            ),
        ),
        payload_schema_by_component={
            "arm/tcp_pose": pose,
            "arm/tcp_wrench": wrench,
        },
    )


def _capability(
    schema: ComponentActionSchema,
    *,
    allowed_sets: tuple[tuple[str, ...], ...],
) -> ComponentTargetCapabilitySpec:
    return ComponentTargetCapabilitySpec(
        capability_id="hybrid-target-v1",
        capability_version="1",
        action_schema=schema,
        component_routes=tuple(
            TargetComponentRouteSpec(
                component_id=component_id,
                input_payload_schema_hash=schema.payload_schema_by_component[
                    component_id
                ].semantic_content_hash,
                handler_contract_id="hybrid-target-handler",
                handler_contract_hash="hybrid-target-handler-hash",
                output_target_schema_hash="hybrid-target-output-hash",
            )
            for component_id in schema.component_ids
        ),
        resource_combinations=(
            TargetResourceCombinationSpec(
                resource_id="arm/tool_tcp",
                participant_component_ids=(
                    "arm/tcp_pose",
                    "arm/tcp_wrench",
                ),
                allowed_active_component_sets=allowed_sets,
                handler_contract_id="hybrid-target-handler",
                handler_contract_hash="hybrid-target-handler-hash",
            ),
        ),
    )


def _candidate(
    schema: ComponentActionSchema,
    *component_ids: str,
) -> ComponentAction:
    values = {
        "arm/tcp_pose": np.asarray(
            [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "arm/tcp_wrench": np.asarray(
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    }
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            component_id: values[component_id]
            for component_id in component_ids
        },
    )


def test_resource_contract_uses_exact_sets_without_silent_priority() -> None:
    schema = _shared_tcp_schema()
    exclusive = _capability(
        schema,
        allowed_sets=(("arm/tcp_pose",), ("arm/tcp_wrench",)),
    )

    pose_only = assess_component_target_capability(
        exclusive,
        _candidate(schema, "arm/tcp_pose"),
    )
    pair = assess_component_target_capability(
        exclusive,
        _candidate(schema, "arm/tcp_pose", "arm/tcp_wrench"),
    )

    assert pose_only.accepted is True
    assert pose_only.rejection_reasons == ()
    assert pair.accepted is False
    assert pair.rejection_reasons == (
        "resource_combination_not_accepted:"
        "arm/tool_tcp:arm/tcp_pose,arm/tcp_wrench",
    )
    assert pair.active_components_by_resource["arm/tool_tcp"] == (
        "arm/tcp_pose",
        "arm/tcp_wrench",
    )


def test_explicit_hybrid_set_accepts_pose_and_wrench_together() -> None:
    schema = _shared_tcp_schema()
    hybrid = _capability(
        schema,
        allowed_sets=(
            ("arm/tcp_pose",),
            ("arm/tcp_wrench",),
            ("arm/tcp_wrench", "arm/tcp_pose"),
        ),
    )

    assessment = assess_component_target_capability(
        hybrid,
        _candidate(schema, "arm/tcp_pose", "arm/tcp_wrench"),
    )

    assert assessment.accepted is True
    assert hybrid.resource_combinations[0].allowed_active_component_sets[-1] == (
        "arm/tcp_pose",
        "arm/tcp_wrench",
    )


def test_capability_requires_exact_routes_and_resource_claims() -> None:
    schema = _shared_tcp_schema()
    capability = _capability(
        schema,
        allowed_sets=(("arm/tcp_pose",), ("arm/tcp_wrench",)),
    )

    with pytest.raises(ValueError, match="every action-schema component"):
        replace(
            capability,
            component_routes=capability.component_routes[:1],
        )
    with pytest.raises(ValueError, match="every claimed resource"):
        replace(capability, resource_combinations=())
    with pytest.raises(ValueError, match="binds another payload schema"):
        replace(
            capability,
            component_routes=(
                replace(
                    capability.component_routes[0],
                    input_payload_schema_hash="another-schema",
                ),
                capability.component_routes[1],
            ),
        )


def test_schema_mismatch_is_rejected_without_becoming_a_hold_command() -> None:
    schema = _shared_tcp_schema()
    capability = _capability(
        schema,
        allowed_sets=(("arm/tcp_pose",), ("arm/tcp_wrench",)),
    )
    changed_schema = replace(schema, schema_id="another-activation")

    assessment = assess_component_target_capability(
        capability,
        _candidate(changed_schema, "arm/tcp_pose"),
    )

    assert assessment.accepted is False
    assert "action_schema_mismatch" in assessment.rejection_reasons
    assert not hasattr(assessment, "hold")
    assert not hasattr(assessment, "dispatchable")


def test_robot_action_v1_builder_rejects_shared_resource_semantics() -> None:
    schema = _shared_tcp_schema()

    with pytest.raises(ValueError, match="cannot bind shared resources"):
        build_robot_action_v1_target_capability(
            capability_id="legacy-v1-target",
            action_schema=schema,
            handler_contract_id="robot-action-v1-adapter",
            handler_contract_hash="robot-action-v1-adapter-hash",
            output_target_schema_hash="robot-action-v1-output-hash",
        )


def test_robot_action_v1_builder_preserves_existing_single_route_profiles() -> None:
    pose = _pose_schema()
    schema = ComponentActionSchema(
        schema_id="legacy-pose-action",
        schema_version="1",
        component_catalog=ActionComponentCatalogV2(
            catalog_id="legacy-pose-catalog",
            catalog_version="1",
            components=(
                ControlComponentSpecV2(
                    component_id="right_arm",
                    controlled_entity_id="right_arm",
                    command_channel_id="right_arm",
                    selection_group_id="arms",
                    resource_claims=("right_arm",),
                    accepted_payload_schema_hashes=(
                        pose.semantic_content_hash,
                    ),
                ),
            ),
        ),
        payload_schema_by_component={"right_arm": pose},
    )
    capability = build_robot_action_v1_target_capability(
        capability_id="legacy-v1-target",
        action_schema=schema,
        handler_contract_id="robot-action-v1-adapter",
        handler_contract_hash="robot-action-v1-adapter-hash",
        output_target_schema_hash="robot-action-v1-output-hash",
    )

    assessment = assess_component_target_capability(
        capability,
        ComponentAction(
            action_schema=schema,
            values_by_component={
                "right_arm": [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
            },
        ),
    )

    assert assessment.accepted is True
    assert capability.component_routes[0].component_id == "right_arm"
    assert capability.resource_combinations[
        0
    ].allowed_active_component_sets == (("right_arm",),)


def test_typed_legacy_adaptation_preserves_candidate_lineage() -> None:
    action = project_robot_action_v1_to_component_action(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            arm_pose_by_side={
                "right": np.asarray(
                    [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                ),
            },
            active_effectors=("right_arm",),
            frame_by_effector={"right_arm": "robot_base"},
            controlled_frame_by_effector={"right_arm": "tool_tcp"},
            source="fixture",
        )
    )
    schema = action.action_schema
    capability = build_robot_action_v1_target_capability(
        capability_id="legacy-v1-target",
        action_schema=schema,
        handler_contract_id=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        handler_contract_hash=ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        output_target_schema_hash="robot-action-v1-output-hash",
    )
    request = build_action_adaptation_request(
        authoritative_decision_id="decision-1",
        authoritative_decision_hash="decision-hash-1",
        arbitration_result_id="result-1",
        candidate_action=action,
        component_target_capability=capability,
        target_effective_at=12.5,
        supported_until=13.0,
        authority_sequence=3,
        execution_gate_sequence=4,
        execution_fence_sequence=5,
        requested_at=12.0,
        clock_domain_id="control-monotonic",
    )
    adapter = RobotActionV1RepresentationAdapter(
        receiver_id="legacy-adapter",
        component_target_capability=capability,
        clock=lambda: 12.1,
    )

    result = adapter.adapt(request)

    assert isinstance(result, ActionAdaptationResult)
    assert result.receipt.accepted is True
    assert result.receipt.request_id == request.request_id
    assert result.adapted_candidate is not None
    adapted = require_robot_action_v1(result.adapted_candidate)
    assert adapted.timestamp == 12.5
    assert adapted.source == "action_arbiter"
    assert adapted.active_effectors == ("right_arm",)


def test_typed_adaptation_rejection_has_no_candidate_or_execution_verdict() -> None:
    schema = _shared_tcp_schema()
    capability = _capability(
        schema,
        allowed_sets=(("arm/tcp_pose",), ("arm/tcp_wrench",)),
    )
    # Rebind the test capability to the concrete legacy adapter contract while
    # retaining the deliberately exclusive shared-resource rule.
    capability = replace(
        capability,
        component_routes=tuple(
            replace(
                route,
                handler_contract_id=(
                    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID
                ),
                handler_contract_hash=(
                    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH
                ),
                output_target_schema_hash="robot-action-v1-output-hash",
            )
            for route in capability.component_routes
        ),
    )
    request = build_action_adaptation_request(
        authoritative_decision_id="decision-2",
        authoritative_decision_hash="decision-hash-2",
        arbitration_result_id="result-2",
        candidate_action=_candidate(
            schema,
            "arm/tcp_pose",
            "arm/tcp_wrench",
        ),
        component_target_capability=capability,
        target_effective_at=13.0,
        supported_until=14.0,
        authority_sequence=3,
        execution_gate_sequence=4,
        execution_fence_sequence=5,
        requested_at=12.0,
        clock_domain_id="control-monotonic",
    )
    adapter = RobotActionV1RepresentationAdapter(
        receiver_id="legacy-adapter",
        component_target_capability=capability,
        clock=lambda: 12.1,
    )

    result = adapter.adapt(request)

    assert result.receipt.accepted is False
    assert result.receipt.rejection_code == "target_capability_rejected"
    assert result.adapted_candidate is None
    assert not hasattr(result.receipt, "hold")
    assert not hasattr(result.receipt, "dispatchable")


def test_target_can_inject_component_to_robot_action_materializer() -> None:
    schema = _shared_tcp_schema()
    capability = _capability(
        schema,
        allowed_sets=(("arm/tcp_pose",), ("arm/tcp_wrench",)),
    )
    candidate = _candidate(schema, "arm/tcp_pose")
    request = build_action_adaptation_request(
        authoritative_decision_id="decision-custom-materializer",
        authoritative_decision_hash="decision-custom-materializer-hash",
        arbitration_result_id="result-custom-materializer",
        candidate_action=candidate,
        component_target_capability=capability,
        target_effective_at=14.0,
        supported_until=15.0,
        authority_sequence=3,
        execution_gate_sequence=4,
        execution_fence_sequence=5,
        requested_at=13.0,
        clock_domain_id="control-monotonic",
    )

    def materialize(action, *, timestamp, source):
        return RobotAction(
            action_space="bimanual_ee_pose_v1",
            timestamp=timestamp,
            arm_pose_by_side={
                    "right": np.asarray(
                        action.values_by_component["arm/tcp_pose"].values,
                    dtype=np.float32,
                )
            },
            active_effectors=("right_arm",),
            frame_by_effector={"right_arm": "robot_base"},
            controlled_frame_by_effector={"right_arm": "tool_tcp"},
            source=source,
        )

    adapter = RobotActionV1RepresentationAdapter(
        receiver_id="custom-component-materializer",
        component_target_capability=capability,
        adapter_contract_id="hybrid-target-handler",
        adapter_contract_hash="hybrid-target-handler-hash",
        materializer=materialize,
        expected_output_effectors=("right_arm",),
        clock=lambda: 13.1,
    )
    result = adapter.adapt(request)

    assert result.receipt.accepted is True
    assert result.adapted_candidate is not None
    adapted = require_robot_action_v1(result.adapted_candidate)
    assert adapted.timestamp == 14.0
    assert adapted.active_effectors == ("right_arm",)
    assert adapted.arm_pose_by_side["right"] == pytest.approx(
        candidate.values_by_component["arm/tcp_pose"].values
    )

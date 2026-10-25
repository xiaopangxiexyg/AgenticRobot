from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from agent_gateway.profiles.real.motion_primitive_profiles import (
    RealAgentPrimitiveConvergenceProfile,
    build_real_agent_primitive_contract_bundle,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.skills.motion.definition import (
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from teleop_runtime.real_teleop.agent_motion_port import RealAgentMotionPortConfig

COMPONENT_ID = "right_arm"


def _inputs():
    execution_frame = CoordinateFrameSpec(frame_id="dual_arm_solver_world")
    station_frame = CoordinateFrameSpec(
        frame_id="canonical_station",
        semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    graph = CoordinateFrameGraph(
        graph_id="real-agent-motion-test-graph",
        frames=(execution_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="solver-world-to-station",
                source_frame_id=execution_frame.frame_id,
                target_frame_id=station_frame.frame_id,
                matrix_target_source=tuple(tuple(row) for row in np.eye(4)),
            ),
        ),
        graph_version="1",
    )
    attachment = RealTcpPoseObserveAttachmentSpec(
        runtime_session_id="real-session:motion-test",
        control_domain_id="real-control-domain:motion-test",
        embodiment_instance_id="dual-rm65:test",
        target_binding_id="real-target:dual-rm65:test",
        component_ids=(COMPONENT_ID,),
        station_manifest_hash="station-manifest:real:test",
        execution_manifest_hash="execution-manifest:real:test",
        backend_inventory_hash="backend-inventory:real:test",
        source_binding_id="real-owner:measured-qpos:test",
        source_binding_hash="source-binding-hash:real:test",
        observation_binding_id="real-owner:tool-tcp-fk:test",
        observation_binding_hash="observation-binding-hash:real:test",
        observation_plan_hash="observation-plan-hash:real:test",
        observation_capability_hash="observation-capability-hash:real:test",
        pose_schema_hash="pose-schema-hash:xyz-quat:test",
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=graph.semantic_content_hash,
        tool_config_hash="tool-config-hash:real:test",
        calibration_hash="calibration-hash:real:test",
        clock_domain_id="real-owner-wall-clock",
        producer_epoch=1,
    )
    port_config = RealAgentMotionPortConfig(
        active_sides=("right",),
        component_id_by_side={"right": COMPONENT_ID},
        execution_frame_id=execution_frame.frame_id,
        execution_frame_spec_hash=execution_frame.semantic_content_hash,
    )
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-agent-motion-envelope:test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-agent-motion-envelope-hash:test",
        constraint_set_hash="real-agent-motion-constraints-hash:test",
        allowed_route_ids=real_agent_motion_route_ids_v1(),
    )
    return attachment, port_config, graph, envelope


def _bundle():
    attachment, port_config, graph, envelope = _inputs()
    return build_real_agent_primitive_contract_bundle(
        attachment=attachment,
        port_config=port_config,
        frame_graph=graph,
        primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=(envelope,),
    )


@pytest.mark.parametrize("component", ["left_arm", "right_arm"])
def test_spatial_goal_is_optional_and_binds_the_current_component(component):
    from agent_gateway.profiles.spatial_goal import calculate_spatial_goal

    attachment, config, graph, envelope = _inputs()
    side = component.removesuffix("_arm")
    attachment = replace(attachment, component_ids=(component,))
    config = replace(config, active_sides=(side,), component_id_by_side={side: component})
    kwargs = {"attachment": attachment, "port_config": config, "frame_graph": graph,
        "primitive_reference_frame_id": "canonical_station",
        "safety_envelope_refs": (envelope,)}
    baseline = build_real_agent_primitive_contract_bundle(**kwargs)
    assert baseline.spatial_goal_registration is None
    bundle = build_real_agent_primitive_contract_bundle(**kwargs, enable_spatial_goals=True)
    registration = bundle.spatial_goal_registration
    assert registration.entry.route.control_effect_class == "read_only"
    assert registration.entry.route.privilege_tier == "t0_read_only"
    assert registration.entry.route.may_move_robot is False
    assert registration.entry.route.requires_operator_confirmation is False
    assert registration.entry.evaluation_plan_hash is None
    assert len(bundle.route_manifest.skill_routes) == len(baseline.route_manifest.skill_routes) + 1
    for entry in baseline.route_manifest.skill_routes:
        assert entry == bundle.route_manifest.route(entry.route.skill_id, entry.route.skill_version)
    parameters = {"component_id": component, "reference_frame_id": "canonical_station",
        "current_tcp_pose_xyz_quat_xyzw": [0., 0., .3, 0., 0., 0., 1.],
        "anchor_xyz_m": [.2, -.1, .3], "offset_xyz_m": [0., 0., .03],
        "offset_frame": "reference", "orientation_mode": "align_axis",
        "tool_axis": "z", "axis_reference_xyz": [0., 0., -1.]}
    validator = registration.payload_validator
    result = calculate_spatial_goal(parameters)
    validator.validate_parameters(parameters)
    validator.validate_execution_result(result)
    validator.validate_execution_result_for_parameters(parameters, result)
    assert result["rotation_distance_rad"] == pytest.approx(np.pi)
    for changes in ({"reference_frame_id": "camera"}, {"component_id": "head"},
                    {"component_id": "right_arm" if component == "left_arm" else "left_arm"},
                    {"anchor_xyz_m": [True, 0, 0]}, {"unexpected": 1}):
        with pytest.raises(ValueError):
            validator.validate_parameters({**parameters, **changes})
    for changes in ({"path_checked": True}, {"motion_executed": True},
                    {"rotation_distance_rad": 0.},
                    {"pose_targets_by_component": {component: [float("nan"), 0., .3, 0., 0., 0., 1.]}},
                    {"pose_targets_by_component": {component: [0., 0., .3, 0., 0., 0., 2.]}},
                    {"pose_targets_by_component": {component: [0., 0., .4, 0., 0., 0., 1.]}}):
        with pytest.raises(ValueError):
            validator.validate_execution_result_for_parameters(parameters, {**result, **changes})


@pytest.mark.parametrize("primary", ["left", "right"])
def test_dual_pose_routes_allow_either_one_arm_but_not_both(primary):
    attachment, config, graph, envelope = _inputs()
    attachment = replace(attachment, component_ids=("left_arm", "right_arm"))
    config = replace(config, active_sides=("left", "right"),
        component_id_by_side={"left": "left_arm", "right": "right_arm"})
    envelope = replace(envelope, allowed_route_ids=(*envelope.allowed_route_ids,
        "real.rm65.agent_skill_tcp_pose_target_safe"))
    kwargs = {"attachment": attachment, "port_config": config, "frame_graph": graph,
        "primitive_reference_frame_id": "canonical_station", "safety_envelope_refs": (envelope,)}
    with pytest.raises(ValueError, match="explicit primary"):
        build_real_agent_primitive_contract_bundle(**kwargs)
    bundle = build_real_agent_primitive_contract_bundle(**kwargs, primary_component_id=primary+"_arm")
    profile = bundle.pose_target_profile
    assert profile.component_ids == ("left_arm", "right_arm")
    assert bundle.cardinal_profile.side == primary
    for skill in ("motion.move_tcp_to_pose", "motion.move_tcp_to_pose_safe"):
        route = bundle.route_manifest.route(skill, "1").route
        schema = bundle.schema_catalog.entry(skill, "1").parameter_schema
        assert schema["properties"]["pose_targets_by_component"]["maxProperties"] == 1
        payload = {"reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp", "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash, "calibration_hash": profile.calibration_hash}
        for arm in ("left", "right"):
            assert profile.side_for_component(arm+"_arm") == arm
            payload["pose_targets_by_component"] = {arm+"_arm": [0, 0, .3, 0, 0, 0, 1]}
            bundle.payload_validators.validate_parameters(route=route, payload=payload)
        payload["pose_targets_by_component"] = {arm+"_arm": [0, 0, .3, 0, 0, 0, 1]
                                              for arm in ("left", "right")}
        with pytest.raises(ValueError, match="exactly the bound component"):
            bundle.payload_validators.validate_parameters(route=route, payload=payload)


def test_real_bundle_exposes_read_diagnostic_and_fixed_motion_primitives() -> None:
    bundle = _bundle()

    assert bundle.attachment.reference_frame_id == "canonical_station"
    assert (
        bundle.attachment.reference_frame_spec_hash
        == bundle.cardinal_profile.reference_frame_spec_hash
    )
    assert (
        bundle.attachment.execution_reference_frame_id
        == "dual_arm_solver_world"
    )
    assert (
        bundle.attachment.execution_reference_frame_spec_hash
        == bundle.cardinal_profile.execution_reference_frame_spec_hash
    )
    assert bundle.tcp_pose_observe.attachment is bundle.attachment
    assert len(bundle.route_manifest.skill_routes) == 12
    assert bundle.route_manifest.route("motion.move_tcp_to_pose", "1")
    assert bundle.pose_target_profile.component_id == COMPONENT_ID
    assert len(bundle.cardinal_profile.primitives) == 6
    assert {
        entry.route.skill_id for entry in bundle.route_manifest.skill_routes
    } >= set(FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1)
    assert {item.skill_id for item in bundle.orientation_profiles} == set(
        FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    )
    read = bundle.route_manifest.route("state.read_tcp_pose", "1")
    assert read.route.privilege_tier == "t0_read_only"
    assert read.route.control_effect_class == "read_only"
    assert read.route.supported_modes == ("real_active",)
    assert read.route.requires_operator_confirmation is False
    assert read.route.may_move_robot is False
    for entry in bundle.route_manifest.skill_routes:
        route = entry.route
        if route.control_effect_class == "read_only":
            continue
        assert route.privilege_tier == "t2_real_motion"
        assert route.supported_modes == ("real_active",)
        assert route.requires_operator_confirmation is True
        assert route.requires_deadman is False
        assert route.cancellation_policy == "hold_on_cancel_disconnect_or_timeout"
        assert entry.deployment_binding.safety_envelope_refs


def test_legacy_execution_frame_attachment_gets_a_public_station_pose_schema() -> None:
    attachment, port_config, graph, envelope = _inputs()
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=attachment,
        port_config=port_config,
        frame_graph=graph,
        primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=(envelope,),
    )

    assert bundle.attachment.pose_schema_hash != attachment.pose_schema_hash
    assert bundle.attachment.reference_frame_id == "canonical_station"
    assert (
        bundle.attachment.execution_reference_frame_id
        == attachment.reference_frame_id
    )


def test_real_cardinal_payload_is_exact_and_selects_all_six_primitives() -> None:
    bundle = _bundle()
    profile = bundle.cardinal_profile
    route = bundle.route_manifest.route("motion.step_tcp_cardinal", "1").route
    for primitive in profile.primitives:
        payload = {
            "component_id": profile.component_id,
            "direction": primitive.direction.value,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_catalog_hash": profile.primitive_catalog_hash,
        }
        bundle.payload_validators.validate_parameters(route=route, payload=payload)

    bad = dict(payload)
    bad["direction"] = "diagonal"
    with pytest.raises(ValueError):
        bundle.payload_validators.validate_parameters(route=route, payload=bad)


def test_real_absolute_pose_payload_binds_tool_tcp_frame_and_bounded_speed() -> None:
    bundle = _bundle()
    profile = bundle.pose_target_profile
    route = bundle.route_manifest.route("motion.move_tcp_to_pose", "1").route
    schema = bundle.schema_catalog.entry("motion.move_tcp_to_pose", "1")
    speed_schema = schema.parameter_schema["properties"]["speed_scale"]
    assert speed_schema["minimum"] == profile.min_speed_scale
    assert speed_schema["maximum"] == profile.max_speed_scale
    assert route.max_operation_duration_ms == 30_000

    payload = {
        "pose_targets_by_component": {
            profile.component_id: [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0],
        },
        "reference_frame_id": profile.reference_frame_id,
        "reference_frame_spec_hash": profile.reference_frame_spec_hash,
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": profile.frame_graph_hash,
        "tool_config_hash": profile.tool_config_hash,
        "calibration_hash": profile.calibration_hash,
        "speed_scale": 0.25,
    }
    bundle.payload_validators.validate_parameters(route=route, payload=payload)
    without_speed = dict(payload)
    without_speed.pop("speed_scale")
    bundle.payload_validators.validate_parameters(
        route=route,
        payload=without_speed,
    )

    too_slow = dict(payload, speed_scale=profile.min_speed_scale / 2.0)
    with pytest.raises(ValueError, match="outside the exact profile bounds"):
        bundle.payload_validators.validate_parameters(route=route, payload=too_slow)
    wrong_component = dict(payload)
    wrong_component["pose_targets_by_component"] = {
        "left_arm": payload["pose_targets_by_component"][profile.component_id]
    }
    with pytest.raises(ValueError, match="unsupported components"):
        bundle.payload_validators.validate_parameters(
            route=route,
            payload=wrong_component,
        )


def test_real_contract_refuses_missing_or_unbound_lower_safety_envelope() -> None:
    attachment, port_config, graph, envelope = _inputs()
    kwargs = {
        "attachment": attachment,
        "port_config": port_config,
        "frame_graph": graph,
        "primitive_reference_frame_id": "canonical_station",
    }
    with pytest.raises(ValueError, match="require lower safety-envelope"):
        build_real_agent_primitive_contract_bundle(
            **kwargs,
            safety_envelope_refs=(),
        )

    unbound = SkillDeploymentSafetyEnvelopeRef(
        object_id=envelope.object_id,
        schema=envelope.schema,
        content_hash=envelope.content_hash,
        constraint_set_hash=envelope.constraint_set_hash,
        allowed_route_ids=("real.some_other_route",),
    )
    with pytest.raises(ValueError, match="absent from all lower envelopes"):
        build_real_agent_primitive_contract_bundle(
            **kwargs,
            safety_envelope_refs=(unbound,),
        )


def test_real_contract_refuses_an_explicit_nonstation_public_frame() -> None:
    attachment, port_config, graph, envelope = _inputs()
    alternate = CoordinateFrameSpec(frame_id="alternate_public_frame")
    extended_graph = CoordinateFrameGraph(
        graph_id=graph.graph_id,
        graph_version=graph.graph_version,
        frames=(*graph.frames, alternate),
        transforms=graph.transforms,
    )
    explicit_mismatch = replace(
        attachment,
        reference_frame_id=alternate.frame_id,
        reference_frame_spec_hash=alternate.semantic_content_hash,
        execution_reference_frame_id=port_config.execution_frame_id,
        execution_reference_frame_spec_hash=(
            port_config.execution_frame_spec_hash
        ),
        frame_graph_hash=extended_graph.semantic_content_hash,
    )

    with pytest.raises(ValueError, match="public frame must be the primitive Station"):
        build_real_agent_primitive_contract_bundle(
            attachment=explicit_mismatch,
            port_config=port_config,
            frame_graph=extended_graph,
            primitive_reference_frame_id="canonical_station",
            safety_envelope_refs=(envelope,),
        )


def test_convergence_profile_is_content_addressed() -> None:
    first = RealAgentPrimitiveConvergenceProfile()
    second = RealAgentPrimitiveConvergenceProfile(stable_measurement_count=4)
    assert first.semantic_content_hash != second.semantic_content_hash


def test_convergence_profile_requires_positive_hold_confirmation_timeout() -> None:
    with pytest.raises(ValueError, match="hold_confirmation_timeout_sec"):
        RealAgentPrimitiveConvergenceProfile(hold_confirmation_timeout_sec=0.0)


@pytest.mark.parametrize("component", ["left_arm", "right_arm"])
@pytest.mark.parametrize("skill", ["motion.move_tcp_to_pose_safe", "motion.move_tcp_to_pose",
                                   "motion.approach_grasp_pose"])
def test_safe_goal_policy_contract_is_confined_to_safe_pose_route(component, skill):
    attachment, config, graph, envelope = _inputs()
    side = component.removesuffix("_arm")
    attachment = replace(attachment, component_ids=(component,))
    config = replace(config, active_sides=(side,), component_id_by_side={side: component})
    envelope = replace(envelope, allowed_route_ids=real_agent_motion_route_ids_v1(
        include_safe_pose=True, include_grasp_approach=True))
    bundle = build_real_agent_primitive_contract_bundle(attachment=attachment,
        port_config=config, frame_graph=graph, primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=(envelope,))
    profile = bundle.pose_target_profile
    route = bundle.route_manifest.route(skill, "1").route
    entry = bundle.schema_catalog.entry(skill, "1")
    payload = {"pose_targets_by_component": {component: [0., 0., .3, 0., 0., 0., 1.]},
        "reference_frame_id": profile.reference_frame_id,
        "reference_frame_spec_hash": profile.reference_frame_spec_hash,
        "controlled_frame_id": "tool_tcp", "frame_graph_hash": profile.frame_graph_hash,
        "tool_config_hash": profile.tool_config_hash, "calibration_hash": profile.calibration_hash}
    bundle.payload_validators.validate_parameters(route=route, payload=payload)
    safe = skill == "motion.move_tcp_to_pose_safe"
    assert ("pose_goal_policy" in entry.parameter_schema["properties"]) is safe
    assert ("pose_goal_report" in entry.execution_result_schema["properties"]) is safe
    assert "pose_goal_policy" not in entry.parameter_schema["required"]
    assert "pose_goal_report" not in entry.execution_result_schema["required"]
    assert route.parameter_schema_hash == entry.parameter_schema_hash
    if safe:
        assert entry.parameter_schema["properties"]["pose_goal_policy"]["default"] == "within_arrival_tolerance"
    for policy in ("preserve_target", "within_arrival_tolerance", "bad", "", None, True, [], {}):
        if safe and policy in ("preserve_target", "within_arrival_tolerance"):
            bundle.payload_validators.validate_parameters(route=route,
                payload={**payload, "pose_goal_policy": policy})
        else:
            with pytest.raises((ValueError, TypeError)):
                bundle.payload_validators.validate_parameters(route=route,
                    payload={**payload, "pose_goal_policy": policy})
    legacy_result = {"execution_receipt_refs": ["execution:test"], "final_measured_state_hash": "state:test"}
    bundle.payload_validators.validate_execution_result(route=route, payload=legacy_result)
    report = {"pose_goal_policy": "preserve_target", "reference_frame_id": profile.reference_frame_id,
        "requested_pose_xyz_quat_xyzw": [0., 0., .3, 0., 0., 0., 1.],
        "achieved_pose_xyz_quat_xyzw": [0., 0., .306, 0., 0., 0., 1.],
        "translation_error_m": .006, "rotation_error_rad": 0.,
        "arrival_translation_tolerance_m": .01, "arrival_rotation_tolerance_rad": .1}
    if safe:
        bundle.payload_validators.validate_execution_result(route=route,
            payload={**legacy_result, "pose_goal_report": report})
    else:
        with pytest.raises(ValueError):
            bundle.payload_validators.validate_execution_result(route=route,
                payload={**legacy_result, "pose_goal_report": report})


@pytest.mark.parametrize("changes", [
    {"extra": True}, {"pose_goal_policy": None}, {"reference_frame_id": " "},
    {"requested_pose_xyz_quat_xyzw": [0.] * 7},
    {"achieved_pose_xyz_quat_xyzw": [0., 0., .3, 0., 0., 0., 2.]},
    {"translation_error_m": True}, {"translation_error_m": float("inf")},
    {"rotation_error_rad": float("nan")}, {"rotation_error_rad": -.1},
    {"arrival_translation_tolerance_m": 0.}, {"arrival_rotation_tolerance_rad": -.1},
])
def test_safe_pose_goal_report_rejects_malformed_values(changes):
    from agent_gateway.skills.motion.schemas import validate_safe_tcp_pose_goal_report_v1
    report = {"pose_goal_policy": "preserve_target", "reference_frame_id": "canonical_station",
        "requested_pose_xyz_quat_xyzw": [0., 0., .3, 0., 0., 0., 1.],
        "achieved_pose_xyz_quat_xyzw": [0., 0., .306, 0., 0., 0., 1.],
        "translation_error_m": .006, "rotation_error_rad": 0.,
        "arrival_translation_tolerance_m": .01, "arrival_rotation_tolerance_rad": .1}
    with pytest.raises((ValueError, TypeError)):
        validate_safe_tcp_pose_goal_report_v1({**report, **changes})

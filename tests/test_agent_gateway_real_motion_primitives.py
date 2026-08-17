from __future__ import annotations

import numpy as np
import pytest

from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    RealAgentPrimitiveConvergenceProfile,
    build_real_agent_primitive_contract_bundle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.skills.motion.definition import (
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
    route_ids = (
        REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
        *tuple(
            REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[skill_id]
            for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        ),
    )
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-agent-motion-envelope:test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-agent-motion-envelope-hash:test",
        constraint_set_hash="real-agent-motion-constraints-hash:test",
        allowed_route_ids=route_ids,
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


def test_real_bundle_exposes_read_plus_six_plus_three_motion_primitives() -> None:
    bundle = _bundle()

    assert len(bundle.route_manifest.skill_routes) == 5
    assert len(bundle.cardinal_profile.primitives) == 6
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


def test_convergence_profile_is_content_addressed() -> None:
    first = RealAgentPrimitiveConvergenceProfile()
    second = RealAgentPrimitiveConvergenceProfile(stable_measurement_count=4)
    assert first.semantic_content_hash != second.semantic_content_hash


def test_convergence_profile_requires_positive_hold_confirmation_timeout() -> None:
    with pytest.raises(ValueError, match="hold_confirmation_timeout_sec"):
        RealAgentPrimitiveConvergenceProfile(hold_confirmation_timeout_sec=0.0)

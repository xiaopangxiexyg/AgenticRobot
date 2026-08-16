from __future__ import annotations

from dataclasses import MISSING, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.motion_factory import (
    attach_real_agent_primitive_gateway,
)
from agent_gateway.profiles.real.motion_gateway_session import (
    assemble_real_agent_primitive_gateway_session,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    build_real_agent_primitive_contract_bundle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealControlOwnerState,
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.runtime.access_policy import issue_skill_execution_grant
from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from agent_gateway.skills.motion.schemas import CARDINAL_TCP_STEP_DIRECTIONS_V1
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.real_teleop.components import RealTeleopComponents

CLIENT_ID = "rpent-real-motion-test"
COMPONENT_ID = "right_arm"


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _TargetBuilder:
    def __init__(self) -> None:
        self.last_target = None

    def current_tool_tcp_world_targets(self):
        return {
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        }

    def build_command_for_target_pair(self, target):
        self.last_target = target
        command = TeleopCommand(timestamp=target.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.zeros(6, dtype=float),
            )
        )
        return command

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        del left_qpos, right_qpos
        return self.current_tool_tcp_world_targets()


class _OwnerSession:
    def __init__(self) -> None:
        self.state_machine = SimpleNamespace(state=TeleopRunState.ARMED)
        self.holds: list[tuple[str, bool]] = []

    def hold_hardware(self, reason, *, require_home_recalibration=False):
        self.holds.append((reason, require_home_recalibration))
        self.state_machine.state = TeleopRunState.DEBUG_ONLY
        return True


def _runtime_fixture():
    execution_frame = CoordinateFrameSpec(frame_id="dual_arm_solver_world")
    station_frame = CoordinateFrameSpec(
        frame_id="canonical_station",
        semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    frame_graph = CoordinateFrameGraph(
        graph_id="real-agent-session-test-graph",
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
        runtime_session_id="real-session:gateway-test",
        control_domain_id="real-domain:gateway-test",
        embodiment_instance_id="dual-rm65:gateway-test",
        target_binding_id="real-target:gateway-test",
        component_ids=(COMPONENT_ID,),
        station_manifest_hash="station-manifest:gateway-test",
        execution_manifest_hash="execution-manifest:gateway-test",
        backend_inventory_hash="backend-inventory:gateway-test",
        source_binding_id="source-binding:gateway-test",
        source_binding_hash="source-binding-hash:gateway-test",
        observation_binding_id="observation-binding:gateway-test",
        observation_binding_hash="observation-binding-hash:gateway-test",
        observation_plan_hash="observation-plan-hash:gateway-test",
        observation_capability_hash="observation-capability-hash:gateway-test",
        pose_schema_hash="pose-schema-hash:gateway-test",
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash="tool-config-hash:gateway-test",
        calibration_hash="calibration-hash:gateway-test",
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
        object_id="real-envelope:gateway-test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:gateway-test",
        constraint_set_hash="real-constraint-hash:gateway-test",
        allowed_route_ids=route_ids,
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=attachment,
        port_config=port_config,
        frame_graph=frame_graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
    )
    target_builder = _TargetBuilder()
    port = RealAgentMotionPort(
        config=port_config,
        target_builder=target_builder,
    )
    return bundle, port, target_builder


def _owner_state(bundle, *, captured_at: float) -> RealControlOwnerState:
    attachment = bundle.attachment
    return RealControlOwnerState(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        execution_manifest_hash=attachment.execution_manifest_hash,
        backend_inventory_hash=attachment.backend_inventory_hash,
        clock_domain_id=attachment.clock_domain_id,
        owner_token_id="real-owner-token:gateway-test",
        owner_kind="realteleop_control_loop",
        owner_generation=3,
        authority_epoch=5,
        execution_epoch=7,
        lifecycle_state="running",
        execution_gate="ready",
        owner_state_revision=11,
        captured_at=captured_at,
        source_owner_state_hash="lower-owner-state:gateway-test",
        evidence_refs=("lower-owner-evidence:gateway-test",),
    )


def _request(bundle, snapshot, *, skill_id: str, direction: str | None):
    entry = bundle.route_manifest.route(skill_id, "1")
    route = entry.route
    if skill_id == "motion.step_tcp_cardinal":
        profile = bundle.cardinal_profile
        parameters = {
            "component_id": profile.component_id,
            "direction": direction,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_catalog_hash": profile.primitive_catalog_hash,
        }
    else:
        profile = bundle.orientation_profile(skill_id)
        parameters = {
            "component_id": profile.component_id,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_hash": profile.primitive.semantic_content_hash,
        }
    return AgentSkillRequest(
        client_id=CLIENT_ID,
        request_id=f"request:{skill_id}:{direction or 'fixed'}",
        idempotency_key=f"idempotency:{skill_id}:{direction or 'fixed'}",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        parameters=parameters,
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="real_active",
        requested_timeout_ms=5_000,
    )


PRIMITIVE_CASES = (
    *tuple(
        ("motion.step_tcp_cardinal", direction)
        for direction in CARDINAL_TCP_STEP_DIRECTIONS_V1
    ),
    *tuple(
        (skill_id, None)
        for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    ),
)


@pytest.mark.parametrize(("skill_id", "direction"), PRIMITIVE_CASES)
def test_real_gateway_executes_each_reviewed_primitive_through_grant_and_hold(
    skill_id: str,
    direction: str | None,
) -> None:
    bundle, port, target_builder = _runtime_fixture()
    clock = _Clock()
    owner_session = _OwnerSession()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:initial",),
    )
    owner_state = _owner_state(bundle, captured_at=clock())
    publish_sequence = 0

    def owner_sleep(duration: float) -> None:
        nonlocal publish_sequence
        clock.value += duration
        lower = port.apply_owner_requests(session=owner_session, now=clock())
        if lower.phase != RealAgentMotionPhase.ACTIVE:
            return
        observation = TeleopObservation(timestamp=clock())
        command = port.command_for_observation(observation, clock())
        assert isinstance(command, TeleopCommand)
        assert port.admit_command(command, observation, clock())
        port.observe_session_step(command=command, command_dispatched=True)
        assert target_builder.last_target is not None
        publish_sequence += 1
        port.publish_measured_pose(
            pose_world_by_component={
                COMPONENT_ID: target_builder.last_target.target_pose_by_side["right"]
            },
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=(f"real-hardware-cut:{publish_sequence}",),
        )

    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner_state,
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
        sleep=owner_sleep,
    )
    observation_outcome = gateway.observation_producer.capture_and_publish(
        gateway.service
    )
    assert observation_outcome.accepted
    snapshot = observation_outcome.snapshot
    assert snapshot is not None
    pre_grant_request = _request(
        bundle,
        snapshot,
        skill_id=skill_id,
        direction=direction,
    )
    rejected = gateway.service.submit_request(
        request=pre_grant_request,
        current_station_revision=port.snapshot().revision,
        now=clock(),
    )
    assert rejected.accepted is False
    assert rejected.reason_code == "skill_execution_grant_required"

    route = bundle.route_manifest.route(skill_id, "1").route
    grant = issue_skill_execution_grant(
        access_policy=gateway.service.access_policy,
        manifest=bundle.route_manifest,
        route=route,
        request=pre_grant_request,
        operator_authorization_receipt_id=(
            f"operator-confirmation:{skill_id}:{direction or 'fixed'}"
        ),
        client_id=CLIENT_ID,
        parameter_profile_id=pre_grant_request.parameter_profile_id,
        parameter_profile_hash=pre_grant_request.parameter_profile_hash,
        max_invocations=1,
        issued_at=clock(),
        expires_at=clock() + 5.0,
    )
    request = replace(
        pre_grant_request,
        skill_execution_grant_id=grant.grant_id,
    )
    submission = gateway.service.submit_request(
        request=request,
        current_station_revision=port.snapshot().revision,
        grant=grant,
        now=clock(),
    )
    assert submission.accepted
    assert submission.operation is not None
    launch = gateway.service.launch_operation(submission.operation.operation_id)
    assert launch.accepted
    assert gateway.wait(timeout_sec=1.0)

    operation = gateway.service.get_operation(submission.operation.operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.execution_result is not None
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert port.snapshot().dispatch_count >= 3
    assert len(owner_session.holds) == 1
    assert gateway.close().accepted


def test_real_gateway_refuses_a_port_from_another_execution_binding() -> None:
    bundle, _, target_builder = _runtime_fixture()
    mismatched = RealAgentMotionPort(
        config=RealAgentMotionPortConfig(
            active_sides=("right",),
            component_id_by_side={"right": COMPONENT_ID},
            execution_frame_id="another_execution_frame",
            execution_frame_spec_hash="another_execution_frame_hash",
        ),
        target_builder=target_builder,
    )

    with pytest.raises(ValueError, match="execution_frame_id mismatch"):
        assemble_real_agent_primitive_gateway_session(
            contract_bundle=bundle,
            port=mismatched,
            owner_state_provider=lambda: _owner_state(bundle, captured_at=10.0),
            allowed_client_ids=(CLIENT_ID,),
        )


def test_real_factory_attaches_gateway_and_realteleop_to_one_lower_port() -> None:
    bundle, port, target_builder = _runtime_fixture()
    session = SimpleNamespace(
        command_override_provider=None,
        command_dispatch_admission=None,
    )
    component_values = {}
    for item in fields(RealTeleopComponents):
        if item.default is MISSING and item.default_factory is MISSING:
            component_values[item.name] = None
    component_values.update(
        session=session,
        active_sides=("right",),
        dual_arm_retargeter=target_builder,
    )
    components = RealTeleopComponents(**component_values)
    clock = _Clock()

    attached = attach_real_agent_primitive_gateway(
        components,
        port_config=port.config,
        observation_attachment=bundle.attachment,
        frame_graph=bundle.frame_graph,
        primitive_reference_frame_id=(
            bundle.cardinal_profile.reference_frame_id
        ),
        safety_envelope_refs=(
            bundle.deployment_bindings[0].safety_envelope_refs
        ),
        owner_state_provider=lambda: _owner_state(
            bundle,
            captured_at=clock(),
        ),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )

    assert components.agent_motion_port is None
    assert attached.components.agent_motion_port is attached.gateway_session.port
    assert session.command_override_provider == (
        attached.gateway_session.port.command_for_observation
    )
    assert set(attached.gateway_session.executors_by_skill_id) == {
        "motion.step_tcp_cardinal",
        *FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    }


def test_wire_adapter_atomically_binds_one_use_grant_to_exact_request() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:wire",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(
            bundle,
            captured_at=clock(),
        ),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )
    observation = gateway.observation_producer.capture_and_publish(gateway.service)
    assert observation.accepted and observation.snapshot is not None
    request = _request(
        bundle,
        observation.snapshot,
        skill_id="motion.step_tcp_cardinal",
        direction="forward",
    )
    assert request.skill_execution_grant_id is None

    issued = []

    def grant_resolver(parsed_request):
        route = bundle.route_manifest.route(
            parsed_request.skill_id,
            parsed_request.skill_version,
            parsed_request.route_id,
            parsed_request.route_version,
        ).route
        grant = issue_skill_execution_grant(
            access_policy=gateway.service.access_policy,
            manifest=bundle.route_manifest,
            route=route,
            request=parsed_request,
            operator_authorization_receipt_id="operator-confirmation:wire",
            client_id=CLIENT_ID,
            parameter_profile_id=parsed_request.parameter_profile_id,
            parameter_profile_hash=parsed_request.parameter_profile_hash,
            max_invocations=1,
            issued_at=clock(),
            expires_at=clock() + 5.0,
        )
        issued.append(grant)
        return grant

    adapter = BoundAgentGatewayWireAdapter(
        authenticated_client_id=CLIENT_ID,
        definition_catalog=bundle.definition_catalog,
        schema_catalog=bundle.schema_catalog,
        service=gateway.service,
        publish_observation=lambda: gateway.observation_producer.capture_and_publish(
            gateway.service
        ),
        current_station_revision=lambda: (
            gateway.observation_producer.current_station_revision
        ),
        operation_launcher=lambda _operation_id: SimpleNamespace(
            accepted=True,
            status="intentionally_not_launched_in_admission_test",
        ),
        grant_resolver=grant_resolver,
    )
    result = adapter.submit_skill_request(request.to_wire())

    assert result["accepted"] is True
    assert len(issued) == 1
    operation = result["operation"]
    assert operation is not None
    assert operation["status"] == "admitted"

from __future__ import annotations

import time
from dataclasses import MISSING, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import gripper_skill as gripper_module
from agent_gateway.profiles.real.gripper_skill import (
    RealDmtacGripperControlPort,
    build_real_dmtac_gripper_skill_contract_bundle,
    build_real_dmtac_grippers_skill_contract_bundle,
)
from agent_gateway.profiles.real.motion_factory import (
    attach_real_agent_primitive_gateway,
)
from agent_gateway.profiles.real.motion_gateway_session import (
    RealAgentDomainEmergencyStopPort,
    assemble_real_agent_primitive_gateway_session,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    build_real_agent_primitive_contract_bundle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealControlOwnerState,
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.profiles.real.one_shot_authorization import (
    RealAgentPrimitiveOneShotAuthorization,
)
from agent_gateway.profiles.real.perception_deployment import (
    build_real_perception_capture_registration,
)
from agent_gateway.profiles.real.placement import RealPlacementRobotStatePort
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    RealAgentPrimitiveSessionAuthorization,
)
from agent_gateway.profiles.segmentation import build_segmentation_registration
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.runtime.access_policy import issue_skill_execution_grant
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.skills.motion.definition import (
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from agent_gateway.skills.motion.schemas import CARDINAL_TCP_STEP_DIRECTIONS_V1
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectSegment,
    encode_grounded_object_segment_artifacts,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.real_teleop.components import RealTeleopComponents
from teleop_runtime.skill_deploy.control_runtime import (
    RealSkillMotionEmergencyStopReceipt,
)
from teleop_runtime.video_frame import VideoFrame

CLIENT_ID = "rpent-real-motion-test"
COMPONENT_ID = "right_arm"


def test_real_domain_emergency_stop_port_preserves_dual_arm_proof() -> None:
    reasons = []

    def emergency_stop(reason: str) -> RealSkillMotionEmergencyStopReceipt:
        reasons.append(reason)
        return RealSkillMotionEmergencyStopReceipt(
            succeeded=True,
            evidence_refs=("rm65-estop:left", "rm65-estop:right"),
            status_by_side={"left": "confirmed", "right": "confirmed"},
        )

    port = RealAgentDomainEmergencyStopPort(
        port_id="real-emergency-stop:test",
        emergency_stop=emergency_stop,
        clock=lambda: 10.0,
    )

    receipt = port.request_emergency_stop(
        operation=None,
        reason="agent_task_aborted",
    )

    assert receipt.succeeded
    assert receipt.operation_kind == "emergency_stop"
    assert receipt.proves_motion_or_hold
    assert receipt.evidence_refs == ("rm65-estop:left", "rm65-estop:right")
    assert reasons == ["agent_task_aborted"]


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

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

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


class _PersistentOwnerSession(_OwnerSession):
    def hold_hardware(self, reason, *, require_home_recalibration=False):
        self.holds.append((reason, require_home_recalibration))
        self.state_machine.state = TeleopRunState.ARMED
        return True


def _runtime_fixture(
    *,
    exclusive_owner: bool = False,
    station_from_execution: np.ndarray | None = None,
    spatial: bool = False,
):
    if station_from_execution is None:
        station_from_execution = np.eye(4, dtype=float)
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
                matrix_target_source=tuple(
                    tuple(row)
                    for row in np.asarray(station_from_execution, dtype=float)
                ),
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
        exclusive_command_claim=exclusive_owner,
    )
    route_ids = (
        REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
        *tuple(
            REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1[skill_id]
            for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
        ),
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
        enable_spatial_goals=spatial,
        attachment=attachment,
        port_config=port_config,
        frame_graph=frame_graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
    )
    target_builder = _TargetBuilder()
    owner_identity = (
        RealAgentMotionOwnerIdentity(
            runtime_session_id=attachment.runtime_session_id,
            control_domain_id=attachment.control_domain_id,
            embodiment_instance_id=attachment.embodiment_instance_id,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=attachment.execution_manifest_hash,
            backend_inventory_hash=attachment.backend_inventory_hash,
            clock_domain_id=attachment.clock_domain_id,
        )
        if exclusive_owner
        else None
    )
    port = RealAgentMotionPort(
        config=port_config,
        target_builder=target_builder,
        owner_identity=owner_identity,
    )
    if exclusive_owner:
        port.confirm_exclusive_attachment()
        port.publish_owner_state(session=_OwnerSession(), captured_at=10.0)
    return bundle, port, target_builder


def test_real_placement_state_uses_fresh_station_frame_tcp_measurement() -> None:
    station_from_execution = np.eye(4, dtype=float)
    station_from_execution[:3, 3] = (0.2, -0.1, 0.3)
    bundle, port, _ = _runtime_fixture(
        station_from_execution=station_from_execution
    )
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:placement",),
    )
    state_port = RealPlacementRobotStatePort(
        port=port,
        frame_graph=bundle.frame_graph,
        component_id=COMPONENT_ID,
        execution_reference_frame_id=(
            bundle.attachment.execution_reference_frame_id
        ),
        reference_frame_id=bundle.cardinal_profile.reference_frame_id,
        maximum_measurement_age_ms=200,
        current_station_revision=lambda: 4,
        clock=clock,
    )

    captured = state_port.capture_placement_robot_state(
        expected_station_revision=4
    )

    assert captured.tool_tcp_reference_xyz_quat_xyzw == pytest.approx(
        (0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0)
    )
    assert captured.gripper_close_fraction is None
    assert captured.station_revision == 4


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


def _request(
    bundle,
    snapshot,
    *,
    skill_id: str,
    direction: str | None,
    client_sequence: int = 1,
    request_suffix: str | None = None,
):
    entry = bundle.route_manifest.route(skill_id, "1")
    route = entry.route
    if skill_id == "motion.step_tcp_cardinal" or skill_id in (
        FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
    ):
        profile = bundle.cardinal_profile
        fixed_direction = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1.get(skill_id)
        parameters = {
            "component_id": profile.component_id,
            "direction": fixed_direction or direction,
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
    identity = request_suffix or direction or "fixed"
    return AgentSkillRequest(
        client_id=CLIENT_ID,
        request_id=f"request:{skill_id}:{identity}",
        idempotency_key=f"idempotency:{skill_id}:{identity}",
        client_sequence=client_sequence,
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


def _read_request(bundle, snapshot) -> AgentSkillRequest:
    entry = bundle.route_manifest.route("state.read_tcp_pose", "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=CLIENT_ID,
        request_id="request:state.read_tcp_pose:1",
        idempotency_key="idempotency:state.read_tcp_pose:1",
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
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={
            "component_ids": list(bundle.attachment.component_ids),
            "reference_frame_id": bundle.attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": bundle.attachment.max_measurement_age_ms,
        },
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="real_active",
        requested_timeout_ms=1_000,
    )


PRIMITIVE_CASES = (
    *tuple(
        ("motion.step_tcp_cardinal", direction)
        for direction in CARDINAL_TCP_STEP_DIRECTIONS_V1
    ),
    *tuple(
        (
            skill_id,
            FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[skill_id],
        )
        for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
    ),
    *tuple((skill_id, None) for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1),
)


class _ComposedGripperBackend:
    def __init__(self, side="right", *, clock=None):
        self.config = SimpleNamespace(
            side=side, gripper_port="/dev/fake-" + side, gripper_id=1
        )
        self.commands = []
        self.holds = []
        self.close_fraction = 0.0
        self.stop_ok = True
        self.clock = clock

    def is_ready(self):
        return True

    def readiness_report(self):
        return {"ready": True}

    def read_gripper_state_now(self):
        if self.clock is not None:
            self.clock.value += .001
            return {"timestamp": self.clock()}
        return {"timestamp": time.time()}

    def get_state(self, **kwargs):
        return {
            "gripper_measured_scalar": self.close_fraction,
            "gripper_is_moving": False,
            "gripper_error_code": 0,
        }

    def send_fixed_joint_pos(self, value):
        self.commands.append(value)
        self.close_fraction = value
        return True

    def hold_gripper_motion(self, reason):
        self.holds.append(reason)
        return self.stop_ok


@pytest.fixture
def _composed_gripper_fixture(monkeypatch):
    return lambda: _build_composed_gripper_fixture(monkeypatch=monkeypatch)


def _build_composed_gripper_fixture(*, monkeypatch):
    base, port, _ = _runtime_fixture(exclusive_owner=True)
    clock = _Clock()
    # Only the adapter time port is replaced. Service/thread time remains real;
    # acquisition, wall age and HOLD deadlines share this fixture's logical cut.
    monkeypatch.setattr(gripper_module, "time", SimpleNamespace(
        monotonic=clock, time=clock,
        sleep=lambda duration: setattr(clock, "value", clock.value + duration)))
    owner = _PersistentOwnerSession()
    port.request_domain_hold(reason="fixture startup", requested_at=clock())
    port.apply_owner_requests(session=owner, now=clock())
    port.publish_owner_state(session=owner, captured_at=clock())
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("fixture:fresh-qpos",),
    )
    backend = _ComposedGripperBackend(clock=clock)
    hand = RealDmtacGripperControlPort(backend=backend, clock=clock)
    contract = build_real_dmtac_gripper_skill_contract_bundle(
        runtime_session_id=base.attachment.runtime_session_id,
        control_domain_id=base.attachment.control_domain_id,
        target_binding_id=base.attachment.target_binding_id,
        station_manifest_hash=base.attachment.station_manifest_hash,
        execution_manifest_hash=base.attachment.execution_manifest_hash,
        port=hand,
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=base.attachment,
        port_config=port.config,
        frame_graph=base.frame_graph,
        primitive_reference_frame_id=base.cardinal_profile.reference_frame_id,
        safety_envelope_refs=base.deployment_bindings[0].safety_envelope_refs,
        gripper_contract=contract,
    )

    def sleep(duration):
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())
        port.publish_owner_state(session=owner, captured_at=clock())
        port.publish_measured_pose(
            pose_world_by_component={COMPONENT_ID: np.eye(4)},
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=("fixture:fresh-qpos",),
        )

    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        gripper_port=hand,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
        sleep=sleep,
    )
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service
    ).snapshot
    assert snapshot is not None
    entry = bundle.route_manifest.route("manipulation.set_gripper_state", "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    request = replace(
        _read_request(bundle, snapshot),
        request_id="request:gripper",
        idempotency_key="idempotency:gripper",
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={"target_state": "closed"},
        requested_timeout_ms=5000,
    )
    return gateway, backend, request, snapshot, clock


def test_composed_gripper_requires_separate_authorization_and_executes_under_hold(_composed_gripper_fixture):
    gateway, backend, request, snapshot, clock = _composed_gripper_fixture()
    try:
        arm_auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("forward",),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        assert arm_auth.grant_resolver(request) is None
        assert not gateway.service.submit_request(
            request=request, current_station_revision=snapshot.station_revision
        ).accepted
        assert not backend.commands
        auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("set_gripper_state",),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        grant = auth.grant_resolver(request)
        request = replace(request, skill_execution_grant_id=grant.grant_id)
        admitted = gateway.service.submit_request(
            request=request,
            grant=grant,
            current_station_revision=snapshot.station_revision,
        )
        assert admitted.accepted
        assert gateway.service.launch_operation(
            admitted.operation.operation_id
        ).accepted
        assert gateway.wait(timeout_sec=3)
        result = gateway.service.get_operation(admitted.operation.operation_id)
        assert result.execution_outcome == "completed"
        assert result.execution_result["terminal_hold_confirmed"] is True
        assert backend.commands == [1.0]
        assert gateway.port.snapshot().phase == RealAgentMotionPhase.HELD
    finally:
        assert gateway.close().accepted


def test_composed_gripper_reserves_the_same_operation_slot_as_arm_motion(_composed_gripper_fixture):
    gateway, backend, request, snapshot, clock = _composed_gripper_fixture()
    try:
        auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("forward", "set_gripper_state"),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        grant = auth.grant_resolver(request)
        request = replace(request, skill_execution_grant_id=grant.grant_id)
        assert gateway.service.submit_request(
            request=request,
            grant=grant,
            current_station_revision=snapshot.station_revision,
        ).accepted
        arm = _request(
            gateway.contract_bundle,
            snapshot,
            skill_id="motion.step_tcp_forward",
            direction="forward",
            client_sequence=2,
        )
        grant = auth.grant_resolver(arm)
        arm = replace(arm, skill_execution_grant_id=grant.grant_id)
        rejected = gateway.service.submit_request(
            request=arm, grant=grant, current_station_revision=snapshot.station_revision
        )
        assert not rejected.accepted
        assert "active" in rejected.reason_code or "busy" in rejected.reason_code
        assert not backend.commands
    finally:
        assert gateway.close().accepted


def test_composed_gateway_does_not_confirm_shutdown_if_gripper_stop_fails(_composed_gripper_fixture):
    gateway, backend, _, _, _ = _composed_gripper_fixture()
    backend.stop_ok = False
    assert not gateway.close().accepted
    backend.stop_ok = True
    assert gateway.close().accepted


@pytest.mark.parametrize("hand_stop_ok", [False, True])
def test_composed_emergency_stop_requires_arm_and_hand_proof(hand_stop_ok):
    backend = _ComposedGripperBackend()
    backend.stop_ok = hand_stop_ok
    hand = RealDmtacGripperControlPort(backend=backend)
    reasons = []

    def stop_arms(reason):
        reasons.append(reason)
        return RealSkillMotionEmergencyStopReceipt(
            succeeded=True,
            evidence_refs=("arm-estop:left", "arm-estop:right"),
            status_by_side={"left": "confirmed", "right": "confirmed"},
        )

    port = RealAgentDomainEmergencyStopPort(
        port_id="composed-estop:test", emergency_stop=stop_arms, gripper_port=hand
    )
    receipt = port.request_emergency_stop(operation=None, reason="operator stop")
    assert receipt.succeeded is hand_stop_ok
    assert receipt.proves_motion_or_hold is hand_stop_ok
    assert reasons == backend.holds == ["operator stop"]
    assert {"arm-estop:left", "arm-estop:right"}.issubset(receipt.evidence_refs)
    assert hand.last_hold_receipt is not None
    assert set(hand.last_hold_receipt.evidence_refs).issubset(receipt.evidence_refs)


def test_composed_emergency_stop_still_stops_hand_after_arm_exception():
    backend = _ComposedGripperBackend()
    hand = RealDmtacGripperControlPort(backend=backend)

    def stop_arms(reason):
        raise RuntimeError("arm stop unavailable")

    port = RealAgentDomainEmergencyStopPort(
        port_id="composed-estop:test", emergency_stop=stop_arms, gripper_port=hand
    )
    with pytest.raises(RuntimeError, match="arm stop unavailable"):
        port.request_emergency_stop(operation=None, reason="operator stop")
    assert backend.holds == ["operator stop"]
    assert port.receipts == ()


def test_composed_gripper_accepts_fresh_confirmed_hold_observation_without_resuming_arm(_composed_gripper_fixture):
    gateway, _, _, _, clock = _composed_gripper_fixture()
    owner = replace(
        _owner_state(gateway.contract_bundle, captured_at=clock()),
        lifecycle_state="hold",
        execution_gate="hold",
    )
    gateway.observation_producer.owner_state_provider = lambda: owner
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service
    ).snapshot
    assert snapshot.lifecycle_state == "hold"
    assert snapshot.execution_gate == "hold"
    assert gateway.port.snapshot().phase == RealAgentMotionPhase.HELD
    assert gateway.close().accepted


def test_gripper_command_attempt_invalidates_scene_revision_but_passive_reads_do_not(_composed_gripper_fixture):
    gateway, _, _, snapshot, _ = _composed_gripper_fixture()
    try:
        gateway.gripper_port.observe_state()
        assert (
            gateway.observation_producer.current_station_revision
            == snapshot.station_revision
        )
        gateway.gripper_port.command_close(1.0)
        assert (
            gateway.observation_producer.current_station_revision
            > snapshot.station_revision
        )
    finally:
        assert gateway.close().accepted


def test_composed_gripper_rejects_motion_after_arm_owner_becomes_stale(_composed_gripper_fixture):
    gateway, backend, _, _, clock = _composed_gripper_fixture()
    clock.value += 0.3
    with pytest.raises(RuntimeError, match="fresh, healthy"):
        gateway.gripper_port.command_close(1.0)
    assert not backend.commands
    assert gateway.close().accepted


def test_composed_gripper_rejects_fresh_owner_with_stale_arm_measurement(_composed_gripper_fixture):
    gateway, backend, _, _, clock = _composed_gripper_fixture()
    clock.value += 0.3
    gateway.port.publish_owner_state(
        session=_PersistentOwnerSession(), captured_at=clock()
    )
    with pytest.raises(RuntimeError, match="fresh, healthy"):
        gateway.gripper_port.command_close(1.0)
    assert not backend.commands
    assert gateway.close().accepted


@pytest.mark.parametrize("expiring_cut", ("arm_measurement", "arm_owner", "fresh"))
def test_composed_gripper_rechecks_arm_hold_before_io_busy_retry(
    _composed_gripper_fixture, expiring_cut
):
    gateway, backend, _, _, clock = _composed_gripper_fixture()
    maximum_age = gateway.contract_bundle.attachment.max_measurement_age_ms / 1000.0
    # Leave 2 ms of validity; the existing io_busy backoff lasts 5 ms.
    if expiring_cut != "fresh":
        clock.value += maximum_age - 0.002
    if expiring_cut == "arm_measurement":
        gateway.port.publish_owner_state(
            session=_PersistentOwnerSession(), captured_at=clock()
        )
    elif expiring_cut == "arm_owner":
        gateway.port.publish_measured_pose(
            pose_world_by_component={COMPONENT_ID: np.eye(4)},
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=("fixture:fresh-qpos-before-busy",),
        )
    attempts = []
    send = backend.send_fixed_joint_pos
    get_state = backend.get_state

    def busy_once(target):
        attempts.append(clock())
        if len(attempts) == 1:
            return False
        return send(target)

    backend.send_fixed_joint_pos = busy_once
    backend.get_state = lambda **kwargs: {
        **get_state(**kwargs), "gripper_last_send_status": "io_busy"
    }
    try:
        if expiring_cut == "fresh":
            assert gateway.gripper_port.command_close(1.0).accepted
            assert len(attempts) == 2
            assert backend.commands == [1.0]
        else:
            with pytest.raises(RuntimeError, match="fresh, healthy"):
                gateway.gripper_port.command_close(1.0)
            assert len(attempts) == 1
            assert not backend.commands
    finally:
        assert gateway.close().accepted


def test_composed_gripper_client_uses_real_wire_admission_and_hand_constraint_binding(
    monkeypatch,
    _composed_gripper_fixture,
    tmp_path,
):
    from argparse import Namespace

    from agent_gateway.profiles.real import one_shot_client

    gateway, backend, _, _, clock = _composed_gripper_fixture()
    auth = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        authorized_actions=("set_gripper_state",),
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )
    adapter = gateway.wire_adapter(
        authenticated_client_id=CLIENT_ID, grant_resolver=auth.grant_resolver
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        def call(self, method, payload):
            function = getattr(adapter, method)
            return function() if payload is None else function(payload)

        def close(self):
            pass

    monkeypatch.setattr(one_shot_client, "_Client", Client)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "fake.sock"),
        side="right",
        action="set_gripper_state",
        target_state="closed",
        client_id=CLIENT_ID,
        timeout_sec=1,
        operation_timeout_sec=None,
    )
    try:
        result = one_shot_client.execute_one_shot(args)
        assert result["status"] == "completed"
        assert backend.commands == [1.0]
    finally:
        assert gateway.close().accepted


def test_real_wire_adapter_projects_lower_owned_arm_resource_while_idle() -> None:
    bundle, port, _ = _runtime_fixture(exclusive_owner=True)
    assert "own_domain_emergency_stop_v1" not in (
        bundle.route_manifest.supported_transport_features
    )
    assert "agent_estop" in bundle.route_manifest.explicit_exclusions
    clock = _Clock()
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )

    adapter = gateway.wire_adapter(authenticated_client_id=CLIENT_ID)
    resources = adapter.get_hardware_resource_views()
    assert resources["enabled"] is True
    assert [item["resource_id"] for item in resources["resources"]] == [COMPONENT_ID]
    arm = resources["resources"][0]
    assert arm["resource_kind"] == "robot_arm"
    assert arm["properties"] == {"readable": True, "drivable": True}
    assert {item["skill_id"] for item in arm["read_skill_refs"]} == {
        "state.read_tcp_pose"
    }
    assert "motion.step_tcp_forward" in {
        item["skill_id"] for item in arm["drive_skill_refs"]
    }

    statuses = adapter.get_hardware_status_views()
    assert statuses["enabled"] is True
    assert statuses["station_revision"] >= 0
    assert statuses["resources"] == [
        {
            "schema_version": "dexteleop.hardware_status_view.v1",
            "resource_id": COMPONENT_ID,
            "occupancy": "occupied",
            "health": "ready",
            "station_revision": statuses["station_revision"],
            "reason_code": "lower_control_claim_occupied",
            "active_operation_id": None,
            "active_skill_ref": None,
        }
    ]

    clock.value += 1.0
    stale = adapter.get_hardware_status_views()["resources"][0]
    assert stale["occupancy"] == "occupied"
    assert stale["health"] == "unavailable"
    assert stale["reason_code"] == "lower_control_claim_occupied"


def test_real_spatial_goal_route_is_passive_and_uses_no_motion_grant():
    bundle, port, builder = _runtime_fixture(spatial=True, exclusive_owner=True)
    clock = _Clock()
    owner = _PersistentOwnerSession()
    port.request_domain_hold(reason="fixture startup", requested_at=clock())
    port.apply_owner_requests(session=owner, now=clock())
    port.publish_owner_state(session=owner, captured_at=clock())
    port.publish_measured_pose(pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(), available_at=clock(), source_evidence_refs=("fixture:geometry-only",))
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle, port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,), clock=clock,
    )
    try:
        snapshot = gateway.observation_producer.capture_and_publish(gateway.service).snapshot
        registration = bundle.spatial_goal_registration
        route = registration.entry.route
        profile = route.allowed_parameter_profiles[0]
        request = replace(_read_request(bundle, snapshot), skill_id=route.skill_id,
            skill_definition_hash=route.skill_definition_hash, route_id=route.route_id,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=registration.entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
            parameters={"component_id": COMPONENT_ID, "reference_frame_id": "canonical_station",
                "current_tcp_pose_xyz_quat_xyzw": [0., 0., .3, 0., 0., 0., 1.],
                "anchor_xyz_m": [.2, -.1, .3], "offset_xyz_m": [0., 0., .03],
                "offset_frame": "reference", "orientation_mode": "preserve",
                "tool_axis": None, "axis_reference_xyz": None})
        before = port.snapshot()
        adapter = gateway.wire_adapter(authenticated_client_id=CLIENT_ID, read_only_access=True)
        submission = adapter.submit_read_only_skill_request(request.to_wire())
        assert submission["accepted"], submission
        assert gateway.spatial_goal_executor.wait_all(timeout_sec=1.)
        operation = gateway.service.get_operation(submission["operation"]["operation_id"])
        assert operation.status == "completed", operation
        assert operation.task_outcome == "not_assessed"
        np.testing.assert_allclose(operation.execution_result["pose_targets_by_component"][COMPONENT_ID],
                                   [.2, -.1, .33, 0, 0, 0, 1])
        assert port.snapshot() == before
        assert builder.last_target is None
        assert gateway.passive_read_executors_by_skill_ref[(route.skill_id, route.skill_version)] is gateway.spatial_goal_executor
    finally:
        assert gateway.close().accepted


def test_real_gateway_reads_tcp_pose_without_consuming_motion_grant() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:read-only",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )
    authorization = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        action="forward",
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service
    ).snapshot
    assert snapshot is not None
    assert authorization.grant_resolver(_read_request(bundle, snapshot)) is None
    assert authorization.grants_issued == 0
    raw_revision_before_refresh = port.snapshot().revision
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:read-only-refresh",),
    )
    assert port.snapshot().revision > raw_revision_before_refresh
    assert gateway.observation_producer.current_station_revision == (
        snapshot.station_revision
    )
    submission = gateway.service.submit_request(
        request=_read_request(bundle, snapshot),
        current_station_revision=(
            gateway.observation_producer.current_station_revision
        ),
        now=clock(),
    )
    assert submission.accepted
    assert submission.operation is not None
    launch = gateway.service.launch_operation(submission.operation.operation_id)
    assert launch.accepted
    assert gateway.state_executor.wait_all(timeout_sec=1.0)
    operation = gateway.service.get_operation(submission.operation.operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.execution_result["component_ids"] == (COMPONENT_ID,)
    assert operation.execution_result["pose_by_component"][COMPONENT_ID] == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert authorization.grant_issued is False
    assert port.snapshot().dispatch_count == 0


def test_real_gateway_projects_solver_world_observation_into_station_axes() -> None:
    station_from_execution = np.array(
        [
            [-1.0, 0.0, 0.0, 0.40],
            [0.0, -1.0, 0.0, -0.10],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bundle, port, _ = _runtime_fixture(
        station_from_execution=station_from_execution,
    )
    clock = _Clock()
    execution_pose = np.eye(4, dtype=float)
    execution_pose[:3, 3] = (0.10, 0.20, 0.30)
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: execution_pose},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:nonidentity-frame",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )

    snapshot = gateway.observation_producer.capture_snapshot(
        active_operation_id=None,
    )
    summary = snapshot.measurement_summary
    assert summary["reference_frame_id"] == "canonical_station"
    assert summary["execution_reference_frame_id"] == "dual_arm_solver_world"
    snapshot_pose = xyzquat_to_pose_matrix(
        summary["pose_xyz_quat_xyzw_by_component"][COMPONENT_ID],
        name="test_real_gateway_station_snapshot_pose",
    )
    np.testing.assert_allclose(
        snapshot_pose,
        station_from_execution @ execution_pose,
        rtol=0.0,
        atol=1e-7,
    )

    read = gateway.observation_producer.read_tcp_pose(
        active_operation_id="real-read:station-frame",
        parameters={
            "component_ids": [COMPONENT_ID],
            "reference_frame_id": bundle.attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": (bundle.attachment.max_measurement_age_ms),
        },
    )
    assert read.execution_result["reference_frame_id"] == "canonical_station"
    read_pose = xyzquat_to_pose_matrix(
        read.execution_result["pose_by_component"][COMPONENT_ID],
        name="test_real_gateway_station_read_pose",
    )
    np.testing.assert_allclose(
        read_pose,
        station_from_execution @ execution_pose,
        rtol=0.0,
        atol=1e-7,
    )


def test_republishing_one_measured_cut_uses_freshness_in_snapshot_identity() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    owner_state = _owner_state(bundle, captured_at=clock())
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:stable",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner_state,
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )

    first = gateway.observation_producer.capture_and_publish(gateway.service)
    assert first.accepted and first.snapshot is not None
    clock.value += 0.010
    second = gateway.observation_producer.capture_and_publish(gateway.service)

    assert second.accepted and second.snapshot is not None
    assert second.snapshot.snapshot_id != first.snapshot.snapshot_id
    assert (
        second.snapshot.measurement_summary["measured_state_hash"]
        == first.snapshot.measurement_summary["measured_state_hash"]
    )
    assert second.snapshot.station_revision == first.snapshot.station_revision
    assert (
        second.snapshot.measurement_freshness_ms
        > first.snapshot.measurement_freshness_ms
    )


def test_newer_owner_cut_combines_with_still_fresh_deduplicated_measurement() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:deduplicated",),
    )
    clock.value += 0.010
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

    snapshot = gateway.observation_producer.capture_snapshot(
        active_operation_id=None,
    )

    assert snapshot.captured_at == 10.0
    assert snapshot.available_at == 10.01
    assert snapshot.measurement_freshness_ms == 10


def test_newer_owner_cut_does_not_revive_a_stale_measurement() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:stale-deduplicated",),
    )
    clock.value += (bundle.convergence_profile.max_measurement_age_ms / 1_000.0) + 0.001
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

    with pytest.raises(RuntimeError, match="measured TCP-pose cut is stale"):
        gateway.observation_producer.capture_snapshot(
            active_operation_id=None,
        )


def test_future_owner_cut_remains_rejected() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:future-owner",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(
            bundle,
            captured_at=clock() + 0.001,
        ),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )

    with pytest.raises(RuntimeError, match="inputs are not yet available"):
        gateway.observation_producer.capture_snapshot(
            active_operation_id=None,
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
    assert gateway.service.operation_store.max_admissions_by_route_hash == {}
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
        current_station_revision=(
            gateway.observation_producer.current_station_revision
        ),
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
        current_station_revision=(
            gateway.observation_producer.current_station_revision
        ),
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


def test_session_authorization_issues_a_distinct_bound_grant_per_request() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    owner_state = _owner_state(bundle, captured_at=clock())
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:authorization",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner_state,
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service,
    ).snapshot
    assert snapshot is not None
    first_request = _request(
        bundle,
        snapshot,
        skill_id="motion.step_tcp_cardinal",
        direction="forward",
        client_sequence=1,
        request_suffix="session-1",
    )
    second_request = _request(
        bundle,
        snapshot,
        skill_id="motion.step_tcp_cardinal",
        direction="backward",
        client_sequence=2,
        request_suffix="session-2",
    )
    authorization = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        authorized_actions=("forward", "backward"),
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )

    mismatched_client = replace(first_request, client_id="another-client")
    mismatched_side = replace(
        first_request,
        parameters={**first_request.parameters, "component_id": "left_arm"},
    )
    unlisted_action = replace(
        first_request,
        parameters={**first_request.parameters, "direction": "right"},
    )
    assert authorization.grant_resolver(mismatched_client) is None
    assert authorization.grant_resolver(mismatched_side) is None
    assert authorization.grant_resolver(unlisted_action) is None
    assert authorization.grants_issued == 0

    first_grant = authorization.grant_resolver(first_request)
    second_grant = authorization.grant_resolver(second_request)

    assert first_grant is not None
    assert second_grant is not None
    assert first_grant.grant_id != second_grant.grant_id
    assert first_grant.client_id == second_grant.client_id == CLIENT_ID
    assert first_grant.max_invocations == second_grant.max_invocations == 1
    assert (
        first_grant.authorized_request_hash == first_request.authorization_content_hash
    )
    assert (
        second_grant.authorized_request_hash
        == second_request.authorization_content_hash
    )
    assert first_grant.operator_authorization_receipt_id == (
        second_grant.operator_authorization_receipt_id
    )
    assert authorization.grants_issued == 2
    assert authorization.grant_issued is True


def test_session_authorization_rejects_the_retired_one_shot_confirmation() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
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

    with pytest.raises(ValueError, match="confirmation token"):
        RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            action="forward",
            client_id=CLIENT_ID,
            confirmation_token="AUTHORIZE_ONE_REAL_TCP_PRIMITIVE",
            clock=clock,
        )


def test_historic_one_shot_authorization_import_is_a_session_alias() -> None:
    assert (
        RealAgentPrimitiveOneShotAuthorization is RealAgentPrimitiveSessionAuthorization
    )


def test_session_authorization_accepts_requests_from_listed_actions() -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:action-set",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
    )
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service,
    ).snapshot
    assert snapshot is not None
    request = _request(
        bundle,
        snapshot,
        skill_id=FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1["backward"],
        direction="backward",
    )
    authorization = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        authorized_actions=("forward", "backward"),
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )
    grant = authorization.grant_resolver(request)
    assert grant is not None
    assert authorization.action is None
    assert authorization.authorized_actions == ("backward", "forward")


def test_one_gateway_session_executes_two_motion_requests_and_holds_between() -> None:
    bundle, port, target_builder = _runtime_fixture()
    clock = _Clock()
    owner_session = _PersistentOwnerSession()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:persistent-session-initial",),
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
            source_evidence_refs=(
                f"real-hardware-cut:persistent-session-{publish_sequence}",
            ),
        )

    def observe_owner():
        held = port.snapshot().phase == RealAgentMotionPhase.HELD
        return replace(owner_state, captured_at=clock(),
            lifecycle_state="hold" if held else "running", execution_gate="hold" if held else "ready")

    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=observe_owner,
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
        sleep=owner_sleep,
    )
    authorization = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        authorized_actions=("forward", "backward"),
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )
    adapter = gateway.wire_adapter(
        authenticated_client_id=CLIENT_ID,
        grant_resolver=authorization.grant_resolver,
    )
    operation_ids = []

    for client_sequence, direction in enumerate(("forward", "backward"), start=1):
        observation = gateway.observation_producer.capture_and_publish(gateway.service)
        assert observation.accepted and observation.snapshot is not None
        assert observation.snapshot.lifecycle_state == ("running" if client_sequence == 1 else "hold")
        request = _request(
            bundle,
            observation.snapshot,
            skill_id="motion.step_tcp_cardinal",
            direction=direction,
            client_sequence=client_sequence,
            request_suffix=f"persistent-session-{client_sequence}",
        )

        submission = adapter.submit_skill_request(request.to_wire())

        assert submission["accepted"] is True
        assert submission["operation"] is not None
        operation_id = submission["operation"]["operation_id"]
        operation_ids.append(operation_id)
        assert gateway.wait(timeout_sec=1.0)
        operation = gateway.service.get_operation(operation_id)
        assert operation is not None
        assert operation.status == "completed"
        assert operation.execution_result is not None
        assert port.snapshot().phase == RealAgentMotionPhase.HELD

    assert len(set(operation_ids)) == 2
    assert authorization.grants_issued == 2
    assert len(owner_session.holds) == 2
    assert gateway.close().accepted


def test_real_gateway_applies_only_an_explicit_session_admission_budget() -> None:
    bundle, port, _ = _runtime_fixture()
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=10.0),
        allowed_client_ids=(CLIENT_ID,),
        max_retained_operations=2,
        max_session_admissions_per_route=7,
    )

    assert gateway.service.operation_store.max_retained_operations == 2
    assert set(gateway.service.operation_store.max_admissions_by_route_hash) == {
        entry.route.semantic_content_hash
        for entry in bundle.route_manifest.skill_routes
    }
    assert set(
        gateway.service.operation_store.max_admissions_by_route_hash.values()
    ) == {7}


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
        primitive_reference_frame_id=(bundle.cardinal_profile.reference_frame_id),
        safety_envelope_refs=(bundle.deployment_bindings[0].safety_envelope_refs),
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
        *FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
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
        skill_id=FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1["forward"],
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


def test_real_rgb_and_depth_routes_run_passively_and_commit_artifacts(tmp_path) -> None:
    base_bundle, port, _target_builder = _runtime_fixture()
    clock = _Clock()

    class _RgbSource:
        config = SimpleNamespace(
            width=4,
            height=3,
            depth_width=4,
            depth_height=3,
            align_depth_to_color=True,
            enable_depth=True,
            color_space="rgb",
            serial="rgb-test-serial",
        )

        def get_latest_frame(self):
            return VideoFrame(
                image=np.full((3, 4, 3), 127, dtype=np.uint8),
                source_timestamp=clock(),
                receive_timestamp=clock(),
                available_timestamp=clock(),
                metadata={
                    "camera_key": "front",
                    "serial": self.config.serial,
                    "source_generation": 1,
                },
                depth_image=np.zeros((3, 4), dtype=np.uint16),
                depth_metadata={
                    "aligned_to": "color",
                    "depth_scale_m_per_unit": 0.001,
                },
            )

        def device_info(self):
            return {
                "running": True,
                "lifecycle_state": "running",
                "serial_number": self.config.serial,
                "generation": 1,
                "worker_alive": True,
                "intrinsics": {
                    "width": 4,
                    "height": 3,
                    "fx": 4.0,
                    "fy": 3.0,
                    "ppx": 2.0,
                    "ppy": 1.5,
                },
            }

    source_binding = RealPerceptionSourceBinding(
        camera_sources={"front": _RgbSource()},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="real-rgb-source-binding:test",
    )
    catalog_provider = RealSenseAgentPerceptionProvider(
        source_binding=source_binding,
        current_station_revision=lambda: 0,
        clock=clock,
    )
    perception = build_real_perception_capture_registration(
        rgb_source_catalog=catalog_provider.rgb_source_catalog(),
        depth_source_catalog=catalog_provider.depth_source_catalog(),
        runtime_session_id=base_bundle.attachment.runtime_session_id,
        control_domain_id=base_bundle.attachment.control_domain_id,
        target_binding_id=base_bundle.attachment.target_binding_id,
        station_manifest_hash=base_bundle.attachment.station_manifest_hash,
        execution_manifest_hash=base_bundle.attachment.execution_manifest_hash,
        source_binding_hash=source_binding.semantic_content_hash,
    )
    segmentation = build_segmentation_registration(
        target_kind="real",
        current_mode="real_active",
        source_id_by_role={
            str(item["role"]): str(item["source_id"])
            for item in catalog_provider.rgb_source_catalog()
        },
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-localizer:test",
        runtime_session_id=base_bundle.attachment.runtime_session_id,
        control_domain_id=base_bundle.attachment.control_domain_id,
        target_binding_id=base_bundle.attachment.target_binding_id,
        station_manifest_hash=base_bundle.attachment.station_manifest_hash,
        execution_manifest_hash=base_bundle.attachment.execution_manifest_hash,
    )
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-envelope:perception-test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:perception-test",
        constraint_set_hash="real-constraint-hash:perception-test",
        allowed_route_ids=tuple(
            entry.route.route_id for entry in base_bundle.route_manifest.skill_routes
        ),
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=base_bundle.attachment,
        port_config=port.config,
        frame_graph=base_bundle.frame_graph,
        primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=(envelope,),
        perception_capture=perception,
        segmentation_registration=segmentation,
    )
    provider = RealSenseAgentPerceptionProvider(
        source_binding=source_binding,
        current_station_revision=lambda: 0,
        clock=clock,
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )

    class _Segmenter:
        semantic_content_hash = "sam2-localizer:test"
        started = True
        model_spec = SimpleNamespace(
            model_profile_id="sam2:test",
            semantic_content_hash="sam2-model:test",
        )

        @staticmethod
        def segment(image, *, language_prompt, pixel_uv, min_score):
            assert image.size == (4, 3)
            assert language_prompt is None
            assert pixel_uv == (1, 1)
            mask = np.zeros((3, 4), dtype=bool)
            mask[1, 1:3] = True
            return GroundedObjectSegment(
                mask=mask,
                score=max(0.9, min_score),
                label=None,
                box_xyxy=(1, 1, 3, 2),
            )

        encode_segment_artifacts = staticmethod(
            encode_grounded_object_segment_artifacts
        )

    segmenter = _Segmenter()
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:perception",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        perception_provider=provider,
        segmentation_localizer=segmenter,
        artifact_store=store,
        clock=clock,
    )
    adapter = gateway.wire_adapter(authenticated_client_id=CLIENT_ID)
    resources = adapter.get_hardware_resource_views()["resources"]
    assert {item["resource_id"] for item in resources} == {
        "perception.cameras",
        "perception.object_localization_model",
        COMPONENT_ID,
    }
    perception_resource = next(
        item for item in resources if item["resource_id"] == "perception.cameras"
    )
    assert perception_resource["resource_kind"] == "camera_bundle"
    assert perception_resource["properties"] == {
        "readable": True,
        "drivable": False,
    }
    assert {item["skill_id"] for item in perception_resource["read_skill_refs"]} == {
        "perception.capture_depth",
        "perception.capture_rgb",
        "perception.segment",
    }
    statuses = {
        item["resource_id"]: item
        for item in adapter.get_hardware_status_views()["resources"]
    }
    assert statuses["perception.cameras"]["occupancy"] == "occupied"
    assert statuses["perception.cameras"]["health"] == "ready"
    assert statuses["perception.cameras"]["reason_code"] == (
        "lower_control_claim_occupied"
    )
    assert statuses["perception.cameras"]["active_operation_id"] is None
    assert statuses[COMPONENT_ID]["occupancy"] == "occupied"
    assert statuses[COMPONENT_ID]["health"] == "unavailable"
    observation = gateway.observation_producer.capture_and_publish(gateway.service)
    assert observation.accepted and observation.snapshot is not None
    entry = bundle.route_manifest.route("perception.capture_rgb", "1")
    route = entry.route
    parameter_profile = route.allowed_parameter_profiles[0]
    request = AgentSkillRequest(
        client_id=CLIENT_ID,
        request_id="request:real-rgb",
        idempotency_key="idempotency:real-rgb",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=observation.snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=parameter_profile.parameter_profile_id,
        parameter_profile_hash=parameter_profile.parameter_profile_hash,
        parameters={
            "selected_roles": ["main"],
            "width": 2,
            "height": 2,
            "max_total_bytes": 100_000,
        },
        observation_snapshot_id=observation.snapshot.snapshot_id,
        requested_mode="real_active",
        requested_timeout_ms=3_000,
    )
    try:
        submitted = gateway.service.submit_request(
            request=request,
            current_station_revision=(
                gateway.observation_producer.current_station_revision
            ),
            now=clock(),
        )
        assert submitted.accepted and submitted.operation is not None
        launched = gateway.service.launch_operation(submitted.operation.operation_id)
        assert launched.accepted
        assert gateway.rgb_executor is not None
        assert gateway.rgb_executor.wait_all(timeout_sec=1.0)
        operation = gateway.service.get_operation(submitted.operation.operation_id)
        assert operation is not None and operation.status == "completed"
        assert len(operation.output_artifact_bindings) == 1
        artifact = operation.output_artifact_bindings[0].artifact
        assert (
            store.managed_read_path(artifact.artifact_id).read_bytes()[:2]
            == b"\xff\xd8"
        )

        depth_entry = bundle.route_manifest.route("perception.capture_depth", "1")
        depth_route = depth_entry.route
        depth_profile = depth_route.allowed_parameter_profiles[0]
        depth_request = AgentSkillRequest(
            client_id=CLIENT_ID,
            request_id="request:real-depth",
            idempotency_key="idempotency:real-depth",
            client_sequence=2,
            runtime_session_id=bundle.route_manifest.runtime_session_id,
            skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
            expected_station_revision=observation.snapshot.station_revision,
            skill_id=depth_route.skill_id,
            skill_version=depth_route.skill_version,
            skill_definition_hash=depth_route.skill_definition_hash,
            route_id=depth_route.route_id,
            route_version=depth_route.route_version,
            route_hash=depth_route.semantic_content_hash,
            deployment_binding_hash=(
                depth_entry.deployment_binding.deployment_binding_hash
            ),
            parameter_profile_id=depth_profile.parameter_profile_id,
            parameter_profile_hash=depth_profile.parameter_profile_hash,
            parameters={"role": "main"},
            observation_snapshot_id=observation.snapshot.snapshot_id,
            requested_mode="real_active",
            requested_timeout_ms=3_000,
        )
        depth_submitted = gateway.service.submit_request(
            request=depth_request,
            current_station_revision=(
                gateway.observation_producer.current_station_revision
            ),
            now=clock(),
        )
        assert depth_submitted.accepted and depth_submitted.operation is not None
        depth_launched = gateway.service.launch_operation(
            depth_submitted.operation.operation_id
        )
        assert depth_launched.accepted
        assert gateway.depth_executor is not None
        assert gateway.depth_executor.wait_all(timeout_sec=1.0)
        depth_operation = gateway.service.get_operation(
            depth_submitted.operation.operation_id
        )
        assert depth_operation is not None and depth_operation.status == "completed"
        depth_artifact = depth_operation.output_artifact_bindings[0].artifact
        depth_array = np.load(
            store.managed_read_path(depth_artifact.artifact_id),
            allow_pickle=False,
        )
        assert depth_array.dtype == np.float32
        assert depth_array.shape == (3, 4)

        segment_entry = bundle.route_manifest.route("perception.segment", "1")
        segment_route = segment_entry.route
        segment_profile = segment_route.allowed_parameter_profiles[0]
        segment_request = AgentSkillRequest(
            client_id=CLIENT_ID,
            request_id="request:real-segment",
            idempotency_key="idempotency:real-segment",
            client_sequence=3,
            runtime_session_id=bundle.route_manifest.runtime_session_id,
            skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
            expected_station_revision=observation.snapshot.station_revision,
            skill_id=segment_route.skill_id,
            skill_version=segment_route.skill_version,
            skill_definition_hash=segment_route.skill_definition_hash,
            route_id=segment_route.route_id,
            route_version=segment_route.route_version,
            route_hash=segment_route.semantic_content_hash,
            deployment_binding_hash=(
                segment_entry.deployment_binding.deployment_binding_hash
            ),
            parameter_profile_id=segment_profile.parameter_profile_id,
            parameter_profile_hash=segment_profile.parameter_profile_hash,
            parameters={
                "role": "main",
                "language_prompt": None,
                "target_pixel_uv": [1, 1],
                "min_score": 0.5,
            },
            observation_snapshot_id=observation.snapshot.snapshot_id,
            requested_mode="real_active",
            requested_timeout_ms=60_000,
        )
        segment_submitted = gateway.service.submit_request(
            request=segment_request,
            current_station_revision=(
                gateway.observation_producer.current_station_revision
            ),
            now=clock(),
        )
        assert segment_submitted.accepted and segment_submitted.operation is not None
        assert gateway.service.launch_operation(
            segment_submitted.operation.operation_id
        ).accepted
        assert gateway.segmentation_executor is not None
        assert gateway.segmentation_executor.wait_all(timeout_sec=1.0)
        segment_operation = gateway.service.get_operation(
            segment_submitted.operation.operation_id
        )
        assert segment_operation is not None
        assert segment_operation.status == "completed"
        assert segment_operation.execution_result["mask_pixel_count"] == 2
        assert len(segment_operation.output_artifact_bindings) == 2
        assert port.snapshot().dispatch_count == 0
    finally:
        store.close()














def test_motion_only_gateway_reads_confirmed_hold_without_resuming_arm():
    bundle, port, builder = _runtime_fixture(exclusive_owner=True)
    clock, lower_owner = _Clock(), _PersistentOwnerSession()
    port.request_domain_hold(reason="planning rejected", requested_at=clock())
    port.apply_owner_requests(session=lower_owner, now=clock())
    port.publish_measured_pose(pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(), available_at=clock(), source_evidence_refs=("fresh:held",))
    owner = replace(_owner_state(bundle, captured_at=clock()), lifecycle_state="hold", execution_gate="hold")
    gateway = assemble_real_agent_primitive_gateway_session(contract_bundle=bundle, port=port,
        owner_state_provider=lambda: owner, allowed_client_ids=(CLIENT_ID,), clock=clock)
    before = port.snapshot()
    try:
        assert bundle.gripper_contract is None
        snapshot = gateway.observation_producer.capture_and_publish(gateway.service).snapshot
        assert snapshot.lifecycle_state == snapshot.execution_gate == "hold"
        assert port.snapshot() == before
        assert before.phase == RealAgentMotionPhase.HELD and before.hold_succeeded is True
        assert builder.last_target is None
        clock.value += .3
        with pytest.raises(RuntimeError, match="stale"):
            gateway.observation_producer.capture_snapshot(active_operation_id=None)
        assert port.snapshot() == before
    finally:
        gateway.close()



@pytest.mark.parametrize("lifecycle,gate", [("fault", "fault"), ("closed", "hold"), ("hold", "fault")])
def test_motion_only_confirmed_hold_does_not_override_owner_fault_or_closed(lifecycle, gate):
    bundle, port, _ = _runtime_fixture(exclusive_owner=True)
    clock = _Clock()
    port.request_domain_hold(reason="fixture", requested_at=clock())
    port.apply_owner_requests(session=_PersistentOwnerSession(), now=clock())
    owner = replace(_owner_state(bundle, captured_at=clock()), lifecycle_state=lifecycle, execution_gate=gate)
    gateway = assemble_real_agent_primitive_gateway_session(contract_bundle=bundle, port=port,
        owner_state_provider=lambda: owner, allowed_client_ids=(CLIENT_ID,), clock=clock)
    try:
        with pytest.raises(RuntimeError, match="Real motion owner is not"):
            gateway.observation_producer.capture_snapshot(active_operation_id=None)
    finally:
        gateway.close()


@pytest.fixture
def _bilateral_gripper_fixture(monkeypatch):
    return lambda selected_side="right": _build_bilateral_gripper_fixture(
        monkeypatch=monkeypatch, selected_side=selected_side)


def _build_bilateral_gripper_fixture(*, monkeypatch, selected_side="right"):
    base, port, _ = _runtime_fixture(exclusive_owner=True)
    clock = _Clock()
    # Only the adapter time port is replaced. Service/thread time remains real;
    # acquisition, wall age and HOLD deadlines share this fixture's logical cut.
    monkeypatch.setattr(gripper_module, "time", SimpleNamespace(
        monotonic=clock, time=clock,
        sleep=lambda duration: setattr(clock, "value", clock.value + duration)))
    owner = _PersistentOwnerSession()
    port.request_domain_hold(reason="fixture startup", requested_at=clock())
    port.apply_owner_requests(session=owner, now=clock())
    port.publish_owner_state(session=owner, captured_at=clock())
    port.publish_measured_pose(
        pose_world_by_component={COMPONENT_ID: np.eye(4)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("fixture:fresh-qpos",),
    )
    backend = _ComposedGripperBackend(clock=clock)
    hand = RealDmtacGripperControlPort(backend=backend, clock=clock)
    hands = {
        "right_gripper": hand,
        "left_gripper": RealDmtacGripperControlPort(
            backend=_ComposedGripperBackend("left", clock=clock), clock=clock,
            component_id="left_gripper"),
    }
    contract = build_real_dmtac_grippers_skill_contract_bundle(
        runtime_session_id=base.attachment.runtime_session_id,
        control_domain_id=base.attachment.control_domain_id,
        target_binding_id=base.attachment.target_binding_id,
        station_manifest_hash=base.attachment.station_manifest_hash,
        execution_manifest_hash=base.attachment.execution_manifest_hash,
        ports=hands,
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=base.attachment,
        port_config=port.config,
        frame_graph=base.frame_graph,
        primitive_reference_frame_id=base.cardinal_profile.reference_frame_id,
        safety_envelope_refs=base.deployment_bindings[0].safety_envelope_refs,
        gripper_contract=contract,
    )

    def sleep(duration):
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())
        port.publish_owner_state(session=owner, captured_at=clock())
        port.publish_measured_pose(
            pose_world_by_component={COMPONENT_ID: np.eye(4)},
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=("fixture:fresh-qpos",),
        )

    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        gripper_ports=hands,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,),
        clock=clock,
        sleep=sleep,
    )
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service
    ).snapshot
    assert snapshot is not None
    entry = bundle.route_manifest.route("manipulation.set_gripper_state", "2")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    request = replace(
        _read_request(bundle, snapshot),
        request_id="request:gripper",
        idempotency_key="idempotency:gripper",
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={"component_id": selected_side+"_gripper", "target_state": "closed"},
        requested_timeout_ms=5000,
    )
    return gateway, hands[selected_side+"_gripper"].backend, request, snapshot, clock



def test_v2_composed_gripper_requires_separate_authorization_and_executes_under_hold(_bilateral_gripper_fixture):
    gateway, backend, request, snapshot, clock = _bilateral_gripper_fixture()
    try:
        arm_auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("forward",),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        assert arm_auth.grant_resolver(request) is None
        assert not gateway.service.submit_request(
            request=request, current_station_revision=snapshot.station_revision
        ).accepted
        assert not backend.commands
        auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("set_gripper_state",),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        grant = auth.grant_resolver(request)
        request = replace(request, skill_execution_grant_id=grant.grant_id)
        admitted = gateway.service.submit_request(
            request=request,
            grant=grant,
            current_station_revision=snapshot.station_revision,
        )
        assert admitted.accepted
        assert gateway.service.launch_operation(
            admitted.operation.operation_id
        ).accepted
        assert gateway.wait(timeout_sec=3)
        result = gateway.service.get_operation(admitted.operation.operation_id)
        assert result.execution_outcome == "completed"
        assert result.execution_result["terminal_hold_confirmed"] is True
        assert backend.commands == [1.0]
        assert gateway.port.snapshot().phase == RealAgentMotionPhase.HELD
    finally:
        assert gateway.close().accepted



def test_v2_composed_gripper_reserves_the_same_operation_slot_as_arm_motion(_bilateral_gripper_fixture):
    gateway, backend, request, snapshot, clock = _bilateral_gripper_fixture()
    try:
        auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("forward", "set_gripper_state"),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        grant = auth.grant_resolver(request)
        request = replace(request, skill_execution_grant_id=grant.grant_id)
        assert gateway.service.submit_request(
            request=request,
            grant=grant,
            current_station_revision=snapshot.station_revision,
        ).accepted
        arm = _request(
            gateway.contract_bundle,
            snapshot,
            skill_id="motion.step_tcp_forward",
            direction="forward",
            client_sequence=2,
        )
        grant = auth.grant_resolver(arm)
        arm = replace(arm, skill_execution_grant_id=grant.grant_id)
        rejected = gateway.service.submit_request(
            request=arm, grant=grant, current_station_revision=snapshot.station_revision
        )
        assert not rejected.accepted
        assert "active" in rejected.reason_code or "busy" in rejected.reason_code
        assert not backend.commands
    finally:
        assert gateway.close().accepted



def test_v2_composed_gateway_does_not_confirm_shutdown_if_gripper_stop_fails(_bilateral_gripper_fixture):
    gateway, backend, _, _, _ = _bilateral_gripper_fixture()
    backend.stop_ok = False
    assert not gateway.close().accepted
    backend.stop_ok = True
    assert gateway.close().accepted



def test_v2_composed_gripper_accepts_fresh_confirmed_hold_observation_without_resuming_arm(_bilateral_gripper_fixture):
    gateway, _, _, _, clock = _bilateral_gripper_fixture()
    owner = replace(
        _owner_state(gateway.contract_bundle, captured_at=clock()),
        lifecycle_state="hold",
        execution_gate="hold",
    )
    gateway.observation_producer.owner_state_provider = lambda: owner
    snapshot = gateway.observation_producer.capture_and_publish(
        gateway.service
    ).snapshot
    assert snapshot.lifecycle_state == "hold"
    assert snapshot.execution_gate == "hold"
    assert gateway.port.snapshot().phase == RealAgentMotionPhase.HELD
    assert gateway.close().accepted



def test_v2_gripper_command_attempt_invalidates_scene_revision_but_passive_reads_do_not(_bilateral_gripper_fixture):
    gateway, _, _, snapshot, _ = _bilateral_gripper_fixture()
    try:
        gateway.gripper_ports["right_gripper"].observe_state()
        assert (
            gateway.observation_producer.current_station_revision
            == snapshot.station_revision
        )
        gateway.gripper_ports["right_gripper"].command_close(1.0)
        assert (
            gateway.observation_producer.current_station_revision
            > snapshot.station_revision
        )
    finally:
        assert gateway.close().accepted



def test_v2_composed_gripper_rejects_motion_after_arm_owner_becomes_stale(_bilateral_gripper_fixture):
    gateway, backend, _, _, clock = _bilateral_gripper_fixture()
    clock.value += 0.3
    with pytest.raises(RuntimeError, match="fresh, healthy"):
        gateway.gripper_ports["right_gripper"].command_close(1.0)
    assert not backend.commands
    assert gateway.close().accepted



def test_v2_composed_gripper_rejects_fresh_owner_with_stale_arm_measurement(_bilateral_gripper_fixture):
    gateway, backend, _, _, clock = _bilateral_gripper_fixture()
    clock.value += 0.3
    gateway.port.publish_owner_state(
        session=_PersistentOwnerSession(), captured_at=clock()
    )
    with pytest.raises(RuntimeError, match="fresh, healthy"):
        gateway.gripper_ports["right_gripper"].command_close(1.0)
    assert not backend.commands
    assert gateway.close().accepted



def test_v2_gripper_side_authorization_is_independent_and_grants_cannot_switch_hands(_bilateral_gripper_fixture):
    gateway, _, left, snapshot, clock = _bilateral_gripper_fixture("left")
    try:
        auth = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("set_gripper_state",),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        assert auth.grant_resolver(left) is None
        both = RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            authorized_actions=("set_gripper_state",),
            authorized_gripper_sides=("left", "right"),
            client_id=CLIENT_ID,
            confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
            clock=clock,
        )
        grant = both.grant_resolver(left)
        wrong = replace(
            left,
            parameters={"component_id": "right_gripper", "target_state": "closed"},
            skill_execution_grant_id=grant.grant_id,
        )
        assert not gateway.service.submit_request(
            request=wrong,
            grant=grant,
            current_station_revision=snapshot.station_revision,
        ).accepted
        assert all(not port.backend.commands for port in gateway.gripper_ports.values())
    finally:
        assert gateway.close().accepted



@pytest.mark.parametrize("side", ["left", "right"])
def test_v2_composed_gripper_client_uses_real_wire_admission_and_hand_constraint_binding(
    monkeypatch,
    _bilateral_gripper_fixture,
    tmp_path,
    side,
):
    from argparse import Namespace

    from agent_gateway.profiles.real import one_shot_client

    gateway, backend, _, _, clock = _bilateral_gripper_fixture(side)
    auth = RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        authorized_actions=("set_gripper_state",),
        authorized_gripper_sides=(side,),
        client_id=CLIENT_ID,
        confirmation_token=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        clock=clock,
    )
    adapter = gateway.wire_adapter(
        authenticated_client_id=CLIENT_ID, grant_resolver=auth.grant_resolver
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        def call(self, method, payload):
            function = getattr(adapter, method)
            return function() if payload is None else function(payload)

        def close(self):
            pass

    monkeypatch.setattr(one_shot_client, "_Client", Client)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "gateway.sock"),
        side=side,
        action="set_gripper_state",
        target_state="closed",
        client_id=CLIENT_ID,
        timeout_sec=1,
        operation_timeout_sec=None,
    )
    try:
        result = one_shot_client.execute_one_shot(args)
        assert result["status"] == "completed"
        assert backend.commands == [1.0]
    finally:
        assert gateway.close().accepted




@pytest.mark.parametrize("proof", [None, False])
def test_motion_only_hold_observation_requires_lower_stop_proof(monkeypatch, proof):
    bundle, port, _ = _runtime_fixture(exclusive_owner=True)
    clock = _Clock()
    port.request_domain_hold(reason="fixture", requested_at=clock())
    port.apply_owner_requests(session=_PersistentOwnerSession(), now=clock())
    held = replace(port.snapshot(), hold_succeeded=proof)
    monkeypatch.setattr(port, "snapshot", lambda: held)
    owner = replace(_owner_state(bundle, captured_at=clock()), lifecycle_state="hold", execution_gate="hold")
    gateway = assemble_real_agent_primitive_gateway_session(contract_bundle=bundle, port=port,
        owner_state_provider=lambda: owner, allowed_client_ids=(CLIENT_ID,), clock=clock)
    try:
        with pytest.raises(RuntimeError, match="hold_succeeded"):
            gateway.observation_producer.capture_snapshot(active_operation_id=None)
    finally:
        gateway.close()

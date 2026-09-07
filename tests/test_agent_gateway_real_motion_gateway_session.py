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
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    RealAgentPrimitiveSessionAuthorization,
)
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
from teleop_runtime.video_frame import VideoFrame

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
        fixed_direction = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1.get(
            skill_id
        )
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
    *tuple(
        (skill_id, None)
        for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    ),
)


def test_real_wire_adapter_projects_lower_owned_arm_resource_while_idle() -> None:
    bundle, port, _ = _runtime_fixture(exclusive_owner=True)
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
    assert [item["resource_id"] for item in resources["resources"]] == [
        COMPONENT_ID
    ]
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
            "max_measurement_age_ms": (
                bundle.attachment.max_measurement_age_ms
            ),
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
        first_grant.authorized_request_hash
        == first_request.authorization_content_hash
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
        RealAgentPrimitiveOneShotAuthorization
        is RealAgentPrimitiveSessionAuthorization
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

    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner_state,
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
        observation = gateway.observation_producer.capture_and_publish(
            gateway.service
        )
        assert observation.accepted and observation.snapshot is not None
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
        artifact_store=store,
        clock=clock,
    )
    adapter = gateway.wire_adapter(authenticated_client_id=CLIENT_ID)
    resources = adapter.get_hardware_resource_views()["resources"]
    assert [item["resource_id"] for item in resources] == [
        "perception.cameras",
        COMPONENT_ID,
    ]
    perception_resource = resources[0]
    assert perception_resource["resource_kind"] == "camera_bundle"
    assert perception_resource["properties"] == {
        "readable": True,
        "drivable": False,
    }
    assert {item["skill_id"] for item in perception_resource["read_skill_refs"]} == {
        "perception.capture_depth",
        "perception.capture_rgb",
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
        assert store.managed_read_path(artifact.artifact_id).read_bytes()[:2] == b"\xff\xd8"

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
        assert port.snapshot().dispatch_count == 0
    finally:
        store.close()

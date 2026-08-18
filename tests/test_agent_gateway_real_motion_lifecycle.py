from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    build_real_agent_primitive_contract_bundle,
)
from agent_gateway.profiles.real.motion_primitive_skill import (
    RealAgentPrimitiveSkillLifecycle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
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
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)


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
        self.holds = []

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
    graph = CoordinateFrameGraph(
        graph_id="real-agent-lifecycle-test-graph",
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
        runtime_session_id="real-session:lifecycle-test",
        control_domain_id="real-domain:lifecycle-test",
        embodiment_instance_id="dual-rm65:lifecycle-test",
        target_binding_id="real-target:lifecycle-test",
        component_ids=("right_arm",),
        station_manifest_hash="station-manifest:lifecycle-test",
        execution_manifest_hash="execution-manifest:lifecycle-test",
        backend_inventory_hash="backend-inventory:lifecycle-test",
        source_binding_id="source-binding:lifecycle-test",
        source_binding_hash="source-binding-hash:lifecycle-test",
        observation_binding_id="observation-binding:lifecycle-test",
        observation_binding_hash="observation-binding-hash:lifecycle-test",
        observation_plan_hash="observation-plan-hash:lifecycle-test",
        observation_capability_hash="observation-capability-hash:lifecycle-test",
        pose_schema_hash="pose-schema-hash:lifecycle-test",
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=graph.semantic_content_hash,
        tool_config_hash="tool-config-hash:lifecycle-test",
        calibration_hash="calibration-hash:lifecycle-test",
        clock_domain_id="real-owner-wall-clock",
        producer_epoch=1,
    )
    port_config = RealAgentMotionPortConfig(
        active_sides=("right",),
        component_id_by_side={"right": "right_arm"},
        execution_frame_id=execution_frame.frame_id,
        execution_frame_spec_hash=execution_frame.semantic_content_hash,
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
        object_id="real-envelope:lifecycle-test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:lifecycle-test",
        constraint_set_hash="real-constraint-hash:lifecycle-test",
        allowed_route_ids=route_ids,
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=attachment,
        port_config=port_config,
        frame_graph=graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
    )
    builder = _TargetBuilder()
    port = RealAgentMotionPort(config=port_config, target_builder=builder)
    return bundle, port, builder


def _request_and_operation(bundle, *, clock: _Clock):
    entry = bundle.route_manifest.route("motion.step_tcp_cardinal", "1")
    route = entry.route
    profile = bundle.cardinal_profile
    request = AgentSkillRequest(
        client_id="rpent-test-client",
        request_id="real-request-1",
        idempotency_key="real-idempotency-1",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=0,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        parameters={
            "component_id": profile.component_id,
            "direction": "forward",
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_catalog_hash": profile.primitive_catalog_hash,
        },
        observation_snapshot_id="real-observation-1",
        requested_mode="real_active",
        requested_timeout_ms=5_000,
        skill_execution_grant_id="agent-skill-grant:real-grant-hash-1",
    )
    operation = AgentSkillOperation(
        operation_id="real-operation-1",
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="admitted",
        reason_code="admitted",
        station_revision_at_admission=0,
        current_station_revision=0,
        started_at=clock(),
        updated_at=clock(),
        deadline_at=clock() + 5.0,
    )
    return request, operation, route, bundle.binding_for_skill(route.skill_id)


def test_real_lifecycle_resolves_dispatches_converges_and_confirms_hold() -> None:
    bundle, port, builder = _runtime_fixture()
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _request_and_operation(bundle, clock=clock)
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:0",),
    )

    publish_sequence = 0

    def owner_sleep(duration: float) -> None:
        nonlocal publish_sequence
        clock.value += duration
        snapshot = port.apply_owner_requests(session=owner, now=clock())
        if snapshot.phase != RealAgentMotionPhase.ACTIVE:
            return
        command = port.command_for_observation(
            TeleopObservation(timestamp=clock()),
            clock(),
        )
        assert isinstance(command, TeleopCommand)
        assert port.admit_command(
            command,
            TeleopObservation(timestamp=clock()),
            clock(),
        )
        port.observe_session_step(command=command, command_dispatched=True)
        assert builder.last_target is not None
        publish_sequence += 1
        port.publish_measured_pose(
            pose_world_by_component={
                "right_arm": builder.last_target.target_pose_by_side["right"],
            },
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=(f"real-hardware-cut:{publish_sequence}",),
        )

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.cardinal_profile,
        request=request,
        operation=operation,
        clock=clock,
        sleep=owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activation.succeeded is True
    assert port.snapshot().phase == RealAgentMotionPhase.ACTIVE

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert completion.succeeded is True
    assert completion.terminal_hold_confirmed is True
    assert completion.execution_result is not None
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert port.snapshot().dispatch_count >= 3
    assert len(owner.holds) == 1


def test_real_lifecycle_confirms_hold_after_operation_deadline() -> None:
    """HOLD proof gets a lower-control grace window after motion expiry."""

    bundle, port, _builder = _runtime_fixture()
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _request_and_operation(bundle, clock=clock)
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:deadline-hold",),
    )

    def lower_cycle_sleep(duration: float) -> None:
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.cardinal_profile,
        request=request,
        operation=operation,
        clock=clock,
        sleep=lower_cycle_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activation.succeeded is True
    assert port.snapshot().phase == RealAgentMotionPhase.ACTIVE

    # The operation's motion budget is already exhausted when HOLD is asked
    # for.  The lower control loop still needs one cycle to apply the request.
    clock.value = operation.deadline_at
    hold = lifecycle.request_hold(
        operation=operation,
        reason="real_primitive_deadline_exceeded",
    )

    assert hold.succeeded is True
    assert hold.lower_hold_confirmed is True
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert len(owner.holds) == 1

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.motion_gateway_session import (
    assemble_real_agent_primitive_gateway_session,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    RealAgentPrimitiveConvergenceProfile,
    build_real_agent_primitive_contract_bundle,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.motion_primitive_skill import (
    RealAgentPrimitiveSkillLifecycle,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealControlOwnerState,
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
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
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_MOTION_COMMAND_METADATA_KEY,
    RealAgentJointTrajectory,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _TargetBuilder:
    def __init__(self, *, include_inactive_command: bool = False) -> None:
        self.last_target = None
        self.include_inactive_command = include_inactive_command

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
        if self.include_inactive_command:
            command.add_command(
                EffectorCommand(
                    effector_name="left_arm",
                    side="left",
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


def _runtime_fixture(
    *,
    station_from_execution_x: float = 0.0,
    include_inactive_command: bool = False,
    include_grasp_approach: bool = False,
    include_safe_pose: bool = False,
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
):
    execution_frame = CoordinateFrameSpec(frame_id="dual_arm_solver_world")
    station_frame = CoordinateFrameSpec(
        frame_id="canonical_station",
        semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    station_from_execution = np.eye(4, dtype=float)
    station_from_execution[0, 3] = station_from_execution_x
    graph = CoordinateFrameGraph(
        graph_id="real-agent-lifecycle-test-graph",
        frames=(execution_frame, station_frame),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="solver-world-to-station",
                source_frame_id=execution_frame.frame_id,
                target_frame_id=station_frame.frame_id,
                matrix_target_source=tuple(
                    tuple(row) for row in station_from_execution
                ),
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
    envelope = SkillDeploymentSafetyEnvelopeRef(
        object_id="real-envelope:lifecycle-test",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash="real-envelope-hash:lifecycle-test",
        constraint_set_hash="real-constraint-hash:lifecycle-test",
        allowed_route_ids=real_agent_motion_route_ids_v1(
            include_safe_pose=include_safe_pose,
            include_grasp_approach=include_grasp_approach,
        ),
    )
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=attachment,
        port_config=port_config,
        frame_graph=graph,
        primitive_reference_frame_id=station_frame.frame_id,
        safety_envelope_refs=(envelope,),
        convergence_profile=convergence_profile,
    )
    builder = _TargetBuilder(include_inactive_command=include_inactive_command)
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


def _pose_request_and_operation(
    bundle,
    *,
    clock: _Clock,
    speed_scale: float,
    skill_id: str = "motion.move_tcp_to_pose",
):
    entry = bundle.route_manifest.route(skill_id, "1")
    route = entry.route
    profile = bundle.pose_target_profile
    request = AgentSkillRequest(
        client_id="rpent-test-client",
        request_id="real-pose-request-1",
        idempotency_key="real-pose-idempotency-1",
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
            "pose_targets_by_component": {
                profile.component_id: [0.4, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            },
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "speed_scale": speed_scale,
        },
        observation_snapshot_id="real-pose-observation-1",
        requested_mode="real_active",
        requested_timeout_ms=10_000,
        skill_execution_grant_id="agent-skill-grant:real-pose-grant-hash-1",
    )
    operation = AgentSkillOperation(
        operation_id="real-pose-operation-1",
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="admitted",
        reason_code="admitted",
        station_revision_at_admission=0,
        current_station_revision=0,
        started_at=clock(),
        updated_at=clock(),
        deadline_at=clock() + 10.0,
    )
    return request, operation, route, bundle.binding_for_skill(route.skill_id)


def _stationary_grasp_trajectory(
    qpos_by_side: dict[str, np.ndarray],
) -> RealAgentJointTrajectory:
    return RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.5], dtype=float),
        qpos_by_side={
            side: np.stack([qpos_by_side[side], qpos_by_side[side]])
            for side in ("left", "right")
        },
        planner_evidence_refs=("rm65-grasp-path:lifecycle-test",),
    )


def _publish_qpos(
    port: RealAgentMotionPort,
    *,
    qpos_by_side: dict[str, np.ndarray],
    measured_at: float,
    available_at: float,
) -> None:
    port.publish_measured_qpos_pair(
        qpos_by_side=qpos_by_side,
        measured_at=measured_at,
        available_at=available_at,
        source_evidence_refs=("hardware-qpos-cut:lifecycle-test",),
    )


@pytest.mark.parametrize("skill_id, planner_key", [
    ("motion.approach_grasp_pose", "grasp_trajectory_planner"),
    ("motion.move_tcp_to_pose_safe", "grasp_trajectory_planner"),
    ("motion.move_tcp_to_pose", "tcp_trajectory_planner"),
])
def test_grasp_approach_activates_only_the_preplanned_joint_trajectory(skill_id, planner_key) -> None:
    bundle, port, builder = _runtime_fixture(include_grasp_approach=True, include_safe_pose=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id=skill_id,
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )
    trajectory = _stationary_grasp_trajectory(qpos_by_side)
    planner_calls = 0

    def planner(**_kwargs) -> RealAgentJointTrajectory:
        nonlocal planner_calls
        planner_calls += 1
        assert _kwargs.get("require_grasp_approach", True) == (skill_id != "motion.move_tcp_to_pose_safe")
        return trajectory

    def lower_owner_sleep(duration: float) -> None:
        clock.value += duration
        _publish_qpos(port, qpos_by_side=qpos_by_side, measured_at=clock(), available_at=clock())
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        **{planner_key: planner},
        clock=clock,
        sleep=lower_owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert activation.succeeded is True
    assert planner_calls == 1
    assert port.snapshot().phase == RealAgentMotionPhase.ACTIVE
    command = port.command_for_observation(
        TeleopObservation(timestamp=clock()),
        clock(),
    )
    assert isinstance(command, TeleopCommand)
    metadata = command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY]
    assert metadata["joint_trajectory_hash"] == trajectory.semantic_content_hash
    assert builder.last_target is None


@pytest.mark.parametrize("skill_id", ["motion.move_tcp_to_pose", "motion.move_tcp_to_pose_safe"])
@pytest.mark.parametrize("outcome", ["fresh", "stale", "unavailable", "drift", "cancel", "cancel_wait", "future", "deadline"])
def test_preplanned_tcp_waits_for_fresh_state_after_non_motion_window(skill_id, outcome, caplog, monkeypatch):
    bundle, port, builder = _runtime_fixture(include_grasp_approach=True, include_safe_pose=True)
    clock, owner = _Clock(), _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle, clock=clock, speed_scale=.2, skill_id=skill_id)
    qpos = {s: np.zeros(6) for s in ("left", "right")}
    _publish_qpos(port, qpos_by_side=qpos, measured_at=clock(), available_at=clock())
    events = []

    @contextmanager
    def window():
        assert port.snapshot().phase != RealAgentMotionPhase.ACTIVE
        events.append("enter")
        yield
        events.append("exit")

    def calculate(**kwargs):
        assert events == ["enter"]
        if skill_id == "motion.move_tcp_to_pose":
            assert kwargs["time_budget_sec"] > 0
        else:
            assert kwargs["require_grasp_approach"] is False
        clock.value += .3  # Original cut expires while no motion is active.
        if outcome == "deadline":
            clock.value = operation.deadline_at - .025
        if outcome == "future":
            _publish_qpos(port, qpos_by_side=qpos, measured_at=clock()+.1, available_at=clock()+.1)
        if outcome == "unavailable":
            monkeypatch.setattr(port, "latest_measured_qpos", lambda: None)
        return _stationary_grasp_trajectory(qpos)

    def observed_planner(**kwargs):
        # The observed-safe adapter owns its window after correlated capture.
        with window():
            return calculate(**kwargs)

    wait_polls = []
    def sleep(duration):
        clock.value += duration
        if port.snapshot().phase == RealAgentMotionPhase.INACTIVE:
            assert events == ["enter", "exit"]
            wait_polls.append(duration)
        if outcome in {"fresh", "drift"} and len(wait_polls) >= 4:
            pair = {s: v.copy() for s, v in qpos.items()}
            if outcome == "drift":
                pair["right"][0] += .5
            _publish_qpos(port, qpos_by_side=pair, measured_at=clock(), available_at=clock())
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(port=port, contract_bundle=bundle,
        route=route, binding=binding, profile=bundle.pose_target_profile,
        request=request, operation=operation,
        **({"tcp_trajectory_planner": calculate} if skill_id == "motion.move_tcp_to_pose"
           else {"grasp_trajectory_planner": observed_planner}),
        non_motion_read_window=window, clock=clock, sleep=sleep)
    result = lifecycle.activate(request=request, operation=operation,
        cancel_requested=lambda: (outcome == "cancel" and "exit" in events)
            or (outcome == "cancel_wait" and len(wait_polls) >= 2))
    assert events == ["enter", "exit"]
    assert result.succeeded == (outcome == "fresh")
    if outcome != "fresh":
        assert result.lower_hold_confirmed
        assert port.snapshot().phase != RealAgentMotionPhase.ACTIVE
    if outcome in {"fresh", "drift"}:
        assert len(wait_polls) >= 4
    if outcome == "future":
        assert not wait_polls
        assert result.status == "grasp_path_postplan_measured_qpos_from_future"
    if outcome in {"stale", "unavailable"}:
        assert sum(wait_polls) <= .501
        assert result.status == "grasp_path_postplan_measured_qpos_" + outcome
        assert "postplan_qpos" in caplog.text and operation.operation_id in caplog.text
        assert "'poll_count':" in caplog.text and "'maximum_age_ms':" in caplog.text
    if outcome == "drift":
        assert result.status == "grasp_path_start_state_drift"
    if outcome in {"cancel", "cancel_wait"}:
        assert result.status == "real_primitive_cancelled_before_activation"
    assert builder.last_target is None


def test_grasp_approach_activation_evidence_stays_bounded_while_owner_waits() -> None:
    bundle, port, _builder = _runtime_fixture(include_grasp_approach=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id="motion.approach_grasp_pose",
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )
    poll_count = 0

    def delayed_owner_sleep(duration: float) -> None:
        nonlocal poll_count
        clock.value += duration
        poll_count += 1
        _publish_qpos(
            port,
            qpos_by_side=qpos_by_side,
            measured_at=clock(),
            available_at=clock(),
        )
        if poll_count == 150:
            port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        grasp_trajectory_planner=(
            lambda **_kwargs: _stationary_grasp_trajectory(qpos_by_side)
        ),
        clock=clock,
        sleep=delayed_owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert poll_count == 150
    assert activation.succeeded is True
    assert len(activation.evidence_refs) < 16
    assert port.snapshot().phase == RealAgentMotionPhase.ACTIVE


@pytest.mark.parametrize("failure", ["no_path", "stale_start", "cancel", "deadline", "start_drift"])
def test_safe_pose_move_never_falls_back_to_online_ik(failure, monkeypatch):
    from dataclasses import replace

    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspTrajectoryPlanningError,
    )

    bundle, port, builder = _runtime_fixture(include_safe_pose=True)
    clock, owner = _Clock(), _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle, clock=clock, speed_scale=0.2, skill_id="motion.move_tcp_to_pose_safe",
    )
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    _publish_qpos(port, qpos_by_side=qpos, measured_at=clock(), available_at=clock())
    if failure == "stale_start":
        stale_qpos = replace(port.latest_measured_qpos(), measured_at=clock() - 0.3)
        monkeypatch.setattr(port, "latest_measured_qpos", lambda: stale_qpos)
    planned = []

    def planner(**kwargs):
        planned.append(kwargs)
        assert kwargs["require_grasp_approach"] is False
        if failure == "no_path":
            raise RM65GraspTrajectoryPlanningError("path_search_exhausted")
        if failure == "deadline":
            clock.value += 9.8
        changed = {side: values.copy() for side, values in qpos.items()}
        if failure == "start_drift":
            changed["left"][0] += 0.2
        _publish_qpos(port, qpos_by_side=changed, measured_at=clock(), available_at=clock())
        return _stationary_grasp_trajectory(qpos)

    def lower_sleep(duration):
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port, contract_bundle=bundle, route=route, binding=binding,
        profile=bundle.pose_target_profile, request=request, operation=operation,
        grasp_trajectory_planner=planner, clock=clock, sleep=lower_sleep,
    )
    result = lifecycle.activate(request=request, operation=operation,
        cancel_requested=lambda: failure == "cancel" and bool(planned))
    assert not result.succeeded
    assert result.lower_hold_confirmed
    assert port.snapshot().dispatch_count == 0
    assert builder.last_target is None
    assert len(planned) == (0 if failure == "stale_start" else 1)


def test_grasp_approach_rejects_stale_qpos_after_planning_and_holds() -> None:
    bundle, port, _builder = _runtime_fixture(include_grasp_approach=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id="motion.approach_grasp_pose",
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )

    def planner(**_kwargs) -> RealAgentJointTrajectory:
        clock.value += 0.25
        return _stationary_grasp_trajectory(qpos_by_side)

    def lower_owner_sleep(duration: float) -> None:
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        grasp_trajectory_planner=planner,
        clock=clock,
        sleep=lower_owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert activation.succeeded is False
    assert activation.status == "grasp_path_postplan_measured_qpos_stale"
    assert activation.lower_hold_confirmed is True
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert len(owner.holds) == 1


def test_grasp_approach_rejects_future_qpos_after_planning_and_holds() -> None:
    bundle, port, _builder = _runtime_fixture(include_grasp_approach=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id="motion.approach_grasp_pose",
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )

    def planner(**_kwargs) -> RealAgentJointTrajectory:
        _publish_qpos(
            port,
            qpos_by_side=qpos_by_side,
            measured_at=clock() + 0.1,
            available_at=clock() + 0.1,
        )
        return _stationary_grasp_trajectory(qpos_by_side)

    def lower_owner_sleep(duration: float) -> None:
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        grasp_trajectory_planner=planner,
        clock=clock,
        sleep=lower_owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert activation.succeeded is False
    assert activation.status == "grasp_path_postplan_measured_qpos_from_future"
    assert activation.lower_hold_confirmed is True
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert len(owner.holds) == 1


def test_grasp_approach_cancelled_after_planning_never_activates_motion() -> None:
    bundle, port, _builder = _runtime_fixture(include_grasp_approach=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id="motion.approach_grasp_pose",
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )
    cancel_checks = iter((False, True))

    def lower_owner_sleep(duration: float) -> None:
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        grasp_trajectory_planner=(
            lambda **_kwargs: _stationary_grasp_trajectory(qpos_by_side)
        ),
        clock=clock,
        sleep=lower_owner_sleep,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: next(cancel_checks),
    )

    assert activation.succeeded is False
    assert activation.status == "real_primitive_cancelled_before_activation"
    assert activation.lower_hold_confirmed is True
    assert port.snapshot().phase == RealAgentMotionPhase.HELD
    assert port.snapshot().dispatch_count == 0
    assert len(owner.holds) == 1


@pytest.mark.parametrize("skill_id, planner_key", [
    ("motion.approach_grasp_pose", "grasp_trajectory_planner"),
    ("motion.move_tcp_to_pose_safe", "grasp_trajectory_planner"),
    ("motion.move_tcp_to_pose", "tcp_trajectory_planner"),
])
def test_grasp_approach_cannot_complete_before_planned_trajectory_elapsed(
    skill_id, planner_key,
) -> None:
    bundle, port, _builder = _runtime_fixture(
        include_grasp_approach=True, include_safe_pose=True,
    )
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.2,
        skill_id=skill_id,
    )
    qpos_by_side = {
        "left": np.zeros(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    _publish_qpos(
        port,
        qpos_by_side=qpos_by_side,
        measured_at=clock(),
        available_at=clock(),
    )
    trajectory = _stationary_grasp_trajectory(qpos_by_side)
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, 3] = (0.4, 0.2, 0.3)
    dispatch_times: list[float] = []

    def lower_owner_sleep(duration: float) -> None:
        clock.value += duration
        _publish_qpos(port, qpos_by_side=qpos_by_side,
            measured_at=clock(), available_at=clock())
        snapshot = port.apply_owner_requests(session=owner, now=clock())
        if snapshot.phase != RealAgentMotionPhase.ACTIVE:
            return
        command = port.command_for_observation(
            TeleopObservation(timestamp=clock()),
            clock(),
        )
        assert isinstance(command, TeleopCommand)
        port.observe_session_step(command=command, command_dispatched=True)
        dispatch_times.append(clock())
        port.publish_measured_pose(
            pose_world_by_component={"right_arm": target_pose},
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=(
                f"real-hardware-cut:grasp-target:{len(dispatch_times)}",
            ),
        )

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
        request=request,
        operation=operation,
        **{planner_key: lambda **_kwargs: trajectory},
        clock=clock,
        sleep=lower_owner_sleep,
    )
    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activation.succeeded is True
    activated_at = port.snapshot().activated_at
    assert activated_at is not None

    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert completion.succeeded is True
    assert dispatch_times[-1] >= activated_at + 0.5
    assert len(dispatch_times) > 3
    assert len(completion.execution_receipt_refs) < 16
    assert (
        "real-agent-joint-trajectory:" + trajectory.semantic_content_hash
        in completion.execution_receipt_refs
    )


@pytest.mark.parametrize("publication_kind", ["pose", "qpos"])
def test_grasp_activation_accepts_measurement_published_during_read(
    monkeypatch, publication_kind
) -> None:
    bundle, port, _builder = _runtime_fixture(include_grasp_approach=True)
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle, clock=clock, speed_scale=0.2, skill_id="motion.approach_grasp_pose",
    )
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    _publish_qpos(port, qpos_by_side=qpos, measured_at=clock(), available_at=clock())
    method = "latest_measured_" + publication_kind
    original_read = getattr(port, method)

    def publish_during_read():
        clock.value += 0.001
        _publish_qpos(port, qpos_by_side=qpos, measured_at=clock(), available_at=clock())
        return original_read()

    monkeypatch.setattr(port, method, publish_during_read)

    def lower_owner_sleep(duration):
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port, contract_bundle=bundle, route=route, binding=binding,
        profile=bundle.pose_target_profile, request=request, operation=operation,
        grasp_trajectory_planner=lambda **_kwargs: _stationary_grasp_trajectory(qpos),
        clock=clock, sleep=lower_owner_sleep,
    )
    activation = lifecycle.activate(
        request=request, operation=operation, cancel_requested=lambda: False,
    )
    assert activation.succeeded is True
    assert port.snapshot().phase == RealAgentMotionPhase.ACTIVE


def test_real_lifecycle_rejects_future_measured_tcp_pose() -> None:
    bundle, port, _builder = _runtime_fixture()
    clock = _Clock()
    request, operation, route, binding = _request_and_operation(bundle, clock=clock)
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4, dtype=float)},
        measured_at=clock() + 0.1,
        available_at=clock() + 0.1,
        source_evidence_refs=("real-hardware-cut:future",),
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
        sleep=lambda _duration: None,
    )

    with pytest.raises(RuntimeError, match="from the future"):
        lifecycle.activate(
            request=request,
            operation=operation,
            cancel_requested=lambda: False,
        )
    assert port.snapshot().phase == RealAgentMotionPhase.INACTIVE


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


def test_real_lifecycle_waits_for_lower_owner_after_derived_cut_stalls() -> None:
    bundle, port, builder = _runtime_fixture()
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _request_and_operation(bundle, clock=clock)
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:before-ik-stall",),
    )
    publish_sequence = 0

    def lower_owner_sleep(duration: float) -> None:
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
        port.observe_session_step(command=command, command_dispatched=True)
        assert builder.last_target is not None
        publish_sequence += 1
        port.publish_measured_pose(
            pose_world_by_component={
                "right_arm": builder.last_target.target_pose_by_side["right"],
            },
            measured_at=clock(),
            available_at=clock(),
            source_evidence_refs=(
                f"real-hardware-cut:after-ik-stall:{publish_sequence}",
            ),
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
        sleep=lower_owner_sleep,
    )
    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    assert activation.succeeded is True

    # Reproduce the original race: the evaluator observes the last derived
    # pose after its 200 ms age, while the lower owner is about to publish the
    # next physical cut.  This is a wait condition, not an evaluator-owned
    # terminal failure.
    clock.value += 0.25
    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )

    assert completion.succeeded is True
    assert completion.reason_code == "real_primitive_completed_after_lower_hold"
    assert publish_sequence >= 3


def test_absolute_tool_tcp_target_transforms_once_and_propagates_speed_scale() -> None:
    bundle, port, builder = _runtime_fixture(
        station_from_execution_x=0.1,
        include_inactive_command=True,
    )
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle,
        clock=clock,
        speed_scale=0.25,
    )
    bundle.payload_validators.validate_parameters(
        route=route,
        payload=request.parameters,
    )
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4, dtype=float)},
        measured_at=clock(),
        available_at=clock(),
        source_evidence_refs=("real-hardware-cut:absolute-pose",),
    )

    def owner_sleep(duration: float) -> None:
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port,
        contract_bundle=bundle,
        route=route,
        binding=binding,
        profile=bundle.pose_target_profile,
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
    assert lifecycle.target_pose is not None
    np.testing.assert_allclose(
        lifecycle.target_pose[:3],
        [0.3, 0.2, 0.3],
        rtol=0.0,
        atol=2e-8,
    )
    command = port.command_for_observation(
        TeleopObservation(timestamp=clock()),
        clock(),
    )
    assert isinstance(command, TeleopCommand)
    operation_metadata = command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY]
    assert operation_metadata["speed_scale"] == 0.25
    assert command.commands["right_arm"].metadata[
        REAL_AGENT_MOTION_COMMAND_METADATA_KEY
    ] == operation_metadata
    assert (
        REAL_AGENT_MOTION_COMMAND_METADATA_KEY
        not in command.commands["left_arm"].metadata
    )
    assert builder.last_target is not None
    np.testing.assert_allclose(
        builder.last_target.target_pose_by_side["right"][:3, 3],
        [0.3, 0.2, 0.3],
        rtol=0.0,
        atol=2e-8,
    )


def test_authorized_absolute_pose_route_registers_a_real_gateway_executor() -> None:
    bundle, port, _builder = _runtime_fixture()
    attachment = bundle.attachment
    owner = RealControlOwnerState(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        execution_manifest_hash=attachment.execution_manifest_hash,
        backend_inventory_hash=attachment.backend_inventory_hash,
        clock_domain_id=attachment.clock_domain_id,
        owner_token_id="real-owner-token:absolute-pose-test",
        owner_kind="exclusive_real_skill_motion_port",
        owner_generation=1,
        authority_epoch=1,
        execution_epoch=1,
        lifecycle_state="running",
        execution_gate="ready",
        owner_state_revision=1,
        captured_at=10.0,
        source_owner_state_hash="lower-owner-state:absolute-pose-test",
        evidence_refs=("lower-owner-evidence:absolute-pose-test",),
    )

    session = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner,
        allowed_client_ids=("rpent-test-client",),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    assert "motion.move_tcp_to_pose" in session.executors_by_skill_id
    route = bundle.route_manifest.route("motion.move_tcp_to_pose", "1").route
    bound_ports = {
        binding.route_hash: binding.execution_port
        for binding in session.service.executor_registry.bindings
    }
    assert bound_ports[route.semantic_content_hash] is session.executors_by_skill_id[
        "motion.move_tcp_to_pose"
    ]

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


def test_real_gateway_session_retries_unconfirmed_close_hold() -> None:
    bundle, port, _builder = _runtime_fixture()
    clock = _Clock()
    lower_owner = _OwnerSession()
    attachment = bundle.attachment
    owner_state = RealControlOwnerState(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        execution_manifest_hash=attachment.execution_manifest_hash,
        backend_inventory_hash=attachment.backend_inventory_hash,
        clock_domain_id=attachment.clock_domain_id,
        owner_token_id="real-owner-token:close-retry",
        owner_kind="exclusive_real_skill_motion_port",
        owner_generation=1,
        authority_epoch=1,
        execution_epoch=1,
        lifecycle_state="running",
        execution_gate="ready",
        owner_state_revision=1,
        captured_at=clock(),
        source_owner_state_hash="lower-owner-state:close-retry",
        evidence_refs=("lower-owner-evidence:close-retry",),
    )
    apply_lower_hold = False

    def lower_sleep(duration: float) -> None:
        clock.value += duration
        if apply_lower_hold:
            port.apply_owner_requests(session=lower_owner, now=clock())

    session = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: owner_state,
        allowed_client_ids=("rpent-test-client",),
        clock=clock,
        sleep=lower_sleep,
    )

    first = session.close()
    assert not first.accepted
    assert session.service.is_closed
    assert not session._closed_executor_ids

    apply_lower_hold = True
    second = session.close()
    third = session.close()

    assert second.accepted
    assert third is second
    assert len(session._closed_executor_ids) == len(
        {id(executor) for executor in session.executors_by_skill_id.values()}
    )


@pytest.mark.parametrize("lower_reason,reason_code", [
    ("lower_dispatch_failed", "real_primitive_lower_dispatch_failed"),
    ("lower_backend_not_ready", "real_primitive_lower_backend_not_ready"),
    ("exclusive_skill_control_cycle_failed", "real_primitive_lower_control_cycle_failed"),
])
def test_real_lifecycle_reports_lower_failure_and_preserves_reason(caplog, lower_reason, reason_code) -> None:
    bundle, port, _ = _runtime_fixture()
    clock = _Clock()
    owner = _OwnerSession()
    request, operation, route, binding = _request_and_operation(bundle, clock=clock)
    port.publish_measured_pose(
        pose_world_by_component={"right_arm": np.eye(4)},
        measured_at=clock(), available_at=clock(),
        source_evidence_refs=("real-hardware-cut:before-failed-dispatch",),
    )

    def owner_sleep(duration):
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(
        port=port, contract_bundle=bundle, route=route, binding=binding,
        profile=bundle.cardinal_profile, request=request, operation=operation,
        clock=clock, sleep=owner_sleep,
    )
    assert lifecycle.activate(request=request, operation=operation,
        cancel_requested=lambda: False).succeeded
    native_reason = lower_reason + ": native failure details"
    port.request_hold(operation_id=operation.operation_id,
        reason=native_reason, requested_at=clock())
    port.apply_owner_requests(session=owner, now=clock())
    result = lifecycle.run(request=request, operation=operation,
        cancel_requested=lambda: False)
    assert not result.succeeded
    assert result.reason_code == reason_code
    assert result.terminal_hold_confirmed
    assert result.execution_result is None
    assert port.snapshot().hold_reason == native_reason
    assert native_reason in caplog.text
    assert len(owner.holds) == 1


@pytest.mark.parametrize("skill_id", ["motion.move_tcp_to_pose", "motion.move_tcp_to_pose_safe"])
def test_preplanned_tcp_requires_new_cut_even_when_previous_cut_is_still_fresh(skill_id):
    bundle, port, builder = _runtime_fixture(include_grasp_approach=True, include_safe_pose=True,
        convergence_profile=RealAgentPrimitiveConvergenceProfile(max_measurement_age_ms=1000))
    clock, owner = _Clock(), _OwnerSession()
    request, operation, route, binding = _pose_request_and_operation(
        bundle, clock=clock, speed_scale=.2, skill_id=skill_id)
    qpos = {s: np.zeros(6) for s in ("left", "right")}
    _publish_qpos(port, qpos_by_side=qpos, measured_at=clock(), available_at=clock())
    initial_measured_at = port.latest_measured_qpos().measured_at
    events, wait_polls = [], []

    @contextmanager
    def window():
        events.append("enter")
        yield
        events.append("exit")

    def calculate(**kwargs):
        assert events == ["enter"]
        clock.value += .3
        return _stationary_grasp_trajectory(qpos)

    def safe_planner(**kwargs):
        with window():
            return calculate(**kwargs)

    def sleep(duration):
        if port.snapshot().phase == RealAgentMotionPhase.INACTIVE:
            wait_polls.append(duration)
        clock.value += duration
        port.apply_owner_requests(session=owner, now=clock())

    lifecycle = RealAgentPrimitiveSkillLifecycle(port=port, contract_bundle=bundle,
        route=route, binding=binding, profile=bundle.pose_target_profile,
        request=request, operation=operation,
        **({"tcp_trajectory_planner": calculate} if skill_id == "motion.move_tcp_to_pose"
           else {"grasp_trajectory_planner": safe_planner}),
        non_motion_read_window=window, clock=clock, sleep=sleep)
    result = lifecycle.activate(request=request, operation=operation, cancel_requested=lambda: False)
    assert events == ["enter", "exit"]
    assert .49 <= sum(wait_polls) <= .501
    assert port.latest_measured_qpos().measured_at == initial_measured_at
    assert (clock() - initial_measured_at) * 1000 < 1000
    assert not result.succeeded
    assert result.status == "grasp_path_postplan_measured_qpos_not_refreshed"
    assert result.lower_hold_confirmed
    assert port.snapshot().dispatch_count == 0
    assert port.snapshot().phase != RealAgentMotionPhase.ACTIVE
    assert builder.last_target is None

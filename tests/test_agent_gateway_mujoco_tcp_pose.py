from __future__ import annotations

import io
import math
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import agent_gateway.profiles.mujoco.motion_deployment as deployment
from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
    resolve_level_tcp_orientation_primitive_target,
    resolve_tcp_displacement_primitive_target,
)
from agent_gateway.profiles.mujoco import motion_skill
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
    MujocoAgentSkillMotionStep,
)
from agent_gateway.profiles.mujoco.motion_deployment import (
    MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1,
    MujocoAgentTcpPoseDeploymentBinding,
    MujocoAgentTcpPoseMotionProfile,
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ENVELOPE_ID_V1,
    MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1,
    _build_standard_tcp_pose_envelope_runtime,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
    validate_mujoco_agent_tcp_pose_target_config,
)
from agent_gateway.profiles.mujoco.motion_gateway_session import (
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from agent_gateway.profiles.mujoco.motion_primitive_profiles import (
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    MujocoAgentTcpCardinalStepProfile,
    MujocoAgentTcpFixedOrientationProfile,
)
from agent_gateway.profiles.mujoco.motion_skill import (
    MujocoAgentTcpPoseSkillLifecycle,
    mujoco_agent_skill_hold_evidence_from_transition,
)
from agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
)
from agent_gateway.protocol.skill_catalog import (
    SkillExecutionRoute,
    SkillParameterProfileRef,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
)
from agent_gateway.skills.perception.definition import (
    CAPTURE_DEPTH_SKILL_ID_V1,
    CAPTURE_RGB_SKILL_ID_V1,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import cardinal_tcp_short_step, level_tcp_orientation
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy import (
    MujocoDirectComponentProposalSource,
    MujocoSimHILTargetConfig,
    build_mujoco_pose_component_target_profile,
)
from sim.mujoco.depth_snapshot import EncodedMujocoDepthFrame
from sim.mujoco.rgb_snapshot import (
    EncodedMujocoRgbFrame,
    EncodedMujocoRgbSnapshot,
)


def test_motion_primitive_organization_preserves_legacy_profile_exports() -> None:
    assert deployment.MujocoAgentTcpCardinalStepProfile is (
        MujocoAgentTcpCardinalStepProfile
    )
    assert deployment.MujocoAgentTcpFixedOrientationProfile is (
        MujocoAgentTcpFixedOrientationProfile
    )
    assert motion_skill.MujocoAgentTcpTargetResolution is (
        MotionPrimitiveTargetResolution
    )
    assert motion_skill.resolve_tcp_displacement_primitive_target is (
        resolve_tcp_displacement_primitive_target
    )
    assert motion_skill.resolve_level_tcp_orientation_primitive_target is (
        resolve_level_tcp_orientation_primitive_target
    )


def _profile_and_schema(component_id: str = "right_arm"):
    side = component_id.removesuffix("_arm")
    target_profile = build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id=f"fixture.agent_tcp_pose.{component_id}.v1",
        arm_side_by_component={component_id: side},
        arm_reference_frame_by_component={component_id: f"{side}_arm_base"},
        arm_controlled_frame_by_component={component_id: "tool_tcp"},
        selection_group_by_component={component_id: "arms"},
    )
    motion_profile = MujocoAgentTcpPoseMotionProfile(
        reference_frame_id=f"{side}_arm_base",
        reference_frame_spec_hash=f"frame-spec:{side}-arm-base",
        frame_graph_hash="frame-graph:fixture",
        tool_config_hash="tool-config:fixture",
        calibration_hash="calibration:fixture",
        component_id=component_id,
        stable_measurement_count=2,
        max_control_cycles=4,
        physics_ticks_per_control_cycle=3,
    )
    return motion_profile, target_profile.action_schema


def _dual_arm_action_schema():
    target_profile = build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id="fixture.agent_tcp_pose.dual_arm.v1",
        arm_side_by_component={
            "left_arm": "left",
            "right_arm": "right",
        },
        arm_reference_frame_by_component={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        arm_controlled_frame_by_component={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        selection_group_by_component={
            "left_arm": "arms",
            "right_arm": "arms",
        },
    )
    return target_profile.action_schema


def test_cardinal_step_target_is_derived_from_lower_activation_pose() -> None:
    resolution = resolve_tcp_displacement_primitive_target(
        measured_pose=(0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0),
        primitive=cardinal_tcp_short_step("up"),
        reference_frame_spec=CoordinateFrameSpec(
            frame_id="right_arm_base",
            frame_version="fixture",
            semantic_parameters={
                "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
            },
        ),
    )

    assert resolution.target_pose == pytest.approx(
        (0.4, -0.1, 0.53, 0.0, 0.0, 0.0, 1.0)
    )
    assert any(
        item.startswith("tcp-displacement-primitive:")
        for item in resolution.evidence_refs
    )
    assert any(
        item.startswith("tcp-displacement-target:")
        for item in resolution.evidence_refs
    )


def test_fixed_level_orientation_is_absolute_and_preserves_tcp_position() -> None:
    frame = CoordinateFrameSpec(
        frame_id="right_arm_base",
        frame_version="fixture",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    primitive = level_tcp_orientation("forward_left_30")
    first = resolve_level_tcp_orientation_primitive_target(
        measured_pose=(0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0),
        primitive=primitive,
        reference_frame_spec=frame,
    )
    second = resolve_level_tcp_orientation_primitive_target(
        measured_pose=(0.2, 0.3, 0.6, 0.5, 0.5, 0.5, 0.5),
        primitive=primitive,
        reference_frame_spec=frame,
    )

    assert first.target_pose[:3] == pytest.approx((0.4, -0.1, 0.5))
    assert second.target_pose[:3] == pytest.approx((0.2, 0.3, 0.6))
    assert first.target_pose[3:] == pytest.approx(second.target_pose[3:])
    assert any(
        item.startswith("tcp-level-orientation-primitive:")
        for item in first.evidence_refs
    )


def _rotated_station_graph() -> tuple[CoordinateFrameGraph, CoordinateFrameSpec]:
    station = CoordinateFrameSpec(
        frame_id="station",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    graph = CoordinateFrameGraph(
        graph_id="rotated-station-fixture",
        frames=(CoordinateFrameSpec(frame_id="right_arm_base"), station),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="right-base-to-station",
                source_frame_id="right_arm_base",
                target_frame_id="station",
                matrix_target_source=(
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
    )
    return graph, station


def test_cardinal_step_uses_station_axes_not_rotated_arm_base_axes() -> None:
    graph, station = _rotated_station_graph()
    resolution = resolve_tcp_displacement_primitive_target(
        measured_pose=(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        primitive=cardinal_tcp_short_step("forward"),
        reference_frame_spec=station,
        frame_graph=graph,
        measured_reference_frame_id="right_arm_base",
    )

    assert resolution.target_pose[:3] == pytest.approx((1.0, -0.03, 0.5))
    assert sum(
        item.startswith("tcp-primitive-frame-resolution:")
        for item in resolution.evidence_refs
    ) == 1
    assert sum(
        item.startswith("tcp-execution-frame-resolution:")
        for item in resolution.evidence_refs
    ) == 1


def test_fixed_orientation_is_defined_in_station_then_returned_in_arm_base() -> None:
    graph, station = _rotated_station_graph()
    primitive = level_tcp_orientation("center_forward")
    resolution = resolve_level_tcp_orientation_primitive_target(
        measured_pose=(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        primitive=primitive,
        reference_frame_spec=station,
        frame_graph=graph,
        measured_reference_frame_id="right_arm_base",
    )
    target_in_base = xyzquat_to_pose_matrix(
        resolution.target_pose,
        name="fixed_orientation_target_in_base",
    )
    target_in_station, _resolution = graph.transform_pose_matrix(
        target_in_base,
        source_reference_frame_id="right_arm_base",
        target_reference_frame_id="station",
    )

    assert resolution.target_pose[:3] == pytest.approx((1.0, 0.0, 0.5))
    assert target_in_station[:3, :3] == pytest.approx(
        primitive.target_rotation_reference
    )


def test_mujoco_fixed_orientation_routes_use_a_separate_full_rotation_envelope() -> None:
    runtime = _build_standard_tcp_pose_envelope_runtime(
        prepared=SimpleNamespace(
            target_frame_profile=SimpleNamespace(
                semantic_content_hash="target-frame-profile:fixture",
                reference_frame_by_component={"right_arm": "right_arm_base"},
            ),
            tool_frame_profile=SimpleNamespace(
                binding_by_component={
                    "right_arm": SimpleNamespace(
                        semantic_content_hash="tool-binding:right-arm:fixture"
                    )
                },
            ),
        ),
        target_config=SimpleNamespace(clock_domain_id="clock:mujoco:fixture"),
    )
    entry_by_id = {entry.envelope_id: entry for entry in runtime.catalog_entries}
    standard = entry_by_id[MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1]
    orientation = entry_by_id[
        MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ENVELOPE_ID_V1
    ]

    assert standard.profile.side_config_by_side["right"].max_rotation_rad == (
        pytest.approx(0.2617993877991494)
    )
    assert orientation.profile.side_config_by_side["right"].max_rotation_rad == (
        pytest.approx(3.141592653589793)
    )
    assert orientation.profile.side_config_by_side[
        "right"
    ].relative_min_xyz_m == (0.0, 0.0, 0.0)
    assert set(orientation.allowed_route_ids) == set(
        MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1.values()
    )
    assert set(standard.allowed_route_ids).isdisjoint(
        orientation.allowed_route_ids
    )


def _contracts(motion_profile):
    route = SkillExecutionRoute(
        skill_id="motion.move_tcp_to_pose",
        skill_version="1",
        skill_definition_hash="definition:move-tcp-to-pose:fixture",
        route_id="sim.mujoco.rm65.agent_skill_tcp_pose",
        route_version="1",
        parameter_schema_hash="parameter-schema:move-tcp:fixture",
        execution_result_schema_hash="result-schema:move-tcp:fixture",
        task_evaluation_schema_hash="evaluation-schema:move-tcp:fixture",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="right-arm-bounded-tool-tcp-target-v1",
                parameter_profile_hash=motion_profile.semantic_content_hash,
                parameter_schema_hash="parameter-schema:move-tcp:fixture",
            ),
        ),
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=10_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=True,
    )
    binding = MujocoAgentTcpPoseDeploymentBinding(
        deployment_binding_id="deployment:agent-tcp-pose:fixture",
        deployment_binding_version="1",
        route_hash=route.semantic_content_hash,
        runtime_session_id="runtime:agent-tcp-pose:fixture",
        control_domain_id="domain:agent-tcp-pose:fixture",
        target_binding_id="target:agent-tcp-pose:fixture",
        station_manifest_hash="station-manifest:fixture",
        execution_manifest_hash="execution-manifest:fixture",
        execution_domain_binding_hash="execution-domain:fixture",
        target_runtime_action_contract_hash="target-action:fixture",
        component_target_capability_hash="component-target:fixture",
        action_schema_hash="action-schema:fixture",
        observation_binding_id="observation-binding:fixture",
        observation_binding_hash="observation-binding-hash:fixture",
        observation_plan_hash="observation-plan:fixture",
        observation_capability_hash="observation-capability:fixture",
        pose_schema_hash="pose-schema:fixture",
        motion_profile_hash=motion_profile.semantic_content_hash,
        reference_frame_id=motion_profile.reference_frame_id,
        reference_frame_spec_hash=motion_profile.reference_frame_spec_hash,
        frame_graph_hash=motion_profile.frame_graph_hash,
        tool_config_hash=motion_profile.tool_config_hash,
        calibration_hash=motion_profile.calibration_hash,
        trace_requirement_profile_id="trace-profile:fixture",
        executor_id=MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1,
        explicit_exclusions=("hand_control", "requested_duration"),
        component_id=motion_profile.component_id,
    )
    evaluator = SkillEvaluatorSpec(
        evaluator_id="motion.tcp_pose_arrival_rule",
        evaluator_version="1",
        evaluator_kind="rule_based",
        input_evidence_schema_hash="evaluator-input:fixture",
        output_schema_hash=route.task_evaluation_schema_hash,
        role="required",
        execution_location="gateway_worker",
        deadline_ms=1_000,
        failure_policy="veto",
        evidence_domain_id=binding.control_domain_id,
        configuration_hash=motion_profile.semantic_content_hash,
    )
    evaluation_plan = SkillEvaluationPlan(
        evaluation_plan_id="motion.tcp_pose_arrival_rule",
        evaluation_plan_version="1",
        skill_definition_hash=route.skill_definition_hash,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        evaluator_specs=(evaluator,),
        aggregation_policy_id="motion.single-required-rule.v1",
        aggregation_policy_hash="aggregation:fixture",
        required_evaluation_deadline_ms=1_000,
        missing_evidence_policy="fail",
        disagreement_policy="aggregation_policy",
        task_outcome_schema_hash=route.task_evaluation_schema_hash,
    )
    return route, binding, evaluation_plan


def _request_and_operation(
    route,
    binding,
    motion_profile,
    target_pose,
    *,
    component_id: str = "right_arm",
):
    request = AgentSkillRequest(
        client_id="rpent:fixture",
        request_id="request:agent-tcp-pose:1",
        idempotency_key="idempotency:agent-tcp-pose:1",
        client_sequence=1,
        runtime_session_id=binding.runtime_session_id,
        skill_route_manifest_hash="route-manifest:fixture",
        expected_station_revision=3,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        parameter_profile_id=route.allowed_parameter_profiles[0].parameter_profile_id,
        parameter_profile_hash=motion_profile.semantic_content_hash,
        parameters={
            "pose_targets_by_component": {component_id: list(target_pose)},
            "reference_frame_id": motion_profile.reference_frame_id,
            "reference_frame_spec_hash": (
                motion_profile.reference_frame_spec_hash
            ),
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": motion_profile.frame_graph_hash,
            "tool_config_hash": motion_profile.tool_config_hash,
            "calibration_hash": motion_profile.calibration_hash,
        },
        observation_snapshot_id="agent-observation:fixture:3",
        requested_mode="sim_active",
        requested_timeout_ms=5_000,
    )
    operation = AgentSkillOperation(
        operation_id="operation:agent-tcp-pose:1",
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
        status="running",
        reason_code="operation_activated",
        station_revision_at_admission=3,
        current_station_revision=3,
        started_at=10.0,
        updated_at=10.0,
        deadline_at=15.0,
    )
    return request, operation


def _ready_transition():
    return SimpleNamespace(
        snapshot=SimpleNamespace(status="ready"),
        lower_event_ids=("gate-ready:1", "gate-ready-fact:1"),
    )


def _hold_transition():
    return SimpleNamespace(
        snapshot=SimpleNamespace(
            status="hold",
            semantic_content_hash="gate-snapshot-hold:1",
        ),
        event=SimpleNamespace(event_id="gate-hold:1"),
        cancellation=SimpleNamespace(cancellation_id="cancellation:hold:1"),
        cancellation_receipt=SimpleNamespace(
            complete=True,
            timed_out=False,
            requires_fault_escalation=False,
            pending_levels=(),
            rejected_levels=(),
        ),
        lower_event_ids=("gate-hold:1", "cancellation-ack:1"),
    )


def _lifecycle_application(runtime):
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.runtime = runtime
    application._lock = threading.RLock()
    application._started = False
    application._held = False
    application._closed = False
    application._active_operation_id = None
    return application


def test_observation_ready_startup_starts_runtime_and_proves_lower_hold() -> None:
    calls: list[object] = []
    transition = _hold_transition()
    runtime = SimpleNamespace(
        start=lambda: calls.append("start"),
        transition_execution_gate=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or transition
        ),
    )
    application = _lifecycle_application(runtime)

    outcome = application.start_observation_ready()

    assert outcome is transition
    assert application.started
    assert application.held
    assert calls == [
        "start",
        (
            ("hold",),
            {
                "reason": "agent_gateway_observation_ready_startup",
                "source_event_ids": ("agent-gateway-observation-ready",),
                "force_cancellation": True,
            },
        ),
    ]


def test_observation_ready_startup_fails_closed_without_lower_hold_proof() -> None:
    runtime = SimpleNamespace(
        start=lambda: None,
        transition_execution_gate=lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=SimpleNamespace(status="hold"),
            lower_event_ids=(),
        ),
    )
    application = _lifecycle_application(runtime)

    with pytest.raises(RuntimeError, match="did not prove HOLD"):
        application.start_observation_ready()

    assert application.started
    assert not application.held


def _fake_application(
    action_schema,
    *,
    start_pose,
    target_pose,
    component_id: str = "right_arm",
):
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.runtime = SimpleNamespace(
        action_schema=action_schema,
        config=SimpleNamespace(
            runtime_session_id="runtime:agent-tcp-pose:fixture",
            decision_hz=20.0,
        ),
    )
    poses = [start_pose, target_pose, target_pose, target_pose]
    calls: dict[str, list] = {
        "activate": [],
        "submit": [],
        "hold": [],
        "release": [],
    }

    def capture():
        return {component_id: poses.pop(0)}

    def activate_operation(*, operation_id):
        calls["activate"].append(operation_id)
        return _ready_transition()

    def submit_and_step(**kwargs):
        calls["submit"].append(kwargs)
        sequence = len(calls["submit"])
        return MujocoAgentSkillMotionStep(
            proposal_admission=SimpleNamespace(
                accepted=True,
                rejection_code=None,
                proposal_id=f"proposal:{sequence}",
                semantic_content_hash=f"proposal-admission:{sequence}",
            ),
            authoritative_step=SimpleNamespace(
                status="dispatched",
                trace_facts=(SimpleNamespace(fact_id=f"dispatch-fact:{sequence}"),),
                target_execution_request=SimpleNamespace(
                    request_id=f"target-request:{sequence}"
                ),
            ),
        )

    def request_hold(**kwargs):
        calls["hold"].append(kwargs)
        return _hold_transition()

    def release_operation(*, operation_id):
        calls["release"].append(operation_id)

    application.capture_tcp_pose_by_component = capture
    application.activate_operation = activate_operation
    application.submit_and_step = submit_and_step
    application.request_hold = request_hold
    application.release_operation = release_operation
    return application, calls


def test_tcp_pose_lifecycle_dispatches_through_lower_and_finishes_under_hold() -> None:
    motion_profile, action_schema = _profile_and_schema()
    route, binding, evaluation_plan = _contracts(motion_profile)
    target_pose = (0.40, -0.10, 0.50, 0.0, 0.0, 0.0, 1.0)
    start_pose = (0.38, -0.10, 0.50, 0.0, 0.0, 0.0, 1.0)
    request, operation = _request_and_operation(
        route,
        binding,
        motion_profile,
        target_pose,
    )
    application, calls = _fake_application(
        action_schema,
        start_pose=start_pose,
        target_pose=target_pose,
    )
    lifecycle = MujocoAgentTcpPoseSkillLifecycle(
        application=application,
        route=route,
        deployment_binding=binding,
        evaluation_plan=evaluation_plan,
        motion_profile=motion_profile,
        request=request,
        operation=operation,
        clock=lambda: 12.0,
        sleep=lambda _seconds: None,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    lifecycle.close()

    assert activation.succeeded
    assert completion.succeeded
    assert completion.terminal_hold_confirmed
    assert completion.evaluation is not None
    assert completion.evaluation.task_outcome == "succeeded"
    assert len(calls["submit"]) == 2
    assert all(call["observation_cut_id"] is None for call in calls["submit"])
    assert all(
        call["context_ids"] == (request.observation_snapshot_id,)
        for call in calls["submit"]
    )
    assert calls["submit"][0]["physics_ticks_after_dispatch"] == 3
    assert calls["hold"] == [
        {
            "reason": "tcp_pose_terminal_candidate_reached",
            "operation_id": operation.operation_id,
        }
    ]
    assert calls["release"] == [operation.operation_id]


def test_lifecycle_resolves_cardinal_target_at_lower_activation_boundary() -> None:
    motion_profile, action_schema = _profile_and_schema()
    route, binding, evaluation_plan = _contracts(motion_profile)
    start_pose = (0.38, -0.10, 0.50, 0.0, 0.0, 0.0, 1.0)
    target_pose = (0.41, -0.10, 0.50, 0.0, 0.0, 0.0, 1.0)
    request, operation = _request_and_operation(
        route,
        binding,
        motion_profile,
        start_pose,
    )
    application, calls = _fake_application(
        action_schema,
        start_pose=start_pose,
        target_pose=target_pose,
    )
    reference_frame = CoordinateFrameSpec(
        frame_id="right_arm_base",
        frame_version="fixture",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    lifecycle = MujocoAgentTcpPoseSkillLifecycle(
        application=application,
        route=route,
        deployment_binding=binding,
        evaluation_plan=evaluation_plan,
        motion_profile=motion_profile,
        request=request,
        operation=operation,
        target_resolver=lambda measured: (
            resolve_tcp_displacement_primitive_target(
                measured_pose=measured,
                primitive=cardinal_tcp_short_step("forward"),
                reference_frame_spec=reference_frame,
            )
        ),
        clock=lambda: 12.0,
        sleep=lambda _seconds: None,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    lifecycle.close()

    assert activation.succeeded
    assert lifecycle.target_pose == pytest.approx(target_pose)
    assert completion.succeeded
    submitted = calls["submit"][0]["action"].values_by_component[
        "right_arm"
    ].values
    assert submitted == pytest.approx(target_pose)
    assert any(
        ref.startswith("tcp-displacement-target:")
        for ref in completion.execution_receipt_refs
    )


@pytest.mark.parametrize("component_id", ("left_arm", "right_arm"))
def test_lifecycle_resolves_fixed_level_orientation_for_either_arm(
    component_id: str,
) -> None:
    motion_profile, _single_arm_action_schema = _profile_and_schema(component_id)
    action_schema = _dual_arm_action_schema()
    motion_profile = replace(
        motion_profile,
        max_rotation_delta_rad=1.2,
    )
    route, binding, evaluation_plan = _contracts(motion_profile)
    side = component_id.removesuffix("_arm")
    reference_frame = CoordinateFrameSpec(
        frame_id=f"{side}_arm_base",
        frame_version="fixture",
        semantic_parameters={
            "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        },
    )
    primitive = level_tcp_orientation("forward_left_30")
    center_resolution = resolve_level_tcp_orientation_primitive_target(
        measured_pose=(0.38, -0.10, 0.50, 0.0, 0.0, 0.0, 1.0),
        primitive=level_tcp_orientation("center_forward"),
        reference_frame_spec=reference_frame,
    )
    anchor_pose = center_resolution.target_pose
    target_resolution = resolve_level_tcp_orientation_primitive_target(
        measured_pose=anchor_pose,
        primitive=primitive,
        reference_frame_spec=reference_frame,
    )
    target_pose = target_resolution.target_pose
    request, operation = _request_and_operation(
        route,
        binding,
        motion_profile,
        target_pose,
        component_id=component_id,
    )
    application, calls = _fake_application(
        action_schema,
        start_pose=anchor_pose,
        target_pose=target_pose,
        component_id=component_id,
    )
    lifecycle = MujocoAgentTcpPoseSkillLifecycle(
        application=application,
        route=route,
        deployment_binding=binding,
        evaluation_plan=evaluation_plan,
        motion_profile=motion_profile,
        request=request,
        operation=operation,
        target_resolver=lambda measured: (
            resolve_level_tcp_orientation_primitive_target(
                measured_pose=measured,
                primitive=primitive,
                reference_frame_spec=reference_frame,
            )
        ),
        clock=lambda: 12.0,
        sleep=lambda _seconds: None,
    )

    activation = lifecycle.activate(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    completion = lifecycle.run(
        request=request,
        operation=operation,
        cancel_requested=lambda: False,
    )
    lifecycle.close()

    assert activation.succeeded
    assert completion.succeeded
    assert binding.component_id == component_id
    assert lifecycle.action is not None
    assert lifecycle.action.component_coverage == (component_id,)
    assert lifecycle.action.value(component_id).values == pytest.approx(target_pose)
    assert calls["hold"]


def test_hold_evidence_never_claims_unstarted_runtime_is_held() -> None:
    evidence = mujoco_agent_skill_hold_evidence_from_transition(
        None,
        lower_operation_id=None,
    )

    assert not evidence.succeeded
    assert not evidence.lower_hold_confirmed
    assert evidence.evidence_refs == ()
    assert evidence.status == "lower_runtime_not_started_no_hold_transition"


def test_graph_factory_rejects_non_dedicated_target_config_before_scene_load() -> None:
    config = MujocoSimHILTargetConfig(
        task_config_path=Path("unused-agent-motion-task.json"),
        realtime_worker=False,
        execution_observation_plan=(
            mujoco_agent_tcp_pose_execution_observation_plan_v1()
        ),
        exposed_pose_sides=("right",),
        allow_scripted_task_reset=False,
    )
    validate_mujoco_agent_tcp_pose_target_config(config)

    with pytest.raises(ValueError, match="scripted_task_reset"):
        validate_mujoco_agent_tcp_pose_target_config(
            replace(config, allow_scripted_task_reset=True)
        )
    with pytest.raises(ValueError, match="exactly the right"):
        validate_mujoco_agent_tcp_pose_target_config(
            replace(config, exposed_pose_sides=("left", "right"))
        )


def _build_no_model_contract_bundle_fixture(
    monkeypatch,
    *,
    enable_artifact_perception: bool = False,
):
    _profile, action_schema = _profile_and_schema()
    observation_plan = SimpleNamespace(
        enabled=True,
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={
            "right_arm": "cartesian_pose"
        },
        semantic_content_hash="observation-plan:agent-motion-fixture",
    )
    direct = MujocoDirectComponentProposalSource(
        producer_id="control-station-agent-skill-fixture",
        producer_epoch=1,
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            runtime_session_id="runtime:agent-motion-bundle-fixture",
            control_domain_id="domain:agent-motion-bundle-fixture",
            embodiment_instance_id="embodiment:agent-motion-bundle-fixture",
            clock_domain_id="clock:agent-motion-bundle-fixture",
            decision_hz=15.0,
            direct_component_proposal_source=direct,
            enable_primary_policy_authority=False,
            enable_composer_authority=False,
            enable_replay_authority=False,
            initial_control_mode="agent_skill",
            execution_observation_plan=observation_plan,
        ),
        action_schema=action_schema,
        catalog=SimpleNamespace(
            action_space=action_schema.schema_id,
            semantic_content_hash="component-catalog:fixture",
        ),
        component_target_capability=SimpleNamespace(
            semantic_content_hash="component-target-capability:fixture"
        ),
        target_runtime_action_contract=SimpleNamespace(
            semantic_content_hash="target-runtime-action:fixture"
        ),
        execution_capability=SimpleNamespace(
            semantic_content_hash="execution-capability:fixture"
        ),
        reference_activation_spec=SimpleNamespace(
            semantic_content_hash="reference-activation:fixture"
        ),
        execution_observation_binding=SimpleNamespace(
            binding_id="observation-binding:fixture",
            plan_hash=observation_plan.semantic_content_hash,
            semantic_content_hash="observation-binding-hash:fixture",
        ),
        execution_observation_capability=SimpleNamespace(
            semantic_content_hash="observation-capability:fixture",
            representation_by_component={"right_arm": "cartesian_pose"},
            schema_hash_by_component={"right_arm": "pose-schema:fixture"},
        ),
        authoritative_port=SimpleNamespace(
            profile=SimpleNamespace(
                selection_group_ids=("arms",),
                authority_policy_hash="authority-policy:fixture",
                decision_schedule_hash="decision-schedule:fixture",
            ),
        ),
        clock=lambda: 10.0,
    )
    frame_graph = CoordinateFrameGraph(
        graph_id="frame-graph:fixture",
        frames=(
            CoordinateFrameSpec(frame_id="right_arm_base"),
            CoordinateFrameSpec(
                frame_id="mujoco_sim_hil/station",
                semantic_parameters={
                    "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
                },
            ),
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="fixture:right-arm-base-to-station",
                source_frame_id="right_arm_base",
                target_frame_id="mujoco_sim_hil/station",
                matrix_target_source=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
    )
    application = SimpleNamespace(
        runtime=runtime,
        task_config=SimpleNamespace(physics_hz=1_000.0),
        calibration_hash="calibration:agent-motion-bundle-fixture",
        target_frame_profile=SimpleNamespace(
            semantic_content_hash="target-frame-profile:fixture",
            reference_frame_by_component={"right_arm": "right_arm_base"},
            canonical_reference_frame_id="mujoco_sim_hil/station",
            frame_graph=frame_graph,
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash="tool-config:fixture",
            binding_by_component={
                "right_arm": SimpleNamespace(
                    controlled_frame_id="tool_tcp",
                    semantic_content_hash="tool-binding:right-arm:fixture",
                )
            },
        ),
    )
    application.depth_source_catalog = lambda: (
        {"view_id": "front", "native_width": 2, "native_height": 2},
    )
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 640, "native_height": 480},
    )
    application.tcp_pose_envelope_runtime = _build_standard_tcp_pose_envelope_runtime(
        prepared=application,
        target_config=SimpleNamespace(
            clock_domain_id=runtime.config.clock_domain_id,
        ),
    )
    captured_manifest_args = {}

    def fake_execution_manifest(**kwargs):
        captured_manifest_args.update(kwargs)
        return SimpleNamespace(semantic_content_hash="execution-manifest:fixture")

    monkeypatch.setattr(
        deployment,
        "build_control_execution_manifest",
        fake_execution_manifest,
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_owner_registry_id",
        lambda _runtime: "owner-registry:fixture",
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_authority_epoch_source_id",
        lambda _runtime: "authority-epoch-source:fixture",
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_execution_epoch_source_id",
        lambda _runtime: "execution-epoch-source:fixture",
    )

    bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        enable_artifact_perception=enable_artifact_perception,
    )
    return application, runtime, direct, bundle, captured_manifest_args


def test_no_model_contract_bundle_exposes_exact_motion_and_state_routes(
    monkeypatch,
) -> None:
    application, _runtime, _direct, bundle, captured_manifest_args = (
        _build_no_model_contract_bundle_fixture(monkeypatch)
    )
    motion_entry = bundle.route_manifest.route(
        "motion.move_tcp_to_pose",
        "1",
    )
    cardinal_step_entry = bundle.route_manifest.route(
        "motion.step_tcp_cardinal",
        "1",
    )
    state_entry = bundle.route_manifest.route("state.read_tcp_pose", "1")
    schema_entry = bundle.schema_catalog.entry(
        "motion.move_tcp_to_pose",
        "1",
    )
    parameters = {
        "pose_targets_by_component": {
            "right_arm": [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]
        },
        "reference_frame_id": bundle.motion_profile.reference_frame_id,
        "reference_frame_spec_hash": (
            bundle.motion_profile.reference_frame_spec_hash
        ),
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": bundle.motion_profile.frame_graph_hash,
        "tool_config_hash": "tool-config:fixture",
        "calibration_hash": "calibration:agent-motion-bundle-fixture",
    }
    cardinal_parameters = {
        "component_id": "right_arm",
        "direction": "up",
        "reference_frame_id": bundle.cardinal_step_profile.reference_frame_id,
        "reference_frame_spec_hash": (
            bundle.cardinal_step_profile.reference_frame_spec_hash
        ),
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": bundle.cardinal_step_profile.frame_graph_hash,
        "tool_config_hash": bundle.cardinal_step_profile.tool_config_hash,
        "calibration_hash": bundle.cardinal_step_profile.calibration_hash,
        "primitive_catalog_hash": (
            bundle.cardinal_step_profile.primitive_catalog_hash
        ),
    }

    assert bundle.route_manifest.current_mode == "sim_active"
    assert motion_entry.route.control_effect_class == "motion"
    assert motion_entry.route.privilege_tier == "t1_sim_or_no_motion"
    assert motion_entry.route.route_id == (
        "sim.mujoco.rm65.agent_skill_tcp_pose"
    )
    assert cardinal_step_entry.route.control_effect_class == "motion"
    assert cardinal_step_entry.route.route_id == (
        "sim.mujoco.rm65.agent_skill_tcp_cardinal_step"
    )
    assert bundle.cardinal_step_profile.reference_frame_id == (
        "mujoco_sim_hil/station"
    )
    assert bundle.cardinal_step_profile.execution_reference_frame_id == (
        "right_arm_base"
    )
    assert tuple(
        item.object_id
        for item in bundle.cardinal_step_deployment_binding.safety_envelope_refs
    ) == (MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1,)
    published_skill_ids = {
        entry.route.skill_id for entry in bundle.route_manifest.skill_routes
    }
    assert set(FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1) <= published_skill_ids
    assert "distance_m" not in bundle.schema_catalog.entry(
        "motion.step_tcp_cardinal",
        "1",
    ).parameter_schema["properties"]
    assert state_entry.route.control_effect_class == "read_only"
    assert "requested_duration_ms" not in schema_entry.parameter_schema[
        "properties"
    ]
    assert captured_manifest_args["motion_authority_sources"] == (
        "agent_skill:control-station-agent-skill-fixture",
    )
    assert bundle.station_manifest.source_slots[0].source_kind == "agent_skill"
    bundle.payload_validators.validate_parameters(
        route=motion_entry.route,
        payload=parameters,
    )
    bundle.payload_validators.validate_parameters(
        route=cardinal_step_entry.route,
        payload=cardinal_parameters,
    )
    for orientation_contract in bundle.fixed_orientation_contracts:
        orientation_profile = orientation_contract.profile
        orientation_route = bundle.route_manifest.route(
            orientation_profile.skill_id,
            "1",
        ).route
        assert orientation_route.route_id == (
            MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[
                orientation_profile.skill_id
            ]
        )
        assert orientation_profile.reference_frame_id == "mujoco_sim_hil/station"
        assert orientation_profile.execution_reference_frame_id == "right_arm_base"
        assert tuple(
            item.object_id
            for item in orientation_contract.deployment_binding.safety_envelope_refs
        ) == (MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ENVELOPE_ID_V1,)
        assert orientation_contract.motion_profile.max_translation_delta_m == (
            pytest.approx(1e-6)
        )
        assert orientation_contract.motion_profile.max_rotation_delta_rad == (
            pytest.approx(math.pi)
        )
        orientation_parameters = {
            "component_id": orientation_profile.component_id,
            "reference_frame_id": orientation_profile.reference_frame_id,
            "reference_frame_spec_hash": (
                orientation_profile.reference_frame_spec_hash
            ),
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": orientation_profile.frame_graph_hash,
            "tool_config_hash": orientation_profile.tool_config_hash,
            "calibration_hash": orientation_profile.calibration_hash,
            "primitive_hash": (
                orientation_profile.primitive.semantic_content_hash
            ),
        }
        bundle.payload_validators.validate_parameters(
            route=orientation_route,
            payload=orientation_parameters,
        )
        with pytest.raises(ValueError, match="exactly match"):
            bundle.payload_validators.validate_parameters(
                route=orientation_route,
                payload={**orientation_parameters, "yaw_deg": 30.0},
            )
    with pytest.raises(ValueError, match="unsupported direction"):
        bundle.payload_validators.validate_parameters(
            route=cardinal_step_entry.route,
            payload={**cardinal_parameters, "direction": "diagonal"},
        )
    with pytest.raises(ValueError, match="exactly match"):
        bundle.payload_validators.validate_parameters(
            route=motion_entry.route,
            payload={**parameters, "requested_duration_ms": 1_000},
        )

    motion_application = object.__new__(MujocoAgentSkillMotionApplication)
    motion_application.__dict__.update(application.__dict__)
    motion_application._lock = threading.RLock()
    motion_application._started = False
    motion_application._held = False
    motion_application._closed = False
    motion_application._active_operation_id = None
    motion_application.capture_passive_application_cut = lambda **_kwargs: None
    session = assemble_mujoco_agent_tcp_pose_gateway_session(
        application=motion_application,
        contract_bundle=bundle,
        allowed_client_ids=("rpent:fixture",),
        max_retained_operations=3,
    )

    assert session.service.domain_control_port is session.motion_executor
    route_to_port = {
        item.skill_id: item.execution_port
        for item in session.service.executor_registry.bindings
    }
    assert route_to_port["motion.move_tcp_to_pose"] is session.motion_executor
    assert route_to_port["motion.step_tcp_cardinal"] is (
        session.cardinal_step_executor
    )
    assert set(FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1) <= set(route_to_port)
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
        assert route_to_port[skill_id] is session.fixed_orientation_executors[
            skill_id
        ]
    assert route_to_port["state.read_tcp_pose"] is session.state_executor
    assert not motion_application.started

    close_outcome = session.close()
    assert not close_outcome.accepted
    assert close_outcome.reason_code == (
        "station_hold_not_confirmed_no_active_operation"
    )
    assert motion_application.closed


def test_depth_artifact_route_uses_passive_executor_and_artifact_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    application, _runtime, _direct, bundle, _captured_manifest_args = (
        _build_no_model_contract_bundle_fixture(
            monkeypatch,
            enable_artifact_perception=True,
        )
    )
    depth_entry = bundle.route_manifest.route(CAPTURE_DEPTH_SKILL_ID_V1, "1")
    assert depth_entry.route.control_effect_class == "read_only"
    assert depth_entry.route.privilege_tier == "t0_read_only"
    assert not depth_entry.route.may_move_robot
    assert not depth_entry.route.may_open_publication
    assert bundle.depth_capture is not None
    depth_payload = io.BytesIO()
    np.save(
        depth_payload,
        np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    depth_bytes = depth_payload.getvalue()

    motion_application = object.__new__(MujocoAgentSkillMotionApplication)
    motion_application.__dict__.update(application.__dict__)
    motion_application._lock = threading.RLock()
    motion_application._started = False
    motion_application._held = False
    motion_application._closed = False
    motion_application._active_operation_id = None
    motion_application.capture_passive_application_cut = lambda **_kwargs: None
    motion_application.capture_depth_npy_frame = lambda **_kwargs: (
        EncodedMujocoDepthFrame(
            view_id="front",
            captured_at=10.0,
            width=2,
            height=2,
            media_type="application/x-npy",
            schema_id="dexteleop.perception.depth_image_meters.v1",
            encoding="npy.float32.c_order.v1",
            compression="none",
            data=depth_bytes,
            camera_pose_world_xyz_quat_xyzw=(0, 0, 1, 0, 0, 0, 1),
            intrinsics={
                "width": 2,
                "height": 2,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 1.0,
                "cy": 1.0,
                "fovy_deg": 90.0,
            },
        )
    )
    rgb_bytes = b"bounded-rgb-jpeg"
    motion_application.capture_visual_jpeg_snapshot = lambda **_kwargs: (
        EncodedMujocoRgbSnapshot(
            captured_at=10.0,
            frames={
                "front": EncodedMujocoRgbFrame(
                    source_id="front",
                    width=32,
                    height=24,
                    media_type="image/jpeg",
                    data=rgb_bytes,
                )
            },
        )
    )
    artifact_store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    session = assemble_mujoco_agent_tcp_pose_gateway_session(
        application=motion_application,
        contract_bundle=bundle,
        allowed_client_ids=("rpent:fixture",),
        max_retained_operations=3,
        artifact_store=artifact_store,
        owns_artifact_store=True,
    )
    route_to_port = {
        item.skill_id: item.execution_port
        for item in session.service.executor_registry.bindings
    }
    assert route_to_port[CAPTURE_DEPTH_SKILL_ID_V1] is session.depth_executor
    assert route_to_port[CAPTURE_RGB_SKILL_ID_V1] is session.rgb_executor
    assert session.depth_executor is not session.motion_executor
    assert session.service.domain_control_port is session.motion_executor
    capabilities = session.service.protocol_runtime.session_capabilities
    assert capabilities.artifact_transfer_enabled
    assert capabilities.artifact_access_mechanisms == (
        "chunked_rpc",
        "local_path",
    )
    assert bundle.route_manifest.skill_io_catalog_hash == (
        bundle.io_catalog.semantic_content_hash
    )

    source_slot = bundle.station_manifest.source_slots[0]
    snapshot = AgentObservationSnapshot(
        snapshot_id="agent-observation:depth-fixture",
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        station_revision=0,
        captured_at=10.0,
        available_at=10.0,
        clock_domain_id=application.runtime.config.clock_domain_id,
        authority_epoch=0,
        execution_epoch=0,
        dispatch_owner_generation=1,
        lifecycle_state="configured",
        execution_gate="not_started",
        active_operation_id=None,
        source_binding_id=source_slot.source_slot_id,
        producer_epoch=application.runtime.config.direct_component_proposal_source.producer_epoch,
        target_binding_id=bundle.route_manifest.target_binding_id,
        execution_manifest_hash=bundle.route_manifest.execution_manifest_hash,
        frame_graph_hash=bundle.deployment_binding.frame_graph_hash,
        tool_config_hash=bundle.deployment_binding.tool_config_hash,
        calibration_hash=bundle.deployment_binding.calibration_hash,
        measurement_summary={"right_arm_tcp_pose_available": True},
        measurement_freshness_ms=0,
        measurement_valid=True,
    )
    assert session.service.publish_observation(snapshot=snapshot, now=10.0).accepted
    depth_route = depth_entry.route
    depth_binding = depth_entry.deployment_binding
    request = AgentSkillRequest(
        client_id="rpent:fixture",
        request_id="request:depth:1",
        idempotency_key="idempotency:depth:1",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=0,
        skill_id=depth_route.skill_id,
        skill_version=depth_route.skill_version,
        skill_definition_hash=depth_route.skill_definition_hash,
        route_id=depth_route.route_id,
        route_version=depth_route.route_version,
        route_hash=depth_route.semantic_content_hash,
        deployment_binding_hash=depth_binding.deployment_binding_hash,
        parameter_profile_id=(
            depth_route.allowed_parameter_profiles[0].parameter_profile_id
        ),
        parameter_profile_hash=(
            depth_route.allowed_parameter_profiles[0].parameter_profile_hash
        ),
        parameters={"role": "main"},
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="sim_active",
        requested_timeout_ms=1_000,
    )
    admitted = session.service.submit_request(
        request=request,
        current_station_revision=0,
        now=10.0,
    )
    assert admitted.accepted and admitted.operation is not None
    assert session.service.launch_operation(admitted.operation.operation_id).accepted
    assert session.depth_executor is not None
    assert session.depth_executor.wait(
        admitted.operation.operation_id,
        timeout_sec=1.0,
    )
    completed = session.service.operation_store.get(admitted.operation.operation_id)
    assert completed is not None and completed.status == "completed"
    assert completed.execution_result["role"] == "main"
    assert completed.execution_result["source_id"] == "front"
    assert len(completed.output_artifact_bindings) == 1
    output_binding = completed.output_artifact_bindings[0]
    assert (
        artifact_store.managed_read_path(
            output_binding.artifact.artifact_id
        ).read_bytes()
        == depth_bytes
    )
    events = session.service.protocol_runtime.replay_events(
        after_sequence=0,
        max_events=16,
        operations=(completed,),
    ).events
    artifact_events = tuple(
        event for event in events if event.event_kind == "artifact_ready"
    )
    assert len(artifact_events) == 1
    assert artifact_events[0].artifact_binding == output_binding
    assert tuple(
        receipt.operation_kind for receipt in session.depth_executor.port_receipts
    ) == ("activate",)
    assert session.motion_executor.workers == ()

    rgb_entry = bundle.route_manifest.route(CAPTURE_RGB_SKILL_ID_V1, "1")
    rgb_route = rgb_entry.route
    rgb_binding = rgb_entry.deployment_binding
    rgb_request = AgentSkillRequest(
        client_id="rpent:fixture",
        request_id="request:rgb:1",
        idempotency_key="idempotency:rgb:1",
        client_sequence=2,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=0,
        skill_id=rgb_route.skill_id,
        skill_version=rgb_route.skill_version,
        skill_definition_hash=rgb_route.skill_definition_hash,
        route_id=rgb_route.route_id,
        route_version=rgb_route.route_version,
        route_hash=rgb_route.semantic_content_hash,
        deployment_binding_hash=rgb_binding.deployment_binding_hash,
        parameter_profile_id=(
            rgb_route.allowed_parameter_profiles[0].parameter_profile_id
        ),
        parameter_profile_hash=(
            rgb_route.allowed_parameter_profiles[0].parameter_profile_hash
        ),
        parameters={
            "selected_roles": ["main"],
            "width": 32,
            "height": 24,
            "max_total_bytes": 1024,
        },
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="sim_active",
        requested_timeout_ms=1_000,
    )
    rgb_admitted = session.service.submit_request(
        request=rgb_request,
        current_station_revision=0,
        now=10.0,
    )
    assert rgb_admitted.accepted and rgb_admitted.operation is not None
    assert session.service.launch_operation(
        rgb_admitted.operation.operation_id
    ).accepted
    assert session.rgb_executor is not None
    assert session.rgb_executor.wait(
        rgb_admitted.operation.operation_id,
        timeout_sec=1.0,
    )
    rgb_completed = session.service.operation_store.get(
        rgb_admitted.operation.operation_id
    )
    assert rgb_completed is not None and rgb_completed.status == "completed"
    assert rgb_completed.execution_result["station_revision"] == 0
    assert rgb_completed.execution_result["frames"][0]["role"] == "main"
    assert len(rgb_completed.output_artifact_bindings) == 1
    rgb_output = rgb_completed.output_artifact_bindings[0]
    assert rgb_output.ordinal == 0
    assert (
        artifact_store.managed_read_path(rgb_output.artifact.artifact_id).read_bytes()
        == rgb_bytes
    )
    assert session.motion_executor.workers == ()

    session.close()
    assert session._artifact_store_closed


@pytest.mark.integration
@pytest.mark.optional
def test_installed_rpent_toolkit_consumes_exact_motion_and_state_routes(
    monkeypatch,
) -> None:
    client_module = pytest.importorskip(
        "rpent_dexteleop.gateway.client",
        reason="paired RPent integration wheel is not installed",
    )
    toolkit_module = pytest.importorskip(
        "rpent_dexteleop.toolkit",
        reason="paired RPent integration wheel is not installed",
    )
    session_module = pytest.importorskip(
        "rpent_dexteleop.gateway.session",
        reason="paired RPent integration wheel is not installed",
    )
    InMemoryDexTeleopGatewayClient = (
        client_module.InMemoryDexTeleopGatewayClient
    )
    DexTeleopToolkit = toolkit_module.DexTeleopToolkit
    DexTeleopConsumerSession = session_module.DexTeleopConsumerSession
    _application, runtime, direct, bundle, _captured_manifest_args = (
        _build_no_model_contract_bundle_fixture(monkeypatch)
    )
    motion_entry = bundle.route_manifest.route(
        "motion.move_tcp_to_pose",
        "1",
    )
    cardinal_entry = bundle.route_manifest.route(
        "motion.step_tcp_cardinal",
        "1",
    )
    parameters = {
        "pose_targets_by_component": {
            "right_arm": [0.4, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]
        },
        "reference_frame_id": bundle.motion_profile.reference_frame_id,
        "reference_frame_spec_hash": (
            bundle.motion_profile.reference_frame_spec_hash
        ),
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": bundle.motion_profile.frame_graph_hash,
        "tool_config_hash": "tool-config:fixture",
        "calibration_hash": "calibration:agent-motion-bundle-fixture",
    }
    cardinal_parameters = {
        "component_id": bundle.cardinal_step_profile.component_id,
        "direction": "forward",
        "reference_frame_id": bundle.cardinal_step_profile.reference_frame_id,
        "reference_frame_spec_hash": (
            bundle.cardinal_step_profile.reference_frame_spec_hash
        ),
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": bundle.cardinal_step_profile.frame_graph_hash,
        "tool_config_hash": bundle.cardinal_step_profile.tool_config_hash,
        "calibration_hash": bundle.cardinal_step_profile.calibration_hash,
        "primitive_catalog_hash": (
            bundle.cardinal_step_profile.primitive_catalog_hash
        ),
    }

    source_slot = bundle.station_manifest.source_slots[0]
    snapshot = AgentObservationSnapshot(
        snapshot_id="agent-observation:agent-motion-bundle-fixture",
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        station_revision=1,
        captured_at=10.0,
        available_at=10.0,
        clock_domain_id=runtime.config.clock_domain_id,
        authority_epoch=0,
        execution_epoch=0,
        dispatch_owner_generation=1,
        lifecycle_state="configured",
        execution_gate="not_started",
        active_operation_id=None,
        source_binding_id=source_slot.source_slot_id,
        producer_epoch=direct.producer_epoch,
        target_binding_id=bundle.route_manifest.target_binding_id,
        execution_manifest_hash=bundle.route_manifest.execution_manifest_hash,
        frame_graph_hash=bundle.deployment_binding.frame_graph_hash,
        tool_config_hash=bundle.deployment_binding.tool_config_hash,
        calibration_hash=bundle.deployment_binding.calibration_hash,
        measurement_summary={"right_arm_tcp_pose_available": True},
        measurement_freshness_ms=0,
        measurement_valid=True,
    )
    client = InMemoryDexTeleopGatewayClient(
        definition_catalog=bundle.definition_catalog.to_wire(),
        schema_catalog=bundle.schema_catalog.to_wire(),
        route_manifest=bundle.route_manifest.to_wire(),
        observation_snapshot=snapshot.to_wire(),
    )
    consumer_session = DexTeleopConsumerSession(
        client=client,
        client_id="rpent:fixture",
        request_id_factory=lambda: "request:rpent-agent-motion-fixture",
    )
    toolkit = DexTeleopToolkit(session=consumer_session)
    motion_spec = next(
        spec
        for spec in toolkit.get_tools_spec()
        if spec["name"].startswith("dex__motion_move_tcp_to_pose")
    )
    cardinal_spec = next(
        spec
        for spec in toolkit.get_tools_spec()
        if spec["name"].startswith("dex__motion_step_tcp_cardinal")
    )

    assert set(motion_spec["input_schema"]["properties"]) == set(parameters)
    assert "requested_duration_ms" not in motion_spec["input_schema"][
        "properties"
    ]
    response = toolkit.execute_tool(
        motion_spec["name"],
        parameters,
    ).result
    assert response["accepted"]
    assert client.submitted_requests[0]["route_hash"] == (
        motion_entry.route.semantic_content_hash
    )
    assert set(cardinal_spec["input_schema"]["properties"]) == set(
        cardinal_parameters
    )
    assert "distance_m" not in cardinal_spec["input_schema"]["properties"]
    cardinal_response = toolkit.execute_tool(
        cardinal_spec["name"],
        cardinal_parameters,
    ).result
    assert cardinal_response["accepted"]
    assert client.submitted_requests[1]["route_hash"] == (
        cardinal_entry.route.semantic_content_hash
    )
    consumer_session.close()

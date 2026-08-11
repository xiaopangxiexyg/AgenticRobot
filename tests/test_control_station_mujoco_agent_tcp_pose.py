from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import control_station.sim_hil_profile.agent_motion_deployment as deployment
from control_station.agent_gateway import (
    AgentObservationSnapshot,
    AgentSkillOperation,
    AgentSkillRequest,
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
    SkillExecutionRoute,
    SkillParameterProfileRef,
)
from control_station.sim_hil_profile.agent_motion_application import (
    MujocoAgentSkillMotionApplication,
    MujocoAgentSkillMotionStep,
)
from control_station.sim_hil_profile.agent_motion_deployment import (
    MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1,
    MujocoAgentTcpPoseDeploymentBinding,
    MujocoAgentTcpPoseMotionProfile,
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from control_station.sim_hil_profile.agent_motion_factory import (
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
    validate_mujoco_agent_tcp_pose_target_config,
)
from control_station.sim_hil_profile.agent_motion_gateway_session import (
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from control_station.sim_hil_profile.agent_motion_skill import (
    MujocoAgentTcpPoseSkillLifecycle,
    mujoco_agent_skill_hold_evidence_from_transition,
)
from sim.mujoco.deploy import (
    MujocoDirectComponentProposalSource,
    MujocoSimHILTargetConfig,
    build_mujoco_pose_component_target_profile,
)


def _profile_and_schema():
    target_profile = build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id="fixture.agent_tcp_pose.right_arm.v1",
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms"},
    )
    motion_profile = MujocoAgentTcpPoseMotionProfile(
        reference_frame_id="right_arm_base",
        reference_frame_spec_hash="frame-spec:right-arm-base",
        frame_graph_hash="frame-graph:fixture",
        tool_config_hash="tool-config:fixture",
        calibration_hash="calibration:fixture",
        stable_measurement_count=2,
        max_control_cycles=4,
        physics_ticks_per_control_cycle=3,
    )
    return motion_profile, target_profile.action_schema


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


def _request_and_operation(route, binding, motion_profile, target_pose):
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
            "pose_targets_by_component": {"right_arm": list(target_pose)},
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


def _fake_application(action_schema, *, start_pose, target_pose):
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
        return {"right_arm": poses.pop(0)}

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


def test_no_model_contract_bundle_exposes_exact_motion_and_state_routes(
    monkeypatch,
) -> None:
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
        core=SimpleNamespace(
            config=SimpleNamespace(
                active_selection_groups=("arms",),
                authority_policy=SimpleNamespace(
                    semantic_content_hash="authority-policy:fixture"
                ),
                decision_schedule=SimpleNamespace(
                    semantic_content_hash="decision-schedule:fixture"
                ),
            )
        ),
        clock=lambda: 10.0,
    )
    application = SimpleNamespace(
        runtime=runtime,
        task_config=SimpleNamespace(physics_hz=1_000.0),
        calibration_hash="calibration:agent-motion-bundle-fixture",
        target_frame_profile=SimpleNamespace(
            reference_frame_by_component={"right_arm": "right_arm_base"},
            frame_graph=SimpleNamespace(
                semantic_content_hash="frame-graph:fixture",
                frame_by_id={
                    "right_arm_base": SimpleNamespace(
                        semantic_content_hash="frame-spec:right-arm-base"
                    )
                },
            ),
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash="tool-config:fixture",
            binding_by_component={
                "right_arm": SimpleNamespace(controlled_frame_id="tool_tcp")
            },
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
        application=application
    )
    motion_entry = bundle.route_manifest.route(
        "motion.move_tcp_to_pose",
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
        "reference_frame_id": "right_arm_base",
        "reference_frame_spec_hash": "frame-spec:right-arm-base",
        "controlled_frame_id": "tool_tcp",
        "frame_graph_hash": "frame-graph:fixture",
        "tool_config_hash": "tool-config:fixture",
        "calibration_hash": "calibration:agent-motion-bundle-fixture",
    }

    assert bundle.route_manifest.current_mode == "sim_active"
    assert motion_entry.route.control_effect_class == "motion"
    assert motion_entry.route.privilege_tier == "t1_sim_or_no_motion"
    assert motion_entry.route.route_id == (
        "sim.mujoco.rm65.agent_skill_tcp_pose"
    )
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
    assert route_to_port["state.read_tcp_pose"] is session.state_executor
    assert not motion_application.started

    close_outcome = session.close()
    assert not close_outcome.accepted
    assert close_outcome.reason_code == (
        "station_hold_not_confirmed_no_active_operation"
    )
    assert motion_application.closed

    rpent_root = (
        Path(__file__).resolve().parents[2] / "HierachicalVLA" / "RPent"
    )
    assert (rpent_root / "robots" / "dexteleop" / "toolkit.py").is_file()
    sys.path.insert(0, str(rpent_root))
    from robots.dexteleop.client import InMemoryDexTeleopGatewayClient
    from robots.dexteleop.toolkit import DexTeleopToolkit

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
    toolkit = DexTeleopToolkit(
        client=client,
        client_id="rpent:fixture",
        request_id_factory=lambda: "request:rpent-agent-motion-fixture",
    )
    motion_spec = next(
        spec
        for spec in toolkit.get_tools_spec()
        if spec["name"].startswith("dex__motion_move_tcp_to_pose")
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

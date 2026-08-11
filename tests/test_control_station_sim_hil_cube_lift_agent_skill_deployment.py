from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import control_station.sim_hil_profile.agent_skill_deployment as deployment
from control_station.agent_gateway import (
    AgentGatewayService,
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
    AgentOwnOperationQuery,
    AgentSkillAccessPolicy,
    AgentSkillRequest,
    AgentStationRevisionTracker,
    BoundAgentGatewayWireAdapter,
    DeterministicNoIoAgentSkillOperationPort,
    NullAgentSkillOperationPort,
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    validate_skill_route_catalog_closure,
)
from control_station.composition.mujoco_sim_hil import MujocoSimHILComposition
from control_station.sim_hil_profile.agent_gateway_session import (
    assemble_mujoco_cube_lift_agent_gateway_session,
)
from control_station.sim_hil_profile.agent_observation import (
    MUJOCO_CUBE_LIFT_AGENT_TCP_SUMMARY_SCHEMA_V1,
    MujocoCubeLiftAgentObservationProducer,
)
from control_station.sim_hil_profile.agent_skill_deployment import (
    build_mujoco_cube_lift_agent_skill_contract_bundle,
    cube_lift_terminal_evaluator_config_from_task,
)
from control_station.sim_hil_profile.application import (
    MujocoSimHILApplication,
    MujocoSimHILPassiveApplicationCut,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1
from robot_motion.runtime.control_admission import ControlCut

RUNTIME_SESSION_ID = "sim-hil:agent-skill-deployment-test:1"
CONTROL_DOMAIN_ID = "mujoco-sim-hil"
TASK_HASH = "task-hash:cube-lift-deployment-test"


def _hashed(value: str) -> SimpleNamespace:
    return SimpleNamespace(semantic_content_hash=value)


def _composition() -> MujocoSimHILComposition:
    observation_plan = SimpleNamespace(
        enabled=True,
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={"right_arm": "cartesian_pose"},
        semantic_content_hash="observation-plan:test",
    )
    runtime_config = SimpleNamespace(
        runtime_session_id=RUNTIME_SESSION_ID,
        control_domain_id=CONTROL_DOMAIN_ID,
        embodiment_instance_id="mujoco-rm65-sim-hil",
        producer_epoch=1,
        composer_producer_id="control-station-sim-hil-composer",
        enable_composer_authority=True,
        enable_primary_policy_authority=False,
        enable_replay_authority=False,
        clock_domain_id="mujoco.sim-hil.test-clock",
        pose_frame_contract_hash="pose-frame-contract:test",
        execution_observation_plan=observation_plan,
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        catalog=SimpleNamespace(
            action_space=BIMANUAL_EE_POSE_V1,
            semantic_content_hash="catalog:test",
        ),
        action_schema=_hashed("action-schema:test"),
        component_target_capability=_hashed("target-capability:test"),
        target_runtime_action_contract=_hashed("target-action-contract:test"),
        execution_capability=_hashed("execution-capability:test"),
        execution_observation_binding=SimpleNamespace(
            binding_id="observation-binding-id:test",
            plan_hash=observation_plan.semantic_content_hash,
            semantic_content_hash="observation-binding:test",
        ),
        execution_observation_capability=SimpleNamespace(
            semantic_content_hash="observation-capability:test",
            representation_by_component={"right_arm": "cartesian_pose"},
            schema_hash_by_component={
                "right_arm": "cartesian-pose-schema:test"
            },
        ),
        reference_activation_spec=SimpleNamespace(
            activation_mode="immediate_target"
        ),
        core=SimpleNamespace(
            config=SimpleNamespace(
                active_selection_groups=("left_arm", "right_arm"),
                authority_policy=_hashed("authority-policy:test"),
                decision_schedule=_hashed("decision-schedule:test"),
            ),
            coordinator=SimpleNamespace(capture_cut=lambda **_kwargs: None),
        ),
        clock=lambda: 10.0,
    )
    policy_binding = SimpleNamespace(
        source_slot_id="sim-hil-temp-ppo",
        producer_id="sim-hil-temp-ppo",
        producer_epoch=1,
        semantic_content_hash="policy-binding:test",
    )
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = SimpleNamespace(
        initial_human_alpha=0.0,
        human_producer_id="human-passive",
        human_producer_epoch=1,
    )
    composition.policy_input_adapter = SimpleNamespace(binding=policy_binding)
    composition.human_source = SimpleNamespace(
        binding_receipt=SimpleNamespace(
            rig_absent=True,
            authoritative=False,
        )
    )
    composition.human_source_spec = SimpleNamespace(
        source_id="passive-teleop-components",
        semantic_content_hash="human-source-spec:test",
    )
    composition.human_adapter = SimpleNamespace(
        contract_hash="human-adapter:test"
    )
    composition.contracts = SimpleNamespace(
        semantic_content_hash="composition-contracts:test",
        composition_spec=_hashed("composition-spec:test"),
        authority_gate_spec=_hashed("authority-gate:test"),
        strategy_spec=_hashed("strategy-spec:test"),
    )
    return composition


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        source_sha256=TASK_HASH,
        object_body_name="debug_cube",
        grasp_latch_left_finger_body_name="right_gripper_left_finger",
        grasp_latch_right_finger_body_name="right_gripper_right_finger",
        table_body_name="table",
        lift_delta_m=0.04,
        success_hold_physics_ticks=20,
        physics_hz=1000.0,
    )


def _application() -> SimpleNamespace:
    composition = _composition()
    task = _task()
    return SimpleNamespace(
        config=SimpleNamespace(
            runtime_session_id=RUNTIME_SESSION_ID,
            control_domain_id=CONTROL_DOMAIN_ID,
            initial_human_alpha=0.0,
            defer_human_input=True,
            realtime_worker=False,
            mujoco_viewer_enabled=False,
            execution_difference_enabled=True,
        ),
        runtime=composition.runtime,
        composition=composition,
        task_config=task,
        policy_source=SimpleNamespace(
            task_config=task,
            checkpoint_path="checkpoint:test.pt",
            compatibility={
                "task_config_mismatch": False,
                "runtime_task_config_sha256": TASK_HASH,
                "checkpoint_task_config_sha256": TASK_HASH,
            },
        ),
        execution_difference_observer=SimpleNamespace(),
        target_frame_profile=SimpleNamespace(
            semantic_content_hash="target-frame-profile:test",
            reference_frame_by_component={"right_arm": "right_arm_base"},
            frame_graph=SimpleNamespace(
                semantic_content_hash="frame-graph:test",
                frame_by_id={
                    "right_arm_base": _hashed("right-arm-base-spec:test")
                },
            ),
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash="tool-frame-profile:test",
            target_frame_profile_hash="target-frame-profile:test",
            binding_by_component={
                "right_arm": SimpleNamespace(controlled_frame_id="tool_tcp")
            },
        ),
    )


def _execution_manifest(application):
    runtime = application.runtime
    return SimpleNamespace(
        semantic_content_hash="execution-manifest:test",
        action_schema_hash=runtime.action_schema.semantic_content_hash,
        component_target_capability_hash=(
            runtime.component_target_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
    )


def test_contract_bundle_uses_only_exact_lower_graph_identities(monkeypatch) -> None:
    application = _application()
    evaluator = cube_lift_terminal_evaluator_config_from_task(
        application.task_config
    )
    monkeypatch.setattr(
        deployment,
        "build_mujoco_sim_hil_agent_skill_execution_manifest",
        lambda composition, station: _execution_manifest(application),
    )
    checkpoint_paths: list[str] = []

    def checkpoint_hash(path) -> str:
        checkpoint_paths.append(str(path))
        return "checkpoint-hash:test"

    bundle = build_mujoco_cube_lift_agent_skill_contract_bundle(
        application=application,
        evaluator_config=evaluator,
        checkpoint_hash_provider=checkpoint_hash,
    )

    validate_skill_route_catalog_closure(
        definition_catalog=bundle.definition_catalog,
        schema_catalog=bundle.schema_catalog,
        route_manifest=bundle.route_manifest,
    )
    binding = bundle.deployment_binding
    assert checkpoint_paths == ["checkpoint:test.pt"]
    assert binding.runtime_session_id == RUNTIME_SESSION_ID
    assert binding.control_domain_id == CONTROL_DOMAIN_ID
    assert binding.task_config_hash == TASK_HASH
    assert binding.policy_checkpoint_hash == "checkpoint-hash:test"
    assert binding.observation_plan_hash == "observation-plan:test"
    assert binding.frame_graph_hash == "frame-graph:test"
    assert binding.tool_config_hash == "tool-frame-profile:test"
    assert binding.execution_domain_binding_hash == (
        bundle.station_manifest.execution_binding.semantic_content_hash
    )
    assert binding.station_manifest_hash == (
        bundle.station_manifest.semantic_content_hash
    )
    assert binding.execution_manifest_hash == "execution-manifest:test"
    assert bundle.route_manifest.route(
        "task.cube_lift", "1"
    ).evaluation_plan_hash == (
        bundle.evaluation_plan.semantic_content_hash
    )
    assert "agent_estop" in bundle.route_manifest.explicit_exclusions
    assert "raw_action_submission" in bundle.route_manifest.explicit_exclusions


def test_task_projection_reuses_existing_hold_duration_unless_explicitly_changed() -> None:
    task = _task()
    inherited = cube_lift_terminal_evaluator_config_from_task(task)
    overridden = cube_lift_terminal_evaluator_config_from_task(
        task,
        stable_hold_physics_ticks=100,
    )
    assert inherited.stable_hold_physics_ticks == 20
    assert overridden.stable_hold_physics_ticks == 100
    assert inherited.task_config_hash == overridden.task_config_hash == TASK_HASH


def test_contract_bundle_rejects_unbound_or_non_tcp_observation_plan(
    monkeypatch,
) -> None:
    application = _application()
    evaluator = cube_lift_terminal_evaluator_config_from_task(
        application.task_config
    )
    application.runtime.execution_observation_binding.plan_hash = "another-plan"
    monkeypatch.setattr(
        deployment,
        "build_mujoco_sim_hil_agent_skill_execution_manifest",
        lambda composition, station: _execution_manifest(application),
    )
    with pytest.raises(ValueError, match="did not bind the exact observation plan"):
        build_mujoco_cube_lift_agent_skill_contract_bundle(
            application=application,
            evaluator_config=evaluator,
            checkpoint_hash_provider=lambda _: "checkpoint-hash:test",
        )


def _contract_bundle(monkeypatch, application):
    evaluator = cube_lift_terminal_evaluator_config_from_task(
        application.task_config
    )
    monkeypatch.setattr(
        deployment,
        "build_mujoco_sim_hil_agent_skill_execution_manifest",
        lambda composition, station: _execution_manifest(application),
    )
    return build_mujoco_cube_lift_agent_skill_contract_bundle(
        application=application,
        evaluator_config=evaluator,
        checkpoint_hash_provider=lambda _: "checkpoint-hash:test",
    )


def test_station_revision_tracker_changes_only_for_control_fingerprint() -> None:
    tracker = AgentStationRevisionTracker(initial_revision=7)
    first = tracker.observe({"gate": "ready", "operation_id": None})
    repeated = tracker.observe({"operation_id": None, "gate": "ready"})
    changed = tracker.observe({"gate": "hold", "operation_id": None})

    assert first.station_revision == repeated.station_revision == 7
    assert not first.changed
    assert not repeated.changed
    assert changed.changed
    assert changed.station_revision == 8


def test_tcp_snapshot_pose_changes_do_not_advance_station_revision(
    monkeypatch,
) -> None:
    application = _application()
    bundle = _contract_bundle(monkeypatch, application)
    state = {
        "time": 10.0,
        "authority_epoch": 2,
        "execution_epoch": 3,
        "owner_generation": 4,
        "lifecycle": "configured",
        "gate": "not_started",
        "gate_epoch": 0,
        "pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        "closed": False,
    }
    application.runtime.clock = lambda: state["time"]

    def capture_cut(*, scene_projector):
        del scene_projector
        return MujocoSimHILPassiveApplicationCut(
            captured_at=state["time"],
            authority_epoch=state["authority_epoch"],
            execution_epoch=state["execution_epoch"],
            dispatch_owner_generation=state["owner_generation"],
            lifecycle_state=state["lifecycle"],
            execution_gate=state["gate"],
            execution_gate_epoch=state["gate_epoch"],
            fault_class=None,
            fault_rearmable=False,
            scene_projection=(
                None
                if state["closed"]
                else {
                    "finite_state": True,
                    "pose_xyz_quat_xyzw_by_component": {
                        "right_arm": list(state["pose"])
                    },
                }
            ),
        )

    application.capture_passive_application_cut = capture_cut
    producer = MujocoCubeLiftAgentObservationProducer(
        application=application,
        contract_bundle=bundle,
    )

    first = producer.capture(active_operation_id=None)
    state["time"] = 10.1
    state["pose"][0] = 0.15
    moved = producer.capture(active_operation_id=None)

    assert first.station_revision == moved.station_revision == 0
    assert first.snapshot_id != moved.snapshot_id
    assert moved.measurement_valid
    assert moved.measurement_summary["schema"] == (
        MUJOCO_CUBE_LIFT_AGENT_TCP_SUMMARY_SCHEMA_V1
    )
    assert moved.measurement_summary[
        "pose_xyz_quat_xyzw_by_component"
    ]["right_arm"][0] == pytest.approx(0.15)
    assert "qpos" not in repr(moved.measurement_summary).lower()
    assert moved.frame_graph_hash == bundle.deployment_binding.frame_graph_hash
    assert moved.tool_config_hash == bundle.deployment_binding.tool_config_hash
    assert moved.calibration_hash == bundle.deployment_binding.calibration_hash

    state.update(
        time=10.2,
        lifecycle="held",
        gate="hold",
        gate_epoch=1,
    )
    held = producer.capture(active_operation_id="operation:test")
    assert held.station_revision == 1
    assert held.active_operation_id == "operation:test"
    assert held.execution_gate == "hold"

    state.update(time=10.25, pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    invalid_pose = producer.capture(active_operation_id="operation:test")
    assert invalid_pose.station_revision == 1
    assert not invalid_pose.measurement_valid
    assert invalid_pose.measurement_summary[
        "pose_xyz_quat_xyzw_by_component"
    ] == {}

    state.update(time=10.3, lifecycle="closed", gate="closed", closed=True)
    closed = producer.capture(active_operation_id=None)
    assert closed.station_revision == 2
    assert not closed.measurement_valid
    assert closed.measurement_summary["status"] == "invalid"


def test_application_passive_cut_is_no_start_no_step_and_uses_existing_owner() -> None:
    application = object.__new__(MujocoSimHILApplication)
    application._lock = threading.RLock()
    application._started = False
    application._held = False
    application._episode_ended = False
    application._restart_required = False
    application._closed = False
    application._first_control_fault = None
    calls = {"scene": 0, "gate": 0}

    def capture_scene(projector):
        calls["scene"] += 1
        return projector("scene:test")

    runtime = SimpleNamespace(
        clock=lambda: 12.0,
        config=SimpleNamespace(
            runtime_session_id=RUNTIME_SESSION_ID,
            control_domain_id=CONTROL_DOMAIN_ID,
        ),
        core=SimpleNamespace(
            capture_receiver_state=lambda: SimpleNamespace(
                runtime_session_id=RUNTIME_SESSION_ID,
                control_domain_id=CONTROL_DOMAIN_ID,
                current_owner_generation=6,
                current_authority_epoch=7,
                current_execution_epoch=8,
            )
        ),
        rig=SimpleNamespace(capture_scene=capture_scene),
    )

    def capture_gate(*, now):
        calls["gate"] += 1
        assert now == 12.0
        return SimpleNamespace(status="hold", gate_epoch=9)

    runtime.capture_execution_gate_snapshot = capture_gate
    application.runtime = runtime

    configured = application.capture_passive_application_cut(
        scene_projector=lambda scene: {"scene": scene}
    )
    assert configured.lifecycle_state == "configured"
    assert configured.execution_gate == "not_started"
    assert configured.dispatch_owner_generation == 6
    assert configured.scene_projection == {"scene": "scene:test"}
    assert calls == {"scene": 1, "gate": 0}

    application._started = True
    application._held = True
    held = application.capture_passive_application_cut(
        scene_projector=lambda scene: {"scene": scene}
    )
    assert held.lifecycle_state == "held"
    assert held.execution_gate == "hold"
    assert held.execution_gate_epoch == 9
    assert calls == {"scene": 2, "gate": 1}


def test_exact_bundle_wire_facade_admits_rpent_request_without_transport(
    monkeypatch,
) -> None:
    application = _application()
    now = {"value": 20.0}
    application.runtime.clock = lambda: now["value"]
    application.capture_passive_application_cut = lambda *, scene_projector: (
        MujocoSimHILPassiveApplicationCut(
            captured_at=now["value"],
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="configured",
            execution_gate="not_started",
            execution_gate_epoch=0,
            fault_class=None,
            fault_rearmable=False,
            scene_projection={
                "finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {
                    "right_arm": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
                },
            },
        )
    )
    bundle = _contract_bundle(monkeypatch, application)
    policy = AgentSkillAccessPolicy(
        access_policy_id="agent-access:cube-lift-test",
        allowed_client_ids=("rpent:b3",),
        allowed_modes=("sim_active",),
        max_request_timeout_ms=30_000,
        max_grant_lifetime_sec=30.0,
    )
    service = AgentGatewayService(
        manifest=bundle.route_manifest,
        access_policy=policy,
        operation_port=NullAgentSkillOperationPort(clock=application.runtime.clock),
        payload_validators=bundle.payload_validators,
        clock=application.runtime.clock,
        clock_domain_id=application.runtime.config.clock_domain_id,
    )
    producer = MujocoCubeLiftAgentObservationProducer(
        application=application,
        contract_bundle=bundle,
    )
    facade = BoundAgentGatewayWireAdapter(
        authenticated_client_id="rpent:b3",
        definition_catalog=bundle.definition_catalog,
        schema_catalog=bundle.schema_catalog,
        service=service,
        publish_observation=lambda: producer.capture_and_publish(service),
        current_station_revision=lambda: producer.current_station_revision,
    )

    manifest = facade.get_skill_route_manifest()
    assert set(manifest["supported_transport_features"]) >= {
        "own_domain_hold_v1",
        "own_operation_cancel_v1",
        "own_operation_status_v1",
    }

    rpent_root = (
        Path(__file__).resolve().parents[2] / "HierachicalVLA" / "RPent"
    )
    if not (rpent_root / "robots" / "dexteleop" / "toolkit.py").is_file():
        pytest.skip("Live RPent checkout is unavailable for the optional loopback.")
    sys.path.insert(0, str(rpent_root))
    from robots.dexteleop.toolkit import (
        DEX_CONTROL_TOOL_CANCEL_OWN_OPERATION,
        DEX_CONTROL_TOOL_GET_OWN_OPERATION,
        DEX_CONTROL_TOOL_REQUEST_HOLD_OWN_DOMAIN,
        DexTeleopToolExposurePolicy,
        DexTeleopToolkit,
    )

    control_tools = frozenset(
        {
            DEX_CONTROL_TOOL_CANCEL_OWN_OPERATION,
            DEX_CONTROL_TOOL_GET_OWN_OPERATION,
            DEX_CONTROL_TOOL_REQUEST_HOLD_OWN_DOMAIN,
        }
    )
    toolkit = DexTeleopToolkit(
        client=facade,
        client_id="rpent:b3",
        exposure_policy=DexTeleopToolExposurePolicy(
            allowed_control_tools=control_tools
        ),
        request_id_factory=lambda: "request:cube-lift:1",
    )
    specs = {item["name"]: item for item in toolkit.get_tools_spec()}
    assert control_tools <= set(specs)
    route_tools = set(specs) - set(control_tools)
    assert len(route_tools) == 2
    motion_tool = next(
        name for name in route_tools if "task_cube_lift" in name
    )
    assert specs[motion_tool]["input_schema"]["properties"] == {}
    assert "read_text_file" not in specs

    response = toolkit.execute_tool(motion_tool, {}).result
    assert response["accepted"]
    assert response["operation"]["status"] == "admitted"
    assert service.operation_store.active_operation_id == (
        response["operation"]["operation_id"]
    )
    operation_id = response["operation"]["operation_id"]
    request = service.operation_store.request_for_operation(operation_id)
    assert request is not None
    assert request.parameters == {}
    assert response["rpent_contract_binding"]["route_hash"] == (
        bundle.route_manifest.route(
            "task.cube_lift", "1"
        ).route.semantic_content_hash
    )

    forged = request.to_wire()
    forged["client_id"] = "another-client"
    with pytest.raises(PermissionError, match="authenticated connection"):
        facade.submit_skill_request(forged)


def test_live_rpent_toolkit_uses_authenticated_uds_disconnect_control_path(
    monkeypatch,
) -> None:
    application = _application()
    now = {"value": 25.0}
    application.runtime.clock = lambda: now["value"]
    application.capture_passive_application_cut = lambda *, scene_projector: (
        MujocoSimHILPassiveApplicationCut(
            captured_at=now["value"],
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="configured",
            execution_gate="not_started",
            execution_gate_epoch=0,
            fault_class=None,
            fault_rearmable=False,
            scene_projection={
                "finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {
                    "right_arm": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
                },
            },
        )
    )
    bundle = _contract_bundle(monkeypatch, application)
    service = AgentGatewayService(
        manifest=bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="agent-access:cube-lift-uds-test",
            allowed_client_ids=("rpent:b3",),
            allowed_modes=("sim_active",),
            max_request_timeout_ms=30_000,
            max_grant_lifetime_sec=30.0,
        ),
        operation_port=DeterministicNoIoAgentSkillOperationPort(
            succeeds_by_kind={"hold": True},
            clock=application.runtime.clock,
        ),
        payload_validators=bundle.payload_validators,
        clock=application.runtime.clock,
        clock_domain_id=application.runtime.config.clock_domain_id,
    )
    producer = MujocoCubeLiftAgentObservationProducer(
        application=application,
        contract_bundle=bundle,
    )

    def facade_factory(client_id):
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=client_id,
            definition_catalog=bundle.definition_catalog,
            schema_catalog=bundle.schema_catalog,
            service=service,
            publish_observation=lambda: producer.capture_and_publish(service),
            current_station_revision=lambda: producer.current_station_revision,
        )

    socket_root = Path(__file__).resolve().parents[1] / "temp" / "u"
    socket_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(socket_root, 0o700)
    socket_path = socket_root / (
        f"b3-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    )
    server = AgentGatewayUnixJsonServer(
        socket_path=socket_path,
        peer_client_map=AgentGatewayUnixPeerClientMap(
            {os.getuid(): "rpent:b3"}
        ),
        facade_factory=facade_factory,
        close_gateway=lambda: service.close(
            current_station_revision=producer.current_station_revision
        ),
        peer_idle_timeout_sec=1.0,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = None
    operation_id = None
    try:
        rpent_root = (
            Path(__file__).resolve().parents[2] / "HierachicalVLA" / "RPent"
        )
        if not (
            rpent_root / "robots" / "dexteleop" / "unix_client.py"
        ).is_file():
            pytest.skip(
                "Live RPent checkout is unavailable for the optional UDS loopback."
            )
        sys.path.insert(0, str(rpent_root))
        from robots.dexteleop.toolkit import DexTeleopToolkit
        from robots.dexteleop.unix_client import UnixDexTeleopGatewayClient

        client = UnixDexTeleopGatewayClient(
            socket_path=socket_path,
            client_id="rpent:b3",
            expected_server_uid=os.getuid(),
            expected_server_gid=os.getgid(),
            heartbeat_interval_sec=0.1,
        )
        toolkit = DexTeleopToolkit(
            client=client,
            client_id="rpent:b3",
            request_id_factory=lambda: "request:cube-lift:uds:1",
        )
        specs = {item["name"]: item for item in toolkit.get_tools_spec()}
        assert len(specs) == 2
        tool_name = next(
            name for name in specs if "task_cube_lift" in name
        )
        assert specs[tool_name]["input_schema"]["properties"] == {}
        admitted = toolkit.execute_tool(tool_name, {}).result
        assert admitted["accepted"]
        operation_id = admitted["operation"]["operation_id"]
        assert service.get_operation(operation_id).status == "admitted"

        client.close()
        client = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            operation = service.get_operation(operation_id)
            if operation is not None and operation.terminal:
                break
            time.sleep(0.01)
        operation = service.get_operation(operation_id)
        assert operation is not None
        assert operation.status == "cancelled", (
            operation,
            server.active_client_ids,
            server.disconnect_events,
        )
        assert operation.reason_code == (
            "operation_cancelled_after_fake_no_io_disconnect_control"
        )
        assert server.disconnect_events[-1].outcome_accepted
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
    assert operation_id is not None
    assert not socket_path.exists()


def test_exact_gateway_session_launches_one_fail_closed_station_worker(
    monkeypatch,
) -> None:
    application = _application()
    now = {"value": 30.0}
    application.runtime.clock = lambda: now["value"]
    application.capture_passive_application_cut = lambda *, scene_projector: (
        MujocoSimHILPassiveApplicationCut(
            captured_at=now["value"],
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="configured",
            execution_gate="not_started",
            execution_gate_epoch=0,
            fault_class=None,
            fault_rearmable=False,
            scene_projection={
                "finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {
                    "right_arm": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
                },
            },
        )
    )
    bundle = _contract_bundle(monkeypatch, application)
    route_entry = bundle.route_manifest.route("task.cube_lift", "1")
    route = route_entry.route
    profile = route.allowed_parameter_profiles[0]

    class FailClosedLifecycle:
        executor_id = bundle.deployment_binding.executor_id

        def __init__(self):
            self.application = application
            self.route = route
            self.deployment_binding = bundle.deployment_binding
            self.evaluation_plan = bundle.evaluation_plan
            self.station_manifest = bundle.station_manifest
            self.execution_manifest = bundle.execution_manifest
            self.closed = False

        def activate(self, *, request, operation, cancel_requested):
            assert request.semantic_content_hash == operation.request_hash
            assert not cancel_requested()
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="fake_activation_rejected_under_hold",
                lower_operation_id="lower-operation:test",
                evidence_refs=("lower-hold:test",),
                lower_hold_confirmed=True,
            )

        def run(self, **kwargs):
            raise AssertionError("failed activation must not run")

        def cancel(self, *, request, operation, reason):
            return self.request_hold(operation=operation, reason=reason)

        def request_hold(self, *, operation, reason):
            assert reason
            return StationLifecycleControlEvidence(
                succeeded=True,
                status="fake_lower_hold_confirmed",
                lower_operation_id="lower-operation:test",
                evidence_refs=("lower-hold:test",),
                lower_hold_confirmed=True,
            )

        def close(self):
            self.closed = True

    lifecycle = FailClosedLifecycle()
    session = assemble_mujoco_cube_lift_agent_gateway_session(
        application=application,
        contract_bundle=bundle,
        lifecycle=lifecycle,
        allowed_client_ids=("rpent:b3",),
        clock=application.runtime.clock,
    )
    facade = session.wire_adapter(authenticated_client_id="rpent:b3")
    observation = facade.get_observation_snapshot()
    request = AgentSkillRequest(
        client_id="rpent:b3",
        request_id="request:cube-lift:session:1",
        idempotency_key="rpent:b3:0:request:cube-lift:session:1",
        client_sequence=0,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=observation["station_revision"],
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=(
            route_entry.deployment_binding.deployment_binding_hash
        ),
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={},
        observation_snapshot_id=observation["snapshot_id"],
        requested_mode="sim_active",
        requested_timeout_ms=30_000,
    )
    admitted = facade.submit_skill_request(request.to_wire())
    assert admitted["accepted"]
    assert session.wait(timeout_sec=1.0)
    operation_id = admitted["operation"]["operation_id"]
    query = AgentOwnOperationQuery(
        client_id="rpent:b3",
        query_id="query:cube-lift:session:1",
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        operation_id=operation_id,
    )
    terminal = facade.get_own_operation(query.to_wire())
    assert terminal["accepted"]
    assert terminal["operation"]["status"] == "failed"
    assert terminal["operation"]["reason_code"] == (
        "station_activation_rejected"
    )
    assert lifecycle.closed

    refreshed = facade.get_observation_snapshot()
    second_request = replace(
        request,
        request_id="request:cube-lift:session:2",
        idempotency_key="rpent:b3:1:request:cube-lift:session:2",
        client_sequence=1,
        expected_station_revision=refreshed["station_revision"],
        observation_snapshot_id=refreshed["snapshot_id"],
    )
    second = facade.submit_skill_request(second_request.to_wire())
    assert not second["accepted"]
    assert second["reason_code"] == "route_admission_limit_reached"
    assert session.service.operation_store.total_admitted == 1

    replay = facade.submit_skill_request(request.to_wire())
    assert replay["accepted"]
    assert replay["idempotent_replay"]
    assert replay["operation"]["operation_id"] == operation_id
    assert session.close().accepted


def test_session_repeats_passive_tcp_pose_reads_without_consuming_motion_worker(
    monkeypatch,
) -> None:
    application = _application()
    now = {"value": 40.0}
    application.runtime.clock = lambda: now["value"]

    def capture_cut(*, scene_projector):
        del scene_projector
        return MujocoSimHILPassiveApplicationCut(
            captured_at=now["value"],
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="configured",
            execution_gate="not_started",
            execution_gate_epoch=0,
            fault_class=None,
            fault_rearmable=False,
            scene_projection={
                "finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {
                    "right_arm": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                },
            },
            control_cut=ControlCut(
                runtime_session_id=RUNTIME_SESSION_ID,
                control_domain_id=CONTROL_DOMAIN_ID,
                clock_domain_id=application.runtime.config.clock_domain_id,
                clock_reset_epoch=0,
                cutoff_at=now["value"],
                cutoff_admission_sequence=0,
            ),
        )

    application.capture_passive_application_cut = capture_cut
    bundle = _contract_bundle(monkeypatch, application)
    motion_route = bundle.route_manifest.route("task.cube_lift", "1").route

    class UnusedMotionLifecycle:
        executor_id = bundle.deployment_binding.executor_id

        def __init__(self) -> None:
            self.application = application
            self.route = motion_route
            self.deployment_binding = bundle.deployment_binding
            self.evaluation_plan = bundle.evaluation_plan
            self.station_manifest = bundle.station_manifest
            self.execution_manifest = bundle.execution_manifest

        def activate(self, **_kwargs):
            raise AssertionError("Passive state read must not activate motion")

        def run(self, **_kwargs):
            raise AssertionError("Passive state read must not run motion")

        def cancel(self, **_kwargs):
            raise AssertionError("Passive state read must not cancel motion")

        def request_hold(self, **_kwargs):
            raise AssertionError("Passive state read must not request HOLD")

        def close(self) -> None:
            pass

    session = assemble_mujoco_cube_lift_agent_gateway_session(
        application=application,
        contract_bundle=bundle,
        lifecycle=UnusedMotionLifecycle(),
        allowed_client_ids=("rpent:b3",),
        clock=application.runtime.clock,
    )
    facade = session.wire_adapter(authenticated_client_id="rpent:b3")

    rpent_root = Path(__file__).resolve().parents[2] / "HierachicalVLA" / "RPent"
    if not (rpent_root / "robots" / "dexteleop" / "toolkit.py").is_file():
        pytest.skip("Live RPent checkout is unavailable for the optional loopback.")
    sys.path.insert(0, str(rpent_root))
    from robots.dexteleop.toolkit import DexTeleopToolkit

    request_ids = iter(("request:state:1", "request:state:2"))
    toolkit = DexTeleopToolkit(
        client=facade,
        client_id="rpent:b3",
        request_id_factory=lambda: next(request_ids),
    )
    specs = {item["name"]: item for item in toolkit.get_tools_spec()}
    state_tool = next(name for name in specs if "state_read_tcp_pose" in name)
    parameters = {
        "component_ids": ["right_arm"],
        "reference_frame_id": "right_arm_base",
        "controlled_frame_id": "tool_tcp",
        "max_measurement_age_ms": 100,
    }

    operation_ids: list[str] = []
    for _ in range(2):
        completed = toolkit.execute_tool(state_tool, parameters).result
        assert completed["accepted"]
        assert completed["rpent_read_completion"]["status"] == "terminal"
        operation_id = completed["operation"]["operation_id"]
        operation_ids.append(operation_id)
        assert session.state_executor.wait(operation_id, timeout_sec=1.0)
        assert completed["operation"]["status"] == "completed"
        result = completed["operation"]["execution_result"]
        assert result["component_ids"] == ["right_arm"]
        assert result["pose_by_component"]["right_arm"] == parameters.get(
            "expected_pose",
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        )
        assert result["measurement_observation_hash"]
        now["value"] += 0.01

    assert len(set(operation_ids)) == 2
    assert session.worker.receipts == ()
    assert tuple(
        item.operation_kind for item in session.state_executor.port_receipts
    ) == ("activate", "activate")
    assert session.service.operation_store.admissions_for_route_hash(
        motion_route.semantic_content_hash
    ) == 0
    assert session.service.operation_store.total_admitted == 2

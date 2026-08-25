# Copyright (c) 2026 DexTeleop Contributors
from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_gateway.profiles.mujoco.motion_deployment as deployment
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.motion_deployment import (
    MujocoAgentTcpPoseHistoryProfile,
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentTcpPoseHistoryStorageConfig,
)
from agent_gateway.profiles.mujoco.motion_gateway_session import (
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from collection_runtime.source_observation_history import (
    SourceObservationHistoryBinding,
    SQLiteSourceObservationHistoryStore,
)
from control_station.sim_hil_profile.application import (
    MujocoSimHILPassiveApplicationCut,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.runtime.control_admission import ControlCut
from sim.mujoco.deploy import (
    MujocoDirectComponentProposalSource,
    build_mujoco_pose_component_target_profile,
)


def _build_application(monkeypatch):
    target_profile = build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id="fixture.agent_tcp_pose_history.right_arm.v1",
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms"},
    )
    action_schema = target_profile.action_schema
    observation_plan = SimpleNamespace(
        enabled=True,
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={"right_arm": "cartesian_pose"},
        semantic_content_hash="observation-plan:agent-history-fixture",
    )
    direct = MujocoDirectComponentProposalSource(
        producer_id="control-station-agent-history-fixture",
        producer_epoch=1,
    )
    state = {
        "time": 10.0,
        "pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    }
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            runtime_session_id="runtime:agent-history-fixture",
            control_domain_id="domain:agent-history-fixture",
            embodiment_instance_id="embodiment:agent-history-fixture",
            clock_domain_id="clock:agent-history-fixture",
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
            semantic_content_hash="component-catalog:history-fixture",
        ),
        component_target_capability=SimpleNamespace(
            semantic_content_hash="component-target-capability:history-fixture"
        ),
        target_runtime_action_contract=SimpleNamespace(
            semantic_content_hash="target-runtime-action:history-fixture"
        ),
        execution_capability=SimpleNamespace(
            semantic_content_hash="execution-capability:history-fixture"
        ),
        reference_activation_spec=SimpleNamespace(
            semantic_content_hash="reference-activation:history-fixture"
        ),
        execution_observation_binding=SimpleNamespace(
            binding_id="observation-binding:history-fixture",
            plan_hash=observation_plan.semantic_content_hash,
            semantic_content_hash="observation-binding-hash:history-fixture",
        ),
        execution_observation_capability=SimpleNamespace(
            semantic_content_hash="observation-capability:history-fixture",
            representation_by_component={"right_arm": "cartesian_pose"},
            schema_hash_by_component={"right_arm": "pose-schema:history-fixture"},
        ),
        authoritative_port=SimpleNamespace(
            profile=SimpleNamespace(
                selection_group_ids=("arms",),
                authority_policy_hash="authority-policy:history-fixture",
                decision_schedule_hash="decision-schedule:history-fixture",
            ),
        ),
        clock=lambda: state["time"],
    )
    frame_graph = CoordinateFrameGraph(
        graph_id="frame-graph:history-fixture",
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
                transform_id="history-fixture:right-arm-base-to-station",
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
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application.__dict__.update(
        runtime=runtime,
        task_config=SimpleNamespace(physics_hz=1_000.0),
        calibration_hash="calibration:agent-history-fixture",
        target_frame_profile=SimpleNamespace(
            reference_frame_by_component={"right_arm": "right_arm_base"},
            canonical_reference_frame_id="mujoco_sim_hil/station",
            frame_graph=frame_graph,
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash="tool-config:history-fixture",
            binding_by_component={
                "right_arm": SimpleNamespace(controlled_frame_id="tool_tcp")
            },
        ),
    )
    application._lock = threading.RLock()
    application._started = False
    application._held = False
    application._closed = False
    application._active_operation_id = None

    def capture_cut(*, scene_projector):
        del scene_projector
        return MujocoSimHILPassiveApplicationCut(
            captured_at=state["time"],
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
                "pose_xyz_quat_xyzw_by_component": {"right_arm": list(state["pose"])},
            },
            control_cut=ControlCut(
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                clock_domain_id=runtime.config.clock_domain_id,
                clock_reset_epoch=0,
                cutoff_at=state["time"],
                cutoff_admission_sequence=0,
            ),
        )

    application.capture_passive_application_cut = capture_cut
    monkeypatch.setattr(
        deployment,
        "build_control_execution_manifest",
        lambda **_kwargs: SimpleNamespace(
            semantic_content_hash="execution-manifest:history-fixture"
        ),
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_owner_registry_id",
        lambda _runtime: "owner-registry:history-fixture",
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_authority_epoch_source_id",
        lambda _runtime: "authority-epoch-source:history-fixture",
    )
    monkeypatch.setattr(
        deployment,
        "authoritative_execution_epoch_source_id",
        lambda _runtime: "execution-epoch-source:history-fixture",
    )
    return application, state


def _history_binding(*, runtime_session_id: str) -> SourceObservationHistoryBinding:
    return SourceObservationHistoryBinding(
        source_id="planning-history.test-source.v1",
        runtime_session_id=runtime_session_id,
        clock_domain_id="clock:history-partition",
        channels=("tcp_pose",),
        capture_profile_hash="profile:history-partition",
        source_contract={"contract_id": "history-partition-test.v1"},
    )


def _history_profile() -> MujocoAgentTcpPoseHistoryProfile:
    return MujocoAgentTcpPoseHistoryProfile(
        max_samples=4,
        max_window_ms=5_000,
        max_result_payload_bytes=16_384,
        retention_sec=10.0,
        max_records=16,
        max_record_payload_bytes=4_096,
        memory_buffer_max_records=4,
        capture_interval_sec=1.0,
    )


def test_history_is_opt_in_exactly_bound_and_default_off(
    monkeypatch,
    tmp_path: Path,
) -> None:
    application, state = _build_application(monkeypatch)
    profile = _history_profile()
    bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        history_profile=profile,
    )
    history_entry = bundle.route_manifest.route(
        "state.read_tcp_pose_history",
        "1",
    )
    assert history_entry.route.control_effect_class == "read_only"
    assert not history_entry.route.may_move_robot
    assert bundle.history_source_binding is not None

    with pytest.raises(ValueError, match="requires an exact injected disk store"):
        assemble_mujoco_agent_tcp_pose_gateway_session(
            application=application,
            contract_bundle=bundle,
            allowed_client_ids=("rpent:history-fixture",),
        )

    with SQLiteSourceObservationHistoryStore(
        root=(tmp_path / "planning-history").resolve(),
        binding=bundle.history_source_binding,
        limits=profile.store_limits,
    ) as store:
        session = assemble_mujoco_agent_tcp_pose_gateway_session(
            application=application,
            contract_bundle=bundle,
            allowed_client_ids=("rpent:history-fixture",),
            source_observation_history_store=store,
            owns_source_observation_history_store=True,
        )
        assert session.history_provider is not None
        assert session.history_executor is not None
        assert session.history_recorder is not None
        assert not session.history_recorder.started
        route_to_port = {
            item.skill_id: item.execution_port
            for item in session.service.executor_registry.bindings
        }
        assert route_to_port["state.read_tcp_pose_history"] is (
            session.history_executor
        )
        assert not application.started

        assert session.history_recorder.capture_once() is not None
        state["time"] = 11.0
        state["pose"][0] = 0.2
        parameters = {
            "component_ids": ["right_arm"],
            "reference_frame_id": "mujoco_sim_hil/station",
            "controlled_frame_id": "tool_tcp",
            "history_window_ms": 5_000,
            "max_samples": 4,
        }
        history_read = session.history_provider.read(
            active_operation_id="operation:tcp-pose-history:lower:1",
            parameters=parameters,
        )
        result = history_read.execution_result
        assert result is not None and result["sample_count"] == 2
        assert result["reference_frame_id"] == "mujoco_sim_hil/station"
        assert bundle.history_source_binding.source_contract[
            "reference_frame_id"
        ] == "mujoco_sim_hil/station"
        assert [
            sample["pose_by_component"]["right_arm"][0] for sample in result["samples"]
        ] == pytest.approx([0.1, 0.2])
        assert len(history_read.evidence_refs) == 3
        assert not application.started
        session.close()
        assert store.closed


@pytest.mark.integration
@pytest.mark.optional
def test_installed_rpent_toolkit_reads_exactly_bound_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    toolkit_module = pytest.importorskip(
        "rpent_dexteleop.toolkit",
        reason="paired RPent integration wheel is not installed",
    )
    session_module = pytest.importorskip(
        "rpent_dexteleop.gateway.session",
        reason="paired RPent integration wheel is not installed",
    )
    DexTeleopToolExposurePolicy = toolkit_module.DexTeleopToolExposurePolicy
    DexTeleopToolkit = toolkit_module.DexTeleopToolkit
    DexTeleopConsumerSession = session_module.DexTeleopConsumerSession
    application, state = _build_application(monkeypatch)
    profile = _history_profile()
    bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        history_profile=profile,
    )

    with SQLiteSourceObservationHistoryStore(
        root=(tmp_path / "paired-planning-history").resolve(),
        binding=bundle.history_source_binding,
        limits=profile.store_limits,
    ) as store:
        session = assemble_mujoco_agent_tcp_pose_gateway_session(
            application=application,
            contract_bundle=bundle,
            allowed_client_ids=("rpent:history-fixture",),
            source_observation_history_store=store,
            owns_source_observation_history_store=True,
        )
        assert session.history_recorder is not None
        assert session.history_recorder.capture_once() is not None
        state["time"] = 11.0
        state["pose"][0] = 0.2
        facade = session.wire_adapter(
            authenticated_client_id="rpent:history-fixture"
        )
        parameters = {
            "component_ids": ["right_arm"],
            "reference_frame_id": "mujoco_sim_hil/station",
            "controlled_frame_id": "tool_tcp",
            "history_window_ms": 5_000,
            "max_samples": 4,
        }
        consumer_session = DexTeleopConsumerSession(
            client=SimpleNamespace(
                get_skill_definition_catalog=facade.get_skill_definition_catalog,
                get_agent_skill_schema_catalog=facade.get_agent_skill_schema_catalog,
                get_skill_route_manifest=facade.get_skill_route_manifest,
                get_observation_snapshot=facade.get_observation_snapshot,
                submit_skill_request=facade.submit_skill_request,
                get_own_operation=facade.get_own_operation,
                cancel_own_operation=facade.cancel_own_operation,
                request_hold_own_domain=facade.request_hold_own_domain,
            ),
            client_id="rpent:history-fixture",
            operation_timeout_cap_ms=1_000,
            request_id_factory=lambda: "request:tcp-pose-history:toolkit:1",
        )
        toolkit = DexTeleopToolkit(
            session=consumer_session,
            exposure_policy=DexTeleopToolExposurePolicy(
                allowed_skill_ids=frozenset({"state.read_tcp_pose_history"})
            ),
        )
        history_tools = [
            binding
            for binding in toolkit.tool_bindings
            if binding.skill_id == "state.read_tcp_pose_history"
        ]
        assert len(history_tools) == 1
        tool_result = toolkit.execute_tool(history_tools[0].tool_name, parameters)
        assert tool_result.result["accepted"]
        assert tool_result.result["rpent_read_completion"]["status"] == "terminal"
        terminal = tool_result.result["operation"]
        assert terminal["status"] == "completed"
        assert terminal["execution_outcome"] == "completed"
        result = terminal["execution_result"]
        assert result is not None and result["sample_count"] == 2
        assert [
            sample["pose_by_component"]["right_arm"][0] for sample in result["samples"]
        ] == pytest.approx([0.1, 0.2])
        assert not application.started
        consumer_session.close()
        session.close()
        assert store.closed


def test_factory_history_storage_config_is_explicit_and_side_effect_free(
    tmp_path: Path,
) -> None:
    relative = Path("temp/agent-history")
    with pytest.raises(ValueError, match="explicit absolute path"):
        MujocoAgentTcpPoseHistoryStorageConfig(root=relative)
    root = (tmp_path / "configured-history").resolve()
    config = MujocoAgentTcpPoseHistoryStorageConfig(root=root)
    assert config.root == root
    assert not root.exists()


def test_factory_history_storage_partitions_exact_source_bindings(
    tmp_path: Path,
) -> None:
    first = _history_binding(runtime_session_id="runtime:history-partition:first")
    second = _history_binding(runtime_session_id="runtime:history-partition:second")
    config = MujocoAgentTcpPoseHistoryStorageConfig(
        root=(tmp_path / "history-catalog").resolve()
    )

    first_root = config.store_root_for(first)
    second_root = config.store_root_for(second)

    assert first_root == (
        config.root / "source_bindings" / first.semantic_content_hash
    )
    assert second_root == (
        config.root / "source_bindings" / second.semantic_content_hash
    )
    assert first_root != second_root
    assert not config.root.exists()


def test_collection_history_module_keeps_source_and_application_boundaries() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "collection_runtime"
        / "source_observation_history.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots.isdisjoint(
        {"control_station", "robot_motion", "sim", "teleop_runtime"}
    )

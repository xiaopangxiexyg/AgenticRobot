"""Offline real-profile wiring; all lower effects are fake or forbidden."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import gateway_cli, rm65_composition
from agent_gateway.profiles.real.exclusive_factory import (
    assemble_exclusive_real_agent_primitive_deployment,
)
from agent_gateway.profiles.real.motion_gateway_session import (
    assemble_real_agent_primitive_gateway_session,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    build_real_agent_primitive_contract_bundle,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.real.one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
)
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.access_policy import (
    assess_skill_request_access,
    issue_skill_execution_grant,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort
from tests.test_agent_gateway_real_exclusive_factory import _assembly_inputs
from tests.test_agent_gateway_real_motion_gateway_session import (
    CLIENT_ID,
    _Clock,
    _owner_state,
    _read_request,
    _TargetBuilder,
)
from tests.test_agent_gateway_real_motion_primitives import _inputs
from tests.test_agent_gateway_real_rm65_cli import _argv
from tests.test_agent_gateway_real_rm65_composition import _lower

pytestmark = pytest.mark.contract


def _bundle(*, enabled=False, side="right", attachment_change=None, public_offset=None):
    attachment, config, graph, envelope = _inputs()
    attachment = replace(attachment, component_ids=(side + "_arm",), **(attachment_change or {}))
    config = replace(config, active_sides=(side,), component_id_by_side={side: side + "_arm"})
    if public_offset is not None:
        matrix = np.eye(4); matrix[:3, 3] = public_offset
        graph = replace(graph, transforms=(replace(graph.transforms[0], matrix_target_source=tuple(tuple(row) for row in matrix)),))
        attachment = replace(
            attachment, reference_frame_id="canonical_station",
            reference_frame_spec_hash=graph.frame_by_id["canonical_station"].semantic_content_hash,
            frame_graph_hash=graph.semantic_content_hash,
        )
    envelope = replace(envelope, allowed_route_ids=real_agent_motion_route_ids_v1(include_safe_pose=True))
    return build_real_agent_primitive_contract_bundle(
        attachment=attachment, port_config=config, frame_graph=graph,
        primitive_reference_frame_id="canonical_station", safety_envelope_refs=(envelope,),
        enable_tcp_pose_comparison=enabled,
    ), config


def _session(*, side="right", public_offset=None, max_retained_operations=16):
    bundle, config = _bundle(enabled=True, side=side, public_offset=public_offset)
    clock = _Clock()
    port = RealAgentMotionPort(config=config, target_builder=_TargetBuilder())
    port.publish_measured_pose(
        pose_world_by_component={side + "_arm": np.eye(4)},
        measured_at=clock(), available_at=clock(), source_evidence_refs=("offline-cut:one-owner",),
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle, port=port,
        owner_state_provider=lambda: _owner_state(bundle, captured_at=clock()),
        allowed_client_ids=(CLIENT_ID,), safe_pose_trajectory_planner=_forbidden,
        clock=clock, max_retained_operations=max_retained_operations,
    )
    snapshot = gateway.observation_producer.capture_and_publish(gateway.service).snapshot
    assert snapshot is not None
    return gateway, snapshot, clock


def _forbidden(*args, **kwargs):
    raise AssertionError("Offline comparison must not read a device, plan, dispatch, or start a resource")


def _read(gateway, snapshot, *, version="2", target=None, sequence=1):
    bundle = gateway.contract_bundle
    entry = bundle.route_manifest.route("state.read_tcp_pose", version)
    profile = entry.route.allowed_parameter_profiles[0]
    request = _read_request(bundle, snapshot)
    parameters = dict(request.parameters)
    if version == "2":
        parameters["target_operation_id"] = target
    return replace(
        request, client_sequence=sequence, request_id=f"comparison-read:{sequence}",
        idempotency_key=f"comparison-read:{sequence}", skill_version=version,
        skill_definition_hash=entry.route.skill_definition_hash, route_id=entry.route.route_id,
        route_version=entry.route.route_version, route_hash=entry.route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
    )


def _run(gateway, request, clock):
    submitted = gateway.service.submit_request(
        request=request, current_station_revision=gateway.observation_producer.current_station_revision,
        now=clock(),
    )
    assert submitted.accepted, submitted
    assert gateway.service.launch_operation(submitted.operation.operation_id).accepted
    executor = gateway.service.execution_port_for_operation(submitted.operation.operation_id)
    assert executor.wait_all(timeout_sec=1.0)
    return gateway.service.get_operation(submitted.operation.operation_id)


def _retained_failed_safe_target(gateway, snapshot, clock) -> tuple[AgentSkillRequest, AgentSkillOperation]:
    """Create store-only history; this fixture command is never submitted or executed."""
    bundle = gateway.contract_bundle
    safe = bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1")
    profile = safe.route.allowed_parameter_profiles[0]
    parameters = {
        "pose_targets_by_component": {"right_arm": [.002, 0, 0, 0, 0, 0, 1]},
        "reference_frame_id": bundle.attachment.reference_frame_id,
        "reference_frame_spec_hash": bundle.attachment.reference_frame_spec_hash,
        "controlled_frame_id": "tool_tcp", "frame_graph_hash": bundle.attachment.frame_graph_hash,
        "tool_config_hash": bundle.attachment.tool_config_hash, "calibration_hash": bundle.attachment.calibration_hash,
    }
    source = replace(
        _read(gateway, snapshot), request_id="offline-store-fixture:not-executed",
        idempotency_key="offline-store-fixture:not-executed",
        skill_id=safe.route.skill_id, skill_version="1",
        skill_definition_hash=safe.route.skill_definition_hash, route_id=safe.route.route_id,
        route_version=safe.route.route_version, route_hash=safe.route.semantic_content_hash,
        deployment_binding_hash=safe.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
    )
    bundle.payload_validators.validate_parameters(route=safe.route, payload=parameters)
    # This pure fixture grant/decision is not a device permit or execution receipt.
    grant = issue_skill_execution_grant(
        access_policy=gateway.service.access_policy, manifest=bundle.route_manifest,
        route=safe.route, request=source,
        operator_authorization_receipt_id="offline-fixture:store-only-not-executed",
        client_id=source.client_id, parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash, max_invocations=1,
        issued_at=clock(), expires_at=clock() + 1.0,
    )
    source = replace(source, skill_execution_grant_id=grant.grant_id)
    decision = assess_skill_request_access(
        policy=gateway.service.access_policy, manifest=bundle.route_manifest,
        request=source, current_station_revision=snapshot.station_revision,
        now=clock(), grant=grant,
    )
    assert decision.accepted, decision
    store = gateway.service.operation_store
    admitted = store.admit(
        request=source, access_decision=decision,
        current_station_revision=snapshot.station_revision, admitted_at=clock(),
    )
    assert admitted.accepted and admitted.operation is not None, admitted
    failed = store.transition_worker_projection(
        operation_id=admitted.operation.operation_id,
        expected_operation_hash=admitted.operation.semantic_content_hash,
        next_status="failed", reason_code="offline_store_fixture_command_not_executed",
        current_station_revision=snapshot.station_revision, updated_at=clock(),
        progress_summary={"phase": "failed", "fixture_only": True, "command_executed": False},
    )
    assert failed.accepted and failed.operation is not None, failed
    original = failed.operation
    assert original.terminal and original.status == original.execution_outcome == "failed"
    assert original.request_hash == source.semantic_content_hash
    assert original.task_outcome == "not_assessed" and original.execution_result is None
    assert original.lower_operation_id is None
    assert store.active_operation_id is None
    assert store.get(original.operation_id) is original
    assert store.request_for_operation(original.operation_id) is source
    assert gateway.port.snapshot().dispatch_count == 0
    return source, original


@pytest.mark.parametrize("side", ["left", "right"])
def test_flag_adds_one_read_only_route_without_changing_v1_safe_route_or_motion_bindings(side):
    baseline, _ = _bundle(side=side)
    enabled, _ = _bundle(enabled=True, side=side)
    assert baseline.tcp_pose_comparison_registration is None
    assert len(enabled.route_manifest.skill_routes) == len(baseline.route_manifest.skill_routes) + 1
    assert enabled.deployment_bindings == baseline.deployment_bindings
    for entry in baseline.route_manifest.skill_routes:
        unchanged = enabled.route_manifest.route(entry.route.skill_id, entry.route.skill_version)
        assert unchanged.to_wire() == entry.to_wire()
    assert enabled.tcp_pose_observe.definition_catalog.to_wire() == baseline.tcp_pose_observe.definition_catalog.to_wire()
    assert enabled.tcp_pose_observe.schema_catalog.to_wire() == baseline.tcp_pose_observe.schema_catalog.to_wire()
    assert enabled.io_catalog.contract("state.read_tcp_pose", "1").to_wire() == baseline.io_catalog.contract("state.read_tcp_pose", "1").to_wire()
    registration = enabled.tcp_pose_comparison_registration
    route = registration.entry.route
    assert route.control_effect_class == "read_only" and route.privilege_tier == "t0_read_only"
    assert not route.may_move_robot and not route.requires_task_evaluation and not route.may_open_publication
    assert route.cancellation_policy == "cancel_only" and registration.entry.evaluation_plan_hash is None
    schema = registration.skill.schema_entry.parameter_schema["properties"]
    assert tuple(schema["component_ids"]["items"]["enum"]) == (side + "_arm",)
    assert tuple(schema["reference_frame_id"]["enum"]) == (enabled.attachment.reference_frame_id,)
    assert schema["max_measurement_age_ms"]["maximum"] == enabled.attachment.max_measurement_age_ms
    assert enabled.payload_validators.validator_for(route) is registration.payload_validator


@pytest.mark.parametrize("changes", [
    {"source_binding_hash": "a" * 64}, {"producer_epoch": 2},
    {"calibration_hash": "b" * 64}, {"tool_config_hash": "c" * 64},
])
def test_comparison_binding_closes_over_the_actual_state_attachment(changes):
    first, _ = _bundle(enabled=True)
    changed, _ = _bundle(enabled=True, attachment_change=changes)
    assert first.tcp_pose_comparison_registration.entry.deployment_binding.deployment_binding_hash != changed.tcp_pose_comparison_registration.entry.deployment_binding.deployment_binding_hash


@pytest.mark.parametrize("enabled", [0, 1, None, "true"])
def test_comparison_flag_rejects_non_boolean_before_assembly(enabled):
    with pytest.raises(TypeError, match="enable_tcp_pose_comparison"):
        _bundle(enabled=enabled)


@pytest.mark.parametrize("side", ["left", "right"])
def test_v1_and_v2_bind_different_route_executors_and_capture_one_existing_cut(monkeypatch, side):
    gateway, snapshot, clock = _session(side=side)
    cuts = []
    capture = gateway.observation_producer._capture_snapshot_with_read_evidence
    monkeypatch.setattr(gateway.observation_producer, "_capture_snapshot_with_read_evidence",
                        lambda **kwargs: cuts.append(kwargs["active_operation_id"]) or capture(**kwargs))
    v1 = _read(gateway, snapshot, version="1", sequence=1)
    v2 = _read(gateway, snapshot, version="2", sequence=2)
    assert gateway.service.executor_registry.execution_port_for_request(v1) is gateway.state_executor
    assert gateway.service.executor_registry.execution_port_for_request(v2) is gateway.tcp_pose_comparison_executor
    assert gateway.state_executor is not gateway.tcp_pose_comparison_executor
    assert gateway.passive_read_executors_by_skill_ref["state.read_tcp_pose", "1"] is gateway.state_executor
    assert gateway.passive_read_executors_by_skill_ref["state.read_tcp_pose", "2"] is gateway.tcp_pose_comparison_executor
    first = _run(gateway, v1, clock); second = _run(gateway, v2, clock)
    assert first.status == second.status == "completed"
    assert first.task_outcome == second.task_outcome == "not_assessed"
    assert "target_comparison" not in first.execution_result
    assert second.execution_result["target_comparison"] is None
    for key in ("source_measurement_hash", "measurement_id", "pose_by_component", "reference_frame_id", "frame_graph_hash", "tool_config_hash", "calibration_hash"):
        assert first.execution_result[key] == second.execution_result[key]
    assert cuts == [first.operation_id, second.operation_id]
    assert gateway.port.snapshot().dispatch_count == 0
    assert gateway.observation_producer.current_station_revision == snapshot.station_revision


@pytest.mark.parametrize("max_retained_operations", [1, 2])
def test_retained_target_lookup_uses_this_service_store_without_executing_the_target(monkeypatch, max_retained_operations):
    gateway, snapshot, clock = _session(max_retained_operations=max_retained_operations)
    source, original = _retained_failed_safe_target(gateway, snapshot, clock)
    store = gateway.service.operation_store
    assert store.max_retained_operations == max_retained_operations
    assert store.total_admitted == store.retained_operation_count == 1
    assert store.total_evicted == 0
    captures = []
    capture = gateway.observation_producer.capture_tcp_pose
    def capture_state(**kwargs):
        captures.append(kwargs["active_operation_id"])
        return capture(**kwargs)
    monkeypatch.setattr(gateway.observation_producer, "capture_tcp_pose", capture_state)
    submitted = gateway.service.submit_request(
        request=_read(gateway, snapshot, target=original.operation_id, sequence=2),
        current_station_revision=snapshot.station_revision, now=clock(),
    )
    assert submitted.accepted and submitted.operation is not None, submitted
    assert captures == []
    assert store.total_admitted == 2
    assert store.retained_operation_count == max_retained_operations
    assert store.total_evicted == (1 if max_retained_operations == 1 else 0)
    # Admission itself evicts terminal history before the passive reader starts.
    if max_retained_operations == 1:
        assert store.get(original.operation_id) is None
        assert store.request_for_operation(original.operation_id) is None
    else:
        assert store.get(original.operation_id) is original
        assert store.request_for_operation(original.operation_id) is source
    assert gateway.service.launch_operation(submitted.operation.operation_id).accepted
    assert gateway.tcp_pose_comparison_executor.wait_all(timeout_sec=1.0)
    result = gateway.service.get_operation(submitted.operation.operation_id)
    assert result is not None
    assert gateway.port.snapshot().dispatch_count == 0
    assert gateway.observation_producer.current_station_revision == snapshot.station_revision
    if max_retained_operations == 1:
        assert result.status == result.execution_outcome == "failed"
        assert result.execution_result is None
        assert result.reason_code == "passive_read_executor_exception:ValueError"
        assert "TCP target is unavailable" in result.progress_summary["failure_diagnostic"]["message"]
        assert captures == []
        assert store.get(original.operation_id) is None
        assert store.request_for_operation(original.operation_id) is None
        return
    assert result.status == "completed", result
    assert captures == [result.operation_id]
    comparison = result.execution_result["target_comparison"]
    assert comparison["target_operation_id"] == original.operation_id
    assert comparison["target_request_hash"] == source.semantic_content_hash
    assert comparison["target_route_hash"] == source.route_hash
    assert dict(comparison["target_pose_by_component"]) == dict(source.parameters["pose_targets_by_component"])
    assert comparison["target_status"] == comparison["target_execution_outcome"] == "failed"
    assert comparison["target_task_outcome"] == "not_assessed"
    assert comparison["translation_error_m_by_component"]["right_arm"] == pytest.approx(.002)
    assert "target_updated_at" not in comparison
    assert "measurement_after_target_terminal" not in comparison
    assert store.get(original.operation_id) is original
    assert store.request_for_operation(original.operation_id) is source


def test_prior_comparison_result_cannot_restore_an_evicted_target(monkeypatch):
    gateway, snapshot, clock = _session(max_retained_operations=2)
    source, original = _retained_failed_safe_target(gateway, snapshot, clock)
    store = gateway.service.operation_store
    first = _run(gateway, _read(gateway, snapshot, target=original.operation_id, sequence=2), clock)
    assert first.status == "completed", first
    assert first.execution_result["target_comparison"]["target_request_hash"] == source.semantic_content_hash
    assert store.get(original.operation_id) is original
    assert store.retained_operation_count == 2 and store.total_evicted == 0
    captures = []
    monkeypatch.setattr(gateway.observation_producer, "capture_tcp_pose",
                        lambda **kwargs: captures.append(kwargs) or _forbidden())
    second = _run(gateway, _read(gateway, snapshot, target=original.operation_id, sequence=3), clock)
    assert second.status == second.execution_outcome == "failed"
    assert second.execution_result is None and captures == []
    assert second.reason_code == "passive_read_executor_exception:ValueError"
    assert "TCP target is unavailable" in second.progress_summary["failure_diagnostic"]["message"]
    assert store.total_admitted == 3 and store.total_evicted == 1
    assert store.get(original.operation_id) is None
    assert store.request_for_operation(original.operation_id) is None
    # A retained successful read contains target data but is not its authority.
    assert store.get(first.operation_id) is first
    assert first.execution_result["target_comparison"]["target_operation_id"] == original.operation_id
    assert gateway.port.snapshot().dispatch_count == 0
    assert gateway.observation_producer.current_station_revision == snapshot.station_revision


@pytest.mark.parametrize("side", ["left", "right"])
def test_v2_uses_the_original_public_frame_projection_and_source_cut(side):
    offset = (.4, -.2, .1)
    gateway, snapshot, clock = _session(side=side, public_offset=offset)
    operation = _run(gateway, _read(gateway, snapshot), clock)
    assert operation.status == "completed"
    result = operation.execution_result
    assert result["reference_frame_id"] == "canonical_station"
    assert result["pose_by_component"][side + "_arm"][:3] == pytest.approx(offset)
    assert result["observation_binding_id"] == gateway.contract_bundle.attachment.observation_binding_id
    assert result["observation_plan_hash"] == gateway.contract_bundle.attachment.observation_plan_hash
    assert result["pose_schema_hash_by_component"][side + "_arm"] == gateway.contract_bundle.attachment.pose_schema_hash
    assert gateway.port.snapshot().dispatch_count == 0


def test_missing_target_fails_before_capture_and_cannot_fall_back_to_current_state(monkeypatch):
    gateway, snapshot, clock = _session()
    monkeypatch.setattr(gateway.observation_producer, "capture_tcp_pose", _forbidden)
    result = _run(gateway, _read(gateway, snapshot, target="not-retained"), clock)
    assert result.status == "failed"
    assert result.execution_result is None
    assert gateway.port.snapshot().dispatch_count == 0


def test_comparison_is_in_unused_and_normal_close_waits(monkeypatch):
    gateway, _, _ = _session()
    # No operation is admitted in this test; close must still inspect V2 readers.
    unused = []; waiter = gateway.tcp_pose_comparison_executor.wait_all
    monkeypatch.setattr(gateway.tcp_pose_comparison_executor, "wait_all",
                        lambda *, timeout_sec: unused.append(timeout_sec) or waiter(timeout_sec=timeout_sec))
    gateway.close_unused()
    assert unused == [0.0]
    gateway, _, _ = _session()
    monkeypatch.setattr(gateway.service, "close", lambda **kwargs: SimpleNamespace(accepted=True))
    monkeypatch.setattr(gateway.tcp_pose_comparison_executor, "wait_all", lambda *, timeout_sec: False)
    with pytest.raises(RuntimeError, match="Passive real reads did not quiesce"):
        gateway.close()
    monkeypatch.undo()
    gateway.close_unused()


@pytest.mark.parametrize("enabled", [False, True])
def test_exclusive_factory_passes_flag_without_starting_or_reading_lower_resources(enabled):
    graph, station, attachment, config, envelope = _assembly_inputs()
    target_builder = _TargetBuilder()
    deployment = assemble_exclusive_real_agent_primitive_deployment(
        target_builder=target_builder, port_config=config, observation_attachment=attachment,
        frame_graph=graph, primitive_reference_frame_id=station.frame_id,
        safety_envelope_refs=(envelope,), allowed_client_ids=(CLIENT_ID,),
        observation_provider=_forbidden, synchronize_target_builder=_forbidden,
        command_admission=_forbidden, command_dispatch=_forbidden, hold_controller=_forbidden,
        enable_tcp_pose_comparison=enabled,
    )
    session = deployment.gateway_session
    assert (session.tcp_pose_comparison_executor is not None) is enabled
    assert (session.contract_bundle.tcp_pose_comparison_registration is not None) is enabled
    assert deployment.control_runtime.motion_port is session.port
    session.close_unused()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("enabled", [False, True])
def test_rm65_composition_to_factory_chain_keeps_safe_owner_and_starts_no_device(monkeypatch, tmp_path, side, enabled):
    lower, guard = _lower(monkeypatch, envelope_side=side)
    monkeypatch.setattr(rm65_composition, "validate_agent_gateway_unix_server_preflight",
                        lambda **kwargs: (str(kwargs["socket_path"]), kwargs["peer_idle_timeout_sec"]))
    monkeypatch.setattr(rm65_composition, "validate_agent_gateway_unix_group_parent", lambda **kwargs: 1000)
    monkeypatch.setattr(rm65_composition, "build_exclusive_rm65_skill_target_builder", lambda *args: object())
    monkeypatch.setattr(rm65_composition, "RM65GraspJointTrajectoryPlanner", lambda **kwargs: _forbidden)
    composition = rm65_composition.assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower, socket_path=tmp_path/"never-opened.sock", authorized_side=side,
        authorized_action="move_tcp_to_pose_safe", confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        client_id=CLIENT_ID, client_uid=1000, runtime_session_id="offline-real-comparison",
        control_domain_id="offline-domain", embodiment_instance_id="offline-dual-arm",
        enable_tcp_pose_comparison=enabled,
    )
    session = composition.deployment.gateway_session
    assert (session.tcp_pose_comparison_executor is not None) is enabled
    assert composition.port_config.active_sides == (side,)
    assert composition.attachment.component_ids == (side + "_arm",)
    assert session.contract_bundle.route_manifest.route("motion.move_tcp_to_pose_safe", "1").route.may_move_robot
    assert composition.deployment.control_runtime.motion_port is session.port
    assert all(backend.starts == 0 for backend in lower.arm_backends.values()) and guard.starts == 0
    assert not composition.socket_path.exists()
    session.close_unused()


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_flag_reaches_composition_before_any_real_resource_start(monkeypatch, short_unix_socket_path_factory, enabled):
    parser = gateway_cli.build_parser()
    assert parser.get_default("enable_tcp_pose_comparison") is False
    lower = object(); captured = {}
    monkeypatch.setattr(gateway_cli, "resolve_exclusive_rm65_skill_deployment_profile",
                        lambda profile: SimpleNamespace(semantic_content_hash="a" * 64))
    monkeypatch.setattr(gateway_cli, "assemble_exclusive_rm65_skill_lower", lambda **kwargs: lower)
    def stop_before_start(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("offline composition boundary")
    monkeypatch.setattr(gateway_cli, "assemble_exclusive_rm65_agent_primitive_composition", stop_before_start)
    args = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
                 socket_path=short_unix_socket_path_factory("offline-comparison"))
    if enabled: args.append("--enable-tcp-pose-comparison")
    with pytest.raises(RuntimeError, match="offline composition boundary"):
        gateway_cli.main(args)
    assert captured["lower"] is lower
    assert captured["enable_tcp_pose_comparison"] is enabled

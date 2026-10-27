"""Backend-free adapter tests; native execution is verified separately."""

import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_gateway_real_motion_primitives import _inputs
from test_agent_gateway_tcp_pose_comparison import _result

from agent_gateway.profiles.real.motion_primitive_profiles import (
    build_real_agent_primitive_contract_bundle,
    real_agent_motion_route_ids_v1,
)
from agent_gateway.profiles.tcp_pose_comparison import (
    build_tcp_pose_comparison_executor,
    build_tcp_pose_comparison_registration,
)
from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.runtime.passive_read_executor import PassiveReadResult
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorRegistry


def _request(manifest, entry, parameters, *, sequence=0):
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="test:comparison", request_id=f"test:request:{sequence}",
        idempotency_key=f"test:key:{sequence}", client_sequence=sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=sequence, skill_id=route.skill_id,
        skill_version=route.skill_version, skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id, route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash, parameters=parameters,
        observation_snapshot_id=f"test:snapshot:{sequence}", requested_mode="real_active",
        requested_timeout_ms=min(1_000, route.max_operation_duration_ms),
    )


def _operation(request, *, operation_id, status="failed", updated_at=11.8):
    return AgentSkillOperation(
        operation_id=operation_id, request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id, status=status, reason_code="test:retained",
        station_revision_at_admission=request.expected_station_revision,
        current_station_revision=max(2, request.expected_station_revision),
        started_at=10, updated_at=updated_at, deadline_at=20,
        execution_outcome="failed" if status == "failed" else "pending",
    )


@pytest.fixture
def context():
    attachment, config, graph, envelope = _inputs()
    bundle = build_real_agent_primitive_contract_bundle(
        attachment=attachment, port_config=config, frame_graph=graph,
        primitive_reference_frame_id="canonical_station",
        safety_envelope_refs=(replace(envelope, allowed_route_ids=real_agent_motion_route_ids_v1(include_safe_pose=True)),),
    )
    base = bundle.tcp_pose_observe
    state_entry = base.route_manifest.route("state.read_tcp_pose", "1")
    base_validator = base.payload_validators.validator_for(state_entry.route)
    registration = build_tcp_pose_comparison_registration(
        base_definition=base.definition_catalog.skill_definitions[0],
        base_schema=base.schema_catalog.schema_entries[0],
        base_entry=state_entry, base_validator=base_validator,
    )
    manifest = replace(bundle.route_manifest, skill_routes=(*bundle.route_manifest.skill_routes, registration.entry))
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id="test:comparison",
        validators=tuple(bundle.payload_validators.validator_for(entry.route)
                         for entry in bundle.route_manifest.skill_routes) + (registration.payload_validator,),
    )
    measured = _result(target=None)
    measured.pop("target_comparison")
    attachment = bundle.attachment
    for key in ("reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
                "tool_config_hash", "calibration_hash", "observation_binding_id", "observation_plan_hash",
                "clock_domain_id"):
        measured[key] = getattr(attachment, key)
    measured["cutoff_clock_domain_id"] = attachment.clock_domain_id
    measured["pose_schema_hash_by_component"] = {"right_arm": attachment.pose_schema_hash}
    target = {key: measured[key] for key in ("reference_frame_id", "reference_frame_spec_hash",
        "controlled_frame_id", "frame_graph_hash", "tool_config_hash", "calibration_hash")}
    target["pose_targets_by_component"] = {"right_arm": [.23, -.06, .3, 0, 0, 1, 0]}
    source_entry = manifest.route("motion.move_tcp_to_pose_safe", "1")
    source_request = _request(manifest, source_entry, target)
    source_operation = _operation(source_request, operation_id="test:retained-motion")
    parameters = {"component_ids": ["right_arm"], "reference_frame_id": attachment.reference_frame_id,
        "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": attachment.max_measurement_age_ms,
        "target_operation_id": source_operation.operation_id}
    request = _request(manifest, registration.entry, parameters, sequence=4)
    operation = _operation(request, operation_id="test:current-read", status="running", updated_at=12)
    reads, lookups = [], []
    retained = {source_operation.operation_id: (source_request, source_operation)}

    def capture(parameters, operation_id):
        reads.append((parameters, operation_id))
        return PassiveReadResult(deepcopy(measured), ("test:measured-cut",))

    def lookup(operation_id):
        lookups.append(operation_id)
        return retained.get(operation_id)

    executor = build_tcp_pose_comparison_executor(
        registration=registration, capture_state=capture, lookup_target=lookup,
        route_manifest=manifest, payload_validators=validators,
        clock_domain_id=attachment.clock_domain_id,
        current_station_revision=lambda: 4, clock=lambda: 12, max_retained_operations=8,
    )
    return SimpleNamespace(**locals())


def test_adapter_uses_one_current_cut_not_old_success_or_target_pose(context):
    result = context.executor._read(context.request, context.operation)
    body = result.execution_result
    compared = body["target_comparison"]
    assert len(context.reads) == len(context.lookups) == 1
    assert "target_operation_id" not in context.reads[0][0]
    assert context.reads[0][1] == context.operation.operation_id
    assert body["pose_by_component"] == context.measured["pose_by_component"]
    assert compared["translation_error_m_by_component"]["right_arm"] == pytest.approx(.05)
    assert compared["rotation_error_rad_by_component"]["right_arm"] == pytest.approx(math.pi)
    assert compared["target_status"] == "failed"
    assert compared["target_execution_outcome"] == "failed"
    assert compared["target_task_outcome"] == "not_assessed"
    assert compared["target_request_hash"] == context.source_request.semantic_content_hash
    assert context.source_operation.status == "failed"
    assert "test:measured-cut" in result.evidence_refs
    assert len(result.evidence_refs) == 3
    assert body["measured_at"] == context.measured["measured_at"]


@pytest.mark.parametrize("omit_target", [False, True], ids=["explicit-null", "omitted"])
def test_state_only_never_looks_up_history_or_rewrites_the_request(context, omit_target):
    parameters = {**context.request.parameters, "target_operation_id": None}
    if omit_target:
        parameters.pop("target_operation_id")
    request = replace(context.request, parameters=parameters)
    before_hash = request.semantic_content_hash
    before_parameters = dict(request.parameters)
    operation = replace(context.operation, request_hash=before_hash)
    result = context.executor._read(request, operation)
    assert result.execution_result["target_comparison"] is None
    assert len(context.reads) == 1 and context.lookups == []
    assert "target_operation_id" not in context.reads[0][0]
    assert result.evidence_refs == ("test:measured-cut",)
    assert request.semantic_content_hash == before_hash
    assert dict(request.parameters) == before_parameters
    assert ("target_operation_id" not in request.parameters) == omit_target


def test_historical_terminal_timestamp_does_not_retime_the_current_measurement(context):
    context.retained[context.source_operation.operation_id] = (
        context.source_request, replace(context.source_operation, updated_at=12),
    )
    result = context.executor._read(context.request, context.operation).execution_result
    assert "target_updated_at" not in result["target_comparison"]
    assert "measurement_after_target_terminal" not in result["target_comparison"]
    assert result["measured_at"] == 11.9


@pytest.mark.parametrize("field,value", [
    ("client_id", "test:other-client"), ("runtime_session_id", "test:other-session"),
    ("skill_route_manifest_hash", "f" * 64), ("route_hash", "f" * 64),
    ("skill_definition_hash", "f" * 64), ("deployment_binding_hash", "f" * 64),
    ("parameter_profile_id", "test:other-profile"), ("parameter_profile_hash", "f" * 64),
    ("route_id", "test.missing_route"), ("route_version", "8"),
    ("skill_id", "motion.move_tcp_to_pose"), ("skill_version", "8"),
])
def test_historical_request_mismatch_is_rejected_before_measurement(context, field, value):
    altered = replace(context.source_request, **{field: value})
    operation = replace(context.source_operation, request_hash=altered.semantic_content_hash)
    context.retained[operation.operation_id] = (altered, operation)
    with pytest.raises((ValueError, KeyError)):
        context.executor._read(context.request, context.operation)
    assert context.reads == []


@pytest.mark.parametrize("field,value", [
    ("operation_id", "test:other-op"), ("request_hash", "f" * 64),
    ("runtime_session_id", "test:other-session"), ("station_revision_at_admission", 3),
])
def test_historical_operation_mismatch_is_rejected_before_measurement(context, field, value):
    altered = replace(context.source_operation, **{field: value})
    context.retained[context.source_operation.operation_id] = (context.source_request, altered)
    with pytest.raises(ValueError):
        context.executor._read(context.request, context.operation)
    assert context.reads == []


def test_pending_target_or_evicted_request_cannot_fallback_to_state(context):
    context.retained[context.source_operation.operation_id] = (
        context.source_request,
        replace(context.source_operation, status="running", execution_outcome="pending"),
    )
    with pytest.raises(ValueError, match="terminal"):
        context.executor._read(context.request, context.operation)
    context.retained.clear()
    with pytest.raises(ValueError, match="unavailable"):
        context.executor._read(context.request, context.operation)
    assert context.reads == []


def test_evicted_and_foreign_client_targets_have_the_same_external_failure(context):
    source = replace(context.source_request, client_id="test:foreign-client")
    context.retained[context.source_operation.operation_id] = (
        source, replace(context.source_operation, request_hash=source.semantic_content_hash),
    )
    with pytest.raises(ValueError) as foreign:
        context.executor._read(context.request, context.operation)
    context.retained.clear()
    with pytest.raises(ValueError) as missing:
        context.executor._read(context.request, context.operation)
    assert str(foreign.value) == str(missing.value)
    assert context.reads == []


@pytest.mark.parametrize("field", ["reference_frame_spec_hash", "frame_graph_hash", "tool_config_hash",
                                  "calibration_hash", "reference_frame_id", "clock_domain_id"])
def test_actual_cut_identity_change_is_rejected_without_rewriting_it(context, field):
    context.measured[field] = "test:wrong-identity"
    with pytest.raises(ValueError):
        context.executor._read(context.request, context.operation)
    assert len(context.reads) == 1
    assert context.measured[field] == "test:wrong-identity"


def test_stale_measurement_is_not_made_fresh_by_comparison(context):
    context.measured["cutoff_at"] = 13.0
    with pytest.raises(ValueError, match="fresh|age|stale"):
        context.executor._read(context.request, context.operation)
    assert context.measured["measured_at"] == 11.9


@pytest.mark.parametrize("field", ["translation_error_m_by_component", "rotation_error_rad_by_component"])
def test_contextual_validator_recomputes_errors(context, field):
    result = context.executor._read(context.request, context.operation).execution_result
    result["target_comparison"][field]["right_arm"] -= .001
    with pytest.raises(ValueError, match="differs"):
        context.registration.payload_validator.validate_execution_result_for_parameters(context.request.parameters, result)


def test_v1_base_is_unmodified_and_new_route_has_no_authority(context):
    original = context.base.route_manifest.route("state.read_tcp_pose", "1")
    assert original.route.skill_version == "1"
    assert "target_operation_id" not in context.base.schema_catalog.schema_entries[0].parameter_schema["properties"]
    added = context.registration.entry
    assert added.route.control_effect_class == "read_only"
    assert not added.route.may_move_robot and not added.route.may_open_publication
    assert not added.route.requires_task_evaluation and added.evaluation_plan_hash is None
    assert added.deployment_binding.target_binding_id == original.deployment_binding.target_binding_id
    assert added.deployment_binding.execution_manifest_hash == original.deployment_binding.execution_manifest_hash


@pytest.mark.parametrize("field,value", [
    ("target_binding_id", "test:other-robot"),
    ("execution_manifest_hash", "test:other-execution-manifest"),
])
def test_matching_frame_does_not_allow_another_robot_or_execution_owner(context, field, value):
    entry = replace(context.source_entry, deployment_binding=replace(context.source_entry.deployment_binding, **{field: value}))
    # The authoritative manifest rejects the foreign target before an adapter
    # can be constructed. Do not bypass that existing owner to fake a scenario.
    with pytest.raises(ValueError, match="manifest"):
        replace(context.manifest, skill_routes=tuple(
            entry if item == context.source_entry else item for item in context.manifest.skill_routes
        ))


def test_comparison_does_not_compare_itself_or_invent_a_missing_component(context):
    request = replace(context.request, parameters={
        **context.request.parameters, "target_operation_id": context.operation.operation_id,
    })
    with pytest.raises(ValueError, match="itself"):
        context.executor._read(request, context.operation)
    context.measured["pose_by_component"] = {}
    with pytest.raises((KeyError, ValueError)):
        context.executor._read(context.request, context.operation)


def test_quaternion_sign_does_not_create_a_rotation_error(context):
    context.measured["pose_by_component"]["right_arm"] = [.23, -.06, .3, 0, 0, -1, 0]
    result = context.executor._read(context.request, context.operation).execution_result
    assert result["target_comparison"]["translation_error_m_by_component"]["right_arm"] == 0
    assert result["target_comparison"]["rotation_error_rad_by_component"]["right_arm"] == 0
    assert result["target_comparison"]["target_status"] == "failed"

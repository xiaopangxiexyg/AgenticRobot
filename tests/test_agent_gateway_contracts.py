from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_gateway.protocol.contracts import (
    AGENT_CONTROL_REASON_CODES_V1,
    AGENT_SKILL_PROTOCOL_VERSION_V2,
    AgentObservationSnapshot,
    AgentOwnDomainEmergencyStopRequest,
    AgentOwnDomainHoldRequest,
    AgentOwnOperationCancelRequest,
    AgentOwnOperationQuery,
    AgentSkillOperation,
    agent_own_domain_emergency_stop_request_from_json,
    agent_own_domain_hold_request_from_json,
    agent_own_operation_cancel_request_from_json,
    agent_own_operation_query_from_json,
    agent_skill_request_from_json,
    agent_skill_schema_snapshot_document,
    agent_skill_schema_snapshot_json,
)
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent_gateway"
    / "protocol"
    / "agent_skill_v2.schema.json"
)


def _route(
    *,
    skill_id: str = "inspection.station_state",
    route_id: str = "real.rm65.station_reader",
    mode: str = "real_observe",
    target_kind: str = "real",
) -> SkillExecutionRoute:
    parameter_schema_hash = f"params:{skill_id}:v1"
    profile = SkillParameterProfileRef(
        parameter_profile_id="safe-v1",
        parameter_profile_hash=f"profile:{skill_id}:safe-v1",
        parameter_schema_hash=parameter_schema_hash,
    )
    return SkillExecutionRoute(
        skill_id=skill_id,
        skill_version="1",
        skill_definition_hash=f"definition:{skill_id}:v1",
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=parameter_schema_hash,
        execution_result_schema_hash=f"execution-result:{skill_id}:v1",
        task_evaluation_schema_hash=f"task-evaluation:{skill_id}:v1",
        allowed_parameter_profiles=(profile,),
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(mode,),
        target_kind=target_kind,
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
    )


def _manifest(*routes: SkillExecutionRoute) -> SkillRouteManifest:
    entries = tuple(
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=SkillDeploymentBindingRef(
                deployment_binding_id=f"deployment:{route.route_id}",
                deployment_binding_hash=f"binding:{route.route_id}:v1",
                target_binding_id="target:rm65-test",
                target_kind="real",
                execution_manifest_hash="execution-manifest:test",
            ),
            evaluation_plan_hash=(
                f"evaluation-plan:{route.route_id}:v1"
                if route.requires_task_evaluation
                else None
            ),
            availability_hash=f"availability:{route.route_id}:ready",
        )
        for route in routes
    )
    return SkillRouteManifest(
        runtime_session_id="agent-runtime:test",
        control_domain_id="domain:arms",
        station_manifest_hash="station-manifest:test",
        execution_manifest_hash="execution-manifest:test",
        skill_definition_catalog_hash="definition-catalog:test",
        skill_schema_catalog_hash="schema-catalog:test",
        target_binding_id="target:rm65-test",
        target_kind="real",
        current_mode="real_observe",
        supported_transport_features=("poll_status_v1", "idempotency_v1"),
        skill_routes=entries,
        explicit_exclusions=("real_motion",),
    )


def _request_payload(
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
) -> dict[str, object]:
    entry = manifest.route(
        route.skill_id,
        route.skill_version,
        route.route_id,
        route.route_version,
    )
    profile = route.allowed_parameter_profiles[0]
    return {
        "client_id": "rpent:test",
        "request_id": "request:test:1",
        "idempotency_key": "idempotency:test:1",
        "client_sequence": 1,
        "runtime_session_id": manifest.runtime_session_id,
        "skill_route_manifest_hash": manifest.semantic_content_hash,
        "expected_station_revision": 7,
        "skill_id": route.skill_id,
        "skill_version": route.skill_version,
        "skill_definition_hash": route.skill_definition_hash,
        "route_id": route.route_id,
        "route_version": route.route_version,
        "route_hash": route.semantic_content_hash,
        "deployment_binding_hash": (
            entry.deployment_binding.deployment_binding_hash
        ),
        "parameter_profile_id": profile.parameter_profile_id,
        "parameter_profile_hash": profile.parameter_profile_hash,
        "parameters": {"labels": ["left", "right"], "stable": True},
        "observation_snapshot_id": "snapshot:test:7",
        "requested_mode": manifest.current_mode,
        "requested_timeout_ms": 500,
        "input_artifact_bindings": [],
    }


def test_manifest_is_order_stable_and_hashes_exact_route_bindings() -> None:
    inspect = _route()
    health = _route(
        skill_id="inspection.runtime_health",
        route_id="real.rm65.health_reader",
    )
    first = _manifest(inspect, health)
    second = _manifest(health, inspect)
    assert first.skill_routes == second.skill_routes
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.to_wire()["route_manifest_hash"] == first.semantic_content_hash
    assert first.protocol_version == AGENT_SKILL_PROTOCOL_VERSION_V2


def test_skill_request_identity_is_upper_framework_neutral() -> None:
    route = _route()
    manifest = _manifest(route)
    payload = _request_payload(manifest, route)
    payload["client_id"] = "other-upper:planner-1"

    request = agent_skill_request_from_json(payload)

    assert request.client_id == "other-upper:planner-1"
    assert request.to_wire() == {
        **payload,
        "skill_execution_grant_id": None,
    }


def test_request_json_shape_binds_route_and_detaches_parameters() -> None:
    route = _route()
    manifest = _manifest(route)
    payload = _request_payload(manifest, route)
    request = agent_skill_request_from_json(payload)
    original_hash = request.semantic_content_hash

    parameters = payload["parameters"]
    assert isinstance(parameters, dict)
    parameters["labels"].append("mutated")  # type: ignore[union-attr]
    assert request.to_wire()["parameters"] == {
        "labels": ["left", "right"],
        "stable": True,
    }
    assert request.semantic_content_hash == original_hash
    assert request.route_hash == route.semantic_content_hash

    payload["raw_qpos"] = [0.0] * 12
    with pytest.raises(ValueError, match="unknown fields"):
        agent_skill_request_from_json(payload)


def test_request_safety_profile_selection_is_optional_but_atomic() -> None:
    route = _route()
    manifest = _manifest(route)
    payload = _request_payload(manifest, route)
    request = agent_skill_request_from_json(payload)

    assert not request.has_safety_profile_selection
    assert "safety_profile_id" not in request.to_wire()

    payload["safety_profile_id"] = "safety.standard@1"
    with pytest.raises(ValueError, match="supplied together"):
        agent_skill_request_from_json(payload)

    payload.update(
        {
            "safety_profile_hash": "safety-profile-hash:standard",
            "deployment_safety_envelope_id": "sim.tcp.standard@1",
            "deployment_safety_envelope_hash": "envelope-hash:standard",
            "safety_constraint_set_hash": "constraints:standard",
        }
    )
    selected = agent_skill_request_from_json(payload)
    assert selected.has_safety_profile_selection
    assert selected.to_wire()["deployment_safety_envelope_id"] == (
        "sim.tcp.standard@1"
    )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("client_sequence", True, "non-negative integer"),
        ("requested_timeout_ms", 0, "positive integer"),
        ("parameters", {"value": float("nan")}, "non-finite"),
        ("route_version", "v1", "positive integer string"),
    ),
)
def test_request_rejects_non_json_or_unsafe_scalar_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    route = _route()
    manifest = _manifest(route)
    payload = _request_payload(manifest, route)
    payload[field_name] = value
    with pytest.raises((TypeError, ValueError), match=message):
        agent_skill_request_from_json(payload)


def test_observation_snapshot_is_causal_and_not_a_safety_permit() -> None:
    summary = {"arm_state": {"finite": True}, "qpos_count": 12}
    snapshot = AgentObservationSnapshot(
        snapshot_id="snapshot:test:7",
        runtime_session_id="agent-runtime:test",
        skill_route_manifest_hash="route-manifest:test",
        station_revision=7,
        captured_at=10.0,
        available_at=10.1,
        clock_domain_id="station-monotonic",
        authority_epoch=3,
        execution_epoch=4,
        dispatch_owner_generation=5,
        lifecycle_state="hold",
        execution_gate="hold",
        active_operation_id=None,
        source_binding_id="source:policy",
        producer_epoch=6,
        target_binding_id="target:rm65",
        execution_manifest_hash="execution:test",
        frame_graph_hash="frames:test",
        tool_config_hash="tool:test",
        calibration_hash="calibration:test",
        measurement_summary=summary,
        measurement_freshness_ms=20,
        measurement_valid=True,
    )
    summary["qpos_count"] = 0
    assert snapshot.to_wire()["measurement_summary"]["qpos_count"] == 12
    assert snapshot.to_wire()["snapshot_hash"] == snapshot.semantic_content_hash

    with pytest.raises(ValueError, match="cannot precede"):
        replace(snapshot, available_at=9.9)
    with pytest.raises(ValueError, match="requires a fault_class"):
        replace(snapshot, fault_rearmable=True)


def test_operation_separates_running_from_terminal_success() -> None:
    running = AgentSkillOperation(
        operation_id="operation:test",
        request_hash="request-hash:test",
        runtime_session_id="agent-runtime:test",
        status="running",
        reason_code="lower_operation_running",
        station_revision_at_admission=7,
        current_station_revision=8,
        started_at=20.0,
        updated_at=20.5,
        deadline_at=25.0,
        lower_operation_id="lower:test",
        execution_receipt_refs=("receipt:admitted",),
        progress_summary={"phase": "running"},
    )
    assert not running.terminal
    assert running.to_wire()["execution_result"] is None
    with pytest.raises(ValueError, match="execution_result requires"):
        replace(running, execution_result={"success": True})
    with pytest.raises(ValueError, match="typed execution_result"):
        replace(
            running,
            status="completed",
            reason_code="completed",
            execution_outcome="completed",
        )

    completed = replace(
        running,
        status="completed",
        reason_code="measured_terminal_evidence",
        execution_outcome="completed",
        updated_at=21.0,
        execution_result={"session_anchor_receipt_id": "anchor:test"},
    )
    assert completed.terminal
    assert completed.task_outcome == "not_assessed"
    assert completed.to_wire()["operation_hash"] == completed.semantic_content_hash

    evaluated_failure = replace(
        completed,
        evaluation_status="completed",
        task_outcome="failed",
        evaluator_receipt_refs=("evaluation:environment:1",),
        aggregate_evaluation_receipt_ref="evaluation:aggregate:1",
        task_evaluation={"verdict": "failed"},
    )
    assert evaluated_failure.execution_outcome == "completed"
    assert evaluated_failure.task_outcome == "failed"


def test_portable_schema_snapshot_is_route_based_and_strict() -> None:
    document = agent_skill_schema_snapshot_document()
    assert document["protocol_version"] == AGENT_SKILL_PROTOCOL_VERSION_V2
    assert len(document["snapshot_hash"]) == 64
    assert len(document["protocol_semantic_hash"]) == 64
    for schema in document["schemas"].values():
        assert schema["additionalProperties"] is False

    assert "skill_route_manifest" in document["schemas"]
    assert "skill_evaluation_plan" in document["schemas"]
    assert "skill_evaluation_receipt" in document["schemas"]
    assert "skill_aggregate_evaluation_receipt" in document["schemas"]
    assert "own_operation_query" in document["schemas"]
    assert "own_operation_cancel_request" in document["schemas"]
    assert "own_domain_hold_request" in document["schemas"]
    assert "own_domain_emergency_stop_request" in document["schemas"]
    assert "control_plane_response" in document["schemas"]
    assert "authoritative_port_receipt" in document["schemas"]
    assert document["hash_domains"]["authoritative_port_receipt"] == (
        "dexteleop.agent_gateway.authoritative_port_receipt.v1"
    )
    assert "capability_manifest" not in document["schemas"]
    request_schema = document["schemas"]["skill_request"]
    route_manifest_schema = document["schemas"]["skill_route_manifest"]
    assert "skill_io_catalog_hash" in route_manifest_schema["properties"]
    assert "skill_io_catalog_hash" in route_manifest_schema["required"]
    assert "route_hash" in request_schema["properties"]
    assert "raw_qpos" not in request_schema["properties"]
    assert "raw_pose" not in request_schema["properties"]
    assert json.loads(agent_skill_schema_snapshot_json()) == document


def test_b2c_control_requests_are_exact_scoped_and_estop_is_observation_free() -> None:
    common = {
        "client_id": "rpent:test",
        "runtime_session_id": "runtime:test",
        "skill_route_manifest_hash": "manifest:test",
    }
    query = AgentOwnOperationQuery(
        **common,
        query_id="query:1",
        operation_id="operation:1",
    )
    assert agent_own_operation_query_from_json(query.to_wire()) == query

    cancel = AgentOwnOperationCancelRequest(
        **common,
        control_request_id="control:cancel:1",
        idempotency_key="idem:cancel:1",
        operation_id="operation:1",
        observation_snapshot_id="snapshot:4",
        expected_station_revision=4,
        reason_code="agent_task_aborted",
    )
    assert agent_own_operation_cancel_request_from_json(cancel.to_wire()) == cancel

    hold = AgentOwnDomainHoldRequest(
        **common,
        control_request_id="control:hold:1",
        idempotency_key="idem:hold:1",
        control_domain_id="domain:arms",
        target_binding_id="target:rm65",
        observation_snapshot_id="snapshot:4",
        expected_station_revision=4,
        reason_code="agent_requested_pause",
    )
    assert agent_own_domain_hold_request_from_json(hold.to_wire()) == hold
    emergency_stop = AgentOwnDomainEmergencyStopRequest(
        **common,
        control_request_id="control:emergency-stop:1",
        idempotency_key="idem:emergency-stop:1",
        control_domain_id="domain:arms",
        target_binding_id="target:rm65",
        reason_code="agent_task_aborted",
    )
    assert (
        agent_own_domain_emergency_stop_request_from_json(
            emergency_stop.to_wire()
        )
        == emergency_stop
    )
    assert "observation_snapshot_id" not in emergency_stop.to_wire()
    assert "expected_station_revision" not in emergency_stop.to_wire()
    assert len(
        {
            cancel.semantic_content_hash,
            hold.semantic_content_hash,
            emergency_stop.semantic_content_hash,
        }
    ) == 3

    document = agent_skill_schema_snapshot_document()
    control_surface = {
        name: document["schemas"][name]
        for name in (
            "own_operation_query",
            "own_operation_cancel_request",
            "own_domain_hold_request",
            "own_domain_emergency_stop_request",
        )
    }
    serialized = json.dumps(control_surface, sort_keys=True).lower()
    for forbidden in ("resume", "rearm"):
        assert forbidden not in serialized
    assert tuple(document["enums"]["agent_control_reason_codes"]) == (
        AGENT_CONTROL_REASON_CODES_V1
    )

    with pytest.raises(ValueError, match="unknown fields"):
        agent_own_domain_hold_request_from_json(
            {**hold.to_wire(), "action": "estop"}
        )
    with pytest.raises(ValueError, match="Unsupported reason_code"):
        replace(hold, reason_code="resume")
    with pytest.raises(ValueError, match="unknown fields"):
        agent_own_domain_emergency_stop_request_from_json(
            {**emergency_stop.to_wire(), "expected_station_revision": 4}
        )


def test_checked_in_schema_matches_the_canonical_generator() -> None:
    assert _SCHEMA_PATH.read_text(encoding="utf-8") == (
        agent_skill_schema_snapshot_json()
    )


def test_agent_gateway_protocol_package_has_no_lower_runtime_imports() -> None:
    package_root = _SCHEMA_PATH.parent
    forbidden_roots = {
        "control_composition",
        "deployments",
        "hardware",
        "robot_motion",
        "sim",
        "teleop_runtime",
    }
    imported_roots: set[str] = set()
    for path in sorted(package_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(forbidden_roots)


def test_agent_gateway_root_is_discovery_only_and_lazy() -> None:
    script = """
import sys
import agent_gateway as gateway

assert gateway.__all__ == ["protocol", "runtime", "transport"]
allowed = {
    "agent_gateway",
    "agent_gateway.protocol",
    "agent_gateway.runtime",
    "agent_gateway.transport",
}
unexpected = sorted(
    name
    for name in sys.modules
    if name.startswith("agent_gateway") and name not in allowed
)
assert not unexpected, unexpected
assert "control_station" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)

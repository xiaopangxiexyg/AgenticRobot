from __future__ import annotations

import io
import json
import socket
import struct
import threading
from types import SimpleNamespace

import pytest

from agent_gateway.protocol.contracts import agent_skill_request_from_json
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
)
from agent_gateway.runtime.access_policy import SkillAccessDecision
from agent_gateway.runtime.hardware_view_provider import (
    HardwareResourceBinding,
    HardwareViewProvider,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.transport.ssh_readonly_proxy import (
    AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1,
    serve_ssh_readonly_proxy,
)
from agent_gateway.transport.unix_transport import (
    AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter


_PREFIX = struct.Struct(">I")


def _route(*, effect: str) -> SkillExecutionRoute:
    read_only = effect == "read_only"
    skill_id = "state.read_tcp_pose" if read_only else "motion.move_tcp_to_pose"
    route_id = "sim.state_reader" if read_only else "sim.pose_driver"
    parameter_hash = f"params:{skill_id}:v1"
    profile = SkillParameterProfileRef(
        parameter_profile_id="default-v1",
        parameter_profile_hash=f"profile:{skill_id}:default-v1",
        parameter_schema_hash=parameter_hash,
    )
    return SkillExecutionRoute(
        skill_id=skill_id,
        skill_version="1",
        skill_definition_hash=f"definition:{skill_id}:v1",
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=parameter_hash,
        execution_result_schema_hash=f"result:{skill_id}:v1",
        task_evaluation_schema_hash=f"evaluation:{skill_id}:v1",
        allowed_parameter_profiles=(profile,),
        control_effect_class=effect,
        privilege_tier="t0_read_only" if read_only else "t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=not read_only,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
    )


def _manifest(*routes: SkillExecutionRoute) -> SkillRouteManifest:
    return SkillRouteManifest(
        runtime_session_id="runtime:hardware-test",
        control_domain_id="domain:hardware-test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        skill_definition_catalog_hash="definitions:test",
        skill_schema_catalog_hash="schemas:test",
        target_binding_id="target:sim-test",
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=("idempotency_v1",),
        skill_routes=tuple(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id=f"deployment:{route.route_id}",
                    deployment_binding_hash=f"binding:{route.route_id}:v1",
                    target_binding_id="target:sim-test",
                    target_kind="sim",
                    execution_manifest_hash="execution:test",
                ),
                evaluation_plan_hash=None,
                availability_hash=f"availability:{route.route_id}:ready",
            )
            for route in routes
        ),
    )


def _request_payload(
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
    *,
    sequence: int = 1,
) -> dict[str, object]:
    entry = manifest.route(
        route.skill_id,
        route.skill_version,
        route.route_id,
        route.route_version,
    )
    profile = route.allowed_parameter_profiles[0]
    return {
        "client_id": "ssh-readonly:test",
        "request_id": f"request:test:{sequence}",
        "idempotency_key": f"idempotency:test:{sequence}",
        "client_sequence": sequence,
        "runtime_session_id": manifest.runtime_session_id,
        "skill_route_manifest_hash": manifest.semantic_content_hash,
        "expected_station_revision": 4,
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
        "parameters": {},
        "observation_snapshot_id": "snapshot:test:4",
        "requested_mode": "sim_active",
        "requested_timeout_ms": 500,
        "input_artifact_bindings": [],
    }


def _provider(
    *,
    manifest: SkillRouteManifest,
    store: BoundedInMemoryAgentSkillOperationStore,
    health_provider=lambda: "ready",
) -> HardwareViewProvider:
    return HardwareViewProvider(
        manifest=manifest,
        operation_store=store,
        bindings=(
            HardwareResourceBinding(
                resource_id="robot.test_arm",
                resource_kind="robot_arm",
                route_hashes=tuple(
                    entry.route.semantic_content_hash
                    for entry in manifest.skill_routes
                ),
                health_provider=health_provider,
                control_claimed_provider=lambda: False,
            ),
        ),
    )


def _admit(
    *,
    store: BoundedInMemoryAgentSkillOperationStore,
    manifest: SkillRouteManifest,
    route: SkillExecutionRoute,
) -> None:
    request = agent_skill_request_from_json(_request_payload(manifest, route))
    entry = manifest.route(
        route.skill_id,
        route.skill_version,
        route.route_id,
        route.route_version,
    )
    outcome = store.admit(
        request=request,
        access_decision=SkillAccessDecision(
            accepted=True,
            reason_code="accepted",
            request_hash=request.semantic_content_hash,
            manifest_hash=manifest.semantic_content_hash,
            access_policy_hash="policy:test",
            route_manifest_entry_hash=entry.semantic_content_hash,
            effective_timeout_ms=500,
        ),
        current_station_revision=4,
        admitted_at=1.0,
    )
    assert outcome.accepted


def test_profile_binding_projects_capabilities_without_owning_hardware() -> None:
    read = _route(effect="read_only")
    drive = _route(effect="motion")
    manifest = _manifest(drive, read)
    provider = _provider(
        manifest=manifest,
        store=BoundedInMemoryAgentSkillOperationStore(),
    )

    resource = provider.resource_views()[0].to_wire()
    assert resource["resource_id"] == "robot.test_arm"
    assert resource["properties"] == {"readable": True, "drivable": True}
    assert [item["skill_id"] for item in resource["read_skill_refs"]] == [
        "state.read_tcp_pose"
    ]
    assert [item["skill_id"] for item in resource["drive_skill_refs"]] == [
        "motion.move_tcp_to_pose"
    ]
    assert set(resource["read_skill_refs"][0]) == {"skill_id", "skill_version"}


@pytest.mark.parametrize(
    ("effect", "expected_occupancy", "expected_reason"),
    [
        ("read_only", "free", "active_shared_read_operation"),
        ("motion", "occupied", "active_operation_reservation"),
    ],
)
def test_status_projects_existing_operation_without_adding_a_lease(
    effect: str,
    expected_occupancy: str,
    expected_reason: str,
) -> None:
    route = _route(effect=effect)
    manifest = _manifest(route)
    store = BoundedInMemoryAgentSkillOperationStore()
    _admit(store=store, manifest=manifest, route=route)

    status = _provider(manifest=manifest, store=store).status_views(
        station_revision=4
    )[0]

    assert status.occupancy == expected_occupancy
    assert status.reason_code == expected_reason
    assert status.active_operation_id is not None


def test_health_failure_is_bounded_and_fail_closed() -> None:
    route = _route(effect="read_only")
    manifest = _manifest(route)

    def unavailable():
        raise RuntimeError("device detail must not escape")

    status = _provider(
        manifest=manifest,
        store=BoundedInMemoryAgentSkillOperationStore(),
        health_provider=unavailable,
    ).status_views(station_revision=8)[0]
    assert status.occupancy == "occupied"
    assert status.health == "unavailable"
    assert status.reason_code == "status_provider_unavailable_fail_closed"


def test_exclusive_lower_control_claim_is_occupied_while_idle() -> None:
    route = _route(effect="read_only")
    manifest = _manifest(route)
    binding = HardwareResourceBinding(
        resource_id="robot.test_arm",
        resource_kind="robot_arm",
        route_hashes=(route.semantic_content_hash,),
        health_provider=lambda: "ready",
        control_claimed_provider=lambda: True,
    )
    status = HardwareViewProvider(
        manifest=manifest,
        operation_store=BoundedInMemoryAgentSkillOperationStore(),
        bindings=(binding,),
    ).status_views(station_revision=8)[0]
    assert status.occupancy == "occupied"
    assert status.reason_code == "lower_control_claim_occupied"
    assert status.active_operation_id is None
    assert status.active_skill_ref is None


class _SubmissionService:
    def __init__(self, manifest: SkillRouteManifest) -> None:
        self.manifest = manifest
        self.requests = []

    def submit_request(self, **kwargs):
        self.requests.append(kwargs["request"])
        return SimpleNamespace(
            accepted=True,
            reason_code="accepted_for_test",
            idempotent_replay=False,
            operation=None,
        )


def _submission_adapter(manifest: SkillRouteManifest):
    adapter = object.__new__(BoundAgentGatewayWireAdapter)
    adapter.authenticated_client_id = "ssh-readonly:test"
    adapter.service = _SubmissionService(manifest)
    adapter._grant_resolver = lambda _request: (_ for _ in ()).throw(
        AssertionError("read-only SSH submission must not resolve grants")
    )
    adapter._operation_launcher = None
    adapter._current_station_revision = lambda: 4
    return adapter


def test_read_only_submission_is_server_enforced_before_admission() -> None:
    read = _route(effect="read_only")
    drive = _route(effect="motion")
    manifest = _manifest(read, drive)
    adapter = _submission_adapter(manifest)

    accepted = adapter.submit_read_only_skill_request(
        _request_payload(manifest, read)
    )
    assert accepted["accepted"]
    assert len(adapter.service.requests) == 1

    with pytest.raises(PermissionError, match="rejects motion and lifecycle"):
        adapter.submit_read_only_skill_request(
            _request_payload(manifest, drive, sequence=2)
        )
    assert len(adapter.service.requests) == 1


@pytest.mark.parametrize(
    "method_name",
    (
        "get_gateway_events",
        "begin_artifact_upload",
        "write_artifact_chunk",
        "commit_artifact_upload",
        "acknowledge_artifact_materialization",
        "submit_skill_request",
        "cancel_own_operation",
        "request_hold_own_domain",
    ),
)
def test_read_only_authenticated_client_cannot_use_privileged_methods(
    method_name: str,
) -> None:
    read = _route(effect="read_only")
    manifest = _manifest(read)
    adapter = _submission_adapter(manifest)
    adapter.read_only_access = True

    with pytest.raises(PermissionError, match="read-only authenticated client"):
        getattr(adapter, method_name)({})


def test_read_only_authenticated_client_can_still_submit_strict_read_route() -> None:
    read = _route(effect="read_only")
    manifest = _manifest(read)
    adapter = _submission_adapter(manifest)
    adapter.read_only_access = True

    accepted = adapter.submit_read_only_skill_request(
        _request_payload(manifest, read)
    )
    assert accepted["accepted"] is True


def _encoded_request(method: str, payload=None) -> bytes:
    body = json.dumps(
        {
            "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
            "request_id": f"request:{method}",
            "method": method,
            "payload": payload,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return _PREFIX.pack(len(body)) + body


def _serve_responses(path, *, responses: tuple[bytes, ...]):
    ready = threading.Event()

    def run():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
            server.listen(1)
            ready.set()
            connection, _ = server.accept()
            with connection:
                for response in responses:
                    prefix = connection.recv(_PREFIX.size)
                    if not prefix:
                        break
                    (size,) = _PREFIX.unpack(prefix)
                    remaining = size
                    while remaining:
                        chunk = connection.recv(remaining)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    connection.sendall(_PREFIX.pack(len(response)) + response)
        finally:
            server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    return thread


def _serve_one_response(path, *, response: bytes):
    return _serve_responses(path, responses=(response,))


def test_ssh_proxy_forwards_only_the_fixed_read_only_surface(
    short_unix_socket_path_factory,
) -> None:
    assert "submit_skill_request" not in AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    assert "request_hold_own_domain" not in AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    assert "cancel_own_operation" not in AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    assert "begin_artifact_upload" not in AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    assert "submit_read_only_skill_request" in (
        AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    )
    assert "cancel_own_read_only_operation" in (
        AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    )

    path = short_unix_socket_path_factory("ssh-ro")
    response = json.dumps({"ok": True}, separators=(",", ":")).encode("utf-8")
    thread = _serve_one_response(path, response=response)
    output = io.BytesIO()
    assert serve_ssh_readonly_proxy(
        socket_path=path,
        input_stream=io.BytesIO(_encoded_request("healthz")),
        output_stream=output,
    ) == 0
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    encoded = output.getvalue()
    (size,) = _PREFIX.unpack(encoded[: _PREFIX.size])
    assert encoded[_PREFIX.size :] == response
    assert size == len(response)

    path = short_unix_socket_path_factory("ssh-deny")
    thread = _serve_one_response(path, response=response)
    output = io.BytesIO()
    assert serve_ssh_readonly_proxy(
        socket_path=path,
        input_stream=io.BytesIO(_encoded_request("submit_skill_request")),
        output_stream=output,
    ) == 3
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    body_size = _PREFIX.unpack(output.getvalue()[: _PREFIX.size])[0]
    denial = json.loads(output.getvalue()[_PREFIX.size : _PREFIX.size + body_size])
    assert denial["error"]["code"] == "permission_denied"


def test_ssh_proxy_scopes_artifact_download_to_visible_own_operation(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("ssh-artifact")
    artifact_id = "artifact:camera:test"
    operation_response = json.dumps(
        {
            "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
            "request_id": "request:get_own_operation",
            "ok": True,
            "result": {
                "operation": {
                    "output_artifact_bindings": [
                        {"artifact": {"artifact_id": artifact_id}}
                    ]
                }
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    grant_response = json.dumps(
        {
            "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
            "request_id": "request:begin_artifact_download",
            "ok": True,
            "result": {"grant_id": "grant:test"},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    thread = _serve_responses(
        path,
        responses=(operation_response, grant_response),
    )
    input_bytes = _encoded_request("get_own_operation", {}) + _encoded_request(
        "begin_artifact_download",
        {"artifact_id": artifact_id, "mechanism": "chunked_rpc"},
    )
    assert serve_ssh_readonly_proxy(
        socket_path=path,
        input_stream=io.BytesIO(input_bytes),
        output_stream=io.BytesIO(),
    ) == 0
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    path = short_unix_socket_path_factory("ssh-artifact-deny")
    thread = _serve_one_response(path, response=grant_response)
    output = io.BytesIO()
    assert serve_ssh_readonly_proxy(
        socket_path=path,
        input_stream=io.BytesIO(
            _encoded_request(
                "begin_artifact_download",
                {"artifact_id": artifact_id, "mechanism": "chunked_rpc"},
            )
        ),
        output_stream=output,
    ) == 3
    thread.join(timeout=1.0)
    assert not thread.is_alive()

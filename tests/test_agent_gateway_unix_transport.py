from __future__ import annotations

import json
import os
import socket
import stat
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.protocol.contracts import (
    AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1,
    AGENT_GATEWAY_DOMAIN_EMERGENCY_STOP_FEATURE_V1,
    AgentOwnDomainEmergencyStopRequest,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillRouteManifest,
)
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_catalog_for_routes
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.operation_port import (
    AgentGatewayAuthoritativePortReceipt,
    DeterministicNoIoAgentSkillOperationPort,
)
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.transport import unix_transport
from agent_gateway.transport.unix_transport import (
    AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
    AgentGatewayUnixPeerCredentials,
)
from agent_gateway.transport.wire_adapter import (
    BoundAgentGatewayWireAdapter,
)

_PREFIX = struct.Struct(">I")


class _FakeBoundFacade(BoundAgentGatewayWireAdapter):
    def __init__(self, client_id: str) -> None:
        self.authenticated_client_id = client_id
        self.requests: list[tuple[str, object]] = []
        self.disconnect_reasons: list[str] = []
        self.domain_state = {
            "schema": "agent_gateway.domain_control_state.v1",
            "runtime_session_id": "runtime:test", "emergency_stop_latched": False,
            "emergency_stop_confirmed": False, "lower_stop_receipt_ref": None,
        }

    def get_domain_control_state(self):
        return dict(self.domain_state)

    def get_skill_definition_catalog(self):
        self.requests.append(("get_skill_definition_catalog", None))
        return {"catalog": "definitions"}

    def get_agent_skill_schema_catalog(self):
        self.requests.append(("get_agent_skill_schema_catalog", None))
        return {"catalog": "schemas"}

    def get_skill_route_manifest(self):
        self.requests.append(("get_skill_route_manifest", None))
        return {"manifest": "routes"}

    def get_hardware_resource_views(self):
        self.requests.append(("get_hardware_resource_views", None))
        return {"resources": []}

    def get_hardware_status_views(self):
        self.requests.append(("get_hardware_status_views", None))
        return {"statuses": []}

    def get_observation_snapshot(self):
        self.requests.append(("get_observation_snapshot", None))
        return {"snapshot": "current"}

    def submit_skill_request(self, payload):
        self._assert_client(payload)
        self.requests.append(("submit_skill_request", dict(payload)))
        return {
            "accepted": True,
            "reason_code": "fake_admitted",
            "idempotent_replay": False,
            "operation": None,
        }

    def submit_read_only_skill_request(self, payload):
        self._assert_client(payload)
        self.requests.append(("submit_read_only_skill_request", dict(payload)))
        return {
            "accepted": True,
            "reason_code": "fake_read_only_admitted",
            "idempotent_replay": False,
            "operation": None,
        }

    def get_own_operation(self, payload):
        self._assert_client(payload)
        self.requests.append(("get_own_operation", dict(payload)))
        return {"accepted": False, "reason_code": "fake_not_found"}

    def cancel_own_operation(self, payload):
        self._assert_client(payload)
        self.requests.append(("cancel_own_operation", dict(payload)))
        return {"accepted": False, "reason_code": "fake_not_found"}

    def cancel_own_read_only_operation(self, payload):
        self._assert_client(payload)
        self.requests.append(("cancel_own_read_only_operation", dict(payload)))
        return {"accepted": False, "reason_code": "fake_read_only_not_found"}

    def request_hold_own_domain(self, payload):
        self._assert_client(payload)
        self.requests.append(("request_hold_own_domain", dict(payload)))
        return {"accepted": True, "reason_code": "fake_hold_confirmed"}

    def request_emergency_stop_own_domain(self, payload):
        self._assert_client(payload)
        self.requests.append(
            ("request_emergency_stop_own_domain", dict(payload))
        )
        return {
            "accepted": True,
            "reason_code": "fake_emergency_stop_confirmed",
        }

    def handle_authenticated_disconnect(self, *, reason: str):
        self.disconnect_reasons.append(reason)
        return SimpleNamespace(
            accepted=True,
            reason_code="fake_disconnect_hold_confirmed",
        )

    def close(self):
        return SimpleNamespace(accepted=True, reason_code="fake_gateway_closed")

    def _assert_client(self, payload) -> None:
        if payload.get("client_id") != self.authenticated_client_id:
            raise PermissionError(
                "Payload client_id does not match authenticated connection."
            )


def _wait_for(predicate, *, timeout_sec: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise AssertionError("socket closed before the complete test frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _raw_call(sock: socket.socket, *, request_id: str, method: str, payload):
    envelope = {
        "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
        "request_id": request_id,
        "method": method,
        "payload": payload,
    }
    body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    sock.sendall(_PREFIX.pack(len(body)) + body)
    (size,) = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    return json.loads(_recv_exact(sock, size))


def _start_server(
    *,
    path: Path,
    facade: _FakeBoundFacade,
    peer_idle_timeout_sec: float = 1.0,
    max_frame_bytes: int = 2 * 1024 * 1024,
    socket_mode: int = 0o600,
):
    close_calls: list[object] = []

    def close_gateway():
        outcome = facade.close()
        close_calls.append(outcome)
        return outcome

    server = AgentGatewayUnixJsonServer(
        socket_path=path,
        peer_client_map=AgentGatewayUnixPeerClientMap(
            {os.getuid(): facade.authenticated_client_id}
        ),
        facade_factory=lambda client_id: (
            facade
            if client_id == facade.authenticated_client_id
            else (_ for _ in ()).throw(AssertionError("unexpected client"))
        ),
        close_gateway=close_gateway,
        peer_idle_timeout_sec=peer_idle_timeout_sec,
        max_frame_bytes=max_frame_bytes,
        socket_mode=socket_mode,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    return server, thread, close_calls


def _stop_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


@pytest.mark.parametrize("confirmed", [False, True])
def test_health_projects_current_owner_state_without_control_requests(short_unix_socket_path_factory, confirmed):
    facade = _FakeBoundFacade("rpent:test")
    path = short_unix_socket_path_factory("health-state")
    server, thread, _ = _start_server(path=path, facade=facade)
    fault = [None]
    server.control_fault_reader = lambda: fault[0]
    try:
        with socket.socket(socket.AF_UNIX) as peer:
            peer.connect(str(path))
            first = _raw_call(peer, request_id="h1", method="healthz", payload=None)
            assert first["ok"] and first["result"]["domain_control_state"] == facade.domain_state
            assert first["result"]["control_fault_reason"] is None
            facade.domain_state.update(emergency_stop_latched=True,
                emergency_stop_confirmed=confirmed, lower_stop_receipt_ref="lower:proof" if confirmed else None)
            fault[0] = "lower_control_cycle_failed"
            second = _raw_call(peer, request_id="h2", method="healthz", payload=None)
            assert second["ok"] and second["result"]["domain_control_state"] == facade.domain_state
            assert second["result"]["control_fault_reason"] == fault[0]
            assert facade.requests == []  # Health reads submit no Skill or stop request.
        with pytest.raises(RuntimeError, match="already bound"):
            server.control_fault_reader = lambda: None
    finally:
        _stop_server(server, thread)


@pytest.mark.parametrize("invalid", ["", "x" * 513, 1])
def test_health_reader_failure_does_not_publish_ready(short_unix_socket_path_factory, invalid):
    facade = _FakeBoundFacade("rpent:test")
    path = short_unix_socket_path_factory("bad-health")
    server, thread, _ = _start_server(path=path, facade=facade)
    server.control_fault_reader = lambda: invalid
    try:
        with socket.socket(socket.AF_UNIX) as peer:
            peer.connect(str(path))
            result = _raw_call(peer, request_id="bad", method="healthz", payload=None)
            assert not result["ok"] and "result" not in result
    finally:
        _stop_server(server, thread)


@pytest.mark.parametrize("error", [RuntimeError("offline health failure"), OSError("offline read failure")])
def test_health_reader_exception_never_returns_a_success_snapshot(short_unix_socket_path_factory, error):
    facade = _FakeBoundFacade("rpent:test")
    path = short_unix_socket_path_factory("health-error")
    server, thread, _ = _start_server(path=path, facade=facade)

    def fail():
        raise error

    server.control_fault_reader = fail
    try:
        with socket.socket(socket.AF_UNIX) as peer:
            peer.connect(str(path))
            result = _raw_call(peer, request_id="error", method="healthz", payload=None)
            assert not result["ok"] and "result" not in result
    finally:
        _stop_server(server, thread)


@pytest.mark.parametrize("stop_succeeded", [None, True, False], ids=["clear", "confirmed", "unconfirmed"])
def test_healthz_publishes_real_service_domain_control_state_without_device_io(
    short_unix_socket_path_factory, monkeypatch, stop_succeeded,
) -> None:
    # Use the real service, facade and Unix dispatcher; only the lower port is
    # synthetic. No upper snapshot or domain_control_state method is mocked.
    class StopPort:
        no_io = False
        port_id = "lower:test-stop"

        def __init__(self):
            self.calls = []

        def request_emergency_stop(self, *, operation, reason):
            assert operation is None
            self.calls.append(reason)
            return AgentGatewayAuthoritativePortReceipt(
                port_id=self.port_id, operation_kind="emergency_stop", operation_id=None,
                succeeded=stop_succeeded is True,
                status="confirmed" if stop_succeeded else "unconfirmed",
                completed_at=10.1, call_sequence=len(self.calls) - 1,
                evidence_refs=("lower:test-stop-receipt",),
                proves_motion_or_hold=stop_succeeded is True)

    client_id = "rpent:local"
    definitions = SkillDefinitionCatalog("health-definitions", "1", ())
    schemas = AgentSkillSchemaCatalog("health-schemas", "1", ())
    io_catalog = inline_only_skill_io_catalog_for_routes(())
    manifest = SkillRouteManifest(
        runtime_session_id="runtime:health-producer", control_domain_id="domain:health-producer",
        station_manifest_hash="station:test", execution_manifest_hash="execution:test",
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
        target_binding_id="target:test", target_kind="null", current_mode="dry_run",
        supported_transport_features=(AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1,
                                      AGENT_GATEWAY_DOMAIN_EMERGENCY_STOP_FEATURE_V1),
        skill_routes=(), explicit_exclusions=())
    stop_port = StopPort()
    operation_port = DeterministicNoIoAgentSkillOperationPort()
    service = AgentGatewayService(manifest=manifest,
        access_policy=AgentSkillAccessPolicy(access_policy_id="access:health",
            allowed_client_ids=(client_id,), allowed_modes=("dry_run",),
            max_request_timeout_ms=1000, max_grant_lifetime_sec=30.),
        operation_port=operation_port, domain_emergency_stop_port=stop_port)

    def forbidden(*_args, **_kwargs):
        pytest.fail("healthz must not observe, poll hardware, launch or grant control")

    facade = BoundAgentGatewayWireAdapter(authenticated_client_id=client_id,
        definition_catalog=definitions, schema_catalog=schemas, service=service,
        publish_observation=forbidden, current_station_revision=forbidden,
        operation_launcher=forbidden, grant_resolver=forbidden)
    for method in ("get_observation_snapshot", "get_hardware_resource_views", "get_hardware_status_views"):
        monkeypatch.setattr(facade, method, forbidden)
    path = short_unix_socket_path_factory("health-producer")
    server, thread, _ = _start_server(path=path, facade=facade)
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(1.0)
    try:
        raw.connect(str(path))
        initial = _raw_call(raw, request_id="health:initial", method="healthz", payload=None)
        assert initial["result"]["domain_control_state"] == {
            "schema": "agent_gateway.domain_control_state.v1",
            "runtime_session_id": manifest.runtime_session_id,
            "emergency_stop_latched": False, "emergency_stop_confirmed": False,
            "lower_stop_receipt_ref": None}
        assert initial["result"]["control_fault_reason"] is None
        assert stop_port.calls == []
        if stop_succeeded is not None:
            outcome = service.request_emergency_stop_own_domain(
                request=AgentOwnDomainEmergencyStopRequest(client_id=client_id,
                    control_request_id="stop:health", idempotency_key="stop:health",
                    runtime_session_id=manifest.runtime_session_id,
                    skill_route_manifest_hash=manifest.semantic_content_hash,
                    control_domain_id=manifest.control_domain_id,
                    target_binding_id=manifest.target_binding_id, reason_code="agent_task_aborted"),
                now=10.1)
            assert outcome.accepted is stop_succeeded
        for index in range(2):
            observed = _raw_call(raw, request_id=f"health:{index}", method="healthz", payload=None)
            assert observed["result"]["domain_control_state"] == {
                "schema": "agent_gateway.domain_control_state.v1",
                "runtime_session_id": manifest.runtime_session_id,
                "emergency_stop_latched": stop_succeeded is not None,
                "emergency_stop_confirmed": stop_succeeded is True,
                "lower_stop_receipt_ref": (None if stop_succeeded is None
                    else service.emergency_stop_receipt.receipt_id)}
        assert initial["result"]["domain_control_state"]["emergency_stop_latched"] is False
        assert stop_port.calls == ([] if stop_succeeded is None else ["agent_task_aborted"])
        assert operation_port.receipts == ()
    finally:
        raw.close()
        _stop_server(server, thread)


@pytest.mark.parametrize("reason", ["control_loop_failed:warn80", "fault:" + "x" * 600])
def test_retained_control_fault_fences_submissions_but_preserves_diagnostics(
    short_unix_socket_path_factory, monkeypatch, reason,
) -> None:
    path = short_unix_socket_path_factory("retained-fault")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, close_calls = _start_server(path=path, facade=facade)
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(1.0)
    try:
        raw.connect(str(path))
        payload = {"client_id": "rpent:local", "parameters": {}}
        for method in ("submit_skill_request", "submit_read_only_skill_request"):
            assert _raw_call(raw, request_id=f"before:{method}", method=method,
                             payload=payload)["result"]["accepted"] is True
        facade.requests.clear()
        assert server.retain_control_fault(reason) is True
        assert server.retain_control_fault("later failure must not replace the first") is True
        for invalid in (None, "", " "):
            with pytest.raises(ValueError):
                server.retain_control_fault(invalid)
        assert facade.requests == []
        for method in ("submit_skill_request", "submit_read_only_skill_request"):
            rejected = _raw_call(raw, request_id=f"after:{method}", method=method, payload=payload)
            assert rejected["ok"] is False
            assert rejected["error"] == {"code": "control_fault_latched", "message": reason[:512]}
        assert facade.requests == []
        health = _raw_call(raw, request_id="health:fault", method="healthz", payload=None)
        assert health["result"]["control_fault_reason"] == reason[:512]
        assert health["result"]["domain_control_state"] == facade.get_domain_control_state()
        assert set(health["result"]["domain_control_state"]) == {
            "schema", "runtime_session_id", "emergency_stop_latched",
            "emergency_stop_confirmed", "lower_stop_receipt_ref"}
        # All other methods keep their original dispatch, including controls,
        # operation reconciliation and retained Artifact access.
        operator_lifecycle = {"read_hardware_connection_state", "request_hardware_connection"}
        permitted = unix_transport.AGENT_GATEWAY_UNIX_JSON_METHODS_V2 - {
            "healthz", "submit_skill_request", "submit_read_only_skill_request"} - operator_lifecycle
        for method in sorted(permitted):
            def query(*args, _method=method):
                facade.requests.append((_method, args))
                return {"queried": _method}

            monkeypatch.setattr(facade, method, query)
            response = _raw_call(raw, request_id=f"allowed:{method}", method=method,
                payload=None if method in unix_transport._NO_PAYLOAD_METHODS else payload)
            assert response["result"] == {"queried": method}
        assert [method for method, _ in facade.requests] == sorted(permitted)
        for method in sorted(operator_lifecycle):
            response = _raw_call(raw, request_id=f"unsupported:{method}", method=method,
                payload=None if method in unix_transport._NO_PAYLOAD_METHODS else {})
            assert response["ok"] is False
            assert response["error"]["code"] == "gateway_request_invalid"
        assert [method for method, _ in facade.requests] == sorted(permitted)
        assert facade.disconnect_reasons == []
        assert close_calls == []
        assert server.control_fault_reason == reason[:512]
    finally:
        raw.close()
        _stop_server(server, thread)


def test_retained_control_fault_does_not_drain_or_replay_inflight_submission(
    short_unix_socket_path_factory, monkeypatch,
) -> None:
    path = short_unix_socket_path_factory("fault-inflight")
    facade = _FakeBoundFacade("rpent:local")
    entered, release = threading.Event(), threading.Event()
    submit = facade.submit_skill_request

    def pending_submit(payload):
        entered.set()
        assert release.wait(timeout=2.0)
        return submit(payload)

    monkeypatch.setattr(facade, "submit_skill_request", pending_submit)
    server, thread, _ = _start_server(path=path, facade=facade)
    worker_channel, query_channel = socket.socket(socket.AF_UNIX), socket.socket(socket.AF_UNIX)
    pool = ThreadPoolExecutor(max_workers=1)
    for channel in (worker_channel, query_channel):
        channel.settimeout(2.0)
    try:
        worker_channel.connect(str(path))
        query_channel.connect(str(path))
        pending = pool.submit(_raw_call, worker_channel, request_id="inflight:submit",
            method="submit_skill_request", payload={"client_id": "rpent:local"})
        assert entered.wait(timeout=1.0)
        assert server.retain_control_fault("control_loop_failed") is True
        assert not pending.done()
        blocked = _raw_call(query_channel, request_id="new:submit",
            method="submit_skill_request", payload={"client_id": "rpent:local"})
        assert blocked["error"]["code"] == "control_fault_latched"
        observed = _raw_call(query_channel, request_id="inflight:query",
            method="get_own_operation", payload={"client_id": "rpent:local"})
        assert observed["result"]["reason_code"] == "fake_not_found"
        release.set()
        assert pending.result(timeout=1.0)["result"]["accepted"] is True
        assert [method for method, _ in facade.requests].count("submit_skill_request") == 1
        assert server.control_fault_reason == "control_loop_failed"
    finally:
        release.set()
        pool.shutdown(wait=True)
        worker_channel.close()
        query_channel.close()
        _stop_server(server, thread)


def test_peer_map_is_one_to_one_and_rejects_unconfigured_uid() -> None:
    mapping = AgentGatewayUnixPeerClientMap({1001: "rpent:local"})
    assert mapping.resolve(
        AgentGatewayUnixPeerCredentials(pid=4, uid=1001, gid=1001)
    ) == "rpent:local"
    with pytest.raises(PermissionError, match="not authorized"):
        mapping.resolve(
            AgentGatewayUnixPeerCredentials(pid=4, uid=1002, gid=1002)
        )
    with pytest.raises(ValueError, match="exactly one peer UID"):
        AgentGatewayUnixPeerClientMap(
            {1001: "rpent:local", 1002: "rpent:local"}
        )


def test_server_preflight_reports_encoded_path_budget_without_relocation() -> None:
    overlong = "/" + "a" * 108

    with pytest.raises(ValueError) as captured:
        unix_transport.validate_agent_gateway_unix_server_preflight(
            socket_path=overlong,
        )

    message = str(captured.value)
    assert "uses 109 encoded bytes" in message
    assert "at most 107" in message
    assert "will not relocate it automatically" in message


def test_server_refuses_existing_endpoint_and_unlinks_only_its_own_socket(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("ownership")
    path.touch()
    try:
        with pytest.raises(FileExistsError, match="Refusing to replace"):
            AgentGatewayUnixJsonServer(
                socket_path=path,
                peer_client_map=AgentGatewayUnixPeerClientMap(
                    {os.getuid(): "rpent:local"}
                ),
                facade_factory=lambda _: _FakeBoundFacade("rpent:local"),
                close_gateway=lambda: None,
            )
        assert path.is_file()
    finally:
        path.unlink()

    facade = _FakeBoundFacade("rpent:local")
    server, thread, close_calls = _start_server(path=path, facade=facade)
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    _stop_server(server, thread)
    assert not path.exists()
    assert len(close_calls) == 1


def test_server_close_retries_gateway_after_transport_is_already_closed(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("close-retry")
    close_calls: list[int] = []

    def close_gateway():
        close_calls.append(len(close_calls) + 1)
        return SimpleNamespace(
            accepted=len(close_calls) > 1,
            reason_code=(
                "station_hold_confirmed" if len(close_calls) > 1 else "hold_failed"
            ),
        )

    server = AgentGatewayUnixJsonServer(
        socket_path=path,
        peer_client_map=AgentGatewayUnixPeerClientMap(
            {os.getuid(): "rpent:local"}
        ),
        facade_factory=lambda _: _FakeBoundFacade("rpent:local"),
        close_gateway=close_gateway,
    )

    server.server_close()
    assert close_calls == [1]
    assert not path.exists()

    server.server_close()
    assert close_calls == [1, 2]


def test_group_shared_server_requires_setgid_parent_and_inherits_group(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("group")
    facade = _FakeBoundFacade("rpent:local")
    with pytest.raises(ValueError, match="must be setgid"):
        _start_server(path=path, facade=facade, socket_mode=0o660)

    os.chmod(path.parent, 0o2750)
    server, thread, close_calls = _start_server(
        path=path,
        facade=facade,
        socket_mode=0o660,
    )
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o660
    assert os.lstat(path).st_gid == os.lstat(path.parent).st_gid
    _stop_server(server, thread)
    assert len(close_calls) == 1


def test_server_cleans_its_bound_inode_when_permission_setup_fails(
    monkeypatch,
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("chmod")

    def reject_chmod(*_args, **_kwargs):
        raise PermissionError("injected chmod failure")

    monkeypatch.setattr(unix_transport.os, "chmod", reject_chmod)
    with pytest.raises(PermissionError, match="injected chmod failure"):
        AgentGatewayUnixJsonServer(
            socket_path=path,
            peer_client_map=AgentGatewayUnixPeerClientMap(
                {os.getuid(): "rpent:local"}
            ),
            facade_factory=lambda _: _FakeBoundFacade("rpent:local"),
            close_gateway=lambda: None,
        )
    assert not os.path.lexists(path)


@pytest.mark.integration
@pytest.mark.optional
def test_live_rpent_client_authenticates_heartbeats_and_binds_payload_identity(
    short_unix_socket_path_factory,
) -> None:
    transport_module = pytest.importorskip(
        "rpent_dexteleop.gateway.unix_transport",
        reason="paired RPent integration wheel is not installed",
    )
    DexTeleopGatewayTransportError = (
        transport_module.DexTeleopGatewayTransportError
    )
    UnixDexTeleopGatewayClient = transport_module.UnixDexTeleopGatewayClient

    path = short_unix_socket_path_factory("live-client")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, close_calls = _start_server(
        path=path,
        facade=facade,
        peer_idle_timeout_sec=0.3,
    )
    clients = []
    try:
        # Each current client consumes one of the four bounded channels.
        for _ in range(4):
            clients.append(UnixDexTeleopGatewayClient(
                socket_path=path,
                client_id="rpent:local",
                expected_server_uid=os.getuid(),
                expected_server_gid=os.getgid(),
                heartbeat_interval_sec=0.05,
            ))
        client = clients[0]
        assert client.get_skill_definition_catalog() == {
            "catalog": "definitions"
        }
        assert client.get_agent_skill_schema_catalog() == {"catalog": "schemas"}
        assert client.get_skill_route_manifest() == {"manifest": "routes"}
        assert client.get_observation_snapshot() == {"snapshot": "current"}
        admitted = client.submit_skill_request(
            {"client_id": "rpent:local", "parameters": {}}
        )
        assert admitted["accepted"]

        time.sleep(0.65)
        for channel in clients:
            assert channel.get_observation_snapshot() == {"snapshot": "current"}
        assert not server.disconnect_events

        with pytest.raises(DexTeleopGatewayTransportError):
            UnixDexTeleopGatewayClient(
                socket_path=path,
                client_id="rpent:local",
                expected_server_uid=os.getuid(),
                expected_server_gid=os.getgid(),
                heartbeat_interval_sec=0.05,
            )

        with pytest.raises(
            DexTeleopGatewayTransportError,
            match="permission_denied",
        ):
            client.submit_skill_request(
                {"client_id": "another-client", "parameters": {}}
            )
        _wait_for(lambda: len(server.disconnect_events) == 1)
        event = server.disconnect_events[0]
        assert event.client_id == "rpent:local"
        assert event.reason == "authenticated_peer_request_rejected"
        assert event.outcome_accepted
        assert facade.disconnect_reasons == [
            "authenticated_peer_request_rejected"
        ]
    finally:
        for channel in clients:
            channel.close()
        _stop_server(server, thread)
    assert len(close_calls) == 1


def test_idle_or_oversized_authenticated_peer_disconnects_fail_closed(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("idle")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(
        path=path,
        facade=facade,
        peer_idle_timeout_sec=0.25,
        max_frame_bytes=1024,
    )
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        raw.connect(str(path))
        _wait_for(lambda: server.active_client_ids == ("rpent:local",))
        _wait_for(lambda: len(server.disconnect_events) == 1)
        assert server.disconnect_events[0].reason == (
            "authenticated_peer_idle_timeout"
        )
    finally:
        raw.close()
        _stop_server(server, thread)

    path = short_unix_socket_path_factory("oversized")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(
        path=path,
        facade=facade,
        max_frame_bytes=1024,
    )
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        raw.connect(str(path))
        raw.sendall(_PREFIX.pack(1025))
        _wait_for(lambda: len(server.disconnect_events) == 1)
        assert server.disconnect_events[0].reason == (
            "authenticated_peer_protocol_violation"
        )
    finally:
        raw.close()
        _stop_server(server, thread)


def test_reserved_safety_channels_survive_bounded_task_connections(
    short_unix_socket_path_factory,
    caplog,
) -> None:
    path = short_unix_socket_path_factory("dual-channel")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    primary = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    emergency = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bootstrap = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    task_emergency = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    rejected = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for channel in (primary, emergency, bootstrap, task_emergency, rejected):
        channel.settimeout(1.0)
    try:
        primary.connect(str(path))
        emergency.connect(str(path))
        assert _raw_call(
            primary,
            request_id="request:primary-health",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        assert _raw_call(
            emergency,
            request_id="request:emergency-health",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        bootstrap.connect(str(path))
        assert _raw_call(
            bootstrap,
            request_id="request:bootstrap-health",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        task_emergency.connect(str(path))
        assert _raw_call(
            task_emergency,
            request_id="request:task-emergency-health",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        _wait_for(lambda: server.active_client_ids == ("rpent:local",))

        rejected.connect(str(path))
        assert rejected.recv(1) == b""
        assert "client_id=rpent:local active=4 limit=4" in caplog.text
        assert not server.disconnect_events
        assert not facade.disconnect_reasons
        # Refusing an excess socket must not evict an admitted
        # channel or invoke the admitted client's disconnect-to-HOLD path.
        for index, channel in enumerate((primary, emergency, bootstrap, task_emergency)):
            assert _raw_call(
                channel,
                request_id=f"request:after-rejection:{index}",
                method="healthz",
                payload=None,
            )["result"]["status"] == "ok"

        bootstrap.close()
        _wait_for(lambda: len(server.disconnect_events) == 1)
        assert server.active_client_ids == ("rpent:local",)
        assert facade.disconnect_reasons == ["authenticated_peer_eof"]

        task_emergency.close()
        _wait_for(lambda: len(server.disconnect_events) == 2)
        assert server.active_client_ids == ("rpent:local",)

        primary.close()
        _wait_for(lambda: len(server.disconnect_events) == 3)
        assert server.active_client_ids == ("rpent:local",)

        emergency.close()
        _wait_for(lambda: len(server.disconnect_events) == 4)
        assert server.active_client_ids == ()
        assert facade.disconnect_reasons == [
            "authenticated_peer_eof",
            "authenticated_peer_eof",
            "authenticated_peer_eof",
            "authenticated_peer_eof",
        ]
    finally:
        primary.close()
        emergency.close()
        bootstrap.close()
        task_emergency.close()
        rejected.close()
        _stop_server(server, thread)


def test_session_channels_survive_sequential_task_connection_pairs(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("session-tasks")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    channels = []

    def open_channel(label):
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channels.append(channel)
        channel.settimeout(1.0)
        channel.connect(str(path))
        assert _raw_call(
            channel,
            request_id=f"request:open:{label}",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        return channel

    try:
        # Bootstrap must finish before the session and task pairs overlap.
        open_channel("bootstrap").close()
        _wait_for(lambda: len(server.disconnect_events) == 1)
        primary, emergency = open_channel("session"), open_channel("session-stop")
        for cycle in range(3):
            task = open_channel(f"task:{cycle}")
            task_stop = open_channel(f"task-stop:{cycle}")
            task.close()
            task_stop.close()
            _wait_for(lambda: len(server.disconnect_events) == 3 + 2 * cycle)
            assert _raw_call(
                primary,
                request_id=f"request:session:{cycle}",
                method="get_observation_snapshot",
                payload=None,
            )["result"] == {"snapshot": "current"}
            # A fake facade proves transport independence, not hardware stop.
            assert _raw_call(
                emergency,
                request_id=f"request:session-stop:{cycle}",
                method="healthz",
                payload=None,
            )["result"]["status"] == "ok"
        assert facade.disconnect_reasons == ["authenticated_peer_eof"] * 7
        assert not any(name == "submit_skill_request" for name, _ in facade.requests)
    finally:
        for channel in channels:
            channel.close()
        _stop_server(server, thread)


def test_connection_handoff_tolerates_delayed_closed_peer_retirement(
    short_unix_socket_path_factory,
    monkeypatch,
) -> None:
    path = short_unix_socket_path_factory("handoff")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    channels = []
    delayed = set()
    retiring = set()
    release = threading.Event()
    original = server._release_authenticated_connection

    def delayed_retirement(**kwargs):
        if kwargs["sock"] in delayed:
            retiring.add(kwargs["sock"])
            assert release.wait(1.0)
        return original(**kwargs)

    def open_channel(index):
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channels.append(channel)
        channel.settimeout(1.0)
        channel.connect(str(path))
        assert _raw_call(
            channel,
            request_id=f"request:handoff:{index}",
            method="healthz",
            payload=None,
        )["result"]["status"] == "ok"
        return channel

    monkeypatch.setattr(server, "_release_authenticated_connection", delayed_retirement)
    timer = None
    try:
        for index in range(4):
            open_channel(index)
        with server._connection_lock:
            delayed.update(list(server._active_connections["rpent:local"].values())[2:])
        channels[2].close()
        channels[3].close()
        _wait_for(lambda: len(retiring) == 2)
        # Local close has returned, but the server still accounts for both old
        # task sockets. The caller must not need a private server-side barrier.
        timer = threading.Timer(0.05, release.set)
        timer.start()
        open_channel(4)
        open_channel(5)
        _wait_for(lambda: len(server.disconnect_events) == 2)
        assert facade.disconnect_reasons == ["authenticated_peer_eof"] * 2
    finally:
        release.set()
        if timer is not None:
            timer.join(timeout=1.0)
        for channel in channels:
            channel.close()
        _stop_server(server, thread)


def test_transport_close_rejects_waiting_peer_before_facade_binding(
    short_unix_socket_path_factory,
    monkeypatch,
) -> None:
    path = short_unix_socket_path_factory("close-waiter")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    channels = []
    factory_calls = []
    waiting, retire = threading.Event(), threading.Event()
    original_wait = server._connection_available.wait_for
    original_release = server._release_authenticated_connection

    def wait_for_slot(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    def delayed_release(**kwargs):
        assert retire.wait(1.0)
        return original_release(**kwargs)

    def factory(client_id):
        factory_calls.append(client_id)
        return facade

    def query(channel, index):
        try:
            return _raw_call(channel, request_id=f"request:close:{index}",
                             method="healthz", payload=None)
        except (OSError, AssertionError):
            return None

    monkeypatch.setattr(server, "_facade_factory", factory)
    monkeypatch.setattr(server._connection_available, "wait_for", wait_for_slot)
    monkeypatch.setattr(server, "_release_authenticated_connection", delayed_release)
    try:
        for index in range(5):
            channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            channels.append(channel)
            channel.settimeout(1.0)
            channel.connect(str(path))
            if index < 4:
                assert query(channel, index)["result"]["status"] == "ok"
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(query, channels[-1], 4)
            try:
                assert waiting.wait(1.0)
                server.shutdown()
                server.server_close()
            finally:
                retire.set()
            assert pending.result(timeout=1.0) is None
        assert len(factory_calls) == 4
        _wait_for(lambda: not server.active_client_ids)
    finally:
        retire.set()
        for channel in channels:
            channel.close()
        _stop_server(server, thread)


def test_protocol_envelope_is_exact_and_method_allowlist_has_no_shutdown(
    short_unix_socket_path_factory,
) -> None:
    assert "shutdown" not in {
        "healthz",
        "get_skill_definition_catalog",
        "get_agent_skill_schema_catalog",
        "get_skill_route_manifest",
        "get_observation_snapshot",
        "submit_skill_request",
        "get_own_operation",
        "cancel_own_operation",
        "request_hold_own_domain",
        "request_emergency_stop_own_domain",
    }
    path = short_unix_socket_path_factory("allowlist")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(1.0)
    try:
        raw.connect(str(path))
        envelope = {
            "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
            "request_id": "request:forbidden",
            "method": "shutdown",
            "payload": None,
        }
        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        raw.sendall(_PREFIX.pack(len(body)) + body)
        (size,) = _PREFIX.unpack(_recv_exact(raw, _PREFIX.size))
        response = json.loads(_recv_exact(raw, size))
        assert not response["ok"]
        assert response["error"]["code"] == "method_not_allowed"
        assert "traceback" not in response
        _wait_for(lambda: len(server.disconnect_events) == 1)
    finally:
        raw.close()
        _stop_server(server, thread)


def test_hardware_views_and_read_only_submission_are_additive_methods(
    short_unix_socket_path_factory,
) -> None:
    path = short_unix_socket_path_factory("hardware-ro")
    facade = _FakeBoundFacade("rpent:local")
    server, thread, _ = _start_server(path=path, facade=facade)
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.settimeout(1.0)
    try:
        raw.connect(str(path))
        resources = _raw_call(
            raw,
            request_id="request:hardware-resources",
            method="get_hardware_resource_views",
            payload=None,
        )
        statuses = _raw_call(
            raw,
            request_id="request:hardware-statuses",
            method="get_hardware_status_views",
            payload=None,
        )
        submitted = _raw_call(
            raw,
            request_id="request:readonly-submit",
            method="submit_read_only_skill_request",
            payload={"client_id": "rpent:local"},
        )
        emergency_stopped = _raw_call(
            raw,
            request_id="request:emergency-stop",
            method="request_emergency_stop_own_domain",
            payload={"client_id": "rpent:local"},
        )
        assert resources["result"] == {"resources": []}
        assert statuses["result"] == {"statuses": []}
        assert submitted["result"]["reason_code"] == "fake_read_only_admitted"
        assert emergency_stopped["result"]["reason_code"] == (
            "fake_emergency_stop_confirmed"
        )
    finally:
        raw.close()
        _stop_server(server, thread)

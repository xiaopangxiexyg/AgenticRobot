from __future__ import annotations

import json
import os
import socket
import stat
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, close_calls


def _stop_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


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
    client = None
    try:
        client = UnixDexTeleopGatewayClient(
            socket_path=path,
            client_id="rpent:local",
            expected_server_uid=os.getuid(),
            expected_server_gid=os.getgid(),
            heartbeat_interval_sec=0.05,
        )
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
        assert client.get_observation_snapshot() == {"snapshot": "current"}
        assert not server.disconnect_events

        with pytest.raises(DexTeleopGatewayTransportError):
            UnixDexTeleopGatewayClient(
                socket_path=path,
                client_id="rpent:local",
                expected_server_uid=os.getuid(),
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
        if client is not None:
            client.close()
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
        assert resources["result"] == {"resources": []}
        assert statuses["result"] == {"statuses": []}
        assert submitted["result"]["reason_code"] == "fake_read_only_admitted"
    finally:
        raw.close()
        _stop_server(server, thread)

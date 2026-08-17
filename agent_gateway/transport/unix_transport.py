"""Authenticated, bounded local Unix transport for the Agent Gateway.

This module owns only a byte/identity boundary.  It does not construct or
start a simulator, device, model, Station application, or operation worker.
The caller supplies a factory for an already-bound Gateway facade and remains
the owner of ``serve_forever``/shutdown process lifecycle.

Linux ``SO_PEERCRED`` is authoritative for the connecting process identity.
The JSON payload's ``client_id`` is never used to authenticate a connection.
One persistent connection is allowed per configured client identity so a
connection EOF/idle timeout has unambiguous disconnect-to-HOLD semantics.
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import stat
import struct
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .wire_adapter import BoundAgentGatewayWireAdapter

AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2 = "dexteleop.agent_gateway.unix_json.v2"
AGENT_GATEWAY_UNIX_JSON_METHODS_V2 = frozenset(
    {
        "healthz",
        "get_skill_definition_catalog",
        "get_agent_skill_schema_catalog",
        "get_skill_route_manifest",
        "get_skill_io_catalog",
        "get_gateway_session_capabilities",
        "get_gateway_events",
        "get_observation_snapshot",
        "submit_skill_request",
        "get_own_operation",
        "cancel_own_operation",
        "request_hold_own_domain",
        "begin_artifact_upload",
        "write_artifact_chunk",
        "commit_artifact_upload",
        "begin_artifact_download",
        "read_artifact_chunk",
        "acknowledge_artifact_download",
        "acknowledge_artifact_materialization",
    }
)

DEFAULT_AGENT_GATEWAY_UNIX_MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_AGENT_GATEWAY_UNIX_FRAME_BYTES = 16 * 1024 * 1024
DEFAULT_AGENT_GATEWAY_UNIX_PEER_IDLE_TIMEOUT_SEC = 10.0
MAX_AGENT_GATEWAY_UNIX_SOCKET_PATH_BYTES = 107

_FRAME_PREFIX = struct.Struct(">I")
_PEER_CREDENTIALS = struct.Struct("3i")
_NO_PAYLOAD_METHODS = frozenset(
    {
        "healthz",
        "get_skill_definition_catalog",
        "get_agent_skill_schema_catalog",
        "get_skill_route_manifest",
        "get_skill_io_catalog",
        "get_gateway_session_capabilities",
        "get_observation_snapshot",
    }
)


class AgentGatewayUnixProtocolError(RuntimeError):
    """A bounded protocol violation safe to project without a traceback."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id


@dataclass(frozen=True)
class AgentGatewayUnixPeerCredentials:
    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        for field_name in ("pid", "uid", "gid"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")


@dataclass(frozen=True)
class AgentGatewayUnixDisconnectEvent:
    client_id: str
    peer: AgentGatewayUnixPeerCredentials
    reason: str
    outcome_accepted: bool | None
    outcome_reason_code: str | None
    error_type: str | None


class AgentGatewayUnixPeerClientMap:
    """Immutable, one-to-one Linux UID to Gateway client identity map."""

    def __init__(self, client_id_by_uid: Mapping[int, str]) -> None:
        if not isinstance(client_id_by_uid, Mapping) or not client_id_by_uid:
            raise ValueError("client_id_by_uid must be a non-empty mapping.")
        normalized: dict[int, str] = {}
        seen_client_ids: set[str] = set()
        for uid, client_id in client_id_by_uid.items():
            if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
                raise ValueError("peer UID keys must be non-negative integers.")
            identity = _required_text(
                client_id,
                field_name="client_id_by_uid value",
                max_length=128,
            )
            if identity in seen_client_ids:
                raise ValueError(
                    "Each Gateway client_id must map to exactly one peer UID."
                )
            normalized[uid] = identity
            seen_client_ids.add(identity)
        self._client_id_by_uid = normalized

    @property
    def client_id_by_uid(self) -> Mapping[int, str]:
        return dict(self._client_id_by_uid)

    def resolve(self, peer: AgentGatewayUnixPeerCredentials) -> str:
        if not isinstance(peer, AgentGatewayUnixPeerCredentials):
            raise TypeError("peer must be AgentGatewayUnixPeerCredentials.")
        client_id = self._client_id_by_uid.get(peer.uid)
        if client_id is None:
            raise PermissionError("Unix peer UID is not authorized.")
        return client_id


@dataclass(frozen=True)
class _ValidatedRequest:
    request_id: str
    method: str
    payload: Mapping[str, Any] | None


def _required_text(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters.")
    return value


def _frame_limit(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1024 <= value <= MAX_AGENT_GATEWAY_UNIX_FRAME_BYTES
    ):
        raise ValueError("max_frame_bytes must be an integer in [1024, 16777216].")
    return value


def _peer_idle_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("peer_idle_timeout_sec must be a finite number.")
    timeout = float(value)
    if not 0.25 <= timeout <= 300.0:
        raise ValueError("peer_idle_timeout_sec must be in [0.25, 300].")
    return timeout


def _socket_mode(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("socket_mode must be an integer permission mode.")
    if value not in {0o600, 0o660}:
        raise ValueError("socket_mode must be exactly 0o600 or 0o660.")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object field: {key!r}")
        result[key] = value
    return result


def _recv_exact_or_eof(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise AgentGatewayUnixProtocolError(
                "truncated_frame",
                f"connection closed after {len(chunks)}/{size} bytes",
            )
        chunks.extend(chunk)
    return bytes(chunks)


def _read_json_frame(
    sock: socket.socket,
    *,
    max_frame_bytes: int,
) -> Mapping[str, Any] | None:
    prefix = _recv_exact_or_eof(sock, _FRAME_PREFIX.size)
    if prefix is None:
        return None
    (length,) = _FRAME_PREFIX.unpack(prefix)
    if not 2 <= length <= max_frame_bytes:
        raise AgentGatewayUnixProtocolError(
            "frame_size_invalid",
            f"frame length {length} is outside [2, {max_frame_bytes}]",
        )
    body = _recv_exact_or_eof(sock, length)
    if body is None:
        raise AgentGatewayUnixProtocolError(
            "truncated_frame",
            "connection closed before the frame body",
        )
    try:
        decoded = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise AgentGatewayUnixProtocolError(
            "json_invalid",
            str(error),
        ) from error
    if not isinstance(decoded, Mapping):
        raise AgentGatewayUnixProtocolError(
            "request_envelope_invalid",
            "request envelope must be a JSON object",
        )
    return decoded


def _encode_json_frame(value: object, *, max_frame_bytes: int) -> bytes:
    try:
        body = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AgentGatewayUnixProtocolError(
            "response_not_json_serializable",
            str(error),
        ) from error
    if not body or len(body) > max_frame_bytes:
        raise AgentGatewayUnixProtocolError(
            "response_frame_size_invalid",
            f"encoded response has {len(body)} bytes; limit is {max_frame_bytes}",
        )
    return _FRAME_PREFIX.pack(len(body)) + body


def _validate_request(payload: Mapping[str, Any]) -> _ValidatedRequest:
    request_id_value = payload.get("request_id")
    request_id = (
        request_id_value
        if isinstance(request_id_value, str)
        and request_id_value.strip() == request_id_value
        and 1 <= len(request_id_value) <= 128
        else None
    )
    if set(payload) != {"protocol", "request_id", "method", "payload"}:
        raise AgentGatewayUnixProtocolError(
            "request_envelope_invalid",
            "request fields must be exactly protocol/request_id/method/payload",
            request_id=request_id,
        )
    if payload["protocol"] != AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2:
        raise AgentGatewayUnixProtocolError(
            "protocol_mismatch",
            "request protocol is not supported",
            request_id=request_id,
        )
    if request_id is None:
        raise AgentGatewayUnixProtocolError(
            "request_id_invalid",
            "request_id must be a trimmed string of 1..128 characters",
        )
    method_value = payload["method"]
    if not isinstance(method_value, str) or method_value not in (
        AGENT_GATEWAY_UNIX_JSON_METHODS_V2
    ):
        raise AgentGatewayUnixProtocolError(
            "method_not_allowed",
            "request method is not in the fixed Gateway allowlist",
            request_id=request_id,
        )
    body = payload["payload"]
    if method_value in _NO_PAYLOAD_METHODS:
        if body is not None:
            raise AgentGatewayUnixProtocolError(
                "method_payload_invalid",
                f"{method_value} requires a null payload",
                request_id=request_id,
            )
        normalized_body = None
    else:
        if not isinstance(body, Mapping):
            raise AgentGatewayUnixProtocolError(
                "method_payload_invalid",
                f"{method_value} requires an object payload",
                request_id=request_id,
            )
        normalized_body = dict(body)
    return _ValidatedRequest(
        request_id=request_id,
        method=method_value,
        payload=normalized_body,
    )


def _peer_credentials(sock: socket.socket) -> AgentGatewayUnixPeerCredentials:
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("Linux SO_PEERCRED is required for this transport.")
    raw = sock.getsockopt(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        _PEER_CREDENTIALS.size,
    )
    pid, uid, gid = _PEER_CREDENTIALS.unpack(raw)
    return AgentGatewayUnixPeerCredentials(pid=pid, uid=uid, gid=gid)


def _success_response(request_id: str, result: object) -> Mapping[str, Any]:
    return {
        "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
        "request_id": request_id,
        "ok": True,
        "result": result,
    }


def _error_response(
    request_id: str,
    *,
    code: str,
    message: str,
) -> Mapping[str, Any]:
    return {
        "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": _required_text(code, field_name="error code", max_length=96),
            "message": _required_text(
                message[:512] or "Gateway request failed",
                field_name="error message",
                max_length=512,
            ),
        },
    }


def _public_error(error: BaseException) -> tuple[str, str, bool]:
    if isinstance(error, PermissionError):
        return "permission_denied", str(error) or "permission denied", True
    if isinstance(error, (TypeError, ValueError)):
        return "gateway_request_invalid", str(error) or "invalid request", False
    if isinstance(error, RuntimeError):
        return "gateway_request_failed", str(error) or "Gateway request failed", False
    return "gateway_internal_error", type(error).__name__, True


class _AgentGatewayUnixRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: AgentGatewayUnixJsonServer = self.server  # type: ignore[assignment]
        facade: BoundAgentGatewayWireAdapter | None = None
        client_id: str | None = None
        peer: AgentGatewayUnixPeerCredentials | None = None
        disconnect_reason = "authenticated_peer_eof"
        try:
            client_id, peer, facade = server._open_authenticated_connection(
                self.request
            )
            self.request.settimeout(server.peer_idle_timeout_sec)
            while True:
                try:
                    envelope = _read_json_frame(
                        self.request,
                        max_frame_bytes=server.max_frame_bytes,
                    )
                except TimeoutError:
                    disconnect_reason = "authenticated_peer_idle_timeout"
                    break
                except AgentGatewayUnixProtocolError as error:
                    disconnect_reason = "authenticated_peer_protocol_violation"
                    if error.request_id is not None:
                        server._try_write_response(
                            self.request,
                            _error_response(
                                error.request_id,
                                code=error.code,
                                message=str(error),
                            ),
                        )
                    break
                if envelope is None:
                    break
                try:
                    request = _validate_request(envelope)
                except AgentGatewayUnixProtocolError as error:
                    disconnect_reason = "authenticated_peer_protocol_violation"
                    if error.request_id is not None:
                        server._try_write_response(
                            self.request,
                            _error_response(
                                error.request_id,
                                code=error.code,
                                message=str(error),
                            ),
                        )
                    break

                response, terminate = server._dispatch_request(
                    facade=facade,
                    client_id=client_id,
                    request=request,
                )
                if not server._try_write_response(self.request, response):
                    disconnect_reason = "authenticated_peer_response_write_failed"
                    break
                if terminate:
                    disconnect_reason = "authenticated_peer_request_rejected"
                    break
        except PermissionError:
            disconnect_reason = "unauthorized_peer_rejected"
        except (OSError, RuntimeError, struct.error):
            disconnect_reason = "authenticated_peer_transport_failure"
        finally:
            if client_id is not None and peer is not None and facade is not None:
                server._release_authenticated_connection(
                    client_id=client_id,
                    peer=peer,
                    facade=facade,
                    sock=self.request,
                    reason=disconnect_reason,
                )


class _ThreadingUnixStreamServerMixIn(socketserver.ThreadingMixIn):
    daemon_threads = True
    block_on_close = False


class AgentGatewayUnixJsonServer(
    _ThreadingUnixStreamServerMixIn,
    socketserver.UnixStreamServer,
):
    """DexTeleop-owned authenticated server; construction does not start it."""

    allow_reuse_address = False

    def __init__(
        self,
        *,
        socket_path: str | os.PathLike[str],
        peer_client_map: AgentGatewayUnixPeerClientMap,
        facade_factory: Callable[[str], BoundAgentGatewayWireAdapter],
        close_gateway: Callable[[], Any],
        socket_mode: int = 0o600,
        max_frame_bytes: int = DEFAULT_AGENT_GATEWAY_UNIX_MAX_FRAME_BYTES,
        peer_idle_timeout_sec: float = (
            DEFAULT_AGENT_GATEWAY_UNIX_PEER_IDLE_TIMEOUT_SEC
        ),
        max_disconnect_events: int = 64,
    ) -> None:
        path = os.fspath(socket_path)
        if not isinstance(path, str) or not path or not os.path.isabs(path):
            raise ValueError("socket_path must be a non-empty absolute path.")
        if "\x00" in path:
            raise ValueError("Abstract or NUL-containing Unix paths are forbidden.")
        if len(os.fsencode(path)) > MAX_AGENT_GATEWAY_UNIX_SOCKET_PATH_BYTES:
            raise ValueError(
                "socket_path exceeds the Linux Unix-socket 107-byte limit."
            )
        if not isinstance(peer_client_map, AgentGatewayUnixPeerClientMap):
            raise TypeError("peer_client_map must be AgentGatewayUnixPeerClientMap.")
        if not callable(facade_factory):
            raise TypeError("facade_factory must be callable.")
        if not callable(close_gateway):
            raise TypeError("close_gateway must be callable.")
        if (
            isinstance(max_disconnect_events, bool)
            or not isinstance(max_disconnect_events, int)
            or not 1 <= max_disconnect_events <= 4096
        ):
            raise ValueError("max_disconnect_events must be in [1, 4096].")

        parent = os.path.dirname(path)
        try:
            parent_stat = os.lstat(parent)
        except OSError as error:
            raise ValueError(
                "Unix socket parent directory must already exist."
            ) from error
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(
                "Unix socket parent must be a real directory, not a symlink."
            )
        if parent_stat.st_mode & stat.S_IWOTH:
            raise ValueError("Unix socket parent directory cannot be world-writable.")
        if os.path.lexists(path):
            raise FileExistsError(
                "Refusing to replace an existing Unix socket path; remove only "
                "the verified stale endpoint explicitly."
            )

        self.socket_path = path
        self.socket_mode = _socket_mode(socket_mode)
        self.max_frame_bytes = _frame_limit(max_frame_bytes)
        self.peer_idle_timeout_sec = _peer_idle_timeout(peer_idle_timeout_sec)
        self.peer_client_map = peer_client_map
        self._facade_factory = facade_factory
        self._close_gateway = close_gateway
        self._connection_lock = threading.RLock()
        self._active_connections: dict[str, socket.socket] = {}
        self._disconnect_events: deque[AgentGatewayUnixDisconnectEvent] = deque(
            maxlen=max_disconnect_events
        )
        self._close_lock = threading.Lock()
        self._transport_closed = False
        self._gateway_close_outcome: Any | None = None
        self._gateway_close_error: BaseException | None = None
        self._bound_endpoint_identity: tuple[int, int] | None = None
        self._bound_here = False

        super().__init__(
            path,
            _AgentGatewayUnixRequestHandler,
            bind_and_activate=False,
        )
        try:
            self.server_bind()
            self._bound_here = True
            endpoint = os.lstat(path)
            if not stat.S_ISSOCK(endpoint.st_mode):
                raise RuntimeError("Bound Agent Gateway endpoint is not a socket.")
            self._bound_endpoint_identity = (endpoint.st_dev, endpoint.st_ino)
            os.chmod(path, self.socket_mode)
            secured_endpoint = os.lstat(path)
            if (
                not stat.S_ISSOCK(secured_endpoint.st_mode)
                or (secured_endpoint.st_dev, secured_endpoint.st_ino)
                != self._bound_endpoint_identity
            ):
                raise RuntimeError(
                    "Unix socket endpoint identity changed while securing it."
                )
            self.server_activate()
        except BaseException:
            try:
                socketserver.UnixStreamServer.server_close(self)
            finally:
                self._unlink_owned_endpoint()
            raise

    @property
    def active_client_ids(self) -> tuple[str, ...]:
        with self._connection_lock:
            return tuple(sorted(self._active_connections))

    @property
    def disconnect_events(self) -> tuple[AgentGatewayUnixDisconnectEvent, ...]:
        with self._connection_lock:
            return tuple(self._disconnect_events)

    @property
    def gateway_close_outcome(self) -> Any | None:
        return self._gateway_close_outcome

    @property
    def gateway_close_error(self) -> BaseException | None:
        return self._gateway_close_error

    def _open_authenticated_connection(
        self,
        sock: socket.socket,
    ) -> tuple[
        str,
        AgentGatewayUnixPeerCredentials,
        BoundAgentGatewayWireAdapter,
    ]:
        peer = _peer_credentials(sock)
        client_id = self.peer_client_map.resolve(peer)
        with self._connection_lock:
            if client_id in self._active_connections:
                raise PermissionError(
                    "Only one persistent connection is allowed per client_id."
                )
            self._active_connections[client_id] = sock
        try:
            facade = self._facade_factory(client_id)
            if not isinstance(facade, BoundAgentGatewayWireAdapter):
                raise TypeError(
                    "facade_factory must return BoundAgentGatewayWireAdapter."
                )
            if facade.authenticated_client_id != client_id:
                raise ValueError(
                    "facade_factory returned a differently bound client identity."
                )
            service = getattr(facade, "service", None)
            protocol_runtime = getattr(service, "protocol_runtime", None)
            if protocol_runtime is not None:
                capabilities = protocol_runtime.session_capabilities
                if (
                    capabilities.max_control_request_bytes != self.max_frame_bytes
                    or capabilities.max_control_response_bytes != self.max_frame_bytes
                ):
                    raise ValueError(
                        "Session control bounds must match the Unix frame bound."
                    )
                if capabilities.artifact_transfer_enabled:
                    encoded_chunk_bound = (
                        4 * ((capabilities.max_artifact_chunk_bytes + 2) // 3) + 65_536
                    )
                    if encoded_chunk_bound > self.max_frame_bytes:
                        raise ValueError(
                            "Artifact chunk plus base64/control overhead exceeds "
                            "the Unix frame bound."
                        )
            return client_id, peer, facade
        except BaseException:
            with self._connection_lock:
                if self._active_connections.get(client_id) is sock:
                    self._active_connections.pop(client_id, None)
            raise

    def _dispatch_request(
        self,
        *,
        facade: BoundAgentGatewayWireAdapter,
        client_id: str,
        request: _ValidatedRequest,
    ) -> tuple[Mapping[str, Any], bool]:
        try:
            if request.method == "healthz":
                result: object = {
                    "status": "ok",
                    "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
                    "authenticated_client_id": client_id,
                    "max_frame_bytes": self.max_frame_bytes,
                    "peer_idle_timeout_sec": self.peer_idle_timeout_sec,
                }
            elif request.method == "get_skill_definition_catalog":
                result = facade.get_skill_definition_catalog()
            elif request.method == "get_agent_skill_schema_catalog":
                result = facade.get_agent_skill_schema_catalog()
            elif request.method == "get_skill_route_manifest":
                result = facade.get_skill_route_manifest()
            elif request.method == "get_skill_io_catalog":
                result = facade.get_skill_io_catalog()
            elif request.method == "get_gateway_session_capabilities":
                result = facade.get_gateway_session_capabilities()
            elif request.method == "get_gateway_events":
                result = facade.get_gateway_events(request.payload or {})
            elif request.method == "get_observation_snapshot":
                result = facade.get_observation_snapshot()
            elif request.method == "submit_skill_request":
                result = facade.submit_skill_request(request.payload or {})
            elif request.method == "get_own_operation":
                result = facade.get_own_operation(request.payload or {})
            elif request.method == "cancel_own_operation":
                result = facade.cancel_own_operation(request.payload or {})
            elif request.method == "request_hold_own_domain":
                result = facade.request_hold_own_domain(request.payload or {})
            elif request.method == "begin_artifact_upload":
                result = facade.begin_artifact_upload(request.payload or {})
            elif request.method == "write_artifact_chunk":
                result = facade.write_artifact_chunk(request.payload or {})
            elif request.method == "commit_artifact_upload":
                result = facade.commit_artifact_upload(request.payload or {})
            elif request.method == "begin_artifact_download":
                result = facade.begin_artifact_download(request.payload or {})
            elif request.method == "read_artifact_chunk":
                result = facade.read_artifact_chunk(request.payload or {})
            elif request.method == "acknowledge_artifact_download":
                result = facade.acknowledge_artifact_download(request.payload or {})
            elif request.method == "acknowledge_artifact_materialization":
                result = facade.acknowledge_artifact_materialization(
                    request.payload or {}
                )
            else:  # pragma: no cover - closed by _validate_request.
                raise AgentGatewayUnixProtocolError(
                    "method_not_allowed",
                    "request method is not in the fixed Gateway allowlist",
                    request_id=request.request_id,
                )
            return _success_response(request.request_id, result), False
        except AgentGatewayUnixProtocolError as error:
            return (
                _error_response(
                    request.request_id,
                    code=error.code,
                    message=str(error),
                ),
                True,
            )
        except BaseException as error:  # noqa: BLE001 - project bounded error.
            code, message, terminate = _public_error(error)
            return (
                _error_response(
                    request.request_id,
                    code=code,
                    message=message,
                ),
                terminate,
            )

    def _try_write_response(
        self,
        sock: socket.socket,
        response: Mapping[str, Any],
    ) -> bool:
        try:
            sock.sendall(
                _encode_json_frame(
                    response,
                    max_frame_bytes=self.max_frame_bytes,
                )
            )
            return True
        except (OSError, AgentGatewayUnixProtocolError):
            return False

    def _release_authenticated_connection(
        self,
        *,
        client_id: str,
        peer: AgentGatewayUnixPeerCredentials,
        facade: BoundAgentGatewayWireAdapter,
        sock: socket.socket,
        reason: str,
    ) -> None:
        with self._connection_lock:
            if self._active_connections.get(client_id) is sock:
                self._active_connections.pop(client_id, None)
        outcome_accepted: bool | None = None
        outcome_reason: str | None = None
        error_type: str | None = None
        try:
            outcome = facade.handle_authenticated_disconnect(reason=reason)
            outcome_accepted = getattr(outcome, "accepted", None)
            outcome_reason = getattr(outcome, "reason_code", None)
        except BaseException as error:  # noqa: BLE001 - record; never claim HOLD.
            error_type = type(error).__name__
        event = AgentGatewayUnixDisconnectEvent(
            client_id=client_id,
            peer=peer,
            reason=reason,
            outcome_accepted=outcome_accepted,
            outcome_reason_code=outcome_reason,
            error_type=error_type,
        )
        with self._connection_lock:
            self._disconnect_events.append(event)

    def _unlink_owned_endpoint(self) -> None:
        if not self._bound_here or self._bound_endpoint_identity is None:
            return
        try:
            endpoint = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISSOCK(endpoint.st_mode)
            and (endpoint.st_dev, endpoint.st_ino) == self._bound_endpoint_identity
        ):
            os.unlink(self.socket_path)

    def server_close(self) -> None:
        with self._close_lock:
            if self._transport_closed:
                return
            self._transport_closed = True
            try:
                self._gateway_close_outcome = self._close_gateway()
            except BaseException as error:  # noqa: BLE001 - cleanup still runs.
                self._gateway_close_error = error
            with self._connection_lock:
                active_sockets = tuple(self._active_connections.values())
            for active_socket in active_sockets:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                super().server_close()
            finally:
                self._unlink_owned_endpoint()


__all__ = [
    "AGENT_GATEWAY_UNIX_JSON_METHODS_V2",
    "AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2",
    "DEFAULT_AGENT_GATEWAY_UNIX_MAX_FRAME_BYTES",
    "DEFAULT_AGENT_GATEWAY_UNIX_PEER_IDLE_TIMEOUT_SEC",
    "MAX_AGENT_GATEWAY_UNIX_SOCKET_PATH_BYTES",
    "AgentGatewayUnixDisconnectEvent",
    "AgentGatewayUnixJsonServer",
    "AgentGatewayUnixPeerClientMap",
    "AgentGatewayUnixPeerCredentials",
    "AgentGatewayUnixProtocolError",
]

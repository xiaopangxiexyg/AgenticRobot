#!/usr/bin/env python3
"""Generic HTTPS/WebSocket development host for Quest Browser WebXR apps.

This file is extracted from the transport and certificate portions of
``quest_pose/server.py``. Optional capabilities live under ``features/`` and
are loaded from config; the core contains no robot, Redis, or control logic.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import secrets
import signal
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, TypeVar
from urllib.parse import quote, urlsplit
from uuid import uuid4

import yaml

try:  # Support both ``python server.py`` and package imports.
    from .protocol import PROTOCOL_VERSION, client_auth_result, server_hello
except ImportError:  # pragma: no cover - exercised by the script entrypoint.
    from protocol import PROTOCOL_VERSION, client_auth_result, server_hello


MODULE_ROOT = Path(__file__).resolve().parent
WEB_ROOT = MODULE_ROOT / "web"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_WS_MESSAGE_BYTES = 1_000_000
PAIRING_CODE_DIGITS = 6

RuntimeT = TypeVar("RuntimeT")


@dataclass
class WebSocketClient:
    client_id: int
    address: str
    writer: BinaryIO
    authenticated: bool = False
    client_instance_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def ws_frame(opcode: int, data: bytes = b"") -> bytes:
    """Build an unmasked server-to-client WebSocket frame."""
    first = 0x80 | (opcode & 0x0F)
    length = len(data)
    if length < 126:
        header = struct.pack("!BB", first, length)
    elif length < 65536:
        header = struct.pack("!BBH", first, 126, length)
    else:
        header = struct.pack("!BBQ", first, 127, length)
    return header + data


def ws_text_frame(payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return ws_frame(0x1, data)


def same_origin_allowed(
    origin: str | None,
    host: str | None,
    *,
    expected_scheme: str | None = None,
) -> bool:
    """Accept browser WebSockets only from the HTTP origin that served the app."""
    if not origin or not host:
        return False
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    allowed_schemes = {expected_scheme} if expected_scheme else {"http", "https"}
    return (
        parsed.scheme in allowed_schemes
        and parsed.netloc.casefold() == str(host).casefold()
    )


def validate_pairing_token(value: Any, name: str = "pairing_token") -> str:
    if (
        not isinstance(value, str)
        or len(value) != PAIRING_CODE_DIGITS
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise ValueError(f"{name} must contain exactly 6 ASCII digits")
    return value


def pairing_token_matches(candidate: Any, expected_token: str) -> bool:
    try:
        validated = validate_pairing_token(candidate)
    except ValueError:
        return False
    return secrets.compare_digest(validated, expected_token)


def bearer_token_matches(header: str | None, expected_token: str) -> bool:
    """Validate an HTTP Authorization bearer token without logging it."""
    if not isinstance(header, str):
        return False
    scheme, separator, candidate = header.partition(" ")
    return bool(
        separator
        and scheme.casefold() == "bearer"
        and pairing_token_matches(candidate, expected_token)
    )


def _bounded_text(value: Any, *, limit: int = 160) -> str | None:
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    return value


def summarize_message(payload: Any) -> dict[str, Any]:
    """Return status-safe metadata without credentials or high-rate pose bodies."""
    if not isinstance(payload, dict):
        return {"type": "invalid"}
    summary: dict[str, Any] = {}
    for key in (
        "type",
        "feature",
        "protocol_version",
        "feature_protocol_version",
        "client_instance_id",
        "xr_session_id",
        "state",
        "reference_space_type",
        "sample_sequence",
    ):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) and key != "pairing_token":
            summary[key] = value
    reference_space = payload.get("reference_space")
    if isinstance(reference_space, dict):
        summary["reference_space"] = {
            key: value
            for key in ("type", "generation")
            if isinstance((value := reference_space.get(key)), (str, int))
        }
    controllers = payload.get("controllers")
    if isinstance(controllers, list):
        summary["controller_count"] = len(controllers)
    robots = payload.get("robots")
    if isinstance(robots, dict):
        summary["robot_count"] = len(robots)
    return summary


class QuestDevHandler(SimpleHTTPRequestHandler):
    server: QuestDevHTTPServer

    def do_GET(self) -> None:
        if self.path == "/api/status":
            if not self.server.is_status_request_authorized(
                self.headers.get("Authorization")
            ):
                self.send_error(401, "Pairing token required")
                return
            self._send_json(self.server.snapshot())
            return
        if self.path == "/api/config":
            self._send_json(self.server.client_config)
            return
        if (
            self.path == "/ws"
            and self.headers.get("Upgrade", "").lower() == "websocket"
        ):
            self._handle_websocket()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_websocket(self) -> None:
        if self.server.require_same_origin and not same_origin_allowed(
            self.headers.get("Origin"),
            self.headers.get("Host"),
            expected_scheme="https" if self.server.use_https else "http",
        ):
            self.send_error(403, "WebSocket Origin must match Host")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return
        try:
            client_id = self.server.add_ws_client(self.client_address[0], self.wfile)
        except ConnectionError:
            self.send_error(503, "Quest Dev client limit reached")
            return
        try:
            accept = base64.b64encode(
                hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
            ).decode("ascii")
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.close_connection = True
            auth_deadline = time.monotonic() + self.server.auth_timeout_s
            self.server.send_to_client(
                client_id,
                server_hello(server_instance_id=self.server.server_instance_id),
            )
            while True:
                authenticated = self.server.is_ws_client_authenticated(client_id)
                if authenticated:
                    self.connection.settimeout(None)
                else:
                    remaining_auth_s = auth_deadline - time.monotonic()
                    if remaining_auth_s <= 0:
                        break
                    self.connection.settimeout(remaining_auth_s)
                try:
                    message = self._read_ws_message(
                        auth_deadline=None if authenticated else auth_deadline
                    )
                except (OSError, TimeoutError, ValueError):
                    break
                if message is None:
                    break
                opcode, data = message
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    self.server.send_raw_to_client(client_id, ws_frame(0xA, data))
                    continue
                if opcode == 0xA:
                    continue
                if opcode != 0x1:
                    break
                try:
                    payload = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if authenticated:
                        continue
                    break
                if not self.server.process_client_message(client_id, payload):
                    break
        finally:
            self.server.remove_ws_client(client_id)

    def _read_exact(self, length: int, *, deadline: float | None) -> bytes:
        chunks: list[bytes] = []
        remaining_bytes = length
        while remaining_bytes:
            if deadline is not None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError("WebSocket authentication deadline expired")
                self.connection.settimeout(remaining_s)
            read_once = getattr(self.rfile, "read1", None)
            if read_once is None:
                read_once = self.rfile.read
            chunk = read_once(remaining_bytes)
            if not chunk:
                break
            chunks.append(chunk)
            remaining_bytes -= len(chunk)
        return b"".join(chunks)

    def _read_ws_message(
        self, *, auth_deadline: float | None = None
    ) -> tuple[int, bytes] | None:
        header = self._read_exact(2, deadline=auth_deadline)
        if len(header) < 2:
            return None
        first, second = header
        if not first & 0x80 or first & 0x70:
            raise ValueError("fragmented or RSV WebSocket frames are unsupported")
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        if not masked:
            raise ValueError("client WebSocket frames must be masked")
        length = second & 0x7F
        if length == 126:
            raw = self._read_exact(2, deadline=auth_deadline)
            if len(raw) != 2:
                return None
            length = struct.unpack("!H", raw)[0]
        elif length == 127:
            raw = self._read_exact(8, deadline=auth_deadline)
            if len(raw) != 8:
                return None
            length = struct.unpack("!Q", raw)[0]
        if opcode not in {0x1, 0x8, 0x9, 0xA}:
            raise ValueError("unsupported WebSocket opcode")
        if opcode >= 0x8 and length > 125:
            raise ValueError("WebSocket control frames must be at most 125 bytes")
        if length > MAX_WS_MESSAGE_BYTES:
            raise ValueError("WebSocket message exceeds size limit")
        mask = self._read_exact(4, deadline=auth_deadline)
        if len(mask) != 4:
            return None
        data = self._read_exact(length, deadline=auth_deadline)
        if len(data) != length:
            return None
        data = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        return opcode, data


class QuestDevHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_cls,
        *,
        client_config: dict[str, Any] | None = None,
        pairing_token: str,
        require_same_origin: bool = True,
        server_instance_id: str | None = None,
        max_ws_clients: int = 4,
        auth_timeout_s: float = 5.0,
        use_https: bool = True,
        monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not callable(monotonic_ns_clock):
            raise TypeError("monotonic_ns_clock must be callable")
        super().__init__(server_address, handler_cls)
        self.client_config = dict(client_config or {})
        self._pairing_token = validate_pairing_token(pairing_token)
        self.require_same_origin = bool(require_same_origin)
        self.max_ws_clients = int(max_ws_clients)
        self.auth_timeout_s = float(auth_timeout_s)
        if not 1 <= self.max_ws_clients <= 32:
            raise ValueError("max_ws_clients must be between 1 and 32")
        if not 0.5 <= self.auth_timeout_s <= 60.0:
            raise ValueError("auth_timeout_s must be between 0.5 and 60 seconds")
        self.use_https = bool(use_https)
        self.monotonic_ns_clock = monotonic_ns_clock
        self.server_instance_id = str(server_instance_id or uuid4().hex)
        self.lock = threading.RLock()
        self.ws_connections: dict[int, WebSocketClient] = {}
        self.next_ws_client_id = 1
        self.last_client_message: dict[str, Any] | None = None
        self.last_feature_messages: dict[str, dict[str, Any]] = {}
        self.client_message_handlers: dict[str, list[Any]] = {}
        self.client_disconnect_handlers: list[Any] = []
        self.feature_status_providers: dict[str, Any] = {}
        self.started_at = time.time()

    def add_ws_client(self, address: str, writer: BinaryIO) -> int:
        with self.lock:
            if len(self.ws_connections) >= self.max_ws_clients:
                raise ConnectionError("Quest Dev WebSocket client limit reached")
            client_id = self.next_ws_client_id
            self.next_ws_client_id += 1
            self.ws_connections[client_id] = WebSocketClient(client_id, address, writer)
            return client_id

    def remove_ws_client(self, client_id: int) -> None:
        with self.lock:
            removed = self.ws_connections.pop(client_id, None)
            handlers = list(self.client_disconnect_handlers)
        if removed is not None:
            for handler in handlers:
                handler(client_id)

    def is_ws_client_authenticated(self, client_id: int) -> bool:
        with self.lock:
            client = self.ws_connections.get(client_id)
            return bool(client and client.authenticated)

    def is_status_request_authorized(self, authorization: str | None) -> bool:
        return bearer_token_matches(authorization, self._pairing_token)

    def _authenticate_ws_client(self, client_id: int, payload: Any) -> bool:
        if not isinstance(payload, dict) or payload.get("type") != "client_auth":
            authenticated = False
            error = "authentication_required"
        elif payload.get("protocol_version") != PROTOCOL_VERSION:
            authenticated = False
            error = "protocol_version_mismatch"
        else:
            token = payload.get("pairing_token")
            client_instance_id = _bounded_text(payload.get("client_instance_id"))
            authenticated = bool(
                client_instance_id and pairing_token_matches(token, self._pairing_token)
            )
            error = None if authenticated else "authentication_failed"
        if authenticated:
            with self.lock:
                client = self.ws_connections.get(client_id)
                if client is None:
                    return False
                client.authenticated = True
                client.client_instance_id = str(payload["client_instance_id"])
        self.send_to_client(
            client_id,
            client_auth_result(
                authenticated=authenticated,
                server_instance_id=self.server_instance_id,
                client_id=client_id if authenticated else None,
                error=error,
            ),
        )
        return authenticated

    def send_raw_to_client(self, client_id: int, frame: bytes) -> bool:
        with self.lock:
            client = self.ws_connections.get(client_id)
        if client is None:
            return False
        try:
            with client.lock:
                client.writer.write(frame)
                client.writer.flush()
            return True
        except OSError:
            self.remove_ws_client(client_id)
            return False

    def send_to_client(self, client_id: int, payload: dict[str, Any]) -> bool:
        return self.send_raw_to_client(client_id, ws_text_frame(payload))

    def broadcast(self, payload: dict[str, Any]) -> int:
        with self.lock:
            client_ids = [
                client_id
                for client_id, client in self.ws_connections.items()
                if client.authenticated
            ]
        return sum(self.send_to_client(client_id, payload) for client_id in client_ids)

    def publish_feature_message(self, feature: str, payload: dict[str, Any]) -> int:
        with self.lock:
            self.last_feature_messages[str(feature)] = summarize_message(payload)
        return self.broadcast(payload)

    def register_client_message_handler(self, message_type: str, handler: Any) -> None:
        with self.lock:
            self.client_message_handlers.setdefault(str(message_type), []).append(
                handler
            )

    def register_client_disconnect_handler(self, handler: Any) -> None:
        with self.lock:
            self.client_disconnect_handlers.append(handler)

    def register_feature_status_provider(self, feature: str, provider: Any) -> None:
        with self.lock:
            self.feature_status_providers[str(feature)] = provider

    def process_client_message(self, client_id: int, payload: Any) -> bool:
        if not self.is_ws_client_authenticated(client_id):
            return self._authenticate_ws_client(client_id, payload)
        if not isinstance(payload, dict):
            return True
        with self.lock:
            client = self.ws_connections.get(client_id)
            authenticated_instance_id = (
                None if client is None else client.client_instance_id
            )
        claimed_instance_id = payload.get("client_instance_id")
        if (
            claimed_instance_id is not None
            and claimed_instance_id != authenticated_instance_id
        ):
            return False
        message_type = payload.get("type")
        record = dict(payload)
        record["client_id"] = client_id
        record["server_timestamp"] = time.time()
        record["server_monotonic_ns"] = int(self.monotonic_ns_clock())
        with self.lock:
            self.last_client_message = {
                **summarize_message(record),
                "client_id": client_id,
                "server_timestamp": record["server_timestamp"],
            }
            handlers = list(self.client_message_handlers.get(str(message_type), []))
        for handler in handlers:
            handler(client_id, record)
        return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            clients = [
                {
                    "id": client.client_id,
                    "address": client.address,
                    "authenticated": client.authenticated,
                }
                for client in self.ws_connections.values()
            ]
            last_message = self.last_client_message
            feature_messages = dict(self.last_feature_messages)
            status_providers = dict(self.feature_status_providers)
        feature_status: dict[str, Any] = {}
        for name, provider in status_providers.items():
            try:
                feature_status[name] = provider()
            except Exception as exc:  # noqa: BLE001 - status must stay diagnostic.
                feature_status[name] = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "type": "server_status",
            "protocol_version": PROTOCOL_VERSION,
            "mode": "quest_dev",
            "uptime_s": time.time() - self.started_at,
            "clients": clients,
            "last_client_message": last_message,
            "last_feature_messages": feature_messages,
            "feature_status": feature_status,
        }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    return config


def resolve_module_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else MODULE_ROOT / path


def detect_lan_ip() -> str:
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        if sock is not None:
            sock.close()


def cert_has_san(cert: Path, lan_ip: str) -> bool:
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", str(cert), "-noout", "-ext", "subjectAltName"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = result.stdout
    return (
        "DNS:localhost" in output
        and "IP Address:127.0.0.1" in output
        and f"IP Address:{lan_ip}" in output
    )


def ensure_certificate(cert: Path, key: Path, lan_ip: str) -> None:
    if cert.exists() and key.exists() and cert_has_san(cert, lan_ip):
        return
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    openssl_config = f"""
[req]
distinguished_name=req_distinguished_name
x509_extensions=v3_req
prompt=no
[req_distinguished_name]
CN={lan_ip}
[v3_req]
subjectAltName=@alt_names
[alt_names]
DNS.1=localhost
IP.1=127.0.0.1
IP.2={lan_ip}
""".strip()
    config_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as handle:
            handle.write(openssl_config)
            config_path = handle.name
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-days",
                "3650",
                "-config",
                config_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if config_path:
            Path(config_path).unlink(missing_ok=True)


def feature_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_features = config.get("features", {})
    if not isinstance(raw_features, dict):
        raise TypeError("features config must be a mapping")
    features: dict[str, dict[str, Any]] = {}
    for name, value in raw_features.items():
        if not str(name).isidentifier() or not isinstance(value, dict):
            raise ValueError(f"invalid feature config: {name!r}")
        features[str(name)] = value
    return features


def build_client_config(
    config: dict[str, Any], features: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    xr_cfg = config.get("xr", {})
    if not isinstance(xr_cfg, dict):
        raise TypeError("xr config must be a mapping")
    client_features: dict[str, dict[str, Any]] = {}
    for name, feature_cfg in features.items():
        client_cfg = feature_cfg.get("client", {})
        if not isinstance(client_cfg, dict):
            raise TypeError(f"features.{name}.client must be a mapping")
        if "enabled" in client_cfg:
            raise ValueError(
                f"features.{name}.client.enabled cannot override host feature state"
            )
        client_features[name] = {
            "enabled": bool(feature_cfg.get("enabled", False)),
            **client_cfg,
        }
    server_cfg = config.get("server", {})
    if not isinstance(server_cfg, dict):
        raise TypeError("server config must be a mapping")
    pairing_cfg = server_cfg.get("pairing", {})
    if not isinstance(pairing_cfg, dict):
        raise TypeError("server.pairing must be a mapping")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "xr": {"reference_space": str(xr_cfg.get("reference_space", "local-floor"))},
        "security": {"pairing_required": bool(pairing_cfg.get("enabled", True))},
        "features": client_features,
    }


def resolve_pairing_settings(server_cfg: dict[str, Any]) -> tuple[str, bool]:
    pairing_cfg = server_cfg.get("pairing", {})
    if not isinstance(pairing_cfg, dict):
        raise TypeError("server.pairing must be a mapping")
    if not bool(pairing_cfg.get("enabled", True)):
        raise ValueError("Quest Dev pairing cannot be disabled")
    token_env = str(pairing_cfg.get("token_env", "QUEST_DEV_PAIRING_TOKEN"))
    configured = os.environ.get(token_env)
    token = (
        configured
        if configured
        else f"{secrets.randbelow(10**PAIRING_CODE_DIGITS):0{PAIRING_CODE_DIGITS}d}"
    )
    return validate_pairing_token(token, token_env), bool(
        pairing_cfg.get("require_same_origin", True)
    )


def resolve_bind_host(server_cfg: dict[str, Any], *, use_https: bool) -> str:
    """Never expose plaintext Quest telemetry beyond the local machine."""
    if not use_https:
        return "127.0.0.1"
    return str(server_cfg.get("host", "0.0.0.0"))


def load_feature_runtime(
    name: str,
    server: QuestDevHTTPServer,
    config: dict[str, Any],
    stop_event: threading.Event,
) -> Any:
    module_name = (
        f"{__package__}.features.{name}.runtime"
        if __package__
        else f"features.{name}.runtime"
    )
    module = importlib.import_module(module_name)
    factory = getattr(module, "create_runtime", None)
    if not callable(factory):
        raise TypeError(f"feature {name!r} does not export create_runtime")
    return factory(server, config, stop_event)


@dataclass(frozen=True)
class QuestDevLaunchInfo:
    """Operator-facing connection facts for one host instance.

    Pairing credentials are intentionally excluded from ``repr`` so an
    application can pass this object around without accidentally copying them
    into diagnostic logs.  They are not projected into the HTTP status endpoint.
    """

    base_url: str
    pairing_url: str = field(repr=False)
    pairing_code: str = field(repr=False)
    use_https: bool


class QuestDevHost:
    """Own one Quest HTTP server and each of its enabled feature runtimes.

    Feature runtimes are constructed and started exactly once.  Application
    leaves may borrow a running runtime by explicit name and type, but the host
    remains its sole lifecycle owner.  There is deliberately no module-global
    runtime registry and no fallback that constructs a missing runtime.
    """

    def __init__(
        self,
        server: QuestDevHTTPServer,
        features: dict[str, dict[str, Any]],
        launch_info: QuestDevLaunchInfo,
        *,
        runtime_loader: Callable[
            [str, QuestDevHTTPServer, dict[str, Any], threading.Event], Any
        ] = load_feature_runtime,
    ) -> None:
        if not callable(runtime_loader):
            raise TypeError("runtime_loader must be callable")
        self.server = server
        self.launch_info = launch_info
        self.stop_event = threading.Event()
        self._feature_configs = tuple(
            (str(name), dict(config)) for name, config in features.items()
        )
        self._runtime_loader = runtime_loader
        self._feature_runtimes: dict[str, Any] = {}
        self._owned_runtimes: list[tuple[str, Any]] = []
        self._lock = threading.RLock()
        self._state = "created"
        self._shutdown_requested = False
        self._serve_loop_started = False
        self._server_shutdown_started = False
        self._server_close_attempted = False
        self._cleanup_failure: tuple[str, ...] = ()

    def start(self) -> None:
        """Start each enabled feature once without entering the HTTP loop."""

        with self._lock:
            if self._state not in {"created", "running", "serving"}:
                raise RuntimeError(f"Quest Dev host cannot start from {self._state}")
            if self._shutdown_requested:
                raise RuntimeError("Quest Dev host shutdown was already requested")
            if self._state in {"running", "serving"}:
                return
            self._state = "starting"
            try:
                for name, feature_cfg in self._feature_configs:
                    if not bool(feature_cfg.get("enabled", False)):
                        continue
                    runtime = self._runtime_loader(
                        name,
                        self.server,
                        feature_cfg,
                        self.stop_event,
                    )
                    start_runtime = getattr(runtime, "start", None)
                    stop_runtime = getattr(runtime, "stop", None)
                    if not callable(start_runtime) or not callable(stop_runtime):
                        raise TypeError(
                            f"feature {name!r} runtime must expose start() and stop()"
                        )
                    # Record ownership before start: a partially-started runtime
                    # must still be included in fail-closed cleanup.
                    self._owned_runtimes.append((name, runtime))
                    start_runtime()
                    self._feature_runtimes[name] = runtime
                    print(f"[quest-dev] feature enabled: {name}", flush=True)
                    if self._shutdown_requested:
                        raise RuntimeError(
                            "Quest Dev host shutdown was requested during startup"
                        )
            except BaseException as start_error:
                cleanup_errors, cleanup_base_error = (
                    self._close_owned_resources_locked()
                )
                if cleanup_errors:
                    self._cleanup_failure = tuple(cleanup_errors)
                self._state = "fault"
                if cleanup_base_error is not None:
                    raise cleanup_base_error from start_error
                if cleanup_errors:
                    raise RuntimeError(
                        "Quest Dev host startup cleanup failed: "
                        + "; ".join(cleanup_errors)
                    ) from start_error
                raise
            self._state = "running"

    def require_running_feature_runtime(
        self,
        name: str,
        expected_type: type[RuntimeT],
    ) -> RuntimeT:
        """Borrow the unique running feature runtime, failing closed otherwise."""

        if not isinstance(expected_type, type):
            raise TypeError("expected_type must be a runtime type")
        with self._lock:
            if self._state not in {"running", "serving"}:
                raise RuntimeError("Quest Dev feature runtimes are not running")
            if self._shutdown_requested:
                raise RuntimeError("Quest Dev host shutdown is in progress")
            try:
                runtime = self._feature_runtimes[str(name)]
            except KeyError:
                raise KeyError(f"Quest Dev feature {name!r} is not enabled") from None
            if not isinstance(runtime, expected_type):
                raise TypeError(
                    f"Quest Dev feature {name!r} runtime is "
                    f"{type(runtime).__name__}, expected {expected_type.__name__}"
                )
            return runtime

    def serve_forever(
        self,
        *,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        """Start features, expose the borrowing seam, then serve until stopped."""

        if ready_callback is not None and not callable(ready_callback):
            raise TypeError("ready_callback must be callable or None")
        try:
            # Keep the pre-start check, one-time start, and serving transition
            # under one re-entrant host lock. A concurrent shutdown request can
            # then observe either the pre-start state or ``serving``, never a
            # socket-owning gap outside this method's cleanup boundary.
            with self._lock:
                if self._shutdown_requested:
                    return
                self.start()
                if self._state != "running":
                    raise RuntimeError("Quest Dev host is not ready to serve")
                self._state = "serving"
            if ready_callback is not None:
                ready_callback()
            with self._lock:
                # Reserve entry before checking the flag. A Python signal can
                # re-enter request_shutdown() at any bytecode boundary even
                # while this thread owns the RLock. Once this flag is true,
                # either that request or this handoff starts exactly one
                # shutdown worker, and we still enter BaseServer so the worker
                # cannot wait forever for a loop that never started.
                self._serve_loop_started = True
                start_shutdown = (
                    self._shutdown_requested and not self._server_shutdown_started
                )
                if start_shutdown:
                    self._server_shutdown_started = True
            if start_shutdown:
                self._start_server_shutdown_worker()
            self.server.serve_forever()
        finally:
            with self._lock:
                self._serve_loop_started = False
                if self._state == "serving":
                    self._state = "running"
            self.close()

    def request_shutdown(self) -> bool:
        """Request stop without ever calling ``shutdown`` on the serving thread."""

        with self._lock:
            if self._state in {"closing", "closed", "fault"}:
                return False
            if self._shutdown_requested:
                return False
            self._shutdown_requested = True
            self.stop_event.set()
            start_shutdown = (
                self._state == "serving"
                and self._serve_loop_started
                and not self._server_shutdown_started
            )
            if start_shutdown:
                self._server_shutdown_started = True
        if start_shutdown:
            self._start_server_shutdown_worker()
        return True

    def _start_server_shutdown_worker(self) -> None:
        try:
            threading.Thread(
                target=self.server.shutdown,
                name="quest-dev-shutdown",
                daemon=True,
            ).start()
        except BaseException:
            with self._lock:
                self._server_shutdown_started = False
            raise

    def close(self) -> None:
        """Stop owned resources once and permanently re-report cleanup failure."""

        with self._lock:
            if self._state == "fault":
                if self._cleanup_failure:
                    raise RuntimeError(
                        "Quest Dev host cleanup failed: "
                        + "; ".join(self._cleanup_failure)
                    )
                return
            if self._state in {"closing", "closed"}:
                return
            if self._state == "serving":
                raise RuntimeError(
                    "cannot close Quest Dev host while serving; request shutdown first"
                )
            self._state = "closing"
            errors, cleanup_base_error = self._close_owned_resources_locked()
            if errors:
                self._cleanup_failure = tuple(errors)
            self._state = "closed" if not errors else "fault"
        if cleanup_base_error is not None:
            raise cleanup_base_error
        if errors:
            raise RuntimeError("Quest Dev host cleanup failed: " + "; ".join(errors))

    def _close_owned_resources_locked(
        self,
    ) -> tuple[list[str], BaseException | None]:
        self._shutdown_requested = True
        self.stop_event.set()
        errors: list[str] = []
        first_base_error: BaseException | None = None
        for name, runtime in reversed(self._owned_runtimes):
            try:
                runtime.stop()
            except BaseException as exc:  # noqa: BLE001 - finish reverse cleanup.
                errors.append(f"feature {name}: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception) and first_base_error is None:
                    first_base_error = exc
        self._feature_runtimes.clear()
        if not self._server_close_attempted:
            self._server_close_attempted = True
            try:
                self.server.server_close()
            except BaseException as exc:  # noqa: BLE001 - aggregate cleanup evidence.
                errors.append(f"server: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception) and first_base_error is None:
                    first_base_error = exc
        return errors, first_base_error


def build_quest_dev_host(
    config_path: Path = MODULE_ROOT / "config.yaml",
    *,
    force_http: bool = False,
    monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
    runtime_loader: Callable[
        [str, QuestDevHTTPServer, dict[str, Any], threading.Event], Any
    ] = load_feature_runtime,
) -> QuestDevHost:
    """Bind one Quest server without starting features or its request loop."""

    if not callable(monotonic_ns_clock):
        raise TypeError("monotonic_ns_clock must be callable")
    if not callable(runtime_loader):
        raise TypeError("runtime_loader must be callable")

    config = load_config(Path(config_path).resolve())
    server_cfg = config.get("server", {})
    if not isinstance(server_cfg, dict):
        raise TypeError("server config must be a mapping")
    port = int(server_cfg.get("port", 8444))
    use_https = bool(server_cfg.get("https", True)) and not force_http
    bind_host = resolve_bind_host(server_cfg, use_https=use_https)
    lan_ip = detect_lan_ip() if use_https else "127.0.0.1"
    pairing_code, require_same_origin = resolve_pairing_settings(server_cfg)

    features = feature_configs(config)
    client_config = build_client_config(config, features)
    handler = partial(QuestDevHandler, directory=str(WEB_ROOT))
    server: QuestDevHTTPServer | None = None
    try:
        server = QuestDevHTTPServer(
            (bind_host, port),
            handler,
            client_config=client_config,
            pairing_token=pairing_code,
            require_same_origin=require_same_origin,
            max_ws_clients=int(server_cfg.get("max_ws_clients", 4)),
            auth_timeout_s=float(server_cfg.get("auth_timeout_s", 5.0)),
            use_https=use_https,
            monotonic_ns_clock=monotonic_ns_clock,
        )
        if use_https:
            cert = resolve_module_path(
                str(server_cfg.get("cert", "assets/certs/quest_dev.crt"))
            )
            key = resolve_module_path(
                str(server_cfg.get("key", "assets/certs/quest_dev.key"))
            )
            ensure_certificate(cert, key, lan_ip)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=cert, keyfile=key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
    except BaseException:
        if server is not None:
            server.server_close()
        raise

    assert server is not None
    bound_port = int(server.server_address[1])
    scheme = "https" if use_https else "http"
    display_host = lan_ip if bind_host in {"0.0.0.0", "::"} else bind_host
    base_url = f"{scheme}://{display_host}:{bound_port}/"
    pairing_url = f"{base_url}#pair={quote(pairing_code, safe='')}"
    launch_info = QuestDevLaunchInfo(
        base_url=base_url,
        pairing_url=pairing_url,
        pairing_code=pairing_code,
        use_https=use_https,
    )
    return QuestDevHost(
        server,
        features,
        launch_info,
        runtime_loader=runtime_loader,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config.yaml")
    parser.add_argument(
        "--http", action="store_true", help="disable HTTPS for local desktop tests"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = build_quest_dev_host(args.config, force_http=args.http)
    launch = host.launch_info
    print(f"[quest-dev] open on Quest: {launch.base_url}", flush=True)
    print(f"[quest-dev] 6-digit pairing code: {launch.pairing_code}", flush=True)
    print(f"[quest-dev] one-step pairing URL: {launch.pairing_url}", flush=True)

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        host.request_shutdown()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    host.serve_forever()


if __name__ == "__main__":
    main()

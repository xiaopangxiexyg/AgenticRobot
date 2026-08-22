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
import signal
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO

import yaml

try:  # Support both ``python server.py`` and package imports.
    from .protocol import PROTOCOL_VERSION, server_hello
except ImportError:  # pragma: no cover - exercised by the script entrypoint.
    from protocol import PROTOCOL_VERSION, server_hello


MODULE_ROOT = Path(__file__).resolve().parent
WEB_ROOT = MODULE_ROOT / "web"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_WS_MESSAGE_BYTES = 1_000_000


@dataclass
class WebSocketClient:
    client_id: int
    address: str
    writer: BinaryIO
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


class QuestDevHandler(SimpleHTTPRequestHandler):
    server: QuestDevHTTPServer

    def do_GET(self) -> None:
        if self.path == "/api/status":
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
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return

        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True

        client_id = self.server.add_ws_client(self.client_address[0], self.wfile)
        self.server.send_to_client(client_id, server_hello())
        try:
            while True:
                try:
                    message = self._read_ws_message()
                except ValueError:
                    break
                if message is None:
                    break
                opcode, data = message
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    self.server.send_raw_to_client(client_id, ws_frame(0xA, data))
                    continue
                if opcode != 0x1:
                    continue
                try:
                    payload = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self.server.process_client_message(client_id, payload)
        finally:
            self.server.remove_ws_client(client_id)

    def _read_ws_message(self) -> tuple[int, bytes] | None:
        header = self.rfile.read(2)
        if len(header) < 2:
            return None
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            raw = self.rfile.read(2)
            if len(raw) != 2:
                return None
            length = struct.unpack("!H", raw)[0]
        elif length == 127:
            raw = self.rfile.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack("!Q", raw)[0]
        if length > MAX_WS_MESSAGE_BYTES:
            raise ValueError("WebSocket message exceeds size limit")
        mask = self.rfile.read(4) if masked else b""
        data = self.rfile.read(length)
        if len(data) != length:
            return None
        if masked:
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
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.client_config = dict(client_config or {})
        self.lock = threading.RLock()
        self.ws_connections: dict[int, WebSocketClient] = {}
        self.next_ws_client_id = 1
        self.last_client_message: dict[str, Any] | None = None
        self.last_feature_messages: dict[str, dict[str, Any]] = {}
        self.client_message_handlers: dict[str, list[Any]] = {}
        self.started_at = time.time()

    def add_ws_client(self, address: str, writer: BinaryIO) -> int:
        with self.lock:
            client_id = self.next_ws_client_id
            self.next_ws_client_id += 1
            self.ws_connections[client_id] = WebSocketClient(client_id, address, writer)
            return client_id

    def remove_ws_client(self, client_id: int) -> None:
        with self.lock:
            self.ws_connections.pop(client_id, None)

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
            client_ids = list(self.ws_connections)
        return sum(self.send_to_client(client_id, payload) for client_id in client_ids)

    def publish_feature_message(self, feature: str, payload: dict[str, Any]) -> int:
        with self.lock:
            self.last_feature_messages[str(feature)] = payload
        return self.broadcast(payload)

    def register_client_message_handler(self, message_type: str, handler: Any) -> None:
        with self.lock:
            self.client_message_handlers.setdefault(str(message_type), []).append(
                handler
            )

    def process_client_message(self, client_id: int, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        message_type = payload.get("type")
        record = dict(payload)
        record["client_id"] = client_id
        record["server_timestamp"] = time.time()
        with self.lock:
            self.last_client_message = record
            handlers = list(self.client_message_handlers.get(str(message_type), []))
        for handler in handlers:
            handler(client_id, record)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            clients = [
                {"id": client.client_id, "address": client.address}
                for client in self.ws_connections.values()
            ]
            last_message = self.last_client_message
            feature_messages = dict(self.last_feature_messages)
        return {
            "type": "server_status",
            "protocol_version": PROTOCOL_VERSION,
            "mode": "quest_dev",
            "uptime_s": time.time() - self.started_at,
            "clients": clients,
            "last_client_message": last_message,
            "last_feature_messages": feature_messages,
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
        client_features[name] = {
            "enabled": bool(feature_cfg.get("enabled", False)),
            **client_cfg,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "xr": {"reference_space": str(xr_cfg.get("reference_space", "local-floor"))},
        "features": client_features,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODULE_ROOT / "config.yaml")
    parser.add_argument(
        "--http", action="store_true", help="disable HTTPS for local desktop tests"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    server_cfg = config.get("server", {})
    host = str(server_cfg.get("host", "0.0.0.0"))
    port = int(server_cfg.get("port", 8444))
    use_https = bool(server_cfg.get("https", True)) and not args.http
    lan_ip = detect_lan_ip()

    features = feature_configs(config)
    client_config = build_client_config(config, features)
    handler = partial(QuestDevHandler, directory=str(WEB_ROOT))
    server = QuestDevHTTPServer((host, port), handler, client_config=client_config)
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

    scheme = "https" if use_https else "http"
    display_host = lan_ip if host in {"0.0.0.0", "::"} else host
    print(f"[quest-dev] {scheme}://{display_host}:{port}", flush=True)

    stop_once = threading.Event()
    runtimes: list[Any] = []

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for name, feature_cfg in features.items():
            if not bool(feature_cfg.get("enabled", False)):
                continue
            runtime = load_feature_runtime(name, server, feature_cfg, stop_once)
            runtimes.append(runtime)
            runtime.start()
            print(f"[quest-dev] feature enabled: {name}", flush=True)
        server.serve_forever()
    finally:
        stop_once.set()
        for runtime in reversed(runtimes):
            runtime.stop()
        server.server_close()


if __name__ == "__main__":
    main()

"""Local operator Web binding for the unique Quest calibration application.

The coordinator owns only its HTTP server, attachment thread, and rebuildable
presentation state.  It waits on the existing factory's public application
port and never starts, closes, or otherwise owns that factory or application.
"""

from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .application import QuestCalibrationOperatorController

QUEST_CALIBRATION_WEB_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.quest_calibration_web_status.v1"
)
_STATIC_DIR = Path(__file__).with_name("static")
_INDEX_HTML = (_STATIC_DIR / "index.html").read_bytes()
_APP_JS = (_STATIC_DIR / "app.js").read_bytes()
_MAX_INTENT_BYTES = 64 * 1024


class QuestCalibrationApplicationFactoryPort(Protocol):
    """Read-only readiness seam; lifecycle ownership remains with deploy."""

    def wait_for_application(self, *, timeout_s: float) -> object: ...


def _positive_float(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


class QuestCalibrationWebCoordinator:
    """Serve a local UI and attach it to one already-owned capture application."""

    def __init__(
        self,
        factory: QuestCalibrationApplicationFactoryPort,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        application_timeout_s: float = 30.0,
        attach_poll_s: float = 0.2,
    ) -> None:
        wait_for_application = getattr(factory, "wait_for_application", None)
        if not callable(wait_for_application):
            raise TypeError("factory must provide wait_for_application")
        normalized_host = str(host).strip()
        if not normalized_host:
            raise ValueError("host must be non-empty")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError("port must be an integer in [0, 65535]")
        self._factory = factory
        self._host = normalized_host
        self._port = port
        self._open_browser = bool(open_browser)
        self._application_timeout_s = _positive_float(
            application_timeout_s,
            name="application_timeout_s",
        )
        self._attach_poll_s = min(
            _positive_float(attach_poll_s, name="attach_poll_s"),
            self._application_timeout_s,
        )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._binding_state = "not_started"
        self._attach_error_code: str | None = None
        self._controller: QuestCalibrationOperatorController | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._attach_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        with self._lock:
            server = self._server
            host = self._host
            port = self._port
        if server is not None:
            host, port = server.server_address[:2]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    @property
    def binding_state(self) -> str:
        with self._lock:
            return self._binding_state

    def _attach(self) -> None:
        deadline = time.monotonic() + self._application_timeout_s
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                with self._lock:
                    if self._binding_state == "waiting_for_application":
                        self._binding_state = "attach_failed"
                        self._attach_error_code = "application_timeout"
                return
            try:
                application = self._factory.wait_for_application(
                    timeout_s=min(self._attach_poll_s, remaining),
                )
            except TimeoutError:
                continue
            except Exception as error:  # noqa: BLE001 - project type only.
                with self._lock:
                    if not self._stop_event.is_set():
                        self._binding_state = "attach_failed"
                        self._attach_error_code = type(error).__name__
                return
            try:
                controller = QuestCalibrationOperatorController(application)
            except Exception as error:  # noqa: BLE001 - project type only.
                with self._lock:
                    if not self._stop_event.is_set():
                        self._binding_state = "attach_failed"
                        self._attach_error_code = type(error).__name__
                return
            with self._lock:
                if self._stop_event.is_set():
                    return
                self._controller = controller
                self._binding_state = "ready"
                self._attach_error_code = None
            return

    def status_payload(self) -> dict[str, object]:
        with self._lock:
            binding_state = self._binding_state
            error_code = self._attach_error_code
            controller = self._controller
        operator: Mapping[str, object] | None = None
        if controller is not None:
            try:
                operator = controller.status()
            except Exception as error:  # noqa: BLE001 - never expose lower detail.
                binding_state = "operator_status_failed"
                error_code = type(error).__name__
        return {
            "schema": QUEST_CALIBRATION_WEB_STATUS_SCHEMA_V1,
            "binding_state": binding_state,
            "attach_error_code": error_code,
            "operator": None if operator is None else dict(operator),
        }

    def submit_intent(
        self, payload: Mapping[str, Any]
    ) -> tuple[int, dict[str, object]]:
        with self._lock:
            controller = self._controller
            binding_state = self._binding_state
        if controller is None:
            return 409, {
                "accepted": False,
                "rejection_code": "application_not_ready",
                "binding_state": binding_state,
            }
        try:
            receipt = controller.submit_json(payload)
        except (TypeError, ValueError) as error:
            return 400, {
                "accepted": False,
                "rejection_code": "invalid_intent",
                "message": type(error).__name__,
            }
        return 200, receipt

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        coordinator = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, status: int, payload: Mapping[str, object]) -> None:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send(status, "application/json; charset=utf-8", encoded)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path in {"/", "/index.html"}:
                    self._send(200, "text/html; charset=utf-8", _INDEX_HTML)
                    return
                if path == "/app.js":
                    self._send(200, "application/javascript; charset=utf-8", _APP_JS)
                    return
                if path == "/status.json":
                    self._send_json(200, coordinator.status_payload())
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:
                if urlsplit(self.path).path != "/intent":
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 < length <= _MAX_INTENT_BYTES:
                    self._send_json(
                        400,
                        {"accepted": False, "rejection_code": "invalid_body_size"},
                    )
                    return
                try:
                    decoded = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(
                        400,
                        {"accepted": False, "rejection_code": "invalid_json"},
                    )
                    return
                if not isinstance(decoded, Mapping):
                    self._send_json(
                        400,
                        {"accepted": False, "rejection_code": "invalid_intent"},
                    )
                    return
                status, response = coordinator.submit_intent(decoded)
                self._send_json(status, response)

            def log_message(self, fmt: str, *args: Any) -> None:
                del fmt, args

        return Handler

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            if self._binding_state == "closed":
                raise RuntimeError("Quest calibration Web coordinator is closed")
            server = ThreadingHTTPServer(
                (self._host, self._port),
                self._build_handler(),
            )
            server.daemon_threads = True
            server_thread = threading.Thread(
                target=server.serve_forever,
                name="quest-calibration-web-server",
                daemon=True,
            )
            attach_thread = threading.Thread(
                target=self._attach,
                name="quest-calibration-web-attach",
                daemon=True,
            )
            self._server = server
            self._server_thread = server_thread
            self._attach_thread = attach_thread
            self._binding_state = "waiting_for_application"
            server_thread.start()
            attach_thread.start()
        if self._open_browser:
            try:
                webbrowser.open(self.url, new=1, autoraise=True)
            except Exception:  # noqa: BLE001 - presentation convenience only.
                return

    def close(self) -> None:
        """Close only Station-owned Web resources; never close lower capture."""

        self._stop_event.set()
        with self._lock:
            server = self._server
            server_thread = self._server_thread
            attach_thread = self._attach_thread
            self._server = None
            self._server_thread = None
            self._attach_thread = None
            self._controller = None
            self._binding_state = "closed"
        if server is not None:
            server.shutdown()
            server.server_close()
        if (
            server_thread is not None
            and server_thread is not threading.current_thread()
        ):
            server_thread.join(timeout=2.0)
        if (
            attach_thread is not None
            and attach_thread is not threading.current_thread()
        ):
            attach_thread.join(timeout=max(1.0, 2.0 * self._attach_poll_s))


__all__ = [
    "QUEST_CALIBRATION_WEB_STATUS_SCHEMA_V1",
    "QuestCalibrationApplicationFactoryPort",
    "QuestCalibrationWebCoordinator",
]

"""Dedicated local web transport for rule-based writing collection."""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from .application import RuleBasedWritingApplication, RuleBasedWritingIntent


_ASSET_DIR = Path(__file__).with_name("static")


class RuleBasedWritingWebUi:
    """Minimal rule-only UI satisfying the lower OperatorUiPort structurally."""

    # The shared real-control logger uses this passive capability marker to
    # add camera/latency snapshots to its existing asynchronous IK frames.
    diagnostic_mode_profile = "rule-writing"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        resizable_sidebars: bool = True,
        input_activity_indicators: bool = True,
    ) -> None:
        del resizable_sidebars, input_activity_indicators
        self._host = str(host)
        self._port = int(port)
        self._open_browser = bool(open_browser)
        self._application = RuleBasedWritingApplication()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}/"
        host, port = self._server.server_address[:2]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                route = self.path.split("?", 1)[0]
                if route in {"/", "/index.html"}:
                    self._send(200, "text/html; charset=utf-8", (_ASSET_DIR / "index.html").read_bytes())
                    return
                if route == "/app.js":
                    self._send(200, "text/javascript; charset=utf-8", (_ASSET_DIR / "app.js").read_bytes())
                    return
                if route == "/styles.css":
                    self._send(200, "text/css; charset=utf-8", (_ASSET_DIR / "styles.css").read_bytes())
                    return
                if route == "/state.json":
                    payload = json.dumps(
                        ui.snapshot(),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if route == "/options.json":
                    payload = json.dumps(
                        ui.configuration_options(),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {
                    "/event",
                    "/configuration",
                    "/height-calibration",
                }:
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8192:
                    self._send(400, "text/plain; charset=utf-8", b"invalid body")
                    return
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(payload, Mapping):
                        raise ValueError("request body must be an object")
                    if self.path == "/event":
                        intent = RuleBasedWritingIntent(str(payload.get("event", "")))
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                    self._send(400, "text/plain; charset=utf-8", b"invalid event")
                    return
                accepted = (
                    ui._application.submit_intent(intent)
                    if self.path == "/event"
                    else ui._application.submit_configuration(payload)
                    if self.path == "/configuration"
                    else ui._application.submit_height_calibration(payload)
                )
                if not accepted:
                    self._send(409, "text/plain; charset=utf-8", b"inactive event")
                    return
                self._send(204, "text/plain; charset=utf-8", b"")

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._server_thread.start()
        if self._open_browser:
            try:
                webbrowser.open(self.url, new=1, autoraise=True)
            except Exception:
                pass

    def close(self) -> None:
        if self._server is None:
            return
        self._application.mark_shutdown("服务已退出，这个网页已经失效。")
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._server_thread = None

    def consume_events(self, *, max_events: int = 64) -> list[str]:
        return self._application.consume_runtime_events(max_events=max_events)

    def publish(self, **updates: Any) -> None:
        self._application.observe_runtime_publication(updates)

    def append_message(self, tag: str, message: str) -> None:
        self._application.append_message(tag, message)

    def bind_rule_runtime(self) -> None:
        self._application.bind_rule_runtime()

    def consume_rule_intents(self) -> tuple[str, ...]:
        return self._application.consume_rule_intents()

    def configure_rule_options(
        self,
        options: Mapping[str, object],
        selected: Mapping[str, object],
    ) -> None:
        self._application.configure_rule_options(options, selected)

    def consume_rule_configuration_requests(self) -> tuple[dict[str, object], ...]:
        return self._application.consume_configuration_requests()

    def consume_rule_height_calibration_requests(
        self,
    ) -> tuple[dict[str, object], ...]:
        return self._application.consume_height_calibration_requests()

    def publish_rule_height_calibration_result(
        self,
        *,
        error: str | None,
    ) -> None:
        self._application.complete_height_calibration(error=error)

    def publish_rule_configuration_result(
        self,
        *,
        selected: Mapping[str, object] | None,
        error: str | None,
    ) -> None:
        self._application.complete_configuration(selected=selected, error=error)

    def publish_writer_prepare_result(self, *, error: str | None) -> None:
        self._application.publish_writer_prepare_result(error=error)

    def publish_device_retry_result(self, *, error: str | None) -> None:
        self._application.complete_device_retry(error=error)

    def configuration_options(self) -> dict[str, Any]:
        return self._application.configuration_options()

    def publish_rule_status(self, status: Mapping[str, object]) -> None:
        self._application.observe_rule_status(status)

    def request_rule_completion_stop(self) -> None:
        self._application.request_completion_stop()

    def request_rule_failure_stop(self) -> None:
        self._application.request_failure_stop()

    def snapshot(self) -> dict[str, object]:
        return self._application.snapshot()


__all__ = ["RuleBasedWritingWebUi"]

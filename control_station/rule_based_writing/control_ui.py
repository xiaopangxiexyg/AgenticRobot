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
                    "/task-chain",
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
                if self.path == "/event":
                    accepted = ui._application.submit_intent(intent)
                elif self.path == "/configuration":
                    accepted = ui._application.submit_configuration(payload)
                elif self.path == "/height-calibration":
                    accepted = ui._application.submit_height_calibration(payload)
                else:
                    accepted = ui._application.submit_task_chain_advance(payload)
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

    def allows_pre_episode_armed_warmup(self) -> bool:
        """Expose the narrow rule-owned writer-open admission capability."""

        return self._application.allows_pre_episode_armed_warmup()

    def pre_episode_start_admission_reason(self) -> str | None:
        """Expose the rule-owned admission diagnostic without owning the gate."""

        return self._application.pre_episode_start_admission_reason()

    def consume_rule_intents(self) -> tuple[str, ...]:
        return self._application.consume_rule_intents()

    def submit_task_chain_advance_from_foot_pad(self) -> bool:
        """Submit one task-chain advance through the existing app boundary.

        The physical input adapter only calls this capability on the
        dedicated rule UI. It must not synthesize a GUI event or own any task
        state, so the application remains the single task-chain authority.
        """

        return self._application.submit_task_chain_advance(
            {"event": "advance", "source": "foot_pad"}
        )

    def task_chain_foot_pad_action_label(self) -> str:
        """Describe the phase actually entered by the latest pedal advance."""

        state = self._application.snapshot()
        if bool(state.get("task_chain_owner_recovery_required")):
            return (
                "正在恢复右夹爪一次性闭合 owner"
                if bool(state.get("gripper_action_pending"))
                else "等待恢复右夹爪一次性闭合 owner"
            )
        return str(state.get("task_chain_title") or "任务链下一步")

    def submit_emergency_stop_from_foot_pad(self) -> bool:
        """Queue the existing rule stop-and-Hold path for pedal 2."""

        return self._application.submit_intent(RuleBasedWritingIntent.STOP)

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

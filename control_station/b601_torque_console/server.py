"""Loopback-only browser UI for owner-held B601 assist experiments.

HTTP threads submit expiring *intent* only. They never access CAN, motor modes,
gravity calculation, or actuator dispatch. The owner samples either the
direct-torque experiment or staged-assist controller during its already-guarded
control cycle.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from .transparency import TransparencyAssessment

_STATIC = Path(__file__).with_name("static")
_CONTENT = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assessment.js": ("assessment.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}

# Only the chart consumes history rows; current gates/receipts stay in the full snapshot.
_PLOT_HISTORY_FIELDS = (
    "time_sec", "desired_assist_nm", "target_offset_rad", "effective_assist_nm",
    "overlay_after_profile_nm", "displacement_rad",
)


def recent_runtime_notices(path: Path, *, limit: int = 14) -> list[str]:
    """Read a bounded operator-facing tail, never the high-rate dynamics trace."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.startswith(("[B601]", "[DRIVE]"))][-limit:]


class _ConsoleIntentController(Protocol):
    def submit(self, action: str, **kwargs: Any) -> None: ...

    def status(self) -> dict[str, object]: ...


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        controller: _ConsoleIntentController,
        status_provider: Callable[[], dict[str, object]],
    ) -> None:
        self.controller = controller
        self.status_provider = status_provider
        self.assessment: TransparencyAssessment | None = None
        self.human_intervention_handler: Callable[[bool], None] | None = None
        # Order permission commits across asynchronous annotation writes.
        # This lock never covers disk I/O; a newer withdrawal stays immediate.
        self.human_context_lock = threading.Lock()
        self.human_context_sequence = 0
        self.token = secrets.token_urlsafe(32)
        self._stop_lock = threading.Lock()
        self._stop_handler: Callable[[str], None] | None = None
        self._status_timing_lock = threading.Lock()
        self._status_requests = 0
        self._status_completed = 0
        # Bounded diagnostics, independent of controller state and intent leases.
        self._status_timings: deque[dict[str, object]] = deque(maxlen=4096)
        super().__init__(("127.0.0.1", 0), _Handler)

    def begin_status_response(self) -> int:
        with self._status_timing_lock:
            self._status_requests += 1
            return self._status_requests

    def record_status_response(self, record: dict[str, object]) -> None:
        with self._status_timing_lock:
            self._status_completed += 1
            self._status_timings.append(record)

    def status_response_diagnostics(self, *, include_records: bool = False) -> dict[str, object]:
        with self._status_timing_lock:
            return {
                "semantics": (
                    "HTTP observer only; perf_counter wall and handler-thread CPU; "
                    "not control latency or client receipt; response contains preceding completed request"
                ),
                "started_count": self._status_requests,
                "completed_count": self._status_completed,
                "dropped_record_count": self._status_completed - len(self._status_timings),
                "last_completed": deepcopy(self._status_timings[-1]) if self._status_timings else None,
                **({"records": deepcopy(list(self._status_timings))} if include_records else {}),
            }

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def set_stop_handler(self, handler: Callable[[str], None] | None) -> None:
        with self._stop_lock:
            self._stop_handler = handler

    @property
    def stop_controls_available(self) -> bool:
        with self._stop_lock:
            return self._stop_handler is not None

    def request_stop(self, action: str) -> None:
        with self._stop_lock:
            if self._stop_handler is None:
                raise ValueError("owner shutdown protocol is not available")
            self._stop_handler(action)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = "B601TorqueConsole/1"

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _send(self, code: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: HTTPStatus, payload: dict[str, object]) -> None:
        self._send(
            code,
            json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _valid_host(self) -> bool:
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self) -> None:
        if not self._valid_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid host"})
            return
        if self.path in ("/api/status", "/api/status?view=operator"):
            self._status_response(operator_view=self.path.endswith("?view=operator"))
            return
        resource = _CONTENT.get(self.path)
        if resource is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        name, content_type = resource
        body = (_STATIC / name).read_bytes()
        if name == "index.html":
            body = body.replace(b"__CSRF_TOKEN__", self.server.token.encode("ascii"))
        self._send(HTTPStatus.OK, body, content_type)

    def _status_response(self, *, operator_view: bool = False) -> None:
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        sequence = self.server.begin_status_response()
        payload: dict[str, object] = {}
        body: bytes | None = None
        stages: dict[str, dict[str, float]] = {}
        completed = False
        previous_wall, previous_cpu = started_wall, started_cpu

        def mark(name: str) -> None:
            nonlocal previous_wall, previous_cpu
            wall, cpu = time.perf_counter(), time.thread_time()
            stages[name] = {"wall_sec": wall - previous_wall, "thread_cpu_sec": cpu - previous_cpu}
            previous_wall, previous_cpu = wall, cpu

        try:
            payload = self.server.controller.status()
            payload.update(self.server.status_provider())
            payload["stop_controls_available"] = self.server.stop_controls_available
            payload["transparency_assessment"] = (
                None if self.server.assessment is None else self.server.assessment.status()
            )
            if operator_view and "history" in payload:
                # Project detached rows only. The controller's audit history and full endpoint are unchanged.
                payload["history"] = [
                    {key: row[key] for key in _PLOT_HISTORY_FIELDS if key in row}
                    for row in payload["history"]
                ]
            payload["http_status_diagnostics"] = self.server.status_response_diagnostics()
            mark("snapshot")
            body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
            mark("encode")
            self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
            mark("write")
            completed = True
        finally:
            # Failed responses remain failed; telemetry never retries or submits intents.
            ended_wall, ended_cpu = time.perf_counter(), time.thread_time()
            dispatch = payload.get("last_dispatch") if isinstance(payload, dict) else None
            if not isinstance(dispatch, dict):
                dispatch = {}
            self.server.record_status_response(
                {
                    "sequence": sequence,
                    "view": "operator" if operator_view else "full",
                    "started_perf_counter_sec": started_wall,
                    "completed_perf_counter_sec": ended_wall,
                    "wall_sec": ended_wall - started_wall,
                    "thread_cpu_sec": ended_cpu - started_cpu,
                    "response_bytes": 0 if body is None else len(body),
                    "completed": completed,
                    "owner_receipt_sequence": dispatch.get("receipt_sequence"),
                    "owner_feedback_timestamp_sec": dispatch.get("feedback_timestamp_sec"),
                    "stages": stages,
                }
            )

    def do_POST(self) -> None:
        if self.path not in ("/api/intent", "/api/stop", "/api/assessment"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if (
            not self._valid_host()
            or self.headers.get("Origin") != self.server.origin
            or self.headers.get("X-B601-Console-Token") != self.server.token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "unauthorized local intent"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 2048:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or "action" not in payload:
                raise ValueError("invalid request fields")
            action = payload.get("action")
            if not isinstance(action, str):
                raise TypeError("action must be a string")
            if self.path == "/api/assessment":
                if self.server.assessment is None:
                    raise ValueError("transparency recording is not configured")
                # Only the explicit probe app binds control context. A true
                # declaration withdraws before disk I/O; permission to resume
                # requires successful annotation. Neither renews the lease.
                handler = self.server.human_intervention_handler
                context_sequence = None
                if handler is not None:
                    if set(payload) != {"action", "human_intervention"} or action != "context" or not isinstance(payload["human_intervention"], bool):
                        raise ValueError("probe context requires human_intervention boolean")
                    with self.server.human_context_lock:
                        self.server.human_context_sequence += 1
                        context_sequence = self.server.human_context_sequence
                        if payload["human_intervention"]:
                            handler(True)
                self.server.assessment.record(payload, self.server.controller.status())
                if handler is not None and not payload["human_intervention"]:
                    with self.server.human_context_lock:
                        if context_sequence == self.server.human_context_sequence:
                            handler(False)
            elif self.path == "/api/stop":
                if set(payload) != {"action"}:
                    raise ValueError("invalid stop request fields")
                if action not in (
                    "protective_hold", "full_disable", "confirm_cycle_full_disable"
                ):
                    raise ValueError("invalid shutdown action")
                self.server.request_stop(action)
            else:
                action_fields = {
                    "set": {"action", "torque_nm"},
                    "set_target": {"action", "target_offset_rad"},
                    "select_joint": {"action", "joint_name"},
                    "select_mode": {"action", "operation_mode"},
                    "arm": {"action"},
                    "authorize_breakaway": {"action"},
                    "stop_and_hold": {"action"},
                    "heartbeat": {"action"},
                    "visibility": {"action", "visible"},
                    "zero": {"action"},
                    "withdraw": {"action"},
                }
                allowed = action_fields.get(action)
                fields_match = allowed is not None and set(payload) == allowed
                if action == "withdraw" and set(payload) == {"action", "source"}:
                    fields_match = True
                if allowed is None or (not fields_match and not (
                    action == "arm" and set(payload) in (
                        {"action", "operation_mode"}, {"action", "operation_mode", "cycle_angle_deg"},
                        {"action", "operation_mode", "return_and_disable"})
                )):
                    raise ValueError("invalid intent request fields")
                if action == "withdraw":
                    source = payload.get("source", "operator_button")
                    if source not in {"operator_button", "page_hidden", "page_unload"}:
                        raise ValueError("invalid withdrawal source")
                    submit_args = {"now_sec": time.monotonic(), "withdraw_source": source}
                else:
                    submit_args = {"now_sec": time.monotonic()}
                if action == "visibility":
                    if type(payload.get("visible")) is not bool:
                        raise ValueError("visibility requires a boolean visible field")
                    submit_args["visible"] = payload["visible"]
                torque = payload.get("torque_nm")
                if torque is not None and (isinstance(torque, bool) or not isinstance(torque, (int, float))):
                    raise ValueError("torque_nm must be numeric")
                target_offset = payload.get("target_offset_rad")
                if target_offset is not None and (
                    isinstance(target_offset, bool)
                    or not isinstance(target_offset, (int, float))
                ):
                    raise ValueError("target_offset_rad must be numeric")
                joint_name = payload.get("joint_name")
                if joint_name is not None and not isinstance(joint_name, str):
                    raise ValueError("joint_name must be a string")
                if "operation_mode" in payload:
                    mode = payload["operation_mode"]
                    if not isinstance(mode, str) or mode not in {"fixed_position", "supported_cycle", "workspace"}:
                        raise ValueError("invalid operation_mode")
                    submit_args["operation_mode"] = mode
                if "cycle_angle_deg" in payload:
                    submit_args["cycle_angle_deg"] = payload["cycle_angle_deg"]
                if torque is not None:
                    submit_args["torque_nm"] = torque
                if target_offset is not None:
                    submit_args["target_offset_rad"] = target_offset
                if joint_name is not None:
                    submit_args["joint_name"] = joint_name
                if "return_and_disable" in payload:
                    if type(payload["return_and_disable"]) is not bool:
                        raise ValueError("return_and_disable must be boolean")
                    submit_args["return_and_disable"] = payload["return_and_disable"]
                self.server.controller.submit(
                    action,
                    **submit_args,
                )
        except (TypeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except OSError as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, {"status": "accepted"})


class B601TorqueConsoleServer:
    """One loopback view bound to the same process as the hardware owner."""

    def __init__(
        self,
        controller: _ConsoleIntentController,
        *,
        status_provider: Callable[[], dict[str, object]] | None = None,
        assessment_path: Path | None = None,
        automatic_recording: bool = False,
        human_intervention_handler: Callable[[bool], None] | None = None,
        recording_failure_handler: Callable[[str], None] | None = None,
    ) -> None:
        self._server = _Server(controller, status_provider or dict)
        if (human_intervention_handler is not None or recording_failure_handler is not None) and (not automatic_recording or assessment_path is None):
            self._server.server_close()
            raise ValueError("control context requires automatic recording")
        self._server.human_intervention_handler = human_intervention_handler
        self._recording_failure_handler = recording_failure_handler
        self._thread: threading.Thread | None = None
        self._observation_stop = threading.Event()
        self._observation_thread: threading.Thread | None = None
        if assessment_path is not None:
            try:
                self._server.assessment = TransparencyAssessment(assessment_path, automatic=automatic_recording)
            except OSError:
                self._server.server_close()
                raise

    def assessment_status(self) -> dict[str, object] | None:
        journal = self._server.assessment
        return None if journal is None else journal.status()

    def status_response_diagnostics(self) -> dict[str, object]:
        return self._server.status_response_diagnostics(include_records=True)

    @property
    def url(self) -> str:
        return self._server.origin

    def set_stop_handler(self, handler: Callable[[str], None] | None) -> None:
        self._server.set_stop_handler(handler)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("torque console already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="b601-torque-console-http",
            daemon=True,
        )
        self._thread.start()
        if self._server.assessment is not None and self._server.assessment.automatic:
            def observe() -> None:
                while not self._observation_stop.wait(0.05):
                    try:
                        self._server.assessment.observe(self._server.controller.status())
                    except (OSError, TypeError, ValueError) as exc:
                        if self._recording_failure_handler is not None:
                            self._recording_failure_handler(str(exc))
                        self._server.assessment.observation_failed(exc)
                        return

            self._observation_thread = threading.Thread(target=observe, name="b601-diagnostic-observer", daemon=True)
            self._observation_thread.start()

    def close(self) -> None:
        self.set_stop_handler(None)
        self._observation_stop.set()
        if self._observation_thread is not None:
            self._observation_thread.join(timeout=2.0)
        try:
            if self._thread is not None:
                self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        finally:
            if self._server.assessment is not None:
                self._server.assessment.close()

"""Dependency-free local Control Station web builders for CS-1a/1b/1c."""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from ..events import BoundedStationEventLog
from ..operation_coordinator import (
    NoIoStationOperationCoordinator,
    SideEffectStationOperationCoordinator,
    StationOperationCoordinatorUnavailable,
)
from ..runtime import ControlStationRuntime
from ..status import build_control_station_status_payload
from .api import (
    LocalIntentAccessPolicy,
    ScopedSimAuthorityAccessPolicy,
    station_intent_draft_from_json,
)


_STATIC_ROOT = Path(__file__).with_name("static")
_STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}
_LOCAL_BIND_HOSTS = frozenset({"127.0.0.1", "localhost"})


class _ControlStationHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        runtime,
        access_policy: LocalIntentAccessPolicy | ScopedSimAuthorityAccessPolicy | None,
        event_log: BoundedStationEventLog | None = None,
        operation_coordinator: NoIoStationOperationCoordinator
        | SideEffectStationOperationCoordinator
        | None = None,
        coordinator_kind: str = "none",
        maturity: str | None = None,
        authority_status_provider: Callable[[], dict[str, object]] | None = None,
    ):
        self.control_station_runtime = runtime
        self.control_station_access_policy = access_policy
        self.control_station_event_log = event_log
        self.control_station_operation_coordinator = operation_coordinator
        self.control_station_coordinator_kind = str(coordinator_kind)
        self.control_station_maturity = maturity
        self.control_station_authority_status_provider = authority_status_provider
        super().__init__(server_address, handler_class)


class _ReadOnlyControlStationHandler(BaseHTTPRequestHandler):
    server_version = "DexTeleopControlStation/1"

    @property
    def station_runtime(self) -> ControlStationRuntime:
        runtime = self.server.control_station_runtime
        if not isinstance(runtime, ControlStationRuntime):
            raise RuntimeError("Control Station web server runtime is invalid.")
        return runtime

    @property
    def access_policy(
        self,
    ) -> LocalIntentAccessPolicy | ScopedSimAuthorityAccessPolicy | None:
        policy = self.server.control_station_access_policy
        if policy is not None and not isinstance(
            policy,
            (LocalIntentAccessPolicy, ScopedSimAuthorityAccessPolicy),
        ):
            raise RuntimeError("Control Station web access policy is invalid.")
        return policy

    @property
    def event_log(self) -> BoundedStationEventLog | None:
        event_log = self.server.control_station_event_log
        if event_log is not None and not isinstance(event_log, BoundedStationEventLog):
            raise RuntimeError("Control Station event log is invalid.")
        return event_log

    @property
    def operation_coordinator(
        self,
    ) -> NoIoStationOperationCoordinator | SideEffectStationOperationCoordinator | None:
        coordinator = self.server.control_station_operation_coordinator
        kind = self.server.control_station_coordinator_kind
        expected_type = (
            NoIoStationOperationCoordinator
            if kind == "no_io"
            else SideEffectStationOperationCoordinator
            if kind == "side_effect"
            else None
        )
        if coordinator is not None and type(coordinator) is not expected_type:
            raise RuntimeError("Control Station operation coordinator is invalid.")
        return coordinator

    def _authority_status_payload(self) -> dict[str, object]:
        provider = self.server.control_station_authority_status_provider
        if provider is None:
            return {
                "status_kind": "unavailable",
                "motion_authorized": False,
                "authoritative": False,
            }
        try:
            payload = dict(provider())
        except Exception:
            return {
                "status_kind": "projection_failed",
                "motion_authorized": False,
                "authoritative": False,
            }
        payload["motion_authorized"] = bool(payload.get("motion_authorized", False))
        payload["authoritative"] = False
        return payload

    def log_message(self, _format: str, *_args) -> None:
        return None

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store")

    def _send_bytes(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
        *,
        headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        for name, value in headers:
            self.send_header(name, value)
        self._security_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _status_payload(self) -> dict[str, object]:
        runtime = self.station_runtime
        access_policy = self.access_policy
        payload = build_control_station_status_payload(
            runtime.manifest,
            runtime.state,
            intent_submission_enabled=access_policy is not None,
            event_fanout_enabled=self.event_log is not None,
            operation_coordinator_enabled=self.operation_coordinator is not None,
        )
        payload["capabilities"]["allowed_intent_kinds"] = (
            []
            if access_policy is None
            else list(access_policy.allowed_intent_kinds)
        )
        payload["capabilities"]["browser_raw_action_submission"] = False
        maturity = self.server.control_station_maturity
        if maturity is not None:
            payload["maturity"] = maturity
        authority = self._authority_status_payload()
        if authority["status_kind"] != "unavailable":
            payload["target"]["motion_authorized"] = authority[
                "motion_authorized"
            ]
            payload["target"]["authorization_status"] = authority["status_kind"]
            payload["control_facts"]["station_publication"] = authority
            payload["control_facts"]["dispatch_owner"] = authority.get(
                "owner_token_id",
                "unavailable",
            )
            payload["control_facts"]["authority_epoch"] = authority.get(
                "authority_epoch",
                "unavailable",
            )
            payload["control_facts"]["execution_epoch"] = authority.get(
                "execution_epoch",
                "unavailable",
            )
            payload["capabilities"]["motion_authority_requests"] = bool(
                access_policy is not None
                and "request_motion_authority"
                in access_policy.allowed_intent_kinds
            )
        return payload

    def _event_payload(self) -> dict[str, object]:
        event_log = self.event_log
        if event_log is None:
            raise RuntimeError("Control Station event fanout is disabled.")
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) - {"after_sequence"}:
            raise ValueError("Unsupported event query field.")
        values = query.get("after_sequence", ["-1"])
        if len(values) != 1:
            raise ValueError("after_sequence must appear at most once.")
        try:
            after_sequence = int(values[0])
        except ValueError as error:
            raise ValueError("after_sequence must be an integer.") from error
        snapshot = event_log.snapshot(after_sequence=after_sequence)
        return {
            "events": [
                {
                    "event_sequence": event.event_sequence,
                    "event_kind": event.event_kind,
                    "source_fact_id": event.source_fact_id,
                    "source_fact_hash": event.source_fact_hash,
                    "state_revision": event.state_revision,
                    "status": event.status,
                    "subject_id": event.subject_id,
                    "projected_at": event.projected_at,
                }
                for event in snapshot.events
            ],
            "requested_after_sequence": snapshot.requested_after_sequence,
            "oldest_available_sequence": snapshot.oldest_available_sequence,
            "latest_sequence": snapshot.latest_sequence,
            "next_sequence": snapshot.next_sequence,
            "gap_detected": snapshot.gap_detected,
            "total_evicted": snapshot.total_evicted,
            "authoritative": False,
            "motion_authorized": self._authority_status_payload()[
                "motion_authorized"
            ],
        }

    def _same_origin_if_present(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlsplit(origin)
        request_host = self.headers.get("Host")
        return (
            parsed.scheme == "http"
            and parsed.hostname in _LOCAL_BIND_HOSTS
            and request_host is not None
            and parsed.netloc == request_host
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path == "/api/health":
            writable = self.access_policy is not None
            coordinated = self.operation_coordinator is not None
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "maturity": self.server.control_station_maturity or (
                        "cs1c_no_io_operation_coordination"
                        if coordinated
                        else (
                            "cs1b_typed_intent_local_gui"
                            if writable
                            else "cs1a_read_only_local_gui"
                        )
                    ),
                    "read_only": not writable,
                    "motion_authorized": self._authority_status_payload()[
                        "motion_authorized"
                    ],
                    **(
                        {
                            "event_fanout_enabled": self.event_log is not None,
                            "operation_coordinator_enabled": coordinated,
                        }
                        if coordinated or self.event_log is not None
                        else {}
                    ),
                },
            )
            return
        if path == "/api/state":
            self._send_json(HTTPStatus.OK, self._status_payload())
            return
        if path == "/api/events":
            if self.event_log is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "event_fanout_not_enabled"},
                )
                return
            try:
                payload = self._event_payload()
            except ValueError as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_event_query", "message": str(error)},
                )
                return
            self._send_json(HTTPStatus.OK, payload)
            return
        static_name = _STATIC_ROUTES.get(path)
        if static_name is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "read_only": True},
            )
            return
        static_path = _STATIC_ROOT / static_name
        payload = static_path.read_bytes()
        media_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        charset = "; charset=utf-8" if media_type.startswith(("text/", "application/javascript")) else ""
        self._send_bytes(HTTPStatus.OK, payload, media_type + charset)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        policy = self.access_policy
        if policy is None:
            self._send_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {
                    "error": "cs1a_read_only",
                    "message": "Intent submission is disabled in Control Station CS-1a.",
                    "motion_authorized": False,
                },
                headers=(("Allow", "GET"),),
            )
            return
        if urlsplit(self.path).path != "/api/intents":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        authorization = self.headers.get("Authorization")
        if authorization is None:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "bearer_token_required", "motion_authorized": False},
                headers=(("WWW-Authenticate", "Bearer"),),
            )
            return
        if not policy.accepts_authorization_header(authorization):
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "invalid_bearer_token", "motion_authorized": False},
            )
            return
        if not self._same_origin_if_present():
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "cross_origin_request_rejected", "motion_authorized": False},
            )
            return
        media_type = self.headers.get("Content-Type", "").partition(";")[0].strip()
        if media_type != "application/json":
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "application_json_required"},
            )
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return
        try:
            content_length = int(raw_length)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty_request_body"})
            return
        if content_length > policy.max_request_bytes:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return
        try:
            decoded = self.rfile.read(content_length).decode("utf-8")

            def reject_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate JSON field: {key!r}.")
                    result[key] = value
                return result

            payload = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
            draft = station_intent_draft_from_json(payload, access_policy=policy)
        except PermissionError as error:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "intent_not_enabled", "message": str(error)},
            )
            return
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "invalid_typed_intent", "message": str(error)},
            )
            return

        coordinator = self.operation_coordinator
        try:
            if coordinator is None:
                transition = self.station_runtime.submit_intent(draft)
                enqueue_receipt = None
            else:
                submission = coordinator.submit_intent(draft)
                transition = submission.transition
                enqueue_receipt = submission.enqueue_receipt
        except StationOperationCoordinatorUnavailable as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": error.status,
                    "intent_admitted": False,
                    "motion_authorized": False,
                },
                headers=(("Retry-After", "1"),),
            )
            return
        receipt = transition.receipt
        self._send_json(
            HTTPStatus.OK,
            {
                "intent": {
                    "intent_id": transition.intent.intent_id,
                    "semantic_content_hash": transition.intent.semantic_content_hash,
                },
                "receipt": {
                    "accepted": receipt.accepted,
                    "status": receipt.status,
                    "admission_sequence": receipt.admission_sequence,
                    "state_revision_before": receipt.state_revision_before,
                    "state_revision_after": receipt.state_revision_after,
                    "resulting_state_hash": receipt.resulting_state_hash,
                    "semantic_content_hash": receipt.semantic_content_hash,
                },
                "station_state": self._status_payload(),
                "motion_authorized": self._authority_status_payload()[
                    "motion_authorized"
                ],
                "adapter_executed_inline": False,
                "operation_enqueue": (
                    None
                    if enqueue_receipt is None
                    else {
                        "coordinator_id": enqueue_receipt.coordinator_id,
                        "source_intent_id": enqueue_receipt.source_intent_id,
                        "accepted": enqueue_receipt.accepted,
                        "status": enqueue_receipt.status,
                        "queue_capacity": enqueue_receipt.queue_capacity,
                        "reserved_work_count": (
                            enqueue_receipt.reserved_work_count
                        ),
                    }
                ),
            },
        )

    def _reject_unsupported_mutating_method(self) -> None:
        allowed = "GET" if self.access_policy is None else "GET, POST"
        self._send_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": "method_not_allowed", "motion_authorized": False},
            headers=(("Allow", allowed),),
        )

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        self._reject_unsupported_mutating_method()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler API
        self._reject_unsupported_mutating_method()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler API
        self._reject_unsupported_mutating_method()


def build_read_only_control_station_server(
    runtime: ControlStationRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Build but do not start the CS-1a server.

    No production entrypoint calls this builder yet.  CS-1a deliberately
    rejects non-loopback binds and every POST request.
    """

    if not isinstance(runtime, ControlStationRuntime):
        raise TypeError("runtime must be a ControlStationRuntime.")
    normalized_host = str(host).strip()
    if normalized_host not in _LOCAL_BIND_HOSTS:
        raise ValueError("CS-1a web server only permits a loopback bind.")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be an integer in [0, 65535].")
    normalized_port = port
    if not 0 <= normalized_port <= 65535:
        raise ValueError("port must be an integer in [0, 65535].")
    return _ControlStationHttpServer(
        (normalized_host, normalized_port),
        _ReadOnlyControlStationHandler,
        runtime=runtime,
        access_policy=None,
        event_log=None,
        operation_coordinator=None,
        coordinator_kind="none",
    )


def build_typed_intent_control_station_server(
    runtime: ControlStationRuntime,
    *,
    access_policy: LocalIntentAccessPolicy,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Build but do not start the null-target CS-1b local intent server."""

    if not isinstance(runtime, ControlStationRuntime):
        raise TypeError("runtime must be a ControlStationRuntime.")
    if not isinstance(access_policy, LocalIntentAccessPolicy):
        raise TypeError("access_policy must be a LocalIntentAccessPolicy.")
    if runtime.manifest.execution_binding.target_kind != "null":
        raise ValueError("CS-1b typed intent server requires a null target.")
    if "motion" not in runtime.manifest.explicit_exclusions:
        raise ValueError("CS-1b manifest must explicitly exclude motion.")
    unsupported = sorted(
        set(access_policy.allowed_intent_kinds)
        - set(runtime.manifest.allowed_intent_kinds)
    )
    if unsupported:
        raise ValueError(
            f"Access policy enables intents absent from the manifest: {unsupported}."
        )
    normalized_host = str(host).strip()
    if normalized_host not in _LOCAL_BIND_HOSTS:
        raise ValueError("CS-1b web server only permits a loopback bind.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535].")
    return _ControlStationHttpServer(
        (normalized_host, port),
        _ReadOnlyControlStationHandler,
        runtime=runtime,
        access_policy=access_policy,
        event_log=None,
        operation_coordinator=None,
        coordinator_kind="none",
    )


def build_coordinated_control_station_server(
    runtime: ControlStationRuntime,
    *,
    access_policy: LocalIntentAccessPolicy,
    event_log: BoundedStationEventLog,
    operation_coordinator: NoIoStationOperationCoordinator,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Build but do not start the null/fake CS-1c application server."""

    if not isinstance(event_log, BoundedStationEventLog):
        raise TypeError("event_log must be a BoundedStationEventLog.")
    if type(operation_coordinator) is not NoIoStationOperationCoordinator:
        raise TypeError(
            "operation_coordinator must be a NoIoStationOperationCoordinator."
        )
    if operation_coordinator.runtime is not runtime:
        raise ValueError("Coordinator and web server must share one runtime.")
    if operation_coordinator.event_log is not event_log:
        raise ValueError("Coordinator and web server must share one event log.")
    server = build_typed_intent_control_station_server(
        runtime,
        access_policy=access_policy,
        host=host,
        port=port,
    )
    server.control_station_event_log = event_log
    server.control_station_operation_coordinator = operation_coordinator
    server.control_station_coordinator_kind = "no_io"
    return server


def build_scoped_sim_authority_control_station_server(
    runtime: ControlStationRuntime,
    *,
    access_policy: ScopedSimAuthorityAccessPolicy,
    event_log: BoundedStationEventLog,
    operation_coordinator: SideEffectStationOperationCoordinator,
    authority_status_provider: Callable[[], dict[str, object]],
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Build but do not start the explicit CS-2a2b scoped-sim API.

    The endpoint accepts typed lifecycle/candidate/authority intents only. It
    cannot carry action values, bind a real target, or serve as a production
    entrypoint.
    """

    if not isinstance(runtime, ControlStationRuntime):
        raise TypeError("runtime must be a ControlStationRuntime.")
    if type(access_policy) is not ScopedSimAuthorityAccessPolicy:
        raise TypeError(
            "access_policy must be a ScopedSimAuthorityAccessPolicy."
        )
    if not isinstance(event_log, BoundedStationEventLog):
        raise TypeError("event_log must be a BoundedStationEventLog.")
    if type(operation_coordinator) is not SideEffectStationOperationCoordinator:
        raise TypeError(
            "operation_coordinator must be a SideEffectStationOperationCoordinator."
        )
    if operation_coordinator.runtime is not runtime:
        raise ValueError("Coordinator and web server must share one runtime.")
    if operation_coordinator.event_log is not event_log:
        raise ValueError("Coordinator and web server must share one event log.")
    if runtime.manifest.execution_binding.target_kind != "sim":
        raise ValueError("Scoped authority Web API requires an exact sim target.")
    required_exclusions = {
        "browser_raw_action_submission",
        "production_entrypoint",
        "real_motion",
    }
    missing_exclusions = sorted(
        required_exclusions - set(runtime.manifest.explicit_exclusions)
    )
    if missing_exclusions:
        raise ValueError(
            "Scoped sim manifest lacks required exclusions: "
            f"{missing_exclusions}."
        )
    unsupported = sorted(
        set(access_policy.allowed_intent_kinds)
        - set(runtime.manifest.allowed_intent_kinds)
    )
    if unsupported:
        raise ValueError(
            f"Access policy enables intents absent from the manifest: {unsupported}."
        )
    if not callable(authority_status_provider):
        raise TypeError("authority_status_provider must be callable.")
    normalized_host = str(host).strip()
    if normalized_host not in _LOCAL_BIND_HOSTS:
        raise ValueError("Scoped sim Web API only permits a loopback bind.")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535].")
    return _ControlStationHttpServer(
        (normalized_host, port),
        _ReadOnlyControlStationHandler,
        runtime=runtime,
        access_policy=access_policy,
        event_log=event_log,
        operation_coordinator=operation_coordinator,
        coordinator_kind="side_effect",
        maturity="cs2a2b_scoped_sim_authority_api",
        authority_status_provider=authority_status_provider,
    )


__all__ = [
    "build_coordinated_control_station_server",
    "build_read_only_control_station_server",
    "build_scoped_sim_authority_control_station_server",
    "build_typed_intent_control_station_server",
]

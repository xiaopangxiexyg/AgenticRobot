from __future__ import annotations

import io
import json
import unittest
from typing import Any

from teleop_runtime.http_security import (
    CONTROL_TOKEN_HEADER,
    WebControlSecurity,
    generate_session_token,
    validate_bind_host,
)
from teleop_runtime.replay.executor import ReplayExecutionSettings
from teleop_runtime.replay.web_gui import make_handler


class _FakeReplaySession:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def state(self) -> dict[str, Any]:
        return {"payload_count": len(self.payloads)}

    def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return self.state()

    def camera_jpeg(self, camera_key: str) -> bytes:
        del camera_key
        return b"jpeg"

    def tactile_jpeg(self, side: str) -> bytes:
        del side
        return b"jpeg"

    def position_collision_debug_snapshot(self, *, limit: int) -> dict[str, Any]:
        return {"limit": int(limit), "events": []}


class _MemorySocket:
    def __init__(self, request: bytes) -> None:
        self._request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BytesIO:
        del args, kwargs
        if "r" not in mode:
            raise AssertionError(f"unexpected makefile mode: {mode}")
        return self._request

    def sendall(self, data: bytes) -> None:
        self.response.extend(data)


class _HttpServer:
    server_address = ("127.0.0.1", 8765)


class _HttpHarness:
    def __init__(self, *, hardware_output_armed: bool = False, max_body: int = 64 * 1024) -> None:
        self.session = _FakeReplaySession()
        self.token = generate_session_token()
        self.handler = make_handler(
            self.session,  # type: ignore[arg-type]
            security=WebControlSecurity(
                token=self.token,
                bind_host="127.0.0.1",
                max_control_body_bytes=max_body,
            ),
            hardware_output_armed=hardware_output_armed,
        )

    @property
    def port(self) -> int:
        return 8765

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(
        self,
        payload: object,
        *,
        token: str | None = None,
        origin: str | None = None,
        content_type: str = "application/json",
        host: str | None = None,
        raw_body: bytes | None = None,
    ) -> tuple[int, dict[str, Any]]:
        body = raw_body if raw_body is not None else json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": content_type,
            "Origin": self.origin if origin is None else origin,
        }
        if token is not None:
            headers[CONTROL_TOKEN_HEADER] = token
        if host is not None:
            headers["Host"] = host
        lines = ["POST /api/control HTTP/1.1"]
        if "Host" not in headers:
            headers["Host"] = f"127.0.0.1:{self.port}"
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        lines.extend(f"{key}: {value}" for key, value in headers.items())
        request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body
        status, response_body = self._exchange(request)
        return status, json.loads(response_body.decode("utf-8"))

    def get(self, path: str = "/") -> tuple[int, bytes]:
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        return self._exchange(request)

    def _exchange(self, request: bytes) -> tuple[int, bytes]:
        connection = _MemorySocket(request)
        self.handler(connection, ("127.0.0.1", 50000), _HttpServer())
        head, separator, body = bytes(connection.response).partition(b"\r\n\r\n")
        if not separator:
            raise AssertionError(f"invalid HTTP response: {connection.response!r}")
        status_line = head.splitlines()[0]
        status = int(status_line.split()[1])
        return status, body


class ReplayWebSecurityTest(unittest.TestCase):
    def test_generated_control_token_is_high_entropy_and_per_session(self) -> None:
        first = generate_session_token()
        second = generate_session_token()
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(len(first), 43)

    def test_control_post_requires_token(self) -> None:
        running = _HttpHarness()
        status, payload = running.request({"type": "pause"})
        self.assertEqual(status, 401)
        self.assertFalse(payload["ok"])
        self.assertEqual(running.session.payloads, [])

    def test_root_page_receives_only_its_per_session_token(self) -> None:
        running = _HttpHarness()
        status, body = running.get()
        self.assertEqual(status, 200)
        self.assertIn(running.token.encode("ascii"), body)
        self.assertNotIn(b"__REPLAY_CONTROL_TOKEN_JSON__", body)

    def test_collision_debug_endpoint_is_bounded_and_read_only(self) -> None:
        running = _HttpHarness()
        status, body = running.get("/api/position-collision-debug?limit=7")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["position_collision_debug"]["limit"], 7)
        self.assertEqual(running.session.payloads, [])

        status, body = running.get("/api/position-collision-debug?limit=bad")
        self.assertEqual(status, 400)
        self.assertIn("must be an integer", json.loads(body)["error"])

    def test_control_post_rejects_cross_origin_and_bad_host(self) -> None:
        running = _HttpHarness()
        status, _ = running.request(
            {"type": "pause"},
            token=running.token,
            origin="http://evil.example",
        )
        self.assertEqual(status, 403)
        status, _ = running.request(
            {"type": "pause"},
            token=running.token,
            host=f"evil.example:{running.port}",
        )
        self.assertEqual(status, 403)
        self.assertEqual(running.session.payloads, [])

    def test_control_post_requires_json_and_limits_body(self) -> None:
        running = _HttpHarness(max_body=32)
        status, _ = running.request(
            {"type": "pause"},
            token=running.token,
            content_type="text/plain",
        )
        self.assertEqual(status, 415)
        status, _ = running.request(
            {},
            token=running.token,
            raw_body=b"{" + b"x" * 64 + b"}",
        )
        self.assertEqual(status, 413)
        self.assertEqual(running.session.payloads, [])

    def test_valid_same_origin_control_request_dispatches(self) -> None:
        running = _HttpHarness()
        status, payload = running.request({"type": "pause"}, token=running.token)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(running.session.payloads, [{"type": "pause"}])

    def test_non_loopback_bind_requires_explicit_unsafe_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            validate_bind_host("192.0.2.10")
        validate_bind_host("192.0.2.10", unsafe_allow_remote=True)
        with self.assertRaisesRegex(ValueError, "wildcard"):
            validate_bind_host("0.0.0.0", unsafe_allow_remote=True)
        validate_bind_host(
            "0.0.0.0",
            unsafe_allow_remote=True,
            allowed_hosts=("robot-console.example",),
        )

    def test_http_cannot_arm_hardware_output_without_local_capability(self) -> None:
        running = _HttpHarness(hardware_output_armed=False)
        for payload in (
            {"type": "set_send_actions", "enabled": True},
            {"type": "configure_hardware", "send_actions": True},
            {"type": "move_to_start"},
            {"type": "start_hardware_replay"},
        ):
            status, response = running.request(payload, token=running.token)
            self.assertEqual(status, 403)
            self.assertIn("not armed", response["error"])
        self.assertEqual(running.session.payloads, [])

        status, _ = running.request(
            {"type": "set_send_actions", "enabled": False},
            token=running.token,
        )
        self.assertEqual(status, 200)

        armed = _HttpHarness(hardware_output_armed=True)
        status, _ = armed.request(
            {"type": "set_send_actions", "enabled": True},
            token=armed.token,
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(armed.session.payloads), 1)

    def test_send_actions_string_cannot_bypass_local_arming_check(self) -> None:
        running = _HttpHarness(hardware_output_armed=False)
        status, _ = running.request(
            {"type": "set_send_actions", "enabled": "false"},
            token=running.token,
        )
        self.assertEqual(status, 200)

        current = ReplayExecutionSettings(send_actions_enabled=False)
        settings = ReplayExecutionSettings.from_payload(
            {"send_actions": "false"},
            current,
        )
        self.assertFalse(settings.send_actions_enabled)
        with self.assertRaisesRegex(ValueError, "send_actions must be a boolean"):
            ReplayExecutionSettings.from_payload(
                {"send_actions": "not-a-boolean"},
                current,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import struct
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from hardware.quest_dev import server
from hardware.quest_dev.features.visualization.protocol import (
    FEATURE_PROTOCOL_VERSION,
    feedback_error_message,
    robot_tcp_message,
)
from hardware.quest_dev.features.visualization.runtime import VisualizationRuntime
from hardware.quest_dev.protocol import PROTOCOL_VERSION

MODULE_ROOT = Path(__file__).resolve().parents[1]


class WebSocketFrameTests(unittest.TestCase):
    def test_short_text_frame_contains_json_payload(self) -> None:
        frame = server.ws_text_frame({"type": "test", "value": 1})
        self.assertEqual(frame[0], 0x81)
        self.assertLess(frame[1], 126)
        payload = json.loads(frame[2:].decode("utf-8"))
        self.assertEqual(payload, {"type": "test", "value": 1})

    def test_medium_frame_uses_16_bit_length(self) -> None:
        data = b"x" * 300
        frame = server.ws_frame(0x1, data)
        self.assertEqual(frame[:2], bytes((0x81, 126)))
        self.assertEqual(struct.unpack("!H", frame[2:4])[0], len(data))
        self.assertEqual(frame[4:], data)

    def test_client_frame_requires_mask_and_decodes_masked_payload(self) -> None:
        handler = object.__new__(server.QuestDevHandler)
        handler.rfile = io.BytesIO(b"\x81\x02{}")
        with self.assertRaisesRegex(ValueError, "masked"):
            handler._read_ws_message()

        payload = b'{"type":"probe"}'
        mask = b"\x01\x02\x03\x04"
        encoded = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        handler.rfile = io.BytesIO(bytes((0x81, 0x80 | len(payload))) + mask + encoded)
        self.assertEqual(handler._read_ws_message(), (0x1, payload))

    def test_pre_auth_read_uses_absolute_deadline(self) -> None:
        class SlowReader:
            def read1(self, _length: int) -> bytes:
                return b"x"

        handler = object.__new__(server.QuestDevHandler)
        handler.rfile = SlowReader()
        handler.connection = mock.Mock()
        with (
            mock.patch.object(server.time, "monotonic", side_effect=[0.0, 0.6]),
            self.assertRaisesRegex(TimeoutError, "deadline"),
        ):
            handler._read_exact(2, deadline=0.5)

    def test_oversized_control_frame_is_rejected(self) -> None:
        handler = object.__new__(server.QuestDevHandler)
        handler.rfile = io.BytesIO(b"\x89\xfe\x00\x7e")
        with self.assertRaisesRegex(ValueError, "control frames"):
            handler._read_ws_message()


class ConfigurationTests(unittest.TestCase):
    def test_default_config_is_mapping_and_read_only(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        self.assertIsInstance(config, dict)
        visualization = config["features"]["visualization"]
        self.assertTrue(visualization["enabled"])
        self.assertFalse(visualization["robot_feedback"]["enabled"])
        self.assertTrue(config["features"]["controller_pose"]["enabled"])
        self.assertTrue(config["server"]["pairing"]["enabled"])

    def test_client_config_contains_only_feature_client_fields(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        features = server.feature_configs(config)
        client = server.build_client_config(config, features)
        visualization = client["features"]["visualization"]
        self.assertTrue(visualization["enabled"])
        self.assertIn("axis_length_m", visualization)
        self.assertNotIn("robot_feedback", visualization)
        self.assertTrue(client["security"]["pairing_required"])
        self.assertNotIn("token_env", client["security"])
        self.assertNotIn("admission", client["features"]["controller_pose"])

    def test_client_config_cannot_override_host_enabled_state(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        config["features"]["visualization"]["client"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "cannot override"):
            server.build_client_config(config, server.feature_configs(config))

    def test_relative_paths_are_module_relative(self) -> None:
        resolved = server.resolve_module_path("assets/certs/example.crt")
        self.assertEqual(resolved, MODULE_ROOT / "assets/certs/example.crt")

    def test_plain_http_is_forced_to_loopback(self) -> None:
        self.assertEqual(
            server.resolve_bind_host({"host": "0.0.0.0"}, use_https=False),
            "127.0.0.1",
        )
        self.assertEqual(
            server.resolve_bind_host({"host": "0.0.0.0"}, use_https=True),
            "0.0.0.0",
        )

    def test_pairing_token_is_bounded_url_safe_ascii(self) -> None:
        self.assertEqual(
            server.validate_pairing_token("valid-token_123456789"),
            "valid-token_123456789",
        )
        for invalid in (
            "short",
            "含非ASCII字符的token-123456",
            "contains space 123456",
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "URL-safe ASCII"),
            ):
                server.validate_pairing_token(invalid)

    def test_lan_ip_probe_falls_back_when_socket_creation_is_denied(self) -> None:
        with mock.patch.object(server.socket, "socket", side_effect=PermissionError):
            self.assertEqual(server.detect_lan_ip(), "127.0.0.1")

    def test_server_protocol_version_has_single_source(self) -> None:
        self.assertEqual(server.PROTOCOL_VERSION, PROTOCOL_VERSION)


class AuthenticationTests(unittest.TestCase):
    @staticmethod
    def make_server() -> server.QuestDevHTTPServer:
        host = object.__new__(server.QuestDevHTTPServer)
        host.client_config = {}
        host._pairing_token = "correct-pairing-token-1234"
        host.require_same_origin = True
        host.max_ws_clients = 2
        host.auth_timeout_s = 5.0
        host.use_https = True
        host.server_instance_id = "server-instance"
        host.lock = threading.RLock()
        host.ws_connections = {
            1: server.WebSocketClient(1, "127.0.0.1", io.BytesIO()),
            2: server.WebSocketClient(2, "127.0.0.2", io.BytesIO()),
        }
        host.next_ws_client_id = 3
        host.last_client_message = None
        host.last_feature_messages = {}
        host.client_message_handlers = {}
        host.client_disconnect_handlers = []
        host.feature_status_providers = {}
        host.started_at = time.time()
        return host

    def test_same_origin_requires_matching_host(self) -> None:
        self.assertTrue(
            server.same_origin_allowed("https://10.0.0.2:8444", "10.0.0.2:8444")
        )
        self.assertFalse(
            server.same_origin_allowed("https://attacker.invalid", "10.0.0.2:8444")
        )
        self.assertFalse(
            server.same_origin_allowed(
                "http://10.0.0.2:8444",
                "10.0.0.2:8444",
                expected_scheme="https",
            )
        )
        self.assertFalse(server.same_origin_allowed(None, "10.0.0.2:8444"))

    def test_status_bearer_auth_and_connection_limit(self) -> None:
        host = self.make_server()
        self.assertTrue(
            server.bearer_token_matches(
                f"Bearer {host._pairing_token}", host._pairing_token
            )
        )
        self.assertFalse(
            server.bearer_token_matches("Bearer wrong", host._pairing_token)
        )
        with self.assertRaisesRegex(ConnectionError, "client limit"):
            host.add_ws_client("127.0.0.3", io.BytesIO())

    def test_feature_messages_require_authentication(self) -> None:
        host = self.make_server()
        handler_calls: list[dict] = []
        host.register_client_message_handler(
            "probe", lambda _, value: handler_calls.append(value)
        )

        self.assertFalse(host.process_client_message(1, {"type": "probe"}))
        self.assertEqual(handler_calls, [])
        self.assertFalse(host.is_ws_client_authenticated(1))

    def test_valid_token_authenticates_without_status_leak(self) -> None:
        host = self.make_server()
        token = host._pairing_token
        authenticated = host.process_client_message(
            1,
            {
                "type": "client_auth",
                "protocol_version": PROTOCOL_VERSION,
                "client_instance_id": "browser-instance",
                "pairing_token": token,
            },
        )
        self.assertTrue(authenticated)
        self.assertTrue(host.is_ws_client_authenticated(1))
        self.assertNotIn(token, json.dumps(host.snapshot()))
        self.assertTrue(host.snapshot()["clients"][0]["authenticated"])

    def test_post_auth_message_must_match_authenticated_instance(self) -> None:
        host = self.make_server()
        self.assertTrue(
            host.process_client_message(
                1,
                {
                    "type": "client_auth",
                    "protocol_version": PROTOCOL_VERSION,
                    "client_instance_id": "browser-instance",
                    "pairing_token": host._pairing_token,
                },
            )
        )
        self.assertFalse(
            host.process_client_message(
                1,
                {"type": "probe", "client_instance_id": "different-browser"},
            )
        )

    def test_broadcast_excludes_unauthenticated_clients(self) -> None:
        host = self.make_server()
        host.ws_connections[1].authenticated = True
        sent = host.broadcast({"type": "feature_message"})
        self.assertEqual(sent, 1)
        self.assertNotEqual(host.ws_connections[1].writer.getvalue(), b"")
        self.assertEqual(host.ws_connections[2].writer.getvalue(), b"")

    def test_status_summary_drops_pose_and_credentials(self) -> None:
        summary = server.summarize_message(
            {
                "type": "controller_pose_sample",
                "pairing_token": "secret",
                "controllers": [{"matrix_reference_from_grip": list(range(16))}],
                "sample_sequence": 4,
            }
        )
        self.assertEqual(summary["controller_count"], 1)
        self.assertEqual(summary["sample_sequence"], 4)
        self.assertNotIn("pairing_token", summary)
        self.assertNotIn("controllers", summary)


class FeatureRuntimeTests(unittest.TestCase):
    def test_visualization_messages_carry_feature_protocol_version(self) -> None:
        for message in (robot_tcp_message({}), feedback_error_message("offline")):
            self.assertEqual(
                message["feature_protocol_version"], FEATURE_PROTOCOL_VERSION
            )

    def test_loop_broadcasts_read_only_feedback(self) -> None:
        stop_event = threading.Event()

        class Reader:
            def get(self, key: str) -> str:
                self.last_key = key
                return json.dumps(
                    {
                        "timestamp": 100.0,
                        "robot": "ar5_1",
                        "control_ready": False,
                        "tcp_pose": {
                            "position": [0, 0, 0],
                            "orientation_quat_xyzw": [0, 0, 0, 1],
                        },
                    }
                )

        class BroadcastServer:
            def __init__(self) -> None:
                self.messages: list[dict] = []

            def publish_feature_message(self, feature: str, payload: dict) -> int:
                self.feature = feature
                self.messages.append(payload)
                stop_event.set()
                return 1

        host = BroadcastServer()
        runtime = VisualizationRuntime(host, {}, stop_event)
        runtime.reader = Reader()
        with mock.patch(
            "hardware.quest_dev.features.visualization.robot_feedback.time.time",
            return_value=100.0,
        ):
            runtime._run_feedback_loop(
                sources={"ar5_1": "ar5:feedback:ar5_1"},
                rate_hz=30,
                max_age_s=0.25,
            )
        self.assertEqual(host.feature, "visualization")
        self.assertEqual(len(host.messages), 1)
        self.assertTrue(host.messages[0]["robots"]["ar5_1"]["fresh"])
        self.assertFalse(host.messages[0]["robots"]["ar5_1"]["control_ready"])


if __name__ == "__main__":
    unittest.main()

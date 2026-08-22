from __future__ import annotations

import json
import struct
import threading
import unittest
from pathlib import Path
from unittest import mock

from hardware.quest_dev import server
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


class ConfigurationTests(unittest.TestCase):
    def test_default_config_is_mapping_and_read_only(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        self.assertIsInstance(config, dict)
        visualization = config["features"]["visualization"]
        self.assertTrue(visualization["enabled"])
        self.assertFalse(visualization["robot_feedback"]["enabled"])

    def test_client_config_contains_only_feature_client_fields(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        features = server.feature_configs(config)
        client = server.build_client_config(config, features)
        visualization = client["features"]["visualization"]
        self.assertTrue(visualization["enabled"])
        self.assertIn("axis_length_m", visualization)
        self.assertNotIn("robot_feedback", visualization)

    def test_relative_paths_are_module_relative(self) -> None:
        resolved = server.resolve_module_path("assets/certs/example.crt")
        self.assertEqual(resolved, MODULE_ROOT / "assets/certs/example.crt")

    def test_lan_ip_probe_falls_back_when_socket_creation_is_denied(self) -> None:
        with mock.patch.object(server.socket, "socket", side_effect=PermissionError):
            self.assertEqual(server.detect_lan_ip(), "127.0.0.1")

    def test_server_protocol_version_has_single_source(self) -> None:
        self.assertEqual(server.PROTOCOL_VERSION, PROTOCOL_VERSION)


class FeatureRuntimeTests(unittest.TestCase):
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

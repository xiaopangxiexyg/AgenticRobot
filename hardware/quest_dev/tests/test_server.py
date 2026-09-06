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

    def test_realman_left_override_is_explicit_and_keeps_right_disabled(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        server.apply_realman_observation_overrides(
            config,
            left_ip="192.168.101.21",
            left_urdf=Path("/reviewed/rm65_arm.urdf"),
        )
        feedback = config["features"]["visualization"]["robot_feedback"]
        self.assertTrue(feedback["enabled"])
        self.assertEqual(
            feedback["realman"]["arms"]["rm65_left"],
            {
                "enabled": True,
                "side": "left",
                "arm_ip": "192.168.101.21",
                "arm_urdf_path": "/reviewed/rm65_arm.urdf",
            },
        )
        self.assertFalse(feedback["realman"]["arms"]["rm65_right"]["enabled"])

    def test_realman_left_override_requires_both_reviewed_inputs(self) -> None:
        config = server.load_config(MODULE_ROOT / "config.yaml")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            server.apply_realman_observation_overrides(
                config,
                left_ip="192.168.101.21",
            )

    def test_plain_http_is_forced_to_loopback(self) -> None:
        self.assertEqual(
            server.resolve_bind_host({"host": "0.0.0.0"}, use_https=False),
            "127.0.0.1",
        )
        self.assertEqual(
            server.resolve_bind_host({"host": "0.0.0.0"}, use_https=True),
            "0.0.0.0",
        )

    def test_pairing_token_is_exactly_six_ascii_digits(self) -> None:
        self.assertEqual(server.validate_pairing_token("042731"), "042731")
        for invalid in (
            "12345",
            "1234567",
            "１２３４５６",
            "12a456",
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "exactly 6 ASCII digits"),
            ):
                server.validate_pairing_token(invalid)

    def test_default_pairing_code_is_six_digits_and_keeps_leading_zeros(self) -> None:
        config = {"pairing": {"token_env": "QUEST_DEV_TEST_PAIRING_TOKEN"}}
        with (
            mock.patch.dict(server.os.environ, {}, clear=True),
            mock.patch.object(server.secrets, "randbelow", return_value=42),
        ):
            code, require_same_origin = server.resolve_pairing_settings(config)
        self.assertEqual(code, "000042")
        self.assertTrue(require_same_origin)

    def test_fixed_pairing_code_must_also_be_six_digits(self) -> None:
        config = {"pairing": {"token_env": "QUEST_DEV_TEST_PAIRING_TOKEN"}}
        with mock.patch.dict(
            server.os.environ,
            {"QUEST_DEV_TEST_PAIRING_TOKEN": "314159"},
            clear=True,
        ):
            code, _ = server.resolve_pairing_settings(config)
        self.assertEqual(code, "314159")

    def test_lan_ip_probe_falls_back_when_socket_creation_is_denied(self) -> None:
        with mock.patch.object(server.socket, "socket", side_effect=PermissionError):
            self.assertEqual(server.detect_lan_ip(), "127.0.0.1")

    def test_server_protocol_version_has_single_source(self) -> None:
        self.assertEqual(server.PROTOCOL_VERSION, PROTOCOL_VERSION)


class _FakeFeatureRuntime:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start
        self.on_stop = None

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise ValueError(f"cannot start {self.name}")

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        callback = self.on_stop
        self.on_stop = None
        if callback is not None:
            callback()


class _OtherFeatureRuntime:
    pass


class _FakeHostServer:
    server_instance_id = "fake-server"

    def __init__(
        self,
        events: list[str],
        *,
        fail_serve: bool = False,
        block_serve: bool = False,
        fail_close: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.fail_serve = fail_serve
        self.block_serve = block_serve
        self.fail_close = fail_close
        self.close_error = close_error
        self.serve_entered = threading.Event()
        self.serve_release = threading.Event()
        self.socket = object()
        self.server_address = ("127.0.0.1", 8444)

    def serve_forever(self) -> None:
        self.events.append("serve")
        self.serve_entered.set()
        if self.fail_serve:
            raise ValueError("serve failed")
        if self.block_serve and not self.serve_release.wait(timeout=2.0):
            raise TimeoutError("fake server was not released")

    def shutdown(self) -> None:
        self.events.append("shutdown")
        self.serve_release.set()

    def server_close(self) -> None:
        self.events.append("server_close")
        if self.close_error is not None:
            raise self.close_error
        if self.fail_close:
            raise RuntimeError("server-close-probe")


class QuestDevHostTests(unittest.TestCase):
    @staticmethod
    def launch_info() -> server.QuestDevLaunchInfo:
        return server.QuestDevLaunchInfo(
            base_url="http://127.0.0.1:8444/",
            pairing_url="http://127.0.0.1:8444/#pair=314159",
            pairing_code="314159",
            use_https=False,
        )

    def make_host(
        self,
        events: list[str],
        features: dict[str, dict],
        *,
        fake_server: _FakeHostServer | None = None,
        fail_start: str | None = None,
    ) -> tuple[
        server.QuestDevHost,
        dict[str, _FakeFeatureRuntime],
        list[threading.Event],
    ]:
        host_server = fake_server or _FakeHostServer(events)
        runtimes: dict[str, _FakeFeatureRuntime] = {}
        stop_events: list[threading.Event] = []

        def load_runtime(
            name: str,
            supplied_server,
            config: dict,
            stop_event: threading.Event,
        ) -> _FakeFeatureRuntime:
            self.assertIs(supplied_server, host_server)
            self.assertTrue(config["enabled"])
            events.append(f"load:{name}")
            stop_events.append(stop_event)
            runtime = _FakeFeatureRuntime(
                name,
                events,
                fail_start=name == fail_start,
            )
            runtimes[name] = runtime
            return runtime

        host = server.QuestDevHost(
            host_server,
            features,
            self.launch_info(),
            runtime_loader=load_runtime,
        )
        return host, runtimes, stop_events

    def test_enabled_runtime_is_started_once_and_borrowed_by_identity(self) -> None:
        events: list[str] = []
        host, runtimes, stop_events = self.make_host(
            events,
            {
                "first": {"enabled": True},
                "disabled": {"enabled": False},
                "second": {"enabled": True},
            },
        )

        with self.assertRaisesRegex(RuntimeError, "not running"):
            host.require_running_feature_runtime("first", _FakeFeatureRuntime)

        host.start()
        host.start()
        first = host.require_running_feature_runtime("first", _FakeFeatureRuntime)
        self.assertIs(first, runtimes["first"])
        self.assertIs(
            host.require_running_feature_runtime("first", _FakeFeatureRuntime),
            first,
        )
        with self.assertRaisesRegex(KeyError, "not enabled"):
            host.require_running_feature_runtime("disabled", _FakeFeatureRuntime)
        with self.assertRaisesRegex(TypeError, "expected _OtherFeatureRuntime"):
            host.require_running_feature_runtime("first", _OtherFeatureRuntime)

        self.assertEqual(stop_events, [host.stop_event, host.stop_event])
        host.close()
        host.close()
        self.assertEqual(
            events,
            [
                "load:first",
                "start:first",
                "load:second",
                "start:second",
                "stop:second",
                "stop:first",
                "server_close",
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "not running"):
            host.require_running_feature_runtime("first", _FakeFeatureRuntime)

    def test_partial_start_failure_stops_failing_runtime_and_prior_runtime(
        self,
    ) -> None:
        events: list[str] = []
        host, _, _ = self.make_host(
            events,
            {
                "first": {"enabled": True},
                "broken": {"enabled": True},
                "never": {"enabled": True},
            },
            fail_start="broken",
        )

        with self.assertRaisesRegex(ValueError, "cannot start broken"):
            host.start()
        host.close()
        self.assertEqual(
            events,
            [
                "load:first",
                "start:first",
                "load:broken",
                "start:broken",
                "stop:broken",
                "stop:first",
                "server_close",
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "cannot start from fault"):
            host.start()

    def test_serve_failure_still_stops_features_and_closes_server(self) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events, fail_serve=True)
        host, runtimes, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
            fake_server=fake_server,
        )
        callback_runtime: list[_FakeFeatureRuntime] = []

        with self.assertRaisesRegex(ValueError, "serve failed"):
            host.serve_forever(
                ready_callback=lambda: callback_runtime.append(
                    host.require_running_feature_runtime(
                        "controller_pose", _FakeFeatureRuntime
                    )
                )
            )

        self.assertEqual(callback_runtime, [runtimes["controller_pose"]])
        self.assertEqual(
            events,
            [
                "load:controller_pose",
                "start:controller_pose",
                "serve",
                "stop:controller_pose",
                "server_close",
            ],
        )

    def test_cleanup_failure_is_permanently_re_reported(self) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events, fail_close=True)
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
            fake_server=fake_server,
        )
        host.start()

        with self.assertRaisesRegex(RuntimeError, "server-close-probe"):
            host.close()
        with self.assertRaisesRegex(RuntimeError, "server-close-probe"):
            host.close()

        self.assertEqual(events.count("stop:controller_pose"), 1)
        self.assertEqual(events.count("server_close"), 1)

    def test_cleanup_base_exception_is_latched_before_propagation(self) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(
            events,
            close_error=KeyboardInterrupt("server-close-interrupt"),
        )
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
            fake_server=fake_server,
        )
        host.start()

        with self.assertRaises(KeyboardInterrupt):
            host.close()
        with self.assertRaisesRegex(RuntimeError, "server-close-interrupt"):
            host.close()

        self.assertEqual(events.count("stop:controller_pose"), 1)
        self.assertEqual(events.count("server_close"), 1)

    def test_shutdown_before_serving_never_calls_blocking_server_shutdown(
        self,
    ) -> None:
        events: list[str] = []
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
        )
        host.start()

        self.assertTrue(host.request_shutdown())
        self.assertFalse(host.request_shutdown())
        with self.assertRaisesRegex(RuntimeError, "shutdown was already requested"):
            host.start()
        host.serve_forever()
        self.assertEqual(
            events,
            [
                "load:controller_pose",
                "start:controller_pose",
                "stop:controller_pose",
                "server_close",
            ],
        )

    def test_shutdown_between_serve_entry_and_start_still_closes_server(self) -> None:
        events: list[str] = []
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
        )
        original_start = host.start

        def request_shutdown_then_start() -> None:
            self.assertTrue(host.request_shutdown())
            original_start()

        with (
            mock.patch.object(host, "start", side_effect=request_shutdown_then_start),
            self.assertRaisesRegex(RuntimeError, "shutdown was already requested"),
        ):
            host.serve_forever()

        self.assertEqual(events, ["server_close"])

    def test_shutdown_after_start_before_serving_enters_loop_to_release_worker(
        self,
    ) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events, block_serve=True)
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
            fake_server=fake_server,
        )
        original_start = host.start

        def start_then_request_shutdown() -> None:
            original_start()
            self.assertTrue(host.request_shutdown())

        with mock.patch.object(
            host,
            "start",
            side_effect=start_then_request_shutdown,
        ):
            host.serve_forever()

        self.assertEqual(events.count("serve"), 1)
        self.assertEqual(events.count("shutdown"), 1)
        self.assertEqual(events.count("server_close"), 1)
        self.assertEqual(events[-2:], ["stop:controller_pose", "server_close"])

    def test_ready_callback_failure_after_shutdown_request_starts_no_worker(
        self,
    ) -> None:
        events: list[str] = []
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
        )

        def fail_ready() -> None:
            self.assertTrue(host.request_shutdown())
            raise ValueError("ready failed")

        with self.assertRaisesRegex(ValueError, "ready failed"):
            host.serve_forever(ready_callback=fail_ready)

        self.assertNotIn("shutdown", events)
        self.assertNotIn("serve", events)
        self.assertEqual(events[-2:], ["stop:controller_pose", "server_close"])

    def test_serving_shutdown_uses_another_thread_and_closes_once(self) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events, block_serve=True)
        host, _, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
            fake_server=fake_server,
        )
        serve_thread = threading.Thread(target=host.serve_forever)
        serve_thread.start()
        self.assertTrue(fake_server.serve_entered.wait(timeout=1.0))

        self.assertTrue(host.request_shutdown())
        serve_thread.join(timeout=2.0)

        self.assertFalse(serve_thread.is_alive())
        self.assertEqual(events.count("shutdown"), 1)
        self.assertEqual(events.count("server_close"), 1)
        self.assertEqual(events[-2:], ["stop:controller_pose", "server_close"])

    def test_close_is_reentrant_when_a_runtime_stop_callback_closes_host(self) -> None:
        events: list[str] = []
        host, runtimes, _ = self.make_host(
            events,
            {"controller_pose": {"enabled": True}},
        )
        host.start()
        runtimes["controller_pose"].on_stop = host.close

        host.close()

        self.assertEqual(events.count("stop:controller_pose"), 1)
        self.assertEqual(events.count("server_close"), 1)

    def test_builder_creates_programmatic_loopback_host_without_starting(self) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events)
        fake_server.server_address = ("127.0.0.1", 49152)
        config = {
            "server": {
                "host": "0.0.0.0",
                "port": 0,
                "https": True,
                "pairing": {
                    "enabled": True,
                    "token_env": "QUEST_DEV_TEST_PAIRING_TOKEN",
                },
            },
            "xr": {"reference_space": "local-floor"},
            "features": {"disabled": {"enabled": False, "client": {}}},
        }
        with (
            mock.patch.object(server, "load_config", return_value=config),
            mock.patch.dict(
                server.os.environ,
                {"QUEST_DEV_TEST_PAIRING_TOKEN": "314159"},
                clear=False,
            ),
            mock.patch.object(
                server, "QuestDevHTTPServer", return_value=fake_server
            ) as constructor,
        ):
            host = server.build_quest_dev_host(Path("unused.yaml"), force_http=True)

        self.assertEqual(constructor.call_args.args[0], ("127.0.0.1", 0))
        self.assertFalse(constructor.call_args.kwargs["use_https"])
        self.assertEqual(host.launch_info.base_url, "http://127.0.0.1:49152/")
        self.assertEqual(
            host.launch_info.pairing_url,
            "http://127.0.0.1:49152/#pair=314159",
        )
        self.assertNotIn("314159", repr(host.launch_info))
        self.assertEqual(events, [])
        host.close()
        self.assertEqual(events, ["server_close"])

    def test_builder_forwards_shared_clock_and_delays_custom_runtime_loader(
        self,
    ) -> None:
        events: list[str] = []
        fake_server = _FakeHostServer(events)
        fake_server.server_address = ("127.0.0.1", 49153)
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": 0,
                "https": False,
                "pairing": {"enabled": True},
            },
            "features": {"controller_pose": {"enabled": True}},
        }
        monotonic_ns_clock = mock.Mock(return_value=123)
        loaded: list[tuple[str, object, dict, threading.Event]] = []

        def runtime_loader(name, supplied_server, feature_config, stop_event):
            loaded.append((name, supplied_server, feature_config, stop_event))
            return _FakeFeatureRuntime(name, events)

        with (
            mock.patch.object(server, "load_config", return_value=config),
            mock.patch.object(
                server, "QuestDevHTTPServer", return_value=fake_server
            ) as constructor,
        ):
            host = server.build_quest_dev_host(
                Path("unused.yaml"),
                force_http=True,
                monotonic_ns_clock=monotonic_ns_clock,
                runtime_loader=runtime_loader,
            )

        self.assertIs(
            constructor.call_args.kwargs["monotonic_ns_clock"],
            monotonic_ns_clock,
        )
        self.assertEqual(loaded, [])
        host.start()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][0], "controller_pose")
        self.assertIs(loaded[0][1], fake_server)
        self.assertEqual(loaded[0][2], {"enabled": True})
        self.assertIs(loaded[0][3], host.stop_event)
        host.close()

    def test_builder_rejects_noncallables_before_server_construction(self) -> None:
        with mock.patch.object(server, "QuestDevHTTPServer") as constructor:
            with self.assertRaisesRegex(TypeError, "monotonic_ns_clock"):
                server.build_quest_dev_host(monotonic_ns_clock=None)  # type: ignore[arg-type]
            with self.assertRaisesRegex(TypeError, "runtime_loader"):
                server.build_quest_dev_host(runtime_loader=None)  # type: ignore[arg-type]
        constructor.assert_not_called()

    def test_cli_delegates_to_the_same_programmatic_host(self) -> None:
        config_path = Path("quest-config.yaml")
        fake_host = mock.Mock()
        fake_host.launch_info = self.launch_info()
        with (
            mock.patch.object(
                server,
                "parse_args",
                return_value=mock.Mock(
                    config=config_path,
                    http=True,
                    realman_left_ip=None,
                    realman_left_urdf=None,
                ),
            ),
            mock.patch.object(
                server,
                "build_quest_dev_host",
                return_value=fake_host,
            ) as build_host,
            mock.patch.object(server.signal, "signal") as install_signal,
            mock.patch("builtins.print"),
        ):
            server.main()

        build_host.assert_called_once_with(
            config_path,
            force_http=True,
            realman_left_ip=None,
            realman_left_urdf=None,
        )
        fake_host.serve_forever.assert_called_once_with()
        self.assertEqual(install_signal.call_count, 2)
        signal_handler = install_signal.call_args_list[0].args[1]
        signal_handler(server.signal.SIGINT, None)
        fake_host.request_shutdown.assert_called_once_with()


class AuthenticationTests(unittest.TestCase):
    @staticmethod
    def make_server() -> server.QuestDevHTTPServer:
        host = object.__new__(server.QuestDevHTTPServer)
        host.client_config = {}
        host._pairing_token = "314159"
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
        host.monotonic_ns_clock = time.monotonic_ns
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

    def test_post_auth_record_uses_authoritative_injected_receipt_clock(self) -> None:
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
        monotonic_ns_clock = mock.Mock(return_value=987654321)
        host.monotonic_ns_clock = monotonic_ns_clock
        records: list[dict] = []
        host.register_client_message_handler(
            "probe", lambda _, record: records.append(record)
        )
        payload = {
            "type": "probe",
            "client_instance_id": "browser-instance",
            "client_id": 999,
            "server_timestamp": -1.0,
            "server_monotonic_ns": 1,
        }

        with mock.patch.object(server.time, "time", return_value=123.5):
            self.assertTrue(host.process_client_message(1, payload))

        self.assertEqual(payload["client_id"], 999)
        self.assertEqual(payload["server_timestamp"], -1.0)
        self.assertEqual(payload["server_monotonic_ns"], 1)
        self.assertEqual(records[0]["client_id"], 1)
        self.assertEqual(records[0]["server_timestamp"], 123.5)
        self.assertEqual(records[0]["server_monotonic_ns"], 987654321)
        monotonic_ns_clock.assert_called_once_with()

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
            def poll(self) -> dict:
                return robot_tcp_message(
                    {
                        "rm65_left": {
                            "robot": "rm65_left",
                            "source": "realman_rm65_direct",
                            "fresh": True,
                            "control_ready": False,
                            "alignment_ready": False,
                        }
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
        runtime._run_feedback_loop(rate_hz=30)
        self.assertEqual(host.feature, "visualization")
        self.assertEqual(len(host.messages), 1)
        self.assertTrue(host.messages[0]["robots"]["rm65_left"]["fresh"])
        self.assertFalse(host.messages[0]["robots"]["rm65_left"]["control_ready"])


if __name__ == "__main__":
    unittest.main()

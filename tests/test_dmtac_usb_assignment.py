from __future__ import annotations

import json
import io
import tempfile
import threading
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
DMTAC_ROOT = ROOT / "hardware" / "DMTAC_gripper"
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from teleop_runtime.configs.dmtac_hardware import dmtac_backend_configs_from_assignment
from portable_dm_tac import DMTacDevice, DMTacMultiDevice, DeviceConfig, TacFrame  # noqa: E402
from portable_dmtac_gripper.device import TactileConfig, tactile_device_config  # noqa: E402
from portable_dmtac_gripper.cli import (  # noqa: E402
    _motion_confirmed_gripper_tokens,
    _print_scan_summary,
    build_parser,
)
from portable_dmtac_gripper import usb_assignment  # noqa: E402
from portable_dmtac_gripper.usb_assignment import (  # noqa: E402
    _extract_unexpected_slave_id,
    _probe_gripper,
    build_gripper_auto_assignment_tokens,
    build_assignment_from_tokens,
    compare_assignment_to_scan,
    load_assignment,
    merge_assignment,
    save_assignment,
    scan_usb_devices,
    tactile_ids_for_side,
)
from portable_dmtac_gripper.tactile_assignment_gui import (  # noqa: E402
    TactileAssignmentGuiConfig,
    TactileGuiDebugRecorder,
    _tactile_gui_record_payload,
    assign_device_to_tactile_slot,
    coarse_force_grid,
    decimate_force_for_gui,
    empty_tactile_slot_map,
    normal_grid_display_components,
    positive_normal_pressure_grid,
    default_tactile_debug_record_root,
    resolve_tactile_debug_record_root,
    tactile_items_for_gui,
    tactile_slot_tokens_from_assignment,
    xz_plane_vector_grid,
)
from portable_dmtac_gripper.force_visualizer import distributed_force_stats  # noqa: E402


def _scan():
    return {
        "schema_version": "dmtac_usb_scan.v1",
        "hostname": "unit-host",
        "grippers": [
            {
                "index": 0,
                "port": "/dev/serial/by-id/left-gripper",
                "serial_number": "LG",
                "hwid": "USB VID:PID=1A86:7523 SER=LG",
                "probe": {"reachable": True},
            },
            {
                "index": 1,
                "port": "/dev/serial/by-id/right-gripper",
                "serial_number": "RG",
                "hwid": "USB VID:PID=1A86:7523 SER=RG",
                "probe": {"reachable": True},
            },
        ],
        "tactile": [
            {"index": 0, "dev_id": "L_IN"},
            {"index": 1, "dev_id": "L_OUT"},
            {"index": 2, "dev_id": "R_IN"},
            {"index": 3, "dev_id": "R_OUT"},
        ],
    }


class DMTacUsbAssignmentTest(unittest.TestCase):
    def test_default_config_dir_lives_under_teleop_runtime_configs(self) -> None:
        self.assertEqual(
            usb_assignment.default_config_dir(),
            ROOT / "teleop_runtime" / "configs" / "hardware" / "dmtac",
        )

    def test_probe_gripper_retries_detected_slave_id(self) -> None:
        class FakeState:
            def to_dict(self):
                return {"joint_pos": 0.25}

        class FakeGripper:
            opened_ids: list[int] = []

            def __init__(self, *, port, gripper_id, baudrate, timeout):
                self.gripper_id = int(gripper_id)
                self.port = port
                self.baudrate = baudrate
                self.timeout = timeout
                self.opened_ids.append(self.gripper_id)

            def __enter__(self):
                if self.gripper_id == 1:
                    raise RuntimeError("Unexpected slave id in response: 32, expect 1")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_state(self):
                return FakeState()

        old_gripper = usb_assignment.InspireGripper
        usb_assignment.InspireGripper = FakeGripper
        try:
            result = _probe_gripper(
                "/dev/serial/by-id/gripper",
                gripper_id=1,
                baudrate=115200,
                timeout=0.03,
            )
        finally:
            usb_assignment.InspireGripper = old_gripper

        self.assertTrue(result["reachable"])
        self.assertEqual(result["detected_gripper_id"], 32)
        self.assertEqual(result["attempted_gripper_ids"], [1, 32])
        self.assertEqual(FakeGripper.opened_ids, [1, 32])
        self.assertEqual(_extract_unexpected_slave_id(result["id_probe_error"]), 32)

    def test_probe_gripper_scans_ids_after_timeout(self) -> None:
        class FakeState:
            def to_dict(self):
                return {"joint_pos": 0.4}

        class FakeGripper:
            opened_ids: list[int] = []

            def __init__(self, *, port, gripper_id, baudrate, timeout):
                self.gripper_id = int(gripper_id)
                self.opened_ids.append(self.gripper_id)

            def __enter__(self):
                if self.gripper_id != 46:
                    raise RuntimeError("Serial read timeout, expect 2 bytes, got 0:")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_state(self):
                return FakeState()

        old_gripper = usb_assignment.InspireGripper
        usb_assignment.InspireGripper = FakeGripper
        try:
            result = _probe_gripper(
                "/dev/serial/by-id/gripper",
                gripper_id=1,
                baudrate=115200,
                timeout=0.03,
            )
        finally:
            usb_assignment.InspireGripper = old_gripper

        self.assertTrue(result["reachable"])
        self.assertEqual(result["detected_gripper_id"], 46)
        self.assertEqual(result["attempted_gripper_ids"], [1, 32, 46])
        self.assertEqual(result["attempted_gripper_id_count"], 3)

    def test_probe_gripper_scans_baudrates(self) -> None:
        class FakeState:
            def to_dict(self):
                return {"joint_pos": 0.6}

        class FakeGripper:
            opened_settings: list[tuple[int, int]] = []

            def __init__(self, *, port, gripper_id, baudrate, timeout):
                self.gripper_id = int(gripper_id)
                self.baudrate = int(baudrate)
                self.opened_settings.append((self.baudrate, self.gripper_id))

            def __enter__(self):
                if (self.baudrate, self.gripper_id) != (57600, 32):
                    raise RuntimeError("Serial read timeout, expect 2 bytes, got 0:")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_state(self):
                return FakeState()

        old_gripper = usb_assignment.InspireGripper
        usb_assignment.InspireGripper = FakeGripper
        try:
            result = _probe_gripper(
                "/dev/serial/by-id/gripper",
                gripper_id=1,
                baudrate=115200,
                timeout=0.03,
                scan_baudrates=(115200, 57600, 19200),
            )
        finally:
            usb_assignment.InspireGripper = old_gripper

        self.assertTrue(result["reachable"])
        self.assertEqual(result["detected_gripper_id"], 32)
        self.assertEqual(result["detected_baudrate"], 57600)
        self.assertEqual(result["attempted_settings"][-1], {"baudrate": 57600, "gripper_id": 32})

    def test_probe_gripper_stops_on_permission_denied(self) -> None:
        class FakeGripper:
            opened_settings: list[tuple[int, int]] = []

            def __init__(self, *, port, gripper_id, baudrate, timeout):
                self.opened_settings.append((int(baudrate), int(gripper_id)))

            def __enter__(self):
                raise RuntimeError(
                    "SerialException: [Errno 13] could not open port /dev/ttyUSB0: "
                    "[Errno 13] Permission denied: '/dev/ttyUSB0'"
                )

            def __exit__(self, exc_type, exc, tb):
                return False

        old_gripper = usb_assignment.InspireGripper
        usb_assignment.InspireGripper = FakeGripper
        try:
            result = _probe_gripper(
                "/dev/ttyUSB0",
                gripper_id=1,
                baudrate=115200,
                timeout=0.03,
                scan_baudrates=(115200, 57600, 19200),
            )
        finally:
            usb_assignment.InspireGripper = old_gripper

        self.assertFalse(result["reachable"])
        self.assertEqual(result["attempted_setting_count"], 1)
        self.assertEqual(result["probe_stopped_reason"], "permission_denied")
        self.assertEqual(FakeGripper.opened_settings, [(115200, 1)])

    def test_probe_all_serial_skips_system_ttys_by_default(self) -> None:
        calls: list[str] = []

        def fake_ports():
            return [
                {
                    "device": "/dev/ttyS0",
                    "description": "n/a",
                    "hwid": "n/a",
                    "manufacturer": None,
                    "product": None,
                },
                {
                    "device": "/dev/ttyUSB0",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523",
                    "manufacturer": None,
                    "product": None,
                },
            ]

        def fake_probe(port: str, **kwargs):
            calls.append(port)
            return {"reachable": False, "error": "timeout"}

        old_ports = usb_assignment.list_serial_devices
        old_probe = usb_assignment._probe_gripper
        old_discover = usb_assignment.discover_devices
        old_candidates = usb_assignment.discover_device_candidates
        usb_assignment.list_serial_devices = fake_ports
        usb_assignment._probe_gripper = fake_probe
        usb_assignment.discover_devices = lambda: []
        usb_assignment.discover_device_candidates = lambda: []
        try:
            scan = scan_usb_devices(probe_all_serial=True)
        finally:
            usb_assignment.list_serial_devices = old_ports
            usb_assignment._probe_gripper = old_probe
            usb_assignment.discover_devices = old_discover
            usb_assignment.discover_device_candidates = old_candidates

        self.assertEqual(calls, ["/dev/ttyUSB0"])
        self.assertFalse(scan["grippers"][0]["probe_attempted"])
        self.assertTrue(scan["grippers"][1]["probe_attempted"])

    def test_probe_all_serial_skips_q6_usb_single_serial(self) -> None:
        calls: list[str] = []

        def fake_ports():
            return [
                {
                    "device": "/dev/ttyACM0",
                    "description": "USB Single Serial",
                    "hwid": "USB VID:PID=1A86:55D3 SER=5B7A137621",
                    "manufacturer": None,
                    "product": "USB Single Serial",
                    "serial_number": "5B7A137621",
                },
                {
                    "device": "/dev/ttyUSB0",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523 LOCATION=5-2.4.4",
                    "manufacturer": None,
                    "product": "USB Serial",
                    "serial_number": None,
                },
            ]

        def fake_probe(port: str, **kwargs):
            calls.append(port)
            return {"reachable": True, "error": None}

        with (
            patch.object(usb_assignment, "list_serial_devices", fake_ports),
            patch.object(usb_assignment, "_probe_gripper", fake_probe),
            patch.object(usb_assignment, "discover_devices", lambda: []),
            patch.object(usb_assignment, "discover_device_candidates", lambda: []),
        ):
            scan = scan_usb_devices(probe_all_serial=True)

        self.assertEqual(calls, ["/dev/ttyUSB0"])
        self.assertTrue(scan["grippers"][0]["known_q6_serial"])
        self.assertFalse(scan["grippers"][0]["probe_attempted"])
        self.assertTrue(scan["grippers"][1]["probe_attempted"])

    def test_scan_progress_callback_reports_probe_start_and_done(self) -> None:
        events: list[dict[str, object]] = []

        def fake_ports():
            return [
                {
                    "device": "/dev/ttyUSB0",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523",
                    "manufacturer": None,
                    "product": "USB Serial",
                },
            ]

        def fake_probe(port: str, **kwargs):
            return {"reachable": False, "error": "timeout", "probe_elapsed_s": 0.03, "attempted_setting_count": 1}

        with (
            patch.object(usb_assignment, "list_serial_devices", fake_ports),
            patch.object(usb_assignment, "_probe_gripper", fake_probe),
            patch.object(usb_assignment, "discover_devices", lambda: []),
            patch.object(usb_assignment, "discover_device_candidates", lambda: []),
        ):
            scan_usb_devices(progress_callback=lambda payload: events.append(dict(payload)))

        self.assertEqual([event["event"] for event in events], [
            "serial_scan_ready",
            "gripper_probe_start",
            "gripper_probe_done",
        ])
        self.assertEqual(events[0]["probe_count"], 1)
        self.assertEqual(events[1]["port"], "/dev/ttyUSB0")

    def test_gripper_probe_stops_after_two_reachable_devices(self) -> None:
        calls: list[str] = []
        events: list[dict[str, object]] = []

        def fake_ports():
            return [
                {
                    "device": f"/dev/ttyUSB{index}",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523",
                    "manufacturer": None,
                    "product": "USB Serial",
                }
                for index in range(4)
            ]

        def fake_probe(port: str, **kwargs):
            calls.append(port)
            return {"reachable": True, "error": None}

        with (
            patch.object(usb_assignment, "list_serial_devices", fake_ports),
            patch.object(usb_assignment, "_probe_gripper", fake_probe),
            patch.object(usb_assignment, "discover_devices", lambda: []),
            patch.object(usb_assignment, "discover_device_candidates", lambda: []),
        ):
            scan = scan_usb_devices(
                progress_callback=lambda payload: events.append(dict(payload))
            )

        self.assertEqual(calls, ["/dev/ttyUSB0", "/dev/ttyUSB1"])
        self.assertEqual(scan["gripper_probe_limit"], 2)
        self.assertEqual(scan["gripper_probe_attempted_count"], 2)
        self.assertEqual(scan["gripper_probe_reachable_count"], 2)
        self.assertTrue(scan["gripper_probe_stopped_early"])
        self.assertFalse(scan["grippers"][2]["probe_attempted"])
        self.assertEqual(
            scan["grippers"][2]["probe_skipped_reason"],
            "reachable_gripper_limit",
        )
        self.assertEqual(events[-1]["event"], "gripper_probe_early_stop")

    def test_gripper_probe_limit_counts_only_reachable_devices(self) -> None:
        calls: list[str] = []

        def fake_ports():
            return [
                {
                    "device": f"/dev/ttyUSB{index}",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523",
                    "manufacturer": None,
                    "product": "USB Serial",
                }
                for index in range(4)
            ]

        def fake_probe(port: str, **kwargs):
            calls.append(port)
            return {"reachable": port != "/dev/ttyUSB0", "error": None}

        with (
            patch.object(usb_assignment, "list_serial_devices", fake_ports),
            patch.object(usb_assignment, "_probe_gripper", fake_probe),
            patch.object(usb_assignment, "discover_devices", lambda: []),
            patch.object(usb_assignment, "discover_device_candidates", lambda: []),
        ):
            scan = scan_usb_devices()

        self.assertEqual(calls, ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"])
        self.assertEqual(scan["gripper_probe_reachable_count"], 2)
        self.assertFalse(scan["grippers"][3]["probe_attempted"])

    def test_gripper_assignment_uses_detected_modbus_id(self) -> None:
        scan = _scan()
        scan["grippers"][0]["probe"] = {
            "reachable": True,
            "requested_gripper_id": 1,
            "detected_gripper_id": 46,
            "detected_baudrate": 57600,
            "id_autodetected": True,
        }

        assignment = build_assignment_from_tokens(scan, left_gripper="0")

        self.assertEqual(assignment["assignments"]["left"]["gripper"]["gripper_id"], 46)
        self.assertEqual(assignment["assignments"]["left"]["gripper"]["baudrate"], 57600)

    def test_build_save_and_load_latest_assignment(self) -> None:
        assignment = build_assignment_from_tokens(
            _scan(),
            left_gripper="0",
            left_inner="L_IN",
            left_outer="L_OUT",
            right_gripper="/dev/serial/by-id/right-gripper",
            right_inner="2",
            right_outer="3",
        )
        self.assertEqual(tactile_ids_for_side(assignment, "left"), ("L_IN", "L_OUT"))

        with tempfile.TemporaryDirectory() as tmp:
            path = save_assignment(assignment, config_dir=tmp)
            loaded = load_assignment("latest", config_dir=tmp)

        self.assertTrue(path.name.endswith("_dmtac_usb_assignment.json"))
        self.assertEqual(loaded["hostname"], assignment["hostname"])
        self.assertEqual(tactile_ids_for_side(loaded, "right"), ("R_IN", "R_OUT"))

    def test_latest_falls_back_to_legacy_config_dir_but_saves_new_default(self) -> None:
        assignment = build_assignment_from_tokens(
            _scan(),
            left_gripper="0",
            left_inner="0",
            left_outer="1",
            right_gripper="1",
            right_inner="2",
            right_outer="3",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_dir = root / "teleop_runtime" / "configs" / "hardware" / "dmtac"
            legacy_dir = root / "teleop_records" / "dmtac_hardware_configs"
            save_assignment(assignment, config_dir=legacy_dir)

            with (
                patch.object(usb_assignment, "default_config_dir", return_value=new_dir),
                patch.object(usb_assignment, "legacy_config_dirs", return_value=(legacy_dir,)),
            ):
                loaded = load_assignment("latest")
                saved = save_assignment(assignment)

        self.assertEqual(tactile_ids_for_side(loaded, "left"), ("L_IN", "L_OUT"))
        self.assertEqual(saved.parent, new_dir)

    def test_auto_gripper_assignment_tokens_match_q6_style_defaults(self) -> None:
        tokens = build_gripper_auto_assignment_tokens(_scan())
        swapped = build_gripper_auto_assignment_tokens(_scan(), swap=True)
        single_scan = _scan()
        single_scan["grippers"] = single_scan["grippers"][:1]

        self.assertEqual(tokens, {"left": "0", "right": "1"})
        self.assertEqual(swapped, {"left": "1", "right": "0"})
        self.assertEqual(
            build_gripper_auto_assignment_tokens(single_scan, single_side="left"),
            {"left": "0", "right": None},
        )

    def test_motion_confirmed_gripper_assignment_jogs_before_side_selection(self) -> None:
        class Args:
            gripper_id = 1
            baudrate = 115200
            timeout = 0.03
            gripper_jog_amplitude = 0.08
            gripper_jog_speed = 0.06
            gripper_jog_torque = 0.08
            json = False

        answers = iter(["", "left", "", "right", ""])
        calls: list[tuple[str, float, float, float]] = []

        def fake_motion(port: str, **kwargs):
            calls.append(
                (
                    port,
                    float(kwargs["amplitude"]),
                    float(kwargs["speed"]),
                    float(kwargs["torque"]),
                )
            )
            return {"port": port, "ok": True}

        tokens = _motion_confirmed_gripper_tokens(
            _scan(),
            Args(),
            motion_func=fake_motion,
            input_func=lambda _prompt: next(answers),
        )

        self.assertEqual(tokens, {"left": "0", "right": "1"})
        self.assertEqual(
            calls,
            [
                ("/dev/serial/by-id/left-gripper", 0.08, 0.06, 0.08),
                ("/dev/serial/by-id/right-gripper", 0.08, 0.06, 0.08),
            ],
        )

    def test_scan_summary_hides_ttys_and_shows_probe_errors(self) -> None:
        scan = _scan()
        scan["grippers"] = [
            {
                "index": 0,
                "port": "/dev/ttyS0",
                "preferred_port": "/dev/ttyS0",
                "description": "n/a",
                "hwid": "n/a",
                "probe_attempted": False,
                "probe": None,
            },
            {
                "index": 1,
                "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_ABC-if00",
                "preferred_port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_ABC-if00",
                "description": "USB Single Serial",
                "hwid": "USB VID:PID=1A86:55D3 SER=ABC",
                "probe_attempted": True,
                "probable_gripper_serial": True,
                "probe": {"reachable": False, "error": "GripperError: timeout"},
            },
        ]
        out = io.StringIO()

        with redirect_stdout(out):
            _print_scan_summary(scan)
        text = out.getvalue()

        self.assertIn("Reachable grippers:\n  none", text)
        self.assertIn("probe_failed", text)
        self.assertIn("probe_failed USB Single Serial", text)
        self.assertIn("GripperError: timeout", text)
        self.assertIn("system ttyS ports hidden", text)
        self.assertNotIn("[0] /dev/ttyS0 unverified", text)
        self.assertIn("[0] L_IN", text)
        self.assertNotIn("{'device_index'", text)

    def test_gripper_assignment_prefers_tty_endpoint_and_remaps(self) -> None:
        scan = _scan()
        scan["grippers"][0]["port"] = "/dev/ttyUSB0"
        scan["grippers"][0]["preferred_port"] = "/dev/serial/by-id/left-gripper"
        scan["grippers"][0]["port_aliases"] = ["/dev/ttyUSB0", "/dev/serial/by-id/left-gripper"]
        assignment = build_assignment_from_tokens(scan, left_gripper="0")

        self.assertEqual(
            assignment["assignments"]["left"]["gripper"]["port"],
            "/dev/ttyUSB0",
        )

        current = _scan()
        current["grippers"][0]["port"] = "/dev/ttyUSB7"
        current["grippers"][0]["preferred_port"] = "/dev/serial/by-id/left-gripper"
        current["grippers"][0]["port_aliases"] = ["/dev/ttyUSB7", "/dev/serial/by-id/left-gripper"]
        comparison = compare_assignment_to_scan(assignment, current)

        self.assertFalse(comparison["missing"])
        self.assertEqual(
            comparison["effective_assignment"]["assignments"]["left"]["gripper"]["port"],
            "/dev/ttyUSB7",
        )

    def test_gripper_preferred_port_uses_tty_before_ch340_aliases(self) -> None:
        preferred = usb_assignment._preferred_port_from_aliases(
            [
                "/dev/ttyUSB0",
                "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.4:1.0-port0",
            ],
            fallback="/dev/ttyUSB0",
            port_info={
                "vid": 6790,
                "pid": 29987,
                "serial_number": None,
            },
        )

        self.assertEqual(
            preferred,
            "/dev/ttyUSB0",
        )

    def test_legacy_by_path_assignment_normalizes_to_recorded_tty(self) -> None:
        scan = _scan()
        scan["grippers"][0].update(
            {
                "port": "/dev/ttyUSB0",
                "device": "/dev/ttyUSB0",
                "preferred_port": "/dev/serial/by-path/old-left",
                "port_aliases": ["/dev/ttyUSB0", "/dev/serial/by-path/old-left"],
                "location": "old-left",
            }
        )
        assignment = build_assignment_from_tokens(scan, left_gripper="0")
        assignment["assignments"]["left"]["gripper"]["port"] = (
            "/dev/serial/by-path/old-left"
        )

        normalized = usb_assignment.normalize_assignment(assignment)

        self.assertEqual(
            normalized["assignments"]["left"]["gripper"]["port"],
            "/dev/ttyUSB0",
        )

    def test_power_cycle_path_change_keeps_tty_side_binding(self) -> None:
        scan = _scan()
        scan["grippers"][0].update(
            {
                "port": "/dev/ttyUSB0",
                "device": "/dev/ttyUSB0",
                "preferred_port": "/dev/serial/by-path/before-power-cycle",
                "port_aliases": [
                    "/dev/ttyUSB0",
                    "/dev/serial/by-path/before-power-cycle",
                ],
                "hwid": "USB VID:PID=1A86:7523 LOCATION=before-power-cycle",
                "location": "before-power-cycle",
            }
        )
        assignment = build_assignment_from_tokens(scan, left_gripper="0")

        current = _scan()
        current["grippers"][0].update(
            {
                "port": "/dev/ttyUSB0",
                "device": "/dev/ttyUSB0",
                "preferred_port": "/dev/serial/by-path/after-power-cycle",
                "port_aliases": [
                    "/dev/ttyUSB0",
                    "/dev/serial/by-path/after-power-cycle",
                ],
                "hwid": "USB VID:PID=1A86:7523 LOCATION=after-power-cycle",
                "location": "after-power-cycle",
            }
        )
        current["grippers"] = current["grippers"][:1]
        current["tactile"] = []

        comparison = compare_assignment_to_scan(assignment, current)

        self.assertFalse(comparison["missing"])
        self.assertFalse(comparison["extra"])
        self.assertEqual(
            comparison["effective_assignment"]["assignments"]["left"]["gripper"]["port"],
            "/dev/ttyUSB0",
        )

    def test_compare_ignores_nonunique_ch340_by_id_collision(self) -> None:
        assignment = build_assignment_from_tokens(_scan(), left_gripper="0", right_gripper="1")
        generic = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
        assignment["assignments"]["left"]["gripper"].update(
            {
                "port": generic,
                "identity": {"location": "3-7.4", "hwid": "USB VID:PID=1A86:7523 LOCATION=3-7.4"},
                "port_aliases": [
                    "/dev/ttyUSB1",
                    generic,
                    "/dev/serial/by-path/pci-0000:00:14.0-usb-0:7.4:1.0-port0",
                ],
            }
        )
        assignment["assignments"]["right"]["gripper"].update(
            {
                "port": generic,
                "identity": {"location": "3-8.4", "hwid": "USB VID:PID=1A86:7523 LOCATION=3-8.4"},
                "port_aliases": [
                    "/dev/ttyUSB0",
                    generic,
                    "/dev/serial/by-path/pci-0000:00:14.0-usb-0:8.4:1.0-port0",
                ],
            }
        )
        current = {
            "grippers": [
                {
                    "index": 0,
                    "port": "/dev/ttyUSB1",
                    "device": "/dev/ttyUSB1",
                    "preferred_port": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:8.4:1.0-port0",
                    "port_aliases": [
                        "/dev/ttyUSB1",
                        generic,
                        "/dev/serial/by-path/pci-0000:00:14.0-usb-0:8.4:1.0-port0",
                    ],
                    "identity": {},
                    "hwid": "USB VID:PID=1A86:7523 LOCATION=3-8.4",
                    "location": "3-8.4",
                    "vid": 6790,
                    "pid": 29987,
                    "serial_number": None,
                    "probe": {"reachable": True},
                },
                {
                    "index": 1,
                    "port": "/dev/ttyUSB0",
                    "device": "/dev/ttyUSB0",
                    "preferred_port": "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.4:1.0-port0",
                    "port_aliases": [
                        "/dev/ttyUSB0",
                        "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.4:1.0-port0",
                    ],
                    "hwid": "USB VID:PID=1A86:7523 LOCATION=3-6.4",
                    "location": "3-6.4",
                    "vid": 6790,
                    "pid": 29987,
                    "serial_number": None,
                    "probe": {"reachable": True},
                },
            ],
            "tactile": [],
        }

        comparison = compare_assignment_to_scan(assignment, current)

        self.assertEqual([(item["kind"], item["side"]) for item in comparison["missing"]], [("gripper", "left")])
        self.assertEqual([(item["kind"], item["side"]) for item in comparison["matched"]], [("gripper", "right")])
        self.assertEqual(comparison["extra"][0]["current"]["port"], "/dev/ttyUSB0")

    def test_merge_assignment_keeps_existing_gripper_when_labeling_tactile_only(self) -> None:
        gripper_assignment = build_assignment_from_tokens(_scan(), left_gripper="0", right_gripper="1")
        tactile_assignment = build_assignment_from_tokens(_scan(), left_inner="L_IN", right_outer="R_OUT")

        merged = merge_assignment(gripper_assignment, tactile_assignment)

        self.assertEqual(
            merged["assignments"]["left"]["gripper"]["port"],
            "/dev/serial/by-id/left-gripper",
        )
        self.assertEqual(
            merged["assignments"]["right"]["gripper"]["port"],
            "/dev/serial/by-id/right-gripper",
        )
        self.assertEqual(tactile_ids_for_side(merged, "left"), ("L_IN",))
        self.assertEqual(tactile_ids_for_side(merged, "right"), ("R_OUT",))

    def test_compare_reports_missing_and_extra_devices(self) -> None:
        assignment = build_assignment_from_tokens(
            _scan(),
            left_gripper="0",
            left_inner="L_IN",
        )
        current = _scan()
        current["grippers"] = current["grippers"][1:]
        current["tactile"].append({"index": 4, "dev_id": "EXTRA"})

        comparison = compare_assignment_to_scan(assignment, current)

        self.assertEqual(comparison["missing"][0]["kind"], "gripper")
        self.assertTrue(any(item["kind"] == "tactile" for item in comparison["extra"]))

    def test_backend_configs_allow_gripper_only_and_tactile_only_sides(self) -> None:
        assignment = build_assignment_from_tokens(
            _scan(),
            left_gripper="0",
            right_inner="R_IN",
            right_outer="R_OUT",
        )

        configs = dmtac_backend_configs_from_assignment(assignment)

        self.assertTrue(configs["left"].connect_hardware)
        self.assertFalse(configs["left"].connect_tactile)
        self.assertFalse(configs["right"].connect_hardware)
        self.assertTrue(configs["right"].connect_tactile)
        self.assertEqual(configs["right"].tactile_ids, ("R_IN", "R_OUT"))

    def test_tactile_assignment_gui_helpers_keep_unique_slot_mapping(self) -> None:
        slots = empty_tactile_slot_map()

        assign_device_to_tactile_slot(slots, "L_IN", side="left", slot="inner")
        assign_device_to_tactile_slot(slots, "L_IN", side="right", slot="outer")
        assign_device_to_tactile_slot(slots, "R_IN", side="left", slot="inner")
        tokens = tactile_slot_tokens_from_assignment(slots)

        self.assertIsNone(tokens["left_outer"])
        self.assertEqual(tokens["left_inner"], "R_IN")
        self.assertEqual(tokens["right_outer"], "L_IN")

    def test_tactile_items_for_gui_accepts_tokens_and_caps_to_four(self) -> None:
        scan = _scan()
        scan["tactile"].append({"index": 4, "dev_id": "EXTRA"})

        selected = tactile_items_for_gui(scan, max_devices=4)
        token_selected = tactile_items_for_gui(scan, tokens=("3", "L_IN"), max_devices=4)

        self.assertEqual([item["dev_id"] for item in selected], ["L_IN", "L_OUT", "R_IN", "R_OUT"])
        self.assertEqual([item["dev_id"] for item in token_selected], ["R_OUT", "L_IN"])

    def test_tactile_gui_force_decimation_reduces_render_work(self) -> None:
        import numpy as np

        force = np.zeros((288, 384, 3), dtype=np.float32)
        decimated = decimate_force_for_gui(force, decimation=2)

        self.assertEqual(decimated.shape, (144, 192, 3))

    def test_tactile_gui_default_grid_is_twenty_by_thirty(self) -> None:
        import numpy as np

        cfg = TactileAssignmentGuiConfig()
        force = np.zeros((288, 384, 3), dtype=np.float32)
        grid = coarse_force_grid(force)

        self.assertEqual(cfg.normal_grid_rows, 20)
        self.assertEqual(cfg.normal_grid_cols, 30)
        self.assertEqual(cfg.normal_grid_vector_stride, 2)
        self.assertFalse(cfg.record_grid)
        self.assertAlmostEqual(cfg.normal_pressure_deadband, 5e-6)
        self.assertAlmostEqual(cfg.normal_vector_deadband, 2e-5)
        self.assertEqual(grid.shape, (20, 30, 3))

    def test_dmtac_multi_device_parallel_read_overlaps_device_getters(self) -> None:
        parallel_phase = False
        sequential_entries: list[str] = []
        parallel_entries: list[str] = []
        parallel_threads: set[int] = set()
        entry_lock = threading.Lock()
        all_parallel_readers_entered = threading.Event()

        class FakeDevice:
            def __init__(self, dev_id: str) -> None:
                self.dev_id = dev_id

            def read(self, **_kwargs):
                if not parallel_phase:
                    sequential_entries.append(self.dev_id)
                    return TacFrame(device_id=self.dev_id, timestamp_ns=0, status=0, fid=1)
                with entry_lock:
                    parallel_entries.append(self.dev_id)
                    parallel_threads.add(threading.get_ident())
                    if len(parallel_entries) == 3:
                        all_parallel_readers_entered.set()
                if not all_parallel_readers_entered.wait(timeout=5.0):
                    raise AssertionError(
                        "parallel readers did not overlap before one returned"
                    )
                return TacFrame(device_id=self.dev_id, timestamp_ns=0, status=0, fid=1)

            def close(self) -> None:
                return None

        multi = DMTacMultiDevice([DeviceConfig("a"), DeviceConfig("b"), DeviceConfig("c")])
        multi.devices = [FakeDevice("a"), FakeDevice("b"), FakeDevice("c")]
        try:
            sequential = multi.read_all(parallel=False)
            parallel_phase = True
            parallel = multi.read_all(parallel=True)
        finally:
            multi.close()

        self.assertEqual(sequential_entries, ["a", "b", "c"])
        self.assertCountEqual(parallel_entries, ("a", "b", "c"))
        self.assertEqual(len(parallel_threads), 3)
        self.assertEqual(tuple(sequential), ("a", "b", "c"))
        self.assertEqual(tuple(parallel), ("a", "b", "c"))

    def test_dmtac_device_converts_distributed_force_to_float32_array(self) -> None:
        import numpy as np

        class FakeSensor:
            def getDevStatus(self) -> int:
                return 0

            def getRawImg(self):
                return (1, None)

            def getInferImg(self):
                return (1, None)

            def getDeformation2D(self):
                return (1, None)

            def getShear(self):
                return (1, None)

            def getDepth(self):
                return (1, None)

            def getForce(self):
                return (1, None)

            def getDistributeForce(self):
                return (7, np.ones((2, 2, 3), dtype=np.float64))

        dev = DMTacDevice(
            DeviceConfig(
                dev_id="fake",
                enable_raw=False,
                enable_deformation=False,
                enable_shear=False,
                enable_depth=False,
                enable_force=False,
                include_infer=False,
                include_distributed_force=True,
                include_contact_area=False,
                include_base_frame=False,
                include_events=False,
                read_raw=False,
                read_deformation=False,
                read_shear=False,
                read_depth=False,
                read_force=False,
            )
        )
        dev.sensor = FakeSensor()

        frame = dev.read(wait=False)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertIsInstance(frame.distributed_force, np.ndarray)
        self.assertEqual(frame.distributed_force.dtype, np.float32)
        self.assertEqual(frame.distributed_force.shape, (2, 2, 3))
        self.assertEqual(frame.fid, 7)

    def test_tactile_compute_profile_dist_only_keeps_force_pipeline_minimal(self) -> None:
        cfg = tactile_device_config(
            "L_IN",
            TactileConfig(fields=("distributed_force",), compute_profile="dist-only"),
        )

        self.assertFalse(cfg.enable_raw)
        self.assertFalse(cfg.enable_deformation)
        self.assertFalse(cfg.enable_shear)
        self.assertFalse(cfg.enable_depth)
        self.assertTrue(cfg.enable_force)
        self.assertTrue(cfg.include_distributed_force)
        self.assertFalse(cfg.should_read_force())

    def test_tactile_default_fields_read_raw_without_force_pipeline(self) -> None:
        tactile = TactileConfig()
        cfg = tactile_device_config("L_IN", tactile)

        self.assertFalse(tactile.denoise_distributed_force)
        self.assertFalse(tactile.denoise_config.enabled)
        self.assertTrue(cfg.enable_raw)
        self.assertTrue(cfg.should_read_raw())
        self.assertFalse(cfg.enable_force)
        self.assertFalse(cfg.include_distributed_force)
        self.assertFalse(cfg.should_read_force())

    def test_tactile_compute_profile_legacy_matches_old_distributed_force_setup(self) -> None:
        cfg = tactile_device_config(
            "L_IN",
            TactileConfig(fields=("distributed_force",), compute_profile="legacy"),
        )

        self.assertTrue(cfg.enable_deformation)
        self.assertTrue(cfg.enable_shear)
        self.assertTrue(cfg.enable_depth)
        self.assertTrue(cfg.enable_force)
        self.assertTrue(cfg.include_distributed_force)
        self.assertFalse(cfg.should_read_shear())
        self.assertFalse(cfg.should_read_depth())

    def test_tactile_gui_coarse_grid_preserves_three_force_channels(self) -> None:
        import numpy as np

        force = np.zeros((4, 6, 3), dtype=np.float32)
        force[..., 0] = np.arange(4, dtype=np.float32)[:, None]
        force[..., 1] = np.arange(6, dtype=np.float32)[None, :]
        force[..., 2] = 2.0

        grid = coarse_force_grid(force, rows=2, cols=3)

        self.assertEqual(grid.shape, (2, 3, 3))
        self.assertAlmostEqual(float(grid[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(grid[0, 0, 1]), 0.5)
        self.assertAlmostEqual(float(grid[1, 2, 0]), 2.5)
        self.assertAlmostEqual(float(grid[1, 2, 1]), 4.5)
        self.assertAlmostEqual(float(grid[1, 2, 2]), 2.0)

    def test_tactile_gui_positive_normal_pressure_clips_negative_values(self) -> None:
        import numpy as np

        grid = np.zeros((2, 3, 3), dtype=np.float32)
        grid[..., 1] = np.array([[-1.0, 0.5, 2.0], [3.0, -4.0, 0.0]], dtype=np.float32)

        pressure = positive_normal_pressure_grid(grid, normal_sign=1.0)
        inverted = positive_normal_pressure_grid(grid, normal_sign=-1.0)

        self.assertEqual(pressure.tolist(), [[0.0, 0.5, 2.0], [3.0, 0.0, 0.0]])
        self.assertEqual(inverted.tolist(), [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    def test_tactile_gui_xz_vector_grid_uses_x_and_z_channels(self) -> None:
        import numpy as np

        grid = np.zeros((2, 3, 3), dtype=np.float32)
        grid[..., 0] = 1.5
        grid[..., 1] = 9.0
        grid[..., 2] = -2.5

        xz = xz_plane_vector_grid(grid)

        self.assertEqual(xz.shape, (2, 3, 2))
        self.assertEqual(xz[0, 0].tolist(), [1.5, -2.5])

    def test_tactile_gui_display_deadband_suppresses_no_load_noise(self) -> None:
        import numpy as np

        grid = np.zeros((2, 2, 3), dtype=np.float32)
        grid[0, 0, :] = [1e-6, 1e-6, 1e-6]
        grid[1, 1, :] = [8e-6, 7e-6, 0.0]

        pressure, xz = normal_grid_display_components(
            grid,
            normal_sign=1.0,
            pressure_deadband=5e-6,
            vector_deadband=5e-6,
        )

        self.assertEqual(float(pressure[0, 0]), 0.0)
        self.assertAlmostEqual(float(pressure[1, 1]), 7e-6)
        self.assertEqual(xz[0, 0].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(xz[1, 1, 0]), 8e-6)
        self.assertAlmostEqual(float(xz[1, 1, 1]), 0.0)

    def test_distributed_force_stats_include_y_normal_and_xz_plane(self) -> None:
        import numpy as np

        force = np.zeros((1, 2, 3), dtype=np.float32)
        force[0, 0, :] = [3.0, 4.0, 0.0]
        force[0, 1, :] = [0.0, -5.0, 12.0]
        prev = np.zeros_like(force)

        stats = distributed_force_stats(force, prev)

        self.assertAlmostEqual(float(stats["xz_max"]), 12.0)
        self.assertAlmostEqual(float(stats["y_abs_max"]), 5.0)
        self.assertAlmostEqual(float(stats["dxz_max"]), 12.0)
        self.assertAlmostEqual(float(stats["dy_abs_max"]), 5.0)

    def test_tactile_gui_record_payload_includes_grid_and_timings(self) -> None:
        import numpy as np

        class Frame:
            timestamp_ns = 123
            status = 0
            fid = 7
            fids = {"distributed_force": 7}
            errors = {}
            timings_ms = {"distributed_force": 3.5, "force": 0.2}
            distributed_force = np.ones((40, 60, 3), dtype=np.float32)

        payload = _tactile_gui_record_payload(
            loop_index=1,
            items=[{"dev_id": "L_IN"}],
            frames_by_id={"L_IN": Frame()},
            fid_age_by_id={"L_IN": 0.012},
            slots={"left": {"inner": "L_IN", "outer": None}, "right": {"inner": None, "outer": None}},
            fps=30.0,
            read_ms=4.0,
            render_ms=2.0,
            display_ms=1.5,
            loop_ms=8.0,
            force_grid_rows=20,
            force_grid_cols=30,
            normal_sign=1.0,
            include_grid=True,
        )

        dev = payload["devices"]["L_IN"]
        self.assertEqual(payload["display_ms"], 1.5)
        self.assertEqual(payload["loop_ms"], 8.0)
        self.assertEqual(dev["grid_shape"], (20, 30, 3))
        self.assertEqual(dev["frame"]["timings_ms"]["distributed_force"], 3.5)
        self.assertEqual(dev["force_grid"].shape, (20, 30, 3))
        self.assertEqual(dev["pressure_y_positive_grid"].shape, (20, 30))

    def test_tactile_gui_debug_recorder_writes_metadata_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TactileGuiDebugRecorder(tmp, metadata={"dev_ids": ["L_IN"]})
            recorder.write_frame({"loop_index": 1, "devices": {"L_IN": {"fid_age_ms": 0.0}}})
            recorder.close()

            metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
            lines = recorder.frames_path.read_text(encoding="utf-8").strip().splitlines()
            frame = json.loads(lines[0])

        self.assertEqual(metadata["schema_version"], "dmtac_tactile_gui_debug.v1")
        self.assertEqual(metadata["dev_ids"], ["L_IN"])
        self.assertEqual(metadata["artifact_storage"]["selection_source"], "explicit")
        self.assertEqual(frame["loop_index"], 1)

    def test_tactile_gui_debug_layout_is_legacy_default_and_canonical_opt_in(
        self,
    ) -> None:
        with patch(
            "portable_dmtac_gripper.tactile_assignment_gui.canonical_tactile_debug_record_root",
            side_effect=AssertionError("legacy resolution must not load canonical paths"),
        ):
            legacy = resolve_tactile_debug_record_root(None)
        self.assertEqual(legacy.path, default_tactile_debug_record_root())
        self.assertEqual(legacy.source, "legacy")

        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "canonical"
            with patch(
                "portable_dmtac_gripper.tactile_assignment_gui.canonical_tactile_debug_record_root",
                return_value=canonical,
            ):
                selected = resolve_tactile_debug_record_root(
                    None,
                    artifact_layout="runtime-artifacts-v1",
                )
            self.assertEqual(selected.path, canonical)
            self.assertEqual(selected.source, "canonical")
            self.assertFalse(canonical.exists())

            explicit = Path(tmp) / "exact"
            with patch(
                "portable_dmtac_gripper.tactile_assignment_gui.canonical_tactile_debug_record_root",
                side_effect=AssertionError("explicit paths must take precedence"),
            ):
                selected = resolve_tactile_debug_record_root(
                    explicit,
                    artifact_layout="runtime-artifacts-v1",
                )
            self.assertEqual(selected.path, explicit)
            self.assertEqual(selected.source, "explicit")

        with self.assertRaisesRegex(ValueError, "unsupported artifact layout"):
            resolve_tactile_debug_record_root(None, artifact_layout="unknown")

        args = build_parser().parse_args(
            [
                "configure-usb",
                "--tactile-gui-artifact-layout",
                "runtime-artifacts-v1",
            ]
        )
        self.assertEqual(args.tactile_gui_artifact_layout, "runtime-artifacts-v1")

        with patch(
            "portable_dmtac_gripper.cli._tactile_gui_artifact_layout_default",
            return_value="runtime-artifacts-v1",
        ):
            configured_default = build_parser().parse_args(["configure-usb"])
        self.assertEqual(
            configured_default.tactile_gui_artifact_layout,
            "runtime-artifacts-v1",
        )


if __name__ == "__main__":
    unittest.main()

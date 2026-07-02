from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hardware.teleop_hand import q6_assignment
from hardware.teleop_hand.q6_assignment import (
    build_assignment_from_tokens,
    compare_assignment_to_scan,
    format_scan_summary,
    load_assignment,
    q6_ports_for_sides,
    save_assignment,
    scan_q6_devices,
)
from teleop_runtime.collection.run_collect import _apply_default_q6_ports


def _serial_ports() -> list[dict[str, object]]:
    return [
        {
            "index": 0,
            "device": "/dev/ttyUSB0",
            "port": "/dev/ttyUSB0",
            "preferred_port": "/dev/serial/by-id/q6-left",
            "description": "USB Single Serial",
            "hwid": "USB VID:PID=1A86:7523 SER=LQ6",
            "serial_number": "LQ6",
            "port_aliases": ["/dev/ttyUSB0", "/dev/serial/by-id/q6-left"],
            "probable_q6_serial": True,
        },
        {
            "index": 1,
            "device": "/dev/ttyUSB1",
            "port": "/dev/ttyUSB1",
            "preferred_port": "/dev/serial/by-id/q6-right",
            "description": "USB Single Serial",
            "hwid": "USB VID:PID=1A86:7523 SER=RQ6",
            "serial_number": "RQ6",
            "port_aliases": ["/dev/ttyUSB1", "/dev/serial/by-id/q6-right"],
            "probable_q6_serial": True,
        },
    ]


def _probe(port: str, **_kwargs):
    return {
        "reachable": True,
        "valid_frames": 1,
        "latest_angles_deg": [float(index) for index in range(6)],
        "error": None,
        "port": port,
    }


def _permission_denied_probe(port: str, **_kwargs):
    return {
        "reachable": False,
        "valid_frames": 0,
        "latest_angles_deg": None,
        "error": f"SerialException: [Errno 13] Permission denied: {port!r}",
        "port": port,
    }


def _partial_probe(port: str, **kwargs):
    if port.endswith("USB0"):
        return _probe(port, **kwargs)
    return _permission_denied_probe(port, **kwargs)


class _NonInteractiveStdin:
    def isatty(self) -> bool:
        return False


class Q6AssignmentTest(unittest.TestCase):
    def test_scan_build_save_and_load_latest_assignment(self) -> None:
        scan = scan_q6_devices(serial_devices=_serial_ports(), probe_func=_probe)
        assignment = build_assignment_from_tokens(scan, left="1", right="0")

        self.assertEqual(
            q6_ports_for_sides(assignment),
            {
                "left": "/dev/serial/by-id/q6-right",
                "right": "/dev/serial/by-id/q6-left",
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = save_assignment(assignment, config_dir=tmp)
            loaded = load_assignment("latest", config_dir=tmp)

        self.assertTrue(path.name.endswith("_hand_q6_assignment.json"))
        self.assertEqual(
            q6_ports_for_sides(loaded),
            {
                "left": "/dev/serial/by-id/q6-right",
                "right": "/dev/serial/by-id/q6-left",
            },
        )

    def test_compare_assignment_remaps_by_identity(self) -> None:
        scan = scan_q6_devices(serial_devices=_serial_ports(), probe_func=_probe)
        assignment = build_assignment_from_tokens(scan, left="0")
        current = {
            **scan,
            "serial_ports": [
                {
                    **_serial_ports()[0],
                    "port": "/dev/ttyUSB7",
                    "device": "/dev/ttyUSB7",
                    "preferred_port": "/dev/serial/by-id/q6-left",
                    "port_aliases": ["/dev/ttyUSB7", "/dev/serial/by-id/q6-left"],
                }
            ],
        }

        comparison = compare_assignment_to_scan(assignment, current)

        self.assertFalse(comparison["missing"])
        self.assertEqual(
            q6_ports_for_sides(comparison["effective_assignment"], active_sides=("left",)),
            {"left": "/dev/serial/by-id/q6-left"},
        )

    def test_collection_default_q6_ports_fill_missing_cli_args(self) -> None:
        scan = scan_q6_devices(serial_devices=_serial_ports(), probe_func=_probe)
        assignment = build_assignment_from_tokens(scan, left="0", right="1")
        with tempfile.TemporaryDirectory() as tmp:
            save_assignment(assignment, config_dir=tmp)
            args = argparse.Namespace(
                active_sides=("left", "right"),
                left_hand_port="",
                right_hand_port="/dev/manual-right",
                q6_default_config=True,
                q6_config="latest",
                q6_config_required=True,
            )

            old_default = None
            from hardware.teleop_hand import q6_assignment

            old_default = q6_assignment.default_config_dir
            q6_assignment.default_config_dir = lambda: Path(tmp)
            try:
                _apply_default_q6_ports(args)
            finally:
                q6_assignment.default_config_dir = old_default

        self.assertEqual(args.left_hand_port, "/dev/serial/by-id/q6-left")
        self.assertEqual(args.right_hand_port, "/dev/manual-right")
        self.assertEqual(args.q6_config_loaded_ports, {"left": "/dev/serial/by-id/q6-left"})

    def test_scan_summary_reports_candidate_probe_errors(self) -> None:
        scan = scan_q6_devices(
            serial_devices=_serial_ports(),
            probe_func=_permission_denied_probe,
        )

        text = format_scan_summary(scan)

        self.assertIn("Detected Q6 devices: 0", text)
        self.assertIn("No reachable Q6 devices. Probe attempts:", text)
        self.assertIn("/dev/serial/by-id/q6-left", text)
        self.assertIn("Permission denied", text)
        self.assertIn("serial permission denied", text)

    def test_scan_summary_reports_failed_candidates_when_partially_detected(self) -> None:
        scan = scan_q6_devices(
            serial_devices=_serial_ports(),
            probe_func=_partial_probe,
        )

        text = format_scan_summary(scan)

        self.assertIn("Detected Q6 devices: 1", text)
        self.assertIn("Other Q6 candidate probe attempts failed:", text)
        self.assertIn("/dev/serial/by-id/q6-right", text)
        self.assertIn("Permission denied", text)

    def test_scan_skips_ch340_gripper_without_serial_as_q6_candidate(self) -> None:
        calls: list[str] = []

        def probe(port: str, **kwargs):
            calls.append(port)
            return _probe(port, **kwargs)

        scan = scan_q6_devices(
            serial_devices=[
                {
                    "index": 32,
                    "device": "/dev/ttyUSB0",
                    "port": "/dev/ttyUSB0",
                    "preferred_port": "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                    "description": "USB Serial",
                    "hwid": "USB VID:PID=1A86:7523 LOCATION=3-8.4",
                    "serial_number": None,
                    "product": "USB Serial",
                },
                {
                    "index": 34,
                    "device": "/dev/ttyACM0",
                    "port": "/dev/ttyACM0",
                    "preferred_port": "/dev/serial/by-id/q6-left",
                    "description": "USB Single Serial",
                    "hwid": "USB VID:PID=1A86:55D3 SER=5B7A137621",
                    "serial_number": "5B7A137621",
                    "product": "USB Single Serial",
                },
            ],
            probe_func=probe,
        )

        self.assertEqual(calls, ["/dev/ttyACM0"])
        self.assertFalse(scan["serial_ports"][0]["probable_q6_serial"])
        self.assertTrue(scan["serial_ports"][1]["probable_q6_serial"])
        self.assertEqual(len(scan["q6_devices"]), 1)

    def test_configure_requires_interactive_confirmation_without_yes(self) -> None:
        scan = scan_q6_devices(serial_devices=_serial_ports(), probe_func=_probe)
        args = argparse.Namespace(
            baudrate=115200,
            timeout=0.05,
            probe_timeout_s=0.1,
            probe_all_serial=False,
            show_raw_lines=False,
            left=None,
            right=None,
            swap=False,
            single_side="right",
            visualize_sec=0.0,
            preview_log_hz=1.0,
            yes=False,
            no_save=True,
            config_dir=None,
        )

        with (
            patch.object(q6_assignment, "scan_q6_devices", return_value=scan),
            patch.object(q6_assignment.sys, "stdin", _NonInteractiveStdin()),
        ):
            with self.assertRaisesRegex(RuntimeError, "interactive stdin"):
                q6_assignment._configure_from_args(args)

    def test_configure_allows_noninteractive_yes(self) -> None:
        scan = scan_q6_devices(serial_devices=_serial_ports(), probe_func=_probe)
        args = argparse.Namespace(
            baudrate=115200,
            timeout=0.05,
            probe_timeout_s=0.1,
            probe_all_serial=False,
            show_raw_lines=False,
            left=None,
            right=None,
            swap=False,
            single_side="right",
            visualize_sec=0.0,
            preview_log_hz=1.0,
            yes=True,
            no_save=True,
            config_dir=None,
        )

        with (
            patch.object(q6_assignment, "scan_q6_devices", return_value=scan),
            patch.object(q6_assignment.sys, "stdin", _NonInteractiveStdin()),
        ):
            self.assertEqual(q6_assignment._configure_from_args(args), 0)


if __name__ == "__main__":
    unittest.main()

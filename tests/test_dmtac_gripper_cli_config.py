from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
DMTAC_ROOT = ROOT / "hardware" / "DMTAC_gripper"
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from portable_dmtac_gripper import cli  # noqa: E402


def _assignment() -> dict:
    return {
        "assignments": {
            "left": {
                "gripper": {
                    "port": "/dev/serial/by-path/left-gripper",
                    "gripper_id": 1,
                    "baudrate": 115200,
                    "timeout": 0.03,
                },
                "tactile": {"inner": None, "outer": None},
            },
            "right": {
                "gripper": {
                    "port": "/dev/serial/by-path/right-gripper",
                    "gripper_id": 7,
                    "baudrate": 57600,
                    "timeout": 0.08,
                },
                "tactile": {"inner": None, "outer": None},
            },
        },
    }


class DMTacGripperCliConfigTest(unittest.TestCase):
    def test_config_resolution_uses_saved_port_without_usb_scan_by_default(self) -> None:
        args = SimpleNamespace(
            config="latest",
            side="right",
            port="/dev/ttyUSB0",
            gripper_id=1,
            baudrate=115200,
            timeout=0.03,
            validate_current_usb=False,
            tactile_slots=None,
        )

        old_load = cli.load_assignment
        old_scan = cli.scan_usb_devices
        cli.load_assignment = lambda path: _assignment()

        def fail_scan(**kwargs):  # noqa: ANN202, ANN003
            raise AssertionError("USB scan should not run for default config resolution")

        cli.scan_usb_devices = fail_scan
        try:
            cli._resolve_config_args(args, need_gripper=True, need_tactile=False)
        finally:
            cli.load_assignment = old_load
            cli.scan_usb_devices = old_scan

        self.assertEqual(args.port, "/dev/serial/by-path/right-gripper")
        self.assertEqual(args.gripper_id, 7)
        self.assertEqual(args.baudrate, 57600)
        self.assertEqual(args.timeout, 0.08)


if __name__ == "__main__":
    unittest.main()

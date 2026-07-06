from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class _FakeClient:
    def close(self) -> None:
        return None


def _fake_read_register(_client: object, _address: int, count: int) -> list[int]:
    return [0] * int(count)


class MaxForcePropagationTests(unittest.TestCase):
    def test_runtime_defaults_match_hardware_modbus_default(self) -> None:
        from hardware.inspire_tac.inspire_tac_modbus import InspireTacModbus
        from teleop_runtime.backends.hands.inspire_tac import (
            DEFAULT_INSPIRE_HAND_MAX_FORCE as TELEOP_RUNTIME_DEFAULT,
        )

        hardware_default = inspect.signature(InspireTacModbus).parameters[
            "max_fingertip_force"
        ].default
        self.assertEqual(hardware_default, 800)
        self.assertEqual(TELEOP_RUNTIME_DEFAULT, hardware_default)

    def test_hardware_modbus_constructor_writes_startup_force(self) -> None:
        from hardware.inspire_tac import inspire_tac_modbus as modbus

        writes: list[tuple[int, list[int]]] = []

        def fake_write_register(_client: object, address: int, values: list[int]) -> None:
            writes.append((int(address), list(values)))

        with patch.object(modbus, "open_modbus", return_value=_FakeClient()), patch.object(
            modbus, "read_register", side_effect=_fake_read_register
        ), patch.object(
            modbus, "write_register", side_effect=fake_write_register
        ), patch(
            "builtins.print"
        ):
            hand = modbus.InspireTacModbus(
                ip="192.0.2.10",
                port=6001,
                max_fingertip_force=1234,
                timeout=0.25,
            )

        self.assertEqual(hand.max_fingertip_force, 1234)
        self.assertEqual(writes, [(1498, [1234] * 6)])
        np.testing.assert_array_equal(hand.last_force_command, np.asarray([1234] * 6))

    def test_teleop_runtime_backend_passes_force_to_hardware_modbus(self) -> None:
        from hardware.inspire_tac import inspire_tac_modbus as modbus
        from teleop_runtime.backends.hands.inspire_tac import (
            InspireTacHandBackend,
            InspireTacHandBackendConfig,
        )

        writes: list[tuple[int, list[int]]] = []

        def fake_write_register(_client: object, address: int, values: list[int]) -> None:
            writes.append((int(address), list(values)))

        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.11",
                side="left",
                max_fingertip_force=2345,
                modbus_timeout_sec=0.25,
            )
        )
        with patch.object(modbus, "open_modbus", return_value=_FakeClient()), patch.object(
            modbus, "read_register", side_effect=_fake_read_register
        ), patch.object(
            modbus, "write_register", side_effect=fake_write_register
        ), patch(
            "builtins.print"
        ):
            hand = backend._create_hand()

        self.assertEqual(backend.get_state()["max_fingertip_force"], 2345)
        self.assertEqual(hand.max_fingertip_force, 2345)
        self.assertEqual(writes, [(1498, [2345] * 6)])

    def test_teleop_collect_right_hand_backend_uses_cli_max_force(self) -> None:
        from teleop_runtime.collection import run_collect

        args = SimpleNamespace(
            hz=30.0,
            arm_control_hz=30.0,
            right_arm_ip="192.0.2.20",
            arm_max_angvel=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
            right_hand_ip="192.0.2.12",
            right_hand_port="",
            with_tactile=False,
            hand_tactile_regions="",
            max_fingertip_force=2345,
            hand_send_threshold=0.002,
            hand_telemetry_hz=10.0,
        )

        with patch.object(run_collect, "require_asset_path", return_value="rm65_arm.urdf"), patch.object(
            run_collect,
            "RM65ArmBackend",
            side_effect=lambda config: SimpleNamespace(config=config),
        ):
            all_backends, _rig_backends = run_collect._build_backends(args, ("right",))

        self.assertEqual(all_backends["right_hand"].config.max_fingertip_force, 2345)

    def test_teleop_collect_parse_args_accepts_max_force_alias_for_right_side(self) -> None:
        from teleop_runtime.collection import run_collect

        with tempfile.TemporaryDirectory(prefix="max_force_test_") as dataset_root:
            argv = [
                "prog",
                "--dataset-root",
                dataset_root,
                "--active-sides",
                "right",
                "--right-arm-ip",
                "192.0.2.20",
                "--right-hand-ip",
                "192.0.2.12",
                "--max-force",
                "2345",
                "--no-camera",
                "--no-video",
            ]
            with patch.object(sys, "argv", argv):
                args = run_collect.parse_args()

        self.assertEqual(args.active_sides, ("right",))
        self.assertEqual(args.max_fingertip_force, 2345)

    def test_force_vector_validation_rejects_bad_values(self) -> None:
        from hardware.inspire_tac.inspire_tac_modbus import _normalize_force_vector

        with self.assertRaises(ValueError):
            _normalize_force_vector([1, 2, 3])
        with self.assertRaises(ValueError):
            _normalize_force_vector([1, 2, 3, 4, 5, -1])


if __name__ == "__main__":
    unittest.main()

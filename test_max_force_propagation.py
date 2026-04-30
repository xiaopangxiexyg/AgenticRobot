from __future__ import annotations

import inspect
import unittest
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
        from inspire_data_collect.hand.hand_backend import (
            DEFAULT_INSPIRE_HAND_MAX_FORCE as DATA_COLLECT_DEFAULT,
        )
        from teleop_runtime.backends.hands.inspire_tac import (
            DEFAULT_INSPIRE_HAND_MAX_FORCE as TELEOP_RUNTIME_DEFAULT,
        )

        hardware_default = inspect.signature(InspireTacModbus).parameters[
            "max_fingertip_force"
        ].default
        self.assertEqual(hardware_default, 800)
        self.assertEqual(TELEOP_RUNTIME_DEFAULT, hardware_default)
        self.assertEqual(DATA_COLLECT_DEFAULT, hardware_default)

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

    def test_inspire_data_collect_backend_passes_force_to_modbus(self) -> None:
        from inspire_data_collect.hand import inspire_tac_modbus as modbus
        from inspire_data_collect.hand.hand_backend import (
            InspireTacHandBackend,
            InspireTacHandBackendConfig,
        )

        writes: list[tuple[int, list[int]]] = []

        def fake_write_register(_client: object, address: int, values: list[int]) -> None:
            writes.append((int(address), list(values)))

        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.12",
                side="right",
                max_fingertip_force=3456,
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

        self.assertEqual(backend.get_state()["max_fingertip_force"], 3456)
        self.assertEqual(hand.max_fingertip_force, 3456)
        self.assertEqual(writes, [(1498, [3456] * 6)])

    def test_force_vector_validation_rejects_bad_values(self) -> None:
        from hardware.inspire_tac.inspire_tac_modbus import _normalize_force_vector

        with self.assertRaises(ValueError):
            _normalize_force_vector([1, 2, 3])
        with self.assertRaises(ValueError):
            _normalize_force_vector([1, 2, 3, 4, 5, -1])


if __name__ == "__main__":
    unittest.main()

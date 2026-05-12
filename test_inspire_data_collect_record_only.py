from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from inspire_data_collect.common import EffectorCommand
from inspire_data_collect.hand.hand_backend import (
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from inspire_data_collect.run_collect import (
    DEFAULT_RECORD_HAND_FIXED_POS,
    apply_record_hand_fixed_pos,
    build_components,
    parse_hand_q6,
)


class _FakeHand:
    def __init__(self):
        self.sent: list[list[float]] = []

    def set_joint_pos(self, values: list[float]) -> None:
        self.sent.append(list(values))


class _FakeModbus:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class InspireDataCollectRecordOnlyTests(unittest.TestCase):
    def test_hand_backend_skips_hw_send_when_send_commands_disabled(self) -> None:
        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.10",
                side="right",
                connect_hardware=True,
                send_commands=False,
            )
        )
        fake_hand = _FakeHand()
        backend.hand = fake_hand

        backend.send_command(
            EffectorCommand(
                effector_name="right_hand",
                side="right",
                value=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
                timestamp=1.0,
                metadata={},
            )
        )

        self.assertEqual(fake_hand.sent, [])
        state = backend.get_state()
        self.assertIsNone(state["last_sent_command"])
        self.assertFalse(state["send_commands"])

    def test_fixed_joint_pos_bypasses_record_only_send_block(self) -> None:
        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.10",
                side="right",
                connect_hardware=True,
                send_commands=False,
            )
        )
        fake_hand = _FakeHand()
        backend.hand = fake_hand

        backend.send_fixed_joint_pos(DEFAULT_RECORD_HAND_FIXED_POS)

        self.assertEqual(len(fake_hand.sent), 1)
        np.testing.assert_allclose(fake_hand.sent[0], DEFAULT_RECORD_HAND_FIXED_POS)
        np.testing.assert_allclose(backend.get_state()["last_sent_command"], DEFAULT_RECORD_HAND_FIXED_POS)

    def test_hand_backend_skips_startup_force_in_record_only_mode(self) -> None:
        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.10",
                side="right",
                connect_hardware=True,
                send_commands=False,
            )
        )

        with patch(
            "inspire_data_collect.hand.inspire_tac_modbus.InspireTacModbus",
            side_effect=lambda **kwargs: _FakeModbus(**kwargs),
        ):
            hand = backend._create_hand()

        self.assertFalse(hand.kwargs["configure_startup_force"])

    def test_build_components_uses_record_only_mode_for_hand_backends(self) -> None:
        args = SimpleNamespace(
            left_hand_port="",
            right_hand_port="/dev/ttyACM0",
            left_hand_ip="",
            right_hand_ip="192.0.2.11",
            hand_modbus_port=6000,
            hand_reader_baudrate=115200,
            hand_reader_timeout=0.05,
            show_hand_raw_lines=False,
            exo_hand_calibration=False,
            exo_hand_calibration_path="",
            silent=True,
            record_hand_only=True,
            record_hand_fixed_pos=DEFAULT_RECORD_HAND_FIXED_POS,
            hand_joint_order=(5, 4, 3, 2, 0, 1),
            hand_invert=True,
            hand_command_alpha=1.0,
            hand_send_threshold=0.002,
            hand_telemetry_hz=10.0,
            hand_tactile_regions="all",
            max_fingertip_force=800,
        )

        with patch("inspire_data_collect.run_collect.HandQ6Source.from_ports") as from_ports:
            from_ports.return_value = SimpleNamespace()
            _source, _retargeters, backends = build_components(args, ("right",))

        self.assertFalse(backends["right"].config.send_commands)

    def test_parse_hand_q6_accepts_default_text(self) -> None:
        parsed = parse_hand_q6("[0.096, 0.057, 0.022, 0.056, 0.495, 0.]")
        self.assertEqual(parsed, DEFAULT_RECORD_HAND_FIXED_POS)

    def test_apply_record_hand_fixed_pos_targets_all_backends(self) -> None:
        left_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.10",
                side="left",
                connect_hardware=True,
                send_commands=False,
            )
        )
        right_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.11",
                side="right",
                connect_hardware=True,
                send_commands=False,
            )
        )
        left_hand = _FakeHand()
        right_hand = _FakeHand()
        left_backend.hand = left_hand
        right_backend.hand = right_hand

        with patch("builtins.print"):
            apply_record_hand_fixed_pos(
                {"left": left_backend, "right": right_backend},
                DEFAULT_RECORD_HAND_FIXED_POS,
                reason="unit_test",
            )

        self.assertEqual(len(left_hand.sent), 1)
        self.assertEqual(len(right_hand.sent), 1)
        np.testing.assert_allclose(left_hand.sent[0], DEFAULT_RECORD_HAND_FIXED_POS)
        np.testing.assert_allclose(right_hand.sent[0], DEFAULT_RECORD_HAND_FIXED_POS)


if __name__ == "__main__":
    unittest.main()

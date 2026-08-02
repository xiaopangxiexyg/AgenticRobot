from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hardware.foot_pad.service import (
    CalibrationHit,
    CandidateDevice,
    FootPadError,
    MAX_PEDALS,
    SPECIAL_KEYS,
    build_config,
    normalize_config,
    validate_config,
)


def candidate(id_path: str, key_codes: set[int] | None = None) -> CandidateDevice:
    return CandidateDevice(
        event_path="/dev/input/event99",
        id_path=id_path,
        name="USB Foot Pad",
        vendor_id="0483",
        model_id="5750",
        key_codes=frozenset(key_codes or {30}),
    )


class FootPadConfigTests(unittest.TestCase):
    def test_supports_ten_pads_with_stable_legacy_prefix(self) -> None:
        self.assertEqual(MAX_PEDALS, 10)
        self.assertEqual(
            SPECIAL_KEYS[:4],
            (("F13", 183), ("F14", 184), ("F15", 185), ("F16", 186)),
        )
        self.assertEqual(
            SPECIAL_KEYS[4:],
            (
                ("F17", 187),
                ("F18", 188),
                ("F19", 189),
                ("F20", 190),
                ("F21", 191),
                ("F22", 192),
            ),
        )

        candidates = [
            candidate(f"pci-usb-0:6.{index}:1.1", {index + 30})
            for index in range(10)
        ]
        config = build_config(
            [
                CalibrationHit(item, index + 30, f"KEY_{index}")
                for index, item in enumerate(candidates)
            ]
        )

        self.assertEqual(config["count"], 10)
        self.assertEqual(
            [pedal["output"]["key_name"] for pedal in config["pedals"]],
            [name for name, _ in SPECIAL_KEYS],
        )
        self.assertEqual(validate_config(config, candidates), [])

    def test_normalize_rejects_more_than_ten_pads(self) -> None:
        first = candidate("pci-usb-0:6.1:1.1", {30})
        config = build_config([CalibrationHit(first, 30, "KEY_A")])
        config["count"] = MAX_PEDALS + 1

        with self.assertRaisesRegex(FootPadError, "between 1 and 10"):
            normalize_config(config)

    def test_build_and_validate_two_pads(self) -> None:
        first = candidate("pci-usb-0:6.1:1.1", {30})
        second = candidate("pci-usb-0:6.2:1.1", {48})
        config = build_config(
            [
                CalibrationHit(first, 30, "KEY_A"),
                CalibrationHit(second, 48, "KEY_B"),
            ]
        )

        self.assertEqual(config["count"], 2)
        self.assertEqual(config["pedals"][0]["output"]["key_code"], SPECIAL_KEYS[0][1])
        self.assertEqual(config["pedals"][1]["output"]["key_code"], SPECIAL_KEYS[1][1])
        self.assertEqual(validate_config(config, [first, second]), [])

    def test_validation_rejects_missing_or_changed_pad(self) -> None:
        first = candidate("pci-usb-0:6.1:1.1", {30})
        config = build_config([CalibrationHit(first, 30, "KEY_A")])

        self.assertIn("not connected", validate_config(config, [] )[0])
        changed = candidate("pci-usb-0:6.1:1.1", {31})
        self.assertIn("no longer exposes", validate_config(config, [changed])[0])

    def test_normalize_rejects_noncanonical_output_key(self) -> None:
        first = candidate("pci-usb-0:6.1:1.1", {30})
        config = build_config([CalibrationHit(first, 30, "KEY_A")])
        config["pedals"][0]["output"]["key_code"] = SPECIAL_KEYS[1][1]

        with self.assertRaises(FootPadError):
            normalize_config(config)

    def test_normalize_rejects_unknown_schema(self) -> None:
        first = candidate("pci-usb-0:6.1:1.1", {30})
        config = build_config([CalibrationHit(first, 30, "KEY_A")])
        config["schema_version"] = "other.v1"

        with self.assertRaises(FootPadError):
            normalize_config(config)

    def test_forward_events_handles_unhashable_input_device(self) -> None:
        from hardware.foot_pad import service

        class UnhashableDevice:
            def __init__(self) -> None:
                self.events = [SimpleNamespace(type=1, code=30, value=1)]

            def __eq__(self, other: object) -> bool:
                return self is other

            def fileno(self) -> int:
                return 123

            def read(self) -> list[SimpleNamespace]:
                events, self.events = self.events, []
                return events

        class FakeUi:
            def __init__(self) -> None:
                self.writes: list[tuple[int, int, int]] = []
                self.sync_count = 0

            def write(self, event_type: int, code: int, value: int) -> None:
                self.writes.append((event_type, code, value))

            def syn(self) -> None:
                self.sync_count += 1

        device = UnhashableDevice()
        self.assertIsNone(device.__hash__)
        first = candidate("pci-usb-0:6.1:1.1", {30})
        config = build_config([CalibrationHit(first, 30, "KEY_A")])
        ui = FakeUi()
        fake_evdev = SimpleNamespace(ecodes=SimpleNamespace(EV_KEY=1))

        with patch.object(
            service.select,
            "select",
            side_effect=[([123], [], []), KeyboardInterrupt],
        ):
            with self.assertRaises(KeyboardInterrupt):
                service._forward_events(config, {first.id_path: device}, ui, fake_evdev)

        self.assertEqual(ui.writes, [(1, SPECIAL_KEYS[0][1], 1), (1, SPECIAL_KEYS[0][1], 0)])
        self.assertEqual(ui.sync_count, 2)


if __name__ == "__main__":
    unittest.main()

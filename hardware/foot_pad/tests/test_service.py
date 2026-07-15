from __future__ import annotations

import unittest

from hardware.foot_pad.service import (
    CalibrationHit,
    CandidateDevice,
    FootPadError,
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


if __name__ == "__main__":
    unittest.main()

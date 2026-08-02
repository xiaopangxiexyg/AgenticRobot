from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hardware.angle_trigger.angle_trigger import (
    ANGLE_SCHEMA_VERSION,
    ANGLE_SCHEMA_VERSION_V2,
    ASSIGNMENT_SCHEMA_VERSION,
    ASSIGNMENT_SCHEMA_VERSION_V2,
    AngleTriggerError,
    DeferredAngleTriggerReader,
    SensorCandidate,
    angle_calibration_path,
    build_angle_calibration,
    build_assignment,
    load_configured_reader,
    prepare_reader,
    resolve_assignment,
    save_angle_calibration_config,
    save_assignment_config,
)


def sensor(device: str, location: str, serial_number: str = "") -> SensorCandidate:
    return SensorCandidate(
        device=device,
        servo_id=1,
        location=location,
        serial_number=serial_number,
        vid="1a86",
        pid="7523",
        description="USB Serial",
    )


class AngleTriggerConfigTests(unittest.TestCase):
    def test_assignment_resolves_by_usb_location(self) -> None:
        left = sensor("/dev/ttyUSB0", "1-6.1")
        right = sensor("/dev/ttyUSB1", "1-6.2")
        assignment = build_assignment({"left": left, "right": right})

        resolved, errors = resolve_assignment(assignment, [right, left])

        self.assertEqual(errors, [])
        self.assertEqual(resolved, {"left": left, "right": right})
        self.assertEqual(assignment["schema_version"], ASSIGNMENT_SCHEMA_VERSION)

    def test_changed_location_requires_reassignment(self) -> None:
        assignment = build_assignment(
            {"left": sensor("/dev/ttyUSB0", "1-6.1"), "right": sensor("/dev/ttyUSB1", "1-6.2")}
        )

        resolved, errors = resolve_assignment(
            assignment,
            [sensor("/dev/ttyUSB2", "1-7.1"), sensor("/dev/ttyUSB3", "1-7.2")],
        )

        self.assertIsNone(resolved)
        self.assertEqual(len(errors), 2)

    def test_angle_endpoints_allow_reverse_direction(self) -> None:
        calibration = build_angle_calibration(
            {
                "left": {"zero_deg": 170.0, "one_deg": 25.0},
                "right": {"zero_deg": 10.0, "one_deg": 95.0},
            }
        )

        self.assertEqual(calibration["schema_version"], ANGLE_SCHEMA_VERSION)
        self.assertEqual(calibration["profiles"]["left"]["one_deg"], 25.0)

    def test_angle_endpoints_reject_nearly_equal_values(self) -> None:
        with self.assertRaises(AngleTriggerError):
            build_angle_calibration(
                {
                    "left": {"zero_deg": 10.0, "one_deg": 10.5},
                    "right": {"zero_deg": 20.0, "one_deg": 50.0},
                }
            )

    def test_single_side_profiles_are_additive_and_resolve_one_sensor(self) -> None:
        right = sensor("/dev/ttyUSB1", "1-6.2")
        assignment = build_assignment({"right": right})
        calibration = build_angle_calibration(
            {"right": {"zero_deg": 10.0, "one_deg": 95.0}}
        )

        resolved, errors = resolve_assignment(
            assignment,
            [right],
            active_sides=("right",),
        )

        self.assertEqual(errors, [])
        self.assertEqual(resolved, {"right": right})
        self.assertEqual(
            assignment["schema_version"],
            ASSIGNMENT_SCHEMA_VERSION_V2,
        )
        self.assertEqual(
            calibration["schema_version"],
            ANGLE_SCHEMA_VERSION_V2,
        )
        self.assertEqual(assignment["active_sides"], ["right"])
        self.assertEqual(calibration["active_sides"], ["right"])

    def test_single_side_config_does_not_overwrite_bilateral_latest(self) -> None:
        left = sensor("/dev/ttyUSB0", "1-6.1")
        right = sensor("/dev/ttyUSB1", "1-6.2")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bilateral = save_assignment_config(
                {"left": left, "right": right},
                config_root=root,
            )
            single = save_assignment_config(
                {"right": right},
                config_root=root,
            )

            self.assertEqual(bilateral.name, "latest.json")
            self.assertEqual(single.name, "right.json")
            self.assertTrue(bilateral.exists())
            self.assertTrue(single.exists())

    def test_single_side_reader_can_use_legacy_bilateral_profile(self) -> None:
        left = sensor("/dev/ttyUSB0", "1-6.1")
        right = sensor("/dev/ttyUSB1", "1-6.2")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_assignment_config(
                {"left": left, "right": right},
                config_root=root,
            )
            save_angle_calibration_config(
                {
                    "left": {"zero_deg": 170.0, "one_deg": 25.0},
                    "right": {"zero_deg": 10.0, "one_deg": 95.0},
                },
                config_root=root,
            )
            with patch(
                "hardware.angle_trigger.angle_trigger.discover_sensors",
                return_value=[right],
            ):
                reader = load_configured_reader(
                    config_root=root,
                    active_sides=("right",),
                )

            self.assertEqual(reader.active_sides, ("right",))
            self.assertEqual(reader.mapping, {"right": right})
            self.assertEqual(
                reader.calibration["profiles"],
                {"left": {"zero_deg": 170.0, "one_deg": 25.0},
                 "right": {"zero_deg": 10.0, "one_deg": 95.0}},
            )
            self.assertFalse(
                angle_calibration_path(
                    root,
                    active_sides=("right",),
                ).exists()
            )

    def test_deferred_reader_does_not_discover_until_open(self) -> None:
        class FakeReader:
            def __init__(self) -> None:
                self.opened = False

            def open(self) -> None:
                self.opened = True

            def close(self) -> None:
                self.opened = False

            def read(self) -> dict[str, float]:
                return {"right": 0.5}

        fake = FakeReader()
        with patch(
            "hardware.angle_trigger.angle_trigger.load_configured_reader",
            return_value=fake,
        ) as loader:
            reader = DeferredAngleTriggerReader(active_sides=("right",))
            loader.assert_not_called()
            reader.open()
            loader.assert_called_once()
            self.assertEqual(reader.read(), {"right": 0.5})
            reader.close()
            self.assertFalse(fake.opened)

    def test_prepare_single_side_writes_only_single_side_profiles(self) -> None:
        right = sensor("/dev/ttyUSB1", "1-6.2")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "hardware.angle_trigger.angle_trigger.discover_sensors",
                    return_value=[right],
                ),
                patch(
                    "hardware.angle_trigger.angle_trigger.calibrate_angles",
                    return_value=build_angle_calibration(
                        {
                            "right": {
                                "zero_deg": 10.0,
                                "one_deg": 95.0,
                            }
                        }
                    ),
                ),
            ):
                reader = prepare_reader(
                    config_root=root,
                    active_sides=("right",),
                )

            self.assertEqual(reader.active_sides, ("right",))
            self.assertTrue((root / "assignment" / "right.json").exists())
            self.assertTrue((root / "angles" / "right.json").exists())
            self.assertFalse((root / "assignment" / "latest.json").exists())
            self.assertFalse((root / "angles" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()

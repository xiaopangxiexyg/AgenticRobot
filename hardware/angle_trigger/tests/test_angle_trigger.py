from __future__ import annotations

import unittest

from hardware.angle_trigger.angle_trigger import (
    ANGLE_SCHEMA_VERSION,
    ASSIGNMENT_SCHEMA_VERSION,
    AngleTriggerError,
    SensorCandidate,
    build_angle_calibration,
    build_assignment,
    resolve_assignment,
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


if __name__ == "__main__":
    unittest.main()

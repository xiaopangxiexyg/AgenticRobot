from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "hardware" / "DMTAC_gripper" / "portable_dmtac_gripper"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import gripper_debug  # noqa: E402


class _FakeGripper:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def set_joint_pos(self, position, **kwargs) -> None:
        self.calls.append(("move", position, kwargs))

    def set_speed(self, value) -> None:
        self.calls.append(("speed", value))

    def set_torque(self, value) -> None:
        self.calls.append(("force", value))

    def get_state(self):
        return SimpleNamespace(to_dict=lambda: {"joint_pos": 0.25})

    def close(self) -> None:
        self.closed = True


class DMTacGripperOnlyDebugTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ROOT / "tests" / "fixtures" / "dmtac_gripper_only_assignment.json"
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text(
            json.dumps(
                {
                    "assignments": {
                        "left": {
                            "gripper": {
                                "port": "/dev/left",
                                "gripper_id": 3,
                                "baudrate": 57600,
                                "timeout": 0.2,
                            }
                        },
                        "right": {
                            "gripper": {
                                "port": "/dev/right",
                                "gripper_id": 4,
                                "baudrate": 115200,
                                "timeout": 0.1,
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.config.unlink(missing_ok=True)

    def test_independent_values_resolve_without_touching_hardware(self) -> None:
        args = gripper_debug.build_parser().parse_args(
            [
                "--config",
                str(self.config),
                "set",
                "--left-opening",
                "0.8",
                "--left-force",
                "0.2",
                "--right-opening",
                "0.3",
                "--right-force",
                "0.4",
            ]
        )

        _, commands = gripper_debug.build_commands(args)

        self.assertEqual([command.endpoint.side for command in commands], ["left", "right"])
        self.assertAlmostEqual(commands[0].joint_pos, 0.2)
        self.assertAlmostEqual(commands[1].joint_pos, 0.7)
        self.assertEqual(commands[0].endpoint.port, "/dev/left")
        self.assertFalse(args.execute)

    def test_force_only_does_not_send_position(self) -> None:
        endpoint = gripper_debug.GripperEndpoint("left", "/dev/left")
        command = gripper_debug.GripperCommand(endpoint=endpoint, force=0.2)
        fake = _FakeGripper()

        result = gripper_debug.execute_commands(
            [command],
            wait=False,
            wait_timeout=1.0,
            gripper_factory=lambda unused: fake,
        )

        self.assertEqual(fake.calls, [("force", 0.2)])
        self.assertTrue(fake.closed)
        self.assertEqual(result[0]["side"], "left")

    def test_write_and_high_force_require_explicit_confirmations(self) -> None:
        command = gripper_debug.GripperCommand(
            endpoint=gripper_debug.GripperEndpoint("right", "/dev/right"),
            opening=0.5,
            force=0.7,
        )
        args = SimpleNamespace(
            execute=True,
            confirmation="",
            allow_high_force=False,
            high_force_confirmation="",
        )
        with self.assertRaisesRegex(ValueError, gripper_debug.WRITE_CONFIRMATION):
            gripper_debug.validate_write_unlock(args, [command])

        args.confirmation = gripper_debug.WRITE_CONFIRMATION
        with self.assertRaisesRegex(ValueError, gripper_debug.HIGH_FORCE_CONFIRMATION):
            gripper_debug.validate_write_unlock(args, [command])

        args.allow_high_force = True
        args.high_force_confirmation = gripper_debug.HIGH_FORCE_CONFIRMATION
        gripper_debug.validate_write_unlock(args, [command])


if __name__ == "__main__":
    unittest.main()

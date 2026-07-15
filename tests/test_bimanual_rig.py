from __future__ import annotations

import unittest

from teleop_runtime.rigs.bimanual import BimanualRig


class _FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        fail_hold: bool = False,
        fail_send: bool = False,
    ) -> None:
        self._name = name
        self.fail_hold = fail_hold
        self.fail_send = fail_send
        self.holds: list[str] = []
        self.send_count = 0

    @property
    def name(self) -> str:
        return self._name

    def send_command(self, command) -> None:
        del command
        self.send_count += 1
        if self.fail_send:
            raise RuntimeError(f"{self.name} send failed")

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.holds.append(reason)
        if self.fail_hold:
            raise RuntimeError(f"{self.name} hold failed")


class BimanualRigHoldMotionTest(unittest.TestCase):
    def test_hold_motion_attempts_all_backends(self) -> None:
        left = _FakeBackend("left", fail_hold=True)
        right = _FakeBackend("right")
        rig = BimanualRig({"left_arm": left, "right_arm": right})

        with self.assertRaisesRegex(RuntimeError, "left_arm"):
            rig.hold_motion("unit test hold")

        self.assertEqual(left.holds, ["unit test hold"])
        self.assertEqual(right.holds, ["unit test hold"])

    def test_hold_motion_reports_all_failures(self) -> None:
        left = _FakeBackend("left", fail_hold=True)
        right = _FakeBackend("right", fail_hold=True)
        rig = BimanualRig({"left_arm": left, "right_arm": right})

        with self.assertRaisesRegex(RuntimeError, "left_arm.*right_arm"):
            rig.hold_motion("unit test hold")

    def test_position_group_dispatch_failure_holds_every_backend(self) -> None:
        left = _FakeBackend("left", fail_send=True)
        right = _FakeBackend("right")
        rig = BimanualRig(
            {"left_arm": left, "right_arm": right},
            parallel_dispatch=True,
            fail_stop_on_dispatch_error=True,
        )
        command = type("Command", (), {"commands": {"left_arm": object(), "right_arm": object()}})()

        with self.assertRaisesRegex(RuntimeError, "group fail-stop"):
            rig.send_commands(command)

        self.assertEqual(left.send_count, 1)
        self.assertEqual(right.send_count, 1)
        self.assertEqual(len(left.holds), 1)
        self.assertEqual(len(right.holds), 1)

    def test_default_dispatch_failure_keeps_legacy_non_fail_stop_behavior(self) -> None:
        left = _FakeBackend("left", fail_send=True)
        right = _FakeBackend("right")
        rig = BimanualRig({"left_arm": left, "right_arm": right})
        command = type("Command", (), {"commands": {"left_arm": object(), "right_arm": object()}})()

        with self.assertRaisesRegex(RuntimeError, "left send failed"):
            rig.send_commands(command)

        self.assertEqual(left.holds, [])
        self.assertEqual(right.holds, [])


if __name__ == "__main__":
    unittest.main()

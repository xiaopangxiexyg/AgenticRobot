from __future__ import annotations

import unittest

from teleop_runtime.rigs.bimanual import BimanualRig


class _FakeBackend:
    def __init__(self, name: str, *, fail_hold: bool = False) -> None:
        self._name = name
        self.fail_hold = fail_hold
        self.holds: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def send_command(self, command) -> None:
        del command

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


if __name__ == "__main__":
    unittest.main()

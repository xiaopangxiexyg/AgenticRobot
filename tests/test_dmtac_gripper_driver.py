from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DMTAC_ROOT = ROOT / "hardware" / "DMTAC_gripper"
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from portable_dmtac_gripper import gripper_driver  # noqa: E402
from portable_dmtac_gripper.gripper_driver import InspireGripper  # noqa: E402


class _State:
    def __init__(self, label: str, moving: bool) -> None:
        self.label = label
        self.is_moving = bool(moving)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


class _FakeGripper(InspireGripper):
    def __init__(self, states: list[_State]) -> None:
        self._states = list(states)
        self._last_state = self._states[-1]
        self.read_labels: list[str] = []

    def get_state(self):  # noqa: ANN201
        if self._states:
            self._last_state = self._states.pop(0)
        self.read_labels.append(self._last_state.label)
        return self._last_state


class DMTacGripperDriverTest(unittest.TestCase):
    def _patch_clock(self, clock: _FakeClock):
        old_time = gripper_driver.time.time
        old_sleep = gripper_driver.time.sleep
        gripper_driver.time.time = clock.time
        gripper_driver.time.sleep = clock.sleep
        return old_time, old_sleep

    def test_wait_until_idle_allows_device_to_enter_moving(self) -> None:
        clock = _FakeClock()
        old_time, old_sleep = self._patch_clock(clock)
        try:
            gripper = _FakeGripper(
                [
                    _State("initial_idle", False),
                    _State("moving", True),
                    _State("final_idle", False),
                ]
            )
            result = gripper.wait_until_idle(
                timeout=1.0,
                poll_interval=0.05,
                start_grace=0.1,
            )
        finally:
            gripper_driver.time.time = old_time
            gripper_driver.time.sleep = old_sleep

        self.assertEqual(result.label, "final_idle")
        self.assertEqual(gripper.read_labels, ["initial_idle", "moving", "final_idle"])

    def test_wait_until_idle_returns_after_grace_for_no_motion(self) -> None:
        clock = _FakeClock()
        old_time, old_sleep = self._patch_clock(clock)
        try:
            gripper = _FakeGripper(
                [
                    _State("idle_1", False),
                    _State("idle_2", False),
                    _State("idle_3", False),
                ]
            )
            result = gripper.wait_until_idle(
                timeout=1.0,
                poll_interval=0.05,
                start_grace=0.1,
            )
        finally:
            gripper_driver.time.time = old_time
            gripper_driver.time.sleep = old_sleep

        self.assertEqual(result.label, "idle_3")
        self.assertEqual(gripper.read_labels, ["idle_1", "idle_2", "idle_3"])


if __name__ == "__main__":
    unittest.main()

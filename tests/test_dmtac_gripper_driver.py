from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DMTAC_ROOT = ROOT / "hardware" / "DMTAC_gripper"
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from portable_dmtac_gripper import gripper_driver  # noqa: E402
from portable_dmtac_gripper.gripper_driver import (  # noqa: E402
    GripperError,
    InspireGripper,
    InspireGripperModbus,
)


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


class _ScriptedSerial:
    scripts: list[bytes] = []
    instances: list["_ScriptedSerial"] = []
    clock: _FakeClock | None = None

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.args = args
        self.kwargs = kwargs
        self.timeout = float(kwargs.get("timeout", 0.03))
        self.is_open = True
        self.writes: list[bytes] = []
        self._rx = bytearray(self.scripts.pop(0) if self.scripts else b"")
        self.close_count = 0
        self.instances.append(self)

    def reset_input_buffer(self) -> None:
        return

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        return

    def read(self, n: int) -> bytes:
        if not self._rx:
            if self.clock is not None:
                self.clock.now += self.timeout + 0.001
            return b""
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def close(self) -> None:
        self.close_count += 1
        self.is_open = False


def _read_register_response(slave_id: int, *regs: int) -> bytes:
    payload = b"".join(int(value).to_bytes(2, byteorder="big") for value in regs)
    frame = bytes([int(slave_id), 0x03, len(payload)]) + payload
    return InspireGripperModbus._append_crc(frame)


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

    def test_modbus_reconnects_and_retries_after_read_timeout(self) -> None:
        clock = _FakeClock()
        old_time, old_sleep = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [b"", _read_register_response(1, 1234)]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            mb = InspireGripperModbus(
                port="/dev/fake0",
                slave_id=1,
                timeout=0.01,
                reconnect_on_error=True,
                reconnect_attempts=1,
                reconnect_delay=0.0,
            )
            regs = mb.read_holding_registers(61, 1)
        finally:
            gripper_driver.serial.Serial = old_serial
            gripper_driver.time.time = old_time
            gripper_driver.time.sleep = old_sleep
            _ScriptedSerial.clock = None

        self.assertEqual(regs, [1234])
        self.assertEqual(len(_ScriptedSerial.instances), 2)
        self.assertEqual(_ScriptedSerial.instances[0].close_count, 1)
        self.assertEqual(mb.reconnect_count, 1)
        self.assertEqual(len(_ScriptedSerial.instances[1].writes), 1)

    def test_modbus_does_not_reconnect_on_unexpected_slave_id(self) -> None:
        clock = _FakeClock()
        old_time, old_sleep = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [_read_register_response(2, 1234)]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            mb = InspireGripperModbus(
                port="/dev/fake0",
                slave_id=1,
                timeout=0.01,
                reconnect_on_error=True,
                reconnect_attempts=1,
                reconnect_delay=0.0,
            )
            with self.assertRaisesRegex(GripperError, "Unexpected slave id"):
                mb.read_holding_registers(61, 1)
        finally:
            gripper_driver.serial.Serial = old_serial
            gripper_driver.time.time = old_time
            gripper_driver.time.sleep = old_sleep
            _ScriptedSerial.clock = None

        self.assertEqual(len(_ScriptedSerial.instances), 1)
        self.assertEqual(mb.reconnect_count, 0)


if __name__ == "__main__":
    unittest.main()

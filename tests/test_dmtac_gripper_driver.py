from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DMTAC_ROOT = ROOT / "hardware" / "DMTAC_gripper"
if str(DMTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(DMTAC_ROOT))

from portable_dmtac_gripper import device as dmtac_device  # noqa: E402
from portable_dmtac_gripper import gripper_driver  # noqa: E402
from portable_dmtac_gripper.device import (  # noqa: E402
    DMTacGripperDevice,
    GripperConfig,
    TactileConfig,
)
from portable_dmtac_gripper.gripper_driver import (  # noqa: E402
    GripperError,
    InspireGripper,
    InspireGripperModbus,
    decode_gripper_error_code,
    describe_gripper_error_code,
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

    def monotonic(self) -> float:
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
        self.write_timeout = kwargs.get("write_timeout")
        self.is_open = True
        self.writes: list[bytes] = []
        self._rx = bytearray(self.scripts.pop(0) if self.scripts else b"")
        self.close_count = 0
        self.open_count = 0
        self.reset_input_count = 0
        self.reset_output_count = 0
        self.instances.append(self)

    def reset_input_buffer(self) -> None:
        self.reset_input_count += 1

    def reset_output_buffer(self) -> None:
        self.reset_output_count += 1

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

    def open(self) -> None:
        self.open_count += 1
        self.is_open = True


def _read_register_response(slave_id: int, *regs: int) -> bytes:
    payload = b"".join(int(value).to_bytes(2, byteorder="big") for value in regs)
    frame = bytes([int(slave_id), 0x03, len(payload)]) + payload
    return InspireGripperModbus._append_crc(frame)


def _write_single_response(slave_id: int, reg_addr: int, value: int) -> bytes:
    frame = (
        bytes([int(slave_id), 0x06])
        + int(reg_addr).to_bytes(2, byteorder="big")
        + int(value).to_bytes(2, byteorder="big")
    )
    return InspireGripperModbus._append_crc(frame)


def _write_multiple_response(slave_id: int, start_addr: int, count: int) -> bytes:
    frame = (
        bytes([int(slave_id), 0x10])
        + int(start_addr).to_bytes(2, byteorder="big")
        + int(count).to_bytes(2, byteorder="big")
    )
    return InspireGripperModbus._append_crc(frame)


class DMTacGripperDriverTest(unittest.TestCase):
    def _patch_clock(self, clock: _FakeClock):
        old_time = gripper_driver.time.time
        old_sleep = gripper_driver.time.sleep
        old_monotonic = gripper_driver.time.monotonic
        gripper_driver.time.time = clock.time
        gripper_driver.time.sleep = clock.sleep
        gripper_driver.time.monotonic = clock.monotonic
        return old_time, old_sleep, old_monotonic

    def _restore_clock(self, patched_clock) -> None:
        old_time, old_sleep, old_monotonic = patched_clock
        gripper_driver.time.time = old_time
        gripper_driver.time.sleep = old_sleep
        gripper_driver.time.monotonic = old_monotonic

    def test_opening_register_decodes_signed_int16_before_range_check(self) -> None:
        self.assertEqual(InspireGripper._decode_opening_register(0xFFFC), 0)
        self.assertEqual(InspireGripper._decode_opening_register(0xFFFE), 0)
        self.assertEqual(InspireGripper._decode_opening_register(0), 0)
        self.assertEqual(InspireGripper._decode_opening_register(1000), 1000)

    def test_error_code_16_is_named_without_masking_unknown_bits(self) -> None:
        self.assertEqual(
            decode_gripper_error_code(16),
            ("internal_communication_fault",),
        )
        self.assertEqual(
            decode_gripper_error_code(17),
            ("internal_communication_fault", "unknown_error_bit_0x0001"),
        )
        self.assertEqual(
            describe_gripper_error_code(16),
            "0x0010(internal_communication_fault)",
        )

    def test_clear_fault_and_confirm_waits_for_fresh_zero(self) -> None:
        class _ConfirmingGripper(InspireGripper):
            def __init__(self) -> None:
                self.errors = [16, 16, 0, 0, 0]
                self.clear_calls = 0

            def clear_fault(self) -> None:
                self.clear_calls += 1

            def get_state(self):  # noqa: ANN201
                error = self.errors.pop(0)
                return type(
                    "State",
                    (),
                    {
                        "error_code": error,
                        "status_code": 2,
                        "status_text": "closed idle",
                    },
                )()

        gripper = _ConfirmingGripper()

        state = gripper.clear_fault_and_confirm(
            timeout=0.1,
            poll_interval=0.001,
            ack_interval=1.0,
        )

        self.assertEqual(state.error_code, 0)
        self.assertEqual(gripper.clear_calls, 1)

    def test_clear_fault_and_confirm_rejects_transient_zero(self) -> None:
        class _OscillatingGripper(InspireGripper):
            def __init__(self) -> None:
                self.errors = iter((16, 0, 16, 0, 16, 16))
                self.clear_calls = 0

            def clear_fault(self) -> None:
                self.clear_calls += 1

            def get_state(self):  # noqa: ANN201
                error = next(self.errors, 16)
                return type(
                    "State",
                    (),
                    {
                        "error_code": error,
                        "status_code": 2,
                        "status_text": "closed idle",
                    },
                )()

        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        try:
            with self.assertRaisesRegex(
                GripperError,
                r"zero_streak=0/3",
            ):
                _OscillatingGripper().clear_fault_and_confirm(
                    timeout=0.006,
                    poll_interval=0.001,
                    ack_interval=1.0,
                )
        finally:
            self._restore_clock(patched_clock)

    def test_modbus_requests_process_exclusive_serial_ownership(self) -> None:
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = []
        _ScriptedSerial.instances = []
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            mb = InspireGripperModbus(
                port="/dev/fake-exclusive",
                slave_id=1,
                reconnect_on_error=False,
            )
            mb.close()
        finally:
            gripper_driver.serial.Serial = old_serial

        self.assertEqual(len(_ScriptedSerial.instances), 1)
        self.assertIs(_ScriptedSerial.instances[0].kwargs.get("exclusive"), True)

    def test_composite_connect_rolls_back_gripper_when_tactile_discovery_fails(
        self,
    ) -> None:
        class _FakeGripper:
            def __init__(self, **_kwargs: object) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        fake_gripper = _FakeGripper()
        device = DMTacGripperDevice(
            gripper=GripperConfig(port="/dev/fake0"),
            tactile=TactileConfig(dev_ids=None),
        )

        with (
            patch.object(dmtac_device, "InspireGripper", return_value=fake_gripper),
            patch.object(dmtac_device, "discover_device_ids", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "No DM-Tac tactile device"),
        ):
            device.connect()

        self.assertTrue(fake_gripper.closed)
        self.assertIsNone(device.gripper)
        self.assertIsNone(device.tactile)

    def test_composite_connect_closes_partially_connected_tactile_candidate(
        self,
    ) -> None:
        class _FakeGripper:
            def __init__(self, **_kwargs: object) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class _FailingTactile:
            def __init__(self, _configs: object) -> None:
                self.closed = False

            def connect(self):  # noqa: ANN201
                raise RuntimeError("tactile connect failed")

            def close(self) -> None:
                self.closed = True

        fake_gripper = _FakeGripper()
        fake_tactile = _FailingTactile([])
        device = DMTacGripperDevice(
            gripper=GripperConfig(port="/dev/fake0"),
            tactile=TactileConfig(dev_ids=["fake-tactile"]),
        )

        with (
            patch.object(dmtac_device, "InspireGripper", return_value=fake_gripper),
            patch.object(dmtac_device, "DMTacMultiDevice", return_value=fake_tactile),
            self.assertRaisesRegex(RuntimeError, "tactile connect failed"),
        ):
            device.connect()

        self.assertTrue(fake_tactile.closed)
        self.assertTrue(fake_gripper.closed)
        self.assertIsNone(device.gripper)
        self.assertIsNone(device.tactile)

    def test_opening_register_still_rejects_values_outside_signed_physical_range(
        self,
    ) -> None:
        with self.assertRaisesRegex(GripperError, "Abnormal opening_raw=-4096"):
            InspireGripper._decode_opening_register(0xF000)

        with self.assertRaisesRegex(GripperError, "Abnormal opening_raw=5000"):
            InspireGripper._decode_opening_register(5000)

    def test_get_state_decodes_negative_signed_opening_register(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [
            _read_register_response(1, 0xFFFC, 0xFFFF, 25, 0, 1)
        ]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                reconnect_on_error=False,
            )
            state = gripper.get_state()
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(state.opening_raw, 0)
        self.assertEqual(state.current_raw, 0xFFFF)
        self.assertEqual(state.current, -1)
        self.assertEqual(state.joint_pos, 1.0)
        self.assertEqual(state.opening_mm, 0.0)

    def test_read_joint_pos_decodes_negative_signed_opening_register(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [_read_register_response(1, 0xFFFC)]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                reconnect_on_error=False,
            )
            joint_pos = gripper.read_joint_pos()
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(joint_pos, [1.0])

    def test_wait_until_idle_allows_device_to_enter_moving(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
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
            self._restore_clock(patched_clock)

        self.assertEqual(result.label, "final_idle")
        self.assertEqual(gripper.read_labels, ["initial_idle", "moving", "final_idle"])

    def test_wait_until_idle_returns_after_grace_for_no_motion(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
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
            self._restore_clock(patched_clock)

        self.assertEqual(result.label, "idle_3")
        self.assertEqual(gripper.read_labels, ["idle_1", "idle_2", "idle_3"])

    def test_modbus_reconnects_and_retries_after_read_timeout(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
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
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(regs, [1234])
        self.assertEqual(len(_ScriptedSerial.instances), 2)
        self.assertEqual(_ScriptedSerial.instances[0].close_count, 1)
        self.assertEqual(mb.reconnect_count, 1)
        self.assertEqual(len(_ScriptedSerial.instances[1].writes), 1)

    def test_modbus_does_not_reconnect_on_unexpected_slave_id(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
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
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(len(_ScriptedSerial.instances), 1)
        self.assertEqual(mb.reconnect_count, 0)

    def test_modbus_sets_write_timeout_and_resets_output_buffer(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [_read_register_response(1, 1234)]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            mb = InspireGripperModbus(
                port="/dev/fake0",
                slave_id=1,
                timeout=0.02,
                inter_command_delay=0.0,
                reconnect_on_error=False,
            )
            regs = mb.read_holding_registers(61, 1)
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(regs, [1234])
        self.assertEqual(_ScriptedSerial.instances[0].write_timeout, 0.02)
        self.assertEqual(_ScriptedSerial.instances[0].reset_input_count, 1)
        self.assertEqual(_ScriptedSerial.instances[0].reset_output_count, 1)

    def test_set_joint_pos_resends_same_target_by_default(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        response = _write_single_response(1, InspireGripperModbus.REG_OPENLEN_SET, 500)
        _ScriptedSerial.scripts = [response + response]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                inter_command_delay=0.0,
                reconnect_on_error=False,
            )
            gripper.set_joint_pos(0.5)
            gripper.set_joint_pos(0.5)
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(len(_ScriptedSerial.instances), 1)
        self.assertEqual(len(_ScriptedSerial.instances[0].writes), 2)
        self.assertTrue(
            all(req[1] == 0x06 for req in _ScriptedSerial.instances[0].writes)
        )

    def test_set_joint_pos_can_skip_same_target_when_force_send_false(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        response = _write_single_response(1, InspireGripperModbus.REG_OPENLEN_SET, 500)
        _ScriptedSerial.scripts = [response]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                inter_command_delay=0.0,
                reconnect_on_error=False,
            )
            gripper.set_joint_pos(0.5)
            gripper.set_joint_pos(0.5, force_send=False)
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        self.assertEqual(len(_ScriptedSerial.instances), 1)
        self.assertEqual(len(_ScriptedSerial.instances[0].writes), 1)

    def test_set_joint_pos_splits_speed_torque_position_by_default(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        speed_raw = 201
        torque_raw = 335
        target_raw = 500
        response = b"".join(
            [
                _write_single_response(
                    1,
                    InspireGripperModbus.REG_SPEED_SET,
                    speed_raw,
                ),
                _write_single_response(
                    1,
                    InspireGripperModbus.REG_FORCE_SET,
                    torque_raw,
                ),
                _write_single_response(
                    1,
                    InspireGripperModbus.REG_OPENLEN_SET,
                    target_raw,
                ),
            ]
        )
        _ScriptedSerial.scripts = [response]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                inter_command_delay=0.0,
                reconnect_on_error=False,
            )
            gripper.set_joint_pos(0.5, speed=0.2, torque=0.3)
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        serial_instance = _ScriptedSerial.instances[0]
        self.assertEqual(
            [req[1] for req in serial_instance.writes],
            [0x06, 0x06, 0x06],
        )
        self.assertEqual(
            [
                int.from_bytes(req[2:4], byteorder="big")
                for req in serial_instance.writes
            ],
            [
                InspireGripperModbus.REG_SPEED_SET,
                InspireGripperModbus.REG_FORCE_SET,
                InspireGripperModbus.REG_OPENLEN_SET,
            ],
        )
        self.assertEqual(
            [
                int.from_bytes(req[4:6], byteorder="big")
                for req in serial_instance.writes
            ],
            [speed_raw, torque_raw, target_raw],
        )

    def test_set_joint_pos_can_use_multi_register_fast_path(self) -> None:
        clock = _FakeClock()
        patched_clock = self._patch_clock(clock)
        old_serial = gripper_driver.serial.Serial
        _ScriptedSerial.scripts = [
            _write_multiple_response(1, InspireGripperModbus.REG_OPENLEN_SET, 3)
        ]
        _ScriptedSerial.instances = []
        _ScriptedSerial.clock = clock
        gripper_driver.serial.Serial = _ScriptedSerial
        try:
            gripper = InspireGripper(
                port="/dev/fake0",
                gripper_id=1,
                timeout=0.01,
                inter_command_delay=0.0,
                reconnect_on_error=False,
                use_write_multiple_registers=True,
            )
            gripper.set_joint_pos(0.5, speed=0.2, torque=0.3)
        finally:
            gripper_driver.serial.Serial = old_serial
            self._restore_clock(patched_clock)
            _ScriptedSerial.clock = None

        serial_instance = _ScriptedSerial.instances[0]
        self.assertEqual(len(serial_instance.writes), 1)
        self.assertEqual(serial_instance.writes[0][1], 0x10)
        self.assertEqual(
            int.from_bytes(serial_instance.writes[0][2:4], byteorder="big"),
            InspireGripperModbus.REG_OPENLEN_SET,
        )


if __name__ == "__main__":
    unittest.main()

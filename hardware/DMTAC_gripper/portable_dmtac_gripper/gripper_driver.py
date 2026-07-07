#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import serial
from serial.tools import list_ports

Number = Union[int, float]


def _clip(v: Number, lo: Number, hi: Number) -> Number:
    return max(lo, min(v, hi))


def _ensure_scalar_or_len1(vals, name: str) -> float:
    if isinstance(vals, (int, float)):
        return float(vals)

    if isinstance(vals, (list, tuple)):
        if len(vals) != 1:
            raise ValueError(
                f"{name} must be a scalar or len-1 list/tuple, got {vals}"
            )
        x = vals[0]
        if not isinstance(x, (int, float)):
            raise TypeError(f"{name} element must be int/float, got {type(x)}")
        return float(x)

    raise TypeError(f"{name} must be a scalar or len-1 list/tuple, got {type(vals)}")


class GripperError(RuntimeError):
    pass


@dataclass
class GripperState:
    timestamp: float
    joint_pos: float
    is_moving: bool
    opening_raw: int
    opening_mm: float
    current_raw: int
    current: int
    temperature_c: int
    error_code: int
    status_code: int
    status_text: str
    commanded_joint_pos: Optional[float]
    commanded_speed: Optional[float]
    commanded_torque: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


class InspireGripperModbus:
    REG_SAVE = 1
    REG_SETDEFAULT = 2
    REG_ID = 3
    REG_BAUD = 4
    REG_CATCH_MODE = 5
    REG_STOP = 6
    REG_FAULT_ACK = 7

    REG_OPENLEN_SET = 10
    REG_SPEED_SET = 11
    REG_FORCE_SET = 12

    REG_MAX_OPENLEN = 16
    REG_MIN_OPENLEN = 17

    REG_OPENLEN_ACT = 61
    REG_CURRENT = 62
    REG_TEMP = 63
    REG_ERRORCODE = 64
    REG_STATUS = 65

    BAUD_TO_MODE = {
        115200: 0,
        57600: 1,
        19200: 2,
    }

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        slave_id: int = 1,
        baudrate: int = 115200,
        timeout: float = 0.03,
        inter_command_delay: float = 0.001,
        reconnect_on_error: bool = True,
        reconnect_attempts: int = 1,
        reconnect_delay: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.port = port
        self.slave_id = int(slave_id)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.inter_command_delay = float(inter_command_delay)
        self.reconnect_on_error = bool(reconnect_on_error)
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_delay = max(0.0, float(reconnect_delay))
        self.debug = bool(debug)
        self._io_lock = threading.Lock()
        self.reconnect_count = 0
        self.last_reconnect_error: Optional[str] = None

        self.ser: serial.Serial | None = None
        self._open_serial()

    def _open_serial(self) -> None:
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _reopen_serial_locked(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        if self.reconnect_delay > 0.0:
            time.sleep(self.reconnect_delay)
        self._open_serial()
        self.reconnect_count += 1
        self.last_reconnect_error = None

    def __enter__(self) -> "InspireGripperModbus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    @staticmethod
    def _crc16_modbus(data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    @classmethod
    def _append_crc(cls, data: bytes) -> bytes:
        crc = cls._crc16_modbus(data)
        return data + crc.to_bytes(2, byteorder="little")

    @classmethod
    def _check_crc(cls, frame: bytes) -> None:
        if len(frame) < 4:
            raise GripperError(f"Frame too short: {frame.hex(' ')}")

        recv_crc = int.from_bytes(frame[-2:], byteorder="little")
        calc_crc = cls._crc16_modbus(frame[:-2])

        if recv_crc != calc_crc:
            raise GripperError(
                f"CRC mismatch: recv=0x{recv_crc:04X}, "
                f"calc=0x{calc_crc:04X}, frame={frame.hex(' ')}"
            )

    def _read_exact(self, n: int) -> bytes:
        if self.ser is None or not self.ser.is_open:
            raise GripperError(f"Serial port is not open: {self.port}")
        data = bytearray()
        t0 = time.time()

        while len(data) < n:
            chunk = self.ser.read(n - len(data))
            if chunk:
                data.extend(chunk)
            elif (time.time() - t0) > self.timeout:
                break

        if len(data) != n:
            raise GripperError(
                f"Serial read timeout, expect {n} bytes, got {len(data)}: "
                f"{bytes(data).hex(' ')}"
            )

        return bytes(data)

    @staticmethod
    def _is_reconnectable_error(exc: BaseException) -> bool:
        if isinstance(exc, (serial.SerialException, OSError)):
            return True
        if not isinstance(exc, GripperError):
            return False
        text = str(exc).lower()
        if "unexpected slave id" in text or "modbus exception response" in text:
            return False
        return any(
            token in text
            for token in (
                "timeout",
                "crc mismatch",
                "frame too short",
                "echo mismatch",
                "serial port is not open",
            )
        )

    def _transact_once_locked(self, function_code: int, payload: bytes) -> bytes:
        if self.ser is None or not self.ser.is_open:
            raise GripperError(f"Serial port is not open: {self.port}")
        req = bytes([self.slave_id, function_code]) + payload
        req = self._append_crc(req)

        self.ser.reset_input_buffer()
        self.ser.write(req)
        self.ser.flush()
        self._log(f"[TX] {req.hex(' ')}")

        head = self._read_exact(2)
        addr, func = head[0], head[1]

        if addr != self.slave_id:
            raise GripperError(
                f"Unexpected slave id in response: {addr}, expect {self.slave_id}"
            )

        if func == (function_code | 0x80):
            tail = self._read_exact(3)
            frame = head + tail
            self._check_crc(frame)
            exc_code = frame[2]
            raise GripperError(
                f"Modbus exception response: func=0x{function_code:02X}, "
                f"exc=0x{exc_code:02X}"
            )

        if func != function_code:
            raise GripperError(
                f"Unexpected function code in response: "
                f"got 0x{func:02X}, expect 0x{function_code:02X}"
            )

        if function_code == 0x03:
            byte_count = self._read_exact(1)
            n = byte_count[0]
            tail = self._read_exact(n + 2)
            frame = head + byte_count + tail
        elif function_code in (0x06, 0x10):
            tail = self._read_exact(6)
            frame = head + tail
        else:
            raise NotImplementedError(
                f"Unsupported function code: 0x{function_code:02X}"
            )

        self._check_crc(frame)
        self._log(f"[RX] {frame.hex(' ')}")

        if self.inter_command_delay > 0:
            time.sleep(self.inter_command_delay)

        return frame

    def _transact(self, function_code: int, payload: bytes) -> bytes:
        with self._io_lock:
            attempts_remaining = self.reconnect_attempts if self.reconnect_on_error else 0
            while True:
                try:
                    return self._transact_once_locked(function_code, payload)
                except Exception as exc:  # noqa: BLE001
                    if attempts_remaining <= 0 or not self._is_reconnectable_error(exc):
                        raise
                    attempts_remaining -= 1
                    self.last_reconnect_error = f"{type(exc).__name__}: {exc}"
                    self._log(f"[RECONNECT] {self.port}: {self.last_reconnect_error}")
                    try:
                        self._reopen_serial_locked()
                    except Exception as reopen_exc:  # noqa: BLE001
                        raise GripperError(
                            f"Serial reconnect failed after {self.last_reconnect_error}: "
                            f"{type(reopen_exc).__name__}: {reopen_exc}"
                        ) from reopen_exc

    def read_holding_registers(self, start_addr: int, count: int) -> list[int]:
        if not (1 <= int(count) <= 125):
            raise ValueError(f"Invalid register count: {count}")

        payload = struct.pack(">HH", int(start_addr), int(count))
        frame = self._transact(0x03, payload)

        byte_count = frame[2]
        data = frame[3 : 3 + byte_count]

        if len(data) != count * 2:
            raise GripperError(
                f"Unexpected register byte count: count={count}, "
                f"byte_count={byte_count}, frame={frame.hex(' ')}"
            )

        regs = []
        for i in range(0, len(data), 2):
            regs.append(int.from_bytes(data[i : i + 2], byteorder="big", signed=False))
        return regs

    def write_single_register(self, reg_addr: int, value: int) -> None:
        value = int(value)
        if not (0 <= value <= 0xFFFF):
            raise ValueError(f"Invalid register value: {value}")

        payload = struct.pack(">HH", int(reg_addr), value)
        frame = self._transact(0x06, payload)

        echo_addr, echo_value = struct.unpack(">HH", frame[2:6])
        if echo_addr != reg_addr or echo_value != value:
            raise GripperError(
                f"Write single register echo mismatch: "
                f"expect ({reg_addr}, {value}), got ({echo_addr}, {echo_value})"
            )

    def write_multiple_registers(self, start_addr: int, values: Sequence[int]) -> None:
        if not values:
            raise ValueError("values cannot be empty")
        if len(values) > 123:
            raise ValueError("At most 123 registers can be written at once")

        vals = [int(v) for v in values]
        for v in vals:
            if not (0 <= v <= 0xFFFF):
                raise ValueError(f"Invalid register value: {v}")

        count = len(vals)
        data = b"".join(struct.pack(">H", v) for v in vals)
        payload = struct.pack(">HHB", int(start_addr), count, len(data)) + data
        frame = self._transact(0x10, payload)

        echo_addr, echo_count = struct.unpack(">HH", frame[2:6])
        if echo_addr != start_addr or echo_count != count:
            raise GripperError(
                f"Write multiple registers echo mismatch: "
                f"expect ({start_addr}, {count}), got ({echo_addr}, {echo_count})"
            )


class InspireGripper:
    STATUS_TEXT = {
        1: "张开侧空闲/停止",
        2: "闭合侧空闲/停止",
        3: "停止",
        4: "正在闭合/夹取",
        5: "正在张开",
        6: "夹取过程中遇到力控停止",
    }

    def __init__(
        self,
        port: str,
        gripper_id: int = 1,
        baudrate: int = 115200,
        timeout: float = 0.03,
        max_open_mm: float = 70.0,
        speed_raw_min: int = 1,
        speed_raw_max: int = 1000,
        torque_raw_min: int = 50,
        torque_raw_max: int = 1000,
        reconnect_on_error: bool = True,
        reconnect_attempts: int = 1,
        reconnect_delay: float = 0.05,
        debug: bool = False,
    ):
        self.mb = InspireGripperModbus(
            port=port,
            slave_id=gripper_id,
            baudrate=baudrate,
            timeout=timeout,
            inter_command_delay=0.001,
            reconnect_on_error=reconnect_on_error,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
            debug=debug,
        )
        self.num_joints = 1
        self.joint_names = ["gripper_close"]

        self.max_open_mm = float(max_open_mm)
        self.speed_raw_min = int(speed_raw_min)
        self.speed_raw_max = int(speed_raw_max)
        self.torque_raw_min = int(torque_raw_min)
        self.torque_raw_max = int(torque_raw_max)

        self.last_joint_pos_cmd: Optional[float] = None
        self.last_speed_cmd: Optional[float] = None
        self.last_torque_cmd: Optional[float] = None
        self.last_state: Optional[GripperState] = None

        self._cfg_lock = threading.Lock()
        self._catch_mode_continuous: Optional[bool] = None

    def close(self):
        self.mb.close()

    def __enter__(self) -> "InspireGripper":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _sanitize_opening_raw(opening_raw: int) -> int:
        x = int(opening_raw)
        if -20 <= x <= 1020:
            return int(_clip(x, 0, 1000))
        raise GripperError(
            f"Abnormal opening_raw={x}, likely communication or device issue"
        )

    def _joint_pos_to_opening_raw(self, joint_pos: float) -> int:
        jp = float(_clip(joint_pos, 0.0, 1.0))
        opening = (1.0 - jp) * 1000.0
        return int(round(_clip(opening, 0.0, 1000.0)))

    def _opening_raw_to_joint_pos(self, opening_raw: int) -> float:
        op = self._sanitize_opening_raw(opening_raw)
        return float(_clip(1.0 - (op / 1000.0), 0.0, 1.0))

    def _norm_to_speed_raw(self, speed: float) -> int:
        s = float(_clip(speed, 0.0, 1.0))
        raw = self.speed_raw_min + s * (self.speed_raw_max - self.speed_raw_min)
        return int(round(_clip(raw, self.speed_raw_min, self.speed_raw_max)))

    def _speed_raw_to_norm(self, speed_raw: int) -> float:
        lo, hi = self.speed_raw_min, self.speed_raw_max
        x = float(_clip(speed_raw, lo, hi))
        return float((x - lo) / (hi - lo)) if hi != lo else 0.0

    def _norm_to_torque_raw(self, torque: float) -> int:
        t = float(_clip(torque, 0.0, 1.0))
        raw = self.torque_raw_min + t * (self.torque_raw_max - self.torque_raw_min)
        return int(round(_clip(raw, self.torque_raw_min, self.torque_raw_max)))

    def _torque_raw_to_norm(self, torque_raw: int) -> float:
        lo, hi = self.torque_raw_min, self.torque_raw_max
        x = float(_clip(torque_raw, lo, hi))
        return float((x - lo) / (hi - lo)) if hi != lo else 0.0

    def _opening_raw_to_mm(self, opening_raw: int) -> float:
        op = self._sanitize_opening_raw(opening_raw)
        return (op / 1000.0) * self.max_open_mm

    @staticmethod
    def _to_signed_16(v: int) -> int:
        return v - 0x10000 if (v & 0x8000) else v

    def clear_fault(self):
        self.mb.write_single_register(self.mb.REG_FAULT_ACK, 1)

    def estop(self):
        self.mb.write_single_register(self.mb.REG_STOP, 1)

    def save_parameters(self):
        self.mb.write_single_register(self.mb.REG_SAVE, 1)

    def restore_defaults(self):
        self.mb.write_single_register(self.mb.REG_SETDEFAULT, 1)

    def set_device_id(self, new_id: int, save: bool = False):
        if not (1 <= int(new_id) <= 254):
            raise ValueError(f"new_id must be in [1,254], got {new_id}")
        self.mb.write_single_register(self.mb.REG_ID, int(new_id))
        self.mb.slave_id = int(new_id)
        if save:
            self.save_parameters()

    def set_baudrate(self, baudrate: int, save: bool = False):
        if int(baudrate) not in self.mb.BAUD_TO_MODE:
            raise ValueError(f"Unsupported baudrate: {baudrate}")
        self.mb.write_single_register(
            self.mb.REG_BAUD,
            self.mb.BAUD_TO_MODE[int(baudrate)],
        )
        if save:
            self.save_parameters()

    def set_continuous_grasp(self, continuous: bool):
        self.mb.write_single_register(
            self.mb.REG_CATCH_MODE,
            1 if continuous else 0,
        )
        with self._cfg_lock:
            self._catch_mode_continuous = bool(continuous)

    def get_continuous_grasp(self) -> bool:
        val = self.mb.read_holding_registers(self.mb.REG_CATCH_MODE, 1)[0]
        return bool(val)

    def set_speed(self, speed):
        s = _ensure_scalar_or_len1(speed, "speed")
        s = float(_clip(s, 0.0, 1.0))
        raw = self._norm_to_speed_raw(s)
        self.mb.write_single_register(self.mb.REG_SPEED_SET, raw)
        self.last_speed_cmd = s

    def get_speed(self):
        raw = self.mb.read_holding_registers(self.mb.REG_SPEED_SET, 1)[0]
        return [self._speed_raw_to_norm(raw)]

    def set_torque(self, torque):
        t = _ensure_scalar_or_len1(torque, "torque")
        t = float(_clip(t, 0.0, 1.0))
        raw = self._norm_to_torque_raw(t)
        self.mb.write_single_register(self.mb.REG_FORCE_SET, raw)
        self.last_torque_cmd = t

    def get_torque(self):
        raw = self.mb.read_holding_registers(self.mb.REG_FORCE_SET, 1)[0]
        return [self._torque_raw_to_norm(raw)]

    def set_joint_pos(
        self,
        joint_pos,
        speed=None,
        torque=None,
        continuous: Optional[bool] = None,
        force_send: bool = False,
    ):
        jp = _ensure_scalar_or_len1(joint_pos, "joint_pos")
        jp = float(_clip(jp, 0.0, 1.0))
        target_opening_raw = self._joint_pos_to_opening_raw(jp)

        sp_norm = None
        tq_norm = None
        sp_raw = None
        tq_raw = None

        if speed is not None:
            sp_norm = float(_clip(_ensure_scalar_or_len1(speed, "speed"), 0.0, 1.0))
            sp_raw = self._norm_to_speed_raw(sp_norm)

        if torque is not None:
            tq_norm = float(_clip(_ensure_scalar_or_len1(torque, "torque"), 0.0, 1.0))
            tq_raw = self._norm_to_torque_raw(tq_norm)

        if continuous is not None:
            with self._cfg_lock:
                if force_send or self._catch_mode_continuous != bool(continuous):
                    self.mb.write_single_register(
                        self.mb.REG_CATCH_MODE,
                        1 if continuous else 0,
                    )
                    self._catch_mode_continuous = bool(continuous)

        if (
            (not force_send)
            and (self.last_joint_pos_cmd is not None)
            and (abs(self.last_joint_pos_cmd - jp) < 1e-6)
            and (speed is None)
            and (torque is None)
            and (continuous is None)
        ):
            return

        if sp_raw is not None and tq_raw is not None:
            self.mb.write_multiple_registers(
                self.mb.REG_OPENLEN_SET,
                [target_opening_raw, sp_raw, tq_raw],
            )
            self.last_speed_cmd = sp_norm
            self.last_torque_cmd = tq_norm
        else:
            if sp_raw is not None:
                self.mb.write_single_register(self.mb.REG_SPEED_SET, sp_raw)
                self.last_speed_cmd = sp_norm

            if tq_raw is not None:
                self.mb.write_single_register(self.mb.REG_FORCE_SET, tq_raw)
                self.last_torque_cmd = tq_norm

            self.mb.write_single_register(self.mb.REG_OPENLEN_SET, target_opening_raw)

        self.last_joint_pos_cmd = jp

    def read_joint_pos(self):
        opening_raw = self.mb.read_holding_registers(self.mb.REG_OPENLEN_ACT, 1)[0]
        pos = self._opening_raw_to_joint_pos(opening_raw)
        return [pos]

    def get_state(self) -> GripperState:
        regs = self.mb.read_holding_registers(self.mb.REG_OPENLEN_ACT, 5)

        opening_raw = self._sanitize_opening_raw(regs[0])
        current_raw = int(regs[1])
        current = self._to_signed_16(current_raw)
        temperature_c = int(regs[2])
        error_code = int(regs[3])
        status_code = int(regs[4])

        st = GripperState(
            timestamp=time.time(),
            joint_pos=self._opening_raw_to_joint_pos(opening_raw),
            is_moving=(status_code in (4, 5)),
            opening_raw=opening_raw,
            opening_mm=self._opening_raw_to_mm(opening_raw),
            current_raw=current_raw,
            current=current,
            temperature_c=temperature_c,
            error_code=error_code,
            status_code=status_code,
            status_text=self.STATUS_TEXT.get(
                status_code, f"未知状态({status_code})"
            ),
            commanded_joint_pos=self.last_joint_pos_cmd,
            commanded_speed=self.last_speed_cmd,
            commanded_torque=self.last_torque_cmd,
        )
        self.last_state = st
        return st

    def get_last_state(self) -> Optional[GripperState]:
        return self.last_state

    def open_hand(self):
        self.set_joint_pos(0.0)

    def close_hand(
        self,
        torque: Optional[float] = None,
        speed: Optional[float] = None,
        continuous: bool = False,
    ):
        self.set_joint_pos(1.0, speed=speed, torque=torque, continuous=continuous)

    def half_close_hand(self):
        self.set_joint_pos(0.5)

    def wait_until_idle(
        self,
        timeout: float = 3.0,
        poll_interval: float = 0.05,
        start_grace: float = 0.2,
    ) -> GripperState:
        timeout_s = max(0.0, float(timeout))
        poll_s = max(0.0, float(poll_interval))
        start_grace_s = max(0.0, float(start_grace))

        start_time = time.time()
        deadline = start_time + timeout_s
        grace_deadline = min(deadline, start_time + start_grace_s)
        last_state = self.get_state()
        observed_motion = bool(last_state.is_moving)

        while time.time() < deadline:
            if last_state.is_moving:
                observed_motion = True
            elif observed_motion or time.time() >= grace_deadline:
                return last_state

            sleep_s = min(poll_s, max(0.0, deadline - time.time()))
            if sleep_s <= 0.0:
                break
            time.sleep(sleep_s)
            last_state = self.get_state()

        return last_state

    def set_open_limits_raw(
        self,
        max_open_raw: int,
        min_open_raw: int,
        save: bool = False,
    ):
        max_open_raw = int(_clip(max_open_raw, 0, 1000))
        min_open_raw = int(_clip(min_open_raw, 0, 1000))

        if min_open_raw > max_open_raw:
            raise ValueError("min_open_raw cannot be greater than max_open_raw")

        self.mb.write_multiple_registers(
            self.mb.REG_MAX_OPENLEN,
            [max_open_raw, min_open_raw],
        )
        if save:
            self.save_parameters()

    def get_open_limits_raw(self):
        regs = self.mb.read_holding_registers(self.mb.REG_MAX_OPENLEN, 2)
        return regs[0], regs[1]

    def print_state(self, state: Optional[GripperState] = None):
        s = state if state is not None else self.get_state()

        print("========== Inspire Gripper State ==========")
        print(f"timestamp : {s.timestamp:.6f}")
        print(f"joint_pos : {s.joint_pos:.4f} (1=closed, 0=open)")
        print(f"is_moving : {s.is_moving}")
        print(f"opening_raw : {s.opening_raw}")
        print(f"opening_mm : {s.opening_mm:.2f}")
        print(f"current_raw : {s.current_raw}")
        print(f"current : {s.current}")
        print(f"temperature_c : {s.temperature_c}")
        print(f"error_code : 0x{s.error_code:04X}")
        print(f"status_code : {s.status_code}")
        print(f"status_text : {s.status_text}")
        print(f"commanded_joint_pos : {s.commanded_joint_pos}")
        print(f"commanded_speed : {s.commanded_speed}")
        print(f"commanded_torque : {s.commanded_torque}")
        print("===========================================")


def list_serial_devices() -> list[dict]:
    ports = []
    for port in list_ports.comports():
        aliases: list[str] = []
        try:
            device_path = Path(str(port.device))
            if device_path.exists():
                realpath = str(device_path.resolve())
                for alias_dir in (Path("/dev/serial/by-id"), Path("/dev/serial/by-path")):
                    if not alias_dir.exists():
                        continue
                    for candidate in sorted(alias_dir.iterdir()):
                        try:
                            if str(candidate.resolve()) == realpath:
                                aliases.append(str(candidate))
                        except OSError:
                            continue
        except OSError:
            pass
        permissions = {}
        try:
            stat = os.stat(port.device)
            permissions = {
                "mode": oct(stat.st_mode & 0o777),
                "uid": stat.st_uid,
                "gid": stat.st_gid,
                "readable": os.access(port.device, os.R_OK),
                "writable": os.access(port.device, os.W_OK),
            }
        except OSError:
            permissions = {
                "readable": False,
                "writable": False,
            }
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
                "manufacturer": port.manufacturer,
                "product": port.product,
                "serial_number": port.serial_number,
                "vid": port.vid,
                "pid": port.pid,
                "location": getattr(port, "location", None),
                "interface": getattr(port, "interface", None),
                "usb_aliases": aliases,
                "permissions": permissions,
            }
        )
    return ports


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspire gripper Modbus driver and debug CLI"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--id", type=int, default=1, dest="gripper_id")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.03)
    parser.add_argument("--max_open_mm", type=float, default=70.0)
    parser.add_argument(
        "--no-reconnect",
        action="store_true",
        help="Disable one-shot serial reconnect/retry after transient Modbus I/O errors",
    )
    parser.add_argument("--reconnect-attempts", type=int, default=1)
    parser.add_argument("--reconnect-delay", type=float, default=0.05)
    parser.add_argument("--debug", action="store_true", help="Print raw Modbus TX/RX frames")
    parser.add_argument("--json", action="store_true", help="Print state as JSON")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list-ports", help="List serial ports and exit")
    sub.add_parser("state", help="Read and print current gripper state")
    sub.add_parser("clear-fault", help="Acknowledge/clear gripper fault")
    sub.add_parser("stop", help="Send stop register command")
    sub.add_parser("open", help="Open gripper")
    sub.add_parser("half", help="Move gripper to joint_pos=0.5")

    close_p = sub.add_parser("close", help="Close gripper")
    close_p.add_argument("--speed", type=float, default=None)
    close_p.add_argument("--torque", type=float, default=None)
    close_p.add_argument("--continuous", action="store_true")

    move_p = sub.add_parser("move", help="Move to normalized joint position")
    move_p.add_argument("joint_pos", type=float, help="0=open, 1=closed")
    move_p.add_argument("--speed", type=float, default=None)
    move_p.add_argument("--torque", type=float, default=None)
    move_p.add_argument("--continuous", action="store_true")

    for p in (sub.choices["open"], sub.choices["half"], close_p, move_p):
        p.add_argument("--wait", action="store_true", help="Wait until gripper stops moving")
        p.add_argument("--wait_timeout", type=float, default=3.0)

    set_speed_p = sub.add_parser("set-speed", help="Set normalized speed [0,1]")
    set_speed_p.add_argument("speed", type=float)

    set_torque_p = sub.add_parser("set-torque", help="Set normalized torque [0,1]")
    set_torque_p.add_argument("torque", type=float)

    sub.add_parser("get-speed", help="Read normalized speed setting")
    sub.add_parser("get-torque", help="Read normalized torque setting")

    return parser


def _print_state(state: GripperState, as_json: bool) -> None:
    if as_json:
        print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
        return

    s = state
    print("========== Inspire Gripper State ==========")
    print(f"timestamp : {s.timestamp:.6f}")
    print(f"joint_pos : {s.joint_pos:.4f} (1=closed, 0=open)")
    print(f"is_moving : {s.is_moving}")
    print(f"opening_raw : {s.opening_raw}")
    print(f"opening_mm : {s.opening_mm:.2f}")
    print(f"current_raw : {s.current_raw}")
    print(f"current : {s.current}")
    print(f"temperature_c : {s.temperature_c}")
    print(f"error_code : 0x{s.error_code:04X}")
    print(f"status_code : {s.status_code}")
    print(f"status_text : {s.status_text}")
    print(f"commanded_joint_pos : {s.commanded_joint_pos}")
    print(f"commanded_speed : {s.commanded_speed}")
    print(f"commanded_torque : {s.commanded_torque}")
    print("===========================================")


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "list-ports":
        print(json.dumps(list_serial_devices(), ensure_ascii=False, indent=2))
        return

    command = args.command or "state"

    with InspireGripper(
        port=args.port,
        gripper_id=args.gripper_id,
        baudrate=args.baudrate,
        timeout=args.timeout,
        max_open_mm=args.max_open_mm,
        reconnect_on_error=not bool(args.no_reconnect),
        reconnect_attempts=args.reconnect_attempts,
        reconnect_delay=args.reconnect_delay,
        debug=args.debug,
    ) as gripper:
        if command == "state":
            _print_state(gripper.get_state(), args.json)
        elif command == "clear-fault":
            gripper.clear_fault()
            _print_state(gripper.get_state(), args.json)
        elif command == "stop":
            gripper.estop()
            _print_state(gripper.get_state(), args.json)
        elif command == "open":
            gripper.open_hand()
            state = gripper.wait_until_idle(args.wait_timeout) if args.wait else gripper.get_state()
            _print_state(state, args.json)
        elif command == "half":
            gripper.half_close_hand()
            state = gripper.wait_until_idle(args.wait_timeout) if args.wait else gripper.get_state()
            _print_state(state, args.json)
        elif command == "close":
            gripper.close_hand(
                speed=args.speed,
                torque=args.torque,
                continuous=args.continuous,
            )
            state = gripper.wait_until_idle(args.wait_timeout) if args.wait else gripper.get_state()
            _print_state(state, args.json)
        elif command == "move":
            gripper.set_joint_pos(
                args.joint_pos,
                speed=args.speed,
                torque=args.torque,
                continuous=args.continuous,
            )
            state = gripper.wait_until_idle(args.wait_timeout) if args.wait else gripper.get_state()
            _print_state(state, args.json)
        elif command == "set-speed":
            gripper.set_speed(args.speed)
            print("speed =", gripper.get_speed())
        elif command == "set-torque":
            gripper.set_torque(args.torque)
            print("torque =", gripper.get_torque())
        elif command == "get-speed":
            print("speed =", gripper.get_speed())
        elif command == "get-torque":
            print("torque =", gripper.get_torque())
        else:
            parser.error(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()

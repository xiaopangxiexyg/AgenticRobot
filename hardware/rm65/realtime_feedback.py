"""Passive, single-owner RealMan JSON UDP feedback; no controller operations.

Wire units follow the vendor UDP protocol, not the SDK callback's float units:
https://develop.realman-robotics.com/robot/json/udpConfig/
Receive timestamps describe kernel arrival, not controller acquisition time.
"""

from __future__ import annotations

import ipaddress
import json
import math
import socket
import struct
import sys
import threading

from utils.owned_thread import OwnedThread
import time
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RM65RealtimeFeedback:
    side: str
    controller_ip: str
    sequence: int
    joint_deg: tuple[float, ...]
    pose_6d: tuple[float, ...]
    raw_force: tuple[float, ...]
    external_force: tuple[float, ...]
    force_coordinate: int
    joint_enabled: tuple[int, ...]
    errors: tuple[int, ...]
    wall_time_sec: float
    monotonic_time_sec: float

    def require_healthy(self) -> None:
        if any(self.errors) or not all(self.joint_enabled):
            raise RuntimeError(f"{self.side} UDP reports controller/joint error or disabled joints")

    def force_sources(self) -> dict[str, tuple[float, ...]]:
        # Never relabel sensor/work-frame data as tool-frame protection.
        source = ("zero", "work", "tool")[self.force_coordinate]
        return {"raw": self.raw_force, source: self.external_force}


def _wire_vector(value: object, *, name: str, count: int, scale: float = 1.) -> tuple[float, ...]:
    if (not isinstance(value, list) or len(value) != count
            or any(isinstance(x, bool) or not isinstance(x, int) for x in value)):
        raise ValueError(f"UDP {name} requires exactly {count} wire integers")
    result = tuple(x * scale for x in value)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"UDP {name} must be finite")
    return result


def decode_rm65_feedback(
    payload: bytes, *, side: str, controller_ip: str, sequence: int,
    wall_time_sec: float, monotonic_time_sec: float,
) -> RM65RealtimeFeedback:
    """Decode the observed six-joint JSON protocol without magnitude heuristics."""
    value = json.loads(payload.rstrip(b"\x00\r\n"))
    if not isinstance(value, Mapping) or value.get("state") != "realtime_arm_joint_state":
        raise ValueError("UDP payload is not realtime_arm_joint_state")
    joint, pose, force = (value.get(key) for key in ("joint_status", "waypoint", "six_force_sensor"))
    if not all(isinstance(part, Mapping) for part in (joint, pose, force)):
        raise ValueError("UDP feedback requires joint, waypoint and six-force fields")
    coordinate = force.get("coordinate")
    if isinstance(coordinate, bool) or not isinstance(coordinate, int) or coordinate not in (0, 1, 2):
        raise ValueError("UDP force coordinate must be sensor=0, work=1 or tool=2")
    enabled = _wire_vector(joint.get("joint_en_flag"), name="joint_en_flag", count=6)
    if any(flag not in (0, 1) for flag in enabled):
        raise ValueError("UDP joint enable flags must be 0 or 1")
    errors = value.get("err")
    if isinstance(errors, int) and not isinstance(errors, bool):
        errors = [errors]
    if not isinstance(errors, list) or not errors:
        raise ValueError("UDP controller err field is missing")
    controller_errors = _wire_vector(errors, name="err", count=len(errors))
    joint_errors = _wire_vector(joint.get("joint_err_code"), name="joint_err_code", count=6)
    if (not math.isfinite(wall_time_sec) or wall_time_sec <= 0
            or not math.isfinite(monotonic_time_sec) or monotonic_time_sec <= 0
            or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0):
        raise ValueError("UDP receive timestamps/sequence are invalid")
    return RM65RealtimeFeedback(
        side=side, controller_ip=controller_ip, sequence=sequence,
        joint_deg=_wire_vector(joint.get("joint_position"), name="joint_position", count=6, scale=.001),
        pose_6d=(*_wire_vector(pose.get("position"), name="position", count=3, scale=.000001),
                 *_wire_vector(pose.get("euler"), name="euler", count=3, scale=.001)),
        raw_force=_wire_vector(force.get("force"), name="force", count=6, scale=.001),
        external_force=_wire_vector(force.get("zero_force"), name="zero_force", count=6, scale=.001),
        force_coordinate=int(coordinate), joint_enabled=tuple(int(x) for x in enabled),
        errors=tuple(int(x) for x in (*controller_errors, *joint_errors)),
        wall_time_sec=wall_time_sec, monotonic_time_sec=monotonic_time_sec,
    )


class RM65UDPFeedbackReceiver:
    """One socket, explicit source-IP demultiplexing and latest immutable cuts.

    Linux kernel timestamps preserve age across queued datagrams and Python/GIL
    delays. SDK callbacks are deliberately not registered: they are process-wide
    and would compete for ownership of the configured UDP receive port.
    """

    def __init__(self, *, arm_ip_by_side: Mapping[str, str], port: int,
                 bind_ip: str = "0.0.0.0") -> None:
        if set(arm_ip_by_side) != {"left", "right"}:
            raise ValueError("UDP feedback requires exact left/right controller addresses")
        ips = {side: str(ipaddress.IPv4Address(ip)) for side, ip in arm_ip_by_side.items()}
        if len(set(ips.values())) != 2:
            raise ValueError("UDP controller addresses must be distinct")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("UDP receive port must be in 1..65535")
        self.port, self.bind_ip = port, str(ipaddress.IPv4Address(bind_ip))
        self._by_ip = {ip: side for side, ip in ips.items()}
        self._condition = threading.Condition()
        self._latest: dict[str, RM65RealtimeFeedback] = {}
        self._invalid: dict[str, str] = {}
        self._counts = {side: 0 for side in ips}
        self._invalid_count = {side: 0 for side in ips}
        self._unknown_count = 0
        self._fault: str | None = None
        self._socket: socket.socket | None = None
        self._thread: OwnedThread | None = None
        self._stop = threading.Event()
        self._closed = False
        self._started = False

    def start(self) -> None:
        if self._closed or self._started:
            raise RuntimeError("UDP receiver has already started or closed")
        self._started = True
        if sys.platform != "linux" or struct.calcsize("@ll") != 16:
            raise RuntimeError("UDP feedback requires Linux 64-bit kernel receive timestamps")
        # SO_TIMESTAMPNS_OLD is 35 on the supported Linux 64-bit runtime;
        # Python 3.10 does not consistently export this socket constant.
        self._timestamp_option = getattr(socket, "SO_TIMESTAMPNS", 35)
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = receiver
        try:
            receiver.setsockopt(socket.SOL_SOCKET, self._timestamp_option, 1)
            # Never share/reuse a port owned by teleop, a probe or another SDK.
            receiver.bind((self.bind_ip, self.port))
            receiver.settimeout(.05)
            self._thread = OwnedThread(target=self._run, name="rm65-udp-feedback", daemon=True)
            self._thread.start()
        except Exception as start_error:
            try:
                self.close()
            except Exception as close_error:
                raise RuntimeError(
                    f"UDP receiver startup failed: {start_error}; "
                    f"startup rollback incomplete: {close_error}"
                ) from start_error
            raise

    def _run(self) -> None:
        receiver = self._socket
        assert receiver is not None
        try:
            while not self._stop.is_set():
                try:
                    payload, ancillary, flags, source = receiver.recvmsg(65535, socket.CMSG_SPACE(16))
                except TimeoutError:
                    continue
                side = self._by_ip.get(source[0])
                if side is None:
                    with self._condition:
                        self._unknown_count += 1
                    continue
                # Sample monotonic first so a scheduling gap can only age the
                # packet conservatively, never move its arrival time forward.
                monotonic = time.monotonic()
                wall = time.time()
                try:
                    timestamps = [data for level, kind, data in ancillary
                        if level == socket.SOL_SOCKET and kind == self._timestamp_option]
                    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(timestamps) != 1:
                        raise ValueError("UDP packet lacks an intact kernel receive timestamp")
                    seconds, nanoseconds = struct.unpack("@ll", timestamps[0])
                    stamp = seconds + nanoseconds * 1e-9
                    if not 0 <= nanoseconds < 1_000_000_000 or stamp > wall:
                        raise ValueError("UDP kernel receive time is invalid/future")
                    with self._condition:
                        self._counts[side] += 1
                        sequence = self._counts[side]
                    sample = decode_rm65_feedback(payload, side=side, controller_ip=source[0],
                        sequence=sequence, wall_time_sec=stamp,
                        monotonic_time_sec=monotonic - (wall - stamp))
                    with self._condition:
                        self._latest[side] = sample
                        self._invalid.pop(side, None)
                        self._condition.notify_all()
                except (ValueError, TypeError, KeyError, OverflowError, struct.error) as error:
                    with self._condition:
                        self._invalid_count[side] += 1
                        self._invalid[side] = str(error)[:240]
                        self._condition.notify_all()
        except Exception as error:  # noqa: BLE001 - receiver failures must reach safety consumers.
            if not self._stop.is_set():
                with self._condition:
                    self._fault = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()

    def snapshot(self, side: str) -> RM65RealtimeFeedback:
        if side not in {"left", "right"}:
            raise ValueError("UDP side must be left or right")
        with self._condition:
            if self._closed or self._fault is not None:
                raise RuntimeError(f"UDP receiver unavailable: {self._fault or 'closed'}")
            if self._started and (self._thread is None or not self._thread.is_alive()):
                raise RuntimeError("UDP feedback receiver thread is not alive")
            if side in self._invalid:
                raise RuntimeError(f"{side} UDP feedback invalid: {self._invalid[side]}")
            sample = self._latest.get(side)
            if sample is None:
                raise RuntimeError(f"{side} UDP feedback has not arrived")
            sample.require_healthy()
            return sample

    def wait_ready(self, *, timeout_sec: float, max_age_sec: float) -> None:
        """Require tool-frame external force before any SDK/motion lifecycle."""
        if any(not math.isfinite(x) or x <= 0 for x in (timeout_sec, max_age_sec)):
            raise ValueError("UDP readiness timing values must be finite and positive")
        deadline = time.monotonic() + timeout_sec
        last_error = "no packets"
        with self._condition:
            while not self._stop.is_set():
                try:
                    for side in ("left", "right"):
                        sample = self.snapshot(side)
                        age = time.monotonic() - sample.monotonic_time_sec
                        if age < 0 or age > max_age_sec:
                            raise RuntimeError(f"{side} UDP sample stale/invalid age: {age:.3f}s")
                        if sample.force_coordinate != 2:
                            raise RuntimeError(f"{side} UDP force_coordinate={sample.force_coordinate}; "
                                "tool-frame force_coordinate=2 is required; no automatic configuration")
                    return
                except RuntimeError as error:
                    last_error = str(error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"UDP startup feedback not ready: {last_error}")
                self._condition.wait(min(.05, remaining))
        raise RuntimeError("UDP receiver closed before startup feedback was ready")

    def diagnostics(self) -> dict[str, object]:
        with self._condition:
            return {"transport": "udp", "port": self.port,
                "timestamp_basis": "kernel_receive_not_controller_acquisition",
                "packet_count_by_side": dict(self._counts),
                "invalid_packet_count_by_side": dict(self._invalid_count),
                "unknown_source_count": self._unknown_count, "fault": self._fault,
                "thread_alive": self._thread is not None and self._thread.is_alive(),
                "closed": self._closed}

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            if not self._thread.join_and_confirm(timeout=1.):
                raise RuntimeError("UDP feedback receiver thread did not stop")
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        with self._condition:
            self._closed = True
            self._condition.notify_all()

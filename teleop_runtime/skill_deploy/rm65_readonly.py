"""Session-owned SDK state sampling and calibrated FK, with no command port."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import pose_matrix_to_xyzquat

from .rm65_lower_assembly import build_exclusive_rm65_skill_target_builder
from .rm65_profile import ResolvedExclusiveRM65SkillDeploymentProfile

RM65_READONLY_POLL_INTERVAL_SEC = 0.05
RM65_READONLY_MAX_CUT_AGE_SEC = 0.2


class ReadonlyRM65Connection(Protocol):
    def Get_Current_Arm_State(self, retry: int = 0): ...

    def Arm_Socket_Close(self): ...


def assert_readonly_rm65_connections_idle(ips: Mapping[str, str]) -> None:
    connections = subprocess.run(
        ["/usr/bin/ss", "-tnH", "state", "established"],
        capture_output=True, text=True, check=True, timeout=3,
    ).stdout
    for line in connections.splitlines():
        if any(f"{ip}:8080" in line for ip in ips.values()):
            raise RuntimeError(f"An existing RM65 connection is present: {line}")


def open_readonly_rm65(ip: str) -> ReadonlyRM65Connection:
    # Construct only a raw SDK connection, never an ArmBackend or control runtime.
    from hardware.rm65.rm_api2_adapter import RM65, Arm

    arm = Arm(RM65, ip)
    if arm.Arm_Socket_State() != 0:
        arm.Arm_Socket_Close()
        raise RuntimeError(f"RM65 read-only connection failed: {ip}")
    return arm


@dataclass(frozen=True)
class RM65ReadonlyTcpSample:
    sequence: int
    measured_at: float
    available_at: float
    qpos_by_side: Mapping[str, tuple[float, ...]]
    pose_by_component: Mapping[str, tuple[float, ...]]
    read_windows_by_side: Mapping[str, tuple[float, float]]

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__, domain="agenticrobot.rm65.readonly_tcp_sample.v1"
        )


class RM65ReadonlyTcpSampler:
    """One worker owns handles, polling and FK; consumers only read its cache.

    A failure latches and stops this session. There is no reconnect, control
    stream, force monitor, SDK tool-frame write, HOLD or STOP on any path.
    """

    def __init__(
        self,
        profile: ResolvedExclusiveRM65SkillDeploymentProfile,
        *,
        arm_factory: Callable[[str], ReadonlyRM65Connection] = open_readonly_rm65,
        target_builder=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile = profile
        self.target_builder = (
            build_exclusive_rm65_skill_target_builder(profile)
            if target_builder is None else target_builder
        )
        self.arm_factory = arm_factory
        self.clock = clock
        self._station_from_world = np.linalg.inv(profile.scene_world_from_canonical_station)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: RM65ReadonlyTcpSample | None = None
        self._fault: BaseException | None = None
        self._cleanup_errors: list[str] = []

    @property
    def latest(self) -> RM65ReadonlyTcpSample:
        with self._lock:
            if self._fault is not None:
                raise RuntimeError("Read-only RM65 sampler failed") from self._fault
            if self._stop.is_set() or self._latest is None:
                raise RuntimeError("Read-only RM65 sampler is not publishing")
            return self._latest

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Read-only RM65 sampler starts only once")
        try:
            assert_readonly_rm65_connections_idle(self.profile.arm_ip_by_side)
        except BaseException as error:
            with self._lock:
                self._fault = error
            self._stop.set()
            raise
        self._thread = threading.Thread(target=self._run, name="rm65-readonly-tcp", daemon=False)
        try:
            self._thread.start()
        except BaseException as error:
            with self._lock:
                self._fault = error
            self._stop.set()
            raise
        if not self._ready.wait(5.0):
            error = TimeoutError("Read-only RM65 sampler did not publish within 5s")
            with self._lock:
                if self._fault is None:
                    self._fault = error
            self._stop.set()
            raise error
        _ = self.latest

    def _sample(self, arms, sequence: int) -> RM65ReadonlyTcpSample | None:
        qpos, windows = {}, {}
        previous = self._latest
        last_timestamp = None if previous is None else previous.available_at
        for side, arm in arms.items():
            if self._stop.is_set():
                return None
            started = self.clock()
            if not math.isfinite(started) or (last_timestamp is not None and started < last_timestamp):
                raise RuntimeError("Read-only SDK measurement clock is invalid or regressed")
            status, degrees, _sdk_pose, arm_error, system_error = arm.Get_Current_Arm_State(retry=0)
            finished = self.clock()
            if self._stop.is_set():
                return None
            if not math.isfinite(finished) or finished < started:
                raise RuntimeError("Read-only SDK measurement clock is invalid or regressed")
            values = np.deg2rad(degrees)
            if status or arm_error or system_error or values.shape != (6,) or not np.isfinite(values).all():
                raise RuntimeError(
                    f"{side} invalid SDK state: status={status}, arm_error={arm_error}, system_error={system_error}"
                )
            qpos[side] = tuple(float(x) for x in values)
            windows[side] = (started, finished)
            last_timestamp = finished
        world_poses = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"], right_qpos=qpos["right"]
        )
        poses = {
            f"{side}_arm": tuple(float(x) for x in pose_matrix_to_xyzquat(
                self._station_from_world @ world_poses[side], name=f"{side}_tcp"
            )) for side in ("left", "right")
        }
        # The SDK does not supply a sensor timestamp. Use the earliest request
        # start, not the return time, so SDK waiting and dual-arm skew age the cut.
        measured = min(window[0] for window in windows.values())
        available = self.clock()
        if (not math.isfinite(available) or available < last_timestamp
                or available - measured > RM65_READONLY_MAX_CUT_AGE_SEC):
            raise RuntimeError("Read-only dual-arm SDK/FK cut exceeds 200ms or has invalid time")
        return RM65ReadonlyTcpSample(
            sequence, measured, available, MappingProxyType(qpos),
            MappingProxyType(poses), MappingProxyType(windows)
        )

    def _run(self) -> None:
        arms = {}
        try:
            for side in ("left", "right"):
                if self._stop.is_set():
                    return
                arms[side] = self.arm_factory(self.profile.arm_ip_by_side[side])
            sequence = 0
            while not self._stop.is_set():
                sequence += 1
                sample = self._sample(arms, sequence)
                if sample is None:
                    break
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._latest = sample
                self._ready.set()
                # No catch-up bursts after a slow read; there are no commands.
                self._stop.wait(RM65_READONLY_POLL_INTERVAL_SEC)
        except BaseException as error:  # noqa: BLE001 - worker faults are re-raised by the consumer.
            with self._lock:
                self._fault = error
            self._stop.set()
            self._ready.set()
        finally:
            for side, arm in arms.items():
                try:
                    code = arm.Arm_Socket_Close()
                    if code not in (None, 0):
                        raise RuntimeError(f"SDK close returned {code}")
                except Exception as error:  # noqa: BLE001 - retain every close failure for the session owner.
                    self._cleanup_errors.append(f"{side}: {type(error).__name__}: {error}")
            self._ready.set()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                # Never destroy a handle concurrently with an in-flight SDK read.
                raise RuntimeError("Read-only RM65 worker still has in-flight SDK I/O")
        if self._cleanup_errors:
            raise RuntimeError("Read-only SDK cleanup failed: " + "; ".join(self._cleanup_errors))

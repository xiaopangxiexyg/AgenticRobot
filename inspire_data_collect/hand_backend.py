from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import EffectorCommand


@dataclass
class InspireTacHandBackendConfig:
    hand_ip: str
    side: str
    effector_name: str | None = None
    port: int = 6000
    connect_hardware: bool = True
    max_fingertip_force: int = 800
    send_threshold: float = 0.002
    enable_telemetry: bool = False
    telemetry_hz: float = 10.0
    tactile_regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if float(self.telemetry_hz) <= 0.0:
            raise ValueError("telemetry_hz must be positive.")


class InspireTacHandBackend:
    def __init__(self, config: InspireTacHandBackendConfig):
        self.config = config
        self.hand: Any | None = None
        self._lock = threading.Lock()
        self._last_sent_command: np.ndarray | None = None
        self._last_sent_command_timestamp: float | None = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_joint_timestamp: float | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_timestamp: float | None = None
        self._latest_tactile_dict: dict[str, np.ndarray] | None = None
        self._latest_telemetry_timestamp: float | None = None
        self._telemetry_error: str | None = None
        self._stop_event: threading.Event | None = None
        self._telemetry_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.hand is not None:
            return
        from .inspire_tac_modbus import InspireTacModbus

        try:
            self.hand = InspireTacModbus(
                ip=self.config.hand_ip,
                port=int(self.config.port),
                max_fingertip_force=int(self.config.max_fingertip_force),
            )
        except Exception as exc:
            self.hand = None
            self._telemetry_error = f"connect_failed: {exc}"
            return
        self._telemetry_error = None
        if self.config.enable_telemetry:
            self._stop_event = threading.Event()
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name=f"{self.name}_telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or (self.hand is not None)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for '{command.effector_name}'.")
        target = np.asarray(command.value, dtype=np.float32).reshape(-1)
        if target.shape != (6,):
            raise ValueError(f"{self.name} expects q6 shape (6,), got {target.shape}.")
        target = np.clip(target, 0.0, 1.0)
        if self._last_sent_command is not None:
            max_delta = float(np.max(np.abs(target - self._last_sent_command)))
            if max_delta < float(self.config.send_threshold):
                return
        command_time = time.time()
        if self.hand is not None:
            with self._lock:
                self.hand.set_joint_pos(target.tolist())
        self._latest_joint_pos = target.copy()
        self._last_sent_command = target.copy()
        self._last_sent_command_timestamp = command_time

    @staticmethod
    def _normalize_joint_pos(joint_pos: object) -> np.ndarray:
        q6 = np.asarray(joint_pos, dtype=float).reshape(-1)
        if q6.shape != (6,) or not np.all(np.isfinite(q6)):
            raise ValueError(f"read_joint_pos() returned invalid q6 shape {q6.shape}.")
        return np.clip(q6, 0.0, 1.0)

    def read_joint_pos_now(self) -> np.ndarray | None:
        if self.hand is None:
            return None
        request_started_at = time.time()
        with self._lock:
            joint_pos = self.hand.read_joint_pos()
        normalized = self._normalize_joint_pos(joint_pos)
        self._latest_joint_pos = normalized.copy()
        self._latest_joint_timestamp = request_started_at
        self._latest_telemetry_timestamp = time.time()
        self._telemetry_error = None
        return normalized.copy()

    def _telemetry_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                with self._lock:
                    if self.hand is None:
                        return
                    joint_request_started_at = time.time()
                    self._latest_joint_pos = self._normalize_joint_pos(self.hand.read_joint_pos())
                    self._latest_joint_timestamp = joint_request_started_at
                    tactile_request_started_at = time.time()
                    if self.config.tactile_regions:
                        tactile_vec, tactile_dict = self.hand.read_tac_data(
                            target_regions=list(self.config.tactile_regions)
                        )
                    else:
                        tactile_vec, tactile_dict = self.hand.read_tac_data()
                self._latest_tactile = np.asarray(tactile_vec, dtype=float)
                self._latest_tactile_timestamp = tactile_request_started_at
                self._latest_tactile_dict = {
                    str(name): np.asarray(value, dtype=float)
                    for name, value in tactile_dict.items()
                }
                self._latest_telemetry_timestamp = time.time()
                self._telemetry_error = None
            except Exception as exc:
                self._telemetry_error = str(exc)
            remaining = (1.0 / float(self.config.telemetry_hz)) - (time.time() - loop_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def get_state(self) -> dict[str, object]:
        return {
            "last_sent_command": None
            if self._last_sent_command is None
            else self._last_sent_command.copy(),
            "last_sent_command_timestamp": self._last_sent_command_timestamp,
            "joint_pos": None if self._latest_joint_pos is None else self._latest_joint_pos.copy(),
            "joint_timestamp": self._latest_joint_timestamp,
            "tactile": None if self._latest_tactile is None else self._latest_tactile.copy(),
            "tactile_timestamp": self._latest_tactile_timestamp,
            "tactile_dict": None
            if self._latest_tactile_dict is None
            else {name: value.copy() for name, value in self._latest_tactile_dict.items()},
            "telemetry_timestamp": self._latest_telemetry_timestamp,
            "tactile_regions": tuple(self.config.tactile_regions),
            "telemetry_error": self._telemetry_error,
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)
            self._telemetry_thread = None
        if self.hand is not None:
            with self._lock:
                self.hand.close()
            self.hand = None
        self._stop_event = None

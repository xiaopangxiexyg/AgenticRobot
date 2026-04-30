from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common import EffectorCommand
from .hand_mapping import parse_tactile_regions


DEFAULT_INSPIRE_HAND_MAX_FORCE = 800


@dataclass
class InspireTacHandBackendConfig:
    hand_ip: str
    side: str
    effector_name: str | None = None
    port: int = 6000
    connect_hardware: bool = True
    max_fingertip_force: int = DEFAULT_INSPIRE_HAND_MAX_FORCE
    send_threshold: float = 0.002
    enable_telemetry: bool = False
    telemetry_hz: float = 10.0
    tactile_regions: tuple[str, ...] = ()
    modbus_timeout_sec: float = 0.5
    reconnect_on_error: bool = True
    reconnect_cooldown_sec: float = 1.0

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if int(self.max_fingertip_force) < 0:
            raise ValueError("max_fingertip_force must be non-negative.")
        if float(self.telemetry_hz) <= 0.0:
            raise ValueError("telemetry_hz must be positive.")
        if float(self.modbus_timeout_sec) <= 0.0:
            raise ValueError("modbus_timeout_sec must be positive.")
        if float(self.reconnect_cooldown_sec) < 0.0:
            raise ValueError("reconnect_cooldown_sec must be non-negative.")
        self.tactile_regions = parse_tactile_regions(self.tactile_regions)


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
        self._last_reconnect_attempt_at: float | None = None
        self._reconnect_count = 0

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _create_hand(self) -> Any:
        from .inspire_tac_modbus import InspireTacModbus

        return InspireTacModbus(
            ip=self.config.hand_ip,
            port=int(self.config.port),
            max_fingertip_force=int(self.config.max_fingertip_force),
            timeout=float(self.config.modbus_timeout_sec),
        )

    def _close_hand_locked(self) -> None:
        if self.hand is None:
            return
        try:
            self.hand.close()
        finally:
            self.hand = None

    def _connect_locked(self) -> bool:
        try:
            self.hand = self._create_hand()
        except Exception as exc:
            self.hand = None
            self._telemetry_error = f"connect_failed: {exc}"
            return False
        self._telemetry_error = None
        return True

    def _handle_io_error_locked(self, prefix: str, exc: BaseException) -> bool:
        error_text = f"{prefix}: {exc}"
        self._telemetry_error = error_text
        self._close_hand_locked()
        if not self.config.connect_hardware or not self.config.reconnect_on_error:
            return False
        now = time.time()
        if (
            self._last_reconnect_attempt_at is not None
            and now - self._last_reconnect_attempt_at < float(self.config.reconnect_cooldown_sec)
        ):
            return False
        self._last_reconnect_attempt_at = now
        try:
            self.hand = self._create_hand()
        except Exception as reconnect_exc:
            self.hand = None
            self._telemetry_error = f"{error_text}; reconnect_failed: {reconnect_exc}"
            return False
        self._reconnect_count += 1
        self._telemetry_error = None
        return True

    def start(self) -> None:
        if not self.config.connect_hardware or self.hand is not None:
            return
        with self._lock:
            connected = self._connect_locked()
        if not connected:
            return
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
        sent = False
        if self.hand is not None:
            with self._lock:
                if self.hand is None:
                    return
                try:
                    self.hand.set_joint_pos(target.tolist())
                except Exception as exc:
                    reconnected = self._handle_io_error_locked("send_failed", exc)
                    if reconnected and self.hand is not None:
                        try:
                            self.hand.set_joint_pos(target.tolist())
                        except Exception as retry_exc:
                            self._handle_io_error_locked("send_retry_failed", retry_exc)
                            return
                    else:
                        return
                sent = True
        elif self.config.connect_hardware:
            with self._lock:
                self._handle_io_error_locked(
                    "send_no_connection",
                    RuntimeError("hand is not connected"),
                )
                if self.hand is not None:
                    try:
                        self.hand.set_joint_pos(target.tolist())
                    except Exception as exc:
                        self._handle_io_error_locked("send_retry_failed", exc)
                        return
                    sent = True
                else:
                    return
        else:
            sent = True
        if sent:
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
        try:
            with self._lock:
                if self.hand is None:
                    return None
                joint_pos = self.hand.read_joint_pos()
        except Exception as exc:
            with self._lock:
                self._handle_io_error_locked("read_joint_pos_failed", exc)
            return None
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
            hand_missing = False
            tactile_regions = tuple(self.config.tactile_regions)
            tactile_request_started_at = None
            tactile_vec = None
            tactile_dict = None
            try:
                with self._lock:
                    if self.hand is None:
                        if self.config.connect_hardware and self.config.reconnect_on_error:
                            self._handle_io_error_locked(
                                "telemetry_no_connection",
                                RuntimeError("hand is not connected"),
                            )
                        hand_missing = self.hand is None
                    if self.hand is not None:
                        joint_request_started_at = time.time()
                        self._latest_joint_pos = self._normalize_joint_pos(
                            self.hand.read_joint_pos()
                        )
                        self._latest_joint_timestamp = joint_request_started_at
                        if tactile_regions:
                            tactile_request_started_at = time.time()
                            tactile_vec, tactile_dict = self.hand.read_tac_data(
                                target_regions=list(tactile_regions)
                            )
                if hand_missing:
                    remaining = (1.0 / float(self.config.telemetry_hz)) - (time.time() - loop_start)
                    if remaining > 0.0:
                        self._stop_event.wait(remaining)
                    continue
                if tactile_vec is not None and tactile_dict is not None:
                    self._latest_tactile = np.asarray(tactile_vec, dtype=float)
                    self._latest_tactile_timestamp = tactile_request_started_at
                    self._latest_tactile_dict = {
                        str(name): np.asarray(value, dtype=float)
                        for name, value in tactile_dict.items()
                    }
                else:
                    self._latest_tactile = None
                    self._latest_tactile_timestamp = None
                    self._latest_tactile_dict = None
                self._latest_telemetry_timestamp = time.time()
                self._telemetry_error = None
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    self._handle_io_error_locked("telemetry_failed", exc)
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
            "connect_hardware": bool(self.config.connect_hardware),
            "is_connected": self.hand is not None,
            "reconnect_count": int(self._reconnect_count),
            "max_fingertip_force": int(self.config.max_fingertip_force),
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)
            self._telemetry_thread = None
        if self.hand is not None:
            with self._lock:
                self._close_hand_locked()
        self._stop_event = None

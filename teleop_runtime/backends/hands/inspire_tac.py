from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ...interfaces import EffectorBackend
from ...registry import register_backend
from ...schemas import EffectorCommand


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
        if self.telemetry_hz <= 0.0:
            raise ValueError("telemetry_hz must be positive.")


@register_backend("inspire_tac_hand")
class InspireTacHandBackend(EffectorBackend):
    def __init__(self, config: InspireTacHandBackendConfig):
        self.config = config
        self.hand = None
        self._lock = threading.Lock()
        self._last_sent_command: np.ndarray | None = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_dict: dict[str, np.ndarray] | None = None
        self._telemetry_error: str | None = None
        self._stop_event: threading.Event | None = None
        self._telemetry_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self.hand is not None:
            return
        from hardware.inspire_tac.inspire_tac_modbus import InspireTacModbus

        self.hand = InspireTacModbus(
            ip=self.config.hand_ip,
            port=self.config.port,
            max_fingertip_force=self.config.max_fingertip_force,
        )
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
            raise ValueError(
                f"{self.name} received command for effector '{command.effector_name}'."
            )
        if command.command_type != "joint_vector":
            raise ValueError(
                f"{self.name} only supports command_type='joint_vector', got '{command.command_type}'."
            )

        target = np.asarray(command.value, dtype=np.float32)
        if target.shape != (6,):
            raise ValueError(
                f"{self.name} expects q6 joint_vector with shape (6,), got {target.shape}."
            )
        target = np.clip(target, 0.0, 1.0)

        if self._last_sent_command is not None:
            max_delta = float(np.max(np.abs(target - self._last_sent_command)))
            if max_delta < self.config.send_threshold:
                return

        if self.hand is not None:
            with self._lock:
                self.hand.set_joint_pos(target.tolist())

        self._last_sent_command = target.copy()

    def _telemetry_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                with self._lock:
                    if self.hand is None:
                        return
                    self._latest_joint_pos = np.asarray(self.hand.read_joint_pos(), dtype=float)
                    if self.config.tactile_regions:
                        tactile_vec, tactile_dict = self.hand.read_tac_data(
                            target_regions=list(self.config.tactile_regions)
                        )
                    else:
                        tactile_vec, tactile_dict = self.hand.read_tac_data()
                self._latest_tactile = np.asarray(tactile_vec, dtype=float)
                self._latest_tactile_dict = {
                    name: np.asarray(value, dtype=float) for name, value in tactile_dict.items()
                }
                self._telemetry_error = None
            except Exception as exc:
                self._telemetry_error = str(exc)

            remaining = (1.0 / self.config.telemetry_hz) - (time.time() - loop_start)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def get_state(self) -> dict[str, object]:
        return {
            "last_sent_command": None
            if self._last_sent_command is None
            else self._last_sent_command.copy(),
            "joint_pos": None if self._latest_joint_pos is None else self._latest_joint_pos.copy(),
            "tactile": None if self._latest_tactile is None else self._latest_tactile.copy(),
            "tactile_dict": None
            if self._latest_tactile_dict is None
            else {name: value.copy() for name, value in self._latest_tactile_dict.items()},
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

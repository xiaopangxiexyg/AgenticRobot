from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common import EffectorCommand


DEFAULT_DMTAC_GRIPPER_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
DEFAULT_DMTAC_GRIPPER_SPEED = 0.2
DEFAULT_DMTAC_GRIPPER_TORQUE = 0.2
GRIPPER_ACTION_MAX = 0.99


@dataclass
class DMTacGripperBackendConfig:
    side: str
    effector_name: str | None = None
    gripper_port: str = DEFAULT_DMTAC_GRIPPER_PORT
    gripper_id: int = 1
    gripper_baudrate: int = 115200
    gripper_timeout: float = 0.03
    max_open_mm: float = 70.0
    connect_hardware: bool = True
    send_commands: bool = True
    send_threshold: float = 0.002
    max_action: float = GRIPPER_ACTION_MAX
    speed: float = DEFAULT_DMTAC_GRIPPER_SPEED
    torque: float = DEFAULT_DMTAC_GRIPPER_TORQUE
    continuous_grasp: bool = False
    clear_fault_on_start: bool = True
    open_on_start: bool = False
    enable_telemetry: bool = True
    telemetry_hz: float = 10.0
    connect_tactile: bool = True
    tactile_ids: tuple[str | int, ...] = ()
    tactile_backend: str = "cpu"
    tactile_mode: str = "standard"
    tactile_max_fps: int = 120
    tactile_timeout_ms: int = 100
    tactile_require_ready: bool = True
    tactile_reset_on_start: bool = False
    tactile_ready_timeout_s: float = 10.0
    tactile_warmup_frames: int = 30
    tactile_denoise: bool = True
    tactile_denoise_mode: str = "delta"
    tactile_baseline_frames: int = 20
    tactile_deadband_abs: float = 5e-5
    tactile_temporal_alpha: float = 0.85
    tactile_spatial_blur_ksize: int = 5
    dmtac_root: str | Path | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if int(self.gripper_id) <= 0:
            raise ValueError("gripper_id must be positive.")
        if int(self.gripper_baudrate) <= 0:
            raise ValueError("gripper_baudrate must be positive.")
        if float(self.gripper_timeout) <= 0.0:
            raise ValueError("gripper_timeout must be positive.")
        if float(self.max_open_mm) <= 0.0:
            raise ValueError("max_open_mm must be positive.")
        if not 0.0 <= float(self.max_action) <= GRIPPER_ACTION_MAX:
            raise ValueError(f"max_action must be in [0, {GRIPPER_ACTION_MAX}].")
        if float(self.telemetry_hz) <= 0.0:
            raise ValueError("telemetry_hz must be positive.")
        if int(self.tactile_max_fps) <= 0:
            raise ValueError("tactile_max_fps must be positive.")
        if int(self.tactile_timeout_ms) <= 0:
            raise ValueError("tactile_timeout_ms must be positive.")
        if float(self.tactile_ready_timeout_s) <= 0.0:
            raise ValueError("tactile_ready_timeout_s must be positive.")
        if int(self.tactile_warmup_frames) < 0:
            raise ValueError("tactile_warmup_frames must be non-negative.")
        if int(self.tactile_baseline_frames) <= 0:
            raise ValueError("tactile_baseline_frames must be positive.")
        self.speed = float(np.clip(float(self.speed), 0.0, 1.0))
        self.torque = float(np.clip(float(self.torque), 0.0, 1.0))
        self.max_action = float(np.clip(float(self.max_action), 0.0, GRIPPER_ACTION_MAX))
        self.tactile_ids = tuple(self.tactile_ids)
        if self.dmtac_root is None:
            self.dmtac_root = Path(__file__).resolve().parents[2] / "hardware" / "DMTAC_gripper"
        else:
            self.dmtac_root = Path(self.dmtac_root)


class DMTacGripperBackend:
    """Backend for the Inspire serial gripper plus DM-Tac getDistributeForce()."""

    def __init__(self, config: DMTacGripperBackendConfig):
        self.config = config
        self.device: Any | None = None
        self._lock = threading.Lock()
        self._last_sent_command: np.ndarray | None = None
        self._last_sent_command_timestamp: float | None = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_joint_timestamp: float | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_timestamp: float | None = None
        self._latest_tactile_dict: dict[str, np.ndarray] | None = None
        self._latest_tactile_devices: list[str] = []
        self._latest_tactile_frame_ids: dict[str, int | None] = {}
        self._latest_telemetry_timestamp: float | None = None
        self._telemetry_error: str | None = None
        self._tactile_ready: dict[str, bool] = {}
        self._stop_event: threading.Event | None = None
        self._telemetry_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _ensure_import_path(self) -> None:
        root = Path(self.config.dmtac_root)
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)

    def _create_device(self) -> Any:
        self._ensure_import_path()
        from portable_dmtac_gripper import (
            DMTacGripperDevice,
            GripperConfig,
            TactileConfig,
            TactileDenoiseConfig,
        )

        gripper = GripperConfig(
            port=str(self.config.gripper_port),
            gripper_id=int(self.config.gripper_id),
            baudrate=int(self.config.gripper_baudrate),
            timeout=float(self.config.gripper_timeout),
            max_open_mm=float(self.config.max_open_mm),
            debug=bool(self.config.debug),
        )
        tactile = TactileConfig(
            dev_ids=list(self.config.tactile_ids) or None,
            backend=str(self.config.tactile_backend),
            mode=str(self.config.tactile_mode),
            max_fps=int(self.config.tactile_max_fps),
            fields=("distributed_force",),
            denoise_distributed_force=bool(self.config.tactile_denoise),
            denoise_config=TactileDenoiseConfig(
                enabled=bool(self.config.tactile_denoise),
                mode=str(self.config.tactile_denoise_mode),
                baseline_frames=int(self.config.tactile_baseline_frames),
                deadband_abs=float(self.config.tactile_deadband_abs),
                temporal_alpha=float(self.config.tactile_temporal_alpha),
                spatial_blur_ksize=int(self.config.tactile_spatial_blur_ksize),
            ),
        )
        return DMTacGripperDevice(
            gripper=gripper,
            tactile=tactile,
            connect_gripper=bool(self.config.connect_hardware),
            connect_tactile=bool(self.config.connect_tactile),
        ).connect()

    def start(self) -> None:
        if not self.config.connect_hardware and not self.config.connect_tactile:
            return
        if self.device is not None:
            return
        try:
            with self._lock:
                self.device = self._create_device()
                if self.config.connect_hardware and self.device.gripper is not None:
                    if self.config.clear_fault_on_start:
                        self.device.clear_fault()
                    self.device.set_speed(float(self.config.speed))
                    self.device.set_torque(float(self.config.torque))
                    if self.config.open_on_start:
                        self.device.open(wait=False)
                if self.config.connect_tactile and self.device.tactile is not None:
                    if self.config.tactile_reset_on_start:
                        self.device.reset_tactile()
                    self._tactile_ready = {
                        str(key): bool(value)
                        for key, value in self.device.wait_tactile_ready(
                            timeout_s=float(self.config.tactile_ready_timeout_s)
                        ).items()
                    }
                    if int(self.config.tactile_warmup_frames) > 0:
                        self.device.warmup_tactile_filter(
                            frames=int(self.config.tactile_warmup_frames),
                            timeout_ms=int(self.config.tactile_timeout_ms),
                        )
            self._telemetry_error = None
        except Exception as exc:
            self.device = None
            self._telemetry_error = f"connect_failed: {exc}"
            return

        if self.config.enable_telemetry:
            self._stop_event = threading.Event()
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name=f"{self.name}_dmtac_gripper_telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def is_ready(self) -> bool:
        if self.config.connect_hardware or self.config.connect_tactile:
            return self.device is not None
        return True

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for '{command.effector_name}'.")
        self._send_joint_pos(command.value, bypass_send_commands=False, force=False)

    def send_fixed_joint_pos(self, joint_pos: object) -> None:
        self._send_joint_pos(joint_pos, bypass_send_commands=True, force=True)

    def _extract_gripper_scalar(self, joint_pos: object) -> float:
        target = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if target.shape == (1,):
            return float(np.clip(target[0], 0.0, float(self.config.max_action)))
        if target.shape != (6,):
            raise ValueError(f"{self.name} expects scalar or q6 shape (6,), got {target.shape}.")
        return float(np.clip(target[1], 0.0, float(self.config.max_action)))

    @staticmethod
    def _q6_from_scalar(value: float, *, max_value: float = 1.0) -> np.ndarray:
        q6 = np.zeros(6, dtype=np.float32)
        q6[1] = float(np.clip(float(value), 0.0, min(float(max_value), 1.0)))
        return q6

    def _send_joint_pos(
        self,
        joint_pos: object,
        *,
        bypass_send_commands: bool,
        force: bool,
    ) -> None:
        target_scalar = self._extract_gripper_scalar(joint_pos)
        target_scalar = float(np.clip(target_scalar, 0.0, float(self.config.max_action)))
        target = self._q6_from_scalar(target_scalar, max_value=float(self.config.max_action))
        if not self.config.send_commands and not bool(bypass_send_commands):
            return
        if self._last_sent_command is not None and not bool(force):
            max_delta = float(np.max(np.abs(target - self._last_sent_command)))
            if max_delta < float(self.config.send_threshold):
                return
        command_time = time.time()
        if self.device is not None and self.config.connect_hardware:
            try:
                with self._lock:
                    if self.device is None:
                        return
                    self.device.move(
                        float(np.clip(target_scalar, 0.0, float(self.config.max_action))),
                        speed=float(self.config.speed),
                        torque=float(self.config.torque),
                        continuous=bool(self.config.continuous_grasp),
                        wait=False,
                    )
            except Exception as exc:
                self._telemetry_error = f"send_failed: {exc}"
                return
        self._latest_joint_pos = target.copy()
        self._last_sent_command = target.copy()
        self._last_sent_command_timestamp = command_time

    def _telemetry_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                self._read_telemetry_once()
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                self._telemetry_error = f"telemetry_failed: {exc}"
            remaining = (1.0 / float(self.config.telemetry_hz)) - (time.time() - loop_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def _read_telemetry_once(self) -> None:
        device = self.device
        if device is None:
            if self.config.connect_hardware or self.config.connect_tactile:
                self._telemetry_error = "device is not connected"
            return

        gripper_state = None
        tactile_frames = {}
        read_started_at = time.time()
        with self._lock:
            if self.device is None:
                return
            if self.config.connect_hardware and self.device.gripper is not None:
                gripper_state = self.device.read_gripper_state()
            if self.config.connect_tactile and self.device.tactile is not None:
                tactile_frames = self.device.read_tactile(
                    timeout_ms=int(self.config.tactile_timeout_ms),
                    wait=True,
                    require_ready=bool(self.config.tactile_require_ready),
                )

        if gripper_state is not None:
            joint_value = float(getattr(gripper_state, "joint_pos"))
            self._latest_joint_pos = self._q6_from_scalar(joint_value)
            self._latest_joint_timestamp = float(getattr(gripper_state, "timestamp", read_started_at))

        tactile_vec, tactile_dict, frame_ids = self._frames_to_tactile(tactile_frames)
        self._latest_tactile = tactile_vec
        self._latest_tactile_timestamp = read_started_at if tactile_vec is not None else None
        self._latest_tactile_dict = tactile_dict if tactile_dict else None
        self._latest_tactile_devices = list(tactile_dict)
        self._latest_tactile_frame_ids = frame_ids
        self._latest_telemetry_timestamp = time.time()
        self._telemetry_error = None

    @staticmethod
    def _frames_to_tactile(
        frames: dict[Any, Any],
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray], dict[str, int | None]]:
        tactile_dict: dict[str, np.ndarray] = {}
        frame_ids: dict[str, int | None] = {}
        vectors: list[np.ndarray] = []
        for dev_id, frame in frames.items():
            key = str(dev_id)
            if frame is None:
                continue
            distributed_force = getattr(frame, "distributed_force", None)
            if distributed_force is None:
                continue
            force = np.asarray(distributed_force, dtype=np.float32)
            if force.ndim != 3 or force.shape[2] < 2:
                continue
            force = force.astype(np.float32, copy=True)
            tactile_dict[key] = force
            frame_ids[key] = getattr(frame, "fid", None)
            vectors.append(force.reshape(-1))
        if not vectors:
            return None, tactile_dict, frame_ids
        return np.concatenate(vectors).astype(np.float32, copy=False), tactile_dict, frame_ids

    def get_state(self) -> dict[str, object]:
        gripper_state = None
        if self.device is not None and getattr(self.device, "gripper", None) is not None:
            gripper_state = getattr(self.device.gripper, "last_state", None)
        return {
            "last_sent_command": None
            if self._last_sent_command is None
            else self._last_sent_command.copy(),
            "last_sent_command_timestamp": self._last_sent_command_timestamp,
            "commanded_joint_pos": None
            if self._last_sent_command is None
            else self._last_sent_command.copy(),
            "joint_pos": None if self._latest_joint_pos is None else self._latest_joint_pos.copy(),
            "measured_joint_pos": None if self._latest_joint_pos is None else self._latest_joint_pos.copy(),
            "joint_timestamp": self._latest_joint_timestamp,
            "tactile": None if self._latest_tactile is None else self._latest_tactile.copy(),
            "tactile_timestamp": self._latest_tactile_timestamp,
            "tactile_dict": None
            if self._latest_tactile_dict is None
            else {name: value.copy() for name, value in self._latest_tactile_dict.items()},
            "telemetry_timestamp": self._latest_telemetry_timestamp,
            "tactile_regions": tuple(self._latest_tactile_devices),
            "tactile_devices": tuple(self._latest_tactile_devices),
            "tactile_frame_ids": dict(self._latest_tactile_frame_ids),
            "telemetry_error": self._telemetry_error,
            "connect_hardware": bool(self.config.connect_hardware),
            "send_commands": bool(self.config.send_commands),
            "is_connected": self.device is not None,
            "reconnect_count": 0,
            "backend_type": "dmtac_gripper",
            "gripper_port": str(self.config.gripper_port),
            "gripper_id": int(self.config.gripper_id),
            "gripper_speed": float(self.config.speed),
            "gripper_torque": float(self.config.torque),
            "gripper_max_action": float(self.config.max_action),
            "gripper_continuous": bool(self.config.continuous_grasp),
            "tactile_ready": dict(self._tactile_ready),
            "raw_gripper_state": None
            if gripper_state is None
            else gripper_state.to_dict(),
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)
            self._telemetry_thread = None
        if self.device is not None:
            with self._lock:
                if self.device is not None:
                    try:
                        self.device.close()
                    finally:
                        self.device = None
        self._stop_event = None

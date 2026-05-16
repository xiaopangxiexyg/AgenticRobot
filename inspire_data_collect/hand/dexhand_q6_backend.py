from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..common import EffectorCommand


DEXHAND_TELEOP_Q6_ORDER = (
    "thumb_yaw",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
)
DEXHAND_TACTILE_FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")
DEXHAND_TACTILE_AXIS_ORDER = ("force_x", "force_y", "force_z")
DEXHAND_TACTILE_DIM = 15


@dataclass
class DexHandQ6BackendConfig:
    side: str
    hardware_hand: str = "left"
    effector_name: str | None = None
    connect_hardware: bool = True
    send_commands: bool = True
    send_threshold: float = 0.002
    enable_telemetry: bool = True
    telemetry_hz: float = 20.0
    device_type: str = "auto"
    device_index: int = 0
    tactile_period_ms: int = 20
    mock: bool = False
    clear_errors_on_start: bool = True
    clear_errors_on_fault: bool = True

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if self.hardware_hand not in {"left", "right"}:
            raise ValueError("hardware_hand must be 'left' or 'right'.")
        if float(self.send_threshold) < 0.0:
            raise ValueError("send_threshold must be non-negative.")
        if float(self.telemetry_hz) <= 0.0:
            raise ValueError("telemetry_hz must be positive.")
        if int(self.tactile_period_ms) < 0:
            raise ValueError("tactile_period_ms must be non-negative.")


class DexHandQ6Backend:
    def __init__(self, config: DexHandQ6BackendConfig):
        self.config = config
        self.controller: Any | None = None
        self.mapper: Any | None = None
        self._lock = threading.RLock()
        self._last_sent_command: np.ndarray | None = None
        self._last_sent_command_timestamp: float | None = None
        self._latest_joint_pos: np.ndarray | None = None
        self._latest_joint_timestamp: float | None = None
        self._latest_joint_dict: dict[str, Any] | None = None
        self._latest_tactile: np.ndarray | None = None
        self._latest_tactile_timestamp: float | None = None
        self._latest_tactile_dict: dict[str, Any] | None = None
        self._latest_telemetry_timestamp: float | None = None
        self._telemetry_error: str | None = None
        self._last_error_clear_timestamp: float | None = None
        self._stop_event: threading.Event | None = None
        self._telemetry_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def _ensure_import_path(self) -> None:
        root = Path(__file__).resolve().parents[2]
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)

    def _create_controller(self) -> Any:
        self._ensure_import_path()
        from hardware.portable_dexhand021 import DexHandController

        return DexHandController(
            hand=str(self.config.hardware_hand),
            device_type=str(self.config.device_type),
            device_index=int(self.config.device_index),
            tactile_period_ms=int(self.config.tactile_period_ms),
            mock=bool(self.config.mock),
        )

    def _create_mapper(self) -> Any:
        self._ensure_import_path()
        control_dir = (
            Path(__file__).resolve().parents[2]
            / "hardware"
            / "portable_dexhand021"
            / "6DOF_control"
        )
        text = str(control_dir)
        if text not in sys.path:
            sys.path.insert(0, text)
        from q6_to_dexhand import DexHandQ6Mapper

        return DexHandQ6Mapper()

    def start(self) -> None:
        if not self.config.connect_hardware:
            return
        if self.controller is not None:
            return
        try:
            controller = self._create_controller()
            controller.connect()
            if bool(self.config.clear_errors_on_start):
                controller.clear_errors()
            self.controller = controller
            self.mapper = self._create_mapper()
            self._telemetry_error = None
        except Exception as exc:
            self.controller = None
            self.mapper = None
            self._telemetry_error = f"connect_failed: {exc}"
            return

        if bool(self.config.enable_telemetry):
            self._stop_event = threading.Event()
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name=f"{self.name}_dexhand_telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or (self.controller is not None)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for '{command.effector_name}'.")
        self._send_joint_pos(command.value, bypass_send_commands=False, force=False)

    def send_fixed_joint_pos(self, joint_pos: object) -> None:
        self._send_joint_pos(joint_pos, bypass_send_commands=True, force=True)

    def _normalize_q6(self, joint_pos: object) -> np.ndarray:
        q6 = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if q6.shape != (6,):
            raise ValueError(f"{self.name} expects q6 shape (6,), got {q6.shape}.")
        if not np.all(np.isfinite(q6)):
            raise ValueError("DexHand q6 command must be finite.")
        return np.clip(q6, 0.0, 1.0)

    def _send_joint_pos(
        self,
        joint_pos: object,
        *,
        bypass_send_commands: bool,
        force: bool,
    ) -> None:
        target = self._normalize_q6(joint_pos)
        if not self.config.send_commands and not bool(bypass_send_commands):
            return
        if self._last_sent_command is not None and not bool(force):
            max_delta = float(np.max(np.abs(target - self._last_sent_command)))
            if max_delta < float(self.config.send_threshold):
                return
        command_time = time.time()
        if self.controller is not None:
            try:
                with self._lock:
                    if self.controller is None or self.mapper is None:
                        return
                    dexhand_q6 = self._teleop_q6_to_internal_q6(target)
                    action = self.mapper.q6_to_dexhand_action(
                        dexhand_q6.tolist(),
                        source_order="internal",
                    )
                    self.controller.send_action(action)
            except Exception as exc:
                self._telemetry_error = f"send_failed: {exc}"
                if bool(self.config.clear_errors_on_fault):
                    self._try_clear_errors()
                return
        elif self.config.connect_hardware:
            self._telemetry_error = "send_failed: controller is not connected"
            return
        self._latest_joint_pos = target.copy()
        self._last_sent_command = target.copy()
        self._last_sent_command_timestamp = command_time

    @staticmethod
    def _teleop_q6_to_internal_q6(q6: np.ndarray) -> np.ndarray:
        return np.asarray(
            [q6[1], q6[0], q6[2], q6[3], q6[4], q6[5]],
            dtype=np.float32,
        )

    @staticmethod
    def _internal_q6_to_teleop_q6(q6: np.ndarray) -> np.ndarray:
        return np.asarray(
            [q6[1], q6[0], q6[2], q6[3], q6[4], q6[5]],
            dtype=np.float32,
        )

    def _joint_angle_to_unit(self, angle_deg: float, low: float, high: float) -> float | None:
        if not np.isfinite(angle_deg):
            return None
        if high <= low:
            return None
        return float(np.clip((float(angle_deg) - float(low)) / (float(high) - float(low)), 0.0, 1.0))

    def _invert_joint_state_to_q6(self, state: Mapping[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
        if self.mapper is None:
            return None, {}
        joints = state.get("joints")
        if not isinstance(joints, Mapping):
            return None, {}

        min_deg = tuple(float(value) for value in self.mapper.min_deg)
        max_deg = tuple(float(value) for value in self.mapper.max_deg)
        ratios = dict(self.mapper.four_finger_ratios)

        details: dict[str, Any] = {}
        internal = np.full(6, np.nan, dtype=np.float32)

        thumb_flex_values: list[float] = []
        thumb_joint_details: dict[str, float | None] = {}
        for joint_name in ("th_dip", "th_mcp"):
            angle = _nested_angle(joints.get(joint_name))
            unit = self._joint_angle_to_unit(angle, min_deg[0], max_deg[0]) if angle is not None else None
            thumb_joint_details[joint_name] = unit
            if unit is not None:
                thumb_flex_values.append(float(unit))
        if thumb_flex_values:
            internal[0] = float(np.mean(thumb_flex_values))
        details["thumb_flex"] = thumb_joint_details

        thumb_rot_angle = _nested_angle(joints.get("th_rot"))
        internal[1] = (
            np.float32(self._joint_angle_to_unit(thumb_rot_angle, min_deg[1], max_deg[1]))
            if thumb_rot_angle is not None
            else np.nan
        )
        details["thumb_rot"] = {
            "th_rot": None if thumb_rot_angle is None else float(internal[1]),
        }

        finger_specs = (
            ("index", "ff_dip", "ff_mcp", 2),
            ("middle", "mf_dip", "mf_mcp", 3),
            ("ring", "rf_dip", "rf_mcp", 4),
            ("little", "lf_dip", "lf_mcp", 5),
        )
        for finger_name, dip_joint, mcp_joint, q6_index in finger_specs:
            ratio = float(ratios.get(finger_name, 1.0))
            candidates: list[float] = []
            finger_detail: dict[str, float | None] = {}

            dip_angle = _nested_angle(joints.get(dip_joint))
            if dip_angle is not None:
                dip_unit = self._joint_angle_to_unit(dip_angle, min_deg[q6_index], max_deg[q6_index])
                finger_detail[dip_joint] = dip_unit
                if dip_unit is not None:
                    candidates.append(float(dip_unit))
            else:
                finger_detail[dip_joint] = None

            mcp_angle = _nested_angle(joints.get(mcp_joint))
            if mcp_angle is not None and ratio > 1e-6:
                dip_equivalent = float(mcp_angle) / ratio
                mcp_unit = self._joint_angle_to_unit(
                    dip_equivalent,
                    min_deg[q6_index],
                    max_deg[q6_index],
                )
                finger_detail[mcp_joint] = mcp_unit
                if mcp_unit is not None:
                    candidates.append(float(mcp_unit))
            else:
                finger_detail[mcp_joint] = None

            if candidates:
                internal[q6_index] = float(np.mean(candidates))
            details[finger_name] = finger_detail

        if not np.any(np.isfinite(internal)):
            return None, details
        return self._internal_q6_to_teleop_q6(np.nan_to_num(internal, nan=0.0)), details

    def _tactile_dict_to_vector(self, tactile: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        per_finger: dict[str, dict[str, float]] = {}
        vector = np.zeros(DEXHAND_TACTILE_DIM, dtype=np.float32)
        for finger_index, finger_name in enumerate(DEXHAND_TACTILE_FINGER_ORDER):
            payload = tactile.get(finger_name)
            payload_dict = dict(payload) if isinstance(payload, Mapping) else {}
            finger_entry: dict[str, float] = {}
            for axis_index, axis_name in enumerate(DEXHAND_TACTILE_AXIS_ORDER):
                value = payload_dict.get(axis_name, 0.0)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = 0.0
                if not np.isfinite(number):
                    number = 0.0
                vector[finger_index * 3 + axis_index] = number
                finger_entry[axis_name] = number
            per_finger[finger_name] = finger_entry
        return vector, per_finger

    def _try_clear_errors(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.clear_errors()
            self._last_error_clear_timestamp = time.time()
        except Exception:
            pass

    def _read_telemetry_once(self) -> None:
        controller = self.controller
        if controller is None:
            if self.config.connect_hardware:
                self._telemetry_error = "controller is not connected"
            return
        read_started_at = time.time()
        with self._lock:
            if self.controller is None:
                return
            state = self.controller.read_state()
        q6, q6_detail = self._invert_joint_state_to_q6(state)
        tactile_dict = dict(state.get("tactile", {})) if isinstance(state.get("tactile"), Mapping) else {}
        tactile_vec, tactile_summary = self._tactile_dict_to_vector(tactile_dict)
        if q6 is not None:
            self._latest_joint_pos = np.clip(q6.astype(np.float32, copy=False), 0.0, 1.0)
            self._latest_joint_timestamp = _state_timestamp_seconds(state)
        else:
            self._latest_joint_pos = None
            self._latest_joint_timestamp = None
        self._latest_joint_dict = {
            "teleop_q6_order": list(DEXHAND_TELEOP_Q6_ORDER),
            "joint_inversion": q6_detail,
            "raw_joints": to_plain_dict(state.get("joints")),
        }
        self._latest_tactile = tactile_vec
        self._latest_tactile_timestamp = _state_timestamp_seconds(state)
        self._latest_tactile_dict = {
            "finger_order": list(DEXHAND_TACTILE_FINGER_ORDER),
            "axis_order": list(DEXHAND_TACTILE_AXIS_ORDER),
            "fingertips": tactile_summary,
            "raw_tactile": to_plain_dict(tactile_dict),
        }
        self._latest_telemetry_timestamp = read_started_at
        self._telemetry_error = None

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
                if bool(self.config.clear_errors_on_fault):
                    self._try_clear_errors()
            remaining = (1.0 / float(self.config.telemetry_hz)) - (time.time() - loop_start)
            if remaining > 0.0:
                self._stop_event.wait(remaining)

    def get_state(self) -> dict[str, object]:
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
            "joint_pos_order": list(DEXHAND_TELEOP_Q6_ORDER),
            "joint_feedback_dict": None if self._latest_joint_dict is None else dict(self._latest_joint_dict),
            "tactile": None if self._latest_tactile is None else self._latest_tactile.copy(),
            "tactile_timestamp": self._latest_tactile_timestamp,
            "tactile_regions": tuple(DEXHAND_TACTILE_FINGER_ORDER),
            "tactile_dict": None if self._latest_tactile_dict is None else dict(self._latest_tactile_dict),
            "telemetry_timestamp": self._latest_telemetry_timestamp,
            "telemetry_error": self._telemetry_error,
            "connect_hardware": bool(self.config.connect_hardware),
            "send_commands": bool(self.config.send_commands),
            "is_connected": self.controller is not None,
            "backend_type": "dexhand_q6",
            "logical_side": str(self.config.side),
            "hardware_hand": str(self.config.hardware_hand),
            "tactile_dim": int(DEXHAND_TACTILE_DIM),
            "last_error_clear_timestamp": self._last_error_clear_timestamp,
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)
            self._telemetry_thread = None
        if self.controller is not None:
            with self._lock:
                if self.controller is not None:
                    try:
                        self.controller.close()
                    finally:
                        self.controller = None
        self._stop_event = None


def _nested_angle(joint: object) -> float | None:
    if not isinstance(joint, Mapping):
        return None
    value = joint.get("angle")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _state_timestamp_seconds(state: Mapping[str, Any]) -> float | None:
    value = state.get("timestamp")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def to_plain_dict(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value

"""Portable DexHand end-effector controller.

This module prefers the installed ``pyzlg_dexhand`` package. A read-only
fallback handles the local checkout layout where the outer source directory
shadows the installed package name.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

try:
    from pyzlg_dexhand import LeftDexHand, MockZCANWrapper, RightDexHand, ZCANWrapper
    from pyzlg_dexhand.dexhand_interface import (
        ControlMode,
        FeedbackMode,
        HandFeedback,
        JointCommand,
        JointFeedback,
        StampedTouchFeedback,
    )
    from pyzlg_dexhand.zcan import ZCANDeviceType
except ImportError:
    from pyzlg_dexhand.pyzlg_dexhand import LeftDexHand, MockZCANWrapper, RightDexHand, ZCANWrapper
    from pyzlg_dexhand.pyzlg_dexhand.dexhand_interface import (
        ControlMode,
        FeedbackMode,
        HandFeedback,
        JointCommand,
        JointFeedback,
        StampedTouchFeedback,
    )
    from pyzlg_dexhand.pyzlg_dexhand.zcan import ZCANDeviceType


JOINT_NAMES = (
    "th_dip",
    "th_mcp",
    "th_rot",
    "ff_spr",
    "ff_dip",
    "ff_mcp",
    "mf_dip",
    "mf_mcp",
    "rf_dip",
    "rf_mcp",
    "lf_dip",
    "lf_mcp",
)

FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")

DEVICE_TYPES = {
    "100U": ZCANDeviceType.ZCAN_USBCANFD_100U,
    "200U": ZCANDeviceType.ZCAN_USBCANFD_200U,
}


ActionValue = Union[
    int,
    float,
    JointCommand,
    Mapping[str, Optional[Union[int, float]]],
]


def normalize_direction(direction: Optional[int]) -> Optional[int]:
    """Return ``None`` for invalid tactile force directions."""
    if direction is None or direction == 65535 or direction < 0 or direction > 359:
        return None
    return int(direction)


def tactile_force_xyz(
    normal_force: float,
    tangential_force: float,
    direction: Optional[int],
) -> Dict[str, Union[bool, float]]:
    """Resolve DexHand fingertip force readings into the local tactile XYZ frame.

    Frame convention:
    +Z is the compressive normal direction into the fingertip pad.
    +Y is the distal fingertip direction on the pad surface.
    +X is the right-hand tangent direction on the pad surface.
    ``direction`` is measured in degrees clockwise from +Y toward +X.
    """
    direction = normalize_direction(direction)
    force_z = float(normal_force)
    if direction is None or tangential_force <= 0:
        return {
            "force_x": 0.0,
            "force_y": 0.0,
            "force_z": force_z,
            "force_direction_valid": False,
        }

    theta = math.radians(direction)
    force_t = float(tangential_force)
    return {
        "force_x": force_t * math.sin(theta),
        "force_y": force_t * math.cos(theta),
        "force_z": force_z,
        "force_direction_valid": True,
    }


def _timestamp_to_seconds(timestamp: Optional[Union[int, float]]) -> Optional[float]:
    if timestamp is None:
        return None
    value = float(timestamp)
    if value > 1e12:
        return value / 1e9
    return value


def _finite_or_none(value: Optional[Union[int, float]]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def joint_to_dict(joint: JointFeedback) -> Dict[str, Optional[Union[int, float]]]:
    """Convert SDK joint feedback to a stable JSON-friendly dict."""
    return {
        "timestamp": _timestamp_to_seconds(joint.timestamp),
        "angle": _finite_or_none(joint.angle),
        "encoder_position": joint.encoder_position,
        "current": joint.current,
        "velocity": joint.velocity,
        "error_code": joint.error_code,
        "impedance": joint.impedance,
    }


def tactile_to_dict(touch: StampedTouchFeedback) -> Dict[str, object]:
    """Convert SDK tactile feedback to a stable JSON-friendly dict."""
    force = tactile_force_xyz(
        normal_force=touch.normal_force,
        tangential_force=touch.tangential_force,
        direction=int(touch.direction),
    )
    return {
        "timestamp": _timestamp_to_seconds(touch.timestamp),
        "force_x": float(force["force_x"]),
        "force_y": float(force["force_y"]),
        "force_z": float(force["force_z"]),
        "force_direction_valid": bool(force["force_direction_valid"]),
        "force_z_delta_raw": int(touch.normal_force_delta),
        "force_xy_delta_raw": int(touch.tangential_force_delta),
        "proximity": int(touch.proximity),
        "temperature": int(touch.temperature),
    }


def feedback_to_dict(feedback: HandFeedback, hand: str) -> Dict[str, Any]:
    """Convert SDK whole-hand feedback to a unified state dict."""
    return {
        "timestamp": feedback.query_timestamp,
        "hand": hand,
        "joints": {
            name: joint_to_dict(joint)
            for name, joint in feedback.joints.items()
        },
        "tactile": {
            finger: tactile_to_dict(touch)
            for finger, touch in feedback.touch.items()
        },
    }


def _control_mode(value: Union[str, ControlMode]) -> ControlMode:
    if isinstance(value, ControlMode):
        return value
    key = value.strip().replace("-", "_").replace(" ", "_").upper()
    try:
        return ControlMode[key]
    except KeyError as exc:
        valid = ", ".join(mode.name.lower() for mode in ControlMode)
        raise ValueError(f"unknown control mode {value!r}; valid modes: {valid}") from exc


def _joint_command(value: ActionValue) -> JointCommand:
    if isinstance(value, JointCommand):
        return value
    if isinstance(value, (int, float)):
        return JointCommand(position=float(value))
    if isinstance(value, Mapping):
        if "position" not in value:
            raise ValueError("joint command mapping must include 'position'")
        position = value["position"]
        if position is None:
            raise ValueError("joint command position cannot be None")
        current = value.get("current")
        velocity = value.get("velocity")
        return JointCommand(
            position=float(position),
            current=None if current is None else int(current),
            velocity=None if velocity is None else int(velocity),
        )
    raise TypeError(f"unsupported joint command value: {type(value).__name__}")


def _validate_joint_command(name: str, command: JointCommand) -> None:
    if command.current is not None and not 10 <= int(command.current) <= 255:
        raise ValueError(
            f"{name}.current={command.current} is outside the currently encodable "
            "SDK range 10-255 mA"
        )
    if command.velocity is not None and not 0 <= int(command.velocity) <= 32767:
        raise ValueError(f"{name}.velocity={command.velocity} must be in 0-32767")


def _dict_from_sequence(values: Sequence[Optional[ActionValue]]) -> Dict[str, ActionValue]:
    if len(values) != len(JOINT_NAMES):
        raise ValueError(f"action sequence must have {len(JOINT_NAMES)} values")
    return {
        name: value
        for name, value in zip(JOINT_NAMES, values)
        if value is not None
    }


class DexHandController:
    """Unified single-hand controller for DexHand as an end effector.

    The wrapper owns one hardware connection by default. Use it as a context
    manager so the ZCAN device is always closed when the control loop exits.
    """

    def __init__(
        self,
        hand: str = "left",
        *,
        device_index: int = 0,
        device_type: str = "auto",
        mock: bool = False,
        zcan: Optional[ZCANWrapper] = None,
        configure_tactile: bool = True,
        tactile_period_ms: int = 20,
        default_control_mode: Union[str, ControlMode] = ControlMode.IMPEDANCE_GRASP,
        use_broadcast: bool = True,
        log_level: Any = None,
    ) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        if device_type not in {"auto", "config", "100U", "200U"}:
            raise ValueError("device_type must be 'auto', 'config', '100U', or '200U'")
        if tactile_period_ms < 0:
            raise ValueError("tactile_period_ms must be non-negative")

        self.hand_name = hand
        self.device_index = device_index
        self.device_type = device_type
        self.mock = mock
        self.configure_tactile = configure_tactile
        self.tactile_period_ms = tactile_period_ms
        self.default_control_mode = _control_mode(default_control_mode)
        self.use_broadcast = use_broadcast
        self.log_level = log_level

        self.zcan = zcan
        self._owns_zcan = zcan is None
        self._matched_device_type: Optional[str] = None
        self._hand: Optional[Union[LeftDexHand, RightDexHand]] = None
        self._connected = False
        self._lock = threading.RLock()

    @property
    def hand(self) -> Union[LeftDexHand, RightDexHand]:
        if self._hand is None or not self._connected:
            raise RuntimeError("DexHandController is not connected; call connect() first")
        return self._hand

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def matched_device_type(self) -> Optional[str]:
        return self._matched_device_type

    def connect(self) -> "DexHandController":
        """Open ZCAN, initialize the hand, and optionally enable periodic feedback."""
        if self._connected:
            return self

        try:
            if self.zcan is None:
                self.zcan = MockZCANWrapper() if self.mock else ZCANWrapper()

            if self._owns_zcan:
                self._matched_device_type = self._open_zcan()

            hand_class = LeftDexHand if self.hand_name == "left" else RightDexHand
            hand_kwargs = {
                "zcan": self.zcan,
                "device_index": self.device_index,
                "auto_init": False,
            }
            if self.log_level is not None:
                hand_kwargs["log_level"] = self.log_level
            self._hand = hand_class(**hand_kwargs)
            if not self._hand.init(device_index=self.device_index):
                raise RuntimeError(f"failed to initialize {self.hand_name} DexHand")

            self._connected = True
            if self.configure_tactile:
                if not self.configure_feedback(self.tactile_period_ms, enable=True):
                    raise RuntimeError("failed to configure periodic tactile feedback")
                time.sleep(max(self.tactile_period_ms / 1000.0 * 2.0, 0.05))
            return self
        except Exception:
            self.close()
            raise

    initialize = connect

    def _open_zcan(self) -> str:
        assert self.zcan is not None
        if self.mock:
            if not self.zcan.open(device_index=self.device_index):
                raise RuntimeError("failed to open mock ZCAN device")
            return "mock"

        if self.device_type in DEVICE_TYPES:
            selected = DEVICE_TYPES[self.device_type]
            if self.zcan.open(device_type=selected, device_index=self.device_index):
                return self.device_type
            raise RuntimeError(f"failed to open USBCANFD-{self.device_type}")

        if self.device_type == "config":
            if self.zcan.open(device_index=self.device_index):
                configured = getattr(self.zcan, "zcan_device_type", "configured")
                return str(configured).replace("ZCAN_USBCANFD_", "")
            raise RuntimeError("failed to open configured ZCAN device")

        configured = getattr(self.zcan, "device_type", None)
        candidates = [candidate for candidate in (configured, *DEVICE_TYPES.values()) if candidate is not None]
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if self.zcan.open(device_type=candidate, device_index=self.device_index):
                for label, enum_value in DEVICE_TYPES.items():
                    if enum_value == candidate:
                        return label
                return str(candidate)

        raise RuntimeError(
            "failed to open USBCANFD adapter; check USB connection, permissions, "
            "config.yaml, and whether another process is using the adapter"
        )

    def configure_feedback(
        self,
        period_ms: int = 20,
        *,
        enable: bool = True,
        mode: FeedbackMode = FeedbackMode.PERIODIC,
    ) -> bool:
        """Configure SDK feedback. Periodic feedback is required for passive tactile reads."""
        if period_ms < 0:
            raise ValueError("period_ms must be non-negative")
        with self._lock:
            return bool(self.hand.set_feedback_mode(mode, period_ms=period_ms, enable=enable))

    def send_action(
        self,
        action: Optional[Union[Mapping[str, ActionValue], Sequence[Optional[ActionValue]]]] = None,
        *,
        control_mode: Optional[Union[str, ControlMode]] = None,
        use_broadcast: Optional[bool] = None,
        clear_error: bool = False,
        request_feedback: bool = True,
        **joint_positions: ActionValue,
    ) -> bool:
        """Send a joint-space action.

        ``action`` can be either a mapping keyed by joint name or a 12-value
        sequence following ``JOINT_NAMES``. Keyword joint positions can be used
        for concise calls and override values from ``action``.
        """
        joint_targets: MutableMapping[str, ActionValue] = {}
        if action is not None:
            if isinstance(action, Mapping):
                joint_targets.update(action)
            elif isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
                joint_targets.update(_dict_from_sequence(action))
            else:
                raise TypeError("action must be a mapping or a 12-value sequence")
        joint_targets.update(joint_positions)

        unknown = sorted(set(joint_targets) - set(JOINT_NAMES))
        if unknown:
            raise ValueError(f"unknown joints: {', '.join(unknown)}")

        normalized = {}
        for name, value in joint_targets.items():
            if value is None:
                continue
            command = _joint_command(value)
            _validate_joint_command(name, command)
            normalized[name] = command
        mode = _control_mode(control_mode) if control_mode is not None else self.default_control_mode
        broadcast = self.use_broadcast if use_broadcast is None else use_broadcast

        with self._lock:
            return bool(
                self.hand.move_joints(
                    control_mode=mode,
                    use_broadcast=broadcast,
                    clear_error=clear_error,
                    request_feedback=request_feedback,
                    **normalized,
                )
            )

    def read_state(self) -> Dict[str, Any]:
        """Read joints and tactile data as one stable dict."""
        with self._lock:
            return feedback_to_dict(self.hand.get_feedback(), self.hand_name)

    def read_joints(self) -> Dict[str, Dict[str, Optional[Union[int, float]]]]:
        """Read joint feedback only from the unified state payload."""
        return self.read_state()["joints"]

    def read_tactile(self) -> Dict[str, Dict[str, Optional[Union[int, float]]]]:
        """Read fingertip tactile feedback only."""
        with self._lock:
            return {
                finger: tactile_to_dict(touch)
                for finger, touch in self.hand.get_tactile_feedback().items()
            }

    def read_errors(self) -> Dict[int, Any]:
        """Return board error info, converted to plain dicts when possible."""
        with self._lock:
            errors = {}
            for board_idx, error in self.hand.get_errors().items():
                if error is None:
                    errors[board_idx] = None
                elif is_dataclass(error):
                    errors[board_idx] = asdict(error)
                else:
                    errors[board_idx] = error
            return errors

    def reset(self, *, use_broadcast: Optional[bool] = None) -> bool:
        """Move all joints to zero."""
        broadcast = self.use_broadcast if use_broadcast is None else use_broadcast
        with self._lock:
            return bool(self.hand.reset_joints(use_broadcast=broadcast))

    def clear_errors(self) -> bool:
        """Clear board errors using the SDK's reliable per-board path."""
        with self._lock:
            return bool(self.hand.clear_errors(clear_all=True, use_broadcast=False))

    def visualize_tactile(
        self,
        *,
        rate_hz: float = 20.0,
        max_normal_force: float = 5.0,
        max_tangential_force: float = 2.0,
    ) -> None:
        """Launch a small Tkinter tactile force viewer using this connection."""
        from .tactile_viewer import run_viewer

        if not self.connected:
            self.connect()
        run_viewer(
            self,
            rate_hz=rate_hz,
            max_normal_force=max_normal_force,
            max_tangential_force=max_tangential_force,
        )

    def close(self) -> None:
        """Close the owned ZCAN device."""
        with self._lock:
            if self._hand is not None:
                self._hand.close()
            if self._owns_zcan and self.zcan is not None:
                self.zcan.close()
            self._connected = False
            self._hand = None

    def __enter__(self) -> "DexHandController":
        return self.connect()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

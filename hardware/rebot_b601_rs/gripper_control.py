"""Single-motor J7 MIT hold boundary.

The normal B601 owner intentionally controls J1--J6 only.  This module is a
separate, explicit J7 owner for commissioning the RobStride gripper gains.
It registers exactly one motor, never calls ``Controller.enable_all`` or
``disable_all``, and sends a bounded hold at the freshly read position.  The
caller must still provide the foreground safety confirmation and arrange
mechanical support before enabling the gripper.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .backend import B601HardwareDependencyError, B601HardwareError
from .gripper import B601RSGripperReaderConfig


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _positive(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True)
class B601RSGripperControlConfig:
    """Runtime J7 MIT settings; no persistent vendor register is written."""

    reader: B601RSGripperReaderConfig
    kp: float
    kd: float
    rate_hz: float = 20.0
    duration_sec: float = 30.0
    max_position_error_rad: float = 0.25
    timeout_ms: int = 500

    def __post_init__(self) -> None:
        object.__setattr__(self, "kp", _nonnegative(self.kp, name="gripper_kp"))
        object.__setattr__(self, "kd", _nonnegative(self.kd, name="gripper_kd"))
        object.__setattr__(self, "rate_hz", _positive(self.rate_hz, name="rate_hz"))
        object.__setattr__(
            self,
            "duration_sec",
            _positive(self.duration_sec, name="duration_sec"),
        )
        object.__setattr__(
            self,
            "max_position_error_rad",
            _positive(self.max_position_error_rad, name="max_position_error_rad"),
        )
        if isinstance(self.timeout_ms, bool) or not isinstance(self.timeout_ms, int):
            raise TypeError("timeout_ms must be an integer")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    @classmethod
    def from_rs_yaml(
        cls,
        path: str | Path,
        *,
        kp: float | None = None,
        kd: float | None = None,
        rate_hz: float = 20.0,
        duration_sec: float = 30.0,
        max_position_error_rad: float = 0.25,
        timeout_ms: int = 500,
    ) -> B601RSGripperControlConfig:
        reader = B601RSGripperReaderConfig.from_rs_yaml(path)
        return cls(
            reader=reader,
            kp=reader.mit_kp if kp is None else kp,
            kd=reader.mit_kd if kd is None else kd,
            rate_hz=rate_hz,
            duration_sec=duration_sec,
            max_position_error_rad=max_position_error_rad,
            timeout_ms=timeout_ms,
        )


class B601RSGripperController:
    """Foreground J7-only MotorBridge controller with injectable runtime."""

    def __init__(
        self,
        config: B601RSGripperControlConfig,
        *,
        controller_factory: Callable[[str], object] | None = None,
        module_loader: Callable[[], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._controller_factory = controller_factory
        self._module_loader = module_loader
        self._clock = clock
        self._sleeper = sleeper
        self._controller: object | None = None
        self._motor: object | None = None
        self._mode: object | None = None
        self._enabled = False

    @property
    def connected(self) -> bool:
        return self._controller is not None and self._motor is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def connect(self) -> None:
        if self.connected:
            return
        controller = None
        try:
            module = self._module_loader() if self._module_loader else import_module("motorbridge")
            if self._controller_factory is not None:
                controller = self._controller_factory(self.config.reader.channel)
            else:
                controller = module.Controller(self.config.reader.channel)
            motor = controller.add_robstride_motor(
                self.config.reader.motor_id,
                self.config.reader.feedback_id,
                self.config.reader.model,
            )
            required = ("robstride_get_param_f32_host_id", "ensure_mode", "enable", "disable", "send_mit")
            missing = [name for name in required if not callable(getattr(motor, name, None))]
            if missing:
                raise B601HardwareDependencyError(
                    "MotorBridge J7 controller is missing: " + ", ".join(missing)
                )
            mode_type = getattr(module, "Mode", None)
            mode = getattr(mode_type, "MIT", None)
            if mode is None:
                raise B601HardwareDependencyError("motorbridge.Mode.MIT is unavailable")
            self._controller = controller
            self._motor = motor
            self._mode = mode
        except ModuleNotFoundError as exc:
            if controller is not None:
                close = getattr(controller, "close", None)
                if callable(close):
                    close()
            raise B601HardwareDependencyError("motorbridge is required for J7 control") from exc
        except Exception:
            if controller is not None:
                close = getattr(controller, "close", None)
                if callable(close):
                    close()
            raise

    def read_position(self) -> float:
        if self._motor is None:
            raise B601HardwareError("B601 J7 controller is not connected")
        try:
            value = self._motor.robstride_get_param_f32_host_id(
                self.config.reader.parameter_id,
                self.config.reader.feedback_id,
                self.config.timeout_ms,
            )
        except Exception as exc:
            raise B601HardwareError(f"B601 J7 mechPos read failed: {exc}") from exc
        return _finite(value, name="gripper position")

    def prepare_hold(self, *, target_position_rad: float | None = None) -> float:
        if self._motor is None or self._mode is None:
            raise B601HardwareError("B601 J7 controller is not connected")
        target = self.read_position() if target_position_rad is None else _finite(
            target_position_rad,
            name="target_position_rad",
        )
        try:
            self._motor.ensure_mode(self._mode, self.config.timeout_ms)
        except Exception as exc:
            raise B601HardwareError(f"B601 J7 MIT mode setup failed: {exc}") from exc
        return target

    def run_hold(self, target_position_rad: float) -> int:
        if self._motor is None:
            raise B601HardwareError("B601 J7 controller is not connected")
        period = 1.0 / self.config.rate_hz
        deadline = self._clock() + self.config.duration_sec
        commands = 0
        self._motor.enable()
        self._enabled = True
        try:
            while self._clock() < deadline:
                current = self.read_position()
                error = abs(current - target_position_rad)
                if error > self.config.max_position_error_rad:
                    raise B601HardwareError(
                        "B601 J7 hold error exceeded bound: "
                        f"{error:.6f} rad > {self.config.max_position_error_rad:.6f} rad"
                    )
                self._motor.send_mit(
                    float(target_position_rad),
                    0.0,
                    self.config.kp,
                    self.config.kd,
                    0.0,
                )
                commands += 1
                self._sleeper(period)
        finally:
            self.disable()
        return commands

    def disable(self) -> None:
        motor = self._motor
        if motor is not None and self._enabled:
            try:
                motor.disable()
            finally:
                self._enabled = False

    def close(self) -> None:
        self.disable()
        motor = self._motor
        controller = self._controller
        self._motor = None
        self._controller = None
        self._mode = None
        if motor is not None:
            close_motor = getattr(motor, "close", None)
            if callable(close_motor):
                close_motor()
        if controller is not None:
            close_controller = getattr(controller, "close", None)
            if callable(close_controller):
                close_controller()


__all__ = ["B601RSGripperControlConfig", "B601RSGripperController"]

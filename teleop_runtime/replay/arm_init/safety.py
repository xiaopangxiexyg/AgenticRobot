from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.rm65 import normalize_side


FORCE_SOURCES = ("raw", "zero", "work", "tool")


@dataclass(frozen=True)
class RM65EmergencyStopResult:
    side: str
    method: str
    code: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "method": self.method,
            "code": self.code,
        }


@dataclass(frozen=True)
class RM65ForceGuardConfig:
    source: str = "tool"
    max_force_n: float = 15.0
    max_torque_nm: float = 2.0
    enabled: bool = True
    emergency_stop_on_limit: bool = True

    def __post_init__(self) -> None:
        source = str(self.source or "tool").strip()
        if source not in FORCE_SOURCES:
            raise ValueError(f"force source must be one of {FORCE_SOURCES}, got {source!r}.")
        max_force = float(self.max_force_n)
        max_torque = float(self.max_torque_nm)
        if not math.isfinite(max_force) or max_force <= 0.0:
            raise ValueError("max_force_n must be a positive finite value.")
        if not math.isfinite(max_torque) or max_torque <= 0.0:
            raise ValueError("max_torque_nm must be a positive finite value.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "max_force_n", max_force)
        object.__setattr__(self, "max_torque_nm", max_torque)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "emergency_stop_on_limit", bool(self.emergency_stop_on_limit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "max_force_n": self.max_force_n,
            "max_torque_nm": self.max_torque_nm,
            "emergency_stop_on_limit": self.emergency_stop_on_limit,
        }


@dataclass(frozen=True)
class RM65ForceReading:
    side: str
    source: str
    values: np.ndarray
    force_norm_n: float
    torque_norm_nm: float
    sources: dict[str, np.ndarray]
    monotonic_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "source": self.source,
            "values": self.values.astype(float).tolist(),
            "force_norm_n": self.force_norm_n,
            "torque_norm_nm": self.torque_norm_nm,
            "sources": {name: value.astype(float).tolist() for name, value in self.sources.items()},
            "monotonic_time_sec": self.monotonic_time_sec,
        }


class RM65ForceLimitExceeded(RuntimeError):
    def __init__(
        self,
        *,
        side: str,
        reading: RM65ForceReading,
        config: RM65ForceGuardConfig,
        stop_result: RM65EmergencyStopResult | None,
    ) -> None:
        message = (
            f"{side}: force guard exceeded on {reading.source}: "
            f"|F|={reading.force_norm_n:.3f}N limit={config.max_force_n:.3f}N, "
            f"|T|={reading.torque_norm_nm:.3f}Nm limit={config.max_torque_nm:.3f}Nm"
        )
        if stop_result is not None:
            message += f"; emergency stop via {stop_result.method} returned {stop_result.code}"
        super().__init__(message)
        self.side = side
        self.reading = reading
        self.config = config
        self.stop_result = stop_result


def read_rm65_force_sources(robot: object, side: str) -> dict[str, np.ndarray]:
    side = normalize_side(side)
    if not hasattr(robot, "Get_Force_Data"):
        raise RuntimeError(f"{side}: robot object does not expose Get_Force_Data")
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": _as_force6(raw, label=f"{side}.raw_force"),
        "zero": _as_force6(zero, label=f"{side}.zero_force"),
        "work": _as_force6(work, label=f"{side}.work_zero_force"),
        "tool": _as_force6(tool, label=f"{side}.tool_zero_force"),
    }


def read_rm65_force_guard(
    robot: object,
    side: str,
    config: RM65ForceGuardConfig | None = None,
) -> RM65ForceReading:
    config = config or RM65ForceGuardConfig()
    side = normalize_side(side)
    sources = read_rm65_force_sources(robot, side)
    values = sources[config.source]
    force_norm = float(np.linalg.norm(values[:3]))
    torque_norm = float(np.linalg.norm(values[3:6]))
    return RM65ForceReading(
        side=side,
        source=config.source,
        values=values,
        force_norm_n=force_norm,
        torque_norm_nm=torque_norm,
        sources=sources,
        monotonic_time_sec=time.monotonic(),
    )


def check_rm65_force_guard(
    robot: object,
    side: str,
    config: RM65ForceGuardConfig | None = None,
) -> RM65ForceReading | None:
    config = config or RM65ForceGuardConfig()
    side = normalize_side(side)
    if not config.enabled:
        return None
    reading = read_rm65_force_guard(robot, side, config)
    exceeded = reading.force_norm_n > config.max_force_n or reading.torque_norm_nm > config.max_torque_nm
    if not exceeded:
        return reading
    stop_result = None
    if config.emergency_stop_on_limit:
        stop_result = emergency_stop_rm65(robot, side, reason="force_guard_exceeded")
    raise RM65ForceLimitExceeded(side=side, reading=reading, config=config, stop_result=stop_result)


def emergency_stop_rm65(robot: object, side: str, *, reason: str = "") -> RM65EmergencyStopResult:
    side = normalize_side(side)
    for method_name in ("Arm_Stop", "Arm_Slow_Stop"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        try:
            code = method()
        except TypeError:
            code = method(True)
        return RM65EmergencyStopResult(side=side, method=method_name, code=_optional_int(code))
    suffix = f" after {reason}" if reason else ""
    raise RuntimeError(f"{side}: cannot emergency-stop{suffix}; robot exposes no Arm_Stop/Arm_Slow_Stop")


def force_guard_summary(
    config: RM65ForceGuardConfig,
    last_reading: RM65ForceReading | None,
) -> dict[str, Any]:
    payload = config.to_dict()
    if last_reading is not None:
        payload["last_reading"] = last_reading.to_dict()
    return payload


def _as_force6(values: object, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - caller receives a labelled validation error.
        raise RuntimeError(f"{label} must be a numeric force6 vector") from exc
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} must contain 6 finite values, got shape={array.shape}")
    return array.astype(float, copy=True)


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        code = value.get("code")
        if code is not None:
            return int(code)
    return int(value)

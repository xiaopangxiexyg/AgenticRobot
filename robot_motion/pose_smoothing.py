from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .transforms import interpolate_pose, parse_pose_matrix, rotation_angle_rad, translation_distance


SCHEMA_VERSION = "teleop_runtime.pose_smoothing.v1"


@dataclass(frozen=True)
class PoseSmoothingConfig:
    enabled: bool = True
    time_constant_sec: float = 0.035
    nominal_dt_sec: float = 1.0 / 15.0
    min_dt_sec: float = 0.0
    max_dt_sec: float = 0.12
    reset_after_gap_sec: float = 0.50
    deadband_translation_m: float = 3e-4
    deadband_rotation_deg: float = 0.05
    snap_translation_m: float = 0.08
    snap_rotation_deg: float = 20.0
    source_path: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.time_constant_sec < 0.0:
            raise ValueError("time_constant_sec must be >= 0.")
        if self.nominal_dt_sec <= 0.0:
            raise ValueError("nominal_dt_sec must be > 0.")
        if self.min_dt_sec < 0.0:
            raise ValueError("min_dt_sec must be >= 0.")
        if self.max_dt_sec <= 0.0:
            raise ValueError("max_dt_sec must be > 0.")
        if self.max_dt_sec < self.min_dt_sec:
            raise ValueError("max_dt_sec must be >= min_dt_sec.")
        if self.reset_after_gap_sec < 0.0:
            raise ValueError("reset_after_gap_sec must be >= 0.")
        if self.deadband_translation_m < 0.0:
            raise ValueError("deadband_translation_m must be >= 0.")
        if self.deadband_rotation_deg < 0.0:
            raise ValueError("deadband_rotation_deg must be >= 0.")
        if self.snap_translation_m < 0.0:
            raise ValueError("snap_translation_m must be >= 0.")
        if self.snap_rotation_deg < 0.0:
            raise ValueError("snap_rotation_deg must be >= 0.")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: str | None = None,
    ) -> "PoseSmoothingConfig":
        values = dict(payload)
        values.pop("schema_version", None)
        values.pop("source_path", None)
        allowed = {
            "enabled",
            "time_constant_sec",
            "nominal_dt_sec",
            "min_dt_sec",
            "max_dt_sec",
            "reset_after_gap_sec",
            "deadband_translation_m",
            "deadband_rotation_deg",
            "snap_translation_m",
            "snap_rotation_deg",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown pose smoothing config field(s): {unknown}")
        return cls(**values, source_path=source_path)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "time_constant_sec": float(self.time_constant_sec),
            "nominal_dt_sec": float(self.nominal_dt_sec),
            "min_dt_sec": float(self.min_dt_sec),
            "max_dt_sec": float(self.max_dt_sec),
            "reset_after_gap_sec": float(self.reset_after_gap_sec),
            "deadband_translation_m": float(self.deadband_translation_m),
            "deadband_rotation_deg": float(self.deadband_rotation_deg),
            "snap_translation_m": float(self.snap_translation_m),
            "snap_rotation_deg": float(self.snap_rotation_deg),
            "source_path": self.source_path,
        }


# Shared position-teleop target shaping used by real RM65 control and Sim-HIL.
# Keep this profile in robot_motion so simulation does not depend upward on
# teleop_runtime.  The real-side JSON loader is checked against this contract.
RM65_POSITION_TARGET_POSE_SMOOTHING_V1 = PoseSmoothingConfig(
    enabled=True,
    time_constant_sec=0.025,
    nominal_dt_sec=0.0666666667,
    min_dt_sec=0.0,
    max_dt_sec=0.12,
    reset_after_gap_sec=0.5,
    deadband_translation_m=0.0003,
    deadband_rotation_deg=0.05,
    snap_translation_m=0.08,
    snap_rotation_deg=20.0,
)


@dataclass(frozen=True)
class PoseSmoothingResult:
    pose: np.ndarray
    metadata: dict[str, object]


class PoseSmoother:
    def __init__(self, config: PoseSmoothingConfig) -> None:
        self.config = config
        self._last_pose: np.ndarray | None = None
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._last_pose = None
        self._last_timestamp = None

    def smooth(
        self,
        target_pose: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> PoseSmoothingResult:
        raw_pose = parse_pose_matrix(target_pose)
        if not self.config.enabled:
            return PoseSmoothingResult(
                pose=raw_pose,
                metadata={
                    "enabled": False,
                    "status": "disabled",
                    "alpha": 1.0,
                    "dt_sec": None,
                    "raw_dt_sec": None,
                    "translation_error_m": 0.0,
                    "rotation_error_deg": 0.0,
                },
            )

        timestamp_value = self._finite_timestamp(timestamp)
        raw_dt = self._raw_dt(timestamp_value)
        reset_reason = self._reset_reason(raw_dt)
        if self._last_pose is None or reset_reason is not None:
            self._last_pose = raw_pose.copy()
            self._last_timestamp = timestamp_value
            return PoseSmoothingResult(
                pose=raw_pose,
                metadata={
                    "enabled": True,
                    "status": "initialized" if reset_reason is None else reset_reason,
                    "alpha": 1.0,
                    "dt_sec": None,
                    "raw_dt_sec": raw_dt,
                    "translation_error_m": 0.0,
                    "rotation_error_deg": 0.0,
                },
            )

        assert self._last_pose is not None
        translation_error_m = translation_distance(self._last_pose, raw_pose)
        rotation_error_deg = math.degrees(rotation_angle_rad(self._last_pose, raw_pose))
        effective_dt = self._effective_dt(raw_dt)

        if self._inside_deadband(translation_error_m, rotation_error_deg):
            output = self._last_pose.copy()
            self._last_timestamp = timestamp_value
            return PoseSmoothingResult(
                pose=output,
                metadata={
                    "enabled": True,
                    "status": "deadband_hold",
                    "alpha": 0.0,
                    "dt_sec": effective_dt,
                    "raw_dt_sec": raw_dt,
                    "translation_error_m": float(translation_error_m),
                    "rotation_error_deg": float(rotation_error_deg),
                },
            )

        if self._should_snap(translation_error_m, rotation_error_deg):
            self._last_pose = raw_pose.copy()
            self._last_timestamp = timestamp_value
            return PoseSmoothingResult(
                pose=raw_pose,
                metadata={
                    "enabled": True,
                    "status": "snap",
                    "alpha": 1.0,
                    "dt_sec": effective_dt,
                    "raw_dt_sec": raw_dt,
                    "translation_error_m": float(translation_error_m),
                    "rotation_error_deg": float(rotation_error_deg),
                },
            )

        alpha = self._alpha(effective_dt)
        output = interpolate_pose(self._last_pose, raw_pose, alpha)
        self._last_pose = output.copy()
        self._last_timestamp = timestamp_value
        return PoseSmoothingResult(
            pose=output,
            metadata={
                "enabled": True,
                "status": "smoothed",
                "alpha": float(alpha),
                "dt_sec": effective_dt,
                "raw_dt_sec": raw_dt,
                "translation_error_m": float(translation_error_m),
                "rotation_error_deg": float(rotation_error_deg),
            },
        )

    def _finite_timestamp(self, timestamp: float | None) -> float | None:
        if timestamp is None:
            return None
        value = float(timestamp)
        return value if np.isfinite(value) else None

    def _raw_dt(self, timestamp: float | None) -> float | None:
        if timestamp is None or self._last_timestamp is None:
            return None
        return float(timestamp - self._last_timestamp)

    def _reset_reason(self, raw_dt: float | None) -> str | None:
        if raw_dt is None:
            return None
        if raw_dt < 0.0:
            return "reset_timestamp_rewind"
        if self.config.reset_after_gap_sec > 0.0 and raw_dt > self.config.reset_after_gap_sec:
            return "reset_after_gap"
        return None

    def _effective_dt(self, raw_dt: float | None) -> float:
        dt = self.config.nominal_dt_sec if raw_dt is None or raw_dt <= 0.0 else raw_dt
        dt = max(float(self.config.min_dt_sec), float(dt))
        return float(min(dt, self.config.max_dt_sec))

    def _alpha(self, dt_sec: float) -> float:
        if self.config.time_constant_sec <= 0.0:
            return 1.0
        return float(1.0 - math.exp(-float(dt_sec) / float(self.config.time_constant_sec)))

    def _inside_deadband(
        self,
        translation_error_m: float,
        rotation_error_deg: float,
    ) -> bool:
        return (
            translation_error_m <= self.config.deadband_translation_m
            and rotation_error_deg <= self.config.deadband_rotation_deg
        )

    def _should_snap(
        self,
        translation_error_m: float,
        rotation_error_deg: float,
    ) -> bool:
        return (
            self.config.snap_translation_m > 0.0
            and translation_error_m >= self.config.snap_translation_m
        ) or (
            self.config.snap_rotation_deg > 0.0
            and rotation_error_deg >= self.config.snap_rotation_deg
        )


class BimanualPoseSmoother:
    def __init__(self, config: PoseSmoothingConfig) -> None:
        self.config = config
        self._smoothers = {
            "left": PoseSmoother(config),
            "right": PoseSmoother(config),
        }

    def reset(self) -> None:
        for smoother in self._smoothers.values():
            smoother.reset()

    def smooth_pair(
        self,
        *,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_active: bool = True,
        right_active: bool = True,
        timestamp: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        left = self._smooth_side(
            "left",
            left_target,
            active=left_active,
            timestamp=timestamp,
        )
        right = self._smooth_side(
            "right",
            right_target,
            active=right_active,
            timestamp=timestamp,
        )
        return (
            {"left": left.pose.copy(), "right": right.pose.copy()},
            {
                "enabled": bool(self.config.enabled),
                "left": dict(left.metadata),
                "right": dict(right.metadata),
            },
        )

    def _smooth_side(
        self,
        side: str,
        target: np.ndarray,
        *,
        active: bool,
        timestamp: float | None,
    ) -> PoseSmoothingResult:
        if active:
            return self._smoothers[side].smooth(target, timestamp=timestamp)
        self._smoothers[side].reset()
        raw_pose = parse_pose_matrix(target)
        return PoseSmoothingResult(
            pose=raw_pose,
            metadata={
                "enabled": bool(self.config.enabled),
                "status": "inactive_passthrough",
                "alpha": 1.0,
                "dt_sec": None,
                "raw_dt_sec": None,
                "translation_error_m": 0.0,
                "rotation_error_deg": 0.0,
            },
        )


def load_pose_smoothing_config(path: str | Path) -> PoseSmoothingConfig:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved} must contain a JSON object.")
    return PoseSmoothingConfig.from_mapping(payload, source_path=str(resolved))


def build_bimanual_pose_smoother(
    config: PoseSmoothingConfig | None = None,
) -> BimanualPoseSmoother:
    return BimanualPoseSmoother(config or PoseSmoothingConfig())


__all__ = [
    "SCHEMA_VERSION",
    "RM65_POSITION_TARGET_POSE_SMOOTHING_V1",
    "BimanualPoseSmoother",
    "PoseSmoother",
    "PoseSmoothingConfig",
    "PoseSmoothingResult",
    "build_bimanual_pose_smoother",
    "load_pose_smoothing_config",
]

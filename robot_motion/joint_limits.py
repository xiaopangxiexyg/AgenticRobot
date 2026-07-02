from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointVelocityLimitProfile:
    name: str
    max_velocity_deg_s: tuple[float, ...]
    max_dt_sec: float = 1.0
    joint_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.max_velocity_deg_s, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("max_velocity_deg_s must contain positive finite values")
        if not math.isfinite(float(self.max_dt_sec)) or float(self.max_dt_sec) <= 0.0:
            raise ValueError("max_dt_sec must be positive and finite")
        if self.joint_names and len(self.joint_names) != values.size:
            raise ValueError("joint_names must match max_velocity_deg_s length")

    @property
    def dof(self) -> int:
        return len(self.max_velocity_deg_s)

    def max_step_deg(self, dt_sec: float) -> np.ndarray:
        dt = self.effective_dt_sec(dt_sec)
        return np.asarray(self.max_velocity_deg_s, dtype=float).reshape(-1) * dt

    def effective_dt_sec(self, dt_sec: float) -> float:
        dt = float(dt_sec)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be positive")
        return min(dt, float(self.max_dt_sec))


@dataclass(frozen=True)
class JointLimitStep:
    qpos: np.ndarray
    limited: bool
    delta_deg: np.ndarray
    max_step_deg: np.ndarray
    limit_ratio: float

    @property
    def max_abs_delta_deg(self) -> float:
        return float(np.max(np.abs(self.delta_deg))) if self.delta_deg.size else 0.0

    @property
    def max_abs_allowed_step_deg(self) -> float:
        return float(np.max(self.max_step_deg)) if self.max_step_deg.size else 0.0


class JointRateLimiter:
    """Per-joint velocity limiter that always chases the newest target.

    The limiter keeps only the last emitted command. A caller may pass a fresh
    target every cycle; no queued old target is replayed.
    """

    def __init__(self, profile: JointVelocityLimitProfile):
        self.profile = profile
        self._command_qpos: np.ndarray | None = None

    def reset(self, qpos: np.ndarray | list[float] | tuple[float, ...] | None = None) -> None:
        self._command_qpos = None if qpos is None else _as_qpos(qpos, self.profile.dof).copy()

    @property
    def command_qpos(self) -> np.ndarray | None:
        return None if self._command_qpos is None else self._command_qpos.copy()

    def step(
        self,
        *,
        target_qpos: np.ndarray | list[float] | tuple[float, ...],
        dt_sec: float,
        reference_qpos: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> JointLimitStep:
        if self._command_qpos is None:
            if reference_qpos is None:
                raise ValueError("reference_qpos is required before the limiter has emitted a command")
            self._command_qpos = _as_qpos(reference_qpos, self.profile.dof).copy()
        result = limit_joint_step(
            reference_qpos=self._command_qpos,
            target_qpos=target_qpos,
            profile=self.profile,
            dt_sec=dt_sec,
        )
        self._command_qpos = result.qpos.copy()
        return result


def default_joint_velocity_limits_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "joint_velocity_limits.json"


def profile_from_dict(name: str, payload: dict[str, Any]) -> JointVelocityLimitProfile:
    raw_velocity = payload.get("max_velocity_deg_s")
    if raw_velocity is None:
        raise ValueError(f"joint velocity profile {name!r} missing max_velocity_deg_s")
    return JointVelocityLimitProfile(
        name=name,
        max_velocity_deg_s=tuple(float(value) for value in raw_velocity),
        max_dt_sec=float(payload.get("max_dt_sec", 1.0)),
        joint_names=tuple(str(value) for value in payload.get("joint_names", ())),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def load_joint_velocity_limit_profiles(path: str | Path | None = None) -> dict[str, JointVelocityLimitProfile]:
    resolved = default_joint_velocity_limits_path() if path in (None, "") else Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{resolved} does not contain any joint velocity limit profiles")
    return {
        str(name): profile_from_dict(str(name), dict(profile_payload))
        for name, profile_payload in profiles.items()
    }


def limit_joint_step(
    *,
    reference_qpos: np.ndarray | list[float] | tuple[float, ...],
    target_qpos: np.ndarray | list[float] | tuple[float, ...],
    profile: JointVelocityLimitProfile,
    dt_sec: float,
) -> JointLimitStep:
    reference = _as_qpos(reference_qpos, profile.dof)
    target = _as_qpos(target_qpos, profile.dof)
    max_step_deg = profile.max_step_deg(dt_sec)
    delta = target - reference
    delta_deg = np.rad2deg(delta)
    clipped_delta_deg = np.clip(delta_deg, -max_step_deg, max_step_deg)
    limited = bool(np.any(np.abs(delta_deg) > max_step_deg + 1e-9))
    qpos = reference + np.deg2rad(clipped_delta_deg)
    ratios = np.divide(
        np.abs(delta_deg),
        max_step_deg,
        out=np.zeros_like(delta_deg, dtype=float),
        where=max_step_deg > 0.0,
    )
    return JointLimitStep(
        qpos=qpos,
        limited=limited,
        delta_deg=clipped_delta_deg,
        max_step_deg=max_step_deg,
        limit_ratio=float(np.max(ratios)) if ratios.size else 0.0,
    )


def plan_limited_joint_path(
    *,
    start_qpos: np.ndarray | list[float] | tuple[float, ...],
    target_qpos: np.ndarray | list[float] | tuple[float, ...],
    profile: JointVelocityLimitProfile,
    dt_sec: float,
    max_steps: int = 10000,
) -> list[JointLimitStep]:
    current = _as_qpos(start_qpos, profile.dof)
    target = _as_qpos(target_qpos, profile.dof)
    steps: list[JointLimitStep] = []
    for _ in range(int(max_steps)):
        result = limit_joint_step(
            reference_qpos=current,
            target_qpos=target,
            profile=profile,
            dt_sec=dt_sec,
        )
        steps.append(result)
        current = result.qpos
        if not result.limited:
            return steps
    raise RuntimeError(f"joint limit path did not converge within max_steps={max_steps}")


def _as_qpos(values: np.ndarray | list[float] | tuple[float, ...], dof: int) -> np.ndarray:
    qpos = np.asarray(values, dtype=float).reshape(-1)
    if qpos.shape != (int(dof),) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"qpos must contain {dof} finite values, got shape={qpos.shape}")
    return qpos

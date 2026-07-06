from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.joint_limits import (
    JointLimitStep,
    JointVelocityLimitProfile,
    load_joint_velocity_limit_profiles,
    plan_limited_joint_path,
)


@dataclass(frozen=True)
class ArmQposTransitionConfig:
    dt_sec: float = 0.05
    joint_velocity_profile_name: str = "rm65_init_slow"
    joint_velocity_profile: JointVelocityLimitProfile | None = None
    max_steps_per_side: int = 10000

    def __post_init__(self) -> None:
        dt = float(self.dt_sec)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be positive and finite.")
        if int(self.max_steps_per_side) < 1:
            raise ValueError("max_steps_per_side must be >= 1.")
        object.__setattr__(self, "dt_sec", dt)
        object.__setattr__(self, "joint_velocity_profile_name", str(self.joint_velocity_profile_name))
        object.__setattr__(self, "max_steps_per_side", int(self.max_steps_per_side))

    def resolved_joint_velocity_profile(self) -> JointVelocityLimitProfile:
        if self.joint_velocity_profile is not None:
            return self.joint_velocity_profile
        profiles = load_joint_velocity_limit_profiles()
        try:
            return profiles[self.joint_velocity_profile_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown joint velocity profile {self.joint_velocity_profile_name!r}; "
                f"available={sorted(profiles)}"
            ) from exc


@dataclass(frozen=True)
class ArmQposTransitionStep:
    index: int
    time_sec: float
    qpos_by_side: dict[str, np.ndarray]
    limited_by_side: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.index) < 1:
            raise ValueError("transition step index must be >= 1.")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "time_sec", float(self.time_sec))
        object.__setattr__(self, "qpos_by_side", validate_qpos_mapping(self.qpos_by_side))
        object.__setattr__(
            self,
            "limited_by_side",
            {str(side): bool(value) for side, value in self.limited_by_side.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def qpos_degrees_by_side(self) -> dict[str, list[float]]:
        return {
            side: np.rad2deg(qpos).astype(float).tolist()
            for side, qpos in self.qpos_by_side.items()
        }


@dataclass(frozen=True)
class ArmQposTransitionPlan:
    sides: tuple[str, ...]
    start_qpos_by_side: dict[str, np.ndarray]
    target_qpos_by_side: dict[str, np.ndarray]
    steps: tuple[ArmQposTransitionStep, ...]
    joint_paths_by_side: dict[str, list[JointLimitStep]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        if not self.steps:
            return all(
                np.allclose(self.start_qpos_by_side[side], self.target_qpos_by_side[side])
                for side in self.sides
            )
        final = self.steps[-1].qpos_by_side
        return all(np.allclose(final[side], self.target_qpos_by_side[side]) for side in self.sides)

    @property
    def final_qpos_by_side(self) -> dict[str, np.ndarray]:
        if not self.steps:
            return {side: qpos.copy() for side, qpos in self.start_qpos_by_side.items()}
        return {side: qpos.copy() for side, qpos in self.steps[-1].qpos_by_side.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "sides": list(self.sides),
            "completed": bool(self.completed),
            "step_count": len(self.steps),
            "duration_sec": float(self.steps[-1].time_sec) if self.steps else 0.0,
            "path_lengths": {side: len(path) for side, path in self.joint_paths_by_side.items()},
            "metadata": dict(self.metadata),
        }


def validate_qpos_mapping(qpos_by_side: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(qpos_by_side, Mapping) or not qpos_by_side:
        raise ValueError("qpos mapping must be a non-empty mapping.")
    out: dict[str, np.ndarray] = {}
    for raw_side, values in qpos_by_side.items():
        side = str(raw_side).strip()
        if not side:
            raise ValueError("qpos side/name must be non-empty.")
        if side in out:
            raise ValueError(f"duplicate qpos side/name {side!r}.")
        qpos = np.asarray(values, dtype=float).reshape(-1)
        if qpos.size == 0 or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{side}.qpos must contain finite values.")
        out[side] = qpos.astype(float, copy=True)
    return out


def load_qpos_by_side_file(path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    raw: Mapping[str, Any]
    if isinstance(payload.get("qpos_by_side"), Mapping):
        raw = payload["qpos_by_side"]
    elif isinstance(payload.get("arms"), Mapping):
        raw = {
            str(side): arm_payload.get("qpos")
            for side, arm_payload in payload["arms"].items()
            if isinstance(arm_payload, Mapping)
        }
    else:
        raw = payload
    return validate_qpos_mapping(raw)


def plan_slow_qpos_transition(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    config: ArmQposTransitionConfig | None = None,
    side_order: Sequence[str] | None = None,
) -> ArmQposTransitionPlan:
    cfg = config if config is not None else ArmQposTransitionConfig()
    start = validate_qpos_mapping(start_qpos_by_side)
    target = validate_qpos_mapping(target_qpos_by_side)
    sides = _ordered_common_sides(start, target, side_order=side_order)
    profile = cfg.resolved_joint_velocity_profile()
    _require_profile_matches_qpos(profile, start, sides)
    _require_profile_matches_qpos(profile, target, sides)

    joint_paths: dict[str, list[JointLimitStep]] = {}
    for side in sides:
        joint_paths[side] = plan_limited_joint_path(
            start_qpos=start[side],
            target_qpos=target[side],
            profile=profile,
            dt_sec=float(cfg.dt_sec),
            max_steps=int(cfg.max_steps_per_side),
        )

    steps: list[ArmQposTransitionStep] = []
    step_count = max((len(path) for path in joint_paths.values()), default=0)
    for index in range(1, step_count + 1):
        qpos_by_side: dict[str, np.ndarray] = {}
        limited_by_side: dict[str, bool] = {}
        per_side_metadata: dict[str, Any] = {}
        for side in sides:
            path = joint_paths[side]
            step = path[min(index, len(path)) - 1]
            qpos_by_side[side] = step.qpos.copy()
            limited_by_side[side] = bool(step.limited)
            per_side_metadata[side] = {
                "delta_deg": step.delta_deg.astype(float).tolist(),
                "max_step_deg": step.max_step_deg.astype(float).tolist(),
                "limit_ratio": float(step.limit_ratio),
            }
        steps.append(
            ArmQposTransitionStep(
                index=index,
                time_sec=float(index) * float(cfg.dt_sec),
                qpos_by_side=qpos_by_side,
                limited_by_side=limited_by_side,
                metadata={"per_side": per_side_metadata},
            )
        )

    return ArmQposTransitionPlan(
        sides=sides,
        start_qpos_by_side={side: start[side].copy() for side in sides},
        target_qpos_by_side={side: target[side].copy() for side in sides},
        steps=tuple(steps),
        joint_paths_by_side=joint_paths,
        metadata={
            "planner": "plan_slow_qpos_transition",
            "dt_sec": float(cfg.dt_sec),
            "joint_velocity_profile": profile.name,
        },
    )


def qpos_transition_plan_to_dict(
    plan: ArmQposTransitionPlan,
    *,
    include_steps: bool = True,
    max_steps: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_type": "robot_motion_arm_init_qpos_transition_plan",
        "schema_version": 1,
        "summary": plan.summary(),
        "start_qpos_by_side": _qpos_mapping_to_lists(plan.start_qpos_by_side),
        "target_qpos_by_side": _qpos_mapping_to_lists(plan.target_qpos_by_side),
        "final_qpos_by_side": _qpos_mapping_to_lists(plan.final_qpos_by_side),
    }
    if include_steps:
        limit = len(plan.steps) if int(max_steps) <= 0 else min(int(max_steps), len(plan.steps))
        payload["steps"] = [_step_to_dict(step) for step in plan.steps[:limit]]
        payload["steps_truncated"] = limit < len(plan.steps)
    return payload


def _ordered_common_sides(
    start: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    *,
    side_order: Sequence[str] | None,
) -> tuple[str, ...]:
    common = tuple(side for side in start if side in target)
    if not common:
        raise ValueError("start and target qpos mappings must share at least one side/name.")
    missing_start = [side for side in target if side not in start]
    missing_target = [side for side in start if side not in target]
    if missing_start or missing_target:
        raise ValueError(
            "start and target qpos mappings must contain the same sides/names; "
            f"missing_start={missing_start}, missing_target={missing_target}."
        )
    if side_order is None:
        return common
    requested = tuple(str(side).strip() for side in side_order if str(side).strip() in common)
    return requested + tuple(side for side in common if side not in set(requested))


def _require_profile_matches_qpos(
    profile: JointVelocityLimitProfile,
    qpos_by_side: Mapping[str, np.ndarray],
    sides: Sequence[str],
) -> None:
    for side in sides:
        if np.asarray(qpos_by_side[side], dtype=float).reshape(-1).shape != (profile.dof,):
            raise ValueError(
                f"{side}.qpos has dof={np.asarray(qpos_by_side[side]).reshape(-1).size}, "
                f"but velocity profile {profile.name!r} has dof={profile.dof}."
            )


def _qpos_mapping_to_lists(qpos_by_side: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        side: np.asarray(qpos, dtype=float).reshape(-1).astype(float).tolist()
        for side, qpos in qpos_by_side.items()
    }


def _step_to_dict(step: ArmQposTransitionStep) -> dict[str, Any]:
    return {
        "index": int(step.index),
        "time_sec": float(step.time_sec),
        "qpos_by_side": _qpos_mapping_to_lists(step.qpos_by_side),
        "qpos_deg_by_side": step.qpos_degrees_by_side(),
        "limited_by_side": dict(step.limited_by_side),
        "metadata": dict(step.metadata),
    }


__all__ = [
    "ArmQposTransitionConfig",
    "ArmQposTransitionPlan",
    "ArmQposTransitionStep",
    "load_qpos_by_side_file",
    "plan_slow_qpos_transition",
    "qpos_transition_plan_to_dict",
    "validate_qpos_mapping",
]

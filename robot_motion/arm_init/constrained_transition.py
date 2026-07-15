from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

import numpy as np

from robot_motion.pose import as_strict_pose_matrix

from .qpos_transition import validate_qpos_mapping


class ArmTransitionRiskLevel(IntEnum):
    NORMAL = 0
    SLOWDOWN = 1
    HOLD = 2
    STOP = 3

    @property
    def label(self) -> str:
        return ("normal", "slowdown", "hold", "stop")[int(self)]


@dataclass(frozen=True)
class ArmTransitionRiskEvent:
    level: ArmTransitionRiskLevel
    code: str
    message: str
    side: str | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": int(self.level),
            "level_label": self.level.label,
            "code": str(self.code),
            "message": str(self.message),
        }
        if self.side is not None:
            payload["side"] = str(self.side)
        if self.value is not None:
            payload["value"] = float(self.value)
        if self.threshold is not None:
            payload["threshold"] = float(self.threshold)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ArmTransitionRiskAssessment:
    events: tuple[ArmTransitionRiskEvent, ...] = ()
    speed_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> ArmTransitionRiskLevel:
        return max((event.level for event in self.events), default=ArmTransitionRiskLevel.NORMAL)

    @property
    def hold(self) -> bool:
        return self.level == ArmTransitionRiskLevel.HOLD

    @property
    def stop(self) -> bool:
        return self.level == ArmTransitionRiskLevel.STOP

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_label": self.level.label,
            "speed_scale": float(self.speed_scale),
            "hold": bool(self.hold),
            "stop": bool(self.stop),
            "events": [event.to_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ArmRuntimeSignals:
    force_norm_n: float | None = None
    torque_norm_nm: float | None = None
    tracking_error_m: float | None = None
    control_error: bool = False
    suspected_collision: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmConstrainedTransitionConfig:
    dt_sec: float = 0.05
    max_joint_step_deg: float = 2.0
    target_tolerance_rad: float = 1e-4
    max_steps: int = 1200
    max_joint_abs_deg: float = 120.0
    joint_slow_margin_deg: float = 12.0
    joint_hold_margin_deg: float = 3.0
    max_endpoint_excursion_m: float = 0.5
    endpoint_slow_margin_m: float = 0.08
    endpoint_hold_margin_m: float = 0.02
    collision_slow_distance_m: float = 0.05
    collision_hold_distance_m: float = 0.02
    collision_stop_distance_m: float = 0.012
    singularity_slow_margin: float = 0.03
    singularity_hold_margin: float = 0.006
    singularity_stop_margin: float | None = None
    level1_speed_scale: float = 0.35
    level2_speed_scale: float = 0.0
    force_slow_n: float | None = 8.0
    force_soft_n: float | None = 15.0
    force_hard_n: float | None = 25.0
    torque_slow_nm: float | None = 1.0
    torque_soft_nm: float | None = 2.0
    torque_hard_nm: float | None = 3.5
    tracking_error_hold_m: float | None = 0.08
    tracking_error_stop_m: float | None = 0.18
    backoff_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    require_kinematic_model: bool = True

    def __post_init__(self) -> None:
        for name in (
            "dt_sec",
            "max_joint_step_deg",
            "target_tolerance_rad",
            "max_joint_abs_deg",
            "max_endpoint_excursion_m",
            "collision_slow_distance_m",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be >= 1.")
        object.__setattr__(self, "max_steps", int(self.max_steps))
        for name in (
            "joint_slow_margin_deg",
            "joint_hold_margin_deg",
            "endpoint_slow_margin_m",
            "endpoint_hold_margin_m",
            "collision_hold_distance_m",
            "collision_stop_distance_m",
            "singularity_slow_margin",
            "singularity_hold_margin",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be >= 0.")
            object.__setattr__(self, name, value)
        if self.collision_stop_distance_m > self.collision_hold_distance_m:
            raise ValueError("collision_stop_distance_m must be <= collision_hold_distance_m.")
        if self.collision_hold_distance_m > self.collision_slow_distance_m:
            raise ValueError("collision_hold_distance_m must be <= collision_slow_distance_m.")
        for name in ("level1_speed_scale", "level2_speed_scale"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be within [0, 1].")
            object.__setattr__(self, name, value)
        _validate_optional_upper_thresholds("force", self.force_slow_n, self.force_soft_n, self.force_hard_n)
        _validate_optional_upper_thresholds("torque", self.torque_slow_nm, self.torque_soft_nm, self.torque_hard_nm)
        _validate_optional_upper_thresholds(
            "tracking_error",
            None,
            self.tracking_error_hold_m,
            self.tracking_error_stop_m,
        )
        if self.singularity_stop_margin is not None:
            stop_margin = float(self.singularity_stop_margin)
            if stop_margin < 0.0 or not np.isfinite(stop_margin):
                raise ValueError("singularity_stop_margin must be >= 0 when set.")
            if stop_margin > self.singularity_hold_margin:
                raise ValueError("singularity_stop_margin must be <= singularity_hold_margin.")
            object.__setattr__(self, "singularity_stop_margin", stop_margin)
        scales = tuple(float(value) for value in self.backoff_scales)
        scales = tuple(value for value in scales if np.isfinite(value) and 0.0 < value <= 1.0)
        if not scales:
            raise ValueError("backoff_scales must contain at least one value in (0, 1].")
        object.__setattr__(self, "backoff_scales", tuple(sorted(set(scales), reverse=True)))
        object.__setattr__(self, "require_kinematic_model", bool(self.require_kinematic_model))


class ArmTransitionModel(Protocol):
    def fk_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        ...

    def min_collision_distance(self, qpos_by_side: Mapping[str, np.ndarray]) -> float | None:
        ...

    def singularity_margin_by_side(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ArmTransitionPlanStep:
    index: int
    time_sec: float
    qpos_by_side: dict[str, np.ndarray]
    assessment: ArmTransitionRiskAssessment
    alpha: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time_sec": float(self.time_sec),
            "alpha": float(self.alpha),
            "risk": self.assessment.to_dict(),
            "qpos_by_side": {
                side: np.asarray(qpos, dtype=float).reshape(-1).tolist()
                for side, qpos in self.qpos_by_side.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ArmConstrainedTransitionPlan:
    status: str
    start_qpos_by_side: dict[str, np.ndarray]
    target_qpos_by_side: dict[str, np.ndarray]
    steps: tuple[ArmTransitionPlanStep, ...]
    initial_assessment: ArmTransitionRiskAssessment
    target_assessment: ArmTransitionRiskAssessment
    blocked_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def final_qpos_by_side(self) -> dict[str, np.ndarray]:
        if self.steps:
            return {side: qpos.copy() for side, qpos in self.steps[-1].qpos_by_side.items()}
        return {side: qpos.copy() for side, qpos in self.start_qpos_by_side.items()}

    def summary(self) -> dict[str, Any]:
        max_level = max(
            [self.initial_assessment.level, self.target_assessment.level]
            + [step.assessment.level for step in self.steps],
            default=ArmTransitionRiskLevel.NORMAL,
        )
        return {
            "status": str(self.status),
            "completed": bool(self.completed),
            "blocked_reason": self.blocked_reason,
            "step_count": len(self.steps),
            "duration_sec": float(self.steps[-1].time_sec) if self.steps else 0.0,
            "max_risk_level": int(max_level),
            "max_risk_label": max_level.label,
            "metadata": dict(self.metadata),
        }

    def to_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": str(self.status),
            "completed": bool(self.completed),
            "blocked_reason": self.blocked_reason,
            "summary": self.summary(),
            "start_qpos_by_side": _qpos_mapping_to_lists(self.start_qpos_by_side),
            "target_qpos_by_side": _qpos_mapping_to_lists(self.target_qpos_by_side),
            "initial_assessment": self.initial_assessment.to_dict(),
            "target_assessment": self.target_assessment.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_steps:
            payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


def evaluate_transition_state(
    *,
    qpos_by_side: Mapping[str, Any],
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    sides: Sequence[str] | None = None,
    model: ArmTransitionModel | None,
    config: ArmConstrainedTransitionConfig | None = None,
    start_pose_by_side: Mapping[str, Any] | None = None,
    target_pose_by_side: Mapping[str, Any] | None = None,
    runtime_signals: ArmRuntimeSignals | None = None,
) -> ArmTransitionRiskAssessment:
    cfg = config if config is not None else ArmConstrainedTransitionConfig()
    qpos = _validate_for_sides(qpos_by_side, sides=sides)
    start = _validate_for_sides(start_qpos_by_side, sides=tuple(qpos))
    target = _validate_for_sides(target_qpos_by_side, sides=tuple(qpos))
    del start, target
    events: list[ArmTransitionRiskEvent] = []
    metadata: dict[str, Any] = {}
    _append_joint_limit_events(events, qpos, cfg)

    if model is None:
        metadata["kinematic_model"] = "missing"
        if cfg.require_kinematic_model:
            events.append(
                ArmTransitionRiskEvent(
                    level=ArmTransitionRiskLevel.STOP,
                    code="missing_kinematic_model",
                    message="transition evaluation requires FK and collision model",
                )
            )
    else:
        fk = _fk_by_side(model, qpos)
        if start_pose_by_side is None:
            start_pose_by_side = _fk_by_side(model, start_qpos_by_side)
        if target_pose_by_side is None:
            target_pose_by_side = _fk_by_side(model, target_qpos_by_side)
        _append_endpoint_excursion_events(events, fk, start_pose_by_side, target_pose_by_side, cfg, metadata)
        _append_collision_events(events, model, qpos, cfg, metadata)
        _append_singularity_events(events, model, qpos, cfg, metadata)

    if runtime_signals is not None:
        _append_runtime_signal_events(events, runtime_signals, cfg, metadata)

    level = max((event.level for event in events), default=ArmTransitionRiskLevel.NORMAL)
    speed_scale = (
        1.0
        if level == ArmTransitionRiskLevel.NORMAL
        else float(cfg.level1_speed_scale)
        if level == ArmTransitionRiskLevel.SLOWDOWN
        else float(cfg.level2_speed_scale)
        if level == ArmTransitionRiskLevel.HOLD
        else 0.0
    )
    return ArmTransitionRiskAssessment(events=tuple(events), speed_scale=speed_scale, metadata=metadata)


def plan_constrained_qpos_transition(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    sides: Sequence[str] | None = None,
    model: ArmTransitionModel | None,
    config: ArmConstrainedTransitionConfig | None = None,
) -> ArmConstrainedTransitionPlan:
    cfg = config if config is not None else ArmConstrainedTransitionConfig()
    start = _validate_for_sides(start_qpos_by_side, sides=sides)
    target = _validate_for_sides(target_qpos_by_side, sides=tuple(start))
    _require_same_shapes(start, target)
    metadata = {
        "dt_sec": float(cfg.dt_sec),
        "max_joint_step_deg": float(cfg.max_joint_step_deg),
        "constraint_space": "qpos",
    }
    start_poses = _fk_by_side(model, start) if model is not None else None
    target_poses = _fk_by_side(model, target) if model is not None else None
    initial_assessment = evaluate_transition_state(
        qpos_by_side=start,
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        sides=tuple(start),
        model=model,
        config=cfg,
        start_pose_by_side=start_poses,
        target_pose_by_side=target_poses,
    )
    target_assessment = evaluate_transition_state(
        qpos_by_side=target,
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        sides=tuple(start),
        model=model,
        config=cfg,
        start_pose_by_side=start_poses,
        target_pose_by_side=target_poses,
    )
    if initial_assessment.stop:
        return _blocked_plan("blocked_start", start, target, (), initial_assessment, target_assessment, _first_event_code(initial_assessment), metadata)
    if target_assessment.stop:
        return _blocked_plan("blocked_target", start, target, (), initial_assessment, target_assessment, _first_event_code(target_assessment), metadata)

    current = {side: qpos.copy() for side, qpos in start.items()}
    steps: list[ArmTransitionPlanStep] = []
    max_step_rad = np.deg2rad(float(cfg.max_joint_step_deg))
    for _iteration in range(int(cfg.max_steps)):
        remaining = {side: target[side] - current[side] for side in current}
        remaining_norm = _max_abs_qpos_delta(remaining)
        if remaining_norm <= float(cfg.target_tolerance_rad):
            return ArmConstrainedTransitionPlan(
                status="completed",
                start_qpos_by_side=start,
                target_qpos_by_side=target,
                steps=tuple(steps),
                initial_assessment=initial_assessment,
                target_assessment=target_assessment,
                metadata=metadata,
            )
        current_assessment = evaluate_transition_state(
            qpos_by_side=current,
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            sides=tuple(start),
            model=model,
            config=cfg,
            start_pose_by_side=start_poses,
            target_pose_by_side=target_poses,
        )
        if current_assessment.stop or current_assessment.hold:
            status = "blocked_current" if current_assessment.stop else "hold_current"
            return _blocked_plan(status, start, target, steps, initial_assessment, target_assessment, _first_event_code(current_assessment), metadata)

        base_alpha = min(1.0, max_step_rad / max(remaining_norm, 1e-12))
        accepted: tuple[dict[str, np.ndarray], ArmTransitionRiskAssessment, float] | None = None
        for backoff in cfg.backoff_scales:
            alpha = min(1.0, base_alpha * float(backoff) * max(float(current_assessment.speed_scale), 1e-6))
            candidate = {side: current[side] + remaining[side] * alpha for side in current}
            assessment = evaluate_transition_state(
                qpos_by_side=candidate,
                start_qpos_by_side=start,
                target_qpos_by_side=target,
                sides=tuple(start),
                model=model,
                config=cfg,
                start_pose_by_side=start_poses,
                target_pose_by_side=target_poses,
            )
            if assessment.stop or assessment.hold:
                continue
            accepted = (candidate, assessment, alpha)
            break
        if accepted is None:
            return _blocked_plan("blocked_candidate", start, target, steps, initial_assessment, target_assessment, "no_constraint_satisfying_step", metadata)
        candidate, assessment, alpha = accepted
        current = {side: qpos.copy() for side, qpos in candidate.items()}
        steps.append(
            ArmTransitionPlanStep(
                index=len(steps) + 1,
                time_sec=(len(steps) + 1) * float(cfg.dt_sec),
                qpos_by_side={side: qpos.copy() for side, qpos in current.items()},
                assessment=assessment,
                alpha=_path_alpha(start, target, current),
                metadata={"local_alpha": float(alpha), "remaining_max_abs_rad_before_step": float(remaining_norm)},
            )
        )
    return _blocked_plan("max_steps_exceeded", start, target, steps, initial_assessment, target_assessment, "max_steps_exceeded", metadata)


def _validate_for_sides(qpos_by_side: Mapping[str, Any], *, sides: Sequence[str] | None) -> dict[str, np.ndarray]:
    qpos = validate_qpos_mapping(qpos_by_side)
    if sides is None:
        return qpos
    out: dict[str, np.ndarray] = {}
    for raw_side in sides:
        side = str(raw_side)
        if side not in qpos:
            raise ValueError(f"qpos mapping missing side/name {side!r}.")
        out[side] = qpos[side]
    return out


def _blocked_plan(
    status: str,
    start: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    steps: Sequence[ArmTransitionPlanStep],
    initial: ArmTransitionRiskAssessment,
    target_assessment: ArmTransitionRiskAssessment,
    reason: str | None,
    metadata: Mapping[str, Any],
) -> ArmConstrainedTransitionPlan:
    return ArmConstrainedTransitionPlan(
        status=status,
        start_qpos_by_side={side: qpos.copy() for side, qpos in start.items()},
        target_qpos_by_side={side: qpos.copy() for side, qpos in target.items()},
        steps=tuple(steps),
        initial_assessment=initial,
        target_assessment=target_assessment,
        blocked_reason=reason,
        metadata=dict(metadata),
    )


def _append_joint_limit_events(events: list[ArmTransitionRiskEvent], qpos_by_side: Mapping[str, np.ndarray], cfg: ArmConstrainedTransitionConfig) -> None:
    hard = np.deg2rad(float(cfg.max_joint_abs_deg))
    slow = np.deg2rad(max(0.0, float(cfg.max_joint_abs_deg) - float(cfg.joint_slow_margin_deg)))
    hold = np.deg2rad(max(0.0, float(cfg.max_joint_abs_deg) - float(cfg.joint_hold_margin_deg)))
    for side, qpos in qpos_by_side.items():
        for joint_index, value in enumerate(np.abs(np.asarray(qpos, dtype=float).reshape(-1)), start=1):
            if value > hard:
                level, code, threshold = ArmTransitionRiskLevel.STOP, "joint_limit_hard", hard
            elif value >= hold:
                level, code, threshold = ArmTransitionRiskLevel.HOLD, "joint_limit_hold_margin", hold
            elif value >= slow:
                level, code, threshold = ArmTransitionRiskLevel.SLOWDOWN, "joint_limit_slow_margin", slow
            else:
                continue
            events.append(
                ArmTransitionRiskEvent(
                    level=level,
                    code=code,
                    message=f"{side} joint {joint_index} is near or outside absolute joint limit",
                    side=side,
                    value=float(value),
                    threshold=float(threshold),
                    metadata={"joint_index": joint_index},
                )
            )


def _append_endpoint_excursion_events(
    events: list[ArmTransitionRiskEvent],
    pose_by_side: Mapping[str, Any],
    start_pose_by_side: Mapping[str, Any],
    target_pose_by_side: Mapping[str, Any],
    cfg: ArmConstrainedTransitionConfig,
    metadata: dict[str, Any],
) -> None:
    hard = float(cfg.max_endpoint_excursion_m)
    hold = max(0.0, hard - float(cfg.endpoint_hold_margin_m))
    slow = max(0.0, hard - float(cfg.endpoint_slow_margin_m))
    distances: dict[str, dict[str, float]] = {}
    for side in pose_by_side:
        pose = as_strict_pose_matrix(np.asarray(pose_by_side[side], dtype=float))
        start_pose = as_strict_pose_matrix(np.asarray(start_pose_by_side[side], dtype=float))
        target_pose = as_strict_pose_matrix(np.asarray(target_pose_by_side[side], dtype=float))
        measurements = {
            "start": float(np.linalg.norm(pose[:3, 3] - start_pose[:3, 3])),
            "target": float(np.linalg.norm(pose[:3, 3] - target_pose[:3, 3])),
        }
        distances[side] = {"from_start_m": measurements["start"], "from_target_m": measurements["target"]}
        for label, distance in measurements.items():
            if distance > hard:
                level, code, threshold = ArmTransitionRiskLevel.STOP, "endpoint_excursion_hard", hard
            elif distance >= hold:
                level, code, threshold = ArmTransitionRiskLevel.HOLD, "endpoint_excursion_hold_margin", hold
            elif distance >= slow:
                level, code, threshold = ArmTransitionRiskLevel.SLOWDOWN, "endpoint_excursion_slow_margin", slow
            else:
                continue
            events.append(
                ArmTransitionRiskEvent(
                    level=level,
                    code=code,
                    message=f"{side} endpoint is near or outside excursion limit",
                    side=side,
                    value=distance,
                    threshold=threshold,
                    metadata={"reference": label},
                )
            )
    metadata["endpoint_excursion_m"] = distances


def _append_collision_events(events: list[ArmTransitionRiskEvent], model: ArmTransitionModel, qpos_by_side: Mapping[str, np.ndarray], cfg: ArmConstrainedTransitionConfig, metadata: dict[str, Any]) -> None:
    value = model.min_collision_distance(qpos_by_side)
    metadata["min_collision_distance_m"] = None if value is None else float(value)
    if value is None:
        metadata["collision_check"] = "unavailable"
        return
    distance = float(value)
    if distance < float(cfg.collision_stop_distance_m):
        level, code, threshold = ArmTransitionRiskLevel.STOP, "collision_hard", float(cfg.collision_stop_distance_m)
    elif distance < float(cfg.collision_hold_distance_m):
        level, code, threshold = ArmTransitionRiskLevel.HOLD, "collision_soft", float(cfg.collision_hold_distance_m)
    elif distance < float(cfg.collision_slow_distance_m):
        level, code, threshold = ArmTransitionRiskLevel.SLOWDOWN, "collision_near", float(cfg.collision_slow_distance_m)
    else:
        return
    events.append(ArmTransitionRiskEvent(level=level, code=code, message="collision distance is near or below margin", value=distance, threshold=threshold))


def _append_singularity_events(events: list[ArmTransitionRiskEvent], model: ArmTransitionModel, qpos_by_side: Mapping[str, np.ndarray], cfg: ArmConstrainedTransitionConfig, metadata: dict[str, Any]) -> None:
    margins = model.singularity_margin_by_side(qpos_by_side)
    if not margins:
        metadata["singularity_check"] = "unavailable"
        return
    parsed: dict[str, float] = {}
    for side, raw_margin in margins.items():
        margin = _extract_margin(raw_margin)
        if margin is None:
            continue
        parsed[str(side)] = float(margin)
        stop_margin = None if cfg.singularity_stop_margin is None else float(cfg.singularity_stop_margin)
        if stop_margin is not None and margin < stop_margin:
            level, code, threshold = ArmTransitionRiskLevel.STOP, "singularity_hard", stop_margin
        elif margin < float(cfg.singularity_hold_margin):
            level, code, threshold = ArmTransitionRiskLevel.HOLD, "singularity_soft", float(cfg.singularity_hold_margin)
        elif margin < float(cfg.singularity_slow_margin):
            level, code, threshold = ArmTransitionRiskLevel.SLOWDOWN, "singularity_near", float(cfg.singularity_slow_margin)
        else:
            continue
        events.append(ArmTransitionRiskEvent(level=level, code=code, message=f"{side} singularity margin is low", side=str(side), value=float(margin), threshold=threshold))
    metadata["singularity_margin"] = parsed


def _append_runtime_signal_events(events: list[ArmTransitionRiskEvent], signals: ArmRuntimeSignals, cfg: ArmConstrainedTransitionConfig, metadata: dict[str, Any]) -> None:
    metadata["runtime_signals"] = {
        "force_norm_n": signals.force_norm_n,
        "torque_norm_nm": signals.torque_norm_nm,
        "tracking_error_m": signals.tracking_error_m,
        "control_error": bool(signals.control_error),
        "suspected_collision": bool(signals.suspected_collision),
        **dict(signals.metadata),
    }
    if signals.control_error:
        events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.STOP, code="control_error", message="runtime control error reported"))
    if signals.suspected_collision:
        events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.STOP, code="suspected_collision", message="runtime suspected collision reported"))
    _append_upper_bound_signal_events(events, value=signals.force_norm_n, slow=cfg.force_slow_n, soft=cfg.force_soft_n, hard=cfg.force_hard_n, code_prefix="force")
    _append_upper_bound_signal_events(events, value=signals.torque_norm_nm, slow=cfg.torque_slow_nm, soft=cfg.torque_soft_nm, hard=cfg.torque_hard_nm, code_prefix="torque")
    tracking = signals.tracking_error_m
    if tracking is not None and np.isfinite(float(tracking)):
        value = float(tracking)
        if cfg.tracking_error_stop_m is not None and value >= float(cfg.tracking_error_stop_m):
            events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.STOP, code="tracking_error_hard", message="tracking error exceeds hard threshold", value=value, threshold=float(cfg.tracking_error_stop_m)))
        elif cfg.tracking_error_hold_m is not None and value >= float(cfg.tracking_error_hold_m):
            events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.HOLD, code="tracking_error_soft", message="tracking error exceeds hold threshold", value=value, threshold=float(cfg.tracking_error_hold_m)))


def _append_upper_bound_signal_events(events: list[ArmTransitionRiskEvent], *, value: float | None, slow: float | None, soft: float | None, hard: float | None, code_prefix: str) -> None:
    if value is None or not np.isfinite(float(value)):
        return
    measured = float(value)
    if hard is not None and measured >= float(hard):
        events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.STOP, code=f"{code_prefix}_hard", message=f"{code_prefix} exceeds hard threshold", value=measured, threshold=float(hard)))
    elif soft is not None and measured >= float(soft):
        events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.HOLD, code=f"{code_prefix}_soft", message=f"{code_prefix} exceeds soft threshold", value=measured, threshold=float(soft)))
    elif slow is not None and measured >= float(slow):
        events.append(ArmTransitionRiskEvent(level=ArmTransitionRiskLevel.SLOWDOWN, code=f"{code_prefix}_light", message=f"{code_prefix} exceeds slow-down threshold", value=measured, threshold=float(slow)))


def _fk_by_side(model: ArmTransitionModel, qpos_by_side: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    result = model.fk_by_side(validate_qpos_mapping(qpos_by_side))
    return {str(side): as_strict_pose_matrix(np.asarray(result[side], dtype=float)) for side in result}


def _require_same_shapes(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> None:
    for side in a:
        if a[side].shape != b[side].shape:
            raise ValueError(f"{side}.qpos shape mismatch: {a[side].shape} != {b[side].shape}.")


def _max_abs_qpos_delta(delta_by_side: Mapping[str, np.ndarray]) -> float:
    return max(float(np.max(np.abs(delta))) for delta in delta_by_side.values())


def _path_alpha(start: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], current: Mapping[str, np.ndarray]) -> float:
    total_sq = 0.0
    remaining_sq = 0.0
    for side in start:
        total = target[side] - start[side]
        remaining = target[side] - current[side]
        total_sq += float(np.dot(total, total))
        remaining_sq += float(np.dot(remaining, remaining))
    if total_sq <= 1e-24:
        return 1.0
    return float(np.clip(1.0 - np.sqrt(remaining_sq / total_sq), 0.0, 1.0))


def _extract_margin(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("margin", "singularity_margin", "min_singular_value", "min_margin"):
            if key in value:
                return _extract_margin(value[key])
        return None
    try:
        margin = float(value)
    except (TypeError, ValueError):
        return None
    return margin if np.isfinite(margin) else None


def _validate_optional_upper_thresholds(label: str, slow: float | None, soft: float | None, hard: float | None) -> None:
    previous_name: str | None = None
    previous_value: float | None = None
    for name, raw in (("slow", slow), ("soft", soft), ("hard", hard)):
        if raw is None:
            continue
        value = float(raw)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{label}_{name} threshold must be >= 0 when set.")
        if previous_value is not None and value < previous_value:
            raise ValueError(f"{label} thresholds must be nondecreasing; {previous_name}={previous_value} > {name}={value}.")
        previous_name = name
        previous_value = value


def _first_event_code(assessment: ArmTransitionRiskAssessment) -> str | None:
    if not assessment.events:
        return None
    highest = assessment.level
    for event in assessment.events:
        if event.level == highest:
            return str(event.code)
    return str(assessment.events[0].code)


def _qpos_mapping_to_lists(qpos_by_side: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {side: np.asarray(qpos, dtype=float).reshape(-1).tolist() for side, qpos in qpos_by_side.items()}


__all__ = [
    "ArmConstrainedTransitionConfig",
    "ArmConstrainedTransitionPlan",
    "ArmRuntimeSignals",
    "ArmTransitionModel",
    "ArmTransitionPlanStep",
    "ArmTransitionRiskAssessment",
    "ArmTransitionRiskEvent",
    "ArmTransitionRiskLevel",
    "evaluate_transition_state",
    "plan_constrained_qpos_transition",
]

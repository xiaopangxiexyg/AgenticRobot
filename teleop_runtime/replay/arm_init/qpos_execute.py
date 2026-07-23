from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.arm_init import (
    ArmQposTransitionConfig,
    load_qpos_by_side_file,
    plan_slow_qpos_transition,
    qpos_transition_plan_to_dict,
)
from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.rm65.constrained_transition import (
    RM65ConstrainedTransitionConfig,
    RM65ConstrainedTransitionPlan,
    RM65DualArmCollisionIKTransitionModel,
    RM65SamplingTransitionConfig,
    plan_dual_arm_constrained_qpos_transition,
    plan_dual_arm_sampling_qpos_transition,
)
from robot_motion.rm65.constants import RM65_ABSOLUTE_Q_MAX_RAD, RM65_ABSOLUTE_Q_MIN_RAD
from robot_motion.rm65.arm_init.qpos_init import DEFAULT_QPOS_TARGET_PATH
from robot_motion.rm65.arm_init.schema import load_qpos_target
from robot_motion.rm65.ik import RM65DualArmCollisionIKConfig
from robot_motion.rm65.qpos_layout import (
    normalize_sides,
    validate_qpos_by_side,
    validate_qpos_within_absolute_limits,
)

from .safety import (
    RM65EmergencyStopResult,
    RM65ForceGuardConfig,
    check_rm65_force_guard,
    emergency_stop_rm65,
)


CONFIRM_TEXT = "INIT_RM65_QPOS_EXECUTE"
DEFAULT_QPOS_INIT_COLLISION_SCENE = "dual_arm_hand_proxy_strict"
DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M = 0.07
DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M = 0.025
DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M = 0.012
DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG = 178.0
DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG = 8.0
DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG = 1.0
DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG = 120.0
DEFAULT_QPOS_INIT_BRANCH_SEARCH = True
DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES = 8
CUSTOM_QPOS_INIT_VELOCITY_PROFILE_NAME = "custom_qpos_init_max_angvel"
# Included in every returned plan so a long-running GUI process can expose
# which qpos-init safety policy was actually imported.  This is diagnostic
# metadata only; it does not alter planning or execution.
QPOS_INIT_PLANNER_CAPABILITY = "rm65_qpos_init_joint_hold_release_v2"
RM65_ARM_JOINT_DOF = 6
DOCUMENTATION_IP_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "192.0.2.0/24",
        "198.51.100.0/24",
        "203.0.113.0/24",
    )
)


@dataclass(frozen=True)
class RM65LiveQposInitConfig:
    qpos_target_path: str | Path = DEFAULT_QPOS_TARGET_PATH
    active_sides: tuple[str, ...] = ("left", "right")
    current_qpos_path: str | Path | None = None
    execute: bool = False
    confirm: str = ""
    qpos_dt_sec: float = 0.05
    joint_velocity_profile: str = "rm65_init_slow"
    max_steps_per_side: int = 10000
    collision_check: bool = True
    collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE
    collision_slow_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M
    collision_hold_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M
    collision_stop_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M
    max_joint_abs_deg: float = DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG
    joint_slow_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG
    joint_hold_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG
    single_arm_max_qpos_delta_deg: float = DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG
    branch_search: bool = DEFAULT_QPOS_INIT_BRANCH_SEARCH
    branch_search_max_branches: int = DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES
    branch_neutral_qpos_path: str | Path | None = DEFAULT_QPOS_TARGET_PATH
    trajectory_mode: int = 0
    radio: int = 0
    state_read_retries: int = 2
    force_guard: RM65ForceGuardConfig = field(default_factory=RM65ForceGuardConfig)
    tracking_check_every: int = 10
    max_tracking_error_deg: float = 30.0
    progress_every: int = 20
    output_path: str | Path | None = None
    include_steps: bool = False
    max_steps_json: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_sides", normalize_sides(tuple(self.active_sides)))
        if bool(self.execute) and str(self.confirm) != CONFIRM_TEXT:
            raise ValueError(f"real execution requires --confirm {CONFIRM_TEXT}")
        if float(self.qpos_dt_sec) <= 0.0 or not math.isfinite(float(self.qpos_dt_sec)):
            raise ValueError("qpos_dt_sec must be positive and finite.")
        if int(self.max_steps_per_side) < 1:
            raise ValueError("max_steps_per_side must be >= 1.")
        if float(self.collision_slow_distance_m) <= 0.0:
            raise ValueError("collision_slow_distance_m must be positive.")
        if float(self.collision_hold_distance_m) < 0.0:
            raise ValueError("collision_hold_distance_m must be >= 0.")
        if float(self.collision_stop_distance_m) < 0.0:
            raise ValueError("collision_stop_distance_m must be >= 0.")
        if float(self.collision_stop_distance_m) > float(self.collision_hold_distance_m):
            raise ValueError("collision_stop_distance_m must be <= collision_hold_distance_m.")
        if float(self.collision_hold_distance_m) > float(self.collision_slow_distance_m):
            raise ValueError("collision_hold_distance_m must be <= collision_slow_distance_m.")
        if float(self.max_joint_abs_deg) <= 0.0:
            raise ValueError("max_joint_abs_deg must be positive.")
        if float(self.joint_slow_margin_deg) < 0.0:
            raise ValueError("joint_slow_margin_deg must be >= 0.")
        if float(self.joint_hold_margin_deg) < 0.0:
            raise ValueError("joint_hold_margin_deg must be >= 0.")
        if float(self.joint_hold_margin_deg) > float(self.max_joint_abs_deg):
            raise ValueError("joint_hold_margin_deg must be <= max_joint_abs_deg.")
        if float(self.joint_slow_margin_deg) > float(self.max_joint_abs_deg):
            raise ValueError("joint_slow_margin_deg must be <= max_joint_abs_deg.")
        if float(self.single_arm_max_qpos_delta_deg) <= 0.0:
            raise ValueError("single_arm_max_qpos_delta_deg must be positive.")
        if int(self.branch_search_max_branches) < 0:
            raise ValueError("branch_search_max_branches must be >= 0.")
        if int(self.tracking_check_every) < 0:
            raise ValueError("tracking_check_every must be >= 0.")
        if float(self.max_tracking_error_deg) <= 0.0:
            raise ValueError("max_tracking_error_deg must be positive.")
        if int(self.progress_every) < 0:
            raise ValueError("progress_every must be >= 0.")


def execute_rm65_live_qpos_init(
    config: RM65LiveQposInitConfig,
    *,
    robots_by_side: Mapping[str, object] | None = None,
    arm_ip_by_side: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    sides = tuple(config.active_sides)
    owned_robots = False
    robots: dict[str, object] = {}
    stop_results: list[RM65EmergencyStopResult] = []
    started_at = time.time()
    try:
        if config.current_qpos_path is not None:
            current_qpos = validate_qpos_by_side(load_qpos_by_side_file(config.current_qpos_path))
        else:
            if robots_by_side is None:
                robots = connect_rm65_arms(arm_ip_by_side or {}, active_sides=sides)
                owned_robots = True
            else:
                robots = {side: robots_by_side[side] for side in sides}
            current_qpos = read_current_qpos_by_side(
                robots,
                active_sides=sides,
                retry=int(config.state_read_retries),
            )

        target_all = load_qpos_target(config.qpos_target_path).qpos_by_side()
        target = {side: target_all[side] for side in sides}
        current = {side: current_qpos[side] for side in sides}
        neutral_qpos = (
            _load_branch_neutral_qpos(config.branch_neutral_qpos_path, sides=sides)
            if bool(config.branch_search) and sides == ("left", "right")
            else None
        )
        plan = build_rm65_qpos_init_plan(
            start_qpos_by_side=current,
            target_qpos_by_side=target,
            active_sides=sides,
            qpos_dt_sec=float(config.qpos_dt_sec),
            joint_velocity_profile=str(config.joint_velocity_profile),
            max_steps_per_side=int(config.max_steps_per_side),
            collision_check=bool(config.collision_check),
            collision_scene_profile=str(config.collision_scene_profile),
            collision_slow_distance_m=float(config.collision_slow_distance_m),
            collision_hold_distance_m=float(config.collision_hold_distance_m),
            collision_stop_distance_m=float(config.collision_stop_distance_m),
            max_joint_abs_deg=float(config.max_joint_abs_deg),
            joint_slow_margin_deg=float(config.joint_slow_margin_deg),
            joint_hold_margin_deg=float(config.joint_hold_margin_deg),
            single_arm_max_qpos_delta_deg=float(config.single_arm_max_qpos_delta_deg),
            branch_search=bool(config.branch_search),
            branch_search_max_branches=int(config.branch_search_max_branches),
            branch_neutral_qpos_by_side=neutral_qpos,
        )
        _require_plan_completed(plan)
        plan_summary = plan.summary()
        if int(config.progress_every) > 0:
            print(
                "rm65-live-qpos-init plan "
                f"planner={_plan_planner_name(plan)} "
                f"steps={plan_summary['step_count']} "
                f"duration_sec={float(plan_summary['duration_sec']):.3f} "
                f"collision_scene={_plan_collision_scene(plan)}",
                flush=True,
            )

        execution_report: dict[str, Any] = {
            "executed": False,
            "sent_step_count": 0,
            "last_sent_step_index": None,
            "stop_results": [],
            "tracking_checks": [],
        }
        if config.execute:
            if not robots:
                if robots_by_side is None:
                    robots = connect_rm65_arms(arm_ip_by_side or {}, active_sides=sides)
                    owned_robots = True
                else:
                    robots = {side: robots_by_side[side] for side in sides}
            execution_report = send_rm65_qpos_plan(
                robots_by_side=robots,
                plan=plan,
                force_guard=config.force_guard,
                dt_sec=float(config.qpos_dt_sec),
                trajectory_mode=int(config.trajectory_mode),
                radio=int(config.radio),
                tracking_check_every=int(config.tracking_check_every),
                max_tracking_error_deg=float(config.max_tracking_error_deg),
                state_read_retries=int(config.state_read_retries),
                progress_every=int(config.progress_every),
                progress_label="rm65-live-qpos-init",
            )

        report = {
            "data_type": "teleop_runtime_rm65_live_qpos_init_report",
            "schema_version": 1,
            "generated_at_unix_sec": started_at,
            "finished_at_unix_sec": time.time(),
            "active_sides": list(sides),
            "qpos_target_path": str(config.qpos_target_path),
            "current_qpos_source": (
                str(config.current_qpos_path)
                if config.current_qpos_path is not None
                else "live_hardware_state"
            ),
            "execute_requested": bool(config.execute),
            "plan": rm65_qpos_init_plan_to_dict(
                plan,
                include_steps=bool(config.include_steps),
                max_steps=int(config.max_steps_json),
            ),
            "execution": execution_report,
            "force_guard": config.force_guard.to_dict(),
            "planning_config": _qpos_init_planning_config_to_dict(config),
        }
        _write_report_if_requested(report, config.output_path)
        return report
    except BaseException as exc:
        if config.execute and robots:
            stop_results = stop_rm65_arms(robots, reason="qpos_init_exception")
        if config.output_path is not None:
            _write_report_if_requested(
                {
                    "data_type": "teleop_runtime_rm65_live_qpos_init_error",
                    "schema_version": 1,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "active_sides": list(sides),
                    "qpos_target_path": str(config.qpos_target_path),
                    "planning_config": _qpos_init_planning_config_to_dict(config),
                    "stop_results": [result.to_dict() for result in stop_results],
                },
                config.output_path,
            )
        raise
    finally:
        if owned_robots:
            close_rm65_arms(robots)


def build_rm65_qpos_init_plan(
    *,
    start_qpos_by_side: Mapping[str, Any],
    target_qpos_by_side: Mapping[str, Any],
    active_sides: Sequence[str],
    qpos_dt_sec: float,
    joint_velocity_profile: str,
    max_steps_per_side: int,
    collision_check: bool = True,
    collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE,
    collision_slow_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M,
    collision_hold_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M,
    collision_stop_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M,
    max_joint_abs_deg: float = DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG,
    joint_slow_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG,
    joint_hold_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG,
    single_arm_max_qpos_delta_deg: float = DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG,
    branch_search: bool = DEFAULT_QPOS_INIT_BRANCH_SEARCH,
    branch_search_max_branches: int = DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES,
    branch_neutral_qpos_by_side: Mapping[str, Any] | None = None,
    collision_model: Any | None = None,
    qpos_init_max_angvel_rad_s: Any = None,
    sampling_search_config: RM65SamplingTransitionConfig | None = None,
) -> Any:
    sides = normalize_sides(tuple(active_sides))
    velocity_profile = qpos_init_velocity_profile_from_max_angvel(qpos_init_max_angvel_rad_s)
    qpos_config = ArmQposTransitionConfig(
        dt_sec=float(qpos_dt_sec),
        joint_velocity_profile_name=str(joint_velocity_profile),
        joint_velocity_profile=velocity_profile,
        max_steps_per_side=int(max_steps_per_side),
    )
    profile_name = velocity_profile.name if velocity_profile is not None else str(joint_velocity_profile)
    validated_start = validate_qpos_within_absolute_limits(
        start_qpos_by_side,
        label="start_qpos_by_side",
    )
    validated_target = validate_qpos_within_absolute_limits(
        target_qpos_by_side,
        label="target_qpos_by_side",
    )
    start = {side: validated_start[side] for side in sides}
    target = {side: validated_target[side] for side in sides}
    if _should_use_dual_arm_collision_qpos_init(collision_check=collision_check, sides=sides):
        model = collision_model
        if model is None:
            model = RM65DualArmCollisionIKTransitionModel.from_config(
                RM65DualArmCollisionIKConfig(scene_profile=str(collision_scene_profile)),
                reference_qpos_by_side=start,
            )
        transition_config = _build_constrained_qpos_init_config(
            qpos_config,
            collision_slow_distance_m=float(collision_slow_distance_m),
            collision_hold_distance_m=float(collision_hold_distance_m),
            collision_stop_distance_m=float(collision_stop_distance_m),
            max_joint_abs_deg=float(max_joint_abs_deg),
            joint_slow_margin_deg=float(joint_slow_margin_deg),
            joint_hold_margin_deg=float(joint_hold_margin_deg),
        )
        plan = plan_dual_arm_constrained_qpos_transition(
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            model=model,
            config=transition_config,
        )
        plan.metadata.update(
            {
                "planner": "plan_dual_arm_constrained_qpos_transition",
                "planner_capability": QPOS_INIT_PLANNER_CAPABILITY,
                "qpos_init_collision_check": "enabled",
                "collision_scene_profile": str(collision_scene_profile),
                "joint_velocity_profile": profile_name,
            }
        )
        if velocity_profile is not None:
            plan.metadata["qpos_init_max_angvel_rad_s"] = list(
                normalize_qpos_init_max_angvel(qpos_init_max_angvel_rad_s) or ()
            )
        if _should_try_dual_arm_branch_search(
            plan,
            branch_search=bool(branch_search),
            branch_search_max_branches=int(branch_search_max_branches),
        ):
            branch_plan = _try_dual_arm_joint_order_branch_plans(
                start_qpos_by_side=start,
                target_qpos_by_side=target,
                config=transition_config,
                collision_scene_profile=str(collision_scene_profile),
                collision_model=model,
                neutral_qpos_by_side=(
                    None
                    if branch_neutral_qpos_by_side is None
                    else {side: validate_qpos_by_side(branch_neutral_qpos_by_side)[side] for side in sides}
                ),
                direct_plan=plan,
                branch_search_max_branches=int(branch_search_max_branches),
            )
            if branch_plan.completed:
                branch_plan.metadata["planner_capability"] = QPOS_INIT_PLANNER_CAPABILITY
                return branch_plan
            branch_attempts = branch_plan.metadata.get("branch_search_attempts", [])
            _set_collision_model_reference_qpos(model, start)
            sampling_plan = plan_dual_arm_sampling_qpos_transition(
                start_qpos_by_side=start,
                target_qpos_by_side=target,
                model=model,
                config=transition_config,
                search_config=sampling_search_config,
            )
            sampling_plan.metadata.update(
                {
                    "planner_capability": QPOS_INIT_PLANNER_CAPABILITY,
                    "qpos_init_collision_check": "enabled",
                    "collision_scene_profile": str(collision_scene_profile),
                    "joint_velocity_profile": profile_name,
                    "branch_search_status": "failed",
                    "branch_search_attempts": branch_attempts,
                    "direct_plan_summary": plan.summary(),
                }
            )
            if velocity_profile is not None:
                sampling_plan.metadata["qpos_init_max_angvel_rad_s"] = list(
                    normalize_qpos_init_max_angvel(qpos_init_max_angvel_rad_s) or ()
                )
            return sampling_plan
        return plan

    if len(sides) == 1:
        _require_single_arm_qpos_delta_within_limit(
            start,
            target,
            side=sides[0],
            max_delta_deg=float(single_arm_max_qpos_delta_deg),
        )
    plan = plan_slow_qpos_transition(
        start_qpos_by_side=start,
        target_qpos_by_side=target,
        config=qpos_config,
        side_order=sides,
    )
    plan.metadata.update(
        {
            "planner_capability": QPOS_INIT_PLANNER_CAPABILITY,
            "qpos_init_collision_check": "disabled"
            if not bool(collision_check)
            else "skipped_requires_both_sides",
            "collision_scene_profile": str(collision_scene_profile),
        }
    )
    if velocity_profile is not None:
        plan.metadata["qpos_init_max_angvel_rad_s"] = list(
            normalize_qpos_init_max_angvel(qpos_init_max_angvel_rad_s) or ()
        )
    return plan


def normalize_qpos_init_max_angvel(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"default", "profile", "none", "off"}:
            return None
        raw_parts: Sequence[Any] = [part for part in text.replace(",", " ").split() if part]
    else:
        raw_parts = list(np.asarray(value, dtype=float).reshape(-1))
        if not raw_parts:
            return None
    try:
        values = [float(part) for part in raw_parts]
    except (TypeError, ValueError) as exc:
        raise ValueError("qpos init max angvel must contain 1 or 6 finite positive values in rad/s.") from exc
    if len(values) == 1:
        values = values * RM65_ARM_JOINT_DOF
    if len(values) != RM65_ARM_JOINT_DOF or not np.all(np.isfinite(values)) or any(value <= 0.0 for value in values):
        raise ValueError("qpos init max angvel must contain 1 or 6 finite positive values in rad/s.")
    return tuple(float(value) for value in values)


def qpos_init_velocity_profile_from_max_angvel(value: Any) -> JointVelocityLimitProfile | None:
    max_angvel = normalize_qpos_init_max_angvel(value)
    if max_angvel is None:
        return None
    return JointVelocityLimitProfile(
        name=CUSTOM_QPOS_INIT_VELOCITY_PROFILE_NAME,
        max_velocity_deg_s=tuple(float(np.rad2deg(value)) for value in max_angvel),
        max_dt_sec=1.0,
        metadata={"max_angvel_rad_s": list(max_angvel)},
    )


def build_rm65_qpos_init_collision_model(
    *,
    reference_qpos_by_side: Mapping[str, Any],
    collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE,
) -> RM65DualArmCollisionIKTransitionModel:
    return RM65DualArmCollisionIKTransitionModel.from_config(
        RM65DualArmCollisionIKConfig(scene_profile=str(collision_scene_profile)),
        reference_qpos_by_side=reference_qpos_by_side,
    )


def _load_branch_neutral_qpos(
    path: str | Path | None,
    *,
    sides: Sequence[str],
) -> dict[str, np.ndarray] | None:
    if path is None or not str(path).strip():
        return None
    qpos = validate_qpos_by_side(load_qpos_target(path).qpos_by_side())
    missing = [side for side in sides if side not in qpos]
    if missing:
        raise ValueError(f"branch neutral qpos {path} is missing sides {missing}.")
    return {side: qpos[side].copy() for side in sides}


def rm65_qpos_init_plan_to_dict(
    plan: Any,
    *,
    include_steps: bool = True,
    max_steps: int = 0,
) -> dict[str, Any]:
    if hasattr(plan, "joint_paths_by_side"):
        return qpos_transition_plan_to_dict(plan, include_steps=include_steps, max_steps=max_steps)

    if not hasattr(plan, "to_dict"):
        raise TypeError(f"unsupported qpos init plan type: {type(plan).__name__}")
    payload = dict(plan.to_dict(include_steps=include_steps))
    payload["data_type"] = "robot_motion_rm65_constrained_qpos_init_plan"
    payload["schema_version"] = 1
    payload["final_qpos_by_side"] = _qpos_mapping_to_lists(plan.final_qpos_by_side)
    if include_steps:
        steps = list(payload.get("steps", []))
        limit = len(steps) if int(max_steps) <= 0 else min(int(max_steps), len(steps))
        payload["steps"] = steps[:limit]
        payload["steps_truncated"] = limit < len(steps)
    return payload


def _should_use_dual_arm_collision_qpos_init(*, collision_check: bool, sides: Sequence[str]) -> bool:
    return bool(collision_check) and tuple(sides) == ("left", "right")


def _build_constrained_qpos_init_config(
    qpos_config: ArmQposTransitionConfig,
    *,
    collision_slow_distance_m: float,
    collision_hold_distance_m: float,
    collision_stop_distance_m: float,
    max_joint_abs_deg: float,
    joint_slow_margin_deg: float,
    joint_hold_margin_deg: float,
) -> RM65ConstrainedTransitionConfig:
    profile = qpos_config.resolved_joint_velocity_profile()
    max_step_deg = float(np.min(profile.max_step_deg(float(qpos_config.dt_sec))))
    return RM65ConstrainedTransitionConfig(
        dt_sec=float(qpos_config.dt_sec),
        max_joint_step_deg=max_step_deg,
        max_steps=int(qpos_config.max_steps_per_side),
        collision_slow_distance_m=float(collision_slow_distance_m),
        collision_hold_distance_m=float(collision_hold_distance_m),
        collision_stop_distance_m=float(collision_stop_distance_m),
        max_joint_abs_deg=float(max_joint_abs_deg),
        joint_lower_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MIN_RAD),
        joint_upper_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MAX_RAD),
        joint_slow_margin_deg=float(joint_slow_margin_deg),
        joint_hold_margin_deg=float(joint_hold_margin_deg),
        # A measured start can legitimately sit inside the soft hold margin.
        # Permit only held joints that stay exactly static or strictly increase
        # distance from that limit toward a target outside the margin. Hard
        # limits and every non-joint HOLD/STOP event still fail closed.
        allow_joint_limit_escape=True,
        require_kinematic_model=True,
        require_collision_check=True,
    )


def _should_try_dual_arm_branch_search(
    plan: Any,
    *,
    branch_search: bool,
    branch_search_max_branches: int,
) -> bool:
    if not bool(branch_search) or int(branch_search_max_branches) <= 0:
        return False
    if bool(getattr(plan, "completed", False)):
        return False
    status = str(getattr(plan, "status", ""))
    return status in {"blocked_candidate", "max_steps_exceeded"}


def _try_dual_arm_joint_order_branch_plans(
    *,
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    config: RM65ConstrainedTransitionConfig,
    collision_scene_profile: str,
    collision_model: Any | None,
    neutral_qpos_by_side: Mapping[str, np.ndarray] | None,
    direct_plan: RM65ConstrainedTransitionPlan,
    branch_search_max_branches: int,
) -> RM65ConstrainedTransitionPlan:
    attempts: list[dict[str, Any]] = []
    if neutral_qpos_by_side is not None:
        neutral_plan = _try_dual_arm_waypoint_branch_plan(
            start_qpos_by_side=start_qpos_by_side,
            target_qpos_by_side=target_qpos_by_side,
            waypoints=(neutral_qpos_by_side,),
            branch_name="neutral_qpos",
            config=config,
            collision_scene_profile=str(collision_scene_profile),
            collision_model=collision_model,
            direct_plan=direct_plan,
            attempts=attempts,
        )
        if neutral_plan.completed:
            return neutral_plan

    for branch_index, branch in enumerate(
        _dual_arm_joint_order_branches(max_branches=int(branch_search_max_branches)),
        start=1,
    ):
        candidate = _try_dual_arm_waypoint_branch_plan(
            start_qpos_by_side=start_qpos_by_side,
            target_qpos_by_side=target_qpos_by_side,
            waypoints=_dual_arm_branch_waypoints(start_qpos_by_side, target_qpos_by_side, branch=branch),
            branch_name=str(branch["name"]),
            config=config,
            collision_scene_profile=str(collision_scene_profile),
            collision_model=collision_model,
            direct_plan=direct_plan,
            attempts=attempts,
        )
        if candidate.completed:
            return candidate

    direct_plan.metadata["branch_search_attempts"] = attempts
    direct_plan.metadata["branch_search_status"] = "failed"
    return direct_plan


def _try_dual_arm_waypoint_branch_plan(
    *,
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    waypoints: Sequence[Mapping[str, np.ndarray]],
    branch_name: str,
    config: RM65ConstrainedTransitionConfig,
    collision_scene_profile: str,
    collision_model: Any | None,
    direct_plan: RM65ConstrainedTransitionPlan,
    attempts: list[dict[str, Any]],
) -> RM65ConstrainedTransitionPlan:
    branch_model = collision_model
    if branch_model is None:
        branch_model = build_rm65_qpos_init_collision_model(
            reference_qpos_by_side=start_qpos_by_side,
            collision_scene_profile=str(collision_scene_profile),
        )
    else:
        _set_collision_model_reference_qpos(branch_model, start_qpos_by_side)
    filtered_waypoints = [
        _copy_qpos_mapping(waypoint)
        for waypoint in waypoints
        if not _qpos_mappings_close(start_qpos_by_side, waypoint)
        and not _qpos_mappings_close(target_qpos_by_side, waypoint)
    ]
    filtered_waypoints.append(_copy_qpos_mapping(target_qpos_by_side))
    segment_plans: list[RM65ConstrainedTransitionPlan] = []
    current = _copy_qpos_mapping(start_qpos_by_side)
    for segment_index, waypoint in enumerate(filtered_waypoints, start=1):
        segment_plan = plan_dual_arm_constrained_qpos_transition(
            start_qpos_by_side=current,
            target_qpos_by_side=waypoint,
            model=branch_model,
            config=config,
        )
        segment_plan.metadata.update(
            {
                "planner": "plan_dual_arm_constrained_qpos_transition",
                "qpos_init_branch": str(branch_name),
                "qpos_init_branch_segment": int(segment_index),
            }
        )
        segment_plans.append(segment_plan)
        if not segment_plan.completed:
            attempts.append(
                {
                    "branch_index": len(attempts) + 1,
                    "branch_name": str(branch_name),
                    "failed_segment": int(segment_index),
                    "summary": segment_plan.summary(),
                }
            )
            return direct_plan
        current = segment_plan.final_qpos_by_side
    return _combine_dual_arm_branch_plan(
        start_qpos_by_side=start_qpos_by_side,
        target_qpos_by_side=target_qpos_by_side,
        segment_plans=segment_plans,
        branch_name=str(branch_name),
        direct_plan=direct_plan,
        attempts=attempts,
        dt_sec=float(config.dt_sec),
    )


def _dual_arm_joint_order_branches(*, max_branches: int) -> tuple[dict[str, Any], ...]:
    base_to_tip = tuple(range(6))
    tip_to_base = tuple(reversed(base_to_tip))
    candidates = (
        {"name": "both_arms_j1_to_j6", "mode": "same_joint_both_arms", "joint_order": base_to_tip},
        {"name": "both_arms_j6_to_j1", "mode": "same_joint_both_arms", "joint_order": tip_to_base},
        {"name": "left_then_right_j1_to_j6", "mode": "single_arm_then_other", "side_order": ("left", "right"), "joint_order": base_to_tip},
        {"name": "right_then_left_j1_to_j6", "mode": "single_arm_then_other", "side_order": ("right", "left"), "joint_order": base_to_tip},
        {"name": "left_then_right_j6_to_j1", "mode": "single_arm_then_other", "side_order": ("left", "right"), "joint_order": tip_to_base},
        {"name": "right_then_left_j6_to_j1", "mode": "single_arm_then_other", "side_order": ("right", "left"), "joint_order": tip_to_base},
        {"name": "alternating_left_right_j1_to_j6", "mode": "alternating_sides", "side_order": ("left", "right"), "joint_order": base_to_tip},
        {"name": "alternating_right_left_j1_to_j6", "mode": "alternating_sides", "side_order": ("right", "left"), "joint_order": base_to_tip},
    )
    return tuple(candidates[: max(0, int(max_branches))])


def _dual_arm_branch_waypoints(
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    *,
    branch: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], ...]:
    current = _copy_qpos_mapping(start_qpos_by_side)
    target = _copy_qpos_mapping(target_qpos_by_side)
    waypoints: list[dict[str, np.ndarray]] = []
    mode = str(branch.get("mode", ""))
    joint_order = tuple(int(joint) for joint in branch.get("joint_order", tuple(range(6))))
    if mode == "same_joint_both_arms":
        for joint in joint_order:
            for side in ("left", "right"):
                current[side][joint] = target[side][joint]
            _append_waypoint_if_changed(waypoints, current)
    elif mode == "single_arm_then_other":
        for side in tuple(branch.get("side_order", ("left", "right"))):
            for joint in joint_order:
                current[side][joint] = target[side][joint]
                _append_waypoint_if_changed(waypoints, current)
    elif mode == "alternating_sides":
        for joint in joint_order:
            for side in tuple(branch.get("side_order", ("left", "right"))):
                current[side][joint] = target[side][joint]
                _append_waypoint_if_changed(waypoints, current)
    else:
        raise ValueError(f"unknown qpos init branch mode {mode!r}")
    if not waypoints or not _qpos_mappings_close(waypoints[-1], target):
        waypoints.append(_copy_qpos_mapping(target))
    return tuple(waypoints)


def _append_waypoint_if_changed(waypoints: list[dict[str, np.ndarray]], qpos_by_side: Mapping[str, np.ndarray]) -> None:
    candidate = _copy_qpos_mapping(qpos_by_side)
    if waypoints and _qpos_mappings_close(waypoints[-1], candidate):
        return
    waypoints.append(candidate)


def _combine_dual_arm_branch_plan(
    *,
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    segment_plans: Sequence[RM65ConstrainedTransitionPlan],
    branch_name: str,
    direct_plan: RM65ConstrainedTransitionPlan,
    attempts: Sequence[Mapping[str, Any]],
    dt_sec: float,
) -> RM65ConstrainedTransitionPlan:
    steps = []
    for segment_index, segment_plan in enumerate(segment_plans, start=1):
        for step in segment_plan.steps:
            step_type = type(step)
            steps.append(
                step_type(
                    index=len(steps) + 1,
                    time_sec=(len(steps) + 1) * float(dt_sec),
                    qpos_by_side=_copy_qpos_mapping(step.qpos_by_side),
                    assessment=step.assessment,
                    alpha=_path_alpha(start_qpos_by_side, target_qpos_by_side, step.qpos_by_side),
                    metadata={
                        **dict(getattr(step, "metadata", {}) or {}),
                        "branch_name": str(branch_name),
                        "branch_segment": int(segment_index),
                        "segment_step_index": int(step.index),
                    },
                )
            )
    return RM65ConstrainedTransitionPlan(
        status="completed",
        start_qpos_by_side=_copy_qpos_mapping(start_qpos_by_side),
        target_qpos_by_side=_copy_qpos_mapping(target_qpos_by_side),
        steps=tuple(steps),
        initial_assessment=segment_plans[0].initial_assessment,
        target_assessment=segment_plans[-1].target_assessment,
        metadata={
            "planner": "plan_dual_arm_constrained_qpos_transition_branch_search",
            "qpos_init_collision_check": "enabled",
            "collision_scene_profile": direct_plan.metadata.get("collision_scene_profile", ""),
            "joint_velocity_profile": direct_plan.metadata.get("joint_velocity_profile", ""),
            "branch_search_status": "completed",
            "branch_name": str(branch_name),
            "branch_segment_count": len(segment_plans),
            "branch_search_attempts": list(attempts),
            "direct_plan_summary": direct_plan.summary(),
            "dt_sec": float(dt_sec),
            "max_joint_step_deg": float(direct_plan.metadata.get("max_joint_step_deg", 0.0) or 0.0),
            "constraint_space": "qpos",
        },
    )


def _copy_qpos_mapping(qpos_by_side: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(side): np.asarray(qpos, dtype=float).reshape(6).copy() for side, qpos in qpos_by_side.items()}


def _set_collision_model_reference_qpos(model: Any, qpos_by_side: Mapping[str, np.ndarray]) -> None:
    method = getattr(model, "fk_by_side", None)
    if method is None:
        return
    try:
        method(_copy_qpos_mapping(qpos_by_side))
    except Exception:
        return


def _qpos_mappings_close(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> bool:
    return all(np.allclose(np.asarray(a[side], dtype=float), np.asarray(b[side], dtype=float)) for side in a)


def _path_alpha(
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    current_qpos_by_side: Mapping[str, np.ndarray],
) -> float:
    total_sq = 0.0
    remaining_sq = 0.0
    for side in start_qpos_by_side:
        total = np.asarray(target_qpos_by_side[side], dtype=float) - np.asarray(start_qpos_by_side[side], dtype=float)
        remaining = np.asarray(target_qpos_by_side[side], dtype=float) - np.asarray(current_qpos_by_side[side], dtype=float)
        total_sq += float(np.dot(total, total))
        remaining_sq += float(np.dot(remaining, remaining))
    if total_sq <= 1e-24:
        return 1.0
    return float(np.clip(1.0 - np.sqrt(remaining_sq / total_sq), 0.0, 1.0))


def _require_single_arm_qpos_delta_within_limit(
    start_qpos_by_side: Mapping[str, np.ndarray],
    target_qpos_by_side: Mapping[str, np.ndarray],
    *,
    side: str,
    max_delta_deg: float,
) -> None:
    start = np.asarray(start_qpos_by_side[side], dtype=float).reshape(6)
    target = np.asarray(target_qpos_by_side[side], dtype=float).reshape(6)
    delta_deg = np.rad2deg(target - start)
    abs_delta = np.abs(delta_deg)
    joint_index = int(np.argmax(abs_delta))
    max_observed = float(abs_delta[joint_index])
    if max_observed <= float(max_delta_deg):
        return
    raise ValueError(
        f"{side} single-arm qpos init rejected: joint {joint_index + 1} delta "
        f"{max_observed:.3f}deg exceeds {float(max_delta_deg):.3f}deg. "
        "Single-arm init only supports slow qpos interpolation for moderate qpos changes."
    )


def _require_plan_completed(plan: Any) -> None:
    if bool(getattr(plan, "completed", False)):
        return
    raise RuntimeError(_format_incomplete_qpos_init_plan_error(plan))


def _format_incomplete_qpos_init_plan_error(plan: Any) -> str:
    summary = plan.summary() if hasattr(plan, "summary") else {"type": type(plan).__name__}
    status = str(summary.get("status") or getattr(plan, "status", ""))
    reason = str(summary.get("blocked_reason") or getattr(plan, "blocked_reason", "") or "unknown")
    if status == "blocked_start":
        prefix = (
            "qpos init refused before execution: start qpos is already unsafe or colliding. "
            "No trajectory was sent; move the arm(s) back to a known safe pose before retrying."
        )
    elif status == "blocked_target":
        prefix = (
            "qpos init refused before execution: target qpos is unsafe or colliding. "
            "Choose another target or recover through a safe intermediate pose first."
        )
    elif status in {"blocked_current", "hold_current"}:
        prefix = (
            "qpos init stopped during planning: current qpos is in a hold/stop risk state. "
            "Keep the arm(s) still and recover to a known safe pose before retrying."
        )
    elif status == "blocked_candidate":
        prefix = (
            "qpos init refused before execution: no collision-safe interpolation step was found. "
            "Use a safer intermediate qpos or a more conservative recovery path."
        )
    elif status == "search_exhausted":
        prefix = (
            "qpos init bounded heuristic and sampling searches were exhausted before execution. "
            "This is not proof that no collision-safe path exists; inspect the rejected-edge diagnostics "
            "or provide a verified intermediate qpos."
        )
    else:
        prefix = "qpos init plan is not executable."
    events = _plan_blocking_events(plan, status=status)
    event_text = "; ".join(_format_risk_event(event) for event in events[:4])
    event_suffix = f" events=[{event_text}]" if event_text else ""
    return (
        f"{prefix} status={status or 'unknown'} blocked_reason={reason}{event_suffix} "
        f"summary={json.dumps(_json_safe(summary), sort_keys=True)}"
    )


def _plan_blocking_events(plan: Any, *, status: str) -> list[Mapping[str, Any]]:
    assessments: list[Any] = []
    metadata = getattr(plan, "metadata", {}) or {}
    diagnostic_assessments: list[Mapping[str, Any]] = []
    for rejection in metadata.get("blocked_candidate_rejections", ()):
        if isinstance(rejection, Mapping) and isinstance(rejection.get("assessment"), Mapping):
            diagnostic_assessments.append(rejection["assessment"])
    last_edge = metadata.get("last_rejected_edge")
    if isinstance(last_edge, Mapping) and isinstance(last_edge.get("assessment"), Mapping):
        diagnostic_assessments.append(last_edge["assessment"])
    rejected_sample = metadata.get("rejected_execution_sample")
    if isinstance(rejected_sample, Mapping) and isinstance(rejected_sample.get("assessment"), Mapping):
        diagnostic_assessments.append(rejected_sample["assessment"])
    if status == "blocked_start":
        assessments.append(getattr(plan, "initial_assessment", None))
    elif status == "blocked_target":
        assessments.append(getattr(plan, "target_assessment", None))
    elif status in {"blocked_current", "hold_current", "blocked_candidate"} and getattr(plan, "steps", ()):
        assessments.append(getattr(plan.steps[-1], "assessment", None))
    assessments.extend(
        assessment
        for assessment in (
            getattr(plan, "initial_assessment", None),
            getattr(plan, "target_assessment", None),
        )
        if assessment is not None and assessment not in assessments
    )
    events: list[Mapping[str, Any]] = []
    for assessment in reversed(diagnostic_assessments):
        for event in assessment.get("events", []):
            if isinstance(event, Mapping):
                events.append(event)
    for assessment in assessments:
        payload = assessment.to_dict() if hasattr(assessment, "to_dict") else {}
        for event in payload.get("events", []):
            if isinstance(event, Mapping):
                events.append(event)
    return events


def _format_risk_event(event: Mapping[str, Any]) -> str:
    parts = [
        str(event.get("code", "unknown")),
        f"level={event.get('level_label', event.get('level', 'unknown'))}",
    ]
    if event.get("side") is not None:
        parts.append(f"side={event['side']}")
    if event.get("value") is not None:
        parts.append(f"value={float(event['value']):.4g}")
    if event.get("threshold") is not None:
        parts.append(f"threshold={float(event['threshold']):.4g}")
    message = str(event.get("message") or "").strip()
    if message:
        parts.append(message)
    return " ".join(parts)


def _plan_planner_name(plan: Any) -> str:
    metadata = getattr(plan, "metadata", {}) or {}
    return str(metadata.get("planner") or type(plan).__name__)


def _plan_collision_scene(plan: Any) -> str:
    metadata = getattr(plan, "metadata", {}) or {}
    return str(metadata.get("collision_scene_profile") or "")


def _qpos_init_planning_config_to_dict(config: RM65LiveQposInitConfig) -> dict[str, Any]:
    return {
        "qpos_dt_sec": float(config.qpos_dt_sec),
        "joint_velocity_profile": str(config.joint_velocity_profile),
        "max_steps_per_side": int(config.max_steps_per_side),
        "collision_check": bool(config.collision_check),
        "collision_scene_profile": str(config.collision_scene_profile),
        "collision_slow_distance_m": float(config.collision_slow_distance_m),
        "collision_hold_distance_m": float(config.collision_hold_distance_m),
        "collision_stop_distance_m": float(config.collision_stop_distance_m),
        "max_joint_abs_deg": float(config.max_joint_abs_deg),
        "joint_slow_margin_deg": float(config.joint_slow_margin_deg),
        "joint_hold_margin_deg": float(config.joint_hold_margin_deg),
        "single_arm_max_qpos_delta_deg": float(config.single_arm_max_qpos_delta_deg),
        "branch_search": bool(config.branch_search),
        "branch_search_max_branches": int(config.branch_search_max_branches),
        "branch_neutral_qpos_path": None if config.branch_neutral_qpos_path is None else str(config.branch_neutral_qpos_path),
        "progress_every": int(config.progress_every),
    }


def connect_rm65_arms(
    arm_ip_by_side: Mapping[str, str],
    *,
    active_sides: Sequence[str],
) -> dict[str, object]:
    Arm, RM65 = _load_rm65_sdk()
    resolved_ips = resolve_rm65_arm_ips(arm_ip_by_side, active_sides=active_sides)
    robots: dict[str, object] = {}
    try:
        for side in active_sides:
            robot = Arm(RM65, resolved_ips[side])
            if int(robot.Arm_Socket_State()) != 0:
                raise RuntimeError(f"{side}: failed to connect ip={resolved_ips[side]}")
            robots[side] = robot
            print(f"{side}: connected ip={resolved_ips[side]} api={robot.API_Version()}")
        return robots
    except BaseException:
        close_rm65_arms(robots)
        raise


def resolve_rm65_arm_ips(
    overrides: Mapping[str, str] | None = None,
    *,
    active_sides: Sequence[str],
    assignment_path: str | Path = "latest",
) -> dict[str, str]:
    overrides = {
        str(side): _normalize_ip_text(ip)
        for side, ip in dict(overrides or {}).items()
        if _normalize_ip_text(ip)
    }
    sides = normalize_sides(tuple(active_sides))
    ips: dict[str, str] = {}
    latest_ips: dict[str, str] = {}
    latest_path = ""
    try:
        from teleop_runtime.configs.rm65_arm_ips import arm_ips_for_sides, load_assignment

        assignment = load_assignment(assignment_path)
        latest_path = str(assignment.get("_config_path") or assignment_path)
        latest_ips.update(arm_ips_for_sides(assignment, sides=sides))
    except FileNotFoundError:
        latest_path = str(assignment_path)

    for side in sides:
        if side in overrides:
            ips[side] = _require_valid_rm65_ip(overrides[side], side=side, source="explicit --*-ip/env")
            continue
        latest_ip = latest_ips.get(side, "")
        if latest_ip:
            valid_latest = _valid_rm65_ip_or_none(latest_ip, side=side, source=f"latest config {latest_path}")
            if valid_latest:
                ips[side] = valid_latest
                continue
    missing = [side for side in sides if not str(ips.get(side, "")).strip()]
    if missing:
        raise ValueError(
            f"missing valid RM65 arm IPs for sides {missing}. "
            "Pass --left-ip/--right-ip, export LEFT_ARM_IP/RIGHT_ARM_IP, or fix "
            f"the RM65 latest IP config ({latest_path or assignment_path}). "
            "No adapter default IP fallback is used for real-motion commands."
        )
    return {side: str(ips[side]).strip() for side in sides}


def read_current_qpos_by_side(
    robots_by_side: Mapping[str, object],
    *,
    active_sides: Sequence[str],
    retry: int = 2,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in active_sides:
        robot = robots_by_side[side]
        code, joint_deg, _pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=int(retry))
        _require_zero(code, f"{side} Get_Current_Arm_State")
        if int(arm_err) != 0 or int(sys_err) != 0:
            raise RuntimeError(f"{side}: arm_err={arm_err}, sys_err={sys_err}")
        out[side] = np.deg2rad(_finite_vector(joint_deg, length=6, label=f"{side}.joint_deg"))
    return validate_qpos_by_side(out)


def send_rm65_qpos_plan(
    *,
    robots_by_side: Mapping[str, object],
    plan: Any,
    force_guard: RM65ForceGuardConfig,
    dt_sec: float,
    trajectory_mode: int = 0,
    radio: int = 0,
    tracking_check_every: int = 10,
    max_tracking_error_deg: float = 30.0,
    state_read_retries: int = 2,
    progress_every: int = 20,
    progress_label: str = "rm65-qpos-init",
) -> dict[str, Any]:
    started = time.monotonic()
    tracking_checks: list[dict[str, Any]] = []
    last_index: int | None = None
    sides = _plan_sides(plan)
    total_steps = len(plan.steps)
    if int(progress_every) > 0:
        print(
            f"{progress_label}: sending {total_steps} steps "
            f"dt={float(dt_sec):.3f}s sides={','.join(sides)}",
            flush=True,
        )
    for step in plan.steps:
        step_start = time.monotonic()
        for side in sides:
            check_rm65_force_guard(robots_by_side[side], side, force_guard)
        for side in sides:
            joint_deg = np.rad2deg(step.qpos_by_side[side]).astype(float).tolist()
            code = robots_by_side[side].Movej_CANFD(
                joint_deg,
                False,
                trajectory_mode=int(trajectory_mode),
                radio=int(radio),
            )
            _require_zero(code, f"{side} Movej_CANFD")
        last_index = int(step.index)
        if int(tracking_check_every) > 0 and (int(step.index) % int(tracking_check_every) == 0):
            tracking_checks.append(
                _check_tracking(
                    robots_by_side,
                    command_qpos_by_side=step.qpos_by_side,
                    active_sides=sides,
                    max_tracking_error_deg=float(max_tracking_error_deg),
                    retry=int(state_read_retries),
                    step_index=int(step.index),
                )
            )
        if _should_print_progress(int(step.index), total_steps, int(progress_every)):
            print(_format_step_progress(progress_label, step, total_steps, started), flush=True)
        elapsed = time.monotonic() - step_start
        sleep_sec = float(dt_sec) - elapsed
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    for side in sides:
        check_rm65_force_guard(robots_by_side[side], side, force_guard)
    if plan.steps:
        tracking_checks.append(
            _check_tracking(
                robots_by_side,
                command_qpos_by_side=plan.final_qpos_by_side,
                active_sides=sides,
                max_tracking_error_deg=float(max_tracking_error_deg),
                retry=int(state_read_retries),
                step_index=int(plan.steps[-1].index),
            )
        )
    return {
        "executed": True,
        "sent_step_count": len(plan.steps),
        "last_sent_step_index": last_index,
        "elapsed_sec": time.monotonic() - started,
        "tracking_checks": tracking_checks,
        "stop_results": [],
    }


def _plan_sides(plan: Any) -> tuple[str, ...]:
    sides = tuple(str(side) for side in getattr(plan, "sides", ()) if str(side))
    if sides:
        return sides
    start = getattr(plan, "start_qpos_by_side", None)
    if isinstance(start, Mapping) and start:
        return tuple(str(side) for side in start)
    if getattr(plan, "steps", ()):
        first_step = plan.steps[0]
        qpos = getattr(first_step, "qpos_by_side", {})
        if isinstance(qpos, Mapping) and qpos:
            return tuple(str(side) for side in qpos)
    raise ValueError(f"cannot infer plan sides for {type(plan).__name__}")


def _should_print_progress(step_index: int, total_steps: int, progress_every: int) -> bool:
    if int(progress_every) <= 0:
        return False
    if int(step_index) <= 1 or int(step_index) >= int(total_steps):
        return True
    return int(step_index) % int(progress_every) == 0


def _format_step_progress(label: str, step: Any, total_steps: int, started: float) -> str:
    index = int(getattr(step, "index", 0))
    pct = 100.0 if int(total_steps) <= 0 else 100.0 * float(index) / float(total_steps)
    parts = [
        f"{label}: step {index}/{int(total_steps)}",
        f"{pct:.1f}%",
        f"elapsed={time.monotonic() - float(started):.1f}s",
    ]
    assessment = getattr(step, "assessment", None)
    if assessment is not None:
        risk = assessment.to_dict() if hasattr(assessment, "to_dict") else {}
        if risk:
            parts.append(f"risk={risk.get('level_label', risk.get('level'))}")
            parts.append(f"speed_scale={float(risk.get('speed_scale', 1.0)):.2f}")
        metadata = getattr(assessment, "metadata", {}) or {}
        distance = metadata.get("min_collision_distance_m")
        if distance is not None:
            parts.append(f"min_collision_m={float(distance):.4f}")
    return " ".join(parts)


def stop_rm65_arms(
    robots_by_side: Mapping[str, object],
    *,
    reason: str,
) -> list[RM65EmergencyStopResult]:
    results: list[RM65EmergencyStopResult] = []
    for side, robot in robots_by_side.items():
        try:
            results.append(emergency_stop_rm65(robot, side, reason=reason))
        except Exception as exc:  # noqa: BLE001 - preserve best-effort stops for other arms.
            print(f"WARNING: {side}: failed to stop after {reason}: {exc}")
    return results


def close_rm65_arms(robots_by_side: Mapping[str, object]) -> None:
    for side, robot in robots_by_side.items():
        close = getattr(robot, "Arm_Socket_Close", None) or getattr(robot, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - close is best-effort.
            print(f"WARNING: {side}: failed to close RM65 connection: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read live RM65 qpos, build a shared robot_motion qpos init plan, "
            "and optionally execute it on real hardware."
        )
    )
    parser.add_argument("--qpos-target", default=str(DEFAULT_QPOS_TARGET_PATH))
    parser.add_argument("--current-qpos", default="", help="Optional JSON current qpos; skips live state read.")
    parser.add_argument("--arm", default="both", help="left, right, or both.")
    parser.add_argument("--left-ip", default=os.environ.get("LEFT_ARM_IP", ""))
    parser.add_argument("--right-ip", default=os.environ.get("RIGHT_ARM_IP", ""))
    parser.add_argument("--qpos-dt-sec", type=float, default=0.05)
    parser.add_argument("--joint-velocity-profile", default="rm65_init_slow")
    parser.add_argument("--max-steps-per-side", type=int, default=10000)
    parser.add_argument(
        "--collision-scene-profile",
        default=DEFAULT_QPOS_INIT_COLLISION_SCENE,
        help="Dual-arm QPOS init collision scene profile.",
    )
    parser.add_argument("--no-collision-check", action="store_true", help="Disable dual-arm constrained collision planning.")
    parser.add_argument("--collision-slow-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M)
    parser.add_argument("--collision-hold-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M)
    parser.add_argument("--collision-stop-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M)
    parser.add_argument("--max-joint-abs-deg", type=float, default=DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG)
    parser.add_argument("--joint-slow-margin-deg", type=float, default=DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG)
    parser.add_argument("--joint-hold-margin-deg", type=float, default=DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG)
    parser.add_argument("--single-arm-max-qpos-delta-deg", type=float, default=DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG)
    parser.add_argument("--no-branch-search", action="store_true", help="Disable dual-arm staged joint-order fallback planning.")
    parser.add_argument("--branch-search-max-branches", type=int, default=DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES)
    parser.add_argument(
        "--branch-neutral-qpos",
        default=str(DEFAULT_QPOS_TARGET_PATH),
        help="Optional neutral qpos target JSON used as the first dual-arm branch-search waypoint; empty disables.",
    )
    parser.add_argument("--trajectory-mode", type=int, default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--state-read-retries", type=int, default=2)
    parser.add_argument("--tracking-check-every", type=int, default=10)
    parser.add_argument("--max-tracking-error-deg", type=float, default=30.0)
    parser.add_argument("--disable-force-guard", action="store_true")
    parser.add_argument("--force-source", default="tool")
    parser.add_argument("--max-force-n", type=float, default=15.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=20, help="Print execution progress every N steps; 0 disables.")
    parser.add_argument("--execute", action="store_true", help="Actually send Movej_CANFD commands.")
    parser.add_argument("--confirm", default="", help=f"Required for --execute: {CONFIRM_TEXT}")
    parser.add_argument("--output", default="", help="Optional report JSON path.")
    parser.add_argument("--include-steps", action="store_true")
    parser.add_argument("--max-steps-json", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(_normalize_cli_argv(argv))
    active_sides = normalize_sides(str(args.arm))
    arm_ip_by_side = {
        side: value
        for side, value in {
            "left": str(args.left_ip).strip(),
            "right": str(args.right_ip).strip(),
        }.items()
        if value
    }
    config = RM65LiveQposInitConfig(
        qpos_target_path=args.qpos_target,
        active_sides=active_sides,
        current_qpos_path=(str(args.current_qpos).strip() or None),
        execute=bool(args.execute),
        confirm=str(args.confirm),
        qpos_dt_sec=float(args.qpos_dt_sec),
        joint_velocity_profile=str(args.joint_velocity_profile),
        max_steps_per_side=int(args.max_steps_per_side),
        collision_check=not bool(args.no_collision_check),
        collision_scene_profile=str(args.collision_scene_profile),
        collision_slow_distance_m=float(args.collision_slow_distance_m),
        collision_hold_distance_m=float(args.collision_hold_distance_m),
        collision_stop_distance_m=float(args.collision_stop_distance_m),
        max_joint_abs_deg=float(args.max_joint_abs_deg),
        joint_slow_margin_deg=float(args.joint_slow_margin_deg),
        joint_hold_margin_deg=float(args.joint_hold_margin_deg),
        single_arm_max_qpos_delta_deg=float(args.single_arm_max_qpos_delta_deg),
        branch_search=not bool(args.no_branch_search),
        branch_search_max_branches=int(args.branch_search_max_branches),
        branch_neutral_qpos_path=(str(args.branch_neutral_qpos).strip() or None),
        trajectory_mode=int(args.trajectory_mode),
        radio=int(args.radio),
        state_read_retries=int(args.state_read_retries),
        force_guard=RM65ForceGuardConfig(
            enabled=not bool(args.disable_force_guard),
            source=str(args.force_source),
            max_force_n=float(args.max_force_n),
            max_torque_nm=float(args.max_torque_nm),
            emergency_stop_on_limit=True,
        ),
        tracking_check_every=int(args.tracking_check_every),
        max_tracking_error_deg=float(args.max_tracking_error_deg),
        progress_every=int(args.progress_every),
        output_path=(str(args.output).strip() or None),
        include_steps=bool(args.include_steps),
        max_steps_json=int(args.max_steps_json),
    )
    report = execute_rm65_live_qpos_init(config, arm_ip_by_side=arm_ip_by_side)
    summary = report["plan"]["summary"]
    execution = report["execution"]
    print(
        "rm65-live-qpos-init "
        f"sides={','.join(active_sides)} "
        f"planner={summary['metadata'].get('planner', report['plan'].get('data_type'))} "
        f"steps={summary['step_count']} "
        f"duration_sec={summary['duration_sec']:.3f} "
        f"execute={bool(execution['executed'])} "
        f"sent={execution['sent_step_count']}"
    )
    if not str(args.output).strip():
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


def _check_tracking(
    robots_by_side: Mapping[str, object],
    *,
    command_qpos_by_side: Mapping[str, np.ndarray],
    active_sides: Sequence[str],
    max_tracking_error_deg: float,
    retry: int,
    step_index: int,
) -> dict[str, Any]:
    observed = read_current_qpos_by_side(robots_by_side, active_sides=active_sides, retry=retry)
    errors: dict[str, float] = {}
    for side in active_sides:
        delta_deg = np.rad2deg(observed[side] - np.asarray(command_qpos_by_side[side], dtype=float).reshape(6))
        errors[side] = float(np.max(np.abs(delta_deg)))
    max_error = max(errors.values(), default=0.0)
    if max_error > float(max_tracking_error_deg):
        raise RuntimeError(
            f"tracking error {max_error:.3f}deg > {float(max_tracking_error_deg):.3f}deg "
            f"at step {int(step_index)}"
        )
    return {
        "step_index": int(step_index),
        "max_tracking_error_deg": max_error,
        "tracking_error_deg_by_side": errors,
    }


def _load_rm65_sdk():
    try:
        from hardware.rm65.rm_api2_adapter import Arm, RM65
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RM65 live qpos init requires the RM65 SDK to be importable.") from exc
    return Arm, RM65


def _normalize_cli_argv(argv: Sequence[str] | None) -> list[str]:
    raw = sys.argv[1:] if argv is None else list(argv)
    return [str(item).strip() for item in raw if str(item).strip()]


def _normalize_ip_text(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and "](" in text:
        text = text[1:].split("](", 1)[0].strip()
    return text


def _valid_rm65_ip_or_none(value: object, *, side: str, source: str) -> str | None:
    ip_text = _normalize_ip_text(value)
    if not ip_text:
        return None
    reason = _invalid_rm65_ip_reason(ip_text)
    if reason:
        print(f"WARNING: ignoring {side} RM65 IP from {source}: {ip_text!r} ({reason})")
        return None
    return ip_text


def _require_valid_rm65_ip(value: object, *, side: str, source: str) -> str:
    ip_text = _normalize_ip_text(value)
    reason = _invalid_rm65_ip_reason(ip_text)
    if reason:
        raise ValueError(f"{side} RM65 IP from {source} is invalid: {ip_text!r} ({reason})")
    return ip_text


def _invalid_rm65_ip_reason(value: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return "not a valid IP address"
    if ip.version != 4:
        return "RM65 adapter expects an IPv4 address"
    if any(ip in network for network in DOCUMENTATION_IP_NETWORKS):
        return "documentation/test network, not a real robot address"
    if ip.is_unspecified:
        return "unspecified address"
    if ip.is_loopback:
        return "loopback address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_link_local:
        return "link-local address"
    return ""


def _require_zero(code: object, label: str) -> None:
    parsed = _result_code(code)
    if parsed != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _result_code(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, tuple) and value:
        return _result_code(value[0])
    if isinstance(value, Mapping):
        for key in ("code", "ret", "result"):
            if key in value:
                return int(value[key])
    text = str(value).split(":", 1)[0].strip()
    return int(text)


def _finite_vector(values: object, *, length: int, label: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"{label} must be a numeric vector.") from exc
    if out.shape != (int(length),) or not np.all(np.isfinite(out)):
        raise RuntimeError(f"{label} must contain {length} finite values, got shape={out.shape}.")
    return out.astype(float, copy=True)


def _write_report_if_requested(report: Mapping[str, Any], output_path: str | Path | None) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _qpos_mapping_to_lists(qpos_by_side: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        str(side): np.asarray(qpos, dtype=float).reshape(-1).astype(float).tolist()
        for side, qpos in qpos_by_side.items()
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())

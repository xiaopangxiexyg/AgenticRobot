from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import PoseTolerance


@dataclass(frozen=True)
class IKSafetyProfile:
    name: str
    target_tolerance: PoseTolerance = PoseTolerance()
    relaxation_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    hard_branch_joint_indices: tuple[int, ...] = (0, 1, 2)
    hard_branch_max_delta_deg: tuple[float, ...] = (90.0, 90.0, 90.0)
    per_step_joint_limit_deg: tuple[float, ...] | None = None
    reject_singular: bool = True
    allow_smoothing_recovery: bool = True
    enable_expensive_seed_expansion: bool = True
    allow_parallel_candidate_generation: bool = False
    max_candidate_count: int = 32
    provider_time_budget_ms: float | None = None
    fk_error_tolerance_m: float = 0.015
    fk_error_tolerance_deg: float = 8.0
    soft_singularity_labels: tuple[str, ...] = ("wrist_singularity",)
    soft_singularity_penalty_deg: float = 25.0
    pose_slack_penalty_deg: float = 10.0
    coarse_tracking_enabled: bool = False
    coarse_tracking_start_fraction: float = 1.0
    coarse_tracking_max_tolerance_fraction: float = 4.0
    coarse_tracking_max_position_error_m: float = 0.0
    coarse_tracking_max_orientation_error_deg: float = 0.0
    coarse_tracking_min_fraction_progress: float = 0.05
    coarse_tracking_orientation_recovery_enabled: bool = False
    coarse_tracking_orientation_recovery_min_progress_deg: float = 15.0
    coarse_tracking_orientation_recovery_max_position_worsen_m: float = 0.08
    local_step_enabled: bool = True
    local_step_first_enabled: bool = False
    local_step_first_full_solve_interval: int = 0
    local_step_first_skip_full_when_far: bool = False
    local_step_first_skip_full_pose_fraction: float = 1.0
    local_step_first_skip_full_if_budget_low: bool = False
    local_step_first_full_solve_min_remaining_ms: float = 5.0
    local_step_trigger_ratio: float = 4.0
    local_step_max_iterations: int = 1
    local_step_iteration_counts: tuple[int, ...] = ()
    local_step_min_progress_fraction: float = 1e-4
    local_step_accept_position_progress_m: float = 5e-4
    local_step_accept_orientation_progress_deg: float = 0.25
    local_step_max_position_worsen_m: float = 0.003
    local_step_max_orientation_worsen_deg: float = 2.0
    local_step_scale_factors: tuple[float, ...] = (1.0,)
    local_step_dynamic_worsen_enabled: bool = False
    local_step_dynamic_worsen_error_start_m: float = 0.05
    local_step_dynamic_worsen_error_start_deg: float = 15.0
    local_step_dynamic_worsen_position_fraction: float = 0.10
    local_step_dynamic_worsen_orientation_fraction: float = 0.10
    local_step_dynamic_worsen_max_position_m: float = 0.02
    local_step_dynamic_worsen_max_orientation_deg: float = 8.0
    local_step_accept_worse_within_tolerance: bool = True
    local_step_orientation_recovery_enabled: bool = False
    local_step_orientation_recovery_min_progress_deg: float = 5.0
    local_step_orientation_recovery_max_position_worsen_m: float = 0.08
    local_step_orientation_recovery_max_after_position_error_m: float = 0.30
    local_step_component_progress_enabled: bool = False
    local_step_component_progress_min_position_m: float = 5e-4
    local_step_component_progress_min_orientation_deg: float = 0.25
    local_step_component_progress_max_score_regression: float = 0.0
    local_step_component_progress_max_position_worsen_m: float = 0.0
    local_step_component_progress_max_orientation_worsen_deg: float = 0.0
    local_step_component_progress_max_after_position_error_m: float = 0.0
    local_step_limit_escape_enabled: bool = False
    local_step_limit_escape_activation_margin_deg: float = 2.0
    local_step_limit_escape_step_fraction: float = 1.0
    local_step_limit_escape_max_candidates: int = 12
    local_step_limit_escape_accept_worse_within_tolerance: bool = True
    local_step_limit_escape_accept_bounded_worse: bool = True
    local_step_limit_escape_max_position_worsen_m: float = 0.03
    local_step_limit_escape_max_orientation_worsen_deg: float = 12.0
    local_step_singularity_escape_enabled: bool = False
    local_step_singularity_escape_step_fraction: float = 1.0
    local_step_singularity_escape_max_candidates: int = 6
    local_step_singularity_escape_max_position_worsen_m: float = 0.02
    local_step_singularity_escape_max_orientation_worsen_deg: float = 8.0
    local_step_full_target_step_enabled: bool = True
    local_step_hold_on_failure: bool = True
    stable_recovery_frames: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def hard_branch_limits_array(self) -> np.ndarray:
        limits = np.asarray(self.hard_branch_max_delta_deg, dtype=float).reshape(-1)
        if limits.shape != (len(self.hard_branch_joint_indices),):
            raise ValueError("hard_branch_max_delta_deg must match hard_branch_joint_indices")
        return limits


def default_profiles_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default_ik_safety_profiles.json"


def _tuple_float(payload: Any, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if payload is None:
        return default
    return tuple(float(value) for value in payload)


def _tuple_int(payload: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if payload is None:
        return default
    return tuple(int(value) for value in payload)


def profile_from_dict(name: str, payload: dict[str, Any]) -> IKSafetyProfile:
    tolerance_payload = payload.get("target_tolerance", {})
    local_step_max_iterations = int(payload.get("local_step_max_iterations", 1))
    return IKSafetyProfile(
        name=name,
        target_tolerance=PoseTolerance(
            position_m=float(tolerance_payload.get("position_m", 0.0)),
            orientation_deg=float(tolerance_payload.get("orientation_deg", 0.0)),
        ),
        relaxation_fractions=_tuple_float(
            payload.get("relaxation_fractions"),
            default=(0.0, 0.25, 0.5, 1.0),
        ),
        hard_branch_joint_indices=_tuple_int(
            payload.get("hard_branch_joint_indices"),
            default=(0, 1, 2),
        ),
        hard_branch_max_delta_deg=_tuple_float(
            payload.get("hard_branch_max_delta_deg"),
            default=(90.0, 90.0, 90.0),
        ),
        per_step_joint_limit_deg=(
            None
            if payload.get("per_step_joint_limit_deg") is None
            else _tuple_float(payload.get("per_step_joint_limit_deg"), default=())
        ),
        reject_singular=bool(payload.get("reject_singular", True)),
        allow_smoothing_recovery=bool(payload.get("allow_smoothing_recovery", True)),
        enable_expensive_seed_expansion=bool(payload.get("enable_expensive_seed_expansion", True)),
        allow_parallel_candidate_generation=bool(payload.get("allow_parallel_candidate_generation", False)),
        max_candidate_count=int(payload.get("max_candidate_count", 32)),
        provider_time_budget_ms=(
            None
            if payload.get("provider_time_budget_ms") in (None, "")
            else float(payload.get("provider_time_budget_ms"))
        ),
        fk_error_tolerance_m=float(payload.get("fk_error_tolerance_m", 0.015)),
        fk_error_tolerance_deg=float(payload.get("fk_error_tolerance_deg", 8.0)),
        soft_singularity_labels=tuple(
            str(value) for value in payload.get("soft_singularity_labels", ("wrist_singularity",))
        ),
        soft_singularity_penalty_deg=float(payload.get("soft_singularity_penalty_deg", 25.0)),
        pose_slack_penalty_deg=float(payload.get("pose_slack_penalty_deg", 10.0)),
        coarse_tracking_enabled=bool(payload.get("coarse_tracking_enabled", False)),
        coarse_tracking_start_fraction=float(payload.get("coarse_tracking_start_fraction", 1.0)),
        coarse_tracking_max_tolerance_fraction=float(
            payload.get("coarse_tracking_max_tolerance_fraction", 4.0)
        ),
        coarse_tracking_max_position_error_m=float(
            payload.get("coarse_tracking_max_position_error_m", 0.0)
        ),
        coarse_tracking_max_orientation_error_deg=float(
            payload.get("coarse_tracking_max_orientation_error_deg", 0.0)
        ),
        coarse_tracking_min_fraction_progress=float(
            payload.get("coarse_tracking_min_fraction_progress", 0.05)
        ),
        coarse_tracking_orientation_recovery_enabled=bool(
            payload.get("coarse_tracking_orientation_recovery_enabled", False)
        ),
        coarse_tracking_orientation_recovery_min_progress_deg=float(
            payload.get("coarse_tracking_orientation_recovery_min_progress_deg", 15.0)
        ),
        coarse_tracking_orientation_recovery_max_position_worsen_m=float(
            payload.get("coarse_tracking_orientation_recovery_max_position_worsen_m", 0.08)
        ),
        local_step_enabled=bool(payload.get("local_step_enabled", True)),
        local_step_first_enabled=bool(payload.get("local_step_first_enabled", False)),
        local_step_first_full_solve_interval=max(
            0,
            int(payload.get("local_step_first_full_solve_interval", 0)),
        ),
        local_step_first_skip_full_when_far=bool(
            payload.get("local_step_first_skip_full_when_far", False)
        ),
        local_step_first_skip_full_pose_fraction=float(
            payload.get("local_step_first_skip_full_pose_fraction", 1.0)
        ),
        local_step_first_skip_full_if_budget_low=bool(
            payload.get("local_step_first_skip_full_if_budget_low", False)
        ),
        local_step_first_full_solve_min_remaining_ms=float(
            payload.get("local_step_first_full_solve_min_remaining_ms", 5.0)
        ),
        local_step_trigger_ratio=float(payload.get("local_step_trigger_ratio", 4.0)),
        local_step_max_iterations=local_step_max_iterations,
        local_step_iteration_counts=_tuple_int(
            payload.get("local_step_iteration_counts"),
            default=(),
        ),
        local_step_min_progress_fraction=float(payload.get("local_step_min_progress_fraction", 1e-4)),
        local_step_accept_position_progress_m=float(payload.get("local_step_accept_position_progress_m", 5e-4)),
        local_step_accept_orientation_progress_deg=float(
            payload.get("local_step_accept_orientation_progress_deg", 0.25)
        ),
        local_step_max_position_worsen_m=float(payload.get("local_step_max_position_worsen_m", 0.003)),
        local_step_max_orientation_worsen_deg=float(payload.get("local_step_max_orientation_worsen_deg", 2.0)),
        local_step_scale_factors=_tuple_float(
            payload.get("local_step_scale_factors"),
            default=(1.0,),
        ),
        local_step_dynamic_worsen_enabled=bool(payload.get("local_step_dynamic_worsen_enabled", False)),
        local_step_dynamic_worsen_error_start_m=float(
            payload.get("local_step_dynamic_worsen_error_start_m", 0.05)
        ),
        local_step_dynamic_worsen_error_start_deg=float(
            payload.get("local_step_dynamic_worsen_error_start_deg", 15.0)
        ),
        local_step_dynamic_worsen_position_fraction=float(
            payload.get("local_step_dynamic_worsen_position_fraction", 0.10)
        ),
        local_step_dynamic_worsen_orientation_fraction=float(
            payload.get("local_step_dynamic_worsen_orientation_fraction", 0.10)
        ),
        local_step_dynamic_worsen_max_position_m=float(
            payload.get("local_step_dynamic_worsen_max_position_m", 0.02)
        ),
        local_step_dynamic_worsen_max_orientation_deg=float(
            payload.get("local_step_dynamic_worsen_max_orientation_deg", 8.0)
        ),
        local_step_accept_worse_within_tolerance=bool(
            payload.get("local_step_accept_worse_within_tolerance", True)
        ),
        local_step_orientation_recovery_enabled=bool(
            payload.get("local_step_orientation_recovery_enabled", False)
        ),
        local_step_orientation_recovery_min_progress_deg=float(
            payload.get("local_step_orientation_recovery_min_progress_deg", 5.0)
        ),
        local_step_orientation_recovery_max_position_worsen_m=float(
            payload.get("local_step_orientation_recovery_max_position_worsen_m", 0.08)
        ),
        local_step_orientation_recovery_max_after_position_error_m=float(
            payload.get("local_step_orientation_recovery_max_after_position_error_m", 0.30)
        ),
        local_step_component_progress_enabled=bool(
            payload.get("local_step_component_progress_enabled", False)
        ),
        local_step_component_progress_min_position_m=float(
            payload.get("local_step_component_progress_min_position_m", 5e-4)
        ),
        local_step_component_progress_min_orientation_deg=float(
            payload.get("local_step_component_progress_min_orientation_deg", 0.25)
        ),
        local_step_component_progress_max_score_regression=float(
            payload.get("local_step_component_progress_max_score_regression", 0.0)
        ),
        local_step_component_progress_max_position_worsen_m=float(
            payload.get("local_step_component_progress_max_position_worsen_m", 0.0)
        ),
        local_step_component_progress_max_orientation_worsen_deg=float(
            payload.get("local_step_component_progress_max_orientation_worsen_deg", 0.0)
        ),
        local_step_component_progress_max_after_position_error_m=float(
            payload.get("local_step_component_progress_max_after_position_error_m", 0.0)
        ),
        local_step_limit_escape_enabled=bool(payload.get("local_step_limit_escape_enabled", False)),
        local_step_limit_escape_activation_margin_deg=float(
            payload.get("local_step_limit_escape_activation_margin_deg", 2.0)
        ),
        local_step_limit_escape_step_fraction=float(
            payload.get("local_step_limit_escape_step_fraction", 1.0)
        ),
        local_step_limit_escape_max_candidates=int(payload.get("local_step_limit_escape_max_candidates", 12)),
        local_step_limit_escape_accept_worse_within_tolerance=bool(
            payload.get("local_step_limit_escape_accept_worse_within_tolerance", True)
        ),
        local_step_limit_escape_accept_bounded_worse=bool(
            payload.get("local_step_limit_escape_accept_bounded_worse", True)
        ),
        local_step_limit_escape_max_position_worsen_m=float(
            payload.get("local_step_limit_escape_max_position_worsen_m", 0.03)
        ),
        local_step_limit_escape_max_orientation_worsen_deg=float(
            payload.get("local_step_limit_escape_max_orientation_worsen_deg", 12.0)
        ),
        local_step_singularity_escape_enabled=bool(
            payload.get("local_step_singularity_escape_enabled", False)
        ),
        local_step_singularity_escape_step_fraction=float(
            payload.get("local_step_singularity_escape_step_fraction", 1.0)
        ),
        local_step_singularity_escape_max_candidates=int(
            payload.get("local_step_singularity_escape_max_candidates", 6)
        ),
        local_step_singularity_escape_max_position_worsen_m=float(
            payload.get("local_step_singularity_escape_max_position_worsen_m", 0.02)
        ),
        local_step_singularity_escape_max_orientation_worsen_deg=float(
            payload.get("local_step_singularity_escape_max_orientation_worsen_deg", 8.0)
        ),
        local_step_full_target_step_enabled=bool(payload.get("local_step_full_target_step_enabled", True)),
        local_step_hold_on_failure=bool(payload.get("local_step_hold_on_failure", True)),
        stable_recovery_frames=int(payload.get("stable_recovery_frames", 3)),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def load_ik_safety_profiles(path: str | Path | None = None) -> dict[str, IKSafetyProfile]:
    resolved = default_profiles_path() if path in (None, "") else Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{resolved} does not contain any IK safety profiles")
    return {
        str(name): profile_from_dict(str(name), dict(profile_payload))
        for name, profile_payload in profiles.items()
    }
